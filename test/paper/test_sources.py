import gzip
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import numpy as np
import paperbroker
import pytest

import ashare.paper
from ashare.paper.market import corporate_actions, normalize, rows, timestamp
from ashare.paper.schedule import Schedule
from ashare.paper.source import (
    Batch,
    LiveSource,
    MinuteSource,
    Monitor,
    SourceError,
    TokenBucket,
    TushareClient,
)
from ashare.paper.storage import ParquetStore

ROOT = Path(__file__).resolve().parents[2]
assert Path(paperbroker.__file__).resolve().is_relative_to(ROOT)
assert Path(ashare.paper.__file__).resolve().is_relative_to(ROOT)


def fixture_directory():
    configured = os.environ.get("ASHARE_DATA_DIR")
    candidates = (
        [Path(configured) / "fixtures/paper/tushare"]
        if configured
        else [
            p / "data/fixtures/paper/tushare" for p in Path(__file__).resolve().parents
        ]
    )
    return next((p for p in candidates if (p / "INDEX.json").is_file()), None)


FIXTURES = fixture_directory()


def fixture(name):
    if FIXTURES is None:
        pytest.skip(
            "Tushare 真实夹具未找到：设置 ASHARE_DATA_DIR 指向含 fixtures/paper/tushare/INDEX.json 的 data 目录"
        )
    return json.loads(gzip.decompress((FIXTURES / f"{name}.json.gz").read_bytes()))[
        "response"
    ]


class FakeClock:
    def __init__(self, now="2026-11-11T09:30:00+08:00"):
        self.value = datetime.fromisoformat(now)
        self.elapsed = 0.0
        self.sleeps = []

    def now(self):
        return self.value

    def monotonic(self):
        return self.elapsed

    def sleep(self, seconds):
        self.elapsed += seconds
        self.value += timedelta(seconds=seconds)
        self.sleeps.append(seconds)


def test_real_snapshot_schema_volume_and_parquet(tmp_path):
    records = sum(
        (rows(fixture(name)) for name in ("rt_k_sh", "rt_k_sz", "rt_k_bj")), []
    )
    symbols = [r["ts_code"] for r in records]
    assert len(symbols) == 5584
    q = normalize(records, symbols)
    assert q.shape[1] == 18 and q.dtype == np.int64 and q.flags.c_contiguous
    record = next(r for r in records if r["ts_code"] == "600000.SH")
    i = symbols.index("600000.SH")
    assert q[i, 7] == 48_504_583
    assert q[i, 9] == 101800 and q[i, 10] == 84441
    assert q[i, 1] == timestamp(record["trade_time"])
    store = ParquetStore(tmp_path)
    store.received_at = timestamp("2026-11-12 00:00:00")
    store.source = "fixture_rt_k"
    ref = store.put(1, q)
    assert np.array_equal(store.get(ref), q)
    from ashare.paper.source import ReplaySource

    replay = ReplaySource(tmp_path, "20261112")
    batch = replay.poll()
    assert batch.received_at == store.received_at and np.array_equal(batch.quotes, q)
    assert replay.poll() is None


def test_transport_retries_limits_redaction_and_single_outlet(
    tmp_path, monkeypatch, caplog
):
    secret = "synthetic-token-for-redaction-check"
    monkeypatch.setenv("TUSHARE_TOKEN", secret)
    clock = FakeClock()
    calls = []
    payloads = [
        dict(code=-200, msg=f"每分钟限流 {secret}"),
        fixture("rt_k_few"),
        fixture("error_bad_api"),
    ]

    def handle(request):
        assert request.url == "https://api.tushare.pro"
        data = json.loads(request.content)
        assert data["token"] == secret and request.headers["user-agent"]
        calls.append(data)
        return httpx.Response(200, json=payloads.pop(0))

    with TushareClient(
        tmp_path / "lock", clock=clock, transport=httpx.MockTransport(handle)
    ) as client:
        with pytest.raises(SourceError, match="占用"):
            TushareClient(tmp_path / "another", transport=httpx.MockTransport(handle))
        source = LiveSource(
            client, ["600000.SH", "000001.SZ", "300750.SZ", "688981.SH"]
        )
        assert source.poll().quotes.shape == (4, 18)
        with pytest.raises(SourceError) as exc:
            client.call("no_such_api")
        assert secret not in str(exc.value)
    assert len(calls) == 3 and clock.elapsed >= 60
    assert "rate_limit" in caplog.text and secret not in caplog.text


def test_outage_staleness_and_bucket(tmp_path, monkeypatch, caplog):
    clock = FakeClock()
    bucket = TokenBucket(clock=clock)
    for _ in range(41):
        bucket.acquire()
    assert clock.elapsed >= 60
    monkeypatch.setenv("TUSHARE_TOKEN", "synthetic")

    def fail(request):
        raise httpx.ConnectError("synthetic should not be exposed")

    with TushareClient(
        tmp_path / "lock", clock=clock, transport=httpx.MockTransport(fail)
    ) as client:
        with pytest.raises(SourceError, match="重试耗尽"):
            client.call("rt_k")
    assert "source_outage" in caplog.text and "synthetic" not in caplog.text
    monitor = Monitor(clock, stale_seconds=10)
    q = normalize(rows(fixture("rt_k_few")), ["600000.SH"])
    batch = Batch(int(q[0, 1]), q, "fixture")
    monitor.observe(batch)
    clock.sleep(11)
    monitor.observe(batch)
    assert "stale_quotes" in caplog.text


def test_calendar_minute_units_and_dividend():
    schedule = Schedule(rows(fixture("trade_cal_2026q4")))
    for when, phase in (
        ("09:00", "preopen"),
        ("09:15", "open_auction"),
        ("09:30", "continuous"),
        ("11:30", "break"),
        ("12:59", "break"),
        ("13:00", "continuous"),
        ("14:57", "close_auction"),
        ("15:00", "closed"),
    ):
        assert (
            schedule.phase(datetime.fromisoformat(f"2026-11-11T{when}:00+08:00"))
            == phase
        )
    assert (
        schedule.phase(datetime.fromisoformat("2026-11-14T10:00:00+08:00")) == "closed"
    )
    data = rows(fixture("stk_mins_600000_SH_20261111"))
    source = MinuteSource(data, ["600000.SH"])
    batches = []
    while (batch := source.poll()) is not None:
        batches.append(batch)
    assert len(batches) == 241
    assert batches[-1].quotes[0, 7] == 48_504_583
    assert batches[1].quotes[0, 7] - batches[0].quotes[0, 7] == 862101
    actions = corporate_actions(
        rows(fixture("dividend_600000")), ["600000.SH"], "20261111"
    )
    assert actions == []
    real = next(
        r
        for r in rows(fixture("dividend_600000"))
        if r["div_proc"] == "实施" and r["cash_div_tax"]
    )
    action = corporate_actions([real, real], ["600000.SH"], real["ex_date"])
    assert len(action) == 1 and action[0]["cash_per_share"] > 0
