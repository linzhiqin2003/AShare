"""审查回归：录制响应、可注入时钟和明确的失败路径。"""

import gzip
import json
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import paperbroker
import pytest
from paperbroker import Broker, money
from paperbroker.journal import canonical

import ashare.paper
from ashare.paper.__main__ import main, read_state
from ashare.paper.importer import import_raw
from ashare.paper.market import (
    corporate_actions,
    normalize,
    rows,
    shares,
    suspension_rules,
    timestamp,
)
from ashare.paper.runner import LiveRunner
from ashare.paper.schedule import Schedule
from ashare.paper.source import (
    Batch,
    LiveSource,
    Monitor,
    ReplaySource,
    SourceError,
    TushareClient,
)
from ashare.paper.storage import ParquetStore

from .test_runner import make_runner
from .test_sources import FakeClock, fixture

ROOT = Path(__file__).resolve().parents[2]
assert Path(paperbroker.__file__).resolve().is_relative_to(ROOT)
assert Path(ashare.paper.__file__).resolve().is_relative_to(ROOT)


def at(clock, day, clock_time):
    new = datetime.fromisoformat(f"{day}T{clock_time}+08:00")
    clock.elapsed += max(0, (new - clock.value).total_seconds())
    clock.value = new


def test_clock_skew_and_rejected_settlement_never_report(tmp_path, monkeypatch):
    runner, client, clock, broker = make_runner(tmp_path)
    original = client.call

    def call(name, params=None, fields=""):
        result = original(name, params, fields)
        if name == "rt_k":
            for r in result:
                r["trade_time"] = (clock.now() + timedelta(seconds=2)).isoformat()
        return result

    client.call = call
    for t in ("09:00:00", "09:30:00", "09:30:03", "09:30:06", "14:57:00"):
        at(clock, "2026-11-11", t)
        runner.step()
    assert runner.target_day == "20261111"
    assert not [e for e in broker.history if e["type"] == "rejected"]
    real_settle = broker.settle
    monkeypatch.setattr(broker, "settle", lambda t, d: real_settle(t - 1, d))
    at(clock, "2026-11-11", "15:00:05")
    with pytest.raises(SourceError, match="time"):
        runner.step()
    assert runner.settled_day != "20261111"
    assert not (tmp_path / "reports/20261111.json").exists()
    monkeypatch.setattr(broker, "settle", real_settle)
    clock.sleep(60)
    runner.step()
    assert json.loads(broker.dump())["settled_day"] == 20261111
    assert (tmp_path / "reports/20261111.json").exists()
    broker.close()


def test_rejected_target_does_not_mark_done(tmp_path, monkeypatch):
    runner, _, clock, broker = make_runner(tmp_path)
    monkeypatch.setattr(
        broker, "target_weights", lambda t, w, **k: [broker.order(t, "A", "buy", 1, 10)]
    )
    at(clock, "2026-11-11", "09:30:00")
    with pytest.raises(SourceError, match="quantity"):
        runner.step()
    assert runner.target_day == ""
    broker.close()


def test_two_days_baseline_report_and_calendar_refresh(tmp_path):
    runner, client, clock, broker = make_runner(tmp_path)
    runner.schedule = Schedule([dict(cal_date="20261111", is_open=1)])
    original = client.call

    def call(name, params=None, fields=""):
        if name == "trade_cal":
            return [dict(cal_date="20261112", is_open=1)]
        return original(name, params, fields)

    client.call = call
    for day in ("2026-11-11", "2026-11-12"):
        opening_volume = 0 if day == "2026-11-11" else 500_000
        client.volume = opening_volume
        for t in (
            "09:00:00",
            "09:30:00",
            "14:56:57",
            "14:57:00",
            "14:58:00",
            "15:00:01",
        ):
            at(clock, day, t)
            runner.step()
        state = json.loads(broker.dump())
        assert state["settled_day"] == int(day.replace("-", ""))
        # 14:57 的 30000 股为基线，14:58、15:00 各增加 10000 股。
        assert runner.close_baseline == {0: opening_volume + 30000}
        recorded = ReplaySource(tmp_path / "quotes", day.replace("-", ""))
        close = []
        while (batch := recorded.poll()) is not None:
            if batch.source == "rt_k_close_volume_proxy":
                close.append(batch)
        assert len(close) == 1 and close[0].quotes[0, 16] == 20000
        daily = json.loads(
            (tmp_path / "reports" / (day.replace("-", "") + ".json")).read_bytes()
        )
        assert all(e["day"] == int(day.replace("-", "")) for e in daily["events"])
    broker.close()


def test_restart_outputs_and_crash_after_settlement_repairs_report(tmp_path):
    runner, client, clock, broker = make_runner(tmp_path)
    for t in ("09:00:00", "09:30:00", "09:30:03"):
        at(clock, "2026-11-11", t)
        runner.step()
    before = [e for e in broker.history if e["type"] == "fill"]
    assert before
    broker.checkpoint()
    broker.close()
    store = ParquetStore(tmp_path / "quotes")
    broker = Broker(["A"], journal=tmp_path, quote_store=store)
    assert [e for e in broker.events() if e["type"] == "fill"] == before
    runner = LiveRunner(
        broker,
        LiveSource(client, ["A"]),
        Schedule(rows(fixture("trade_cal_2026q4"))),
        clock,
        store,
        output=None,
    )
    at(clock, "2026-11-11", "15:00:03")
    runner.step()
    broker.close()
    assert not (tmp_path / "reports/20261111.json").exists()
    broker = Broker(["A"], journal=tmp_path, quote_store=store)
    runner = LiveRunner(
        broker,
        LiveSource(client, ["A"]),
        Schedule(rows(fixture("trade_cal_2026q4"))),
        clock,
        store,
        output=tmp_path / "reports",
    )
    runner.step()
    result = json.loads((tmp_path / "reports/20261111.json").read_bytes())
    assert result["fills"] == before
    broker.close()


def test_missed_close_catch_up_and_daily_retry_backoff(tmp_path):
    runner, client, clock, broker = make_runner(tmp_path)
    at(clock, "2026-11-11", "09:30:00")
    runner.step()
    at(clock, "2026-11-12", "09:00:00")
    runner.step()
    assert json.loads(broker.dump())["settled_day"] == 20261111
    assert broker.account()["day"] == 20261112
    assert (tmp_path / "reports/20261111.json").exists()
    original = client.call
    calls = []

    def call(name, params=None, fields=""):
        calls.append(name)
        if name == "daily":
            return []
        return original(name, params, fields)

    client.call = call
    at(clock, "2026-11-12", "15:01:00")
    with pytest.raises(SourceError, match="尚未发布"):
        runner.step()
    session_count = sum(e["type"] == "session" for e in broker.history)
    for _ in range(5):
        clock.sleep(3)
        runner.step()
    assert calls.count("daily") == 1
    assert sum(e["type"] == "session" for e in broker.history) == session_count
    broker.close()


def test_full_market_missing_metadata_and_daily_rounding(tmp_path):
    records = sum(
        (rows(fixture("rt_k_" + suffix)) for suffix in ("sh", "sz", "bj")), []
    )
    symbols = sorted({r["ts_code"] for r in records})
    clock = FakeClock("2026-11-11T09:00:00+08:00")

    class Client:
        def __init__(self):
            self.clock = clock

        def call(self, name, params=None, fields=""):
            if name == "rt_k":
                suffix = (
                    "sh"
                    if params["ts_code"].startswith("6")
                    else "sz"
                    if params["ts_code"].startswith("3")
                    else "bj"
                )
                return rows(fixture("rt_k_" + suffix))
            if name == "dividend":
                return []
            return rows(fixture(name + "_20261111"))

    b = Broker(symbols)
    source = LiveSource(Client(), symbols, full_market=True)
    runner = LiveRunner(
        b, source, Schedule(rows(fixture("trade_cal_2026q4"))), clock, None
    )
    runner.step()
    at(clock, "2026-11-11", "14:57:00")
    runner.step()
    at(clock, "2026-11-11", "15:01:00")
    runner.step()
    assert json.loads(b.dump())["settled_day"] == 20261111
    daily = {r["ts_code"]: r for r in rows(fixture("daily_20261111"))}
    quotes = json.loads(b.dump())["quotes"]
    for symbol, r in daily.items():
        if symbol in b.symbol_ids:
            q = quotes[b.symbol_ids[symbol]]
            assert q is not None and q[2] == money(r["close"])
            assert q[7] == shares(r["vol"], 100)
    affected = [
        r for r in daily.values() if int(r["vol"] * 100) < shares(r["vol"], 100)
    ]
    assert len(affected) == 264
    rt = {r["ts_code"]: r for r in records}
    previously_dropped = [
        r
        for r in daily.values()
        if r["ts_code"] in rt
        and int(r["vol"] * 100) < int(rt[r["ts_code"]]["vol"] or 0)
    ]
    assert len(previously_dropped) == 267
    actual_corrections = [
        r["ts_code"]
        for r in daily.values()
        if r["ts_code"] in rt
        and shares(r["vol"], 100) < int(rt[r["ts_code"]]["vol"] or 0)
    ]
    assert sorted(actual_corrections) == ["920059.BJ", "920375.BJ", "920964.BJ"]
    assert len(set(b.symbols) - source.limits.keys()) == 4


def test_duplicate_actions_suspend_intervals_bad_dates():
    actions = corporate_actions(
        [
            dict(
                ts_code="A",
                ex_date="20261111",
                div_proc="实施",
                end_date=end,
                cash_div_tax=cash,
                stk_div=stock,
            )
            for end, cash, stock in [("20250630", 0.5, 0.1), ("20251231", 0.3, 0.2)]
        ],
        ["A"],
        "20261111",
    )
    assert actions == [
        dict(
            symbol=0,
            cash_per_share=8000,
            split_num=13,
            split_den=10,
            reference_price=None,
        )
    ]
    full, intervals = suspension_rules(
        [dict(ts_code="A", suspend_type="S", suspend_timing="09:30-10:00")]
    )
    records = [
        dict(ts_code="A", close=10, trade_time=t, vol=0)
        for t in (
            "bad",
            "2026-11-10 10:00:00",
            "2026-11-11 09:45:00",
            "2026-11-11 10:00:00",
        )
    ]
    q = normalize(
        records, ["A"], day="20261111", suspended=full, suspension_intervals=intervals
    )
    assert q.shape == (2, 18) and q[:, 15].tolist() == [1, 0]


def test_open_auction_prime_and_empty_retry(tmp_path):
    runner, client, clock, b = make_runner(tmp_path)
    runner.targets = dict(at="09:20:00", weights={"A": 0.01})
    original = client.call
    attempts = []

    def call(name, params=None, fields=""):
        if name == "stk_auction":
            attempts.append(name)
            if len(attempts) == 1:
                return []
        result = original(name, params, fields)
        if name == "stk_limit":
            result[0]["pre_close"] = 10
        return result

    client.call = call
    for t in ("09:00:00", "09:20:00", "09:26:00"):
        at(clock, "2026-11-11", t)
        runner.step()
    assert runner.auction_day == ""
    assert json.loads(b.dump())["orders"][0]["time"] == timestamp("2026-11-11 09:20:00")
    clock.sleep(3)
    runner.step()
    assert runner.auction_day == "20261111"
    at(clock, "2026-11-11", "09:30:00")
    runner.step()
    assert any(e["type"] == "fill" and e["auction"] for e in b.history)
    b.close()


def test_target_cadence_and_unknown_symbols(tmp_path, monkeypatch):
    runner, _, clock, b = make_runner(tmp_path)
    at(clock, "2026-11-11", "09:30:00")
    runner.step()
    first = len(json.loads(b.dump())["orders"])
    calls = []
    original = b.target_weights

    def tracked(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(b, "target_weights", tracked)
    for _ in range(4):
        clock.sleep(3)
        runner.step()
    assert len(json.loads(b.dump())["orders"]) == first
    assert calls == []
    assert not [
        e for e in b.history if e["type"] == "ended" and e["status"] == "cancelled"
    ]
    with pytest.raises(ValueError, match="未注册"):
        LiveRunner(
            b, runner.source, runner.schedule, clock, None, dict(weights={"BAD": 0.1})
        )
    b.close()


def test_readonly_status_does_not_truncate(tmp_path):
    runner, _, _, b = make_runner(tmp_path)
    runner.step()
    b.close()
    log = tmp_path / "events.jsonl"
    log.write_bytes(log.read_bytes() + b'{"torn')
    before = {
        str(p.relative_to(tmp_path)): p.read_bytes()
        for p in tmp_path.rglob("*")
        if p.is_file()
    }
    assert read_state(tmp_path)["account"]["day"] == 20261111
    after = {
        str(p.relative_to(tmp_path)): p.read_bytes()
        for p in tmp_path.rglob("*")
        if p.is_file()
    }
    assert before == after


def test_interface_buckets_40203_and_limit_name_not_rate_error(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setenv("TUSHARE_TOKEN", "synthetic")
    clock = FakeClock()
    responses = [
        dict(code=40203, msg="抱歉，您访问接口(trade_cal)频率超限(500次/分钟)"),
        dict(code=0, data=dict(fields=[], items=[])),
        dict(code=-1, msg="stk_limit failure"),
    ]

    def handle(request):
        return httpx.Response(200, json=responses.pop(0))

    with TushareClient(
        tmp_path / "lock", clock=clock, transport=httpx.MockTransport(handle)
    ) as client:
        client.call("trade_cal")
        assert clock.elapsed >= 60
        assert (
            client.buckets["trade_cal"].rate == 400 / 60
            and client.buckets["rt_k"].rate == 40 / 60
        )
        before = clock.elapsed
        with pytest.raises(SourceError, match="api=stk_limit, code=-1"):
            client.call("stk_limit")
        assert clock.elapsed == before
    assert "rate_limit" in caplog.text


def test_monitor_suppresses_halted_and_alert_storm(caplog):
    clock = FakeClock()
    q = normalize(
        [dict(ts_code="A", close=10, trade_time="2026-11-11 09:00:00")],
        ["A"],
        suspended={"A"},
    )
    monitor = Monitor(clock)
    for _ in range(3):
        monitor.observe(Batch(timestamp(clock.now()), q, "test"))
        clock.sleep(3)
    assert not caplog.records
    q[0, 15] = 0
    for _ in range(3):
        monitor.observe(Batch(timestamp(clock.now()), q, "test"))
        clock.sleep(3)
    assert len(caplog.records) == 1


def test_raw_import_three_clocks_and_replay(tmp_path, capsys):
    rawdir = tmp_path / "raw"
    rawdir.mkdir()

    def write(name, api, sent, recv, records):
        fields = list(records[0]) if records else []
        payload = dict(
            api_name=api,
            params={},
            fields="",
            t_send=sent,
            t_recv=recv,
            response=dict(
                code=0,
                data=dict(
                    fields=fields, items=[[r[k] for k in fields] for r in records]
                ),
            ),
        )
        (rawdir / name).write_bytes(gzip.compress(canonical(payload)))

    write(
        "1.json.gz",
        "stk_limit",
        "2026-11-11T09:00:00+08:00",
        "2026-11-11T09:00:01+08:00",
        [dict(ts_code="A", up_limit=11, down_limit=9)],
    )
    write(
        "2.json.gz",
        "rt_k",
        "2026-11-11T09:29:00+08:00",
        "2026-11-11T09:29:01+08:00",
        [dict(ts_code="A", close=10, vol=100000, trade_time="2026-11-10 15:00:00")],
    )
    for n, vol in ((3, 1000), (4, 900)):
        write(
            f"{n}.json.gz",
            "rt_k",
            f"2026-11-11T09:30:0{n}+08:00",
            f"2026-11-11T09:30:0{n + 1}+08:00",
            [
                dict(
                    ts_code="A",
                    close=10,
                    vol=vol,
                    trade_time=f"2026-11-11 09:30:0{n - 1}",
                )
            ],
        )
    result = import_raw(rawdir, tmp_path / "import", "20261111")
    assert (
        result["counts"]["old_date_rows"] == 1
        and result["counts"]["volume_reversal"] == 1
    )
    assert result["trade_lag_ms"]["p50"] == 2000
    assert result["network_ms"]["p50"] == 1000
    source = ReplaySource(tmp_path / "import/quotes")
    batch = source.poll()
    assert batch.sent_at == timestamp("2026-11-11 09:30:03")
    assert batch.received_at - batch.sent_at == 1000
    main(
        [
            "replay",
            str(tmp_path / "import/quotes"),
            "--day",
            "20261111",
            "--script",
            str(tmp_path / "import/script.json"),
            "--output",
            str(tmp_path / "result.json"),
        ]
    )
    assert json.loads(capsys.readouterr().out)["phase"] == "closed"


def test_midday_restart_does_not_apply_dividend_to_new_buys(tmp_path):
    runner, client, clock, b = make_runner(tmp_path)
    for t in ("09:00:00", "09:30:00", "09:30:03"):
        at(clock, "2026-11-11", t)
        runner.step()
    assert b.account()["positions"][0]["total"] > 0
    before = b.account()["cash"]
    b.checkpoint()
    b.close()
    original = client.call
    called = []

    def call(name, params=None, fields=""):
        called.append(name)
        if name == "dividend":
            return [
                dict(
                    ts_code="A",
                    ex_date="20261111",
                    end_date="20251231",
                    div_proc="实施",
                    cash_div_tax=1,
                )
            ]
        return original(name, params, fields)

    client.call = call
    b = Broker(["A"], journal=tmp_path, quote_store=ParquetStore(tmp_path / "quotes"))
    runner = LiveRunner(
        b,
        LiveSource(client, ["A"]),
        Schedule(rows(fixture("trade_cal_2026q4"))),
        clock,
        ParquetStore(tmp_path / "quotes"),
    )
    at(clock, "2026-11-11", "11:30:00")
    runner.step()
    assert "dividend" not in called
    assert b.account()["cash"] == before
    b.close()


def test_preopen_dividend_failure_retries_before_trading(tmp_path):
    runner, client, clock, b = make_runner(tmp_path)
    for t in ("09:00:00", "09:30:00", "09:30:03", "15:01:00"):
        at(clock, "2026-11-11", t)
        runner.step()
    quantity = b.account()["positions"][0]["total"]
    before = b.account()["cash"]
    assert quantity > 0
    original = client.call
    attempts = []

    def call(name, params=None, fields=""):
        if name == "dividend":
            attempts.append(params)
            if len(attempts) == 1:
                raise SourceError("合成临时故障")
            return [
                dict(
                    ts_code="A",
                    ex_date="20261112",
                    end_date="20251231",
                    div_proc="实施",
                    cash_div_tax=1,
                )
            ]
        return original(name, params, fields)

    client.call = call
    at(clock, "2026-11-12", "09:00:00")
    with pytest.raises(SourceError, match="合成临时故障"):
        runner.step()
    assert b.account()["phase"] == "preopen"
    clock.sleep(3)
    runner.step()
    assert attempts == [{"ex_date": "20261112"}, {"ex_date": "20261112"}]
    assert b.account()["cash"] == before + quantity * money(1)
    b.close()
