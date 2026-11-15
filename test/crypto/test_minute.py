"""C7：全合成归档，网络由 MockTransport 提供。"""

import hashlib
import io
import json
import zipfile

import httpx
import polars as pl
import pytest

from ashare.data.crypto import minute as m
from ashare.data.crypto.sources import PublicClient
from ashare.data.crypto.timeutil import DAY, HOUR, millis
from ashare.data.crypto.verify_minute import verify

T = millis("2025-03-01")


def zipped(times, *, micro=False, header=False, change=0):
    rows = []
    if header:
        rows.append(
            "open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore"
        )
    for i, t in enumerate(times):
        scale = 1000 if micro else 1
        rows.append(
            f"{t * scale},{100 + i + change},{105 + i + change},{95 + i + change},{102 + i + change},2,{(t + 60000) * scale - 1},200,3,1,100,0"
        )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("synthetic.csv", "\n".join(rows) + "\n")
    return buf.getvalue()


def client_for(payloads, *, bad_checksum=False, fail=None):
    calls = []
    sleeps = []

    def respond(request):
        key = request.url.path.lstrip("/")
        calls.append(key)
        if fail and fail(key):
            raise httpx.ConnectError("synthetic network failure", request=request)
        original = key.removesuffix(".CHECKSUM")
        if original not in payloads:
            return httpx.Response(404)
        payload = payloads[original]
        if key.endswith(".CHECKSUM"):
            sha = "0" * 64 if bad_checksum else hashlib.sha256(payload).hexdigest()
            return httpx.Response(
                200, content=f"{sha}  {original.rsplit('/', 1)[-1]}\n".encode()
            )
        return httpx.Response(200, content=payload)

    client = PublicClient(transport=httpx.MockTransport(respond), sleep=sleeps.append)
    return client, calls, sleeps


def sync_one(root, payloads, *, start=T, end=T + DAY, **kwargs):
    with client_for(payloads, **kwargs)[0] as client:
        return m.sync(
            root=root,
            start=start,
            end=end,
            now=end + DAY,
            series=[("BTCUSDT", "spot", "klines")],
            client=client,
        )


def built(root, times=None):
    times = [T, T + 60000] if times is None else times
    key = m.archive_key("spot", "klines", "BTCUSDT", "2025-03")
    sync_one(root, {key: zipped(times)})
    return m.build(root=root)


@pytest.mark.parametrize("micro", [False, True])
@pytest.mark.parametrize("header", [False, True])
def test_units_and_headers(micro, header):
    df = m.parse_archive(
        zipped([T, T + 60000], micro=micro, header=header),
        "klines",
        start=T,
        end=T + DAY,
    )
    assert df["open_time"].to_list() == [T, T + 60000]
    assert df["count"].to_list() == [3, 3]
    assert df["taker_buy_volume"].to_list() == [1, 1]


@pytest.mark.parametrize("time", [T // 1000, T * 1_000_000, T + 1, T * 1000 + 1])
def test_unknown_units_and_off_grid_rejected(time):
    with pytest.raises(ValueError):
        m.parse_archive(zipped([time]), "klines", start=T, end=T + DAY)


def test_duplicates_and_unsorted_raise():
    for times in ([T, T], [T + 60000, T]):
        with pytest.raises(ValueError, match="重复|递增"):
            m.parse_archive(zipped(times), "klines", start=T, end=T + DAY)


def test_gaps_are_full_month_segments(tmp_path):
    version = built(tmp_path, [T + 60000, T + 180000])
    meta = m.metadata(root=tmp_path, version=version)
    q = next(iter(meta["partitions"].values()))["quality"]
    assert q["expected_rows"] == 31 * 1440
    assert q["rows"] == 2
    assert q["gaps"] == [
        dict(start=T, end=T + 60000, minutes=1),
        dict(start=T + 120000, end=T + 180000, minutes=1),
        dict(start=T + 240000, end=millis("2025-04-01"), minutes=31 * 1440 - 4),
    ]
    assert verify(root=tmp_path)["ok"]


def test_daily_only_missing_days_and_resume(tmp_path):
    monthly = m.archive_key("spot", "klines", "BTCUSDT", "2025-03")
    d1 = m.archive_key("spot", "klines", "BTCUSDT", "2025-03-01", frequency="daily")
    d2 = m.archive_key("spot", "klines", "BTCUSDT", "2025-03-02", frequency="daily")
    with client_for(
        {monthly: zipped([T]), d1: zipped([T], change=100), d2: zipped([T + DAY])}
    )[0] as client:
        calls = []
        original = client.get
        client.get = lambda url, **kw: (calls.append(url), original(url, **kw))[1]
        args = dict(
            root=tmp_path,
            start=T,
            end=T + 2 * DAY,
            now=T + 3 * DAY,
            series=[("BTCUSDT", "spot", "klines")],
            client=client,
        )
        m.sync(**args)
        assert not any(d1 in url for url in calls)
        calls.clear()
        m.sync(**args)
        assert calls == []
    m.build(root=tmp_path)
    assert m.bars("BTCUSDT", "spot", "klines", T, T + 2 * DAY, root=tmp_path)[
        "open"
    ].to_list() == [100, 100]
    assert verify(root=tmp_path)["ok"]


def test_checksum_fails_and_no_partial_object(tmp_path):
    key = m.archive_key("spot", "klines", "BTCUSDT", "2025-03")
    with pytest.raises(m.SyncError) as exc:
        sync_one(tmp_path, {key: zipped([T])}, bad_checksum=True)
    assert exc.value.report["failures"][0]["key"] == key
    assert not list(tmp_path.glob("objects/*"))
    with pytest.raises(ValueError):
        m.build(root=tmp_path)


def test_interrupted_resume_and_bounded_backoff(tmp_path):
    key = m.archive_key("spot", "klines", "BTCUSDT", "2025-03-01", frequency="daily")
    with client_for({key: zipped([T])}, fail=lambda k: k == key)[0] as client:
        with pytest.raises(m.SyncError):
            m.sync(
                root=tmp_path,
                start=T,
                end=T + DAY,
                now=T + 2 * DAY,
                series=[("BTCUSDT", "spot", "klines")],
                client=client,
            )
    with client_for({key: zipped([T])})[0] as client:
        m.sync(
            root=tmp_path,
            start=T,
            end=T + DAY,
            now=T + 2 * DAY,
            series=[("BTCUSDT", "spot", "klines")],
            client=client,
        )
    m.build(root=tmp_path)
    assert verify(root=tmp_path)["ok"]
    client, calls, sleeps = client_for({}, fail=lambda k: True)
    with client, pytest.raises(httpx.ConnectError):
        client.get("https://data.binance.vision/synthetic.zip")
    assert len(calls) == 4
    assert all(s in sleeps for s in (1, 2, 4))


def test_cross_month_lazy_reads_and_integrity(tmp_path, monkeypatch):
    apr = millis("2025-04-01")
    for label, t in [
        ("2025-03", apr - 60000),
        ("2025-04", apr),
        ("2025-05", millis("2025-05-01")),
    ]:
        key = m.archive_key("spot", "klines", "BTCUSDT", label)
        sync_one(
            tmp_path, {key: zipped([t])}, start=t // DAY * DAY, end=t // DAY * DAY + DAY
        )
    version = m.build(root=tmp_path)
    paths = []
    original = pl.read_parquet

    def tracked(path, *args, **kwargs):
        paths.append(str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(pl, "read_parquet", tracked)
    df = m.bars("BTCUSDT", "spot", "klines", apr - 60000, apr + 60000, root=tmp_path)
    assert df["open_time"].to_list() == [apr - 60000, apr]
    assert len(paths) == 2 and not any("2025-05" in p for p in paths)
    meta = m.metadata(root=tmp_path, version=version)
    part = meta["partitions"]["spot/BTCUSDT/klines/2025-03"]
    (tmp_path / part["file"]).write_bytes(b"broken")
    with pytest.raises(ValueError, match="损坏"):
        m.bars("BTCUSDT", "spot", "klines", apr - 60000, apr, root=tmp_path)


def test_aggregate_ohlcv_and_incomplete():
    frame = m.parse_archive(
        zipped([T + i * 60000 for i in range(60)]), "klines", start=T, end=T + HOUR
    )
    hour = m.aggregate_hourly(frame).row(0, named=True)
    assert [hour[f] for f in ("open", "high", "low", "close")] == [100, 164, 95, 161]
    assert [
        hour[f]
        for f in (
            "volume",
            "quote_volume",
            "count",
            "taker_buy_volume",
            "taker_buy_quote_volume",
        )
    ] == [120, 12000, 180, 60, 6000]
    assert hour["minutes"] == 60 and hour["complete"]
    assert not m.aggregate_hourly(frame.head(59))["complete"].item()
    with pytest.raises(ValueError):
        m.aggregate_hourly(pl.concat([frame, frame.head(1)]))


def test_quality_notes_affect_hash_and_old_version_readable(tmp_path):
    v1 = built(tmp_path)
    v2 = m.build(root=tmp_path, quality_notes=[dict(reason="synthetic review")])
    assert v1 != v2
    assert (
        m.bars(
            "BTCUSDT", "spot", "klines", T, T + DAY, root=tmp_path, version=v1
        ).height
        == 2
    )
    assert verify(root=tmp_path, version=v1)["ok"]
    assert verify(root=tmp_path, version=v2)["ok"]
    path = tmp_path / "versions" / f"{v2}.json"
    meta = json.loads(path.read_text())
    meta["quality_notes"] = []
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="版本"):
        m.metadata(root=tmp_path)


@pytest.mark.parametrize("kind", ["mark", "index"])
def test_price_only_schema(tmp_path, kind):
    key = m.archive_key("usdm", kind, "ETHUSDT", "2025-03")
    with client_for({key: zipped([T], micro=True, header=True)})[0] as client:
        m.sync(
            root=tmp_path,
            start=T,
            end=T + DAY,
            now=T + 2 * DAY,
            series=[("ETHUSDT", "usdm", kind)],
            client=client,
        )
    m.build(root=tmp_path)
    df = m.bars("ETHUSDT", "usdm", kind, T, T + DAY, root=tmp_path)
    assert df.columns == ["open_time", "open", "high", "low", "close"]
    assert verify(root=tmp_path)["ok"]


def test_failure_is_not_cleared_by_unrelated_sync(tmp_path):
    """开发自审回归：续跑其他月份不能抹掉旧失败。"""
    failed = m.archive_key("spot", "klines", "BTCUSDT", "2025-03")
    with pytest.raises(m.SyncError):
        sync_one(tmp_path, {failed: zipped([T])}, bad_checksum=True)
    apr = millis("2025-04-01")
    key = m.archive_key("spot", "klines", "BTCUSDT", "2025-04")
    with pytest.raises(m.SyncError) as exc:
        sync_one(tmp_path, {key: zipped([apr])}, start=apr, end=apr + DAY)
    assert failed in [f["key"] for f in exc.value.report["failures"]]
    with pytest.raises(ValueError):
        m.build(root=tmp_path)
    sync_one(tmp_path, {failed: zipped([T])})
    m.build(root=tmp_path)
    assert verify(root=tmp_path)["ok"]


def test_mixed_time_units_even_within_same_month():
    data = zipped([T, (T + 60000) * 1000])
    with pytest.raises(ValueError, match="单位"):
        m.parse_archive(data, "klines", start=T, end=T + DAY)


def test_corrupt_raw_rejected_on_read_and_verify(tmp_path):
    built(tmp_path)
    meta = m.metadata(root=tmp_path)
    ref = next(iter(next(iter(meta["partitions"].values()))["archives"].values()))
    (tmp_path / "objects" / ref["sha256"]).write_bytes(b"not a zip")
    with pytest.raises(ValueError, match="损坏"):
        m.bars("BTCUSDT", "spot", "klines", T, T + DAY, root=tmp_path)
    assert not verify(root=tmp_path)["ok"]


def _republish(root, meta):
    """测试故意重签畸形产物，让独立校验必须检查语义而非只看哈希。"""
    from ashare.data.crypto.storage import digest, encoded
    from ashare.data.store import atomic_json

    meta.pop("version")
    version = digest(encoded(meta))
    meta["version"] = version
    atomic_json(root / "versions" / f"{version}.json", meta)
    atomic_json(root / "latest.json", dict(version=version))


@pytest.mark.parametrize(
    "damage", ["gap", "value", "missing_row", "duplicate", "schema", "checksum"]
)
def test_independent_verifier_rejects_resigned_damage(tmp_path, damage):
    from ashare.data.crypto.storage import digest

    built(tmp_path)
    meta = m.metadata(root=tmp_path)
    part_id, part = next(iter(meta["partitions"].items()))
    if damage == "gap":
        part["quality"]["gaps"] = []
    elif damage == "checksum":
        next(iter(part["archives"].values()))["checksum_text"] = "0" * 64 + " wrong.zip"
    else:
        df = pl.read_parquet(tmp_path / part["file"])
        if damage == "value":
            df = df.with_columns(pl.col("open") + 1)
        elif damage == "missing_row":
            df = df.head(1)
        elif damage == "duplicate":
            df = pl.concat([df, df.head(1)])
        elif damage == "schema":
            df = df.with_columns(pl.col("open_time").cast(pl.Float64))
        buf = io.BytesIO()
        df.write_parquet(buf)
        payload = buf.getvalue()
        part.update(sha256=digest(payload), bytes=len(payload))
        part["file"] = f"bars/{part_id}/{part['sha256']}.parquet"
        (tmp_path / part["file"]).write_bytes(payload)
    _republish(tmp_path, meta)
    assert not verify(root=tmp_path)["ok"]


def test_independent_verifier_does_not_call_builder_predicates(tmp_path, monkeypatch):
    built(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("shared builder predicate")

    for function in (
        "parse_archive",
        "_quality",
        "aggregate_hourly",
        "metadata",
        "_raw",
    ):
        monkeypatch.setattr(m, function, forbidden)
    assert verify(root=tmp_path)["ok"]


def test_new_monthly_replaces_daily_without_mutating_old_version(tmp_path):
    day = m.archive_key("spot", "klines", "BTCUSDT", "2025-03-01", frequency="daily")
    sync_one(tmp_path, {day: zipped([T])})
    v1 = m.build(root=tmp_path)
    month = m.archive_key("spot", "klines", "BTCUSDT", "2025-03")
    sync_one(tmp_path, {month: zipped([T], change=10), day: zipped([T])})
    v2 = m.build(root=tmp_path)
    assert v1 != v2
    assert (
        m.bars("BTCUSDT", "spot", "klines", T, T + DAY, root=tmp_path, version=v1)[
            "open"
        ].item()
        == 100
    )
    assert (
        m.bars("BTCUSDT", "spot", "klines", T, T + DAY, root=tmp_path)["open"].item()
        == 110
    )
    assert verify(root=tmp_path, version=v1)["ok"]
    assert verify(root=tmp_path, version=v2)["ok"]


def test_hourly_reconciliation_lists_every_difference():
    from ashare.data.crypto.verify_minute import reconcile_hourly

    frame = m.parse_archive(
        zipped([T + i * 60000 for i in range(60)]), "klines", start=T, end=T + HOUR
    )
    hourly = m.aggregate_hourly(frame).drop("minutes", "complete")
    assert reconcile_hourly(frame, hourly, "klines")["matches"]
    changed = hourly.with_columns(
        pl.col("open") + 1, pl.col("close") + 2, pl.col("volume") + 3
    )
    report = reconcile_hourly(frame, changed, "klines")
    assert report["material_differences"] == 3
    assert report["max_absolute"]["open"] == 1
    assert report["max_absolute"]["close"] == 2
    assert report["max_absolute"]["volume"] == 3
    assert len(report["differences"]) == 3
    missing = reconcile_hourly(frame.head(59), hourly, "klines")
    assert len(missing["incomplete_hours"]) == 1 and not missing["matches"]


def test_cli_commands_and_readonly(tmp_path):
    import subprocess
    import sys

    built(tmp_path)
    for command in ("status", "build", "verify"):
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "ashare.data.crypto.minute",
                command,
                "--root",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        json.loads(result.stdout)
    before = {
        p.relative_to(tmp_path): p.read_bytes()
        for p in tmp_path.rglob("*")
        if p.is_file()
    }
    m.bars("BTCUSDT", "spot", "klines", T, T + DAY, root=tmp_path)
    verify(root=tmp_path)
    after = {
        p.relative_to(tmp_path): p.read_bytes()
        for p in tmp_path.rglob("*")
        if p.is_file()
    }
    assert before == after


def test_failed_month_preserves_original_requested_days(tmp_path):
    """开发自审回归：失败月先不发计划时，缩短窗口续跑会遗忘原请求。"""
    month = m.archive_key("spot", "klines", "BTCUSDT", "2025-03")
    with pytest.raises(m.SyncError):
        sync_one(tmp_path, {month: zipped([T])}, end=T + 2 * DAY, bad_checksum=True)
    day1 = m.archive_key("spot", "klines", "BTCUSDT", "2025-03-01", frequency="daily")
    day2 = m.archive_key("spot", "klines", "BTCUSDT", "2025-03-02", frequency="daily")
    sync_one(tmp_path, {day1: zipped([T]), day2: zipped([T + DAY])})
    m.build(root=tmp_path)
    assert (
        m.bars("BTCUSDT", "spot", "klines", T, T + 2 * DAY, root=tmp_path).height == 2
    )


def test_review_empty_reconciliation_is_not_a_match():
    """来源：C7 独立审查 P2-1。没有任何小时被比较时不能报对账通过。"""
    from ashare.data.crypto.verify_minute import reconcile_hourly

    frame = m.parse_archive(
        zipped([T + i * 60000 for i in range(60)]), "klines", start=T, end=T + HOUR
    )
    hourly = m.aggregate_hourly(frame).drop("minutes", "complete")
    report = reconcile_hourly(frame.head(0), hourly.head(0), "klines")
    assert report["compared_hours"] == 0
    assert not report["matches"]


def test_review_bars_end_is_exclusive_within_and_across_months(tmp_path):
    """来源：C7 独立审查 P2-2。终点恰有记录时不能返回，同月与跨月都要守住。"""
    apr = millis("2025-04-01")
    for label, times in [
        ("2025-03", [apr - 120000, apr - 60000]),
        ("2025-04", [apr, apr + 60000]),
    ]:
        key = m.archive_key("spot", "klines", "BTCUSDT", label)
        t = times[0]
        sync_one(
            tmp_path,
            {key: zipped(times)},
            start=t // DAY * DAY,
            end=t // DAY * DAY + DAY,
        )
    m.build(root=tmp_path)
    same = m.bars("BTCUSDT", "spot", "klines", apr - 120000, apr - 60000, root=tmp_path)
    assert same["open_time"].to_list() == [apr - 120000]
    cross = m.bars("BTCUSDT", "spot", "klines", apr - 60000, apr + 60000, root=tmp_path)
    assert cross["open_time"].to_list() == [apr - 60000, apr]
    edge = m.bars("BTCUSDT", "spot", "klines", apr - 60000, apr, root=tmp_path)
    assert edge["open_time"].to_list() == [apr - 60000]
