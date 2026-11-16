"""来源：C8 任务书四个真实异常、判定反例；全部归档为合成数据。"""

import csv
import io
import json
import zipfile

import numpy as np
import polars as pl
import pytest

from ashare.data.crypto import metrics as m
from ashare.data.crypto import minute
from ashare.data.crypto.storage import RawStore, digest
from ashare.data.crypto.timeutil import DAY, iso, millis
from ashare.data.crypto.verify_metrics import _alignment, verify
from ashare.data.store import atomic_json
from ashare.paths import DATASETS
from test.crypto.test_minute import client_for

T = millis("2024-03-03")


def zipped(rows):
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["create_time", "symbol", *m.FIELDS])
    writer.writerows(rows)
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("synthetic.csv", buf.getvalue())
    return out.getvalue()


def raw_row(t, ratio=1, *, empty=False):
    return [
        iso(t).replace("T", " ").removesuffix("+00:00"),
        "BTCUSDT",
        10,
        1000,
        *(["", "", "", ""] if empty else [1, 2, 3, ratio]),
    ]


def test_duplicate_exact_only():
    row = raw_row(T)
    frame, duplicates = m.parse_archive(zipped([row, row]), "BTCUSDT", "2024-03-03")
    assert frame.height == 1 and duplicates == 1
    changed = row.copy()
    changed[2] = "10.0"  # 数值相等也不是原字符串完全一致。
    with pytest.raises(ValueError, match="重复"):
        m.parse_archive(zipped([row, changed]), "BTCUSDT", "2024-03-03")


def test_empty_strings_are_null_and_marked():
    rows = [raw_row(T + i * m.STEP, empty=i == 1) for i in range(288)]
    frame, _ = m.parse_archive(zipped(rows), "BTCUSDT", "2024-03-03")
    error = np.tile([0.0001, 0.4], (288, 1))
    error[1] = np.nan
    out = m.normalize(frame, error)
    assert out["top_account_ratio"][:3].to_list() == [1, None, 1]
    assert out["quality"].to_list() == [0, 1] + [0] * 286
    assert out["available_at"].to_list() == [T + i * m.STEP for i in range(288)]


def test_missing_rows_not_filled():
    frame, _ = m.parse_archive(
        zipped([raw_row(T), raw_row(T + 2 * m.STEP)]), "BTCUSDT", "2024-03-03"
    )
    assert frame.height == 2
    assert m.gaps(frame["create_time"], T, T + 3 * m.STEP) == [
        dict(start=T + m.STEP, end=T + 2 * m.STEP, rows=1)
    ]


def assert_alignment(times, error, expected, inherited):
    a, i = m.resolve_alignment(times, error)
    assert a.tolist() == expected and i.tolist() == inherited
    assert _alignment(times, error, m.MATCH_ERROR, m.REJECT_ERROR)[:2] == (
        expected,
        inherited,
    )


def daily_evidence(votes, day_offsets=None):
    offsets = range(len(votes)) if day_offsets is None else day_offsets
    return (
        np.concatenate([T + d * DAY + np.arange(288) * m.STEP for d in offsets]),
        np.repeat(np.asarray(votes, float), 288, axis=0),
    )


def test_alignment_registered_day_boundary():
    # 来源：第二次复核，登记的 UTC 日界决定逐行对齐。
    times, error = daily_evidence([[0.001, 0.4], [0.4, 0.001]], [0, 1])
    assert_alignment(times, error, [0] * 288 + [1] * 288, [False] * 576)
    frame = pl.DataFrame(
        {"create_time": times, **{f: [1.0] * 576 for f in m.FIELDS.values()}}
    )
    result = m.normalize(frame, error)
    np.testing.assert_equal(
        result["time"], times - np.r_[np.full(288, m.STEP), np.zeros(288)]
    )
    np.testing.assert_equal(
        result["available_at"], times + np.r_[np.zeros(288), np.full(288, m.STEP)]
    )


@pytest.mark.parametrize(
    "error",
    [
        [[0.001, 0.4], [0.4, 0.001], [0.001, 0.4]],
        [[0.001, 0.4], [0.4, 0.001], [0.001, 0.4], [0.4, 0.001]],
        [[0.001, 0.4], [0.02, 0.03], [0.4, 0.001]],
    ],
)
def test_alignment_rejects_unregistered_switches(error):
    times, error = daily_evidence(error, range(-10, -10 + len(error)))
    with pytest.raises(ValueError, match="REGISTERED_ALIGNMENT"):
        m.alignment_days(times, error)
    with pytest.raises(ValueError, match="REGISTERED_ALIGNMENT"):
        _alignment(times, error, 0.01, 0.1)


@pytest.mark.parametrize(
    "error",
    [
        [0.01, 0.4],
        [0.001, 0.1],
        [0.01 - 1e-12, 0.1 - 1e-12],
        [0.01 + 1e-12, 0.4],
    ],
)
def test_threshold_gray_zone_does_not_vote(error):
    # 来源：审查阈值严格不等；使用字面量防止常量与断言一起漂移。
    times, error = daily_evidence([error])
    assert_alignment(times, error, [0] * 288, [True] * 288)


def test_threshold_strict_success_and_same_segment_inheritance():
    for votes, inherited in [
        (
            [[0.01 - 1e-12, 0.1 + 1e-12], [np.nan, np.nan], [0.001, 0.4]],
            [False, True, False],
        ),
        (
            [[0.001, 0.4], [0.02, 0.03], [np.nan, np.nan], [0.001, 0.4]],
            [False, True, True, False],
        ),
    ]:
        times, error = daily_evidence(votes, range(-10, -10 + len(votes)))
        assert_alignment(
            times, error, [0] * len(times), np.repeat(inherited, 288).tolist()
        )


@pytest.mark.parametrize("neighbors", [False, True])
def test_noon_switch_is_not_a_day_boundary(neighbors):
    # 来源：审查反例，一天中午切换，两种全天中位数均不过。
    times = np.arange(T, T + DAY, m.STEP)
    error = np.array([[0.001, 0.4]] * 144 + [[0.4, 0.001]] * 144)
    if neighbors:
        times = np.concatenate(
            [
                T - DAY + np.arange(288) * m.STEP,
                times,
                T + DAY + np.arange(288) * m.STEP,
            ]
        )
        error = np.vstack(
            [np.tile([0.001, 0.4], (288, 1)), error, np.tile([0.4, 0.001], (288, 1))]
        )
    with pytest.raises(ValueError, match="SWITCH_WINDOW"):
        m.alignment_days(times, error)
    with pytest.raises(ValueError, match="SWITCH_WINDOW"):
        _alignment(times, error, 0.01, 0.1)


def test_daily_median_uses_paired_rows_and_keeps_outliers():
    # 远离登记切换窗口：160 行双边可比，40 个孤立反向点，128 行单边可比。
    times = T - 10 * DAY + np.arange(288) * m.STEP
    error = np.vstack(
        [
            np.tile([[0.001, 0.4], [0.002, 0.3], [0.004, 0.6], [0.4, 0.001]], (40, 1)),
            np.tile([np.nan, 0.0], (128, 1)),
        ]
    )
    assert_alignment(times, error, [0] * 288, [False] * 288)
    days = m.alignment_days(times, error)
    assert days[0]["both_compared"] == 160
    assert days[0]["medians"] == [0.003, 0.35]


def test_mismatch_threshold_and_correct_rows():
    times = T + np.arange(288) * m.STEP
    frame = pl.DataFrame(
        {"create_time": times, **{f: [1.0] * 288 for f in m.FIELDS.values()}}
    )
    error = np.tile([0.001, 0.4], (288, 1))
    error[3] = [0.05, 0.4]
    error[-1] = [0.05 + 1e-12, 0.001]
    out = m.normalize(frame, error)
    assert out["alignment"].to_list() == [0] * 288
    assert out["quality"].to_list() == [0] * 287 + [m.TAKER_MISMATCH]
    assert out["available_at"][-1] == T + DAY - m.STEP


def test_missing_midnight_row_reports_day_boundary():
    times, error = daily_evidence([[0.001, 0.4], [0.4, 0.001]])
    days = m.alignment_days(np.delete(times, 288), np.delete(error, 288, axis=0))
    assert [d["day"] for d in days] == [T, T + DAY]
    assert [d["alignment"] for d in days] == [0, 1]


def test_minute_window_missing_zero_volume_and_recalculation():
    times = np.arange(T, T + 3 * m.STEP, 60000)
    frame = pl.DataFrame(
        dict(
            open_time=times,
            volume=np.repeat([3.0, 0.0, 10.0], 5),
            taker_buy_volume=np.repeat([1.0, 0.0, 4.0], 5),
        )
    )
    ratios = m.minute_windows(frame)
    assert ratios == {T: 0.5, T + 2 * m.STEP: 2 / 3}
    assert T not in m.minute_windows(frame.slice(1))
    metrics, _ = m.parse_archive(
        zipped([raw_row(T + m.STEP, 0.5)]), "BTCUSDT", "2024-03-03"
    )
    error = m.errors(metrics, ratios)
    assert error[0, 0] == 0 and np.isnan(error[0, 1])


@pytest.mark.parametrize("change", ["grid", "symbol", "negative", "inf"])
def test_invalid_raw_rejected(change):
    row = raw_row(T)
    if change == "grid":
        row[0] = iso(T + 1)
    elif change == "symbol":
        row[1] = "ETHUSDT"
    else:
        row[-1] = -1 if change == "negative" else "inf"
    with pytest.raises(
        ValueError, match="时间|标的" if change in ("grid", "symbol") else "数值"
    ):
        m.parse_archive(zipped([row]), "BTCUSDT", "2024-03-03")


@pytest.mark.parametrize("change", ["duplicate_header", "extra_cell"])
def test_ambiguous_csv_columns_rejected(change):
    # 来源：开发复核，DictReader 会覆盖重复表头并收容多余单元格。
    header = ["create_time", "symbol", *m.FIELDS]
    row = raw_row(T)
    if change == "duplicate_header":
        header.append("sum_open_interest")
    row.append(10)
    text = io.StringIO()
    writer = csv.writer(text)
    writer.writerows([header, row])
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("synthetic.csv", text.getvalue())
    with pytest.raises(ValueError, match="表头|列数"):
        m.parse_archive(buf.getvalue(), "BTCUSDT", "2024-03-03")


def test_independent_decoder_respects_explicit_timezone():
    # 来源：开发复核，独立解码不能用 replace 抹掉原字符串的 UTC 偏移。
    from ashare.data.crypto.verify_metrics import _decode

    row = raw_row(T)
    row[0] = "2024-03-03T08:00:00+08:00"
    payload = zipped([row])
    assert m.parse_archive(payload, "BTCUSDT", "2024-03-03")[0]["create_time"][0] == T
    assert list(_decode(payload, "BTCUSDT", "2024-03-03")[0]) == [T]


def fixture(root):
    mr, kr = root / "metrics", root / "minute"
    # 两天、实际切换时刻为 2024-03-04 00:00；相邻窗口的比值显著不同。
    times = np.arange(T - m.STEP, T + 2 * DAY + m.STEP, 60000)
    rows = []
    for t in times:
        buy = 1 if (t // m.STEP) % 2 else 3
        rows.append(f"{t},100,101,99,100,4,{t + 59999},400,1,{buy},{buy * 100},0")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("synthetic.csv", "\n".join(rows) + "\n")
    key = minute.archive_key("usdm", "klines", "BTCUSDT", "2024-03")
    with client_for({key: buf.getvalue()})[0] as client:
        minute.sync(
            root=kr,
            start=T,
            end=T + 2 * DAY,
            now=T + 3 * DAY,
            series=[("BTCUSDT", "usdm", "klines")],
            client=client,
        )
    minute.build(root=kr)
    refs = {}
    store = RawStore(mr)
    for day in (T, T + DAY):
        metrics_rows = []
        for t in range(day, day + DAY, m.STEP):
            start = t - m.STEP if t < T + DAY else t
            ratio = 1 / 3 if (start // m.STEP) % 2 else 3
            metrics_rows.append(raw_row(t, ratio))
        payload = zipped(metrics_rows)
        key = m.archive_key("BTCUSDT", iso(day)[:10])
        ref = store.put(
            key,
            payload,
            checksum=digest(payload),
            checksum_text=f"{digest(payload)}  {key.rsplit('/', 1)[-1]}\n",
        )
        refs[key] = {
            f: ref[f] for f in ("sha256", "bytes", "checksum", "checksum_text")
        }
    store.close()
    atomic_json(
        mr / "metrics_plan.json",
        dict(definition=m.DEFINITION, archives=refs, failures=[]),
    )
    return mr, kr


def test_build_verify_content_version_and_cold_registration(tmp_path):
    assert not DATASETS["crypto_metrics"].hot
    mr, kr = fixture(tmp_path)
    version = m.build(root=mr, minute_root=kr)
    assert m.build(root=mr, minute_root=kr) == version
    meta = m.metadata(root=mr)
    assert meta["quality"]["BTCUSDT"]["switch_at"] == [T + DAY]
    assert verify(root=mr, minute_root=kr)["ok"]
    frame = m.read("BTCUSDT", T - m.STEP, T + 2 * DAY, root=mr, version=version)
    assert frame.height == 576
    assert frame["available_at"].max() == T + 2 * DAY
    path = mr / next(iter(meta["partitions"].values()))["file"]
    path.write_bytes(path.read_bytes() + b"bad")
    with pytest.raises(ValueError, match="parquet"):
        m.read("BTCUSDT", T, T + DAY, root=mr)
    with pytest.raises(ValueError, match="parquet"):
        verify(root=mr, minute_root=kr)


def test_sync_resume_checksum_and_failure_guard(tmp_path):
    key = m.archive_key("BTCUSDT", "2024-03-03")
    payload = zipped([raw_row(T)])
    with client_for({key: payload})[0] as client:
        client.listing = lambda _: iter([key])
        first = m.sync(root=tmp_path, symbols=["BTCUSDT"], client=client, workers=1)
        second = m.sync(root=tmp_path, symbols=["BTCUSDT"], client=client, workers=1)
    assert first["downloaded"] == 1 and second["downloaded"] == 0
    with client_for({key: payload}, bad_checksum=True)[0] as client:
        client.listing = lambda _: iter([key])
        with pytest.raises(ValueError, match="同步失败"):
            m.sync(
                root=tmp_path,
                symbols=["BTCUSDT"],
                client=client,
                workers=1,
                refresh=True,
            )
    assert json.loads((tmp_path / "metrics_plan.json").read_text())["failures"]


def test_full_missing_month_error_shape_and_gap():
    # 来源：开发复核，整月无归档也必须保留缺口，而不是构造一维空误差数组。
    empty = pl.DataFrame(
        schema={"create_time": pl.Int64, **dict.fromkeys(m.FIELDS.values(), pl.Float64)}
    )
    assert m.errors(empty, {}).shape == (0, 2)
    assert m.gaps([], T, T + DAY) == [dict(start=T, end=T + DAY, rows=288)]


def test_unanchored_unknown_days_use_registry_without_extrapolation():
    # 来源：第二次复核，不再从中间证据向两侧推断；登记常量覆盖未知日。
    error = [[np.nan, np.nan], [0.4, 0.0001], [np.nan, np.nan]]
    times, error = daily_evidence(error)
    assert_alignment(
        times, error, [0] * 288 + [1] * 576, [True] * 288 + [False] * 288 + [True] * 288
    )


def test_normalized_gap_includes_timestamp_transition(tmp_path):
    # 来源：开发复核，旧23:55为结束、新00:00为开始，中间5分钟没有观测。
    mr, kr = fixture(tmp_path)
    m.build(root=mr, minute_root=kr)
    q = m.metadata(root=mr)["quality"]["BTCUSDT"]
    assert q["normalized_gaps"] == [dict(start=T + DAY - m.STEP, end=T + DAY, rows=1)]
    assert q["gaps"] == []


def test_read_detects_corrupt_raw_and_verifier_detects_resigned_quality(tmp_path):
    # 来源：C7 内容版本合同，读取必须校验源对象；独立审计不能仅验证自报哈希。
    mr, kr = fixture(tmp_path)
    version = m.build(root=mr, minute_root=kr)
    meta = m.metadata(root=mr)
    raw_path = mr / "objects" / next(iter(meta["archives"].values()))["sha256"]
    original = raw_path.read_bytes()
    raw_path.write_bytes(original + b"damage")
    with pytest.raises(ValueError, match="CHECKSUM"):
        m.read("BTCUSDT", T, T + DAY, root=mr)
    raw_path.write_bytes(original)
    meta["quality"]["BTCUSDT"]["duplicates"] += 1
    from ashare.data.crypto.storage import encoded

    body = {k: v for k, v in meta.items() if k != "version"}
    damaged = digest(encoded(body))
    atomic_json(mr / "versions" / f"{damaged}.json", dict(body, version=damaged))
    result = verify(root=mr, minute_root=kr, version=damaged)
    assert not result["ok"] and result["differences"] == 1
    report = json.loads((mr / "verification" / f"{damaged}.json").read_text())
    assert report["details"] == [dict(symbol="BTCUSDT", reason="quality_summary")]
    assert verify(root=mr, minute_root=kr, version=version)["ok"]


def test_zero_buy_volume_still_reconciles_and_wrong_ratio_rejected():
    # 来源：开发复核，零主动买量但有成交，不等于无成交；不能跳过对账。
    frame, _ = m.parse_archive(
        zipped([raw_row(T, 0), raw_row(T + m.STEP, 1)]), "BTCUSDT", "2024-03-03"
    )
    error = m.errors(frame, {T - m.STEP: 0.0, T: 0.0, T + m.STEP: 0.0})
    assert error[0].tolist() == [0.0, 0.0]
    assert np.all(error[1] > m.REJECT_ERROR)


def test_interrupted_sync_marks_unfinished_request(tmp_path):
    # 来源：C7 续传合同；进程中断后不得把未完成请求误称为可构建完整计划。
    key = m.archive_key("BTCUSDT", "2024-03-03")

    class Interrupted:
        def listing(self, prefix):
            return [key]

        def get(self, *a, **kw):
            raise KeyboardInterrupt("synthetic interruption")

    with pytest.raises(KeyboardInterrupt, match="synthetic interruption"):
        m.sync(root=tmp_path, symbols=["BTCUSDT"], client=Interrupted(), workers=1)
    plan = json.loads((tmp_path / "metrics_plan.json").read_text())
    assert plan["pending"] == [key]
    with client_for({key: zipped([raw_row(T)])})[0] as client:
        client.listing = lambda _: iter([key])
        m.sync(root=tmp_path, symbols=["BTCUSDT"], client=client, workers=1)
    assert json.loads((tmp_path / "metrics_plan.json").read_text())["pending"] == []


def test_build_retains_mismatch_and_independent_verify_checks_bit(tmp_path):
    # 来源：d402835，离群点保留；独立校验须核对质量位而非拒绝数据。
    mr, kr = fixture(tmp_path)
    plan = json.loads((mr / "metrics_plan.json").read_text())
    key = next(iter(plan["archives"]))
    payload = m._raw(mr, key, plan["archives"][key])
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        rows = list(csv.reader(io.StringIO(z.read(z.namelist()[0]).decode())))[1:]
    rows[10][-1] = "10000"
    blob = zipped(rows)
    store = RawStore(mr)
    ref = store.put(
        key,
        blob,
        checksum=digest(blob),
        checksum_text=f"{digest(blob)}  {key.rsplit('/', 1)[-1]}\n",
    )
    store.close()
    plan["archives"][key] = {
        f: ref[f] for f in ("sha256", "bytes", "checksum", "checksum_text")
    }
    atomic_json(mr / "metrics_plan.json", plan)
    version = m.build(root=mr, minute_root=kr)
    frame = m.read("BTCUSDT", T - m.STEP, T + 2 * DAY, root=mr)
    assert frame.height == 576
    assert frame.filter(pl.col("quality") == m.TAKER_MISMATCH)[
        "create_time"
    ].to_list() == [T + 10 * m.STEP]
    report = verify(root=mr, minute_root=kr, version=version)
    assert report["ok"] and report["differences"] == 0
    q = m.metadata(root=mr)["quality"]["BTCUSDT"]
    assert q["mismatches"] == 1 and q["mismatch_ratio"] == 1 / 576
    assert q["days"][0]["mismatches"] == 1
    assert q["years"]["2024"]["mismatches"] == 1
    details = json.loads((mr / "verification" / f"{version}.json").read_text())[
        "taker_mismatches"
    ]
    assert len(details) == 1 and details[0]["create_time"] == T + 10 * m.STEP


def test_status_distinguishes_downloaded_from_published(tmp_path):
    # 来源：真实原档完成但严格构建拒绝，status 必须区分下载与可读版本。
    atomic_json(
        tmp_path / "metrics_plan.json",
        dict(
            definition=m.DEFINITION,
            archives={"synthetic": {}},
            pending=["missing"],
            failures=[],
        ),
    )
    result = m.status(root=tmp_path)
    assert (
        not result["built"] and result["raw_archives"] == 1 and result["pending"] == 1
    )


def test_binary_checksum_marker_builds_and_verifies(tmp_path):
    # 来源：最终一致性复核，下载/构建已支持 sha256sum 的 * 标记，独立校验也须支持。
    mr, kr = fixture(tmp_path)
    path = mr / "metrics_plan.json"
    plan = json.loads(path.read_text())
    for key, ref in plan["archives"].items():
        ref["checksum_text"] = f"{ref['sha256']} *{key.rsplit('/', 1)[-1]}\n"
    atomic_json(path, plan)
    version = m.build(root=mr, minute_root=kr)
    assert verify(root=mr, minute_root=kr, version=version)["ok"]


def test_whole_missing_day_uses_registered_alignment():
    times, error = daily_evidence([[0.001, 0.4], [0.001, 0.4]], [-10, -8])
    days = m.alignment_days(times, error)
    assert days[1] == dict(
        day=T - 9 * DAY,
        rows=0,
        both_compared=0,
        medians=[],
        alignment=0,
        inherited=1,
        opposite_rows=0,
        opposite_max_run=0,
    )
    assert_alignment(times, error, [0] * 576, [False] * 576)
    error[288:] = [0.4, 0.001]
    with pytest.raises(ValueError, match="REGISTERED_ALIGNMENT"):
        m.alignment_days(times, error)
    with pytest.raises(ValueError, match="REGISTERED_ALIGNMENT"):
        _alignment(times, error, 0.01, 0.1)


@pytest.mark.parametrize(
    "field",
    [
        "day_median",
        "day_inheritance",
        "year_mismatch",
        "row_quality",
        "segment_error",
        "schema",
    ],
)
def test_independent_verifier_rejects_resigned_daily_evidence(tmp_path, field):
    # 来源：审查要求独立判定；哈希重签不能掩盖错误的日统计/年统计/质量位。
    from ashare.data.crypto.storage import encoded

    mr, kr = fixture(tmp_path)
    m.build(root=mr, minute_root=kr)
    meta = m.metadata(root=mr)
    q = meta["quality"]["BTCUSDT"]
    if field == "day_median":
        q["days"][0]["medians"][0] += 0.001
    elif field == "day_inheritance":
        q["days"][0]["inherited"] = 1
    elif field == "year_mismatch":
        q["years"]["2024"]["mismatches"] += 1
    elif field == "segment_error":
        q["segments"][0]["errors"]["0"][2] += 0.001
    else:
        part_id, part = next(iter(meta["partitions"].items()))
        frame = pl.read_parquet(mr / part["file"])
        frame = frame.with_columns(
            pl.col("quality").cast(pl.Int64)
            if field == "schema"
            else (pl.col("quality") + m.TAKER_MISMATCH).alias("quality")
        )
        buf = io.BytesIO()
        frame.write_parquet(buf, compression="zstd")
        payload = buf.getvalue()
        sha = digest(payload)
        rel = f"metrics/{part_id}/{sha}.parquet"
        (mr / rel).write_bytes(payload)
        part.update(file=rel, sha256=sha, bytes=len(payload))
    body = {k: v for k, v in meta.items() if k != "version"}
    version = digest(encoded(body))
    atomic_json(mr / "versions" / f"{version}.json", dict(body, version=version))
    if field == "schema":
        with pytest.raises(ValueError, match="schema"):
            verify(root=mr, minute_root=kr, version=version)
    else:
        result = verify(root=mr, minute_root=kr, version=version)
        assert not result["ok"] and result["differences"] > 0
        report = json.loads((mr / "verification" / f"{version}.json").read_text())
        assert {d["reason"] for d in report["details"]} == {
            "normalized_fields" if field == "row_quality" else "quality_summary"
        }


def test_raw_v1_plan_rebuilds_into_daily_version(tmp_path):
    # 来源：本轮迁移，原档格式未变；v1 下载计划可复用，新内容版本须记录新规则。
    mr, kr = fixture(tmp_path)
    plan = json.loads((mr / "metrics_plan.json").read_text())
    plan["definition"] = "crypto-metrics-v1"
    atomic_json(mr / "metrics_plan.json", plan)
    m.build(root=mr, minute_root=kr)
    assert m.metadata(root=mr)["definition"] == "crypto-metrics-v4-registered"
    assert verify(root=mr, minute_root=kr)["ok"]


def test_scoped_verification_is_explicit_and_not_full_coverage(tmp_path):
    # 来源：真实全库验证合并命令超时；分币续验不得把部分覆盖冒充全库通过。
    from ashare.data.crypto.storage import encoded

    mr, kr = fixture(tmp_path)
    m.build(root=mr, minute_root=kr)
    meta = m.metadata(root=mr)
    key, ref = next(iter(meta["archives"].items()))
    extra_key = key.replace("BTCUSDT", "ETHUSDT")
    meta["registered_switches"]["ETHUSDT"] = 1709510400000
    meta["archives"][extra_key] = dict(
        ref, checksum_text=f"{ref['sha256']}  {extra_key.rsplit('/', 1)[-1]}\n"
    )
    part_id, part = next(iter(meta["partitions"].items()))
    meta["partitions"][part_id.replace("BTCUSDT", "ETHUSDT")] = part
    body = {k: v for k, v in meta.items() if k != "version"}
    version = digest(encoded(body))
    atomic_json(mr / "versions" / f"{version}.json", dict(body, version=version))
    report = verify(root=mr, minute_root=kr, version=version, symbols=["BTCUSDT"])
    assert report["ok"] and not report["complete"]
    assert set(report["symbols"]) == {"BTCUSDT"}
    assert report["version"] == version
    assert (mr / "verification" / f"{version}-BTCUSDT.json").exists()
    assert not (mr / "verification" / f"{version}.json").exists()
    with pytest.raises(ValueError, match="CSV date/grid/symbol"):
        verify(root=mr, minute_root=kr, version=version)
    for symbols in ([], ["INVALID"], ["BTCUSDT", "BTCUSDT"]):
        with pytest.raises(ValueError, match="invalid verification scope/coverage"):
            verify(root=mr, minute_root=kr, version=version, symbols=symbols)


def test_scoped_reports_merge_only_matching_full_coverage(tmp_path):
    # 来源：840 秒限时后分币合并；缺报告或版本/范围不符都不得发布完整通过记录。
    from bench.crypto_metrics_validation import collect

    mr, kr = fixture(tmp_path)
    version = m.build(root=mr, minute_root=kr)
    with pytest.raises(FileNotFoundError, match="BTCUSDT.json"):
        collect(mr, version)
    assert not (mr / "verification" / f"{version}.json").exists()
    verify(root=mr, minute_root=kr, version=version, symbols=["BTCUSDT"])
    path = mr / "verification" / f"{version}-BTCUSDT.json"
    report = json.loads(path.read_text())
    report["seconds"] = 1.0
    atomic_json(path, report)
    atomic_json(mr / "sync_report.json", dict(seconds=0.0))
    result = collect(mr, version)
    assert result["ok"] == 1 and result["built"] == 1 and result["differences"] == 0
    for key, value in [
        ("version", "f" * 64),
        ("minute_version", "f" * 64),
        ("symbols", {}),
        ("differences", 1),
    ]:
        atomic_json(path, dict(report, **{key: value}))
        with pytest.raises(
            ValueError,
            match="verification detail count mismatch"
            if key == "differences"
            else "verification scope/version mismatch",
        ):
            collect(mr, version)


def test_build_with_no_comparable_taker_rows(tmp_path):
    # 来源：合并前复核 recheck2_c8 P2-1。真实 BTCUSDT 2022-02 主动买卖比全空，
    # 构建区间内没有任何可比行时，汇总比例不能除零，应与每日、年度汇总一样记 0。
    mr, kr = fixture(tmp_path)
    plan = json.loads((mr / "metrics_plan.json").read_text())
    store = RawStore(mr)
    for key, ref in list(plan["archives"].items()):
        payload = m._raw(mr, key, ref)
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            rows = list(csv.reader(io.StringIO(z.read(z.namelist()[0]).decode())))[1:]
        for row in rows:
            row[-1] = ""
        blob = zipped(rows)
        new = store.put(
            key,
            blob,
            checksum=digest(blob),
            checksum_text=f"{digest(blob)}  {key.rsplit('/', 1)[-1]}\n",
        )
        plan["archives"][key] = {
            f: new[f] for f in ("sha256", "bytes", "checksum", "checksum_text")
        }
    store.close()
    atomic_json(mr / "metrics_plan.json", plan)
    m.build(root=mr, minute_root=kr)
    quality = m.metadata(root=mr)["quality"]["BTCUSDT"]
    assert quality["compared"] == 0
    assert quality["comparable_mismatch_ratio"] == 0.0
    assert verify(root=mr, minute_root=kr)["ok"]
