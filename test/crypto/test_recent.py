"""C5 任务书及设计审查 6/12：全部使用独立常数合成数据。"""

import copy
import importlib

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ashare.data import crypto
from ashare.data.crypto.timeutil import HOUR, millis
from ashare.data.crypto.verify import verify_frame

NAME = "usdm:PERPUSDT"
START = millis("2024-07-01")


@pytest.fixture
def recent(fixture, monkeypatch):
    p = copy.deepcopy(fixture["profiles"][0])
    p["listed_at"] = START
    bars = [
        dict(
            time=START + h * HOUR,
            open=100.0,
            high=102.0,
            low=99.0,
            close=101.0,
            volume=10.0,
            quote_volume=1000.0,
            taker_buy_quote_volume=500.0,
            source="archive",
        )
        for h in range(49)
    ]
    rates = [
        dict(time=START + h * HOUR, interval=8, rate=0.0001, source="archive")
        for h in (0, 8, 16)
    ]
    prices = [
        dict(fundingTime=START + h * HOUR, fundingRate="0.0001", markPrice="100")
        for h in (0, 8, 16, 24, 32, 40, 48)
    ]
    data = {
        NAME: dict(
            klines=bars,
            markPriceKlines=copy.deepcopy(bars),
            indexPriceKlines=copy.deepcopy(bars),
            fundingRate=rates,
            funding_prices=prices,
        )
    }
    plan = dict(
        profiles=[p],
        events=[],
        start=START,
        end=START + 48 * HOUR,
        raw_version=None,
        funding_info=dict(fetched_at=START + 50 * HOUR, records=[], sha256="synthetic"),
    )
    module = importlib.import_module("ashare.data.crypto.build")
    monkeypatch.setattr(
        module, "load_plan", lambda _: (copy.deepcopy(plan), copy.deepcopy(data))
    )
    return plan, data


def inferred(tmp_path):
    return crypto.build(tmp_path, allow_inferred_funding=True)


def test_strict_unchanged_default_and_separate_latest(tmp_path, recent):
    """来源 C5：不开选项逐位不变，建推断版不能污染默认读取。"""
    from test.crypto.test_c5_review_r1 import assert_legacy

    strict = crypto.build(tmp_path)
    strict_frame = strict.read()
    assert_legacy(strict_frame, "single")
    assert strict.meta["end"] == START + 23 * HOUR
    extended = inferred(tmp_path)
    assert extended.version != strict.version
    assert extended.meta["end"] == START + 48 * HOUR
    assert extended.meta["allow_inferred_funding"] is True
    assert crypto.open(tmp_path).version == strict.version
    assert (
        crypto.open(tmp_path, allow_inferred_funding=True).version == extended.version
    )
    assert crypto.open(tmp_path, extended.version).version == extended.version
    assert_frame_equal(
        extended.read(end=START + 23 * HOUR), strict_frame, check_exact=True
    )
    assert crypto.verify(tmp_path, allow_inferred_funding=True)["verdict"] == "pass"


def test_equal_intervals_and_quality(tmp_path, recent):
    ds = inferred(tmp_path)
    rows = ds.read().filter((pl.col("quality") & 4096) != 0)
    assert rows["time"].to_list() == [START + h * HOUR for h in (23, 31, 39, 47)]
    assert rows["funding_interval_hours"].to_list() == [8.0] * 4
    assert rows["funding"].to_list() == [0.0001] * 4
    assert crypto.verify(tmp_path, ds.version)["errors"] == []


@pytest.mark.parametrize(
    "case,expected",
    [
        ("missing_middle", 24),
        ("changed_4h", 24),
        ("info_changed", 23),
        ("info_missing_4h", 19),
        ("missing_last", 40),
        ("missing_mark", 24),
        ("off_grid", 24),
        ("jitter", 24),
        ("missing_snapshot", 23),
    ],
)
def test_stop_at_first_untrusted_settlement(tmp_path, recent, case, expected):
    """来源 C5/审查 12：坏记录后即便恢复正常也不能跨过去。"""
    plan, data = recent
    rows = data[NAME]["funding_prices"]
    if case == "missing_middle":
        rows[:] = [r for r in rows if r["fundingTime"] != START + 32 * HOUR]
    elif case == "changed_4h":
        rows.insert(
            4,
            dict(fundingTime=START + 28 * HOUR, fundingRate="0.0001", markPrice="100"),
        )
    elif case == "info_changed":
        plan["funding_info"]["records"] = [
            dict(symbol="PERPUSDT", fundingIntervalHours=4)
        ]
    elif case == "info_missing_4h":
        data[NAME]["fundingRate"] = [
            dict(time=START + h * HOUR, interval=4, rate=0.0001)
            for h in (0, 4, 8, 12, 16)
        ]
        rows[:] = [
            dict(fundingTime=START + h * HOUR, fundingRate="0.0001", markPrice="100")
            for h in range(0, 49, 4)
        ]
    elif case == "missing_last":
        rows.pop()
    elif case == "missing_mark":
        rows[4]["markPrice"] = ""
    elif case == "off_grid":
        rows[4]["fundingTime"] += HOUR
    elif case == "jitter":
        rows[4]["fundingTime"] += 1001
    else:
        plan.pop("funding_info")
    ds = inferred(tmp_path)
    assert ds.meta["end"] == START + expected * HOUR
    assert crypto.verify(tmp_path, ds.version)["errors"] == []


def test_info_listed_same_and_daily_cap(tmp_path, recent):
    plan, data = recent
    plan["funding_info"]["records"] = [
        dict(symbol="PERPUSDT", fundingIntervalHours=8, updateTime=0)
    ]
    data[NAME]["indexPriceKlines"] = data[NAME]["indexPriceKlines"][:35]
    ds = inferred(tmp_path)
    assert ds.meta["end"] == START + 35 * HOUR
    assert crypto.verify(tmp_path, ds.version)["errors"] == []


def test_verifier_independent_conditions_and_terminal(tmp_path, recent):
    ds = inferred(tmp_path)
    frame, inputs = ds.read(), ds.inputs()
    for change in ("spacing", "info", "price", "terminal", "flag"):
        bad = copy.deepcopy(inputs)
        mutant = frame
        if change == "spacing":
            bad["data"][NAME]["funding_prices"].pop(4)
        elif change == "info":
            bad["recent"]["funding_info"]["records"] = [
                dict(symbol="PERPUSDT", fundingIntervalHours=4)
            ]
        elif change == "price":
            bad["data"][NAME]["funding_prices"][4]["markPrice"] = ""
        elif change == "terminal":
            bad["data"][NAME]["indexPriceKlines"] = bad["data"][NAME][
                "indexPriceKlines"
            ][:35]
        else:
            mutant = frame.with_columns((pl.col("quality") & ~4096).alias("quality"))
        report = verify_frame(mutant, bad)
        assert any(e["check"].startswith("inferred_") for e in report["errors"]), change


@pytest.mark.parametrize("change", [None, "rate", "interval", "price", "time"])
def test_archive_reconciliation_keeps_old_and_does_not_block(tmp_path, recent, change):
    """来源审查 12：只比经济字段；相同来源变化无报告，纠错可重复构建。"""
    _, data = recent
    old = inferred(tmp_path)
    old_frame = old.read()
    src = data[NAME]
    src["fundingRate"] += [
        dict(time=START + h * HOUR, interval=8, rate=0.0001, source="archive")
        for h in (24, 32, 40, 48)
    ]
    if change == "rate":
        src["fundingRate"][4]["rate"] = 0.0002
        src["funding_prices"][4]["fundingRate"] = "0.0002"
    elif change == "interval":
        src["fundingRate"][3]["interval"] = 4
        src["fundingRate"].insert(
            3, dict(time=START + 20 * HOUR, interval=4, rate=0.0001, source="archive")
        )
        src["funding_prices"].insert(
            3,
            dict(fundingTime=START + 20 * HOUR, fundingRate="0.0001", markPrice="100"),
        )
    elif change == "price":
        src["funding_prices"][4]["markPrice"] = "105"
    elif change == "time":
        # 24h 结算取消，32h 按新 8h 不能自洽；以合法 4h 转换替换旧 24h 后一段。
        src["fundingRate"][4:5] = [
            dict(time=START + h * HOUR, interval=4, rate=0.0001, source="archive")
            for h in (28, 32)
        ]
        src["funding_prices"].insert(
            4,
            dict(fundingTime=START + 28 * HOUR, fundingRate="0.0001", markPrice="100"),
        )
    new = crypto.build(tmp_path)
    report = crypto.verify(tmp_path, new.version)
    assert report["verdict"] == "pass", report["errors"]
    assert bool(report["funding_reconciliations"]) == (change is not None)
    assert bool(list((tmp_path / "funding_reconciliations").glob("*.json"))) == (
        change is not None
    )
    assert crypto.build(tmp_path).version == new.version
    assert_frame_equal(
        crypto.open(tmp_path, old.version).read(), old_frame, check_exact=True
    )
    assert bool(crypto.verify(tmp_path, old.version)["funding_reconciliations"]) == (
        change is not None
    )


def test_bad_rest_beyond_common_archive_cap_is_irrelevant(tmp_path, recent):
    """C5 开发自检：不能让覆盖外的异常把已覆盖窗口再次缩短。"""
    _, data = recent
    data[NAME]["indexPriceKlines"] = data[NAME]["indexPriceKlines"][:35]
    data[NAME]["funding_prices"][5]["markPrice"] = ""
    assert inferred(tmp_path).meta["end"] == START + 35 * HOUR


def test_backtest_diagnoses_only_selected_inference(tmp_path, recent):
    """来源 C5：最小 C3 适配改动，诊断必须随读取窗口裁剪。"""
    from ashare.backtest.crypto import crypto_market

    ds = inferred(tmp_path)
    risk = dict(
        maint_rate=0.004,
        tier_cap=50000,
        max_leverage=125,
        clearance_fee_rate=0.0125,
        reason="synthetic",
        source_url="https://example.test/synthetic",
    )

    def market(end):
        return crypto_market(
            ds, [NAME], START, end, kind="usdm", assume=dict(rules={NAME: risk})
        )

    assert market(START + 23 * HOUR).diagnostics["inferred_funding_intervals"] == []
    notes = market(START + 35 * HOUR).diagnostics["inferred_funding_intervals"]
    assert len(notes) == 2
    assert all(r["symbol"] == NAME for r in notes)


@pytest.mark.parametrize("revised_rate", [0.0002, 0.0001 + 1e-15])
def test_archive_rate_revision_overrides_cached_rest_rate(
    tmp_path, recent, revised_rate
):
    """C5 开发自检：归档更正费率而旧 REST 缓存未修订时，纠错构建不能永久失败。"""
    _, data = recent
    old = inferred(tmp_path)
    src = data[NAME]
    src["fundingRate"] += [
        dict(
            time=START + h * HOUR,
            interval=8,
            rate=revised_rate if h == 32 else 0.0001,
            source="archive",
        )
        for h in (24, 32, 40, 48)
    ]
    new = crypto.build(tmp_path)
    report = crypto.verify(tmp_path, new.version)
    assert report["verdict"] == "pass"
    assert report["funding_reconciliations"][0]["inferred_version"] == old.version
    assert (
        new.read().filter(pl.col("time") == START + 31 * HOUR)["funding_rate"].item()
        == revised_rate
    )
    assert any(
        r["check"] == "funding_rate_archive_override" for r in report["diagnostics"]
    )
    assert crypto.build(tmp_path).version == new.version


def test_verify_terminal_formula_even_without_new_settlement(tmp_path, recent):
    """来源审查 12：没有新增结算的小时也必须受归档和请求终点约束。"""
    plan, _ = recent
    plan["end"] = START + 35 * HOUR
    ds = inferred(tmp_path)
    inputs = ds.inputs()
    inputs["recent"]["requested_end"] -= HOUR
    report = verify_frame(ds.read(), inputs)
    assert "inferred_terminal" in {r["check"] for r in report["errors"]}


def test_snapshot_provenance_independent_raw_verification(tmp_path, recent):
    """C5 自检：冻结输入的快照内容/时间/哈希必须与原始响应一致。"""
    from ashare.data.crypto.storage import RawStore, encoded
    from ashare.data.crypto.verify_raw import verify_raw

    ds = inferred(tmp_path)
    store = RawStore(tmp_path)
    payload = encoded([])
    entry = store.put("fundingInfo/test", payload, source="rest", known_at=START)
    version = store.freeze()
    store.close()
    for field, value in (
        ("records", [{"symbol": "OTHER"}]),
        ("sha256", "0" * 64),
        ("fetched_at", START + HOUR),
    ):
        inputs = ds.inputs()
        inputs["raw_version"] = version
        inputs["recent"]["funding_info"] = dict(
            key="fundingInfo/test", records=[], sha256=entry["sha256"], fetched_at=START
        )
        inputs["recent"]["funding_info"][field] = value
        errors, _ = verify_raw(tmp_path, inputs)
        assert "inferred_snapshot_provenance" in {e["check"] for e in errors}
