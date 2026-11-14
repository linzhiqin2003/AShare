import copy
import math

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from ashare.data.crypto.dataset import aggregate_daily, rank_coins, write_panel
from ashare.data.crypto.events import asof, normalize_event
from ashare.data.crypto.funding import funding_schedule
from ashare.data.crypto.instruments import (
    coin_price,
    coin_quantity,
    eligible,
    rule_at,
    validate_profile,
)
from ashare.data.crypto.panel import build_frame
from ashare.data.crypto.timeutil import HOUR, millis
from ashare.data.crypto.verify import verify_frame

PERP = "usdm:PERPUSDT"
QUARTER = "usdm:FUTUSDT_240628"


def row(frame, name, hour, start):
    return frame.filter(
        (pl.col("instrument") == name) & (pl.col("time") == start + hour * HOUR)
    ).row(0, named=True)


def test_shared_fixture_hand_calculation_and_lifecycle(fixture):
    """来源：4.17 输入合同、R1-9/12、R3 共用夹具要求。"""
    frame, notes = build_frame(**fixture)
    assert frame.height == 24 and notes == {}
    t = fixture["start"]
    r = row(frame, PERP, 0, t)
    assert r["settlement"] == 103.5 and math.isnan(r["mark_open"]) and r["funding"] == 0
    assert not r["tradable"] and r["lifecycle"] == "pre_listing"
    r = row(frame, PERP, 3, t)
    assert r["funding"] == 0.0001 * 105 * 0.01
    assert math.isnan(r["last_funding_rate"])
    assert row(frame, PERP, 4, t)["last_funding_rate"] == 0.0001
    assert r["volume_engine"] == 1000
    assert row(frame, PERP, 7, t)["funding"] == -0.0002 * 109 * 0.01
    r = row(frame, QUARTER, 7, t)
    assert r["expires"] and r["settlement"] == 199 and r["mark_close"] == 208.5
    assert (r["settlement"] - 200) * 0.1 == -0.1
    r = row(frame, QUARTER, 8, t)
    assert not r["tradable"] and not r["expires"] and r["settlement"] == 199
    assert math.isnan(r["mark_open"]) and r["funding"] == 0


@pytest.mark.parametrize("index", [0, 1, 2])
def test_funding_missing_first_middle_last_is_error(fixture, index):
    """来源：6.10 缺一笔不当零，包括区间首尾。"""
    fixture["data"][PERP]["fundingRate"].pop(index)
    with pytest.raises(ValueError, match="资金费"):
        build_frame(**fixture)


def test_funding_interval_change_comes_from_field():
    start = millis("2024-05-01")
    records = [
        dict(time=start + hour * HOUR, interval=interval, rate=0.0001)
        for hour, interval in [(0, 8), (8, 8), (12, 4), (16, 4), (17, 1), (18, 1)]
    ]
    assert set(funding_schedule(records, start, start + 18 * HOUR, start)) == {
        start + h * HOUR for h in (8, 12, 16, 17, 18)
    }
    wrong = copy.deepcopy(records)
    wrong[2]["interval"] = 8
    with pytest.raises(ValueError, match="整点"):
        funding_schedule(wrong, start, start + 18 * HOUR, start)
    # 不能把漏了 12:00 当作从 4h 自动改成 8h。
    wrong = [
        dict(time=start + hour * HOUR, interval=4, rate=0) for hour in (0, 4, 8, 16)
    ]
    with pytest.raises(ValueError, match="缺少"):
        funding_schedule(wrong, start, start + 16 * HOUR, start)


def test_final_funding_price_truncates_whole_rectangle(fixture):
    fixture["data"][PERP]["funding_prices"][-1]["markPrice"] = ""
    frame, notes = build_frame(**fixture)
    assert frame.height == 22
    assert frame["time"].max() == fixture["start"] + 10 * HOUR
    assert notes[PERP]["reason"] == "final_funding_price_missing"


def test_real_delivery_api_uses_midnight_date_label(fixture):
    """开发实测：BTC 240628 官方 deliveryTime=00:00，实际交割仍为 08:00。"""
    fixture["data"][QUARTER]["deliveries"][0]["deliveryTime"] -= 8 * HOUR
    frame, _ = build_frame(**fixture)
    r = row(frame, QUARTER, 7, fixture["start"])
    assert r["settlement"] == 199 and r["expires"]
    assert r["delivery_record_time"] == fixture["start"]
    assert verify_frame(frame, fixture)["verdict"] == "pass"


def test_missing_internal_funding_price_is_not_future_filled(fixture):
    fixture["data"][PERP]["funding_prices"][1]["markPrice"] = ""
    with pytest.raises(LookupError, match="缺价格"):
        build_frame(**fixture)


def test_official_price_difference_only_diagnostic(fixture):
    """来源：R2 N2/R3 V4；官方价格不能为了相等校验而覆盖。"""
    fixture["data"][PERP]["funding_prices"][0]["markPrice"] = "120"
    frame, _ = build_frame(**fixture)
    r = row(frame, PERP, 3, fixture["start"])
    assert r["funding_price"] == 120
    result = verify_frame(frame, fixture)
    assert result["verdict"] == "pass", result["errors"]
    assert any(
        d["check"] == "funding_mark_deviation" and d["review"]
        for d in result["diagnostics"]
    )


def test_old_funding_uses_exact_right_open_and_marks_approximation(fixture):
    delta = millis("2021-06-25") - fixture["start"]
    fixture["start"] += delta
    fixture["end"] += delta
    for p in fixture["profiles"]:
        p["listed_at"] += delta
        for r in p["rules"]:
            r["known_at"] += delta
            r["effective_at"] += delta
    for e in fixture["events"]:
        e.pop("version_id")
        e["known_at"] += delta
        e["effective_at"] += delta
    for datasets in fixture["data"].values():
        for name, records in datasets.items():
            for r in records:
                for k in ("time", "raw_time", "fundingTime", "deliveryTime"):
                    if k in r:
                        r[k] += delta
                if name == "funding_prices":
                    r["markPrice"] = ""
    frame, _ = build_frame(**fixture)
    r = row(frame, PERP, 3, fixture["start"])
    assert r["funding_price"] == 104.5 and r["quality"] & 16
    del fixture["data"][PERP]["markPriceKlines"][2]  # 04:00 缺价，不得拿 05:00 顶替
    with pytest.raises(LookupError):
        build_frame(**fixture)


def test_gap_suspend_resume_and_zombie_do_not_delist(fixture):
    t = fixture["start"]
    fixture["events"] += [
        dict(
            event_id="pause",
            instrument=PERP,
            kind="suspend",
            known_at=t,
            effective_at=t + 4 * HOUR,
            source_url="fixture://maintenance",
            original_text="maintenance",
        ),
        dict(
            event_id="resume",
            instrument=PERP,
            kind="resume",
            known_at=t,
            effective_at=t + 6 * HOUR,
            source_url="fixture://maintenance",
            original_text="resumed",
        ),
    ]
    fixture["data"][PERP]["klines"] = [
        r
        for r in fixture["data"][PERP]["klines"]
        if r["time"] not in (t + 4 * HOUR, t + 5 * HOUR)
    ]
    frame, _ = build_frame(**fixture)
    assert row(frame, PERP, 4, t)["quality"] & 8
    assert row(frame, PERP, 6, t)["tradable"]
    assert verify_frame(frame, fixture)["verdict"] == "pass"
    # 尾部连续零成交仅从第 3 个已观测小时标记，不能改前两小时。
    for bar in fixture["data"][PERP]["klines"]:
        if bar["time"] >= t + 8 * HOUR:
            bar.update(
                open=109,
                high=109,
                low=109,
                close=109,
                volume=0,
                quote_volume=0,
                taker_buy_quote_volume=0,
            )
    frame, _ = build_frame(**fixture)
    assert row(frame, PERP, 10, t)["quality"] & 4
    assert not row(frame, PERP, 8, t)["quality"] & 4
    assert row(frame, PERP, 11, t)["lifecycle"] == "active"


def test_unexplained_gap_errors_even_when_model_allows_carry(fixture):
    fixture["data"][PERP]["klines"].pop(3)
    with pytest.raises(ValueError, match="缺少行情"):
        build_frame(**fixture)
    frame, _ = build_frame(**fixture, allow_missing=True)
    report = verify_frame(frame, fixture)
    assert report["verdict"] == "fail"
    assert any(e["check"] == "unexplained_gap" for e in report["errors"])


def test_1000pepe_units_rules_and_field_filters(fixture):
    p = fixture["profiles"][0]
    p.update(
        coin="PEPE",
        base_asset="1000PEPE",
        coin_units=1000,
        mapping_source="fixture://1000pepe-spec",
    )
    frame, _ = build_frame(**fixture)
    r = row(frame, PERP, 3, fixture["start"])
    assert coin_price(r["close"], p) == r["close"] / 1000
    assert coin_quantity(r["volume"], p) == r["volume"] * 1000
    assert r["quote_volume"] == 1030 and r["min_notional"] == 5
    assert r["funding"] == 0.0001 * 105 * 0.01
    good = dict(
        contractType="PERPETUAL",
        underlyingType="COIN",
        quoteAsset="USDT",
        marginAsset="USDT",
    )
    assert eligible(good, "usdm")
    for change in [
        dict(contractType="TRADIFI_PERPETUAL"),
        dict(underlyingType="INDEX"),
        dict(quoteAsset="USDC"),
        dict(marginAsset="USDC"),
    ]:
        assert not eligible(dict(good, **change), "usdm")


def test_frozen_multiplier_and_nonhour_rule_effective(fixture):
    p = fixture["profiles"][0]
    r = dict(
        p["rules"][0],
        effective_at=fixture["start"] + HOUR // 2,
        known_at=fixture["start"],
        qty_step=0.02,
        source="announcement",
        old_positions_exempt=True,
    )
    p["rules"].append(r)
    assert rule_at(p, fixture["start"])["qty_step"] == 0.01
    assert rule_at(p, fixture["start"] + HOUR)["qty_step"] == 0.02
    validate_profile(p)
    p["rules"].append(dict(r, qty_step=0.001))
    with pytest.raises(ValueError, match="冻结"):
        validate_profile(p)


def test_spot_delist_preserves_unrealized_asset(fixture):
    """来源：R1-10，现货停止交易不能虚构现金兑付。"""
    p = fixture["profiles"][0]
    p.update(
        instrument="spot:PERPUSDT",
        market="spot",
        contract_type="SPOT",
        margin_asset=None,
    )
    event = dict(
        event_id="spot-stop",
        instrument=p["instrument"],
        kind="delist",
        known_at=fixture["start"],
        effective_at=fixture["start"] + 6 * HOUR,
        source_url="fixture://spot-delist",
        original_text="Synthetic spot delisting",
    )
    frame, _ = build_frame(
        [p],
        {p["instrument"]: {"klines": fixture["data"][PERP]["klines"]}},
        [event],
        fixture["start"],
        fixture["end"],
    )
    r = row(frame, p["instrument"], 6, fixture["start"])
    assert r["close"] == 106 and r["settlement_source"] == "unrealized_spot_carry"
    assert not r["expires"] and not r["tradable"] and r["funding"] == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("quote_asset", "USDC"),
        ("margin_asset", "USDC"),
        ("contract_type", "TRADIFI_PERPETUAL"),
    ],
)
def test_invalid_profile_rejected_by_builder(fixture, field, value):
    fixture["profiles"][0][field] = value
    with pytest.raises(ValueError, match="USDT"):
        build_frame(**fixture)


def test_append_future_data_versions_history_and_ranking_unchanged(fixture):
    """来源：6.10 截断测试同时截事件版本；R1-10/R3 V3。"""
    cut = fixture["start"] + 6 * HOUR
    short = copy.deepcopy(fixture)
    short["end"] = cut
    for datasets in short["data"].values():
        for key, records in datasets.items():
            datasets[key] = [
                r
                for r in records
                if r.get("time", r.get("fundingTime", r.get("deliveryTime", 0)))
                <= cut + 1000
            ]
    short["events"] = [e for e in short["events"] if e["known_at"] <= cut]
    prefix, _ = build_frame(**short)
    # 未来一版延期公告：不能改修订公布之前的历史视图。
    old = fixture["events"][0]
    replacement = {k: v for k, v in old.items() if k != "version_id"}
    replacement.update(
        known_at=cut + HOUR,
        effective_at=fixture["end"] + HOUR,
        supersedes=old["version_id"],
    )
    fixture["events"].append(normalize_event(replacement))
    full, _ = build_frame(**fixture)
    assert_frame_equal(prefix, full.filter(pl.col("time") < cut))
    assert asof(fixture["events"], cut) == asof(short["events"], cut)
    assert_frame_equal(
        rank_coins(prefix, fixture["profiles"], cut, n=1, min_age="0D", lookback="1D"),
        rank_coins(full, fixture["profiles"], cut, n=1, min_age="0D", lookback="1D"),
    )
    # 不接 panelbt；固定持仓的资金费现金路径逐位不变。
    np.testing.assert_array_equal(
        prefix.filter(pl.col("instrument") == PERP)["funding"].cum_sum(),
        full.filter((pl.col("instrument") == PERP) & (pl.col("time") < cut))[
            "funding"
        ].cum_sum(),
    )


def test_future_first_event_does_not_change_prefix_schema(fixture):
    """真实 E2E 截断发现：首次交割前全 null 列曾被推断为 Null 而非固定类型。"""
    full, _ = build_frame(**fixture)
    fixture["end"] = fixture["start"] + 6 * HOUR
    prefix, _ = build_frame(**fixture)
    assert_frame_equal(prefix, full.filter(pl.col("time") < fixture["end"]))


def test_daily_trade_hour_and_arrays_version(tmp_path, fixture):
    frame, notes = build_frame(**fixture)
    ds = write_panel(
        tmp_path,
        frame,
        profiles=fixture["profiles"],
        events=fixture["events"],
        data=fixture["data"],
        notes=notes,
    )
    arrays = ds.arrays(["settlement", "funding"], instruments=[PERP, QUARTER])
    assert arrays["settlement"].shape == (12, 2)
    assert arrays["settlement"][7, 1] == 199
    # 日线的 01:00..次日01:00，标签与 available_at 明确，不移位。
    one = frame.filter(pl.col("instrument") == PERP).filter(
        pl.col("time") == fixture["start"] + 3 * HOUR
    )
    expanded = pl.concat(
        [
            one.with_columns(pl.lit(fixture["start"] + i * HOUR).alias("time"))
            for i in range(49)
        ]
    )
    daily = aggregate_daily(expanded, trade_hour=1)
    assert daily["day"].to_list() == [
        fixture["start"] + HOUR,
        fixture["start"] + 25 * HOUR,
    ]
    assert daily["available_at"].to_list() == [
        fixture["start"] + 25 * HOUR,
        fixture["start"] + 49 * HOUR,
    ]
    assert daily["volume"].to_list() == [240, 240]
