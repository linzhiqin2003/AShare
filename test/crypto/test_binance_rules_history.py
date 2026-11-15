"""币安历史风险规则（2026-11-15 调研补齐，证据见 docs/markets/BINANCE.md 5.2、5.3）。

来源：第一批遗留「2021-06 以前的保证金参数、清算费历史没查到」；调研后按公告、FAQ 存档、
exchangeInfo 与档位接口存档写进档案。每条期望值都直接来自公告或存档原文，不从实现取。
"""

import numpy as np
import pytest
from panelbt import markets
from panelbt.markets import binance

P = markets.get("binance_usdm")


def rules(symbols, times, **kw):
    return P.rules(symbols, np.array(times, dtype="datetime64[h]"), **kw)


def test_2021_05_19_uses_the_announced_clearance_fee_without_assumption():
    # 公告 aeeccb6d…：2021-02-07 04:00 起 BTCUSDT 2.00%、ETHUSDT 3.00%；
    # 公告 839a865d…：2021-05-20 09:00 起 0.30% / 0.50%。旧档案在这里外推 1.25%。
    # ETH 第一档上限在 2021 年没有证据，这里只看清算费：用保守假设放行其他字段。
    r, notes = rules(
        ["BTCUSDT", "ETHUSDT"],
        ["2021-05-19T00", "2021-05-20T08", "2021-05-20T09"],
        assume="conservative",
    )
    np.testing.assert_array_equal(r["clearance_fee_rate"][:, 0], [0.02, 0.02, 0.003])
    np.testing.assert_array_equal(r["clearance_fee_rate"][:, 1], [0.03, 0.03, 0.005])
    clearance = [n for n in notes if n["parameter"] == "clearance_fee_rate"]
    assert clearance and all(n["confidence"] == "official" for n in clearance)


def test_btc_first_tier_history():
    # 2019-11-14 FAQ 存档起第一档 0–50,000、MMR 0.4%；公告 e778988e…：
    # 2025-06-17 06:30 第一档上限改为 300,000（下一个整点起生效）；2025-08-19 06:30 杠杆 150x。
    times = [
        "2020-01-01T00",
        "2025-06-17T06",
        "2025-06-17T07",
        "2025-08-19T06",
        "2025-08-19T07",
        "2026-11-15T00",
    ]
    r, _ = rules(["BTCUSDT"], times, assume="conservative")
    np.testing.assert_array_equal(r["maint_rate"][:, 0], [0.004] * 6)
    np.testing.assert_array_equal(
        r["tier_cap"][:, 0], [50000, 50000, 300000, 300000, 300000, 300000]
    )
    np.testing.assert_array_equal(
        r["max_leverage"][:, 0], [125, 125, 125, 125, 150, 150]
    )


def test_eth_first_tier_history_and_conservative_gaps():
    # 2020-05/09 FAQ 表：首 10,000、MMR 0.65%；2021-01-24 FAQ：首 10,000、0.5%；
    # 两次观测之间（2020-10）没有证据，保守取较高的 0.65% 和较低的上限 10,000。
    times = ["2020-06-01T00", "2020-11-01T00", "2021-02-01T00", "2023-12-24T10"]
    with pytest.raises(ValueError, match="ETHUSDT maint_rate"):
        rules(["ETHUSDT"], times)
    r, notes = rules(["ETHUSDT"], times, assume="conservative")
    np.testing.assert_array_equal(r["maint_rate"][:, 0], [0.0065, 0.0065, 0.005, 0.004])
    # 2021-02 的上限介于 2021-01-24 的 10,000 与 2022-11-17 的 100,000 之间，保守取 10,000。
    np.testing.assert_array_equal(r["tier_cap"][:, 0], [10000, 10000, 10000, 50000])
    gap = [
        n
        for n in notes
        if n["parameter"] == "maint_rate" and n["start"].startswith("2020-11-01")
    ]
    assert gap and gap[0]["confidence"] == "assumption"
    assert "conservative" in gap[0]["reason"]


def test_conservative_picks_the_adverse_neighbour_for_each_field():
    # exchangeInfo 存档：2021-09-14 BTC 0.6%、2021-10-14 BTC 2.0%；之间取较高者。
    r, _ = rules(["BTCUSDT"], ["2021-10-01T00"], assume="conservative")
    assert r["clearance_fee_rate"][0, 0] == 0.02
    # 同一缺口按「沿用上一个观测」得到 0.6%：两种策略确实不同。
    r, _ = rules(["BTCUSDT"], ["2021-10-01T00"], assume="previous")
    assert r["clearance_fee_rate"][0, 0] == 0.006
    assert binance.ADVERSE == dict(
        maint_rate=max, clearance_fee_rate=max, tier_cap=min, max_leverage=min
    )


def test_clearance_gaps_still_require_explicit_assumption():
    # 2021-06-10 公告之后到 2021-07-16 存档之间没有证据：默认报错。
    with pytest.raises(ValueError, match="BTCUSDT clearance_fee_rate"):
        rules(["BTCUSDT"], ["2021-07-01T00"])
    with pytest.raises(ValueError, match="conservative"):
        rules(["BTCUSDT"], ["2021-07-01T00"], assume="later")


def test_non_hourly_effective_times_apply_from_the_next_hour():
    _, notes = rules(["BTCUSDT"], ["2025-06-17T07"], assume="conservative")
    cap = [n for n in notes if n["parameter"] == "tier_cap"][0]
    assert cap["effective_at"].startswith("2025-06-17T06:30")
    assert cap["applied_at"].startswith("2025-06-17T07:00")
    assert cap["approximation"] is True


def test_evidence_records_are_sorted_and_disjoint():
    for symbol in ("BTCUSDT", "ETHUSDT", "BTCUSDT_261225"):
        for field, records in binance._observations(symbol).items():
            for a, b in zip(records, records[1:], strict=False):
                assert a[0] < a[1] <= b[0], (symbol, field)
            for _lo, _hi, value, source, confidence, _timing in records:
                assert source.startswith("https://") and confidence
                assert np.isfinite(value) and value > 0


def test_conservative_fill_is_linear_in_rows():
    # 回归（自查）：逐小时生成说明会让 7 万行的缺口变成 O(T²)。
    times = np.arange(
        np.datetime64("2020-01-01T00"),
        np.datetime64("2026-11-15T00"),
        np.timedelta64(1, "h"),
    )
    _, notes = P.rules(["BTCUSDT", "ETHUSDT"], times, assume="conservative")
    assert len(notes) < 400
