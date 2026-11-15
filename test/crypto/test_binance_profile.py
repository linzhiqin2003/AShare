"""来源：BINANCE.md 唯一费用表、4.17 缺规则默认拒绝。"""

import numpy as np
import pytest
from panelbt import markets


def test_binance_identity_and_fee_segments():
    spot = markets.get("binance_spot")
    future = markets.get("binance_usdm")
    for p in (spot, future):
        assert p.currency == "USDT" and p.periods_per_year == 8760
        assert not p.sell_lock and not p.price_limits
    dates = ["2019-12-20T08", "2023-05-31T23", "2023-06-01T00"]
    np.testing.assert_array_equal(
        future.fee_terms(dates)["taker"], [0.0004, 0.0004, 0.0005]
    )
    np.testing.assert_array_equal(
        future.fee_terms(dates)["delivery"], [0.0004, 0.0004, 0.0005]
    )
    np.testing.assert_array_equal(spot.fee_terms(dates)["taker"], [0.001] * 3)
    for before in ("2019-12-19", "2019-12-20T07"):
        with pytest.raises(ValueError, match="fee"):
            future.fee_terms([before])
    provenance = future.fee_terms(dates)["provenance"]
    assert all(x["source_url"] and x["confidence"] for x in provenance)
    assert provenance[-1]["confidence"] == "assumption"
    assert spot.costs(dates).commission_rate == 0
    assert spot.costs(dates).slippage == 0


def test_rule_gaps_require_explicit_assumption():
    p = markets.get("binance_usdm")
    times = ["2021-05-19T00", "2024-01-01T00"]
    with pytest.raises(ValueError, match="BTCUSDT.*clearance_fee_rate"):
        p.rules(["BTCUSDT"], times)
    rules, notes = p.rules(["BTCUSDT"], times, assume="previous")
    assert rules["maint_rate"][0, 0] == 0.004
    assert rules["tier_cap"][0, 0] == 50000
    # 2026-11-15 调研补齐历史后：2021-05-19 由公告直接给出 2.00%（此前外推 1.25%）。
    assert rules["clearance_fee_rate"][0, 0] == 0.02
    assert any(
        n["symbol"] == "BTCUSDT" and n["start"] == times[0] + ":00:00.000"
        for n in notes
    )
    with pytest.raises(ValueError, match="UNKNOWN"):
        p.rules(["UNKNOWN"], times, assume="previous")


def test_risk_rules_are_symbol_and_date_specific_and_bounded():
    p = markets.get("binance_usdm")
    dates = ["2023-12-24T09", "2023-12-24T10", "2025-08-19T06", "2025-08-19T07"]
    rules, _ = p.rules(["BTCUSDT", "ETHUSDT"], dates, assume="previous")
    np.testing.assert_array_equal(
        rules["maint_rate"][:, 1], [0.005, 0.004, 0.004, 0.004]
    )
    np.testing.assert_array_equal(
        rules["tier_cap"][:, 1], [200000, 50000, 300000, 300000]
    )
    np.testing.assert_array_equal(rules["max_leverage"][:, 1], [100, 125, 125, 150])
    with pytest.raises(ValueError, match="leverage"):
        p.rules(["ETHUSDT"], dates, assume="previous", leverage=126)
    with pytest.raises(ValueError, match="2027"):
        p.rules(["BTCUSDT"], ["2027-01-01"])


def test_fee_sensitivity_is_explicit_and_diagnosed():
    p = markets.get("binance_usdm")
    fees = p.fee_terms(["2023-06-01", "2024-01-01"], fee_switch="2024-01-01")
    np.testing.assert_array_equal(fees["taker"], [0.0004, 0.0005])
    assert all(x["confidence"] == "assumption" for x in fees["provenance"])
