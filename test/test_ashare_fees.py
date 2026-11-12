"""A 股历史费率表：覆盖面板起点、每次调整落在正确日期、旧写法结果不变。"""

import numpy as np
import pytest
from panelbt.markets import ASHARE, FeeSchedule, MarketProfile

ROWS = [
    # date, tax_buy, tax_sell, fee_rate, fee_per_share
    ("2007-01-04", 0.001, 0.001, 0.0000255, 0.001),
    ("2007-05-29", 0.001, 0.001, 0.0000255, 0.001),
    ("2007-05-30", 0.003, 0.003, 0.0000255, 0.001),
    ("2008-04-23", 0.003, 0.003, 0.0000255, 0.001),
    ("2008-04-24", 0.001, 0.001, 0.0000255, 0.001),
    ("2008-09-18", 0.001, 0.001, 0.0000255, 0.001),
    ("2008-09-19", 0.0, 0.001, 0.0000255, 0.001),
    ("2012-08-31", 0.0, 0.001, 0.0000255, 0.001),
    ("2012-09-03", 0.0, 0.001, 0.0000255, 0.0003),
    ("2015-07-31", 0.0, 0.001, 0.0000255, 0.0003),
    ("2015-08-03", 0.0, 0.001, 0.00002, 0.0),
    ("2021-11-12", 0.0, 0.001, 0.00002, 0.0),
    ("2021-11-15", 0.0, 0.001, 0.000025, 0.0),
    ("2022-04-28", 0.0, 0.001, 0.000025, 0.0),
    ("2022-04-29", 0.0, 0.001, 0.00001, 0.0),
    ("2023-08-25", 0.0, 0.001, 0.00001, 0.0),
    ("2023-08-28", 0.0, 0.0005, 0.00001, 0.0),
    ("2026-11-12", 0.0, 0.0005, 0.00001, 0.0),
]


def test_full_history_needs_no_overrides():
    dates = [r[0] for r in ROWS]
    c = ASHARE.costs(dates)
    expect = np.array([r[1:] for r in ROWS])
    np.testing.assert_array_equal(c.tax_buy, expect[:, 0])
    np.testing.assert_array_equal(c.tax_sell, expect[:, 1])
    for side in ("buy", "sell"):
        np.testing.assert_array_equal(getattr(c, f"fee_rate_{side}"), expect[:, 2])
        np.testing.assert_array_equal(getattr(c, f"fee_per_share_{side}"), expect[:, 3])


def test_before_schedule_still_fails():
    with pytest.raises(ValueError, match="precede"):
        ASHARE.costs(["2005-01-21"])


def test_overriding_transfer_fee_replaces_whole_charge():
    """旧写法：2022-04-29 之前显式给过户费率；结果与历史表无关、与改动前一致。"""
    dates = ["2008-09-19", "2015-07-31", "2015-08-03", "2022-04-29"]
    c = ASHARE.costs(dates, fee_rate_buy=1e-5, fee_rate_sell=1e-5)
    np.testing.assert_array_equal(c.fee_rate_buy, 1e-5)
    np.testing.assert_array_equal(c.fee_per_share_buy, [0, 0, 0, 0])
    np.testing.assert_array_equal(c.fee_per_share_sell, [0, 0, 0, 0])
    np.testing.assert_array_equal(c.tax_sell, 0.001)
    np.testing.assert_array_equal(c.tax_buy, 0.0)
    # 只覆盖买方：卖方仍按历史表。
    one = ASHARE.costs(["2012-01-04"], fee_rate_buy=0.0)
    assert one.fee_per_share_buy[0] == 0 and one.fee_per_share_sell[0] == 0.001


def test_old_calls_unchanged_after_2022():
    """改动前可用的调用（2022-04-29 之后、或显式给过户费），逐位不变。"""
    dates = np.arange("2022-04-29", "2026-11-13", dtype="datetime64[D]")
    c = ASHARE.costs(dates)
    old = np.where(dates >= np.datetime64("2023-08-28"), 0.0005, 0.001)
    np.testing.assert_array_equal(c.tax_sell, old)
    for name in ("tax_buy", "fee_per_share_buy", "fee_per_share_sell"):
        np.testing.assert_array_equal(getattr(c, name), 0.0)
    np.testing.assert_array_equal(c.fee_rate_buy, 1e-5)


def test_fixed_rates_explicit():
    """全区间固定费率：四个分量都覆盖，历史表不起作用。"""
    dates = ["2007-01-04", "2026-11-12"]
    c = ASHARE.costs(
        dates, tax_buy=0.0, tax_sell=0.001, fee_rate_buy=1e-5, fee_rate_sell=1e-5
    )
    np.testing.assert_array_equal(c.tax_buy, 0.0)
    np.testing.assert_array_equal(c.tax_sell, 0.001)
    np.testing.assert_array_equal(c.fee_per_share_sell, 0.0)


def test_components_must_be_disjoint_fee_keys():
    with pytest.raises(ValueError, match="components"):
        FeeSchedule(
            (("2025-01-01", {"tax_buy": 0.1}),), components=(("tax_buy",), ("tax_buy",))
        )
    with pytest.raises(ValueError, match="components"):
        FeeSchedule(
            (("2025-01-01", {"tax_buy": 0.1}),), components=(("commission_rate",),)
        )
    plain = MarketProfile(
        "x", "X", 252, FeeSchedule((("2025-01-01", {"tax_buy": 0.1}),))
    )
    assert plain.costs(["2025-01-02"], tax_sell=0.2).tax_buy[0] == 0.1
