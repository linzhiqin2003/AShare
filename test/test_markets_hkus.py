"""港股（本地账户、港股通）与美股市场档案：手算例子、分段边界、近似与工具函数。

期望值一律按调研文档（docs/markets/hk_rules.md、us_rules.md）在测试里直接手写，
不调用档案里的费率表，档案改错一处这里就对不上。
"""

import warnings

import numpy as np
import panelbt as bt
import pytest
from panelbt import markets
from panelbt.markets import (
    ASHARE,
    HK_DIRECT,
    HK_SOUTHBOUND,
    US,
    US_WITHHOLDING,
    connect_home_equity,
    hk_margin,
    home_currency_nav,
    net_dividends,
    reg_t_margin,
    settlement_days,
)

REL = 1e-12


def close(a, b):
    return a == pytest.approx(b, rel=REL, abs=1e-12)


def run_one(profile, dates, price, div, shares, costs, config, **market):
    t = len(dates)
    p = np.asarray(price, float).reshape(t, 1)
    m = profile.market(p, p, cash_div=np.asarray(div, float).reshape(t, 1), **market)
    targets = bt.Targets.from_dense(
        np.asarray(shares, float).reshape(t, 1), kind="shares"
    )
    return bt.run(m, targets, costs, config)


# ---------------------------------------------------------------------------
# 手算例子：一买、一次分红、一卖，买卖分别落在费率分段两侧。
# ---------------------------------------------------------------------------


def test_hk_direct_hand_example():
    """买在 2022-12-30（交易费 0.005%、使用费 HK$0.5、印花税 0.13%），
    卖在 2023-01-03（交易费 0.00565%、无使用费）。H 股分红扣 10%。"""
    dates = ["2022-12-30", "2023-01-02", "2023-01-03"]
    costs = HK_DIRECT.costs(dates, slippage=0)
    config = HK_DIRECT.config(
        lot_sizes=[500], h_share=[True], initial_cash=1_000_000, record_trades=True
    )
    r = run_one(
        HK_DIRECT, dates, [50, 52, 55], [0, 2, 0], [1000, np.nan, 0], costs, config
    )
    buy, sell = r.trades
    fv = 20_000.0
    a = 50_000.0
    # 印花税 0.13% + 取整余量 HK$1 按 20,000 折成比例。
    assert close(buy.tax, 0.0013 * a + a / fv)
    # 证监会 0.0027% + 会财局 0.00015% + 交易费 0.005% + 交收费 0.002%，
    # 外加使用费 0.5 与交收费下限超出部分 (2 - 0.002%×20,000) 按 20,000 折算。
    rate = 0.000027 + 0.0000015 + 0.00005 + 0.00002
    assert close(buy.other_fee, rate * a + (0.5 + (2 - 0.00002 * fv)) * a / fv)
    assert close(buy.commission, 15.0)  # 假设：每笔 HK$15 平台费
    a2 = 55_000.0
    assert close(sell.tax, 0.0013 * a2 + a2 / fv)
    rate2 = 0.000027 + 0.0000015 + 0.0000565 + 0.00002
    assert close(sell.other_fee, rate2 * a2 + (2 - 0.00002 * fv) * a2 / fv)
    assert close(sell.commission, 15.0)
    assert close(r.dividend_gross[1], 2000.0) and close(r.dividend_tax[1], 200.0)
    cash = (
        1_000_000
        - a
        - buy.tax
        - buy.other_fee
        - 15
        + 1800
        + a2
        - sell.tax
        - sell.other_fee
        - 15
    )
    assert close(r.cash[-1], cash)
    # 真实逐笔规则下（非近似）的数：印花税 65 向上取整仍是 65，其他费用 4.925+0.5+2。
    assert np.ceil(0.0013 * a) == 65 and close(rate * a + 0.5 + 2, 7.425)


def test_hk_southbound_hand_example():
    """买在 2023-11-16（印花税 0.13%），卖在 2023-11-20（0.1%）；汇兑 0.05%/边，
    佣金万三无下限，分红一律扣 20%，交收费 2025-06-30 前有 HK$2 下限。"""
    dates = ["2023-11-16", "2023-11-17", "2023-11-20"]
    costs = HK_SOUTHBOUND.costs(dates, slippage=0)
    config = HK_SOUTHBOUND.config(
        lot_sizes=[200], initial_cash=500_000, record_trades=True
    )
    r = run_one(
        HK_SOUTHBOUND, dates, [100, 101, 98], [0, 3, 0], [400, np.nan, 0], costs, config
    )
    buy, sell = r.trades
    fv, a, a2 = 20_000.0, 40_000.0, 39_200.0
    rate = 0.000027 + 0.0000015 + 0.0000565 + 0.00002 + 0.0005
    floor = 2 - 0.00002 * fv
    assert close(buy.tax, 0.0013 * a + a / fv)
    assert close(buy.other_fee, rate * a + floor * a / fv)
    assert close(buy.commission, 0.0003 * a)
    assert close(sell.tax, 0.001 * a2 + a2 / fv)
    assert close(sell.other_fee, rate * a2 + floor * a2 / fv)
    assert close(sell.commission, 0.0003 * a2)
    assert close(r.dividend_gross[1], 1200.0) and close(r.dividend_tax[1], 240.0)


def test_us_hand_example():
    """2026-09-29 买 300 股；09-30 分红并卖 100 股（TAF 0.000195/股）；
    10-01 卖 200 股（TAF 暂停为 0）。SEC 费 $20.60/百万，按笔进位到分的余量各 $0.01。"""
    dates = ["2026-09-29", "2026-09-30", "2026-10-01"]
    costs = US.costs(dates, slippage=0)
    config = US.config(n=1, initial_cash=100_000, record_trades=True)
    r = run_one(US, dates, [40, 41, 42], [0, 0.5, 0], [300, 200, 0], costs, config)
    buy, sell1, sell2 = r.trades
    fv = 10_000.0
    assert close(buy.commission, 1.5) and buy.other_fee == 0 and buy.tax == 0
    assert close(sell1.commission, 1.0)  # 0.005×100 = 0.5，被每单 $1 下限抬上来
    a1 = 4100.0
    assert close(sell1.other_fee, 20.6e-6 * a1 + 0.000195 * 100 + 0.02 * a1 / fv)
    a2 = 8400.0
    assert close(sell2.commission, 1.0)
    assert close(sell2.other_fee, 20.6e-6 * a2 + 0.01 * a2 / fv)
    assert close(r.dividend_gross[1], 150.0) and close(r.dividend_tax[1], 15.0)
    # 佣金参数照 IBKR Fixed：按股 0.005、每单下限 1、上限成交额 1%（上限由内核执行）。
    c = US.costs(["2026-09-29"], slippage=0)
    assert c.commission_max_rate == 0.01 and c.commission_per_share == 0.005


# ---------------------------------------------------------------------------
# 分段边界：生效日当天取新值，前一天取旧值。期望值逐条手写。
# ---------------------------------------------------------------------------

# (日期, 印花税, 证监会, 会财局, 交易费, 交收费率, 交收下限, 使用费)
HK_ROWS = [
    ("2010-01-01", 0.001, 0.00004, 0, 0.00005, 0.00002, 2, 0.5),
    ("2010-09-30", 0.001, 0.00004, 0, 0.00005, 0.00002, 2, 0.5),
    ("2010-10-01", 0.001, 0.00003, 0, 0.00005, 0.00002, 2, 0.5),
    ("2014-10-31", 0.001, 0.00003, 0, 0.00005, 0.00002, 2, 0.5),
    ("2014-11-01", 0.001, 0.000027, 0, 0.00005, 0.00002, 2, 0.5),
    ("2021-07-31", 0.001, 0.000027, 0, 0.00005, 0.00002, 2, 0.5),
    ("2021-08-01", 0.0013, 0.000027, 0, 0.00005, 0.00002, 2, 0.5),
    ("2021-12-31", 0.0013, 0.000027, 0, 0.00005, 0.00002, 2, 0.5),
    ("2022-01-01", 0.0013, 0.000027, 0.0000015, 0.00005, 0.00002, 2, 0.5),
    ("2022-12-31", 0.0013, 0.000027, 0.0000015, 0.00005, 0.00002, 2, 0.5),
    ("2023-01-01", 0.0013, 0.000027, 0.0000015, 0.0000565, 0.00002, 2, 0),
    ("2023-11-16", 0.0013, 0.000027, 0.0000015, 0.0000565, 0.00002, 2, 0),
    ("2023-11-17", 0.001, 0.000027, 0.0000015, 0.0000565, 0.00002, 2, 0),
    ("2025-06-27", 0.001, 0.000027, 0.0000015, 0.0000565, 0.00002, 2, 0),
    ("2025-06-30", 0.001, 0.000027, 0.0000015, 0.0000565, 0.000042, 0, 0),
    ("2026-11-13", 0.001, 0.000027, 0.0000015, 0.0000565, 0.000042, 0, 0),
]


def expected_hk(rows, fv, fx=0.0):
    tax, fee = [], []
    for _, stamp, sfc, afrc, trading, settle, smin, tariff in rows:
        tax.append(stamp + 1.0 / fv)
        fee.append(
            sfc
            + afrc
            + trading
            + settle
            + fx
            + (tariff + max(smin - settle * fv, 0)) / fv
        )
    return np.array(tax), np.array(fee)


@pytest.mark.parametrize("fv", [20_000.0, 1_000.0, 1e6])
def test_hk_direct_boundaries(fv):
    dates = [r[0] for r in HK_ROWS]
    c = HK_DIRECT.costs(dates, fill_value=fv)
    tax, fee = expected_hk(HK_ROWS, fv)
    for side in ("buy", "sell"):
        np.testing.assert_allclose(getattr(c, f"tax_{side}"), tax, rtol=REL)
        np.testing.assert_allclose(getattr(c, f"fee_rate_{side}"), fee, rtol=REL)
        assert not getattr(c, f"fee_per_share_{side}").any()


def test_hk_statutory_only_and_itemised():
    dates = [r[0] for r in HK_ROWS]
    c = HK_DIRECT.costs(dates, fill_value=None)
    np.testing.assert_array_equal(c.tax_buy, [r[1] for r in HK_ROWS])
    np.testing.assert_allclose(
        c.fee_rate_sell, [r[2] + r[3] + r[4] + r[5] for r in HK_ROWS], rtol=REL
    )
    it = HK_DIRECT.charges(dates)
    np.testing.assert_array_equal(it["tariff"], [r[7] for r in HK_ROWS])
    np.testing.assert_array_equal(it["settlement_min"], [r[6] for r in HK_ROWS])
    assert it["settlement_max"][0] == 100 and np.isinf(it["settlement_max"][-1])


# 港股通 2014-11-17 开通；2023-01-03 才停收使用费（香港本地 2023-01-01）。
SB_ROWS = [
    ("2014-11-17", 0.001, 0.000027, 0, 0.00005, 0.00002, 2, 0.5),
    ("2021-07-31", 0.001, 0.000027, 0, 0.00005, 0.00002, 2, 0.5),
    ("2021-08-01", 0.0013, 0.000027, 0, 0.00005, 0.00002, 2, 0.5),
    ("2022-01-01", 0.0013, 0.000027, 0.0000015, 0.00005, 0.00002, 2, 0.5),
    ("2022-12-31", 0.0013, 0.000027, 0.0000015, 0.00005, 0.00002, 2, 0.5),
    ("2023-01-01", 0.0013, 0.000027, 0.0000015, 0.0000565, 0.00002, 2, 0.5),
    ("2023-01-02", 0.0013, 0.000027, 0.0000015, 0.0000565, 0.00002, 2, 0.5),
    ("2023-01-03", 0.0013, 0.000027, 0.0000015, 0.0000565, 0.00002, 2, 0),
    ("2023-11-16", 0.0013, 0.000027, 0.0000015, 0.0000565, 0.00002, 2, 0),
    ("2023-11-17", 0.001, 0.000027, 0.0000015, 0.0000565, 0.00002, 2, 0),
    ("2025-06-29", 0.001, 0.000027, 0.0000015, 0.0000565, 0.00002, 2, 0),
    ("2025-06-30", 0.001, 0.000027, 0.0000015, 0.0000565, 0.000042, 0, 0),
]


def test_southbound_boundaries():
    dates = [r[0] for r in SB_ROWS]
    c = HK_SOUTHBOUND.costs(dates)
    tax, fee = expected_hk(SB_ROWS, 20_000.0, fx=0.0005)
    np.testing.assert_allclose(c.tax_sell, tax, rtol=REL)
    np.testing.assert_allclose(c.fee_rate_buy, fee, rtol=REL)
    assert HK_SOUTHBOUND.costs(dates, fx_cost=0).fee_rate_buy[0] == pytest.approx(
        fee[0] - 0.0005, rel=REL
    )
    with pytest.raises(ValueError, match="precede"):
        HK_SOUTHBOUND.costs(["2014-11-14"])
    with pytest.raises(ValueError, match="precede"):
        HK_DIRECT.costs(["2009-12-31"])


# (日期, SEC $/百万, TAF $/股)
US_ROWS = [
    ("2010-01-01", 25.70, 0.000075),
    ("2010-01-14", 25.70, 0.000075),
    ("2010-01-15", 12.70, 0.000075),
    ("2010-04-01", 16.90, 0.000075),
    ("2011-01-21", 19.20, 0.000075),
    ("2011-06-30", 19.20, 0.000075),
    ("2011-07-01", 19.20, 0.00009),
    ("2012-02-21", 18.00, 0.00009),
    ("2012-03-01", 18.00, 0.000095),
    ("2012-04-01", 22.40, 0.000095),
    ("2012-07-01", 22.40, 0.000119),
    ("2013-05-25", 17.40, 0.000119),
    ("2014-03-18", 22.10, 0.000119),
    ("2015-02-14", 18.40, 0.000119),
    ("2016-02-15", 18.40, 0.000119),
    ("2016-02-16", 21.80, 0.000119),
    ("2017-07-04", 23.10, 0.000119),
    ("2018-05-22", 13.00, 0.000119),
    ("2019-04-16", 20.70, 0.000119),
    ("2020-02-18", 22.10, 0.000119),
    ("2021-02-24", 22.10, 0.000119),
    ("2021-02-25", 5.10, 0.000119),
    ("2022-01-01", 5.10, 0.00013),
    ("2022-05-14", 22.90, 0.00013),
    ("2023-01-01", 22.90, 0.000145),
    ("2023-02-27", 8.00, 0.000145),
    ("2024-01-01", 8.00, 0.000166),
    ("2024-05-22", 27.80, 0.000166),
    ("2025-05-13", 27.80, 0.000166),
    ("2025-05-14", 0.0, 0.000166),
    ("2026-01-01", 0.0, 0.000195),
    ("2026-04-03", 0.0, 0.000195),
    ("2026-04-04", 20.60, 0.000195),
    ("2026-09-30", 20.60, 0.000195),
    ("2026-10-01", 20.60, 0.0),
    ("2026-12-31", 20.60, 0.0),
    ("2027-01-01", 20.60, 0.000195),
]


def test_us_boundaries():
    dates = [r[0] for r in US_ROWS]
    c = US.costs(dates, fill_value=None)
    np.testing.assert_allclose(
        c.fee_rate_sell, [r[1] * 1e-6 for r in US_ROWS], rtol=REL
    )
    np.testing.assert_array_equal(c.fee_per_share_sell, [r[2] for r in US_ROWS])
    for k in ("tax_buy", "tax_sell", "fee_rate_buy", "fee_per_share_buy"):
        assert not getattr(c, k).any()
    rounded = US.costs(dates, fill_value=5000.0)
    extra = [((r[1] > 0) + (r[2] > 0)) * 0.01 / 5000 for r in US_ROWS]
    np.testing.assert_allclose(
        rounded.fee_rate_sell - c.fee_rate_sell, extra, atol=1e-15
    )
    caps = US.charges(dates)["taf_cap"]
    assert caps[0] == 3.75 and caps[-1] == 9.79 and caps[-3] == 0


# ---------------------------------------------------------------------------
# 覆盖、档案入口与 A 股不变。
# ---------------------------------------------------------------------------


def test_registry_and_ashare_identity():
    assert markets.get("ashare") is ASHARE
    assert markets.get("HK-Southbound") is HK_SOUTHBOUND
    assert markets.get("hk_direct") is HK_DIRECT and markets.get("us") is US
    assert set(markets.PROFILES) == {"ashare", "hk_direct", "hk_southbound", "us"}
    with pytest.raises(ValueError, match="unknown market"):
        markets.get("jp")
    d = ["2008-09-19", "2023-08-28"]
    a, b = markets.get("ashare").costs(d), ASHARE.costs(d)
    for k in ("tax_buy", "tax_sell", "fee_rate_buy", "fee_rate_sell"):
        np.testing.assert_array_equal(getattr(a, k), getattr(b, k))
    assert type(ASHARE) is markets.MarketProfile  # 仍是原来的类，没有被换掉


def test_overrides_replace_final_values():
    c = HK_DIRECT.costs(
        ["2026-01-05"],
        commission_rate=0.001,
        commission_min=50,
        tax_buy=0.002,
        slippage=0.001,
    )
    assert c.commission_rate == 0.001 and c.commission_min == 50
    np.testing.assert_array_equal(c.tax_buy, 0.002)
    assert c.tax_sell[0] == pytest.approx(0.001 + 1 / 20_000, rel=REL)
    with pytest.raises(ValueError, match="unknown cost"):
        HK_DIRECT.costs(["2026-01-05"], comission_rate=0.001)
    with pytest.raises(ValueError, match="Stock Connect"):
        HK_DIRECT.costs(["2026-01-05"], fx_cost=0.001)
    with pytest.raises(ValueError, match="fill_value"):
        HK_DIRECT.costs(["2026-01-05"], fill_value=0)


def test_fill_value_converts_fixed_charges():
    """近似的含义：成交额恰好等于 fill_value 时，比例费用等于逐笔规则的真实金额上限。"""
    fv = 7_777.0
    c = HK_DIRECT.costs(["2022-06-01"], fill_value=fv)
    true_other = (
        0.000027 * fv + 0.0000015 * fv + 0.00005 * fv + max(0.00002 * fv, 2) + 0.5
    )
    true_tax_upper = np.ceil(0.0013 * fv)
    assert c.fee_rate_buy[0] * fv == pytest.approx(true_other, rel=1e-12)
    assert c.tax_buy[0] * fv == pytest.approx(0.0013 * fv + 1, rel=1e-12)
    assert c.tax_buy[0] * fv >= true_tax_upper


# ---------------------------------------------------------------------------
# config：每手、分红税、退市、卖出锁定、保证金。
# ---------------------------------------------------------------------------


def test_lot_sizes():
    cfg = HK_DIRECT.config(lot_sizes=[100, 2000], h_share=[False, False])
    np.testing.assert_array_equal(cfg.min_qty, [100, 2000])
    np.testing.assert_array_equal(cfg.qty_step, [100, 2000])
    with pytest.warns(UserWarning, match="largest lot"):
        cfg = HK_DIRECT.config(
            lot_sizes=[[100, 500], [100, 1000], [100, 200]], h_share=[False, False]
        )
    np.testing.assert_array_equal(cfg.min_qty, [100, 1000])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        HK_DIRECT.config(lot_sizes=[[100, 500], [100, 500]], h_share=[False, False])
    with pytest.raises(ValueError, match="lot_sizes"):
        HK_DIRECT.config(n=3)
    with pytest.raises(ValueError, match="board"):
        HK_DIRECT.config(board=[0], lot_sizes=[100])
    np.testing.assert_array_equal(US.config(n=3).qty_step, [1, 1, 1])
    np.testing.assert_array_equal(US.config(n=2, fractional=True).min_qty, [0, 0])


def test_dividend_withholding():
    assert HK_DIRECT.config(lot_sizes=[100, 100]).dividend_tax_rate == 0.10
    assert (
        HK_DIRECT.config(lot_sizes=[100, 100], h_share=[False, False]).dividend_tax_rate
        == 0.0
    )
    with pytest.warns(UserWarning, match="highest rate"):
        cfg = HK_DIRECT.config(lot_sizes=[100, 100], h_share=[True, False])
    assert cfg.dividend_tax_rate == 0.10
    dates = ["2026-08-31", "2026-09-01"]
    rates = HK_DIRECT.withholding(
        3, dates, h_share=[True, True, False], fie_h_share=[False, True, False]
    )
    np.testing.assert_array_equal(rates, [[0.1, 0.0, 0.0], [0.1, 0.2, 0.0]])
    with pytest.warns(UserWarning):
        cfg = HK_DIRECT.config(
            lot_sizes=[100] * 3,
            dates=dates,
            h_share=[True, True, False],
            fie_h_share=[False, True, False],
        )
    np.testing.assert_array_equal(cfg.dividend_tax_rate, [0.1, 0.2])
    with pytest.raises(ValueError, match="dates"):
        HK_DIRECT.config(lot_sizes=[100], h_share=[True], fie_h_share=[True])
    with pytest.raises(ValueError, match="H shares"):
        HK_DIRECT.withholding(1, dates, h_share=[False], fie_h_share=[True])
    # 港股通一律 20%，H 股标记不影响。
    assert (
        HK_SOUTHBOUND.config(
            lot_sizes=[100, 100], h_share=[True, False]
        ).dividend_tax_rate
        == 0.2
    )
    assert US.config(n=2).dividend_tax_rate == 0.10
    assert (
        US.config(n=2, withholding=US_WITHHOLDING["no_treaty"]).dividend_tax_rate
        == 0.30
    )
    assert US.config(n=2, withholding=0.0).dividend_tax_rate == 0.0
    with pytest.warns(UserWarning):
        assert (
            US.config(n=2, withholding=0.3, adr=[True, False]).dividend_tax_rate == 0.3
        )
    np.testing.assert_array_equal(
        US.withholding(2, adr=[True, False], adr_withholding=0.0), [0.0, 0.1]
    )


def test_net_dividends_exact_long_only():
    div = np.array([[0.0, 0.0], [1.0, 2.0]])
    np.testing.assert_array_equal(
        net_dividends(div, [0.1, 0.0], bt.MarginModel()), [[0, 0], [0.9, 2.0]]
    )
    with pytest.raises(ValueError):
        net_dividends(div, [1.5, 0], bt.MarginModel())


def test_delisting_modes():
    p = np.array([[10.0], [np.nan], [np.nan]])
    delist = np.array([[False], [False], [True]])
    t = bt.Targets.from_dense([[1000.0], [np.nan], [np.nan]], kind="shares")
    costs = bt.Costs(
        commission_rate=0, commission_min=0, fee_rate_buy=0, fee_rate_sell=0, slippage=0
    )
    navs = {}
    for mode in ("zero", "last"):
        cfg = HK_DIRECT.config(
            lot_sizes=[100], h_share=[False], delisting=mode, initial_cash=20_000
        )
        assert cfg.delist_recovery == (0.0 if mode == "zero" else 1.0)
        r = bt.run(HK_DIRECT.market(p, p, delist=delist), t, costs, cfg)
        navs[mode] = r.nav
    # 冻结到除牌日：停牌期间按最后价格估值；除牌日按 0 或按最后价格了结。
    np.testing.assert_allclose(navs["zero"], [1.0, 1.0, 0.5])
    np.testing.assert_allclose(navs["last"], [1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="delisting"):
        HK_DIRECT.config(lot_sizes=[100], delisting="par")


def test_sell_lock_and_margin_rules():
    with pytest.raises(ValueError, match="sell_lock=False"):
        US.config(n=1, rules=bt.Rules(fill_time="close"))
    US.config(n=1, rules=bt.Rules(fill_time="close", sell_lock=False))
    assert HK_DIRECT.sell_lock == 0 and US.sell_lock == 0 and not HK_DIRECT.price_limits
    with pytest.raises(ValueError, match="neither short"):
        HK_SOUTHBOUND.config(lot_sizes=[100], margin=bt.MarginModel(allow_short=True))
    us = reg_t_margin(financing_rate=0.06)
    m = us.config(n=1, delisting="last").margin
    assert m.allow_short and m.long_initial == 0.5 and m.short_initial == 0.5
    assert m.long_maintenance == 0.25 and m.short_maintenance == 0.30
    assert m.borrow_rate == pytest.approx(
        0.003 / 252
    ) and m.financing_rate == pytest.approx(0.06 / 252)
    hk = hk_margin(
        long_initial=0.5,
        long_maintenance=0.3,
        short_initial=0.5,
        short_maintenance=0.3,
        financing_rate=0.07,
    )
    assert hk.config(lot_sizes=[100], delisting="last").margin.allow_short
    assert hk.items == HK_DIRECT.items
    # 默认借券费年 0.3%（假设），按 252 期折算。
    assert hk.margin.borrow_rate == pytest.approx(0.003 / 252, rel=1e-12)


def test_connect_eligible_blocks_buys_only():
    p = np.full((2, 2), 10.0)
    e = np.array([[True, False], [True, False]])
    m = HK_SOUTHBOUND.market(p, p, connect_eligible=e)
    np.testing.assert_array_equal(m.can_buy, e)
    assert m.can_sell.all()
    with pytest.raises(ValueError, match="Stock Connect"):
        HK_DIRECT.market(p, p, connect_eligible=e)
    with pytest.raises(ValueError, match="shape"):
        HK_SOUTHBOUND.market(p, p, connect_eligible=e[:1])
    m = HK_DIRECT.market(p, p, suspended=~e)
    np.testing.assert_array_equal(m.can_buy, e)


def test_settlement_days():
    np.testing.assert_array_equal(
        settlement_days(["2017-09-01", "2017-09-05", "2024-05-24", "2024-05-28"]),
        [3, 2, 2, 1],
    )


# ---------------------------------------------------------------------------
# 折算成人民币的小工具。
# ---------------------------------------------------------------------------


def test_home_currency_nav():
    np.testing.assert_allclose(
        home_currency_nav([1.0, 1.1], [0.9, 0.95]), [1.0, 1.1 * 0.95 / 0.9]
    )
    np.testing.assert_allclose(
        home_currency_nav([1.0], [0.9], base_fx=0.8), [0.9 / 0.8]
    )
    with pytest.raises(ValueError):
        home_currency_nav([1.0, 1.0], [0.9, -1])


def test_connect_home_equity_hand():
    """10,000 港币起步：第 0 天全部换成股票（现金 0），第 1 天卖出一半（现金 5,500）。
    人民币现金：10,000×0.9 − 10,000×0.9 = 0；再 + 5,500×0.92 = 5,060。
    第 1 天股票 5,500 港币按 0.92 折算 5,060。合计 10,120。"""
    eq = connect_home_equity(
        [10_000.0, 11_000.0], [0.0, 5_500.0], [0.9, 0.92], 10_000.0
    )
    np.testing.assert_allclose(eq, [9_000.0, 10_120.0])
    # 汇率不变时等于港币权益 × 汇率。
    e, c = np.array([1e4, 1.2e4, 0.9e4]), np.array([2e3, 5e3, 9e3])
    np.testing.assert_allclose(connect_home_equity(e, c, np.full(3, 0.9), 1e4), e * 0.9)


def test_run_end_to_end_all_profiles():
    """三份档案在合成数据上都能跑通，费用全为非负。"""
    rng = np.random.default_rng(0)
    p = 20 * np.cumprod(1 + rng.normal(0, 0.01, (30, 4)), axis=0)
    dates = np.arange("2026-01-05", 30, dtype="datetime64[D]")
    w = np.full((30, 4), np.nan)
    w[0], w[10], w[20] = 0.25, [0.4, 0.1, 0.4, 0.1], 0
    t = bt.Targets.from_dense(w)
    for prof, cfg in (
        (HK_DIRECT, dict(lot_sizes=[100] * 4, h_share=[False] * 4)),
        (HK_SOUTHBOUND, dict(lot_sizes=[100] * 4)),
        (US, dict(n=4)),
    ):
        r = bt.run(
            prof.market(p, p),
            t,
            prof.costs(dates),
            prof.config(**cfg, initial_cash=1e6),
        )
        assert (
            np.isfinite(r.nav).all()
            and (r.other_fee >= 0).all()
            and r.commission.sum() > 0
        )


def test_readme_examples_run():
    """README 新小节里的三个例子各自能单独照抄运行，去掉导入后十行以内，不触发告警。"""
    import pathlib
    import re

    root = pathlib.Path(__file__).resolve().parents[1] / "packages/panelbt"
    for name in ("README.md", "README.zh-CN.md"):
        text = (root / name).read_text()
        start = text.index(
            "## Hong Kong and US profiles"
            if name == "README.md"
            else "## 港股与美股档案"
        )
        section = text[start : text.index("\n## ", start + 5)]
        blocks = re.findall(r"```python\n(.*?)```", section, re.S)
        assert len(blocks) == 3
        for code in blocks:
            body = [
                line
                for line in code.splitlines()
                if line.strip() and not line.startswith(("import ", "from "))
            ]
            assert len(body) <= 10, (name, body)
            scope: dict = {}  # 每段在全新的作用域里跑：不能依赖前一段的数据
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                exec(code, scope)
            assert np.isfinite(scope["result"].nav).all()


# ---------------------------------------------------------------------------
# 复核返工（P1、P2）新增的行为。
# ---------------------------------------------------------------------------


def _short_delisted(delisting):
    """做空 1000 股 @10，之后停牌，第 3 行除牌。"""
    p = np.array([[10.0], [np.nan], [np.nan]])
    delist = np.array([[False], [False], [True]])
    t = bt.Targets.from_dense([[-1000.0], [np.nan], [np.nan]], kind="shares")
    costs = bt.Costs(
        commission_rate=0, commission_min=0, fee_rate_buy=0, fee_rate_sell=0, slippage=0
    )
    us = reg_t_margin(financing_rate=0.0, borrow_rate=0.0)
    cfg = us.config(n=1, delisting=delisting, initial_cash=20_000)
    return bt.run(us.market(p, p, delist=delist), t, costs, cfg).nav


def test_short_delisting_requires_explicit_choice():
    us = reg_t_margin(financing_rate=0.06)
    with pytest.raises(ValueError, match="optimistic for shorts"):
        us.config(n=1)
    with pytest.raises(ValueError, match="explicitly"):
        HK_DIRECT.config(lot_sizes=[100], h_share=[False], margin=us.margin)
    # 显式给出即可；直接给 delist_recovery 也算显式。
    assert us.config(n=1, delist_recovery=0.5).delist_recovery == 0.5
    # 只做多仍默认 "zero"。
    assert HK_DIRECT.config(lot_sizes=[100], h_share=[False]).delist_recovery == 0.0
    # 记录：显式 "zero" 时空头按 0 回补，净值 1 → 1.5（乐观）；"last" 按前收回补，净值不变。
    np.testing.assert_allclose(_short_delisted("zero"), [1.0, 1.0, 1.5])
    np.testing.assert_allclose(_short_delisted("last"), [1.0, 1.0, 1.0])


def test_resident_enterprise_withholding():
    rates = HK_DIRECT.withholding(
        3, h_share=[True, False, False], resident_enterprise=[False, True, False]
    )
    np.testing.assert_array_equal(rates, [0.10, 0.10, 0.0])
    with pytest.warns(UserWarning, match="highest rate"):
        cfg = HK_DIRECT.config(
            lot_sizes=[100, 100],
            h_share=[False, False],
            resident_enterprise=[True, False],
        )
    assert cfg.dividend_tax_rate == 0.10
    # 默认不启用：非 H 股仍按 0%。
    assert HK_DIRECT.config(lot_sizes=[100], h_share=[False]).dividend_tax_rate == 0.0
    # 港股通不受影响，一律 20%。
    assert (
        HK_SOUTHBOUND.config(
            lot_sizes=[100], resident_enterprise=[True]
        ).dividend_tax_rate
        == 0.2
    )
    with pytest.raises(ValueError, match="resident_enterprise"):
        HK_DIRECT.withholding(2, h_share=[False, False], resident_enterprise=[True])


def test_missing_h_share_warns_pessimistic_fallback():
    with pytest.warns(UserWarning, match="h_share not given"):
        cfg = HK_DIRECT.config(lot_sizes=[100, 100])
    assert cfg.dividend_tax_rate == 0.10
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        HK_DIRECT.config(
            lot_sizes=[100, 100], h_share=[True, False], cash_div=np.zeros((2, 2))
        )
        HK_SOUTHBOUND.config(lot_sizes=[100])  # 港股通不需要 H 股标记


def test_dividend_rate_only_over_paying_securities():
    div = np.array([[0.0, 0.5], [0.3, 0.0], [0.0, 0.0], [0.2, 0.4]])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        cfg = HK_DIRECT.config(
            lot_sizes=[100, 100], h_share=[True, False], cash_div=div[:3]
        )
    # 第 0 行只有非 H 股派息 → 0%；第 1 行只有 H 股 → 10%；无派息的行取最低值。
    np.testing.assert_array_equal(cfg.dividend_tax_rate, [0.0, 0.10, 0.0])
    with pytest.warns(UserWarning, match="highest rate"):
        cfg = HK_DIRECT.config(
            lot_sizes=[100, 100], h_share=[True, False], cash_div=div
        )
    np.testing.assert_array_equal(cfg.dividend_tax_rate, [0.0, 0.10, 0.0, 0.10])
    with pytest.raises(ValueError, match="cash_div"):
        HK_DIRECT.config(
            lot_sizes=[100, 100], h_share=[True, False], cash_div=np.zeros((3, 3))
        )


def test_net_dividends_refuses_short_accounts():
    div = np.array([[0.0], [1.0]])
    us = reg_t_margin(financing_rate=0.06)
    with pytest.raises(ValueError, match="gross"):
        net_dividends(div, [0.1], us.margin)
    with pytest.raises(ValueError, match="gross"):
        net_dividends(div, [0.1], us.config(n=1, delisting="last"))
    np.testing.assert_array_equal(
        net_dividends(div, [0.1], us.margin, allow_short=True), [[0.0], [0.9]]
    )
    np.testing.assert_array_equal(
        net_dividends(div, [0.1], bt.MarginModel()), [[0.0], [0.9]]
    )
    with pytest.raises(ValueError, match="account"):
        net_dividends(div, [0.1], None)


def test_fx_cost_survives_fee_rate_override():
    d = ["2026-01-05"]
    c = HK_SOUTHBOUND.costs(d, fee_rate_buy=1e-4, fee_rate_sell=2e-4)
    assert float(c.fee_rate_buy) == pytest.approx(1e-4 + 0.0005, rel=1e-12)
    assert float(c.fee_rate_sell) == pytest.approx(2e-4 + 0.0005, rel=1e-12)
    c = HK_SOUTHBOUND.costs(d, fee_rate_buy=1e-4, fx_cost=0)
    assert float(c.fee_rate_buy) == 1e-4
    assert HK_SOUTHBOUND.costs(d, commission_rate=0.001).commission_rate == 0.001


def test_us_adr_default_withholding():
    # 不给 adr_withholding 时 ADR 按 10%（假设）；与普通股默认相同，所以不告警。
    np.testing.assert_array_equal(
        US.withholding(2, adr=[True, False], withholding=0.0), [0.10, 0.0]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert US.config(n=2, adr=[True, False]).dividend_tax_rate == 0.10


def test_converters_reject_missing_rates():
    with pytest.raises(ValueError):
        home_currency_nav([1.0, 1.0], [0.9, np.nan])
    with pytest.raises(ValueError):
        connect_home_equity([1.0, 1.0], [0.0, 0.0], [0.9, np.nan], 1.0)
    with pytest.raises(ValueError):
        home_currency_nav([1.0, 1.0, 1.0], [0.9, 0.9])  # 行数不对齐
