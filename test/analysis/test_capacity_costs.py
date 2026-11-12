"""容量与成本：单笔成本公式、与 panelbt 实跑对账、容量求解、变现天数、成本拆解。"""

import json
import math

import numpy as np
import panelbt
import pytest
from analysis_fixtures import make_panel
from analysis_reference import ref_trade_cost

import ashare.analysis as an
from ashare.analysis.capacity import MarketArrays, _dense_targets, _fills


def _event_panel(seed=3, t=200, n=20):
    panel = make_panel(t=t, n=n, seed=seed)  # 第 2 只 40–42 日停牌（无成交量）
    f = panel.fields
    f["split"][30, 1] = 1.25
    f["cash_dividend"][50, 4] = 0.3
    f["close"][50:, 4] -= 0.3
    f["open"][50:, 4] -= 0.3
    f["delist"][150, 5] = True
    f["close"][150:, 5] = np.nan
    f["open"][150:, 5] = np.nan
    return panel


def _plan(t, n, seed=0, every=20, invest=0.98):
    rng = np.random.default_rng(seed)
    dense = np.full((t, n), np.nan)
    for d in range(1, t, every):
        raw = rng.uniform(0, 1, n)
        dense[d] = invest * raw / raw.sum()
    return dense


def test_single_fill_costs_match_reference():
    panel = _event_panel()
    m = MarketArrays.from_panel(panel)
    costs = panelbt.Costs(tax_sell=0.001, commission_min=5.0)
    k = np.full(m.exec_price.shape, 0.3)
    i = np.array([0, 3, 7])
    q = np.array([1000.0, 25000.0, 7.0])
    for side in (1.0, -1.0):
        f = _fills(q, side, 12, i, m, costs, k, 0.5)
        for j, col in enumerate(i):
            ref = ref_trade_cost(
                q[j] * m.exec_price[12, col],
                side,
                m.exec_price[12, col],
                m.volume[12, col],
                0.3,
                0.5,
                5e-4,
                2.5e-4,
                5.0,
                0.001,
                1e-5,
            )
            for key in (
                "impact",
                "slippage",
                "commission",
                "tax",
                "fee",
                "participation",
            ):
                assert math.isclose(
                    f[key][j], ref[key], rel_tol=1e-13, abs_tol=1e-12
                ), key
    assert f["commission"][2] == 5.0  # 最低佣金分支


@pytest.mark.parametrize("aum", [3e5, 1e8, 2e9])
def test_simulation_reconciles_with_panelbt_relaxed_rules(aum):
    """同规则（不取整、无成交量上限）下，解析模拟与 panelbt 的成本和净值逐笔对上。"""
    panel = _event_panel()
    t, n = len(panel.dates), len(panel.symbols)
    dense = _plan(t, n)
    m = MarketArrays.from_panel(panel)
    k = an.impact_coefficients(panel)
    costs = an.default_costs(panel.dates)
    days, rows = _dense_targets(dense, (t, n))
    sim = an.simulate(days, rows, m, costs, k, 0.5, aum)
    market = an.market_from_panel(panel)
    run_costs = panelbt.Costs(tax_sell=costs.tax_sell, impact_k=k)
    cfg = panelbt.Config(initial_cash=aum, min_qty=0, qty_step=0)
    res = panelbt.run(market, panelbt.Targets.from_dense(dense), run_costs, cfg)
    assert sim.blocked > 0  # 走到了挂单重试分支（停牌、无成交量）
    assert 41 in sim.days or 42 in sim.days or 43 in sim.days
    for j, name in enumerate(
        ("commission", "tax", "other_fee", "slippage_cost", "impact_cost")
    ):
        ours = sim.costs[:, j].sum()
        theirs = getattr(res, name).sum()
        assert math.isclose(ours, theirs, rel_tol=1e-9), (name, ours, theirs)
    assert math.isclose(sim.final_equity, res.equity[-1], rel_tol=1e-10)
    # 逐日：成本率分母与 panelbt 的 E[t-1] 相同。
    np.testing.assert_allclose(sim.equity, res.equity[sim.days - 1], rtol=1e-10)
    np.testing.assert_allclose(
        sim.costs[:, 4], res.impact_cost[sim.days], rtol=1e-9, atol=1e-6
    )
    if aum >= 2e9:
        assert res.diagnostics["cash_scaled"] > 0  # 买单同比例缩小的分支


def test_capacity_rerun_table_reconciles():
    panel = _event_panel()
    t, n = len(panel.dates), len(panel.symbols)
    cap = an.capacity(
        _plan(t, n), panel, aum=[1e6, 1e8], rerun=True, benchmark="weight_bench"
    )
    r = cap.rerun
    np.testing.assert_allclose(r["relaxed_impact_ratio"], 1, rtol=1e-9)
    np.testing.assert_allclose(r["relaxed_cost_ratio"], 1, rtol=1e-9)
    np.testing.assert_allclose(r["analytic_impact"], r["relaxed_impact"], rtol=1e-9)
    np.testing.assert_allclose(
        r["analytic_net_excess"], r["relaxed_net_excess"], rtol=1e-9
    )
    # 真实规则（整手、10% 上限）在大规模下成交不全，冲击成本低于宽松规则。
    assert r["real_impact"][-1] < r["relaxed_impact"][-1]
    assert json.dumps(cap.to_dict(), allow_nan=False)


def test_capacity_scaling_and_solutions():
    panel = _event_panel()
    t, n = len(panel.dates), len(panel.symbols)
    dense = _plan(t, n)
    sizes = [1e6, 1e7, 1e8]
    cap = an.capacity(dense, panel, aum=sizes)
    tab = cap.table
    # 成交量占比与规模成正比（同一条成交路径，资金规模不改变权重）。
    ratio = tab["participation_p50"][1] / tab["participation_p50"][0]
    assert 9.5 < ratio < 10.5
    # 冲击成本按平方根律增长：规模 ×10，年化冲击约 ×sqrt(10)。
    growth = tab["cost_impact"][2] / tab["cost_impact"][1]
    assert 3.0 < growth < 3.3
    assert np.all(np.diff(tab["cost_total"]) > 0)
    np.testing.assert_allclose(
        tab["cost_total"],
        tab.values[
            :,
            [
                tab.columns.index(f"cost_{c}")
                for c in an.capacity.__globals__["COST_ITEMS"]
            ],
        ].sum(1),
    )
    c = cap.capacity
    assert c["gross_excess"] > 0
    at = an.capacity(dense, panel, aum=[c["by_cost"]])
    assert math.isclose(
        at.table["cost_total"][0], 0.5 * c["gross_excess"], rel_tol=1e-6
    )
    at = an.capacity(
        dense, panel, aum=[c["by_participation"] * 1.001, c["by_participation"] * 0.999]
    )
    assert at.table["over_cap_share"][0] < 0.05 <= at.table["over_cap_share"][1]


def test_capacity_with_impossible_gross_excess():
    panel = _event_panel()
    t, n = len(panel.dates), len(panel.symbols)
    cap = an.capacity(_plan(t, n), panel, aum=[1e7], gross_excess=-0.01)
    assert cap.capacity["by_cost"] == 0.0
    cap = an.capacity(_plan(t, n), panel, aum=[1e7], gross_excess=1e-5)
    assert cap.capacity["by_cost"] == 1e4  # 固定成本已超过毛超额的一半：任何规模都不行


def test_liquidation_days_by_hand():
    panel = _event_panel()
    t, n = len(panel.dates), len(panel.symbols)
    dense = np.full((t, n), np.nan)
    dense[60, :2] = [0.6, 0.4]
    cap = an.capacity(dense, panel, aum=[1e8])
    adv = panel["amount"][40:60, :2].mean(axis=0)
    need = np.array([0.6, 0.4]) * 1e8 / (0.1 * adv)
    liq = cap.liquidation
    assert math.isclose(
        liq["days_weighted_mean"][0], 0.6 * need[0] + 0.4 * need[1], rel_tol=1e-12
    )
    assert math.isclose(liq["days_worst_max"][0], need.max(), rel_tol=1e-12)


def test_cost_breakdown_by_year():
    panel = _event_panel(t=400)
    t, n = len(panel.dates), len(panel.symbols)
    market = an.market_from_panel(panel)
    costs = panelbt.Costs(tax_sell=0.001, impact_k=np.full((t, n), 0.2))
    res = panelbt.run(
        market,
        panelbt.Targets.from_dense(_plan(t, n)),
        costs,
        panelbt.Config(initial_cash=1e7),
    )
    tab = an.cost_breakdown(res, panel.dates)
    assert list(tab.index) == ["2019", "2020", "all"]
    items = ["commission", "stamp_tax", "other_fee", "slippage", "impact", "borrow"]
    np.testing.assert_allclose(
        tab.values[:, [tab.columns.index(c) for c in items]].sum(1), tab["total_cost"]
    )
    year = np.array([d[:4] for d in panel.dates]) == "2019"
    denominator = np.r_[1e7, res.equity[:-1]]
    expected = (res.commission / denominator)[year].sum()
    assert math.isclose(tab["commission"][0], expected, rel_tol=1e-13)
    assert math.isclose(
        tab["net_return"][0], np.prod(1 + res.returns[year]) - 1, rel_tol=1e-13
    )
    np.testing.assert_allclose(tab["turnover"], 2 * tab["turnover_one_sided"])
    assert math.isclose(
        tab["turnover_one_sided"][1], res.turnover[~year].sum(), rel_tol=1e-13
    )
    gross = tab["net_return"] + tab["total_cost"]
    np.testing.assert_allclose(tab["gross"], gross)
    assert tab["days"][-1] == t
    assert json.dumps(tab.to_dict(), allow_nan=False)


def test_volatility_uses_only_past_days():
    panel = make_panel(seed=50, suspend=False)
    vol = an.volatility(panel, window=10)
    r = an.total_returns(panel)
    day = 40
    assert math.isclose(
        vol[day, 3], np.std(r[day - 10 : day, 3], ddof=1), rel_tol=1e-12
    )
    panel.fields["close"][day:, 3] *= 1.5  # 改当日及以后的价格，不影响当日的 k
    assert vol[day, 3] == an.volatility(panel, window=10)[day, 3]
