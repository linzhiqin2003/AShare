"""容量与成本：单笔成本公式、与 panelbt 实跑对账、容量求解、变现天数、成本拆解。"""

import dataclasses
import json
import math

import numpy as np
import panelbt
import pytest
from analysis_fixtures import make_panel
from analysis_reference import ref_trade_cost

import ashare.analysis as an
from ashare.analysis.adapters import ashare_costs, market_from_panel
from ashare.analysis.capacity import MarketArrays, _dense_targets, _fills


def _event_panel(seed=3, t=200, n=20):
    panel = make_panel(t=t, n=n, seed=seed)  # 第 2 只 40–42 日停牌（无成交量）
    f = panel.fields
    f["split"][30, 1] = 1.25
    f["cash_dividend"][50, 4] = 0.3
    f["close"][50:, 4] -= 0.3
    f["open"][50:, 4] -= 0.3
    # 同一天先派现（按送转前股数）再 10 送 10。
    f["cash_dividend"][70, 6] = 0.2
    f["split"][70, 6] = 2.0
    f["adj_factor"][70:, 6] *= f["close"][69, 6] / ((f["close"][69, 6] - 0.2) / 2.0)
    f["close"][70:, 6] = (f["close"][70:, 6] - 0.2) / 2.0
    f["open"][70:, 6] = (f["open"][70:, 6] - 0.2) / 2.0
    # 停牌中（第 2 只 40–42 日）送转、派现：估值价按最近收盘逐次除权。
    ref = f["close"][39, 2] / 2.0
    f["split"][41, 2] = 2.0
    f["cash_dividend"][42, 2] = 0.1
    f["close"][43:, 2] = f["close"][43:, 2] / 2.0 - 0.1
    f["open"][43:, 2] = f["open"][43:, 2] / 2.0 - 0.1
    f["adj_factor"][41:, 2] *= 2.0
    f["adj_factor"][42:, 2] *= ref / (ref - 0.1)
    f["delist"][150, 5] = True
    f["close"][150:, 5] = np.nan
    f["open"][150:, 5] = np.nan
    # 价格改过之后重算高低价和涨跌停（否则除权后的价格会落在旧的跌停价以下）。
    f["high"] = np.fmax(f["open"], f["close"])
    f["low"] = np.fmin(f["open"], f["close"])
    f["upper_limit"] = np.nan_to_num(f["close"], nan=10) * 1.5
    f["lower_limit"] = np.nan_to_num(f["close"], nan=10) * 0.5
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
    # 主干的 A 股费用表，另加按股收取的佣金和过户费（走到 per-share 分支）。
    costs = dataclasses.replace(
        ashare_costs(panel.dates),
        commission_per_share=2e-4,
        fee_per_share_buy=1e-3,
        fee_per_share_sell=5e-4,
    )
    days, rows = _dense_targets(dense, (t, n))
    sim = an.simulate(days, rows, m, costs, k, 0.5, aum)
    market = market_from_panel(panel)
    run_costs = dataclasses.replace(costs, impact_k=k)
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
    assert at.table["over_cap_share"][0] <= 0.05 < at.table["over_cap_share"][1]


def _u_curve_case():
    """成本曲线先降后升：小规模被最低佣金 5 元拖累，大规模被冲击拖累。"""
    panel = make_panel(t=500, n=12, suspend=False)
    panel.fields["volume"] = panel.fields["volume"] * 100
    dense = np.full((500, 12), np.nan)
    for d in range(1, 500, 20):
        dense[d] = 0.0
        dense[d, :10] = 0.099
    return panel, dense


def test_capacity_takes_rising_branch_crossing_with_minimum_commission():
    panel, dense = _u_curve_case()
    cap = an.capacity(dense, panel, aum=[1e6], gross_excess=0.03)
    c = cap.capacity
    assert c["by_cost_status"] == "found" and c["min_viable_status"] == "found"
    assert c["min_viable_aum"] < c["min_cost_aum"] < c["by_cost"]
    # 两个交点上的成本都等于门槛（毛超额的一半），中间的规模低于门槛。
    probe = an.capacity(
        dense,
        panel,
        aum=[c["min_viable_aum"], c["min_cost_aum"], c["by_cost"]],
        gross_excess=0.03,
    ).table["cost_total"]
    assert math.isclose(probe[0], 0.015, rel_tol=1e-6)
    assert probe[1] < 0.015
    assert math.isclose(probe[2], 0.015, rel_tol=1e-6)
    # 小规模一侧确实是最低佣金造成的：去掉它，下降段的交点消失，上升段的交点不变。
    free = dataclasses.replace(ashare_costs(panel.dates), commission_min=0.0)
    nomin = an.capacity(dense, panel, aum=[1e6], gross_excess=0.03, costs=free)
    assert nomin.capacity["min_viable_status"] == "below_search_floor"
    assert math.isnan(nomin.capacity["min_viable_aum"])
    assert math.isclose(nomin.capacity["by_cost"], c["by_cost"], rel_tol=1e-3)
    assert c["by_cost"] != c["search_range"][0]


def test_capacity_boundary_statuses_never_return_search_floor():
    panel, dense = _u_curve_case()
    neg = an.capacity(dense, panel, aum=[1e7], gross_excess=-0.01).capacity
    assert neg["by_cost_status"] == "no_gross_excess" and math.isnan(neg["by_cost"])
    tiny = an.capacity(dense, panel, aum=[1e7], gross_excess=1e-5).capacity
    assert tiny["by_cost_status"] == "never_viable" and math.isnan(tiny["by_cost"])
    assert tiny["min_cost"] > 0.5e-5
    huge = an.capacity(dense, panel, aum=[1e7], gross_excess=50.0).capacity
    assert huge["by_cost_status"] == "beyond_search_range"
    assert huge["by_cost"] == float("inf")
    # breach_share = 0：「出现任何超限成交」的最小规模，不是搜索下限。
    zero = an.capacity(dense, panel, aum=[1e7], breach_share=0.0).capacity
    assert zero["by_participation_status"] == "found"
    assert zero["by_participation"] > zero["search_range"][0]
    at = an.capacity(
        dense,
        panel,
        aum=[zero["by_participation"] * 0.999, zero["by_participation"] * 1.001],
    ).table["over_cap_share"]
    assert at[0] == 0.0 < at[1]
    # 搜索下限处已经越过：NaN 加状态，而不是返回下限。
    low = an.capacity(
        dense, panel, aum=[1e7], breach_share=0.0, search=(1e9, 1e12)
    ).capacity
    assert low["by_participation_status"] == "below_search_floor"
    assert math.isnan(low["by_participation"])
    assert json.dumps(an.capacity(dense, panel, aum=[1e7]).to_dict()) is not None


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
    market = market_from_panel(panel)
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


def test_capacity_treats_wiped_out_small_accounts_as_not_viable():
    """几千元的账户被最低佣金打穿（权益 ≤ 0）时，不能把负的成本率当成「成本最低」。"""
    panel, dense = _u_curve_case()
    c = an.capacity(dense, panel, aum=[1e6], gross_excess=0.03, search=(1e2, 1e12))
    c = c.capacity
    assert c["min_cost"] > 0
    assert c["min_cost_aum"] > 1e5
    assert c["min_viable_status"] == "found" and c["min_viable_aum"] > 1e3


def test_wiped_out_rows_in_table_are_nan_and_flagged():
    """用户给的规模里有被最低佣金打穿的点（权益变负）：成本率与扣成本后超额为 NaN，
    wiped_out 列为 1。"""
    panel, dense = _u_curve_case()
    cap = an.capacity(dense, panel, aum=[300.0, 3000.0, 1e6], gross_excess=0.03)
    tab = cap.table
    assert cap.simulations[0].final_equity < 0
    # 3000 元：最低佣金让现金有几天小额透支，但权益始终为正，不算打穿。
    assert cap.simulations[1].overdrawn > 0 and cap.simulations[1].equity.min() > 0
    assert list(tab["wiped_out"]) == [1.0, 0.0, 0.0]
    for col in ("cost_total", "cost_commission", "net_excess", "net_excess_path"):
        assert math.isnan(tab[col][0]), col
        assert np.isfinite(tab[col][1:]).all(), col
    assert json.dumps(cap.to_dict(), allow_nan=False)


def test_simulate_many_matches_one_size_at_a_time():
    """一批规模一起推进，与逐个单独模拟的结果相同（只差浮点舍入）。"""
    panel = _event_panel()
    t, n = len(panel.dates), len(panel.symbols)
    dense = _plan(t, n)
    m = MarketArrays.from_panel(panel)
    k = an.impact_coefficients(panel)
    costs = ashare_costs(panel.dates)
    days, rows = _dense_targets(dense, (t, n))
    # 只拦卖或只拦买的格子：不同规模对同一只股票的方向可能相反（大规模冲击大、少买了
    # 流动性差的股票，下次调仓要补买；小规模则要卖），挂单要按规模各记各的。
    rng = np.random.default_rng(7)
    m.can_sell = m.can_sell & (rng.uniform(size=m.can_sell.shape) > 0.15)
    m.can_buy = m.can_buy & (rng.uniform(size=m.can_buy.shape) > 0.15)
    sizes = [3e3, 3e5, 1e8, 2e9, 5e10]
    many = an.simulate_many(days, rows, m, costs, k, 0.5, sizes)
    for size, sim in zip(sizes, many, strict=True):
        one = an.simulate(days, rows, m, costs, k, 0.5, size)
        np.testing.assert_array_equal(sim.days, one.days)
        np.testing.assert_allclose(sim.costs, one.costs, rtol=1e-12, atol=1e-9)
        np.testing.assert_allclose(sim.equity, one.equity, rtol=1e-12)
        np.testing.assert_allclose(sim.participation, one.participation, rtol=1e-12)
        np.testing.assert_allclose(sim.notional, one.notional, rtol=1e-12)
        assert math.isclose(sim.final_equity, one.final_equity, rel_tol=1e-12)
        assert sim.blocked == one.blocked


def test_capacity_crossings_within_declared_tolerance():
    """多分法收敛到 rtol：交点再往可行一侧挪 rtol，就落回门槛以内。"""
    panel, dense = _u_curve_case()
    rtol = 1e-6
    cap = an.capacity(dense, panel, aum=[1e6], gross_excess=0.03, rtol=rtol)
    c = cap.capacity
    up, down = c["by_cost"], c["min_viable_aum"]
    # 批数：表与网格一批，多分法每轮把区间缩小 16 倍，0.2 个十倍缩到 1e-6 要 5 轮。
    batches = cap.assumptions["simulation_batches"]
    assert len(batches) <= 6 and batches[0] <= 1 + 46
    assert max(batches[1:]) <= 3 * 15
    probe = an.capacity(
        dense,
        panel,
        aum=[up * (1 - rtol), up, down, down * (1 + rtol)],
        gross_excess=0.03,
    ).table["cost_total"]
    # 表按规模排序：down, down·(1+rtol), up·(1−rtol), up。
    assert probe[0] >= 0.015 > probe[1]
    assert probe[2] < 0.015 <= probe[3]
    # 与很细的二分（rtol 1e-12）相比，差别在 rtol 以内。
    fine = an.capacity(dense, panel, aum=[1e6], gross_excess=0.03, rtol=1e-12)
    for key in ("by_cost", "min_viable_aum", "by_participation"):
        assert math.isclose(c[key], fine.capacity[key], rel_tol=2 * rtol), key


def test_grid_does_not_miss_cost_minimum():
    """每十倍 5 个点的网格找到的最低成本，与每十倍 40 个点的网格相差不到 2%。"""
    panel, dense = _u_curve_case()
    coarse = an.capacity(dense, panel, aum=[1e6], gross_excess=0.03).capacity
    sizes = np.logspace(4, 9, 201)
    fine = an.capacity(dense, panel, aum=sizes, gross_excess=0.03).table["cost_total"]
    assert coarse["min_cost"] <= np.nanmin(fine) * 1.02
