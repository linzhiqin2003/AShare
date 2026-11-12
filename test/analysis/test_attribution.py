"""收益归因：参考实现对拍、恒等式、已知答案的合成用例、与 panelbt 的对账。"""

import math

import numpy as np
import panelbt
import pytest
from analysis_fixtures import FakePanel, make_panel, random_history, random_weights
from analysis_reference import (
    ref_attribution,
    ref_carino_total,
    ref_grap_total,
    ref_total_returns,
)

import ashare
import ashare.analysis as an

EPS = np.finfo(float).eps


def test_imports_point_at_worktree():
    assert "/.worktrees/analysis/" in ashare.__file__
    assert "/.worktrees/analysis/" in panelbt.__file__


def test_total_returns_matches_reference():
    panel = make_panel()
    ours = an.total_returns(panel)
    ref = np.array(
        ref_total_returns(panel["close"].tolist(), panel["adj_factor"].tolist())
    )
    np.testing.assert_allclose(ours, ref, rtol=0, atol=1e-15, equal_nan=True)
    # 除权日（复权因子跳升、价格下跌）总收益连续；停牌日收益为 0。
    assert abs(ours[30, 1]) < 0.2
    assert np.all(ours[41:43, 2] == 0)


@pytest.mark.parametrize("with_result", [False, True])
def test_decompose_matches_reference(with_result):
    panel = make_panel(seed=3)
    hist = random_history(panel, drop=(np.s_[50], np.s_[4]))  # 第 50 日第 4 只无暴露
    hist.factor_returns[70] = np.nan  # 一天没有因子收益
    w = random_weights(panel)
    held = an.held_from_weights(w)
    hb = an.benchmark_held(panel, "weight_bench", an.total_returns(panel))
    r = an.total_returns(panel)
    rng = np.random.default_rng(9)
    rp = cost = None
    if with_result:
        cost = rng.uniform(0, 1e-3, len(r))
        rp = (held * np.nan_to_num(r)).sum(axis=1) + rng.normal(0, 1e-3, len(r)) - cost
    out = an.decompose(held, hb, r, hist.exposures, hist.factor_returns, rp, cost)
    ref = ref_attribution(
        held.tolist(),
        hb.tolist(),
        r.tolist(),
        hist.exposures.tolist(),
        hist.factor_returns.tolist(),
        None if rp is None else rp.tolist(),
        None if cost is None else cost.tolist(),
    )
    np.testing.assert_allclose(out["factor"], [x[0] for x in ref], atol=1e-15)
    np.testing.assert_allclose(out["specific"], [x[1] for x in ref], atol=1e-15)
    np.testing.assert_allclose(out["trading"], [x[2] for x in ref], atol=1e-15)
    np.testing.assert_allclose(out["cost"], [x[3] for x in ref], atol=1e-15)
    side = ref_attribution(
        held.tolist(),
        np.zeros_like(held).tolist(),
        r.tolist(),
        hist.exposures.tolist(),
        hist.factor_returns.tolist(),
    )
    np.testing.assert_allclose(
        out["specific_portfolio"], [x[1] for x in side], atol=1e-15
    )
    np.testing.assert_allclose(
        out["specific_portfolio"] - out["specific_benchmark"],
        out["specific"],
        atol=1e-16,
    )
    # 用例确实走到了未覆盖与无因子收益两个分支。
    assert out["uncovered"][51] > 0 and out["uncovered"][52] == 0
    assert not out["covered_factor_days"][70]
    assert np.all(out["factor"][70] == 0)
    total = out["factor"].sum(1) + out["specific"] + out["trading"] + out["cost"]
    np.testing.assert_allclose(total, out["portfolio"] - out["benchmark"], atol=1e-16)


@pytest.mark.parametrize("method", ["grap", "carino"])
def test_link_matches_direct_formula(method):
    rng = np.random.default_rng(4)
    t, m = 60, 5
    rp, rb = rng.normal(0.001, 0.02, t), rng.normal(0.0005, 0.015, t)
    rb[10] = rp[10]  # 某日组合与基准收益相等（Carino 的极限分支）
    a = rng.normal(0, 1, (t, m))
    a *= ((rp - rb) / a.sum(axis=1))[:, None]
    ours = an.link(a, rp, rb, method)
    ref = (ref_grap_total if method == "grap" else ref_carino_total)(
        a.tolist(), rp.tolist(), rb.tolist()
    )
    np.testing.assert_allclose(ours[-1], ref, rtol=1e-11, atol=1e-14)
    target = np.prod(1 + rp) - np.prod(1 + rb)
    assert abs(ours[-1].sum() - target) < 64 * EPS
    # 逐日的累计值也满足恒等式。
    np.testing.assert_allclose(ours.sum(1), an.cumulative_active(rp, rb), atol=1e-14)


def test_link_absolute_and_equal_returns():
    rp = np.array([0.01, -0.02, 0.03])
    a = np.column_stack([rp * 0.4, rp * 0.6])
    for method in ("grap", "carino"):
        out = an.link(a, rp, None, method)
        assert abs(out[-1].sum() - (np.prod(1 + rp) - 1)) < 8 * EPS
        np.testing.assert_allclose(out[-1, 0] / out[-1, 1], 0.4 / 0.6)
    same = an.link(np.zeros((3, 2)), rp, rp, "carino")
    assert np.all(same == 0)


@pytest.mark.parametrize("method", ["grap", "carino"])
def test_attribute_identity_on_weights(method):
    panel = make_panel(seed=5)
    hist = random_history(panel)
    w = random_weights(panel)
    rep = an.attribute(w, panel, hist, benchmark="weight_bench", method=method)
    ident = rep.summary["identity"]
    assert ident["daily_max_abs"] < 1e-16
    assert ident["cumulative_max_eps"] < 64, ident
    assert ident["groups_cumulative_max_abs"] < 64 * EPS
    # 因子明细按类别相加 = 类别表。
    detail = rep.cumulative
    groups = rep.groups_cumulative.last()
    ind = [c for c in detail.columns if c.startswith("industry")]
    sty = [c for c in detail.columns if c.startswith("style")]
    assert abs(sum(detail.last()[c] for c in ind) - groups["industry"]) < 1e-15
    assert abs(sum(detail.last()[c] for c in sty) - groups["style"]) < 1e-15
    assert rep.to_dict()["summary"]["method"] == method
    assert (
        abs(sum(rep.summary["groups"].values()) - rep.summary["active_cumulative"])
        < 1e-14
    )


def _single_factor_case(seed=7):
    """收益完全由因子决定（无特质），主动组合只暴露 style_1。"""
    base = make_panel(seed=seed, suspend=False)
    hist = random_history(base, seed=seed)
    t, n, k = hist.exposures.shape
    r = np.zeros((t, n))
    r[1:] = np.einsum("tnk,tk->tn", hist.exposures[:-1], hist.factor_returns[1:])
    close = 10 * np.cumprod(1 + r, axis=0)
    fields = dict(base.fields)
    fields["close"] = close
    fields["adj_factor"] = np.ones((t, n))
    panel = FakePanel(base.dates, base.symbols, fields)
    target = hist.factors.index("style_1")
    bench_end = np.zeros((t, n))
    port_end = np.zeros((t, n))
    hb = an.benchmark_held(panel, "weight_bench", an.total_returns(panel))
    bench_end[:-1] = hb[1:]
    for day in range(t):
        b = hist.exposures[day]
        others = np.delete(b, target, axis=1)
        # 投影到「其余因子暴露为 0」的子空间，再沿 style_1 方向取一个主动组合。
        q, _ = np.linalg.qr(others)
        v = b[:, target] - q @ (q.T @ b[:, target])
        port_end[day] = bench_end[day] + 0.05 * v / np.abs(v).sum()
    return panel, hist, port_end, bench_end, target


def test_single_factor_portfolio_attributes_everything_to_that_factor():
    panel, hist, w, _, target = _single_factor_case()
    rep = an.attribute(w, panel, hist, benchmark="weight_bench")
    daily = rep.daily.values
    name = hist.factors[target]
    others = [j for j, c in enumerate(rep.daily.columns) if c != name]
    assert np.abs(daily[:, others]).max() < 1e-15
    assert np.abs(rep.daily[name]).max() > 1e-5
    exp = rep.exposures.values
    assert np.abs(np.delete(exp, target, axis=1)).max() < 1e-13
    assert abs(rep.cumulative.last()[name] - rep.summary["active_cumulative"]) < 1e-14


def test_identical_to_benchmark_gives_zero():
    panel = make_panel(seed=8)
    hist = random_history(panel)
    r = an.total_returns(panel)
    hb = an.benchmark_held(panel, "weight_bench", r)
    w = np.zeros_like(hb)
    w[:-1] = hb[1:]
    rep = an.attribute(w, panel, hist, benchmark="weight_bench")
    assert np.all(rep.daily.values == 0)
    assert np.all(rep.cumulative.values == 0)
    assert rep.summary["active_cumulative"] == 0


def test_benchmark_drifts_between_snapshots():
    panel = make_panel(seed=10, monthly=20)
    r = an.total_returns(panel)
    hb = an.benchmark_held(panel, "weight_bench", r)
    raw = panel["weight_bench"]
    np.testing.assert_allclose(hb[20], raw[20] / raw[20].sum())  # 新快照当天
    grown = hb[20] * (1 + np.nan_to_num(r[20]))
    np.testing.assert_allclose(hb[21], grown / grown.sum())  # 次日漂移
    assert not np.allclose(hb[21], raw[21])


def _market(panel):
    p = panel.fields
    return panelbt.Market(
        exec_price=p["open"],
        close=p["close"],
        can_buy=np.isfinite(p["open"]),
        can_sell=np.isfinite(p["open"]),
        volume=p["volume"],
        split=p["split"],
        cash_div=p["cash_dividend"],
        delist=p["delist"],
    )


def _result_case(seed=11):
    panel = make_panel(seed=seed)
    # 让 panelbt 看到除权：10 送 2.5（复权因子 ×1.25 那天）。
    panel.fields["split"][30, 1] = 1.25
    t, n = len(panel.dates), len(panel.symbols)
    rng = np.random.default_rng(seed)
    dense = np.full((t, n), np.nan)
    for day in range(1, t, 15):
        raw = rng.uniform(0, 1, n)
        dense[day] = 0.95 * raw / raw.sum()
    costs = panelbt.Costs(impact_k=np.full((t, n), 0.2), tax_sell=0.001, slippage=5e-4)
    config = panelbt.Config(initial_cash=1e7, min_qty=0, qty_step=0, record_trades=True)
    result = panelbt.run(
        _market(panel), panelbt.Targets.from_dense(dense), costs, config
    )
    return panel, result, dense


def test_attribute_panelbt_result_trading_and_cost():
    panel, result, dense = _result_case()
    hist = random_history(panel)
    rep = an.attribute(result, panel, hist, benchmark="weight_bench")
    assert rep.summary["identity"]["daily_max_abs"] < 1e-16
    assert rep.summary["identity"]["cumulative_max_eps"] < 64
    trade_days = set(np.flatnonzero(~np.all(np.isnan(dense), axis=1)))
    rows = {d: i for i, d in enumerate(rep.daily.index)}
    idx = np.array([i for i, d in enumerate(panel.dates) if d in rows])
    trading, cost = rep.daily["trading"], rep.daily["cost"]
    quiet = [rows[panel.dates[d]] for d in idx if d not in trade_days]
    busy = [rows[panel.dates[d]] for d in idx if d in trade_days]
    # 不交易的日子：持仓收益完全解释净值收益（公司行为口径一致，交易项只剩浮点误差）。
    assert np.abs(trading[quiet]).max() < 1e-12
    assert np.all(cost[quiet] == 0)
    assert np.all(cost[busy] < 0) and np.abs(trading[busy]).max() > 1e-6
    # 成本项 = −(佣金+税+过户费+滑点+冲击) / (E[t-1] + 资金流)。
    e = result.equity
    day = sorted(trade_days)[2]
    expected = (
        -(
            result.commission[day]
            + result.tax[day]
            + result.other_fee[day]
            + result.slippage_cost[day]
            + result.impact_cost[day]
        )
        / e[day - 1]
    )
    assert math.isclose(cost[rows[panel.dates[day]]], expected, rel_tol=1e-13)
    # 净值收益与报告里的组合收益一致。
    np.testing.assert_allclose(
        rep.returns["portfolio"], result.returns[idx], rtol=0, atol=0
    )


def test_attribute_requires_positions():
    panel, result, _ = _result_case()
    result.positions = None
    with pytest.raises(ValueError, match="record_trades"):
        an.attribute(result, panel, random_history(panel))


def test_attribute_from_model_iterable_matches_history():
    """直接传风险模型序列：内部收集只需要的列，结果与先建 FactorHistory 一致。"""
    from ashare.risk import model_series

    panel = make_panel(t=140, n=14, seed=12)
    w = random_weights(panel)
    source, cfg = an.risk_source(panel)
    hist = an.FactorHistory.collect(
        model_series(source, config=cfg, threads=1), source, config=cfg
    )
    a = an.attribute(w, panel, hist, benchmark="weight_bench")
    b = an.attribute(
        w, panel, model_series(source, config=cfg, threads=1), benchmark="weight_bench"
    )
    np.testing.assert_array_equal(a.daily.values, b.daily.values)
