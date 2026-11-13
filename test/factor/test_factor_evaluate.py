"""单因子评估对直白参考实现（pandas / NumPy 逐日循环，不经 panelops、不经 ashare.ml）。"""

import math

import numpy as np
import pandas as pd
import panelbt
import pytest

pytest.importorskip("lightgbm")  # 标签复用 ashare.ml（可选依赖组 ml）

from ashare.factor import Evaluator  # noqa: E402
from ashare.factor.evaluate import (  # noqa: E402
    FactorEvaluator,
    equal_weight_benchmark,
    evaluate_factor,
    ic_summary,
)
from ashare.factor.mining.synthetic import market, planted_expression  # noqa: E402


@pytest.fixture(scope="module")
def panel():
    return market(t=260, n=40, seed=7, strength=0.003, blocked=0.05)


def _universe(p):
    return (
        np.asarray(p["is_listed"])
        & ~np.asarray(p["is_st"])
        & (np.asarray(p["listing_age"]) >= 120)
        & np.asarray(p["has_quote"])
    )


def _ref_label(p, h, lag=1):
    """t+lag 开盘买入、t+lag+h 开盘卖出；买入日不可买置缺失（逐格循环，独立写法）。"""
    o = np.asarray(p["open"]) * np.asarray(p["adj_factor"])
    can = np.asarray(p["can_buy_open"])
    t, n = o.shape
    out = np.full((t, n), np.nan)
    for i in range(t - lag - h):
        for j in range(n):
            if can[i + lag, j]:
                out[i, j] = o[i + lag + h, j] / o[i + lag, j] - 1
    return out


def _ref_rank_ic(x, y):
    out = []
    for a, b in zip(x, y, strict=True):
        s = pd.DataFrame({"a": a, "b": b}).dropna()
        if len(s) < 2 or s["a"].nunique() < 2 or s["b"].nunique() < 2:
            out.append(np.nan)
        else:
            out.append(s["a"].rank().corr(s["b"].rank()))
    return np.array(out)


def _ref_groups(x, y, q):
    out = np.full((len(x), q), np.nan)
    for i, (a, b) in enumerate(zip(x, y, strict=True)):
        cols = [j for j in range(len(a)) if np.isfinite(a[j])]
        cols.sort(key=lambda j: (a[j], j))
        groups = [[] for _ in range(q)]
        for pos, j in enumerate(cols):
            groups[pos * q // len(cols)].append(j)
        for g, members in enumerate(groups):
            vals = [b[j] for j in members if np.isfinite(b[j])]
            if vals:
                out[i, g] = sum(vals) / len(vals)
    return out


def test_ic_and_groups_match_reference(panel):
    values = Evaluator().evaluate(planted_expression(), panel)
    fe = FactorEvaluator(panel, horizons=(1, 5), rebalance=5, groups=5)
    uni = _universe(panel)
    scores = np.where(uni & np.isfinite(values), values, np.nan)
    _, partial = fe.cheap(values)
    for h in (1, 5):
        label = _ref_label(panel, h)
        label = np.where(uni, label, np.nan)
        ref = _ref_rank_ic(scores, label)
        np.testing.assert_allclose(partial["ic"][h], ref, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        partial["groups"], _ref_groups(scores, np.where(uni, _ref_label(panel, 5), np.nan), 5),
        rtol=1e-12, atol=1e-14,
    )  # fmt: skip
    rep = fe.finish("x", partial)
    seg = rep.segments["all"]
    # 段内只用标签完整落在段内的信号日：t + 1 + h < T
    ref_ic = _ref_rank_ic(scores, np.where(uni, _ref_label(panel, 5), np.nan))[
        : 260 - 6
    ]
    assert seg["ic"][5]["mean"] == pytest.approx(np.nanmean(ref_ic), rel=1e-12)
    assert seg["ic"][5]["days"] == int(np.isfinite(ref_ic).sum())


def test_label_alignment_off_by_one_would_differ(panel):
    # 断言用例确实区分「T+1 开盘买」与「T 日开盘买」：两种标签的 IC 明显不同
    values = Evaluator().evaluate(planted_expression(), panel)
    uni = _universe(panel)
    scores = np.where(uni & np.isfinite(values), values, np.nan)
    good = np.nanmean(_ref_rank_ic(scores, _ref_label(panel, 1, lag=1)))
    shifted = np.nanmean(_ref_rank_ic(scores, _ref_label(panel, 1, lag=2)))
    fe = FactorEvaluator(panel, horizons=(1,))
    got = fe.evaluate(values, long=False).segments["all"]["ic"][1]["mean"]
    assert got == pytest.approx(
        np.nanmean(
            _ref_rank_ic(scores, np.where(uni, _ref_label(panel, 1), np.nan))[: 260 - 2]
        )
    )
    assert good > 0.1 and abs(good - shifted) > 0.03


def test_ic_summary_hansen_hodrick():
    rng = np.random.default_rng(0)
    ic = rng.standard_normal(200) * 0.1 + 0.02
    s = ic_summary(ic, horizon=5)
    e = ic - ic.mean()
    var = e @ e / 200
    for j in range(1, 5):
        var += 2 * (e[j:] @ e[:-j]) / 200  # 均匀核：滞后 1..h−1 权重都是 1
    assert s["t"] == pytest.approx(ic.mean() / math.sqrt(var / 200), rel=1e-12)
    assert s["icir"] == pytest.approx(ic.mean() / ic.std(ddof=1))
    # p 值：自由度 n/h − 1 = 39 的 Student t 上尾（与正态近似不同）
    from scipy import stats as sps

    assert s["p"] == pytest.approx(sps.t.sf(s["t"], 39), rel=1e-10)
    assert s["p"] > 0.5 * math.erfc(s["t"] / math.sqrt(2))


def test_benchmark_reference(panel):
    uni = _universe(panel)
    b = equal_weight_benchmark(panel, uni)
    c = np.asarray(panel["close"]) * np.asarray(panel["adj_factor"])
    for d in (1, 50, 259):
        vals = [c[d, j] / c[d - 1, j] - 1 for j in range(40) if uni[d - 1, j]]
        assert b[d] == pytest.approx(sum(vals) / len(vals), rel=1e-12)
    assert math.isnan(b[0])


def _ref_targets(scores, frac, start, every, t):
    """信号日 s 选前 ceil(有效数·frac) 只等权，成交行 s + 1。"""
    dense = np.full(scores.shape, np.nan)
    for s in range(start, t, every):
        if s + 1 >= t:
            break
        cols = [j for j in range(scores.shape[1]) if np.isfinite(scores[s, j])]
        cols.sort(key=lambda j: (-scores[s, j], j))
        k = math.ceil(len(cols) * frac)
        dense[s + 1] = 0.0
        for j in cols[:k]:
            dense[s + 1, j] = 1.0 / k
    return panelbt.Targets.from_dense(dense)


def test_long_side_matches_direct_backtest(panel):
    values = Evaluator().evaluate(planted_expression(), panel)
    fe = FactorEvaluator(panel, horizons=(1,), rebalance=5, top_fraction=0.2)
    rep = fe.evaluate(values)
    scores = np.where(_universe(panel) & np.isfinite(values), values, np.nan)
    start = int(np.flatnonzero(np.isfinite(scores).any(axis=1))[0])
    ref = panelbt.run(
        fe._market, _ref_targets(scores, 0.2, start, 5, 260), fe.costs, fe.config
    )
    np.testing.assert_array_equal(rep.series["long"]["returns"], ref.returns)
    r = ref.returns[:]
    bench = fe.benchmark
    ok = np.isfinite(r) & np.isfinite(bench)
    ex = (r - bench)[ok]
    lo = rep.segments["all"]["long"]
    assert lo["excess_return"] == pytest.approx(ex.mean() * 252, rel=1e-12)
    assert lo["information_ratio"] == pytest.approx(
        ex.mean() / ex.std(ddof=1) * math.sqrt(252), rel=1e-12
    )
    assert lo["annual_return"] == pytest.approx(
        np.prod(1 + r[ok]) ** (252 / ok.sum()) - 1, rel=1e-12
    )
    w = np.r_[1.0, np.cumprod(1 + r[ok])]
    assert lo["max_drawdown"] == pytest.approx((w / np.maximum.accumulate(w) - 1).min())
    # 成本：佣金 + 税 + 其他费用 + 滑点，按前一日权益折成比例，年化
    spent = ref.commission + ref.tax + ref.other_fee + ref.slippage_cost
    prev = np.r_[1.0, ref.nav[:-1]] * fe.config.initial_cash
    assert lo["cost"] == pytest.approx(
        (spent / prev)[ok].sum() / ok.sum() * 252, rel=1e-12
    )
    assert lo["cost"] > 0.01
    assert lo["turnover"] == pytest.approx(
        ref.turnover[ok].sum() / ok.sum() * 252, rel=1e-12
    )


def test_costs_are_deducted(panel):
    values = Evaluator().evaluate(planted_expression(), panel)
    paid = evaluate_factor(values, panel, horizons=(1,))
    free = evaluate_factor(values, panel, horizons=(1,), costs=panelbt.Costs(
        commission_rate=0.0, commission_min=None, fee_rate_buy=0.0, fee_rate_sell=0.0, slippage=0.0
    ))  # fmt: skip
    a, b = paid.segments["all"]["long"], free.segments["all"]["long"]
    assert b["cost"] < a["cost"] * 0.6  # 印花税仍在
    assert a["excess_return"] < b["excess_return"] - 0.5 * (a["cost"] - b["cost"])


def test_batch_equals_single_and_segments(panel):
    ev = Evaluator()
    e1 = planted_expression()
    rng = np.random.default_rng(1)
    noise = rng.standard_normal(panel.shape)
    fe = FactorEvaluator(
        panel, segments={"a": (0, 150), "b": (150, 260)}, horizons=(1, 5)
    )
    both = fe.evaluate_many({"p": ev.evaluate(e1, panel), "n": noise})
    one = fe.evaluate(ev.evaluate(e1, panel), "p")
    assert both["p"].to_dict() == one.to_dict()
    d = both["n"].to_dict()
    assert set(d["segments"]) == {"a", "b"}
    assert d["segments"]["a"]["end"] < d["segments"]["b"]["start"]
    # 段内 IC 只用标签完整落在段内的信号日：t + 1 + 5 < 150
    assert d["segments"]["a"]["ic"]["5"]["days"] == 150 - 6
    assert d["segments"]["b"]["ic"]["5"]["days"] == 260 - 6 - 150
    # 分年度、市值段、自相关、与已有因子的相关都在
    fe2 = FactorEvaluator(
        panel, horizons=(1,), existing={"same": ev.evaluate(e1, panel)}
    )
    r2 = fe2.evaluate(ev.evaluate(e1, panel), long=False)
    assert r2.correlation["same"] == pytest.approx(1.0)
    assert set(r2.segments["all"]["size"]) == {"小", "中", "大"}
    assert set(r2.segments["all"]["years"]) == {"2012", "2013"}
    assert r2.autocorr[1] > 0.5
    text = r2.text()
    assert "Rank IC" in text and max(len(x) for x in text.splitlines()) <= 120


def test_signal_beats_noise(panel):
    ev = Evaluator()
    rng = np.random.default_rng(2)
    fe = FactorEvaluator(panel)
    rep = fe.evaluate_many({"p": ev.evaluate(planted_expression(), panel),
                            "n": rng.standard_normal(panel.shape)})  # fmt: skip
    assert rep["p"].metric("all", "long.information_ratio") > 1
    assert rep["p"].metric("all", "ic.5.mean") > 0.05
    assert rep["n"].metric("all", "long.excess_return") < rep["p"].metric(
        "all", "long.excess_return"
    )
    assert rep["p"].segments["all"]["groups"]["monotonicity"] > 0.8


def test_ordered_rank_ic_bitwise(panel):
    import panelops

    fe = FactorEvaluator(panel, horizons=(1, 5))
    rng = np.random.default_rng(4)
    x = rng.standard_normal(panel.shape)
    x[rng.random(panel.shape) < 0.1] = np.nan
    for h in (1, 5):
        np.testing.assert_array_equal(fe._ic[h](x), panelops.rank_ic(x, fe.returns[h]))
    assert fe._size_ic
    for k, m in fe._size_masks.items():
        got = fe._size_ic[k](x)
        np.testing.assert_array_equal(got, panelops.rank_ic(x, fe.returns[5], mask=m))
    # 粗筛的指标与完整评估的同名字段一致
    scores, ic = fe.screen(x)
    full = fe.evaluate(x, long=False).to_dict()["segments"]["all"]["ic"]["5"]
    assert fe.screen_metrics(ic)["all"]["ic"] == full
