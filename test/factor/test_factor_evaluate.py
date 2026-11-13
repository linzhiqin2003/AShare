"""单因子评估对直白参考实现（pandas / NumPy 逐日循环，不经 panelops、不经 ashare.ml）。"""

import math

import numpy as np
import pandas as pd
import panelbt
import pytest

from ashare.factor import Evaluator  # noqa: E402
from ashare.factor.evaluate import (  # noqa: E402
    FactorEvaluator,
    equal_weight_benchmark,
    evaluate_factor,
    ic_summary,
)
from ashare.factor.mining.synthetic import (  # noqa: E402
    Frictions,
    market,
    planted_expression,
)

ZERO = panelbt.Costs(
    commission_rate=0, commission_min=None, fee_rate_buy=0, fee_rate_sell=0, slippage=0
)


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


def test_ic_summary_overlap_bound():
    """t = 均值 / √(s²·h/n)，自由度 ⌊n/h⌋ − 1；不重叠的块不足 10 个时不给 p 值。"""
    rng = np.random.default_rng(0)
    ic = rng.standard_normal(200) * 0.1 + 0.02
    s = ic_summary(ic, horizon=5)
    assert s["t"] == pytest.approx(
        ic.mean() / math.sqrt(ic.var(ddof=1) * 5 / 200), rel=1e-12
    )
    assert s["df"] == 39 and s["warning"] is None
    assert s["icir"] == pytest.approx(ic.mean() / ic.std(ddof=1))
    from scipy import stats as sps

    assert s["p"] == pytest.approx(sps.t.sf(s["t"], 39), rel=1e-10)
    # h = 1 就是普通单样本 t 检验
    one = ic_summary(ic, horizon=1)
    ref = sps.ttest_1samp(ic, 0.0, alternative="greater")
    assert one["t"] == pytest.approx(ref.statistic, rel=1e-12)
    assert one["p"] == pytest.approx(ref.pvalue, rel=1e-9)
    # 200 天、h = 20：只有 10 个不重叠的块，刚好够；h = 25 只有 8 个，不给 p 值
    assert math.isfinite(ic_summary(ic, horizon=20)["p"])
    short = ic_summary(ic, horizon=25)
    assert math.isnan(short["p"]) and "不给 p 值" in short["warning"]
    assert math.isfinite(short["t"])


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
    first = int(np.flatnonzero(np.isfinite(scores).any(axis=1))[0])
    start = -(-first // 5) * 5  # 对齐到调仓日表 0, 5, 10, …
    ref = panelbt.run(
        fe._market, _ref_targets(scores, 0.2, start, 5, 260), fe.costs, fe.config
    )
    np.testing.assert_array_equal(rep.series["long"]["returns"], ref.returns)
    # 基准：同一份行情、同样的调仓日，持有信号日股票池全体等权
    uni = np.where(_universe(panel), 1.0, np.nan)
    bref = panelbt.run(
        fe._market, _ref_targets(uni, 1.0, 0, 5, 260), fe.costs, fe.config
    )
    np.testing.assert_array_equal(fe.benchmark[1:], bref.returns[1:])
    assert np.isnan(fe.benchmark[0])
    r = ref.returns[:].copy()
    r[: start + 1] = np.nan  # 指标从第一次成交那天算起
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


def test_random_scores_have_zero_excess_against_default_benchmark():
    """零成本、每期重抽的随机分数：相对默认基准（同一套成交机制持有股票池全体）的超额
    信息比率平均接近 0；相对旧基准（收盘到收盘）明显为负——旧基准看不到停牌期间的跌幅、
    退市清零和戴帽期间的漂移。"""
    t, n = 750, 200
    p = market(t=t, n=n, seed=1, frictions=Frictions.typical())
    out = {}
    for mode in ("universe", "close"):
        fe = FactorEvaluator(p, horizons=(5,), benchmark=mode, costs=ZERO,
                             top_fraction=0.5, size_buckets=1)  # fmt: skip
        irs = [
            fe.evaluate(np.random.default_rng(k).standard_normal((t, n))).metric(
                "all", "long.information_ratio"
            )
            for k in range(60)
        ]
        out[mode] = float(np.mean(irs))
    assert abs(out["universe"]) < 0.25  # 60 次的标准误约 0.09
    assert out["close"] < -0.35


def test_benchmark_uses_signal_day_universe():
    """一只股票在信号日 r 才进股票池：基准 r+1 日开盘才买到它，拿不到 r 日开盘到 r+1 日
    开盘之间的翻倍；按成交日的股票池（前视一天）建基准就会拿到。"""
    p = market(t=80, n=6, seed=2)
    r = 40
    scale = np.ones(p.shape)
    scale[r + 1 :, 0] = 2.0  # 开盘价从 r+1 起翻倍
    cscale = np.ones(p.shape)
    cscale[r:, 0] = 2.0  # 收盘价从 r 起翻倍（r 日日内翻倍）
    arrays = {k: np.array(p[k], copy=True) for k in p.fields()}
    arrays["open"] = arrays["open"] * scale
    for k in ("close", "high", "vwap"):
        arrays[k] = arrays[k] * cscale
    arrays["low"] = arrays["low"] * scale
    arrays["previous_close"] = np.vstack(
        [arrays["previous_close"][:1], arrays["close"][:-1]]
    )
    q = p.replace(**arrays)
    uni = np.ones(p.shape, bool)
    uni[:r, 0] = False
    fe = FactorEvaluator(q, horizons=(1,), rebalance=1, universe=uni, costs=ZERO,
                         size_buckets=1)  # fmt: skip
    b = fe.benchmark
    assert abs(b[r]) < 0.1  # r 日：这只股票还没买
    assert abs(b[r + 1]) < 0.1  # r+1 日开盘才买进，翻倍已经过去


def test_exec_price_must_match_market(panel):
    from ashare.backtest.market import market_inputs

    wrong = panelbt.Market(**market_inputs(panel, "close"), validate=True)
    with pytest.raises(ValueError, match="成交价"):
        FactorEvaluator(panel, horizons=(1,), market=wrong)
    fe = FactorEvaluator(panel, horizons=(1,), market=wrong, exec="close")
    assert fe.exec == "close"
    from ashare.labels import Label, compute

    v = compute(panel, Label(horizon=1, exec="close"), fe.universe)
    np.testing.assert_array_equal(fe.returns[1], np.where(v.buyable, v.raw, np.nan))
    assert not np.array_equal(
        fe.returns[1], FactorEvaluator(panel, horizons=(1,)).returns[1], equal_nan=True
    )


def test_default_universe_same_as_ml(panel):
    pytest.importorskip("lightgbm")
    from ashare.factor.evaluate import default_universe
    from ashare.ml.walk import default_universe as ml_universe

    p = market(t=300, n=30, seed=5, frictions=Frictions.typical(st=0.3, suspend=0.01))
    np.testing.assert_array_equal(default_universe(p), ml_universe(p))
    np.testing.assert_array_equal(default_universe(panel), ml_universe(panel))


def test_long_summary_var_ratio():
    from ashare.factor import stats
    from ashare.factor.evaluate import long_summary

    rng = np.random.default_rng(1)
    e = rng.standard_normal(2000) * 0.01
    x = np.empty_like(e)
    x[0] = e[0]
    for i in range(1, len(e)):
        x[i] = 0.5 * x[i - 1] + e[i]
    lo = long_summary(x + 0.001, np.zeros_like(x))
    assert lo["var_ratio"] == pytest.approx(stats.variance_ratio(x, 20), rel=1e-12)
    assert long_summary(x, np.zeros_like(x), lags=5)["var_ratio"] == pytest.approx(
        stats.variance_ratio(x, 5), rel=1e-12
    )
    assert lo["var_ratio"] > 2


def test_rank_ic_fallback_warns_once(panel, monkeypatch):
    import panelops

    from ashare.factor import evaluate as ev

    monkeypatch.setitem(panelops.api.BACKENDS, "rank_ic", "numpy")
    monkeypatch.setattr(ev._RankIC, "_warned", False)
    fe = FactorEvaluator(panel, horizons=(1,), size_buckets=1)
    x = np.random.default_rng(0).standard_normal(panel.shape)
    with pytest.warns(RuntimeWarning, match="慢"):
        a = fe._ic[1](x)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        b = fe._ic[1](x)
    np.testing.assert_array_equal(a, b)


def test_default_capital_makes_minimum_commission_irrelevant():
    """默认账户规模下，持有全体的基准的成本与「没有最低佣金」时几乎相同；1 亿元时
    最低佣金让它多扣四成以上（小账户下基准成本偏高、超额偏正）。"""
    from panelbt.markets import ASHARE

    from ashare.backtest.market import iso_dates
    from ashare.factor.evaluate import CAPITAL

    assert CAPITAL >= 1e10
    p = market(t=300, n=1500, seed=4, frictions=Frictions.typical())
    full = ASHARE.costs(iso_dates(p.dates))
    no_min = ASHARE.costs(iso_dates(p.dates), commission_min=None)

    def bench_cost(costs, **kw):
        fe = FactorEvaluator(p, horizons=(1,), costs=costs, size_buckets=1,
                             top_fraction=1.0, **kw)  # fmt: skip
        return fe.evaluate(np.ones(p.shape)).metric("all", "long.cost")

    default = bench_cost(full)
    assert default == pytest.approx(bench_cost(no_min), rel=0.01)
    small = bench_cost(full, capital=1e8)
    assert small > 1.3 * bench_cost(no_min, capital=1e8)
    assert (
        FactorEvaluator(p, horizons=(1,), size_buckets=1).config.initial_cash == CAPITAL
    )


def test_random_scores_not_positive_with_default_costs():
    """默认费用、默认账户规模、带摩擦：随机分数的超额信息比率不显著为正。"""
    t, n = 750, 200
    p = market(t=t, n=n, seed=1, frictions=Frictions.typical())
    fe = FactorEvaluator(p, horizons=(5,), top_fraction=0.5, size_buckets=1)
    irs = [
        fe.evaluate(np.random.default_rng(k).standard_normal((t, n))).metric(
            "all", "long.information_ratio"
        )
        for k in range(60)
    ]
    assert np.mean(irs) < 0.25  # 60 次的标准误约 0.09


def test_rebalance_days_follow_fixed_grid(panel):
    """因子第 7 行才有值、每 5 天调仓：信号日是固定日表上的 10、15、20…（与基准同一天），
    不是从第 7 行起每 5 行。"""
    t = panel.shape[0]
    x = np.random.default_rng(3).standard_normal(panel.shape)
    x[:7] = np.nan
    fe = FactorEvaluator(panel, horizons=(1,), rebalance=5, size_buckets=1)
    days = fe.rebalance_days(fe._scores(x))
    np.testing.assert_array_equal(days, np.arange(10, t, 5))
    np.testing.assert_array_equal(
        fe.targets(fe._scores(x)).days, days[days + 1 < t] + 1
    )
    rep = fe.evaluate(x)
    assert rep.metric("all", "long.days") == t - 11  # 从第一次成交（第 11 行）算起


def test_user_config_wins_over_capital(panel):
    """传 BacktestMarket 和 config= 时用用户的账户，不被默认规模覆盖。"""
    from panelbt.markets import Board

    from ashare.backtest.market import BacktestMarket, market_inputs
    from ashare.factor.evaluate import CAPITAL

    board = np.full(len(panel.symbols), int(Board.MAIN), dtype=np.int64)
    bm = BacktestMarket(panelbt.Market(**market_inputs(panel)), np.asarray(panel.dates),
                        np.asarray(panel.symbols), board, panel)  # fmt: skip
    fe = FactorEvaluator(panel, horizons=(1,), market=bm, size_buckets=1,
                         config=bm.config(initial_cash=3e7))  # fmt: skip
    assert fe.config.initial_cash == 3e7
    assert (
        FactorEvaluator(
            panel, horizons=(1,), market=bm, size_buckets=1
        ).config.initial_cash
        == CAPITAL
    )
    assert FactorEvaluator(panel, horizons=(1,), market=bm, size_buckets=1,
                           capital=5e8).config.initial_cash == 5e8  # fmt: skip


def test_old_benchmark_uses_previous_day_universe(panel):
    """旧基准（benchmark="close"）第 d 日用 d−1 日在池内的股票：股票池逐日变化时与参考一致。"""
    rng = np.random.default_rng(11)
    uni = rng.random(panel.shape) < 0.6
    b = equal_weight_benchmark(panel, uni)
    c = np.asarray(panel["close"]) * np.asarray(panel["adj_factor"])
    for d in range(1, panel.shape[0]):
        vals = [
            c[d, j] / c[d - 1, j] - 1 for j in range(panel.shape[1]) if uni[d - 1, j]
        ]
        assert b[d] == pytest.approx(sum(vals) / len(vals), rel=1e-12)
    fe = FactorEvaluator(
        panel, horizons=(1,), universe=uni, benchmark="close", size_buckets=1
    )
    np.testing.assert_array_equal(fe.benchmark, b)
