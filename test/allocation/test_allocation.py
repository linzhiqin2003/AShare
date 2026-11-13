"""资产配置：对拍、性质、防前视、病态输入。"""

import json
from pathlib import Path

import numpy as np
import pytest

import ashare
import ashare.allocation as aa
from ashare.allocation import reference as ref
from ashare.allocation.riskparity import risk_contributions, solve_risk_parity
from ashare.backtest.timing import execution_days

assert (
    Path(ashare.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[2])
)


def random_cov(rng, n, rank=None):
    rank = rank or n
    a = rng.normal(size=(n, rank)) * rng.uniform(0.005, 0.03, size=(n, 1))
    return a @ a.T + (0 if rank < n else 1e-6) * np.eye(n)


def synthetic(seed=0, t=900, n=5, vols=None):
    rng = np.random.default_rng(seed)
    vols = np.asarray(vols if vols is not None else rng.uniform(0.004, 0.02, n))
    corr = np.full((n, n), 0.3) + 0.7 * np.eye(n)
    chol = np.linalg.cholesky(corr)
    r = rng.normal(size=(t, n)) @ chol.T * vols + 0.0002
    r[0] = np.nan
    return r


# ---------- 风险平价：正式实现对参考实现 ----------


@pytest.mark.parametrize("seed", range(12))
def test_risk_parity_matches_reference(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 12))
    sigma = random_cov(rng, n)
    budget = None if seed % 2 else rng.uniform(0.2, 1.0, n)
    w, info = solve_risk_parity(sigma, budget)
    expected = ref.risk_parity_reference(sigma, budget)
    assert info["converged"] and info["residual"] < 1e-10
    assert np.max(np.abs(w - expected)) < 1e-5
    b = np.full(n, 1 / n) if budget is None else budget / budget.sum()
    assert np.max(np.abs(risk_contributions(w, sigma) - b)) < 1e-10
    assert (w > 0).all() and abs(w.sum() - 1) < 1e-14


def test_risk_parity_singular_covariance():
    rng = np.random.default_rng(3)
    # 秩 2、载荷全正：奇异，但不存在方差为 0 的长仓组合，解存在、不需要 ridge。
    loadings = rng.uniform(0.002, 0.02, size=(6, 2))
    sigma = loadings @ loadings.T
    assert np.linalg.matrix_rank(sigma) == 2
    w, info = solve_risk_parity(sigma)
    assert info["converged"] and info["residual"] < 1e-9 and info["ridge"] == 0
    # 载荷有正有负：存在方差为 0 的长仓组合，等风险贡献无解，靠 ridge 给出有定义的结果。
    a = np.array([[1.0, 0.0], [-1.0, 1.0], [0.0, -1.0]]) * 0.01  # 等权组合方差恰为 0
    signed = a @ a.T
    w, info = solve_risk_parity(signed)
    assert info["ridge"] > 0 and np.isfinite(w).all()
    # 原协方差上无解：不能报 converged，单列「靠 ridge 解到」。
    assert not info["converged"] and info["solved_with_ridge"]
    assert info["status"] == "ridge"
    assert info["residual"] < 1e-8 and abs(w.sum() - 1) < 1e-12
    # 两只完全相同的资产平分。
    x = random_cov(rng, 3)
    dup = np.block([[x, x[:, :1]], [x[:1, :], x[:1, :1]]])
    w, info = solve_risk_parity(dup)
    assert info["converged"] and abs(w[0] - w[3]) < 1e-10


def test_risk_parity_zero_budget_and_single():
    sigma = random_cov(np.random.default_rng(1), 4)
    w, _ = solve_risk_parity(sigma, [0.5, 0.5, 0, 0])
    assert w[2] == 0 and w[3] == 0 and abs(w.sum() - 1) < 1e-14
    w, info = solve_risk_parity(np.array([[0.0004]]))
    assert w.tolist() == [1.0] and info["iterations"] == 0
    with pytest.raises(ValueError, match="对角线必须为正"):
        solve_risk_parity(np.diag([0.0004, 0.0]))


def test_min_and_mean_variance_match_reference():
    rng = np.random.default_rng(7)
    r = synthetic(7, t=400, n=6)
    for method, check in [
        (aa.MinVariance(), lambda s, w: ref.min_variance_reference(s)),
        (
            aa.MinVariance(max_weight=0.25),
            lambda s, w: ref.min_variance_reference(s, 0.25),
        ),
        (aa.MeanVariance(risk_aversion=20.0), None),
    ]:
        res = aa.allocate(r, method, rebalance=[300], lookback=252)
        cov = aa.Covariance()
        sigma, _ = cov.estimate(r[300 - 252 : 300])
        w = res.weights[0]
        if check is None:
            mu = np.nanmean(r[300 - 252 : 300], axis=0)
            expected = ref.mean_variance_reference(sigma, mu, 20.0)
        else:
            expected = check(sigma, w)
        assert np.max(np.abs(w - expected)) < 2e-4, method
        assert abs(w.sum() - 1) < 1e-9 and (w >= 0).all()
    del rng


# ---------- 性质 ----------

METHODS = [
    aa.EqualWeight(),
    aa.InverseVolatility(),
    aa.RiskParity(),
    aa.RiskParity(budget=[0.4, 0.3, 0.1, 0.1, 0.1]),
    aa.MinVariance(),
    aa.MeanVariance(),
    aa.FixedWeights([0.4, 0.2, 0.2, 0.1, 0.1]),
    aa.TargetVolatility(aa.RiskParity(), target=0.05),
    aa.TrendFilter(aa.EqualWeight(), window=60),
    aa.TrendFilter(aa.TargetVolatility(aa.InverseVolatility(), 0.06), window=40),
]


@pytest.mark.parametrize("method", METHODS, ids=lambda m: m.name)
def test_weight_properties(method):
    r = synthetic(1)
    res = aa.allocate(r, method, rebalance="monthly")
    assert len(res.rows) > 20
    w = res.weights
    assert (w >= 0).all() and np.isfinite(w).all()
    sums = w.sum(axis=1)
    if isinstance(method, (aa.TargetVolatility, aa.TrendFilter)):
        assert (sums <= 1 + 1e-12).all()
    else:
        np.testing.assert_allclose(sums, 1.0, atol=1e-9)
    for d in res.diagnostics:
        if "risk_parity" in d:
            assert d["risk_parity"]["residual"] < 1e-9
    json.dumps(res.to_dict())  # 只有基本类型


def test_branches_reached():
    """叠加层确实改动了权重（防止性质测试在没走到的分支上空过）。"""
    r = synthetic(1)
    tv = aa.allocate(r, aa.TargetVolatility(aa.RiskParity(), target=0.05))
    assert any(d["target_volatility"]["scale"] < 0.99 for d in tv.diagnostics)
    tf = aa.allocate(r, aa.TrendFilter(aa.EqualWeight(), window=60))
    assert any(d["trend_off"] for d in tf.diagnostics)
    assert any(not d["trend_off"] for d in tf.diagnostics)


def test_target_volatility_realized():
    r = synthetic(5, t=3000, n=4, vols=[0.012, 0.008, 0.015, 0.01])
    res = aa.allocate(
        r,
        aa.TargetVolatility(aa.RiskParity(), target=0.06, max_leverage=1.0),
        rebalance=21,
    )
    bt = aa.run(res, returns=r)
    nav = bt.result.nav
    daily = nav[1:] / nav[:-1] - 1
    realized = np.std(daily[400:], ddof=1) * np.sqrt(252)
    assert 0.06 * 0.85 < realized < 0.06 * 1.15, realized
    # 杠杆上限：放大后的权重和不超过 max_leverage。
    up = aa.allocate(
        r,
        aa.TargetVolatility(aa.RiskParity(), target=0.5, max_leverage=1.5),
        rebalance=63,
    )
    sums = up.weights.sum(axis=1)
    assert sums.max() <= 1.5 + 1e-12 and sums.max() > 1.4
    with pytest.raises(ValueError, match="杠杆"):
        aa.run(up, returns=r)


# ---------- 防前视 ----------


@pytest.mark.parametrize("method", METHODS, ids=lambda m: m.name)
def test_no_lookahead(method):
    """改动 t 日及以后的收益，第 t 个调仓日及以前的权重不变。"""
    r = synthetic(2)
    base = aa.allocate(r, method, rebalance="monthly", band=0.02)
    rng = np.random.default_rng(9)
    for k in (5, 15, len(base.rows) - 3):
        t = int(base.rows[k])
        changed = r.copy()
        changed[t:] = rng.normal(0, 0.05, changed[t:].shape)
        after = aa.allocate(changed, method, rebalance="monthly", band=0.02)
        upto = base.rows <= t
        np.testing.assert_array_equal(after.weights[upto], base.weights[upto])
        np.testing.assert_array_equal(after.traded[upto], base.traded[upto])
        # 之后的调仓日确实变了（改动有效）：权重或是否下单（与数据无关的等权、固定比例只看后者）。
        assert not (
            np.array_equal(after.weights[~upto], base.weights[~upto])
            and np.array_equal(after.traded[~upto], base.traded[~upto])
        )


def test_schedule_matches_timing():
    dates = [
        str(d) for d in np.busday_offset("2021-01-04", np.arange(300), roll="forward")
    ]
    rows = aa.schedule(dates, "monthly")
    assert rows.tolist() == ref.rebalance_rows_reference(dates)
    # 成交行 = 信号日（上月最后一个交易日）+ 1，与 ashare.backtest.timing 一致。
    np.testing.assert_array_equal(execution_days(rows - 1, len(dates)), rows)
    assert aa.schedule(dates, 10).tolist() == list(range(10, 300, 10))
    q = aa.schedule(dates, "quarterly")
    assert all(dates[i][5:7] in ("01", "04", "07", "10") for i in q)


# ---------- 病态输入 ----------


def test_zero_volatility_asset():
    r = synthetic(3, n=4)
    r[1:, 2] = 0.0  # 价格从不变动
    res = aa.allocate(r, aa.RiskParity(), rebalance=63)
    assert (res.weights[:, 2] == 0).all()
    assert all(d["zero_volatility"] == [2] for d in res.diagnostics)
    ivol = aa.allocate(r, aa.InverseVolatility(), rebalance=63)
    assert (ivol.weights[:, 2] == 0).all()
    allzero = np.zeros((300, 3))
    allzero[0] = np.nan
    eq = aa.allocate(allzero, aa.RiskParity(), rebalance=100)
    np.testing.assert_allclose(eq.weights, 1 / 3)
    assert "fallback" in eq.diagnostics[0]


def test_single_asset_and_late_listing():
    r = synthetic(4, n=3)
    one = aa.allocate(r[:, :1], aa.RiskParity(), rebalance=63)
    assert (one.weights == 1.0).all()
    late = r.copy()
    late[:500, 1] = np.nan  # 第二个资产第 500 行才上市
    res = aa.allocate(late, aa.RiskParity(), rebalance=21, min_history=60)
    before = res.rows < 560
    assert (res.weights[before, 1] == 0).all()
    assert (res.weights[~before, 1] > 0).all()
    assert all(
        d["available"] == 2 for d, b in zip(res.diagnostics, before, strict=True) if b
    )


def test_band_controls_trading():
    r = synthetic(6)
    never = aa.allocate(r, aa.FixedWeights([0.5, 0.5, 0, 0, 0]), rebalance=21, band=1.0)
    assert never.traded.sum() == 1  # 只有第一次
    always = aa.allocate(
        r, aa.FixedWeights([0.5, 0.5, 0, 0, 0]), rebalance=21, band=0.0
    )
    assert always.traded.all()
    mid = aa.allocate(r, aa.FixedWeights([0.5, 0.5, 0, 0, 0]), rebalance=21, band=0.02)
    assert 1 < mid.traded.sum() < len(mid.rows)
    dense = mid.dense()
    assert np.isnan(dense[mid.rows[~mid.traded]]).all()


def test_fixed_weights_inputs():
    r = synthetic(6, n=3)
    res = aa.allocate(
        r, aa.FixedWeights({"a": 0.6, "b": 0.4}), symbols=["a", "b", "c"], rebalance=63
    )
    np.testing.assert_allclose(res.weights[-1], [0.6, 0.4, 0.0])
    with pytest.raises(ValueError, match="未知资产"):
        aa.allocate(r, aa.FixedWeights({"x": 1.0}), symbols=["a", "b", "c"])
    late = r.copy()
    late[:400, 1] = np.nan
    keep = aa.allocate(late, aa.FixedWeights([0.6, 0.4, 0]), rebalance=[300])
    spread = aa.allocate(
        late, aa.FixedWeights([0.6, 0.4, 0], redistribute=True), rebalance=[300]
    )
    np.testing.assert_allclose(keep.weights[0], [0.6, 0, 0])
    np.testing.assert_allclose(spread.weights[0], [1.0, 0, 0])


def test_bad_inputs_are_explained():
    r = synthetic(0, n=2)
    with pytest.raises(ValueError, match="大于 -1"):
        aa.allocate(np.full((10, 2), -1.0), aa.EqualWeight())
    with pytest.raises(TypeError):
        aa.allocate(r, "risk_parity")
    with pytest.raises(ValueError, match="min_history"):
        aa.allocate(r, aa.EqualWeight(), min_history=1000)


def test_pandas_and_tearsheet_roundtrip():
    pd = pytest.importorskip("pandas")
    r = synthetic(8, n=3, t=700)
    frame = pd.DataFrame(
        r,
        columns=["stock", "bond", "gold"],
        index=pd.bdate_range("2019-01-02", periods=700),
    )
    res = aa.allocate(frame, aa.RiskParity())
    assert res.symbols == ["stock", "bond", "gold"] and res.dates[0].startswith("2019-")
    ts = aa.tearsheet(res, returns=frame)
    text = ts.text()
    assert "年化收益" in text
    assert res.frame().shape == res.weights.shape


def test_covariance_matches_hand_written_ew():
    """不收缩时等于直白写的指数加权协方差（逐列有效权重归一，与 ashare.risk 约定一致）。"""
    r = synthetic(11, t=300, n=3)[1:]
    half = 20.0
    sigma, _ = aa.Covariance(half_life=half, shrinkage=0).estimate(r)
    w = np.array([0.5 ** ((len(r) - 1 - i) / half) for i in range(len(r))])
    w = w / w.sum()
    mean = (r * w[:, None]).sum(axis=0)
    z = r - mean
    expected = (z * w[:, None]).T @ z
    np.testing.assert_allclose(sigma, expected, rtol=1e-10)
    flat, _ = aa.Covariance(half_life=None, shrinkage=0).estimate(r)
    np.testing.assert_allclose(flat, np.cov(r.T, ddof=0), rtol=1e-10)


def test_inverse_volatility_proportions():
    r = synthetic(12, t=400, n=3, vols=[0.01, 0.02, 0.04])
    res = aa.allocate(
        r,
        aa.InverseVolatility(),
        rebalance=[300],
        covariance=aa.Covariance(half_life=None, shrinkage=0),
    )
    vol = np.std(r[300 - 252 : 300], axis=0)
    expected = (1 / vol) / (1 / vol).sum()
    np.testing.assert_allclose(res.weights[0], expected, rtol=1e-10)


def test_trend_filter_direction():
    t = 400
    r = np.zeros((t, 3))
    r[0] = np.nan
    r[1:, 0] = 0.001  # 一路上涨：保留
    r[1:, 1] = -0.001  # 一路下跌：过滤成现金
    r[1:, 2] = np.where(np.arange(1, t) % 2, 0.002, -0.001)  # 震荡上行：保留
    res = aa.allocate(r, aa.TrendFilter(aa.EqualWeight(), window=100), rebalance=[300])
    np.testing.assert_allclose(res.weights[0], [1 / 3, 0.0, 1 / 3])
    assert res.diagnostics[0]["trend_off"] == [1]
    spread = aa.allocate(
        r,
        aa.TrendFilter(aa.EqualWeight(), window=100, redistribute=True),
        rebalance=[300],
    )
    np.testing.assert_allclose(spread.weights[0], [0.5, 0.0, 0.5])


def test_risk_parity_iteration_limit_is_not_ridge():
    """迭代次数不够时报 max_iter，不加 ridge、不算收敛；allocate 发警告并计数。"""
    sigma = random_cov(np.random.default_rng(5), 6)
    w, info = solve_risk_parity(sigma, max_iter=1)
    assert info["status"] == "max_iter" and info["ridge"] == 0
    assert not info["converged"] and not info["solved_with_ridge"]
    assert np.isfinite(w).all() and abs(w.sum() - 1) < 1e-12
    r = synthetic(13)
    with pytest.warns(RuntimeWarning, match="风险平价"):
        res = aa.allocate(r, aa.RiskParity(max_iter=1), rebalance=126)
    status = res.settings["risk_parity_status"]
    assert status.get("max_iter", 0) >= 1 and sum(status.values()) == len(res.rows)
    quiet = aa.allocate(r, aa.RiskParity(), rebalance=126)
    assert quiet.settings["risk_parity_status"] == {"converged": len(quiet.rows)}


def test_target_volatility_zero_ex_ante_uses_base():
    """事前波动为 0 时不许直接用满杠杆：退回基础配置并标出。"""
    r = np.zeros((300, 3))
    r[0] = np.nan
    res = aa.allocate(
        r,
        aa.TargetVolatility(aa.EqualWeight(), target=0.1, max_leverage=2.0),
        rebalance=100,
    )
    np.testing.assert_allclose(res.weights.sum(axis=1), 1.0)
    assert all("fallback" in d["target_volatility"] for d in res.diagnostics)


def test_risk_budget_aligned_by_symbol():
    """预算按代码对齐：前面有资产尚未上市时，其余资产的风险贡献仍按各自的预算。"""
    r = synthetic(14, n=3)
    r[:500, 0] = np.nan  # 第一个资产第 500 行才上市
    budget = {"A": 0.6, "B": 0.3, "C": 0.1}
    res = aa.allocate(
        r,
        aa.RiskParity(budget=budget),
        symbols=["A", "B", "C"],
        rebalance=[300, 700],
        covariance=aa.Covariance(half_life=None, shrinkage=0),
    )
    before, after = res.diagnostics
    assert before["available"] == 2 and res.weights[0, 0] == 0
    np.testing.assert_allclose(before["risk_shares"], [0.75, 0.25], atol=1e-9)
    np.testing.assert_allclose(after["risk_shares"], [0.6, 0.3, 0.1], atol=1e-9)


def test_trend_filter_uses_previous_close():
    """调仓日 t 用 t−1 日收盘与均线比较：t−1 日一根大阴线跌破均线，就要过滤（用 t−2 收盘则不会）。"""
    t = 400
    r = np.full((t, 2), 0.001)
    r[0] = np.nan
    row = 300
    r[row - 1, 0] = -0.2  # t−1 日跌 20%，跌破 100 日均线
    res = aa.allocate(r, aa.TrendFilter(aa.EqualWeight(), window=100), rebalance=[row])
    assert res.diagnostics[0]["trend_off"] == [0]
    assert res.weights[0].tolist() == [0.0, 0.5]


def test_sample_volatility_matches_nanstd():
    from ashare.allocation.covariance import sample_volatility

    rng = np.random.default_rng(15)
    x = rng.normal(size=(50, 4))
    x[rng.random(x.shape) < 0.2] = np.nan
    x[:, 3] = np.nan
    x[0, 3] = 0.1
    got = sample_volatility(x)
    want = np.array([np.nanstd(x[:, j], ddof=1) for j in range(3)])
    np.testing.assert_allclose(got[:3], want, rtol=1e-12)
    assert np.isnan(got[3])
