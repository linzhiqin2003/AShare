from pathlib import Path

import numpy as np
import portopt
import pytest

import ashare
from ashare.risk import RiskModel
from ashare.risk.kernels import cross_section, ew_covariance, ew_variance, rolling_bias
from ashare.risk.statistical import statistical_model

ROOT = Path(__file__).resolve().parents[2]
assert Path(ashare.__file__).resolve().is_relative_to(ROOT)
assert Path(portopt.__file__).resolve().is_relative_to(ROOT)


def test_risk_structure_and_derivative():
    rng = np.random.default_rng(12)
    b = rng.normal(size=(12, 4))
    z = rng.normal(size=(4, 4))
    m = RiskModel(
        tuple(map(str, range(12))), b, tuple("abcd"), z @ z.T, np.linspace(0.1, 0.3, 12)
    )
    w, bench = rng.normal(size=(2, 12))
    a = w - bench
    assert m.variance(w, bench) == pytest.approx(a @ m.dense() @ a, rel=1e-12)
    assert m.decomposition(a)["total"] == pytest.approx(m.variance(a), rel=1e-12)
    assert m.contributions(a).sum() == pytest.approx(m.volatility(a), rel=1e-12)
    for i in range(12):
        step = np.eye(12)[i] * 1e-6
        derivative = (m.volatility(a + step) - m.volatility(a - step)) / (2e-6)
        assert m.marginal_risk(a)[i] == pytest.approx(derivative, abs=1e-8)
    with pytest.raises(ValueError):
        RiskModel(
            tuple(map(str, range(257))),
            np.ones((257, 1)),
            ("a",),
            np.eye(1),
            np.ones(257),
        ).dense()
    with pytest.raises(ValueError):
        m.B[0, 0] = 2


def test_nw_sign_and_missing_lag():
    x = np.array([[1.0, np.nan], [2.0, 1.0], [3.0, 2.0], [4.0, 3.0], [5.0, 4.0]])
    z = x[:, 0] - x[:, 0].mean()
    expected = (z @ z + z[1:] @ z[:-1]) / 5
    assert ew_variance(x, np.inf, 1)[0] == pytest.approx(expected)
    np.testing.assert_allclose(
        np.diag(ew_covariance(x, np.inf, 1)), ew_variance(x, np.inf, 1), atol=1e-14
    )
    assert np.linalg.eigvalsh(ew_covariance(x, np.inf, 2)).min() > -1e-12


def test_regression_weight_and_industry_constraint():
    rng = np.random.default_rng(192)
    n = 70
    cap = np.exp(rng.normal(4, 2, n))
    industries = np.eye(3)[np.arange(n) % 3]
    b = np.column_stack((np.ones(n), industries, rng.normal(size=(n, 2))))
    r = rng.normal(size=n)
    coef, residual = cross_section(r, b, cap, np.ones(n, bool), (1, 2, 3))
    industry_weights = cap @ industries
    assert abs(coef[1:4] @ industry_weights) < 1e-9
    # 独立 KKT 解作为权重口径的预言机。
    w = np.diag(np.sqrt(cap))
    c = np.r_[0, industry_weights, 0, 0]
    kkt = np.block([[b.T @ w @ b, c[:, None]], [c[None, :], np.zeros((1, 1))]])
    expected = np.linalg.solve(kkt, np.r_[b.T @ w @ r, 0])[:-1]
    np.testing.assert_allclose(coef, expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(residual, r - b @ expected, rtol=1e-10, atol=1e-10)


def test_sklearn_ledoit_wolf():
    from sklearn.covariance import LedoitWolf

    rng = np.random.default_rng(81)
    for t, n in ((100, 8), (40, 80), (500, 20)):
        x = rng.normal(size=(t, n)) @ np.diag(np.linspace(0.01, 0.06, n))
        reference = LedoitWolf().fit(x)
        model = statistical_model(x, half_life=np.inf, components=None)
        assert model.metadata["shrinkage"] == pytest.approx(
            reference.shrinkage_, abs=1e-12
        )
        np.testing.assert_allclose(
            model.dense(), reference.covariance_, rtol=1e-11, atol=1e-13
        )


def test_statistical_missing_and_pca():
    rng = np.random.default_rng(821)
    x = rng.normal(size=(300, 25))
    x[:200, 0] = np.nan
    x[:, 1] = np.nan
    for target in ("identity", "constant_correlation"):
        m = statistical_model(x, components=5, target=target)
        assert np.isfinite(m.D).all() and np.all(m.D > 0)
        assert np.linalg.eigvalsh(m.dense()).min() > 0


def test_rust_kernels():
    rng = np.random.default_rng(294)
    x = rng.normal(size=(350, 40))
    x[rng.random(x.shape) < 0.02] = np.nan
    for threads in (1, 3):
        for lags in (0, 2, 5):
            np.testing.assert_allclose(
                portopt.ew_variance(x, 63, lags, threads),
                ew_variance(x, 63, lags),
                rtol=2e-13,
                atol=2e-13,
            )
        p = np.full_like(x, 1.1)
        np.testing.assert_allclose(
            portopt.rolling_bias(x, p, 20, threads),
            rolling_bias(x, p, 20),
            rtol=1e-12,
            atol=1e-12,
        )
    b = np.column_stack(
        (np.ones(40), np.eye(3)[np.arange(40) % 3], rng.normal(size=(40, 2)))
    )
    cap = np.exp(rng.normal(size=x.shape))
    est = rng.random(x.shape) > 0.1
    f, u = portopt.regress(x, b, cap, est, (1, 2, 3), threads=3)
    for day in range(len(x)):
        ff, uu = cross_section(x[day], b, cap[day], est[day], (1, 2, 3))
        np.testing.assert_allclose(f[day], ff, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(u[day], uu, atol=1e-10, rtol=1e-10)
    one = portopt.regress(x, b, cap, est, (1, 2, 3), threads=1)
    np.testing.assert_array_equal(f, one[0])
    np.testing.assert_array_equal(u, one[1])


def test_rust_covariance_and_invalid_inputs():
    rng = np.random.default_rng(13)
    x = rng.normal(size=(400, 15))
    x[:30, 2] = np.nan
    for lag in (0, 2, 5):
        for threads in (1, 4):
            np.testing.assert_allclose(
                portopt.ew_covariance(x, 84, lag, threads),
                ew_covariance(x, 84, lag),
                rtol=1e-12,
                atol=1e-12,
            )
    for value in (np.zeros((0, 3)), np.full((2, 3), np.inf)):
        with pytest.raises(ValueError):
            portopt.ew_variance(value)
    with pytest.raises(ValueError):
        portopt.rolling_bias(x, x, 0)
    with pytest.raises(ValueError):
        portopt.regress(x, np.zeros((3, 2)), np.ones_like(x), np.ones_like(x, bool))


def test_constant_series_and_extreme_decay():
    x = np.random.default_rng(32).normal(size=(100, 8))
    x[:, 0] = 0
    for half_life in (0.01, 20):
        for target in ("identity", "constant_correlation"):
            m = statistical_model(x, half_life=half_life, target=target, components=3)
            assert np.isfinite(m.D).all()
            assert np.linalg.eigvalsh(m.F).min() >= 0


def test_varying_exposure_panel():
    from ashare.risk.kernels import cross_section_batch

    rng = np.random.default_rng(117)
    r = rng.normal(size=(20, 70))
    b = rng.normal(size=(20, 70, 7))
    b[:, :, 0] = 1
    cap = np.exp(rng.normal(size=r.shape))
    est = np.ones_like(r, bool)
    f, u = portopt.regress(r, b, cap, est, threads=3)
    reference = cross_section_batch(r, b, cap, est)
    np.testing.assert_allclose(f, reference[0], atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(u, reference[1], atol=1e-10, rtol=1e-10)


def test_gram_reference_and_rank_deficiency():
    from ashare.risk.kernels import cross_section, cross_section_gram

    rng = np.random.default_rng(723)
    n = 150
    industry = np.eye(4)[np.arange(n) % 4]
    b = np.column_stack((np.ones(n), industry, rng.normal(size=(n, 3))))
    r = rng.normal(size=n)
    cap = np.exp(rng.normal(size=n))
    est = rng.random(n) > 0.1
    for design in (
        b,
        np.column_stack((b, b[:, -1])),
        np.column_stack((b, np.zeros(n))),
    ):
        a = cross_section(r, design, cap, est, np.arange(1, 5))
        z = cross_section_gram(r, design, cap, est, np.arange(1, 5))
        np.testing.assert_allclose(a[0], z[0], atol=1e-11, rtol=1e-11)
        np.testing.assert_allclose(a[1], z[1], atol=1e-11, rtol=1e-11)


def test_eigen_adjustment_repeatable_psd():
    from ashare.risk.fundamental import eigen_adjustment

    f = np.diag([0.00001, 0.0001, 0.001, 0.003])
    a = eigen_adjustment(f, 100, 42, simulations=20)
    b = eigen_adjustment(f, 100, 42, simulations=20)
    np.testing.assert_array_equal(a, b)
    assert np.linalg.eigvalsh(a).min() > 0
    assert np.max(np.abs(a - f)) > 1e-5
