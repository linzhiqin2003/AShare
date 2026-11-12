"""审查补充：手算矩、缓存键、估计池和正式执行路径。"""

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import portopt
import pytest

import ashare
from ashare.risk import ModelCache, RiskModel, engine, kernels
from ashare.risk.data import MemoryPanel, adjusted_returns
from ashare.risk.fundamental import FundamentalConfig, exposures_at, specific_variance
from ashare.risk.rolling import RollingDescriptors
from ashare.risk.series import model_series
from ashare.risk.statistical import shrinkage_components, statistical_model
from test.risk.test_series import source

ROOT = Path(__file__).resolve().parents[2]
assert Path(ashare.__file__).resolve().is_relative_to(ROOT)
assert Path(portopt.__file__).resolve().is_relative_to(ROOT)


def config(**kw):
    return replace(
        FundamentalConfig(
            industry_codes=tuple(range(5)),
            min_listing_days=5,
            min_descriptor_days=3,
            history=20,
            structural_days=10,
        ),
        **kw,
    )


def state_at(directory, date):
    for path in directory.glob("*.npz"):
        with np.load(path, allow_pickle=False) as data:
            if json.loads(str(data["meta"]))["date"] == str(date):
                return {k: data[k].copy() for k in data.files}
    raise AssertionError("缺少缓存日期")


@pytest.mark.parametrize("change", ("version", "config", "values", "dates"))
def test_cache_keys_and_one_generation(tmp_path, change):
    p, c = source(32, 40), config()
    before = list(model_series(p, config=c, cache=tmp_path))
    keys = {x.name for x in tmp_path.glob("*.npz")}
    if change == "version":
        p = MemoryPanel(p.dates, p.symbols, "revision-two", p.fields)
    elif change == "config":
        c = replace(c, vol_half_life=3)
    elif change == "dates":
        p = MemoryPanel(
            np.datetime64("2021-01-01") + np.arange(32), p.symbols, p.version, p.fields
        )
    else:
        fields = {k: v.copy() for k, v in p.fields.items()}
        fields["close"][17:, 5] *= np.linspace(1, 1.3, 15)
        p = MemoryPanel(p.dates, p.symbols, p.version, fields)
    actual = list(model_series(p, config=c, cache=tmp_path))
    expected = list(model_series(p, config=c))
    assert {x.name for x in tmp_path.glob("*.npz")} != keys
    assert actual[-1].metadata != before[-1].metadata
    for a, b in zip(actual, expected, strict=True):
        assert a.date == b.date and a.metadata == b.metadata
        for field in ("B", "F", "D"):
            np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
    cache = ModelCache(tmp_path)
    assert cache.stats()["snapshots"] == 32 and cache.stats()["generations"] == 1
    assert len(list(tmp_path.glob("*.npz"))) == 32
    assert cache.clear() > 0 and cache.stats()["bytes"] == 0


def test_cache_lossless_history_exposure_rebuild(tmp_path, monkeypatch):
    p, c = source(35, 45), config()
    full = list(model_series(p, config=c, cache=tmp_path))
    with np.load(next(tmp_path.glob("*.npz")), allow_pickle=False) as data:
        assert "B" not in data.files
    calls = []
    import ashare.risk.series as series

    original = series.exposures_at

    def record(*args, **kwargs):
        calls.append(args[2])
        return original(*args, **kwargs)

    monkeypatch.setattr(series, "exposures_at", record)
    retrieved = list(model_series(p, config=c, cache=tmp_path, start=33))
    assert 34 not in calls and 33 in calls
    for a, b in zip(retrieved, full[33:], strict=True):
        for name in ("B", "F", "D"):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


def test_half_life_and_covariance_hand_moments():
    weights = kernels.exp_weights(5, 2)
    assert weights[2] / weights[4] == pytest.approx(0.5, abs=1e-15)
    x = np.array([[1.0, 4.0], [3.0, -2.0], [2.0, 5.0], [8.0, 1.0]])

    def cov(h):
        w = np.array([2 ** (-3 / h), 2 ** (-2 / h), 2 ** (-1 / h), 1.0])
        w /= w.sum()
        mean = np.sum(x * w[:, None], axis=0)
        return sum(w[i] * np.outer(x[i] - mean, x[i] - mean) for i in range(4))

    v, c = cov(1), cov(2)
    expected = (
        c
        / np.sqrt(np.outer(np.diag(c), np.diag(c)))
        * np.sqrt(np.outer(np.diag(v), np.diag(v)))
    )
    np.testing.assert_allclose(
        engine.factor_covariance(x, 1, 2, 0, 0), expected, rtol=1e-13, atol=1e-13
    )
    np.testing.assert_allclose(portopt.ew_covariance(x, 2), c, rtol=1e-13, atol=1e-13)
    for matrix in (kernels.psd(np.array([[0.2, 0.03], [0.03, 0.1]])), expected):
        model = RiskModel(
            ("a", "b"),
            np.eye(2),
            ("x", "y"),
            matrix + np.array([[0, 1e-18], [0, 0]]),
            np.ones(2),
        )
        np.testing.assert_array_equal(model.F, model.F.T)


def test_structural_blend_zero_residual_and_floor():
    r = np.full((20, 4), np.nan)
    r[:, :2] = np.tile([-0.02, 0.02], 10)[:, None]
    r[-4:, 2] = [-0.001, 0.001, -0.001, 0.001]
    r[:, 3] = 0
    c = config(
        specific_half_life=1e12,
        specific_corr_half_life=1e12,
        specific_lags=0,
        structural_days=20,
        shrinkage=0,
    )
    d, _ = specific_variance(r, np.ones((4, 1)), np.ones(4), c, np.ones(4, bool))
    assert np.sqrt(d[2]) == pytest.approx(0.2 * 0.001 + 0.8 * 0.02, abs=1e-12)
    assert np.sqrt(d[3]) == pytest.approx(0.02, abs=1e-12)
    high = replace(c, variance_floor=0.01)
    floor, _ = specific_variance(np.zeros((20, 4)), np.ones((4, 1)), np.ones(4), high)
    np.testing.assert_array_equal(floor, np.full(4, 0.01))


def test_specific_groups_ignore_non_estimation_stocks():
    # 两边均超过 64 列，固定为同一 Rust 内核，隔离分组成员变化。
    rng = np.random.default_rng(47)
    r = rng.normal(size=(80, 80)) * np.linspace(0.005, 0.04, 80)
    b = np.ones((80, 1))
    cap = np.exp(np.linspace(1, 10, 80))
    c = config(structural_days=30, shrinkage=0.7)
    expected, _ = specific_variance(r, b, cap, c, np.ones(80, bool))
    actual, _ = specific_variance(
        np.column_stack((r, rng.normal(size=(80, 80)) * 10)),
        np.ones((160, 1)),
        np.r_[cap, np.ones(80)],
        c,
        np.r_[np.ones(80, bool), np.zeros(80, bool)],
    )
    np.testing.assert_array_equal(actual[:80], expected)


def test_statistical_short_history_and_pca_diagonal():
    rng = np.random.default_rng(99)
    r = rng.normal(0, 0.02, (200, 12))
    r[:-3, 0] = np.nan
    r[-3:, 0] = [-0.0001, 0, 0.0001]
    u, d, _ = shrinkage_components(r, np.inf, "identity", 0.3)
    diag = np.sum(u * u, axis=1) + d
    prior = np.median(diag[1:])
    m = statistical_model(r, half_life=np.inf, components=None, shrinkage=0.3)
    actual = np.sum((m.B @ m.F) * m.B, axis=1) + m.D
    assert actual[0] == pytest.approx((3 / 60) * diag[0] + (57 / 60) * prior, rel=1e-12)
    # 独立稠密特征分解只出现在小样本测试中。
    complete = rng.normal(0, 0.02, (100, 10))
    centered = complete - complete.mean(0)
    sample = centered.T @ centered / 100
    sigma = 0.7 * sample + 0.3 * np.trace(sample) / 10 * np.eye(10)
    values, vectors = np.linalg.eigh(sigma)
    common = (vectors[:, -3:] * values[-3:]) @ vectors[:, -3:].T
    m = statistical_model(complete, half_life=np.inf, components=3, shrinkage=0.3)
    np.testing.assert_allclose(m.D, np.diag(sigma - common), atol=1e-15, rtol=1e-12)


def test_previous_estimation_unknown_industry_and_history(tmp_path, monkeypatch):
    p = source(40, 60)
    fields = {k: v.copy() for k, v in p.fields.items()}
    fields["is_suspended"][18, 4] = True
    fields["industry"][:, :8] = 999
    p = MemoryPanel(p.dates, p.symbols, p.version, fields)
    c = config(history=7, regime=False)
    import ashare.risk.series as series

    sizes = []
    original = series.estimate_covariances

    def record(fs, *args, **kwargs):
        sizes.append(len(fs))
        return original(fs, *args, **kwargs)

    monkeypatch.setattr(series, "estimate_covariances", record)
    models = list(model_series(p, config=c, cache=tmp_path))
    assert sizes and max(sizes) == 7 and all(size <= 7 for size in sizes)
    r = adjusted_returns(p, 39)
    r[p["is_suspended"]] = np.nan
    _, _, est, cap = exposures_at(p, r, 18, c)
    assert not est[4]
    state = state_at(tmp_path, p.dates[19])
    b = models[18].B
    ok = est & np.isfinite(r[19])
    inds = np.arange(1, 7)
    constraint = np.zeros(b.shape[1])
    constraint[inds] = cap[ok] @ b[ok][:, inds]
    # 约束及一阶最优条件为独立预言机，含 unknown 行业。
    assert abs(constraint @ state["f"]) / cap[ok].sum() < 1e-11
    residual = r[19, ok] - b[ok] @ state["f"]
    gradient = b[ok].T @ (np.sqrt(cap[ok]) * residual)
    projected = gradient - constraint * (constraint @ gradient) / (
        constraint @ constraint
    )
    assert np.max(np.abs(projected)) < 1e-7
    assert constraint[6] > 0


def test_regime_multipliers_and_estimation_bias(tmp_path, monkeypatch):
    p = source(32, 45)
    c = config(regime=True, history=8)
    import ashare.risk.series as series

    def covariance(fs, rs, b, cap, config, est):
        return (
            np.eye(b.shape[1]) * 4,
            np.full(len(cap), 9.0),
            np.full(b.shape[1], 4.0),
            np.full(len(cap), 9.0),
        )

    monkeypatch.setattr(series, "estimate_covariances", covariance)
    original = series.cross_sectional_bias
    weights_seen = []

    def bias(realized, predicted, weights=None):
        if weights is None:
            return 4.0
        weights_seen.append(weights.copy())
        return 9.0

    monkeypatch.setattr(series, "cross_sectional_bias", bias)
    models = list(model_series(p, config=c, cache=tmp_path))
    np.testing.assert_allclose(
        models[-1].F, np.eye(models[-1].F.shape[0]) * 16, atol=1e-13
    )
    np.testing.assert_allclose(models[-1].D, 81, atol=1e-12)
    assert weights_seen and all(w[0] == 0 for w in weights_seen)  # stock 0 尚未上市。
    assert np.isnan(original(np.array([100.0]), np.ones(1), np.zeros(1)))


def test_rust_minimum_norm_and_thread_determinism():
    rng = np.random.default_rng(13)
    n = 90
    ind = np.eye(3)[np.arange(n) % 3]
    ind[:, -1] = 0
    ind[0] = [0, 0, 1]  # 第三个行业仅一只，另加空列。
    b = np.column_stack((np.ones(n), ind, np.zeros(n), rng.normal(size=n)))
    cap = np.exp(rng.normal(size=n))
    est = np.ones(n, bool)
    r = rng.normal(size=n)
    b = np.column_stack((b, b[:, -1]))
    for exposure in (
        b,
        np.column_stack((b[:, :-1], b[:, -1] + rng.normal(size=n) * 1e-9)),
    ):
        f, u = engine.cross_section(r, exposure, cap, est, (1, 2, 3, 4))
        expected, residual = kernels.cross_section(r, exposure, cap, est, (1, 2, 3, 4))
        np.testing.assert_allclose(u, residual, atol=2e-6, rtol=1e-6)
        if np.array_equal(exposure[:, -1], exposure[:, -2]):
            np.testing.assert_allclose(f, expected, atol=1e-11)
        f1, u1 = engine.cross_section_batch(
            np.tile(r, (5, 1)),
            exposure,
            np.tile(cap, (5, 1)),
            np.tile(est, (5, 1)),
            (1, 2, 3, 4),
            1,
        )
        f4, u4 = engine.cross_section_batch(
            np.tile(r, (5, 1)),
            exposure,
            np.tile(cap, (5, 1)),
            np.tile(est, (5, 1)),
            (1, 2, 3, 4),
            4,
        )
        np.testing.assert_array_equal(f1, f4)
        np.testing.assert_array_equal(u1, u4)
        np.testing.assert_array_equal(f, f1[0])
        np.testing.assert_array_equal(u, u1[0])
    previous = portopt.get_num_threads()
    try:
        portopt.set_num_threads(2)
        assert portopt.get_num_threads() == 2
        portopt.set_num_threads(1)
        assert portopt.get_num_threads() == 1
    finally:
        portopt.set_num_threads(previous)


def test_rolling_descriptors_independent_windows_and_pipeline_chunks(tmp_path):
    p = source(600, 80)
    c = config(beta_window=70, momentum_skip=5, history=30)
    r = adjusted_returns(p, 599)
    r[p["is_suspended"]] = np.nan
    ages = np.cumsum(p["is_listed"], axis=0, dtype=np.int32)
    rolling = RollingDescriptors(p, r, c, ages)
    for day in (22, 150, 260, 599):
        direct = exposures_at(p, r, day, c, listing_ages=ages)[0]
        incremental = exposures_at(p, r, day, c, listing_ages=ages, rolling=rolling)[0]
        np.testing.assert_allclose(incremental, direct, atol=2e-10, rtol=2e-9)
    small = source(55, 45)
    original = portopt.get_num_threads()
    try:
        portopt.set_num_threads(1)
        single = list(model_series(small, config=c, batch_size=1))
        portopt.set_num_threads(4)
        whole = list(model_series(small, config=c, batch_size=17))
        prefix = list(
            model_series(small, config=c, cache=tmp_path, end=54, batch_size=17)
        )
        append = list(
            model_series(small, config=c, cache=tmp_path, start=54, batch_size=1)
        )
        for a, b, d in zip(single, whole, prefix + append, strict=True):
            for name in ("B", "F", "D"):
                np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
                np.testing.assert_array_equal(getattr(a, name), getattr(d, name))
    finally:
        portopt.set_num_threads(original)


def test_market_beta_uses_previous_cap_hand_calculation():
    p = source(5, 4)
    fields = {k: v.copy() for k, v in p.fields.items()}
    fields["is_listed"][:] = True
    fields["total_mv"] = np.array(
        [[1, 2, 4, 8], [8, 4, 2, 1], [2, 8, 1, 4], [4, 1, 8, 2], [20, 1, 1, 1]], float
    )
    p = MemoryPanel(p.dates, p.symbols, p.version, fields)
    r = np.array(
        [
            [np.nan] * 4,
            [0.01, 0.02, -0.01, 0.03],
            [0.02, -0.01, 0.04, -0.02],
            [-0.02, 0.03, 0.01, 0.04],
            [0.05, 0.01, -0.03, 0.02],
        ]
    )
    c = config(
        min_listing_days=1, min_descriptor_days=2, beta_window=4, beta_half_life=2
    )
    market = np.array(
        [
            sum(fields["total_mv"][j - 1] * r[j]) / sum(fields["total_mv"][j - 1])
            for j in range(1, 5)
        ]
    )
    weights = np.array([2 ** (-3 / 2), 0.5, 2 ** (-1 / 2), 1.0])
    weights /= sum(weights)
    dx = market - sum(weights * market)
    dy = r[1:] - sum(weights[:, None] * r[1:])
    expected = sum(weights[:, None] * dx[:, None] * dy) / sum(weights * dx * dx)
    raw = RollingDescriptors(p, r, c, np.cumsum(fields["is_listed"], axis=0)).at(4)
    np.testing.assert_allclose(raw["beta"], expected, atol=1e-13, rtol=1e-13)


def test_well_conditioned_gram_avoids_svd_and_psd_exact(monkeypatch):
    rng = np.random.default_rng(112)
    n = 70
    b = np.column_stack(
        (np.ones(n), np.eye(3)[np.arange(n) % 3], rng.normal(size=(n, 2)))
    )

    def unexpected(*args, **kwargs):
        raise AssertionError("良态 Gram/KKT 不应退回 SVD")

    monkeypatch.setattr(kernels, "cross_section", unexpected)
    kernels.cross_section_gram(
        rng.normal(size=n), b, np.exp(rng.normal(size=n)), np.ones(n, bool), (1, 2, 3)
    )
    a = rng.normal(size=(7, 7))
    matrix = kernels.psd(a @ a.T)
    np.testing.assert_array_equal(matrix, matrix.T)


def test_checkpoint_restores_only_one_day_and_clear_preserves_unrelated(
    tmp_path, monkeypatch
):
    p, c = source(80, 45), config()
    list(model_series(p, config=c, cache=tmp_path, end=79))
    steps = []
    original = RollingDescriptors.restore

    def record(self, state):
        original(self, state)
        steps.append(self.day)

    monkeypatch.setattr(RollingDescriptors, "restore", record)
    actual = list(model_series(p, config=c, cache=tmp_path, start=79))
    assert steps == [78]
    expected = list(model_series(p, config=c))[-1]
    for name in ("B", "F", "D"):
        np.testing.assert_array_equal(getattr(actual[0], name), getattr(expected, name))
    unrelated = tmp_path / "user-data.npz"
    unrelated.write_bytes(b"unrelated")
    cache = ModelCache(tmp_path)
    size = cache.stats()["bytes"]
    assert cache.clear() == size and unrelated.read_bytes() == b"unrelated"


def test_singleton_industry_residual_is_uninformative(tmp_path):
    p = source(40, 60)
    fields = {k: v.copy() for k, v in p.fields.items()}
    fields["industry"][:, 5] = 6
    p = MemoryPanel(p.dates, p.symbols, p.version, fields)
    c = config(industry_codes=tuple(range(7)))
    models = list(model_series(p, config=c, cache=tmp_path))
    state = state_at(tmp_path, p.dates[-1])
    assert np.isnan(state["residual"][5])
    assert models[-1].D[5] > c.variance_floor * 100


@pytest.mark.parametrize("perturbation", (0.0, 1e-9))
def test_pipeline_calls_native_and_degenerate_append(
    tmp_path, monkeypatch, perturbation
):
    p, c = source(35, 50), config()
    rng = np.random.default_rng(141)
    column = rng.normal(size=(35, 50))
    external = np.stack(
        (column, column + perturbation * rng.normal(size=column.shape)), axis=2
    )
    calls = []
    original = portopt.regress

    def record(returns, *args, **kwargs):
        calls.append(len(returns))
        return original(returns, *args, **kwargs)

    monkeypatch.setattr(portopt, "regress", record)
    options = dict(
        config=c, external=external, external_names=("duplicate_a", "duplicate_b")
    )
    full = list(model_series(p, batch_size=12, **options))
    assert calls and sum(calls) == 30 and max(calls) > 1
    list(model_series(p, cache=tmp_path, end=34, batch_size=7, **options))
    calls.clear()
    append = list(model_series(p, cache=tmp_path, start=34, **options))
    assert calls == [1]
    for field in ("B", "F", "D"):
        np.testing.assert_array_equal(
            getattr(append[0], field), getattr(full[-1], field)
        )


def test_external_model_covariance_is_exactly_symmetric():
    f = np.array([[0.2, 0.03 + 1e-16], [0.03, 0.1]])
    assert not np.array_equal(f, f.T)
    m = RiskModel(("a", "b"), np.eye(2), ("x", "y"), f, np.ones(2))
    np.testing.assert_array_equal(m.F, (f + f.T) / 2)
    np.testing.assert_array_equal(m.F, m.F.T)


def test_cmra_endpoints_missing_and_checkpoint():
    p = source(600, 8)
    c = config()
    # 手工十二个月收益：CMRA 不包含零端点，而是 1..12 个月累计的极差。
    logs = np.tile(
        np.array(
            [0.01, -0.02, 0.03, -0.01, 0.02, 0.01, -0.03, 0.04, 0.01, -0.01, 0.02, 0.03]
        ),
        2,
    )
    r = np.zeros((600, 8))
    r[1:505] = np.expm1(np.repeat(logs / 21, 21))[:, None]
    r[1:505, 0] = np.expm1(np.arange(1, 505) * 0.00001)
    r[510, 1] = np.nan
    ages = np.cumsum(p["is_listed"], axis=0)
    rolling = RollingDescriptors(p, r, c, ages)
    a = rolling.at(504)
    expected = np.full(8, np.ptp(np.cumsum(logs[-12:][::-1])))
    expected[0] = sum(range(253, 484)) * 0.00001
    np.testing.assert_allclose(a["cmra"], expected, atol=1e-14, rtol=1e-13)
    checkpoint = rolling.checkpoint()
    full = rolling.at(599)
    restored = RollingDescriptors(p, r, c, ages)
    restored.restore(checkpoint)
    appended = restored.at(599)
    assert np.isnan(full["cmra"][1]) and np.isfinite(full["cmra"][0])
    np.testing.assert_array_equal(full["cmra"], appended["cmra"])
    direct = exposures_at(p, r, 599, c, listing_ages=ages)[0]
    fast = exposures_at(p, r, 599, c, listing_ages=ages, rolling=restored)[0]
    np.testing.assert_allclose(fast, direct, atol=2e-10, rtol=2e-9)
