# ruff: noqa: E402
from pathlib import Path

import panelops

ROOT = Path(__file__).resolve().parents[2]
assert Path(panelops.__file__).resolve().is_relative_to(ROOT)
assert Path(panelops._api._core.__file__).resolve().is_relative_to(ROOT)

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from panelops import api, reference

PAIRED = {"corr", "cov", "regression_slope", "regression_residual", "sumif"}


def compare(a, b):
    np.testing.assert_allclose(a, b, rtol=2e-10, atol=2e-10, equal_nan=True)


def invoke(op, x, y, **kwargs):
    if op == "ffill":
        kwargs.pop("window", None)
        kwargs.pop("min_periods", None)
    fn = getattr(panelops, "ts_" + op)
    return fn(x, y, **kwargs) if op in PAIRED else fn(x, **kwargs)


def test_random_differential():
    counts = dict(ties=0, all_missing=0, zero_variance=0, masked=0, oversized=0, unit=0)
    for seed in range(100):
        rng = np.random.default_rng(seed)
        t, n = int(rng.integers(2, 25)), int(rng.integers(3, 9))
        x = rng.integers(-3, 4, size=(t, n)).astype(float)
        y = rng.integers(-4, 4, size=(t, n)).astype(float)
        x[:, 0] = np.nan
        x[:, 1] = 3.0
        x[rng.random(x.shape) < 0.2] = np.nan
        y[rng.random(x.shape) < 0.15] = np.nan
        mask = rng.random(x.shape) > 0.2
        window = [1, t + 2, int(rng.integers(2, t + 1))][seed % 3]
        mp = int(rng.integers(1, window + 1))
        effective = np.where(mask, x, np.nan)
        for row in range(t):
            for col in range(n):
                lo = max(0, row - window + 1)
                values = effective[lo : row + 1, col]
                valid = np.isfinite(values)
                sample = values[valid]
                counts["all_missing"] += len(sample) == 0
                if mask[row, col] and np.isfinite(x[row, col]) and len(sample) >= mp:
                    counts["ties"] += len(sample) != len(np.unique(sample))
                paired = values[valid & np.isfinite(y[lo : row + 1, col])]
                if mask[row, col] and len(paired) >= max(mp, 2):
                    counts["zero_variance"] += np.ptp(paired) == 0
        counts["masked"] += (~mask).sum()
        counts["oversized"] += window > t
        counts["unit"] += window == 1
        snapshots = [(a, a.copy()) for a in (x, y, mask)]
        for op in api.TS_OPS:
            kw = dict(mask=mask)
            if op in ("delay", "delta"):
                kw["periods"] = seed % (t + 2)
            else:
                kw.update(window=window, min_periods=mp)
            if op == "rank":
                kw["normalize"] = bool(seed % 2)
            if op in ("std", "cov"):
                kw["ddof"] = seed % 3
            if op == "quantile":
                kw["q"] = (seed % 5) / 4
            b = reference.ts(x, op, y=y if op in PAIRED else None, **kw)
            a = invoke(op, x, y, **kw)
            compare(a, b)
            assert a.flags.c_contiguous and a.dtype == np.float64
            assert not np.shares_memory(a, x) and not np.shares_memory(a, y)
        for original, snapshot in snapshots:
            np.testing.assert_array_equal(
                original.view(np.uint8), snapshot.view(np.uint8)
            )
        assert np.isnan(x[:, 0]).all()
    assert all(v > 0 for v in counts.values()), counts
    print("coverage", {k: int(v) for k, v in counts.items()})


def test_random_cross_sections_and_evaluation():
    rng = np.random.default_rng(722)
    for _ in range(50):
        x = rng.integers(-3, 4, size=(8, 13)).astype(float)
        y = rng.normal(size=x.shape)
        x[rng.random(x.shape) < 0.2] = np.nan
        y[rng.random(y.shape) < 0.2] = np.nan
        mask = rng.random(x.shape) > 0.2
        groups = rng.integers(-1, 6, size=x.shape, dtype=np.int64)
        exposures = [rng.normal(size=x.shape)]
        exposures.append(np.ascontiguousarray(exposures[0] * 2))
        weights = rng.uniform(0.1, 3, size=x.shape)
        snapshots = [(a, a.copy()) for a in (x, y, mask, groups, weights, *exposures)]
        for op in api.CS_OPS:
            kw = dict(mask=mask)
            if op in ("group_demean", "neutralize"):
                kw["groups"] = groups
            if op == "neutralize":
                kw.update(exposures=exposures, weights=weights)
            compare(getattr(panelops, "cs_" + op)(x, **kw), reference.cs(x, op, **kw))
            compare(
                getattr(panelops, "cs_" + op)(x, backend="numpy", **kw),
                reference.cs(x, op, **kw),
            )
        compare(panelops.ic(x, y, mask=mask), reference.ic(x, y, mask=mask))
        compare(
            panelops.rank_ic(x, y, mask=mask), reference.ic(x, y, mask=mask, rank=True)
        )
        for backend in ("rust", "numpy"):
            compare(
                panelops.group_returns(
                    x, y, 4, mask=mask, weights=weights, backend=backend
                ),
                reference.group_returns(x, y, 4, mask=mask, weights=weights),
            )
            compare(
                panelops.group_turnover(x, 4, mask=mask, backend=backend),
                reference.group_turnover(x, 4, mask=mask),
            )
            for kw in [dict(k=3), dict(fraction=0.31)]:
                got = panelops.targets(x, mask=mask, backend=backend, **kw)
                want = reference.targets(x, mask=mask, **kw)
                for a, b in zip(
                    (got.days, got.indptr, got.cols, got.weights), want, strict=True
                ):
                    compare(a, b)
        for original, snapshot in snapshots:
            np.testing.assert_array_equal(
                original.view(np.uint8), snapshot.view(np.uint8)
            )


@given(st.integers(1, 10000), st.integers(1, 12), st.integers(0, 10))
@settings(max_examples=35, deadline=None)
def test_no_future(seed, window, cut):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(12, 5))
    y = rng.normal(size=x.shape)
    xx, yy = x.copy(), y.copy()
    xx[cut + 1 :] = rng.normal(100, 20, size=xx[cut + 1 :].shape)
    yy[cut + 1 :] = np.nan
    mask = rng.random(x.shape) > 0.2
    future_mask = mask.copy()
    future_mask[cut + 1 :] = ~future_mask[cut + 1 :]
    for op in api.TS_OPS:
        kw = dict(mask=mask)
        if op not in ("delay", "delta"):
            kw.update(window=window, min_periods=1)
        a = invoke(op, x, y, **kw)[: cut + 1]
        b = invoke(op, xx, yy, **(kw | {"mask": future_mask}))[: cut + 1]
        np.testing.assert_array_equal(a.view(np.uint64), b.view(np.uint64))


@given(
    st.integers(0, 1000),
    st.floats(0.1, 5, allow_nan=False),
    st.floats(-10, 10, allow_nan=False),
)
@settings(max_examples=30, deadline=None)
def test_invariances(seed, scale, shift):
    rng = np.random.default_rng(seed)
    x, y = rng.normal(size=(2, 15, 8))
    compare(panelops.ic(x * scale, y), panelops.ic(x, y))
    compare(panelops.cs_zscore(x * scale + shift), panelops.cs_zscore(x))
    compare(
        panelops.ts_std(x * scale + shift, 5, min_periods=2),
        panelops.ts_std(x, 5, min_periods=2) * scale,
    )
    compare(
        panelops.cs_rank(x * scale + shift, normalize=True),
        panelops.cs_rank(x, normalize=True),
    )
    rank = panelops.cs_rank(x, normalize=True)
    assert np.all((rank >= 0) & (rank <= 1))
    tied = np.round(x)
    tied[:, 1] = tied[:, 0]
    tied_rank = panelops.cs_rank(tied, normalize=True)
    compare(tied_rank, reference.cs(tied, "rank", normalize=True))
    np.testing.assert_array_equal(tied_rank[:, 0], tied_rank[:, 1])
    compare(panelops.ts_corr(x * scale + shift, y, 5), panelops.ts_corr(x, y, 5))
    compare(panelops.ts_skew(x * scale + shift, 5), panelops.ts_skew(x, 5))
    compare(panelops.ts_kurt(x * scale + shift, 5), panelops.ts_kurt(x, 5))


@given(st.integers(0, 1000))
@settings(max_examples=25, deadline=None)
def test_neutralization_orthogonality(seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(3, 20))
    e = rng.normal(size=x.shape)
    groups = np.tile(np.array([0, 1, 1] + [2] * 17, dtype=np.int64), (3, 1))
    weights = rng.uniform(0.2, 2, size=x.shape)
    residual = panelops.cs_neutralize(
        x, groups=groups, exposures=[e, e * 2, np.ones_like(e)], weights=weights
    )
    for t in range(3):
        for exposure in [np.ones(20), e[t], *[(groups[t] == g) for g in range(3)]]:
            assert abs(np.dot(residual[t] * weights[t], exposure)) < 2e-10
    assert np.all(abs(residual[:, 0]) < 2e-10)


def test_thread_determinism_and_layout():
    rng = np.random.default_rng(333)
    x, y = rng.normal(size=(2, 37, 17))
    x[3::4, 2::3] = np.nan
    try:
        for op in api.TS_OPS:
            kw = {} if op in ("delay", "delta") else dict(window=7, min_periods=2)
            panelops.set_num_threads(1)
            a = invoke(op, x, y, **kw)
            panelops.set_num_threads(0)
            b = invoke(op, x, y, **kw)
            np.testing.assert_array_equal(a.view(np.uint64), b.view(np.uint64))
            for threads in (2, 3, 7):
                panelops.set_num_threads(threads)
                c = invoke(op, x, y, **kw)
                np.testing.assert_array_equal(a.view(np.uint64), c.view(np.uint64))
        for op in api.CS_OPS:
            kw = (
                dict(groups=np.zeros(x.shape, np.int64))
                if op in ("group_demean", "neutralize")
                else {}
            )
            panelops.set_num_threads(1)
            a = getattr(panelops, "cs_" + op)(x, **kw)
            panelops.set_num_threads(0)
            b = getattr(panelops, "cs_" + op)(x, **kw)
            np.testing.assert_array_equal(a.view(np.uint64), b.view(np.uint64))
    finally:
        panelops.set_num_threads(0)


def test_long_dynamic_range_and_compensation():
    x = np.tile(np.array([1e16, 1.0, -1e16, 3.0, 1e-12, -2.0])[:, None], (1200, 2))
    for window in (5, 60, 250):
        for op in ("sum", "mean"):
            actual = invoke(op, x, x, window=window, min_periods=1)
            expected = reference.ts(x, op, window=window, min_periods=1)
            bounds = np.empty_like(x)
            for t in range(len(x)):
                values = x[max(0, t - window + 1) : t + 1]
                bounds[t] = (
                    64
                    * np.finfo(float).eps
                    * np.abs(values).sum(0)
                    / (len(values) if op == "mean" else 1)
                )
            assert np.all(np.abs(actual - expected) <= bounds)
    x = np.ascontiguousarray(
        1e10 + np.arange(3000)[:, None] * np.array([[0.001, 0.002]])
    )
    compare(panelops.ts_std(x, 250), reference.ts(x, "std", window=250))


def test_contract_and_empty():
    x = np.ones((5, 4))
    for bad in [x.T, x[:, ::2], np.ones((5, 4, 1))]:
        with pytest.raises(ValueError):
            panelops.ts_mean(bad, 2)
    with pytest.raises(TypeError):
        panelops.ts_mean(x.astype("f4"), 2)
    with pytest.raises(TypeError):
        panelops.ts_mean(x, 2, ddof=-1)
    with pytest.raises(ValueError):
        panelops.ts_std(x, 2, ddof=-1)
    for kw in [
        dict(window=0),
        dict(window=2, min_periods=3),
        dict(window=2, min_periods=0),
        dict(window=2, mask=np.ones((3, 4), bool)),
    ]:
        with pytest.raises(ValueError):
            panelops.ts_mean(x, **kw)
    with pytest.raises(TypeError):
        panelops.ts_mean(x, True)
    for shape in [(0, 4), (4, 0), (0, 0)]:
        empty = np.empty(shape)
        for op in api.TS_OPS:
            assert invoke(op, empty, empty).shape == shape
        for op in api.CS_OPS:
            for backend in ("rust", "numpy"):
                result = getattr(panelops, "cs_" + op)(
                    empty, groups=np.zeros(shape, np.int64), backend=backend
                )
                assert result.shape == shape
        assert panelops.ic(empty, empty).shape == (shape[0],)
        assert panelops.group_returns(empty, empty).shape == (shape[0], 5)
    y = np.array([[1.0, np.inf, -np.inf, np.nan]])
    compare(panelops.cs_rank(y), np.array([[1.0, np.nan, np.nan, np.nan]]))
    original = x.copy()
    panelops.ts_mean(x, 2)
    np.testing.assert_array_equal(x, original)


# Focused mutation witnesses keep each numerical rule visible.
def test_window_includes_today():
    x = np.array([[1.0], [2.0], [9.0]])
    compare(panelops.ts_mean(x, 2), np.array([[np.nan], [1.5], [5.5]]))
    compare(panelops.ts_min(x, 2), np.array([[np.nan], [1.0], [2.0]]))


def test_min_periods():
    x = np.array([[1.0], [np.nan], [3.0]])
    assert np.isnan(panelops.ts_sum(x, 3, min_periods=2)[0, 0])
    assert panelops.ts_sum(x, 3, min_periods=2)[2, 0] == 4


def test_average_ties():
    x = np.array([[2.0, 2.0, 8.0]])
    compare(panelops.cs_rank(x), np.array([[1.5, 1.5, 3.0]]))
    compare(panelops.cs_rank(np.array([[1.0]]), normalize=True), np.array([[0.5]]))


def test_ddof():
    compare(
        panelops.ts_std(np.array([[1.0], [3.0]]), 2),
        np.array([[np.nan], [np.sqrt(2.0)]]),
    )


def test_output_mask():
    x = np.array([[1.0], [2.0], [3.0]])
    mask = np.array([[True], [True], [False]])
    assert np.isnan(panelops.ts_mean(x, 3, min_periods=1, mask=mask)[2, 0])


def test_neutralize_intercept():
    x = np.array([[4.0, 7.0, 8.0, 12.0]])
    e = np.array([[1.0, 2.0, 3.0, 4.0]])
    result = panelops.cs_neutralize(x, exposures=[e])
    assert abs(result.sum()) < 2e-10
    compare(result, reference.cs(x, "neutralize", exposures=[e]))


def test_ic_alignment():
    x = np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0], [1.0, 3.0, 2.0]])
    y = x.copy()
    compare(panelops.ic(x, y), np.ones(3))


def test_target_indptr():
    x = np.array([[1.0, 3.0, 2.0], [np.nan] * 3, [3.0, 2.0, 1.0]])
    target = panelops.targets(x, k=2)
    np.testing.assert_array_equal(target.indptr, [0, 2, 2, 4])
    np.testing.assert_array_equal(target.cols, [1, 2, 0, 1])


def test_row_boundaries():
    x = np.arange(257 * 3, dtype=float).reshape(257, 3)
    expected = np.tile([1.0, 2.0, 3.0], (257, 1))
    compare(panelops.cs_rank(x), expected)
