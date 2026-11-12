# ruff: noqa: E402
from pathlib import Path

import panelops as po

ROOT = Path(__file__).resolve().parents[2]
assert Path(po.__file__).resolve().is_relative_to(ROOT)
assert Path(po._api._core.__file__).resolve().is_relative_to(ROOT)

import numpy as np
import pytest
from panelops import reference

ELEMENTS = (
    "gt ge lt le eq ne and_ or_ not_ where maximum minimum sign power "
    "signed_power sqrt exp clip is_nan fill_nan"
).split()
BINARY = {
    "gt",
    "ge",
    "lt",
    "le",
    "eq",
    "ne",
    "and_",
    "or_",
    "maximum",
    "minimum",
    "power",
    "signed_power",
    "fill_nan",
}


@pytest.mark.parametrize("op", ELEMENTS)
def test_elementwise_independent_reference(op):
    x = np.array(
        [
            [np.nan, np.inf, -np.inf, 0.0, -0.0, -2.0, 1.0, 3.0],
            [2.0, 0.1, -0.1, 1e100, 700.0, -700.0, 3.0, np.nan],
        ]
    )
    y = np.array(
        [
            [1.0, 0.0, 0.0, -1.0, 0.0, 0.5, np.nan, 2.0],
            [np.nan, -2.0, 1.0, 2.0, 2.0, 2.0, 3.0, 2.0],
        ]
    )
    z = np.flip(y, axis=1).copy()
    mask = np.ones(x.shape, dtype=bool)
    mask[1, 0] = False
    args = (x, y, z) if op == "where" else (x, y) if op in BINARY else (x,)
    expected = reference.elementwise(op, *args, mask=mask)
    for backend in ("rust", "numpy"):
        actual = po.elementwise(op, *args, mask=mask, backend=backend)
        np.testing.assert_allclose(
            actual, expected, rtol=2e-10, atol=2e-10, equal_nan=True
        )
    if op in BINARY:
        np.testing.assert_allclose(
            po.elementwise(op, x, 2.0),
            reference.elementwise(op, x, 2.0),
            rtol=2e-10,
            atol=2e-10,
            equal_nan=True,
        )
    np.testing.assert_allclose(
        po.elementwise(op, *args),
        reference.elementwise(op, *args),
        rtol=2e-10,
        atol=2e-10,
        equal_nan=True,
    )
    assert np.isnan(x[0, 0]) and x[0, 1] == np.inf


def test_lagged_sort_reuse_boundaries_and_common_universe():
    rng = np.random.default_rng(92)
    x = rng.integers(-4, 5, size=(143, 97)).astype(float)
    x[rng.random(x.shape) < 0.2] = np.nan
    mask = rng.random(x.shape) > 0.15
    try:
        for lag in (0, 1, 2, 7, 32, 33, 70, 143):
            expected = reference.group_turnover(x, q=7, periods=lag, mask=mask)
            for rank in (False, True):
                shifted = reference.ts(x, "delay", periods=lag, mask=mask)
                corr = reference.ic(x, shifted, rank=rank, mask=mask)
                po.set_num_threads(1)
                a = po.factor_autocorr(x, periods=lag, rank=rank, mask=mask)
                po.set_num_threads(3)
                b = po.factor_autocorr(x, periods=lag, rank=rank, mask=mask)
                np.testing.assert_allclose(
                    a, corr, rtol=2e-10, atol=2e-10, equal_nan=True
                )
                np.testing.assert_array_equal(a.view(np.uint64), b.view(np.uint64))
            np.testing.assert_allclose(
                po.group_turnover(x, q=7, periods=lag, mask=mask),
                expected,
                rtol=2e-10,
                atol=2e-10,
                equal_nan=True,
            )
    finally:
        po.set_num_threads(0)


def test_random_long_windows_rebuilds():
    from test.panelops.test_adversarial import invoke

    coverage = {"constant": 0, "ties": 0, "missing": 0, "long": 0}
    for seed in (17, 103, 819):
        rng = np.random.default_rng(seed)
        x, y = rng.normal(size=(2, 603, 4))
        x[130:400, 0] = 0.7
        x[::7, 1] = x[::7, 2] = 0.1
        x[::13, 3] = np.nan
        mask = rng.random(x.shape) > 0.1
        for w in (2, 5, 20, 60, 250):
            for op in po.api.TS_OPS:
                with np.errstate(all="ignore"):
                    actual, expected = invoke(op, x, y, w, mask)
                np.testing.assert_allclose(
                    actual,
                    expected,
                    rtol=2e-10,
                    atol=2e-10,
                    equal_nan=True,
                    err_msg=f"{op} seed={seed} W={w}",
                )
                coverage["constant"] += int(np.all(x[130:400, 0] == 0.7))
                coverage["ties"] += int(x[0, 1] == x[0, 2])
                coverage["missing"] += int(np.isnan(x).any())
                coverage["long"] += len(x) > 600
    assert all(v > 0 for v in coverage.values())


def test_wide_single_thread_blocks_match_reference():
    rng = np.random.default_rng(504)
    x = rng.normal(size=(29, 2051))
    x[::7, ::11] = np.nan
    try:
        for op in ("mean", "std", "rank", "quantile", "min", "max"):
            po.set_num_threads(1)
            a = getattr(po, "ts_" + op)(x, 5, min_periods=3)
            po.set_num_threads(3)
            b = getattr(po, "ts_" + op)(x, 5, min_periods=3)
            np.testing.assert_array_equal(a.view(np.uint64), b.view(np.uint64))
            expected = reference.ts(x, op, 5, 3, rows=[28])
            np.testing.assert_allclose(
                a[-1], expected[-1], rtol=2e-10, atol=2e-10, equal_nan=True
            )
    finally:
        po.set_num_threads(0)


def test_scalar_operands_use_universe_shape():
    mask = np.array([[True, False, True], [False, True, False]])
    for backend in ("rust", "numpy"):
        result = po.elementwise("maximum", 2.0, 1.0, mask=mask, backend=backend)
        np.testing.assert_array_equal(result, np.where(mask, 2.0, np.nan))


def test_single_observation_product_preserves_finite_bits():
    x = np.array([[-0.0, 0.0, 0.1, 12.34, np.nextafter(0.0, 1.0), 1e308]])
    actual = po.ts_product(x, 1)
    expected = reference.ts(x, "product", window=1)
    np.testing.assert_array_equal(actual.view(np.uint64), x.view(np.uint64))
    np.testing.assert_array_equal(expected.view(np.uint64), x.view(np.uint64))


def test_fill_nan_contiguous_blocks_and_threads():
    rng = np.random.default_rng(735)
    x, y = rng.normal(size=(2, 3, 65537))
    x[:, ::3] = np.nan
    x[:, ::11] = np.inf
    y[:, ::7] = -np.inf
    y[:, ::19] = -0.0
    expected = np.where(np.isfinite(x), x, y)
    expected[~np.isfinite(expected)] = np.nan
    try:
        for count in (1, 3):
            po.set_num_threads(count)
            result = po.elementwise("fill_nan", x, y)
            np.testing.assert_array_equal(
                result.view(np.uint64), expected.view(np.uint64)
            )
            result = po.elementwise("is_nan", x)
            wanted = (~np.isfinite(x)).astype(np.float64)
            np.testing.assert_array_equal(
                result.view(np.uint64), wanted.view(np.uint64)
            )
    finally:
        po.set_num_threads(0)
