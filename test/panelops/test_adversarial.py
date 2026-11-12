# ruff: noqa: E402
from pathlib import Path

import panelops as po

ROOT = Path(__file__).resolve().parents[2]
assert Path(po.__file__).resolve().is_relative_to(ROOT)
assert Path(po._api._core.__file__).resolve().is_relative_to(ROOT)

import math

import numpy as np
import pytest
from panelops import reference

PAIRED = {"corr", "cov", "regression_slope", "regression_residual", "sumif"}


def corpus():
    """Finite binary floats, nonbinary constants and transitions over 620 rows."""
    rng = np.random.default_rng(31415)
    t = 620
    x, y = rng.normal(size=(2, t, 12))
    x[:, 0] = 0.1
    x[:, 1] = np.repeat([12.34, 0.7, 123456.789, 1e-3], 155)
    x[:, 2] *= 1e-3
    x[[5, 100, 127, 128, 129, 250, 300], 2] = 1e12
    y[:, 2] *= 1e-3
    y[[130, 300], 2] = [-1e12, 3e11]
    x[:, 3] *= np.geomspace(1e-6, 1e9, t)
    y[:, 3] *= np.geomspace(1e9, 1e-6, t)
    x[:, 4] = rng.uniform(10, 100, t) * rng.uniform(1e8, 1e9, t)
    for j, level in enumerate([1e3, 1e4, 1e5, 3e6, 6e6, 1e7], 5):
        x[129:, j] += level
        y[129:, j] -= level
    x[:, 11] = 0.7
    y[:, 11] = 12.34
    x[::19, 1:] = np.nan
    y[::23, 1:] = np.nan
    mask = rng.random(x.shape) > 0.03
    return x, y, mask


def invoke(op, x, y, window, mask):
    kw = dict(mask=mask)
    if op in ("delay", "delta"):
        kw["periods"] = window
    elif op == "ffill":
        kw["limit"] = window
    else:
        kw.update(window=window, min_periods=max(1, window // 2))
    if op in PAIRED:
        result = getattr(po, "ts_" + op)(x, y, **kw)
    else:
        result = getattr(po, "ts_" + op)(x, **kw)
    expected = reference.ts(x, op, y=y if op in PAIRED else None, **kw)
    return result, expected


@pytest.mark.parametrize("op", po.api.TS_OPS)
@pytest.mark.parametrize("window", [2, 5, 20, 60, 250])
def test_adversarial_windows(op, window):
    x, y, mask = corpus()
    with np.errstate(all="ignore"):
        actual, expected = invoke(op, x, y, window, mask)
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=2e-10,
        atol=2e-10,
        equal_nan=True,
        err_msg=f"{op} W={window}",
    )


def test_exact_float_constants_and_piecewise_constants():
    for constant in (0.1, 0.7, 12.34, 123456.789, 1e-3):
        x = np.full((620, 3), constant)
        y = np.ascontiguousarray(np.arange(620 * 3, dtype=float).reshape(620, 3))
        for w in (2, 5, 20, 60, 250):
            np.testing.assert_array_equal(po.ts_std(x, w)[w - 1 :], 0)
            np.testing.assert_array_equal(po.ts_var(x, w)[w - 1 :], 0)
            for op in ("skew", "kurt", "zscore", "rsquare"):
                assert np.isnan(getattr(po, "ts_" + op)(x, w)).all()
            for op in ("corr", "regression_slope", "regression_residual"):
                assert np.isnan(getattr(po, "ts_" + op)(y, x, w)).all()
        for backend in ("rust", "numpy"):
            assert np.isnan(po.cs_zscore(x, backend=backend)).all()
            assert np.isnan(po.ic(x, x * 0 + 0.7, backend=backend)).all()
            np.testing.assert_array_equal(po.cs_demean(x, backend=backend), 0)


def test_cross_section_ddof_and_new_conventions():
    x = np.array([[0.1, 0.2, 0.4, np.nan]])
    for ddof in (0, 1, 2, 4):
        np.testing.assert_allclose(
            po.cs_zscore(x, ddof=ddof),
            reference.cs(x, "zscore", ddof=ddof),
            rtol=2e-10,
            atol=2e-10,
            equal_nan=True,
        )
    x = np.array([[4.0], [np.nan], [4.0], [2.0], [np.nan]])
    np.testing.assert_equal(
        po.ts_idxmax(x, 3, min_periods=1), 3 - po.ts_argmax(x, 3, min_periods=1)
    )
    np.testing.assert_equal(
        po.ts_idxmin(x, 3, min_periods=1), 3 - po.ts_argmin(x, 3, min_periods=1)
    )
    np.testing.assert_equal(
        po.ts_ffill(x, limit=1), [[4.0], [4.0], [4.0], [2.0], [2.0]]
    )
    np.testing.assert_equal(
        po.ts_count(np.array([[0.0], [np.nan], [-2.0], [3.0]]), 3, min_periods=2),
        [[np.nan], [np.nan], [1.0], [2.0]],
    )
    np.testing.assert_equal(
        po.ts_sumif(
            np.array([[1.0], [2.0], [3.0]]), np.array([[0.0], [1.0], [0.0]]), 2
        ),
        [[np.nan], [2.0], [2.0]],
    )
    np.testing.assert_equal(po.ts_mad(np.array([[1.0], [2.0], [9.0]]), 3)[-1], [1.0])
    assert po.ts_rsquare(np.arange(10.0, dtype=float)[:, None], 5)[
        -1, 0
    ] == pytest.approx(1.0)
    assert math.isnan(po.cs_scale(np.zeros((1, 2)))[0, 0])


def test_streaming_column_boundaries_and_prefix():
    rng = np.random.default_rng(89)
    x, y = rng.normal(size=(2, 620, 193))
    x[::17] = np.nan
    try:
        for op in (
            "sum",
            "mean",
            "std",
            "cov",
            "corr",
            "slope",
            "residual",
            "regression_slope",
            "regression_residual",
            "decay_linear",
            "ema",
            "skew",
            "kurt",
            "rank",
            "min",
            "max",
        ):
            args = (x, y) if op in PAIRED else (x,)
            fn = getattr(po, "ts_" + op)
            po.set_num_threads(1)
            a = fn(*args, 20, min_periods=10)
            po.set_num_threads(3)
            b = fn(*args, 20, min_periods=10)
            np.testing.assert_array_equal(a.view(np.uint64), b.view(np.uint64))
            changed = x.copy()
            changed[257:] = 1e12
            args = (changed, y) if op in PAIRED else (changed,)
            c = fn(*args, 20, min_periods=10)
            np.testing.assert_array_equal(
                a[:257].view(np.uint64), c[:257].view(np.uint64)
            )
    finally:
        po.set_num_threads(0)


def test_multiple_scales_cancel_without_losing_small_sum():
    a, b = float(2**200), float(2**100)
    x = np.tile([a, b, 1.0, -a, -b], 125)[:, None]
    y = np.ones_like(x)
    for op in ("sum", "mean", "sumif"):
        for window in (5, 20, 60, 250):
            args = (x, y) if op == "sumif" else (x,)
            actual = getattr(po, "ts_" + op)(*args, window, min_periods=1)
            expected = reference.ts(x, op, window=window, min_periods=1, y=y)
            np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-10)
    x = np.tile([10 * a, 5 * b, 1.0, -2.5 * a, -2 * b], 125)[:, None]
    expected = reference.ts(x, "decay_linear", window=5, min_periods=1)
    np.testing.assert_allclose(
        po.ts_decay_linear(x, 5, min_periods=1), expected, rtol=2e-10, atol=2e-10
    )


def test_dense_fast_path_switch_cannot_change_past_bits():
    rng = np.random.default_rng(818)
    x, y = rng.normal(size=(2, 620, 137))
    changed = x.copy()
    changed[257:] = np.nan
    mask = np.ones(x.shape, dtype=bool)
    masked = mask.copy()
    masked[257:] = False
    for op in po.api.TS_OPS:
        args = (
            dict(periods=20)
            if op in ("delay", "delta")
            else dict(limit=20)
            if op == "ffill"
            else dict(window=20, min_periods=10)
        )
        call = getattr(po, "ts_" + op)
        original = call(x, y, **args) if op in PAIRED else call(x, **args)
        future = call(changed, y, **args) if op in PAIRED else call(changed, **args)
        masked_result = (
            call(x, y, mask=masked, **args)
            if op in PAIRED
            else call(x, mask=masked, **args)
        )
        np.testing.assert_array_equal(
            original[:257].view(np.uint64), future[:257].view(np.uint64), err_msg=op
        )
        np.testing.assert_array_equal(
            original[:257].view(np.uint64),
            masked_result[:257].view(np.uint64),
            err_msg=op,
        )
