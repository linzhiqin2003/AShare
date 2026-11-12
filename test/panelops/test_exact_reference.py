# ruff: noqa: E402
"""Exact rational reference for the streaming sum, decay and moment kernels.

Every finite float is an integer multiple of a power of two, so each window's power
sums are exact Python integers taken from prefix sums. Each statistic is formed
from exact integers and rounded once; a square root is taken of a once-rounded
square. Neither the native kernels nor ``panelops.reference`` (a floating two-pass
reference that shares their cancellation limits) supply expected values here.
"""

from functools import cache
from pathlib import Path

import panelops as po

ROOT = Path(__file__).resolve().parents[2]
assert Path(po._api._core.__file__).resolve().is_relative_to(ROOT)

import math
from fractions import Fraction

import numpy as np
import pytest

T = 620
WINDOWS = (2, 5, 20, 60, 250)
PAIRED = {"cov", "corr", "regression_slope", "regression_residual", "sumif"}
EXACT_OPS = (
    "sum",
    "mean",
    "sumif",
    "decay_linear",
    "std",
    "var",
    "zscore",
    "corr",
    "cov",
    "skew",
    "kurt",
    "slope",
    "residual",
    "regression_slope",
    "regression_residual",
    "rsquare",
)
# Covariance and regression outputs inherit the conditioning of the window: with a
# 1e12 observation beside 1e-3 neighbours, a double result carries an absolute error
# of a few eps times the window's largest centered magnitude. SPEC states this
# window-scaled bound for these five operators. It is admitted only in the columns
# built for extreme dynamic range, and there only while such a value is inside the
# window: every other cell must meet the plain tolerance.
WINDOW_SCALED = {"cov", "slope", "residual", "regression_slope", "regression_residual"}
WIDE_RANGE_COLUMNS = (3, 4, 12, 14)
SCALED_BOUND = 1e-12


def corpus():
    """Columns chosen to trigger sum events, expansions and moment rebuilds."""
    rng = np.random.default_rng(271828)
    cols = []
    cols.append(rng.normal(size=T))
    cols.append(np.full(T, 0.1))
    cols.append(np.repeat([12.34, 0.7, 123456.789, 1e-3], T // 4 + 1)[:T])
    s = 1e-3 * rng.normal(size=T)
    s[[5, 100, 127, 128, 129, 250, 300, 511, 512, 513]] = 1e12
    cols.append(s)
    s = np.full(T, 0.1)
    s[:50] = rng.normal(size=50)
    s[[50, 51, 300, 520]] = 1e12
    cols.append(s)
    s = 1e-3 * rng.normal(size=T)
    s[:310] = 1e15 * (1 + 1e-3 * rng.normal(size=310))
    cols.append(s)
    cols.append(rng.normal(size=T) * np.geomspace(1e-6, 1e9, T))
    cols.append(rng.uniform(10, 100, T) * rng.uniform(1e8, 1e9, T))
    for level in (1e3, 1e5, 3e6, 1e7):
        s = rng.normal(size=T)
        s[129:] += level
        s[400:] -= level / 2
        cols.append(s)
    a, b = float(2**200), float(2**100)
    cols.append(np.tile([a, b, 1.0, -a, -b], T // 5 + 1)[:T])
    s = np.full(T, 0.1)
    s[:200] = 1e6
    cols.append(s)
    cols.append(np.where(rng.random(T) < 0.5, 1e8, 1e-8) * rng.normal(size=T))
    s = rng.normal(size=T)
    s[200:260] = 3.3
    s[260:] = 3.3 + 1e-12 * rng.normal(size=T - 260)
    cols.append(s)
    cols.append(rng.integers(-3, 4, T).astype(float))
    # Level steps of 25-45 noise units drift the window mean far from the moment
    # anchor between rebuilds; the higher-moment cancellation guard must catch it.
    stairs = np.random.default_rng(12)
    cols.append(25 * (np.arange(T) // 23) + stairs.normal(size=T))
    cols.append(45 * (np.arange(T) // 17) + stairs.normal(size=T))
    x = np.ascontiguousarray(np.column_stack(cols))
    y = rng.normal(size=x.shape)
    y[:, 0] = 7.7
    y[:, 3] *= 1e-3
    y[[130, 300], 3] = [-1e12, 3e11]
    y[:, 1] = 12.34
    y[:, 16] = rng.integers(0, 2, T)
    y[:, 12] = (rng.random(T) < 0.5).astype(float)
    for j in range(2, x.shape[1], 3):
        x[::19, j] = np.nan
    y[::23, 2:] = np.nan
    mask = rng.random(x.shape) > 0.03
    return x, np.ascontiguousarray(y), mask


def scaled(values, valid):
    """Exact integers m and exponent k with value = m / 2**k (0 where invalid)."""
    ratios = [
        v.as_integer_ratio() if ok else (0, 1)
        for v, ok in zip(values, valid, strict=True)
    ]
    k = max(d.bit_length() - 1 for _, d in ratios)
    return np.array([n << (k - d.bit_length() + 1) for n, d in ratios], object), k


def windowed(a, window):
    """Exact sums over rows max(0, t-window+1)..t of an object integer array."""
    prefix = np.concatenate([np.array([0], object), np.cumsum(a)])
    rows = np.arange(len(a))
    return prefix[rows + 1] - prefix[np.maximum(0, rows - window + 1)]


def signed_root(numerator, denominator, factor=1):
    """sign(numerator) * sqrt(factor * numerator**2 / denominator) for integers.

    The square is one correctly rounded integer division, so the result is within
    about one unit in the last place of the exact value.
    """
    square = factor * numerator * numerator / denominator
    return math.copysign(math.sqrt(square), numerator)


@cache
def exact(window):
    return exact_panel(*corpus(), window)


def exact_panel(x, y, mask, window):
    """Expected values of every EXACT_OPS operator, min_periods=max(1, window//2)."""
    T = len(x)
    mp = max(1, window // 2)
    out = {op: np.full(x.shape, np.nan) for op in EXACT_OPS}
    rows = np.arange(T, dtype=object)
    for j in range(x.shape[1]):
        single = np.isfinite(x[:, j]) & mask[:, j]
        paired = single & np.isfinite(y[:, j])
        for kind, valid in (("single", single), ("paired", paired)):
            xs, kx = scaled(x[:, j], valid)
            if kind == "single":
                ys, ky = np.where(valid, rows, 0), 0
            else:
                ys, ky = scaled(y[:, j], valid)
            flags = valid.astype(object)
            n = windowed(flags, window)
            s1, s2 = windowed(xs, window), windowed(xs * xs, window)
            if kind == "single":
                s3, s4 = windowed(xs**3, window), windowed(xs**4, window)
            sy, syy = windowed(ys, window), windowed(ys * ys, window)
            sxy = windowed(xs * ys, window)
            positions = windowed(np.where(valid, rows, 0), window)
            weighted = windowed(xs * rows, window)
            chosen = windowed(np.where(y[:, j] != 0, xs, 0), window)
            for t in range(T):
                c = int(n[t])
                if not mask[t, j] or c < mp:
                    continue
                today = bool(valid[t])
                a1, a2 = s1[t], c * s2[t] - s1[t] ** 2
                ay = c * syy[t] - sy[t] ** 2
                cross = c * sxy[t] - s1[t] * sy[t]
                if kind == "paired":
                    out["sumif"][t, j] = chosen[t] / (1 << kx)
                else:
                    out["sum"][t, j] = a1 / (1 << kx)
                    out["mean"][t, j] = a1 / (c << kx)
                    shift = t - window
                    out["decay_linear"][t, j] = (weighted[t] - shift * a1) / (
                        (positions[t] - shift * c) << kx
                    )
                    if c > 1:
                        out["var"][t, j] = a2 / ((c * (c - 1)) << (2 * kx))
                        out["std"][t, j] = math.sqrt(out["var"][t, j])
                        if today and a2 > 0:
                            out["zscore"][t, j] = signed_root(
                                c * xs[t] - a1, c * a2, c - 1
                            )
                    if c >= 3 and a2 > 0:
                        a3 = c * c * s3[t] - 3 * c * a1 * s2[t] + 2 * a1**3
                        out["skew"][t, j] = signed_root(a3, a2**3)
                    if c >= 4 and a2 > 0:
                        a4 = (
                            c**3 * s4[t]
                            - 4 * c * c * a1 * s3[t]
                            + 6 * c * a1 * a1 * s2[t]
                            - 3 * a1**4
                        )
                        out["kurt"][t, j] = (a4 - 3 * a2 * a2) / (a2 * a2)
                if kind == "paired":
                    slope_name, residual_name = (
                        "regression_slope",
                        "regression_residual",
                    )
                    if c > 1:
                        out["cov"][t, j] = cross / ((c * (c - 1)) << (kx + ky))
                else:
                    slope_name, residual_name = "slope", "residual"
                if c >= 2 and ay > 0:
                    out[slope_name][t, j] = (cross << ky) / (ay << kx)
                    if today:
                        numerator = (c * xs[t] - a1) * ay - cross * (c * ys[t] - sy[t])
                        out[residual_name][t, j] = numerator / ((c * ay) << kx)
                    if a2 > 0:
                        if kind == "paired":
                            out["corr"][t, j] = signed_root(cross, a2 * ay)
                        else:
                            out["rsquare"][t, j] = cross * cross / (a2 * ay)
    return out


def actual(op, window):
    x, y, mask = corpus()
    fn = getattr(po, "ts_" + op)
    args = (x, y) if op in PAIRED else (x,)
    with np.errstate(all="ignore"):
        return fn(*args, window, min_periods=max(1, window // 2), mask=mask)


def window_scale(op, window):
    """SPEC's scale M per cell: centered magnitudes of the regression's observations.

    cov: max|x-mean x| * max|p-mean p|; slopes: max|x-mean x| / std(p);
    residuals: max|x-mean x|. p is y for paired operators and the row position
    otherwise. Also returns whether the window holds a value of magnitude >= 1e6.
    """
    x, y, mask = corpus()
    valid = np.isfinite(x) & mask
    if op in PAIRED:
        valid &= np.isfinite(y)
    rows = np.broadcast_to(np.arange(T, dtype=float)[:, None], x.shape)
    p = y if op in PAIRED else rows
    out, large = np.zeros(x.shape), np.zeros(x.shape, bool)
    for t in range(T):
        part = slice(max(0, t - window + 1), t + 1)
        ok = valid[part]
        for j in range(x.shape[1]):
            v, q = x[part, j][ok[:, j]], p[part, j][ok[:, j]]
            if not len(v):
                continue
            mx = np.abs(v - v.mean()).max()
            large[t, j] = np.abs(v).max() >= 1e6
            if op == "cov":
                out[t, j] = mx * np.abs(q - q.mean()).max()
            elif op.endswith("slope"):
                out[t, j] = mx / q.std() if q.std() > 0 else 0.0
            else:
                out[t, j] = mx
    return out, large


@pytest.mark.parametrize("op", EXACT_OPS)
def test_exact_reference_adversarial_windows(op):
    """sum/mean/sumif event rebuilds, decay and moment guards against exact values."""
    for window in WINDOWS:
        expected, result = exact(window)[op], actual(op, window)
        np.testing.assert_array_equal(
            np.isnan(result), np.isnan(expected), err_msg=f"{op} W={window} NaN"
        )
        bound = 2e-10 + 2e-10 * np.abs(np.nan_to_num(expected))
        if op in WINDOW_SCALED:
            scale, large = window_scale(op, window)
            wide = np.zeros(scale.shape, bool)
            wide[:, list(WIDE_RANGE_COLUMNS)] = True
            bound = np.where(wide & large, bound + SCALED_BOUND * scale, bound)
        error = np.abs(np.nan_to_num(result - expected))
        worst = np.unravel_index(np.argmax(error / bound), error.shape)
        assert (error <= bound).all(), (
            f"{op} W={window}: {(error > bound).sum()} cells, worst {worst} "
            f"got {result[worst]!r} expected {expected[worst]!r}"
        )


def rounding_up_series(rows, start, magnitude, candidates, square=False):
    """W=2 series whose plain running sum (or sum of squares) rounds up each step.

    Additions of two values on one grid land on a grid twice as coarse; a tie rounds
    to even, so choosing the low bits fixes the direction. Every candidate is exact.
    """
    x, total = [start], start * start if square else start
    for t in range(1, rows):
        if t >= 2:
            total -= x[t - 2] * x[t - 2] if square else x[t - 2]
        options = list(candidates(t))
        chosen = next(
            (
                c
                for c in options
                if Fraction(total + (c * c if square else c))
                > Fraction(total) + Fraction(c * c if square else c)
            ),
            options[0],  # the first addition to an exact zero cannot round
        )
        x.append(chosen)
        total += chosen * chosen if square else chosen
    return x


def test_sum_compensation_needed_between_rebuilds():
    """Without compensated updates, 509 upward ties accumulate 2.3e-10 below the
    event threshold (peak 5686 < 5742 for a value of 0.02) before the 512-row rebuild.
    """
    step = 2.0**-41
    x = rounding_up_series(
        510, 2843.0, None, lambda t: [2843.0 + j * step for j in range(1, 200)]
    )
    x = np.array([*x, 0.01, 0.01])[:, None]
    for op, args in (("sum", (x,)), ("sumif", (x, np.ones_like(x)))):
        result = getattr(po, "ts_" + op)(*args, 2, min_periods=1)[-1, 0]
        assert abs(result - 0.02) <= 2e-10 * 1.02, (op, result)


def test_variance_bound_counts_updates_since_rebuild():
    """Plain two-moment updates of +/-A drift by one tie per row; with the final pair
    centred at 0 the drift is 2.7e-10 of the variance. A bound that ignored the update
    count since the last rebuild would accept it; the counted bound rebuilds.
    """
    rng = np.random.default_rng(7)
    a = 2.0**20 * (1 + 2.0**-30)

    def candidates(t):
        sign = 1.0 if t % 2 else -1.0
        return sign * (a + rng.integers(0, 2**27, size=64) * 2.0**-32)

    x = rounding_up_series(510, 0.0, None, candidates, square=True)
    e = 0.0145 * a
    x = np.array([*x, -e, e])[:, None]
    exact = float((Fraction(e) - Fraction(-e)) ** 2 / 2)
    result = po.ts_var(x, 2)[-1, 0]
    assert abs(result - exact) <= 2e-10 * (1 + exact), result
