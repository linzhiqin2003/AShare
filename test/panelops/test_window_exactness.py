# ruff: noqa: E402
"""Degenerate windows and history independence of the rolling-sum family.

Expected values come from Python integers and ``fractions`` (a correctly rounded
sum is ``float(sum(map(Fraction, window)))``), never from panelops itself.
"""

from pathlib import Path

import panelops as po

ROOT = Path(__file__).resolve().parents[2]
assert Path(po._api._core.__file__).resolve().is_relative_to(ROOT)

from fractions import Fraction

import numpy as np
import pytest

SHORT = 3  # SHORT_SUM_WINDOW in crates/panelops-core/src/time.rs
SUMS = ("sum", "mean", "sumif")


def call(op, x, w, *, cond=None, exact=False, mp=None, mask=None):
    kw = dict(min_periods=mp, mask=mask)
    if op == "sumif":
        return po.ts_sumif(x, cond, w, exact=exact, **kw)
    return getattr(po, "ts_" + op)(x, w, exact=exact, **kw)


def rounded(values):
    return float(sum(map(Fraction, values), Fraction(0)))


def reference(op, x, cond, w, mp, t, k):
    lo = max(0, t - w + 1)
    vals, n = [], 0
    for r in range(lo, t + 1):
        v = x[r, k]
        if not np.isfinite(v) or (op == "sumif" and not np.isfinite(cond[r, k])):
            continue
        n += 1
        if op == "sumif" and cond[r, k] == 0:
            continue
        vals.append(v)
    if n < mp:
        return np.nan
    s = rounded(vals)
    if op == "mean":
        if len(set(vals)) == 1:
            return vals[0] + 0.0
        return s / n
    return s


def same(a, b):
    return np.array_equal(
        np.asarray(a).view(np.int64), np.asarray(b).view(np.int64)
    ) or np.array_equal(a, b, equal_nan=True)


# The two minimal reproductions reported against the incremental kernel.


def test_reproduction_all_zero_window_after_large_values():
    x = np.zeros((30, 1))
    x[:10, 0] = [171.2, 3.3, 0.1, 55.5, 12.25, 7.7, 0.33, 99.9, 1e3, 2.2]
    c = np.zeros((30, 1))
    c[:10, 0] = 1.0
    for exact in (False, True):
        for got in (
            po.ts_sum(x, 12, min_periods=12, exact=exact)[-1, 0],
            po.ts_mean(x, 12, min_periods=12, exact=exact)[-1, 0],
            po.ts_sumif(x, c, 12, min_periods=12, exact=exact)[-1, 0],
        ):
            assert got == 0.0 and not np.signbit(got)


def test_reproduction_equal_windows_in_different_order():
    x = np.array(
        [
            99.23849100000001,
            30.226922000000005,
            67.64233300000001,
            72.53838900000001,
            31.098254000000004,
            99.23849100000001,
        ]
    ).reshape(-1, 1)
    assert sum(map(Fraction, x[0:5, 0])) == sum(map(Fraction, x[1:6, 0]))
    exact = po.ts_sum(x, 5, exact=True)
    assert exact[4, 0] == exact[5, 0] == rounded(x[1:6, 0])
    # w=5 is above the short-window limit, so the default path is incremental and
    # history dependent: it still differs by one unit in the last place here.
    default = po.ts_sum(x, 5)
    assert default[4, 0] != default[5, 0]


# Degenerate windows: exact values, every path.


def degenerate_panel(seed, t=400, n=6):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (t, n)) * rng.choice([1.0, 1e6, 1e-3], (t, n))
    cond = (rng.random((t, n)) < 0.6).astype(float)
    segments = []
    for k in range(n):
        for _ in range(4):
            start = int(rng.integers(0, t - 80))
            length = int(rng.integers(5, 80))
            value = rng.choice([0.0, -0.0, 0.1, 1 / 3, -7.25, 1e9 + 0.1])
            x[start : start + length, k] = value
            segments.append((k, start, length, value))
    x[rng.random((t, n)) < 0.04] = np.nan
    cond[rng.random((t, n)) < 0.02] = np.nan
    return x, cond, segments


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("w", (2, 3, SHORT, SHORT + 1, 12, 60))
@pytest.mark.parametrize("exact", (False, True))
def test_degenerate_windows_are_exact(seed, w, exact):
    x, cond, _ = degenerate_panel(seed)
    mp = max(1, w // 2)
    checked = {"zero": 0, "constant": 0}
    results = {op: call(op, x, w, cond=cond, exact=exact, mp=mp) for op in SUMS}
    var = po.ts_var(x, w, min_periods=mp)
    std = po.ts_std(x, w, min_periods=mp)
    decay = po.ts_decay_linear(x, w, min_periods=mp)
    partner = np.sin(np.arange(len(x)))[:, None] * np.ones((1, x.shape[1])) * 3 + 1
    cov = po.ts_cov(x, partner, w, min_periods=mp)
    corr = po.ts_corr(x, partner, w, min_periods=mp)
    for t in range(len(x)):
        for k in range(x.shape[1]):
            window = x[max(0, t - w + 1) : t + 1, k]
            valid = window[np.isfinite(window)]
            if len(valid) < max(mp, 2) or len(set(valid.tolist())) != 1:
                continue
            v = valid[0] + 0.0
            checked["zero" if v == 0 else "constant"] += 1
            assert results["sum"][t, k] == len(valid) * v + 0.0
            assert not np.signbit(results["sum"][t, k]) or v < 0
            assert results["mean"][t, k] == v
            assert var[t, k] == 0.0 and std[t, k] == 0.0
            if v == 0:  # nonzero constant windows: known last-bit error, see SPEC
                assert decay[t, k] == 0.0
            assert cov[t, k] == 0.0
            assert np.isnan(corr[t, k])
    assert checked["zero"] > 0 and checked["constant"] > 0


@pytest.mark.parametrize("w", (3, SHORT, 12, 60))
@pytest.mark.parametrize("exact", (False, True))
def test_sumif_false_window_is_positive_zero(w, exact):
    rng = np.random.default_rng(w)
    x = rng.normal(1e4, 1e3, (300, 4))
    cond = np.ones_like(x)
    cond[150:220] = 0.0
    got = po.ts_sumif(x, cond, w, exact=exact)
    block = got[150 + w - 1 : 220]
    assert (block == 0).all() and not np.signbit(block).any()


# History independence.


def with_history(window_values, history):
    return np.concatenate([history, window_values]).reshape(-1, 1)


@pytest.mark.parametrize("w", (2, 3, SHORT, SHORT + 1, 20, 60, 250))
def test_exact_path_depends_only_on_window(w):
    rng = np.random.default_rng(100 + w)
    window = rng.uniform(-50, 150, w).round(6)
    seen = {op: set() for op in SUMS}
    for trial in range(60):
        history = rng.normal(0, 1, int(rng.integers(1, 700))) * 10.0 ** int(
            rng.integers(-3, 9)
        )
        order = window if trial % 2 == 0 else rng.permutation(window)
        x = with_history(order, history)
        cond = np.ones_like(x)
        for op in SUMS:
            seen[op].add(call(op, x, w, cond=cond, exact=True)[-1, 0])
    for op in SUMS:
        assert seen[op] == (
            {rounded(window)} if op != "mean" else {rounded(window) / w}
        ), op


@pytest.mark.parametrize("w", (1, 2, SHORT))
def test_short_default_path_depends_only_on_window(w):
    rng = np.random.default_rng(w)
    window = rng.uniform(-50, 150, w).round(6)
    seen = {op: set() for op in SUMS}
    for _ in range(60):
        history = rng.normal(0, 1, int(rng.integers(1, 700))) * 10.0 ** int(
            rng.integers(-3, 9)
        )
        x = with_history(window, history)
        for op in SUMS:
            seen[op].add(call(op, x, w, cond=np.ones_like(x))[-1, 0])
    assert all(len(v) == 1 for v in seen.values()), seen


def test_incremental_default_path_starts_after_short_windows():
    """Proves the boundary: the first long window still depends on history."""
    rng = np.random.default_rng(5)
    w = SHORT + 1
    window = rng.uniform(-50, 150, w).round(6)
    seen = set()
    for _ in range(200):
        history = rng.normal(0, 1, int(rng.integers(1, 400))) * 1e6
        seen.add(po.ts_sum(with_history(window, history), w)[-1, 0])
    assert len(seen) > 1


# Correct rounding of the exact path against fractions, with gaps and masks.


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("w", (1, 3, SHORT, 9, 40))
def test_exact_path_is_correctly_rounded(seed, w):
    rng = np.random.default_rng(seed)
    t, n = 160, 5
    x = (rng.normal(0, 1, (t, n)) * rng.choice([1, 1e8, 1e-8, 1e15], (t, n))).round(3)
    x[rng.random((t, n)) < 0.1] = np.nan
    x[rng.random((t, n)) < 0.02] = np.inf
    cond = (rng.random((t, n)) < 0.7).astype(float)
    mask = rng.random((t, n)) < 0.9
    mp = max(1, w // 2)
    xm = np.where(mask, x, np.nan)
    hard = 0
    for op in SUMS:
        got = call(op, x, w, cond=cond, exact=True, mp=mp, mask=mask)
        default = call(op, x, w, cond=cond, mp=mp, mask=mask)
        assert np.array_equal(np.isnan(got), np.isnan(default))
        for r in range(t):
            for k in range(n):
                want = reference(op, xm, cond, w, mp, r, k)
                if not mask[r, k]:
                    want = np.nan
                assert same(got[r, k], want), (op, r, k, got[r, k], want)
                hard += not same(default[r, k], want)
    if w > 1:
        # The data defeat plain float sums somewhere, so the test can tell paths apart.
        assert hard > 0


def test_exact_overflow_falls_back_without_poisoning_later_windows():
    x = np.array([1e308, 1e308, -1e308, 1.0, 2.0, 3.0]).reshape(-1, 1)
    got = po.ts_sum(x, 2, exact=True)
    assert np.isinf(got[1, 0])
    assert list(got[2:, 0]) == [0.0, -1e308 + 1.0, 3.0, 5.0]


# Threads, NaN pattern and min_periods.


@pytest.mark.parametrize("w", (3, SHORT, 20, 250))
@pytest.mark.parametrize("exact", (False, True))
def test_parallel_matches_serial(w, exact):
    rng = np.random.default_rng(w)
    x = rng.normal(0, 1, (600, 2100)) * 100
    x[rng.random(x.shape) < 0.05] = np.nan
    cond = (rng.random(x.shape) < 0.5).astype(float)
    try:
        po.set_num_threads(1)
        serial = {op: call(op, x, w, cond=cond, exact=exact) for op in SUMS}
        po.set_num_threads(0)
        parallel = {op: call(op, x, w, cond=cond, exact=exact) for op in SUMS}
    finally:
        po.set_num_threads(0)
    for op in SUMS:
        assert same(serial[op], parallel[op]), op


@pytest.mark.parametrize("w", (2, SHORT, 30))
def test_nan_pattern_and_min_periods_match_count(w):
    rng = np.random.default_rng(w)
    x = rng.normal(0, 1, (200, 3))
    x[rng.random(x.shape) < 0.3] = np.nan
    for mp in (1, w):
        count = po.ts_count(np.where(np.isfinite(x), 1.0, np.nan), w, min_periods=1)
        expected_nan = ~(np.nan_to_num(count) >= mp)
        for exact in (False, True):
            for op in ("sum", "mean"):
                got = call(op, x, w, exact=exact, mp=mp)
                assert (np.isnan(got) == expected_nan).all(), (op, exact, mp)


@pytest.mark.parametrize("op", SUMS)
def test_short_default_path_survives_cancellation(op):
    """Three-row windows are summed directly; cancellation must fall back to exact."""
    rng = np.random.default_rng(11)
    big = np.array([1e16, 1.0, -1e16, 2.0**200, 3.0, -(2.0**200), 1e300, -1e300, 0.5])
    x = np.concatenate([big, rng.normal(0, 1, 30) * 1e15, big[::-1]])[:, None]
    cond = np.ones_like(x)
    got = call(op, x, SHORT, cond=cond, mp=1)
    for t in range(len(x)):
        window = x[max(0, t - SHORT + 1) : t + 1, 0]
        exact = sum(map(Fraction, window), Fraction(0))
        if op == "mean":
            exact /= len(window)
        want = float(exact)
        assert abs(got[t, 0] - want) <= 2e-10 * (1 + abs(want)), (t, got[t, 0], want)
