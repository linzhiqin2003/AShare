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
    if w <= SHORT:
        # Short windows are correctly rounded on the default path as well.
        assert hard == 0
    else:
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


# The incremental path after a constant window: its state is reset to the exact sum
# (rounded product plus its exact rounding error), never to the rounded product alone.


@pytest.mark.parametrize("value", (1 / 3, 0.1, 2 / 3, -7.3, 123456.789))
@pytest.mark.parametrize("w", (SHORT + 1, 5, 12, 20))
@pytest.mark.parametrize("tail", ("zeros", "small"))
@pytest.mark.parametrize("history", (False, True))
def test_cells_right_after_a_constant_window(value, w, tail, history):
    head = np.full(w + 3, value)
    if history:
        # Earlier rows leave a residue in the state; the constant window must drop it.
        noise = (
            np.random.default_rng(w).normal(0, 1, 40)
            * 10.0 ** np.arange(-3, 5)[:, None].repeat(5, 1).ravel()
        )
        head = np.concatenate([noise, head])
    after = np.zeros(w + 5) if tail == "zeros" else np.full(w + 5, 0.25)
    x = np.concatenate([head, after])[:, None]
    for op in ("sum", "mean"):
        got = call(op, x, w)
        for t in range(len(head), len(x)):
            window = x[t - w + 1 : t + 1, 0]
            want = rounded(window)
            if op == "mean":
                want /= w
            if tail == "zeros":
                assert got[t, 0] == want, (op, t, got[t, 0], want)
            else:
                # Later sums carry ordinary compensated-sum error: one ulp of the
                # largest magnitude the window passes through.
                scale = np.abs(window).sum() / (w if op == "mean" else 1)
                assert abs(got[t, 0] - want) <= np.spacing(scale), (op, t, got[t, 0])


def test_residue_does_not_reach_windows_after_an_all_zero_window():
    x = np.zeros((40, 1))
    x[:10, 0] = [171.2, 3.3, 0.1, 55.5, 12.25, 7.7, 0.33, 99.9, 1e3, 2.2]
    x[30:, 0] = 0.7
    got = po.ts_sum(x, 12)
    for t in range(30, 40):
        assert got[t, 0] == rounded(x[t - 11 : t + 1, 0]), t


@pytest.mark.parametrize("w", (5, 20))
def test_constant_detection_across_the_periodic_rebuild_row(w):
    # The equal run must advance on the row where the state is rebuilt (t % 512 == 0):
    # a stale run would call [5, 5, 5, 5, 7] constant.
    x = np.full((560, 1), 3.0)
    x[480:512, 0] = 5.0
    x[512:, 0] = 7.0
    got = po.ts_sum(x, w)
    for t in range(505, 520):
        assert got[t, 0] == rounded(x[t - w + 1 : t + 1, 0]), t


# Short windows are correctly rounded by default.


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("w", (1, 2, 3))
def test_short_default_path_is_correctly_rounded(seed, w):
    rng = np.random.default_rng(seed)
    h = 2.0**-53
    x = rng.choice(
        [1.0, -1.0, h, -h, h * h, 0.1, 0.3, -0.4, 1e16, -1e16, 3.0], (300, 6)
    )
    x = x * rng.choice([1.0, 1.0000001, 2.0**-1000, 2.0**1000], x.shape)
    x[rng.random(x.shape) < 0.1] = np.nan
    cond = (rng.random(x.shape) < 0.7).astype(float)
    for op in SUMS:
        got = call(op, x, w, cond=cond, mp=1)
        for t in range(len(x)):
            for k in range(x.shape[1]):
                assert same(got[t, k], reference(op, x, cond, w, 1, t, k)), (op, t, k)
    assert po.ts_sum(np.array([[1.0], [h], [-1.0]]), 3)[-1, 0] == h


# exact=True near the top of the float range: correctly rounded, independent of order.


def test_exact_is_correctly_rounded_when_partial_sums_overflow():
    big = np.finfo(float).max
    rng = np.random.default_rng(3)
    pool = [1e308, -1e308, big, -big, 2.0**970, 1.0, -0.5, 2.0**-1074, 1e-300, 2.0**-53]
    for _ in range(400):
        window = rng.choice(pool, 4)
        want = sum(map(Fraction, window), Fraction(0))
        limit = Fraction(2) ** 1024 - Fraction(2) ** 970
        want = (
            np.inf
            if want >= limit
            else -np.inf
            if want <= -limit
            else float(want) + 0.0
        )
        for order in (window, window[::-1], rng.permutation(window)):
            got = po.ts_sum(np.array(order)[:, None].copy(), 4, exact=True)[-1, 0]
            assert got == want, (order, got, want)


def test_exact_rebuilds_after_an_overflowing_window():
    big = np.finfo(float).max
    x = np.array([big, big, -big, 1.0, 2.0, 3.0, 4.0])[:, None]
    got = po.ts_sum(x, 2, exact=True)
    assert got[1, 0] == np.inf
    assert list(got[2:, 0]) == [0.0, -big, 3.0, 5.0, 7.0]


def test_exact_overflow_path_rounds_half_to_even():
    # Partial sums overflow, so the fixed-point path runs; the exact sum
    # 2**1023 + 2**970 is exactly halfway between two doubles and must round
    # to the even one, 2**1023.
    top, half = 2.0**1023, 2.0**970
    x = np.array([top, top, -top, half])[:, None]
    want = float(Fraction(top) + Fraction(half))
    assert want == top
    assert po.ts_sum(x, 4, exact=True)[3, 0] == want
    # Just above halfway rounds up.
    y = np.array([top, top, -top, half, 2.0**900])[:, None]
    above = float(Fraction(top) + Fraction(half) + Fraction(2.0**900))
    assert above == np.nextafter(top, np.inf)
    assert po.ts_sum(y, 5, exact=True)[4, 0] == above


# Parameter types.


@pytest.mark.parametrize("value", ("no", 0, 1, None, 1.0))
def test_exact_accepts_only_bool(value):
    x = np.ones((5, 1))
    for op in SUMS:
        with pytest.raises(TypeError, match="exact"):
            call(op, x, 3, cond=x, exact=value)
    assert po.ts_sum(x, 3, exact=np.bool_(True))[-1, 0] == 3.0
