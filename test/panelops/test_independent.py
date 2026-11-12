# ruff: noqa: E402
"""Checks that do not consult ``panelops.reference``.

``reference.py`` is the executable specification most tests compare against, so a
definition changed in both the kernel and the reference would pass them. Here
expected values are hand-calculated fractions or small loops written directly from
SPEC.md in plain Python.
"""

from pathlib import Path

import panelops as po

ROOT = Path(__file__).resolve().parents[2]
assert Path(po._api._core.__file__).resolve().is_relative_to(ROOT)

import math
from fractions import Fraction

import numpy as np
import pytest

TOL = dict(rtol=2e-10, atol=2e-10, equal_nan=True)


# Hand-calculated moments of x = 1, 2, 4, 8 (n = 4, mean 15/4):
# deviations -11/4, -7/4, 1/4, 17/4; m2 = 115/16, m3 = 405/32, m4 = 25141/256.
M2, M3, M4 = Fraction(115, 16), Fraction(405, 32), Fraction(25141, 256)
BIASED_SKEW = float(M3) / float(M2) ** 1.5  # g1 = m3 / m2^(3/2)
BIASED_KURT = float(M4 / M2**2 - 3)  # g2 = m4 / m2^2 - 3 = -14534/13225
# Unbiased (adjusted Fisher-Pearson) definitions used by pandas/Excel, for contrast.
UNBIASED_SKEW = math.sqrt(4 * 3) / 2 * BIASED_SKEW
UNBIASED_KURT = float((5 * (M4 / M2**2 - 3) + 6) * 3 / (2 * 1))


def test_skew_kurt_hand_values_are_biased_population_moments():
    assert BIASED_KURT == -14534 / 13225
    assert abs(BIASED_SKEW - 0.6568077344996993) < 1e-15  # 40-digit decimal value
    x = np.array([[1.0], [2.0], [4.0], [8.0]])
    for scale, shift in [(1.0, 0.0), (-1.0, 0.0), (2.0**-10, 1e6), (2.0**40, -7.0)]:
        z = np.ascontiguousarray(x * scale + shift)
        sign = math.copysign(1.0, scale)
        skew = po.ts_skew(z, 4)[-1, 0]
        kurt = po.ts_kurt(z, 4)[-1, 0]
        assert skew == pytest.approx(sign * BIASED_SKEW, rel=1e-12, abs=1e-12)
        assert kurt == pytest.approx(BIASED_KURT, rel=1e-12, abs=1e-12)
        assert abs(skew - sign * UNBIASED_SKEW) > 0.1
        assert abs(kurt - UNBIASED_KURT) > 0.1
    # A missing observation is skipped, not zero-filled: windows of 1, 2, NaN, 4, 8.
    y = np.array([[1.0], [2.0], [np.nan], [4.0], [8.0]])
    assert po.ts_kurt(y, 5, min_periods=4)[-1, 0] == pytest.approx(BIASED_KURT)
    assert po.ts_skew(y, 5, min_periods=4)[-1, 0] == pytest.approx(BIASED_SKEW)


def fraction_moments(values):
    v = [Fraction(a) for a in values]
    n = len(v)
    m = sum(v) / n
    m2, m3, m4 = (sum((a - m) ** k for a in v) / n for k in (2, 3, 4))
    skew = float(m3) / float(m2) ** 1.5 if n >= 3 and m2 > 0 else math.nan
    kurt = float(m4 / m2**2 - 3) if n >= 4 and m2 > 0 else math.nan
    return skew, kurt


def test_skew_kurt_rolling_fractions_with_ties_and_gaps():
    rng = np.random.default_rng(17)
    x = rng.integers(-4, 5, size=(80, 4)).astype(float)
    x[rng.random(x.shape) < 0.15] = np.nan
    x[30:40, 2] = 3.0
    for window in (4, 7, 30):
        skew = po.ts_skew(x, window, min_periods=3)
        kurt = po.ts_kurt(x, window, min_periods=3)
        for t in range(len(x)):
            for j in range(x.shape[1]):
                part = x[max(0, t - window + 1) : t + 1, j]
                part = part[np.isfinite(part)]
                s, k = fraction_moments(part) if len(part) >= 3 else (math.nan,) * 2
                np.testing.assert_allclose(skew[t, j], s, **TOL)
                np.testing.assert_allclose(kurt[t, j], k, **TOL)


def small_panel(seed=3, t=40, n=6):
    rng = np.random.default_rng(seed)
    x = rng.integers(-3, 4, size=(t, n)).astype(float)
    x[rng.random(x.shape) < 0.2] = np.nan
    x[:, 0] = np.nan
    x[5:20, 1] = 2.0
    mask = rng.random(x.shape) > 0.15
    return x, mask


def spec_ts(op, x, mask, window, mp, q=0.5, normalize=False):
    """Direct per-cell transcription of SPEC.md for order and recursive operators."""
    t_len, n = x.shape
    ok = np.isfinite(x) & mask
    out = np.full(x.shape, math.nan)
    for j in range(n):
        state, seen, last = math.nan, 0, None
        for t in range(t_len):
            if op in ("ema", "ffill"):
                if ok[t, j]:
                    if op == "ffill" or not seen:
                        state = x[t, j]
                    else:
                        alpha = 2 / (window + 1)
                        state = alpha * x[t, j] + (1 - alpha) * state
                    seen, last = seen + 1, t
                if mask[t, j] and seen >= (mp if op == "ema" else 1):
                    if op == "ema" or window is None or t - last <= window:
                        out[t, j] = state
                continue
            if op in ("delay", "delta"):
                s = t - window
                if mask[t, j] and s >= 0 and ok[s, j] and (op == "delay" or ok[t, j]):
                    out[t, j] = x[s, j] if op == "delay" else x[t, j] - x[s, j]
                continue
            rows = [r for r in range(max(0, t - window + 1), t + 1) if ok[r, j]]
            v = [x[r, j] for r in rows]
            if not mask[t, j] or len(v) < mp:
                continue
            if op == "count":
                out[t, j] = sum(a != 0 for a in v)
            elif op in ("min", "max"):
                out[t, j] = min(v) if op == "min" else max(v)
            elif op in ("argmin", "argmax", "idxmin", "idxmax"):
                best = min(v) if op in ("argmin", "idxmin") else max(v)
                age = t - max(r for r in rows if x[r, j] == best)
                out[t, j] = age if op.startswith("arg") else window - age
            elif op == "product":
                out[t, j] = math.prod(v)
            elif op in ("quantile", "median"):
                out[t, j] = linear_quantile(v, q if op == "quantile" else 0.5)
            elif op == "mad":
                centre = linear_quantile(v, 0.5)
                out[t, j] = linear_quantile([abs(a - centre) for a in v], 0.5)
            elif op == "rank":
                if not ok[t, j]:
                    continue
                today = x[t, j]
                less = sum(a < today for a in v)
                equal = sum(a == today for a in v)
                rank = less + (equal + 1) / 2
                if normalize:
                    rank = 0.5 if len(v) == 1 else (rank - 1) / (len(v) - 1)
                out[t, j] = rank
    return out


def linear_quantile(values, q):
    s = sorted(values)
    at = (len(s) - 1) * q
    lo, hi = math.floor(at), math.ceil(at)
    return s[lo] + (s[hi] - s[lo]) * (at - lo)


@pytest.mark.parametrize(
    "op",
    "delay delta min max rank quantile argmin argmax idxmin idxmax product ema "
    "median mad count ffill".split(),
)
def test_order_and_recursive_operators_follow_spec_text(op):
    x, mask = small_panel()
    for window in (1, 3, 7, 50):
        for mp in sorted({1, max(1, window // 2), window}):
            if op in ("delay", "delta"):
                got = getattr(po, "ts_" + op)(x, window, mask=mask)
                want = spec_ts(op, x, mask, window, 1)
            elif op == "ffill":
                got = po.ts_ffill(x, limit=window, mask=mask)
                want = spec_ts(op, x, mask, window, 1)
            elif op in ("rank", "quantile"):
                for extra in [False, True] if op == "rank" else [0.0, 0.3, 1.0]:
                    kw = dict(normalize=extra) if op == "rank" else dict(q=extra)
                    got = getattr(po, "ts_" + op)(
                        x, window, min_periods=mp, mask=mask, **kw
                    )
                    want = spec_ts(op, x, mask, window, mp, **kw)
                    np.testing.assert_allclose(got, want, **TOL, err_msg=f"{kw}")
                continue
            else:
                got = getattr(po, "ts_" + op)(x, window, min_periods=mp, mask=mask)
                want = spec_ts(op, x, mask, window, mp)
            np.testing.assert_allclose(got, want, **TOL, err_msg=f"W={window} mp={mp}")
    unlimited = po.ts_ffill(x, mask=mask)
    np.testing.assert_allclose(unlimited, spec_ts("ffill", x, mask, None, 1), **TOL)


def average_ranks(v):
    return [sum(b < a for b in v) + (sum(b == a for b in v) + 1) / 2 for a in v]


def pearson(a, b):
    a, b = [Fraction(v) for v in a], [Fraction(v) for v in b]
    n = len(a)
    if n < 2:
        return math.nan
    ma, mb = sum(a) / n, sum(b) / n
    sab = sum((p - ma) * (q - mb) for p, q in zip(a, b, strict=True))
    saa, sbb = sum((p - ma) ** 2 for p in a), sum((q - mb) ** 2 for q in b)
    if saa == 0 or sbb == 0:
        return math.nan
    return math.copysign(math.sqrt(float(sab * sab / (saa * sbb))), sab)


def test_cross_sections_follow_spec_text():
    x, mask = small_panel(seed=8, t=12, n=9)
    x[:, 0] = np.arange(12.0)
    groups = np.random.default_rng(1).integers(-1, 3, size=x.shape, dtype=np.int64)
    for t in range(len(x)):
        ids = [j for j in range(x.shape[1]) if mask[t, j] and np.isfinite(x[t, j])]
        v = [x[t, j] for j in ids]
        if not v:
            continue
        expected = {}
        ranks = average_ranks(v)
        expected["rank"] = ranks
        n = len(v)
        mean = Fraction(sum(Fraction(a) for a in v), n)
        dev = [Fraction(a) - mean for a in v]
        var = sum(d * d for d in dev) / n
        expected["demean"] = [float(d) for d in dev]
        expected["zscore"] = (
            [float(d) / math.sqrt(float(var)) for d in dev]
            if var > 0
            else [math.nan] * n
        )
        absolute = sum(abs(a) for a in v)
        expected["scale"] = [a / absolute if absolute else math.nan for a in v]
        lo, hi = linear_quantile(v, 0.01), linear_quantile(v, 0.99)
        expected["winsor_quantile"] = [min(hi, max(lo, a)) for a in v]
        med = linear_quantile(v, 0.5)
        mad = linear_quantile([abs(a - med) for a in v], 0.5)
        expected["winsor_mad"] = [min(med + 3 * mad, max(med - 3 * mad, a)) for a in v]
        for op, values in expected.items():
            row = getattr(po, "cs_" + op)(x, mask=mask)[t, ids]
            np.testing.assert_allclose(row, values, **TOL, err_msg=f"{op} t={t}")
        norm = po.cs_rank(x, mask=mask, normalize=True)[t, ids]
        want = [0.5] if n == 1 else [(r - 1) / (n - 1) for r in ranks]
        np.testing.assert_allclose(norm, want, **TOL)
        members = [j for j in ids if groups[t, j] >= 0]
        demeaned = po.cs_group_demean(x, groups=groups, mask=mask)[t]
        for j in members:
            same = [x[t, k] for k in members if groups[t, k] == groups[t, j]]
            assert demeaned[j] == pytest.approx(x[t, j] - sum(same) / len(same))
        assert np.isnan(demeaned[[j for j in ids if groups[t, j] < 0]]).all()


def test_neutralize_matches_fraction_normal_equations():
    rng = np.random.default_rng(12)
    x = rng.integers(-5, 6, size=(6, 14)).astype(float)
    groups = rng.integers(0, 3, size=x.shape, dtype=np.int64)
    exposure = rng.integers(-3, 4, size=x.shape).astype(float)
    weights = rng.integers(1, 4, size=x.shape).astype(float)
    got = po.cs_neutralize(x, groups=groups, exposures=[exposure], weights=weights)
    for t in range(len(x)):
        labels = sorted(set(groups[t]))
        # Intercept plus all but the first group indicator spans the same column space.
        design = [
            [Fraction(1)]
            + [Fraction(int(groups[t, j] == g)) for g in labels[1:]]
            + [Fraction(exposure[t, j])]
            for j in range(x.shape[1])
        ]
        w = [Fraction(v) for v in weights[t]]
        k = len(design[0])
        a = [
            [
                sum(w[i] * design[i][p] * design[i][q] for i in range(len(w)))
                for q in range(k)
            ]
            for p in range(k)
        ]
        b = [
            sum(w[i] * design[i][p] * Fraction(x[t, i]) for i in range(len(w)))
            for p in range(k)
        ]
        beta = solve(a, b)
        want = [
            float(
                Fraction(x[t, i])
                - sum(c * d for c, d in zip(beta, design[i], strict=True))
            )
            for i in range(len(w))
        ]
        np.testing.assert_allclose(got[t], want, rtol=2e-10, atol=2e-10)


def solve(a, b):
    """Exact Gauss-Jordan elimination on Fractions."""
    m = [row[:] + [v] for row, v in zip(a, b, strict=True)]
    k = len(m)
    for c in range(k):
        p = next(r for r in range(c, k) if m[r][c] != 0)
        m[c], m[p] = m[p], m[c]
        m[c] = [v / m[c][c] for v in m[c]]
        for r in range(k):
            if r != c and m[r][c] != 0:
                m[r] = [u - m[r][c] * v for u, v in zip(m[r], m[c], strict=True)]
    return [row[-1] for row in m]


def test_evaluation_follows_spec_text():
    rng = np.random.default_rng(21)
    x = rng.integers(-3, 4, size=(10, 11)).astype(float)
    r = rng.integers(-5, 6, size=x.shape).astype(float)
    x[rng.random(x.shape) < 0.2] = np.nan
    r[rng.random(r.shape) < 0.2] = np.nan
    ic, rank_ic = po.ic(x, r), po.rank_ic(x, r)
    autocorr = po.factor_autocorr(x)
    groups = po.group_returns(x, r, q=3)
    turnover = po.group_turnover(x, q=3)
    target = po.targets(x, fraction=0.3)
    for t in range(len(x)):
        both = [
            j
            for j in range(x.shape[1])
            if np.isfinite(x[t, j]) and np.isfinite(r[t, j])
        ]
        a, b = [x[t, j] for j in both], [r[t, j] for j in both]
        np.testing.assert_allclose(ic[t], pearson(a, b), **TOL)
        np.testing.assert_allclose(
            rank_ic[t], pearson(average_ranks(a), average_ranks(b)), **TOL
        )
        if t >= 1:
            common = [
                j
                for j in range(x.shape[1])
                if np.isfinite(x[t, j]) and np.isfinite(x[t - 1, j])
            ]
            want = pearson(
                average_ranks([x[t, j] for j in common]),
                average_ranks([x[t - 1, j] for j in common]),
            )
            np.testing.assert_allclose(autocorr[t], want, **TOL)
        else:
            assert np.isnan(autocorr[t])
        labels = ordinal_groups(x[t], 3)
        for g in range(3):
            rets = [
                r[t, j]
                for j, lab in labels.items()
                if lab == g and np.isfinite(r[t, j])
            ]
            want = sum(rets) / len(rets) if rets else math.nan
            np.testing.assert_allclose(groups[t, g], want, **TOL)
            today = {j for j, lab in labels.items() if lab == g}
            if t == 0 or not today:
                assert np.isnan(turnover[t, g])
            else:
                past = {j for j, lab in ordinal_groups(x[t - 1], 3).items() if lab == g}
                assert turnover[t, g] == pytest.approx(
                    1 - len(today & past) / len(today)
                )
        finite = [j for j in range(x.shape[1]) if np.isfinite(x[t, j])]
        count = math.ceil(len(finite) * 0.3)
        chosen = sorted(sorted(finite, key=lambda j: (-x[t, j], j))[:count])
        cols = target.cols[target.indptr[t] : target.indptr[t + 1]]
        weights = target.weights[target.indptr[t] : target.indptr[t + 1]]
        assert cols.tolist() == chosen
        np.testing.assert_allclose(weights, [1 / count] * count if count else [], **TOL)


def ordinal_groups(row, q):
    finite = sorted((v, j) for j, v in enumerate(row) if np.isfinite(v))
    return {j: i * q // len(finite) for i, (_, j) in enumerate(finite)}


ELEMENT_CASES = [
    # (op, args, expected) with missing operands written as nan.
    ("gt", (2.0, 1.0), 1.0),
    ("gt", (1.0, 1.0), 0.0),
    ("ge", (1.0, 1.0), 1.0),
    ("lt", (-1.0, 1.0), 1.0),
    ("le", (2.0, 1.0), 0.0),
    ("eq", (0.0, -0.0), 1.0),
    ("ne", (1.0, math.nan), math.nan),
    ("and_", (2.0, -3.0), 1.0),
    ("and_", (2.0, 0.0), 0.0),
    ("or_", (0.0, 0.0), 0.0),
    ("or_", (0.0, -1e-300), 1.0),
    ("not_", (0.0,), 1.0),
    ("not_", (math.inf,), math.nan),
    ("where", (1.0, 5.0, math.nan), 5.0),
    ("where", (0.0, math.nan, 6.0), 6.0),
    ("where", (math.nan, 5.0, 6.0), math.nan),
    ("maximum", (3.0, -4.0), 3.0),
    ("minimum", (3.0, math.nan), math.nan),
    ("sign", (-0.5,), -1.0),
    ("sign", (0.0,), 0.0),
    ("power", (2.0, 10.0), 1024.0),
    ("power", (-8.0, 1 / 3), math.nan),
    ("power", (10.0, 400.0), math.nan),
    ("signed_power", (-4.0, 0.5), -2.0),
    ("signed_power", (0.0, -1.0), math.nan),
    ("sqrt", (9.0,), 3.0),
    ("sqrt", (-1.0,), math.nan),
    ("exp", (0.0,), 1.0),
    ("exp", (1000.0,), math.nan),
    ("clip", (5.0,), 1.0),
    ("clip", (-0.25,), -0.25),
    ("is_nan", (math.inf,), 1.0),
    ("is_nan", (0.0,), 0.0),
    ("fill_nan", (math.nan, 7.0), 7.0),
    ("fill_nan", (2.0, 7.0), 2.0),
]


@pytest.mark.parametrize("backend", ["rust", "numpy"])
def test_elementwise_literal_values(backend):
    for op, args, expected in ELEMENT_CASES:
        panels = [np.full((2, 3), v) for v in args]
        got = po.elementwise(op, *panels, backend=backend)
        np.testing.assert_allclose(got, np.full((2, 3), expected), **TOL, err_msg=op)
        mask = np.array([[True, False, True], [True, True, True]])
        assert np.isnan(po.elementwise(op, *panels, mask=mask, backend=backend)[0, 1])


@pytest.mark.parametrize("backend", ["rust", "numpy"])
def test_cross_section_overflow_rows_are_missing(backend):
    rows = np.array(
        [[1e308, -1e308, 1e308], [1e308, 1e308, 5e307], [-1e308, 1e308, 1.0]]
    )
    scale = po.cs_scale(rows, backend=backend)
    demean = po.cs_demean(rows, backend=backend)
    zscore = po.cs_zscore(rows, backend=backend)
    assert np.isnan(scale).all() and np.isnan(zscore).all()
    assert np.isnan(demean[[0, 2]]).all()
    expected = [1e308 / 6, 1e308 / 6, -1e308 / 3]
    np.testing.assert_allclose(demean[1], expected, rtol=1e-15)
