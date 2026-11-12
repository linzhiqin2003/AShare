"""Validated array-only wrappers around the independent Rust core."""

import numpy as np

from . import _core


def available_threads():
    """Return the hardware thread count, independent of the process default."""
    return _core.available_threads()


def set_num_threads(threads):
    """Set the process default; zero selects the available CPU count. Pools are reused."""
    _core.set_num_threads(_threads(threads))


def get_num_threads():
    """Return the effective process default thread count."""
    return _core.get_num_threads()


def _panel(x):
    x = np.ascontiguousarray(x, dtype=np.float64)
    if x.ndim != 2 or 0 in x.shape or np.isinf(x).any():
        raise ValueError("Expected a nonempty T by N panel; missing values must be NaN")
    return x


def _threads(value):
    if not isinstance(value, int) or value < 0:
        raise ValueError("threads must be a nonnegative integer")
    return value


def ew_variance(x, half_life=252.0, lags=0, threads=0):
    x = _panel(x)
    if (
        half_life <= 0
        or np.isnan(half_life)
        or not isinstance(lags, int)
        or not 0 <= lags < len(x)
    ):
        raise ValueError("Invalid half-life or lag count")
    return _core.ew_variance(x, half_life, lags, _threads(threads))


def ew_moments_segment(
    window, first, last, shift, half_life=252.0, lags=0, floor=0, threads=0
):
    """Exponentially weighted moments of rows first..last of a T by N window.

    Built from empty one row at a time; values shifted by ``shift``; lag pairs kept
    only when the earlier row index is at least ``floor``. Returns an N*(3+4*lags)
    column-major vector for ``ew_moments_variance``.
    """
    x = np.ascontiguousarray(window, dtype=np.float64)
    shift = np.ascontiguousarray(shift, dtype=np.float64)
    if x.ndim != 2 or shift.shape != (x.shape[1],):
        raise ValueError("Expected a T by N window and an N shift; missing values NaN")
    if not all(isinstance(v, int) for v in (first, last, floor, lags)) or not (
        0 <= first <= last < len(x) and floor >= 0 and lags >= 0
    ):
        raise ValueError("Invalid segment bounds or lag count")
    if half_life <= 0 or np.isnan(half_life) or np.isinf(x[first : last + 1]).any():
        raise ValueError("Invalid half-life or infinite value")
    return _core.ew_moments_segment(
        x, first, last, floor, shift, half_life, lags, _threads(threads)
    )


def ew_moments_variance(state, n, lags, used, half_life=252.0):
    """Variance and term magnitude from moments, Bartlett weights over ``used`` lags."""
    state = np.ascontiguousarray(state, dtype=np.float64)
    if not isinstance(lags, int) or lags < 0 or state.shape != (n * (3 + 4 * lags),):
        raise ValueError("Moment state must hold n*(3+4*lags) values")
    if not isinstance(used, int) or not 0 <= used <= lags:
        raise ValueError("Invalid lag count")
    return _core.ew_moments_variance(state, n, lags, used, half_life)


def ew_covariance(x, half_life=252.0, lags=0, threads=0):
    x = _panel(x)
    if (
        half_life <= 0
        or np.isnan(half_life)
        or not isinstance(lags, int)
        or not 0 <= lags < len(x)
    ):
        raise ValueError("Invalid half-life or lag count")
    return _core.ew_covariance(x, half_life, lags, _threads(threads)).reshape(
        x.shape[1], x.shape[1]
    )


def rolling_bias(realized, predicted, window=252, threads=0):
    r, p = _panel(realized), _panel(predicted)
    if r.shape != p.shape or not isinstance(window, int) or not 2 <= window <= len(r):
        raise ValueError("Invalid panel shapes or window")
    return _core.rolling_bias(r, p, window, _threads(threads)).reshape(
        len(r) - window + 1, r.shape[1]
    )


def regress(returns, exposures, market_cap, estimation, industries=(), threads=0):
    r, cap = _panel(returns), _panel(market_cap)
    b = np.ascontiguousarray(exposures, dtype=float)
    est = np.ascontiguousarray(estimation, dtype=bool)
    t, n = r.shape
    if (
        b.ndim not in (2, 3)
        or b.shape[-2] != n
        or (b.ndim == 3 and b.shape[0] != t)
        or b.shape[-1] == 0
    ):
        raise ValueError("Invalid exposure shape")
    if cap.shape != r.shape or est.shape != r.shape:
        raise ValueError("Panel shapes must agree")
    indices = np.asarray(industries, dtype=int)
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= b.shape[-1]):
        raise ValueError("Invalid industry columns")
    f, u = _core.regress(r, b, cap, est, indices.tolist(), _threads(threads))
    return f.reshape(t, b.shape[-1]), u.reshape(t, n)


def _csc(value):
    rows, cols, ptr, ix, data = value
    ptr, ix, data = np.asarray(ptr), np.asarray(ix), np.asarray(data, dtype=float)
    if (
        rows < 0
        or cols < 1
        or ptr.shape != (cols + 1,)
        or ptr.dtype.kind not in "iu"
        or ix.dtype.kind not in "iu"
    ):
        raise ValueError("Invalid CSC dimensions or index types")
    if (
        ptr[0] != 0
        or ptr[-1] != len(ix)
        or len(ix) != len(data)
        or np.any(ptr[1:] < ptr[:-1])
        or np.any(ix < 0)
        or np.any(ix >= rows)
        or not np.isfinite(data).all()
    ):
        raise ValueError("Invalid CSC entries")
    for j in range(cols):
        row = ix[ptr[j] : ptr[j + 1]]
        if np.any(row[1:] <= row[:-1]):
            raise ValueError("CSC rows must be sorted and unique")
    return rows, cols, ptr.tolist(), ix.tolist(), data.tolist()


def _problem(p, q, a, b, cones, tolerance):
    p, a = _csc(p), _csc(a)
    q, b = np.asarray(q, dtype=float), np.asarray(b, dtype=float)
    if (
        p[0] != p[1]
        or a[1] != p[1]
        or q.shape != (p[1],)
        or b.shape != (a[0],)
        or not np.isfinite(q).all()
        or not np.isfinite(b).all()
    ):
        raise ValueError("Inconsistent conic problem dimensions")
    if (
        tolerance <= 0
        or not np.isfinite(tolerance)
        or sum(size for _, size in cones) != len(b)
        or any(
            kind not in (0, 1, 2) or size < 0 or (kind == 2 and size < 2)
            for kind, size in cones
        )
    ):
        raise ValueError("Invalid cones or tolerance")
    return p, q.tolist(), a, b.tolist(), cones, tolerance


def solve(p, q, a, b, cones, tolerance=1e-9):
    return _core.solve(*_problem(p, q, a, b, cones, tolerance))


def solve_many(p, qs, a, bs, cones, tolerance=1e-9, threads=0):
    qs, bs = np.asarray(qs, dtype=float), np.asarray(bs, dtype=float)
    if (
        qs.ndim != 2
        or bs.ndim != 2
        or len(qs) != len(bs)
        or len(qs) == 0
        or not np.isfinite(qs).all()
        or not np.isfinite(bs).all()
    ):
        raise ValueError("Expected nonempty finite batches")
    pp, _, aa, _, cc, tt = _problem(p, qs[0], a, bs[0], cones, tolerance)
    return _core.solve_many(pp, qs.tolist(), aa, bs.tolist(), cc, tt, _threads(threads))
