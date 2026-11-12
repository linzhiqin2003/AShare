# ruff: noqa: E402
"""P-013：ts_mad 改用有序窗口后与按定义的两次选择逐位相同。"""

from pathlib import Path

import panelops

ROOT = Path(__file__).resolve().parents[2]
assert Path(panelops.__file__).resolve().is_relative_to(ROOT)

import numpy as np
import panelops as po
import pytest


def select(values, q=0.5):
    v = sorted(values)
    pos = q * (len(v) - 1)
    lo = int(np.floor(pos))
    a = v[lo]
    return a if pos == lo else a + (v[lo + 1] - a) * (pos - lo)


def reference(x, window, min_periods, mask):
    t, n = x.shape
    out = np.full((t, n), np.nan)
    for j in range(n):
        for i in range(t):
            if mask is not None and not mask[i, j]:
                continue
            rows = range(max(0, i - window + 1), i + 1)
            v = [
                float(x[r, j])
                for r in rows
                if np.isfinite(x[r, j]) and (mask is None or mask[r, j])
            ]
            if len(v) < min_periods or not v:
                continue
            m = select(v)
            out[i, j] = select([abs(a - m) for a in v])
    return out


def data(kind, t=160, n=6, seed=0):
    r = np.random.default_rng(seed)
    if kind == "ties":
        x = r.integers(-2, 3, (t, n)).astype(float)
    elif kind == "zeros":
        x = np.where(r.random((t, n)) < 0.5, 0.0, -0.0) + r.integers(0, 2, (t, n))
    elif kind == "huge":
        x = r.choice([-1.7e308, 1.7e308, 1.0, -1.0, 0.0], (t, n))
    elif kind == "tiny":
        x = r.normal(size=(t, n)) * 5e-324 * r.integers(1, 9, (t, n))
    else:
        x = r.normal(size=(t, n))
    x[r.random((t, n)) < 0.1] = np.nan
    x[r.random((t, n)) < 0.02] = np.inf
    return np.ascontiguousarray(x)


@pytest.mark.parametrize("kind", ["normal", "ties", "zeros", "huge", "tiny"])
@pytest.mark.parametrize("window", [1, 2, 3, 4, 7, 20, 70])
@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("threads", [1, 0])
def test_mad_matches_definition(kind, window, masked, threads):
    x = data(kind, seed=window)
    mask = np.random.default_rng(1).random(x.shape) > 0.1 if masked else None
    previous = po.get_num_threads()
    po.set_num_threads(threads)
    try:
        for min_periods in sorted({1, max(1, window // 2), window}):
            got = po.ts_mad(x, window, min_periods=min_periods, mask=mask)
            expected = reference(x, window, min_periods, mask)
            assert got.tobytes() == expected.tobytes(), (kind, window, min_periods)
    finally:
        po.set_num_threads(previous)
