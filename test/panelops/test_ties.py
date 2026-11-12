"""Tie rule for extremum positions: ties="earliest" against direct window scans."""

import numpy as np
import panelops as po
import pytest
from panelops import reference

OPS = ("argmin", "argmax", "idxmin", "idxmax")


def direct(x, op, window, min_periods, ties, mask=None):
    """Independent cell-by-cell scan written from SPEC, not from reference.py."""
    t, n = x.shape
    out = np.full((t, n), np.nan)
    valid = np.isfinite(x) & (True if mask is None else mask)
    for j in range(n):
        for i in range(t):
            if mask is not None and not mask[i, j]:
                continue
            rows = [r for r in range(max(0, i - window + 1), i + 1) if valid[r, j]]
            if len(rows) < min_periods:
                continue
            values = [x[r, j] for r in rows]
            best = min(values) if op.endswith("min") else max(values)
            tied = [r for r, v in zip(rows, values, strict=True) if v == best]
            row = tied[0] if ties == "earliest" else tied[-1]
            age = i - row
            out[i, j] = age if op.startswith("arg") else window - age
    return out


def corpus(seed=0, t=90, n=7):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 4, (t, n)).astype(float)  # many ties
    x[rng.uniform(size=(t, n)) < 0.15] = np.nan
    x[:, 0] = 2.0  # constant column: every window is one big tie
    x[10:40, 1] = np.nan  # long gap
    x[:, 2] = rng.normal(size=t)  # no ties
    x[5, 3] = np.inf  # infinity is missing
    x[:, 4] = np.where(np.arange(t) % 2, 0.0, -0.0)  # signed zeros tie
    return x


@pytest.mark.parametrize("op", OPS)
@pytest.mark.parametrize("window", [1, 2, 5, 20, 60])
@pytest.mark.parametrize("ties", ["earliest", "latest"])
def test_ties_against_direct_scan(op, window, ties):
    x = corpus()
    for mp in sorted({1, max(1, window // 2), window}):
        got = getattr(po, f"ts_{op}")(x, window, min_periods=mp, ties=ties)
        np.testing.assert_array_equal(got, direct(x, op, window, mp, ties))
        ref = reference.ts(x, op, window, mp, ties=ties)
        np.testing.assert_array_equal(got, ref)


@pytest.mark.parametrize("op", OPS)
def test_ties_with_mask_threads_and_wide_panels(op):
    rng = np.random.default_rng(3)
    x = np.round(rng.normal(size=(300, 2100)), 1)  # ties; >1024 columns crosses blocks
    x[rng.uniform(size=x.shape) < 0.05] = np.nan
    mask = rng.uniform(size=x.shape) > 0.03
    fn = getattr(po, f"ts_{op}")
    try:
        po.set_num_threads(1)
        single = fn(x, 20, min_periods=5, mask=mask, ties="earliest")
        po.set_num_threads(4)
        multi = fn(x, 20, min_periods=5, mask=mask, ties="earliest")
    finally:
        po.set_num_threads(0)
    np.testing.assert_array_equal(single, multi)
    cols = [0, 1023, 1024, 2099]
    np.testing.assert_array_equal(
        single[:, cols],
        direct(x[:, cols], op, 20, 5, "earliest", mask[:, cols]),
    )


def test_earliest_matches_numpy_argmax_on_full_windows():
    """np.argmax picks the first tied maximum: idxmax(ties="earliest") = argmax + 1."""
    rng = np.random.default_rng(5)
    x = rng.integers(0, 5, (200, 30)).astype(float)
    w = 10
    windows = np.lib.stride_tricks.sliding_window_view(x, w, axis=0)
    expected = np.full_like(x, np.nan)
    expected[w - 1 :] = windows.argmax(axis=-1) + 1
    np.testing.assert_array_equal(po.ts_idxmax(x, w, ties="earliest"), expected)
    expected[w - 1 :] = windows.argmin(axis=-1) + 1
    np.testing.assert_array_equal(po.ts_idxmin(x, w, ties="earliest"), expected)


def test_ties_parameter_validation():
    x = np.zeros((5, 2))
    with pytest.raises(ValueError, match="ties"):
        po.ts_argmax(x, 3, ties="first")
    with pytest.raises(TypeError):
        po.ts_mean(x, 3, ties="earliest")
    # Values themselves never depend on the tie rule.
    y = corpus(seed=1)
    np.testing.assert_array_equal(po.ts_max(y, 7), reference.ts(y, "max", 7))
