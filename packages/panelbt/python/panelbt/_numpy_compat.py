"""Reject a changed NumPy float64 reduction order before exposing the engine."""

import numpy as np

from . import _core


def check_reduction_order():
    """Compare small, cancellation-sensitive probes, including signed zeros/NaNs."""
    if np.lib.NumpyVersion(np.__version__) < "2.1.0":
        raise ImportError("panelbt weight mode requires NumPy >= 2.1")
    probes = []
    pattern = np.array([1e16, 1.0, -1e16, 1.0, 1.0, 1.0, 1.0, 1.0])
    for n in (1, 7, 8, 9, 127, 128, 129, 1000):
        cancellation = np.resize(pattern, n)
        mixed = np.ldexp(
            ((np.arange(n) * 37) % 101 - 50).astype(float),
            (np.arange(n) * 13) % 80 - 40,
        )
        missing = cancellation.copy()
        missing[n // 2] = np.nan
        probes.extend((cancellation, mixed, np.full(n, -0.0), missing))
    expected = np.array([np.add.reduce(p) for p in probes])
    actual = _core._numpy_sums(probes)
    if actual.tobytes() != expected.tobytes():
        raise ImportError(
            "panelbt: NumPy float64 summation order has changed; "
            "update the NumPy-order Rust kernel before using panelbt. "
            "Bitwise compatibility is required (including signed zero and NaN)."
        )
