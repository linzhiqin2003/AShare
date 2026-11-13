"""Import-time NumPy compatibility guard and its exact-bit failure contract."""

import importlib
import tomllib
from pathlib import Path

import numpy as np
import panelbt
import pytest
from panelbt import _core, _numpy_compat


def test_probe_coverage_and_exact_bits(monkeypatch):
    real = _core._numpy_sums
    lengths = set()

    def checked(probes):
        lengths.update(map(len, probes))
        assert any(np.signbit(p).all() and (p == 0).all() for p in probes)
        assert any(np.isnan(p).any() for p in probes)
        return real(probes)

    monkeypatch.setattr(_core, "_numpy_sums", checked)
    _numpy_compat.check_reduction_order()
    assert lengths == {1, 7, 8, 9, 127, 128, 129, 1000}


@pytest.mark.parametrize("mutation", ["order", "zero", "nan"])
def test_import_rejects_changed_reduction(monkeypatch, mutation):
    real = _core._numpy_sums

    def changed(probes):
        values = real(probes)
        if mutation == "order":
            values[0] = np.nextafter(values[0], np.inf)
        elif mutation == "zero":
            values[2] = -0.0
        else:
            values.view(np.uint64)[3] ^= np.uint64(1)
        return values

    monkeypatch.setattr(_core, "_numpy_sums", changed)
    with pytest.raises(ImportError, match="NumPy float64 summation order has changed"):
        importlib.reload(panelbt)
    monkeypatch.setattr(_core, "_numpy_sums", real)
    importlib.reload(panelbt)


def test_numpy_minimum_matches_metadata_and_native_guard(monkeypatch):
    root = Path(__file__).resolve().parents[1]
    package = root / "packages/panelbt"
    metadata = tomllib.loads((package / "pyproject.toml").read_text())
    assert "numpy>=2.1,<3" in metadata["project"]["dependencies"]
    assert (
        "weight mode requires NumPy >= 2.1"
        in (package / "src/numpy_dot.rs").read_text()
    )
    monkeypatch.setattr(np, "__version__", "2.0.2")
    with pytest.raises(ImportError, match="requires NumPy >= 2.1"):
        _numpy_compat.check_reduction_order()
