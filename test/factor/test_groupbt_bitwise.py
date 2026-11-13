"""冻结 main 的全输出逐位回归；不依赖真实数据或外部路径。"""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from ashare.factor import groupbt
from bench.groupbt.common import cases, compare, snapshot, synthetic
from bench.groupbt.oracle import groupbt as oracle


@pytest.fixture(scope="module")
def panel_factor():
    return synthetic()


def test_main_oracle_sources_are_frozen():
    root = Path(__file__).resolve().parents[2] / "bench/groupbt/oracle"
    manifest = json.loads((root / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        source = root / name.removeprefix("src/ashare/factor/")
        assert hashlib.sha256(source.read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("options", cases())
def test_all_outputs_bitwise(panel_factor, options):
    panel, factor = panel_factor
    expected = snapshot(oracle.run(factor, panel, **options))
    actual = snapshot(groupbt.run(factor, panel, **options))
    compare(expected, actual)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "window",
    [("20150101", "20260101"), ("20240101", "20250101"), ("20300101", "20400101")],
)
@pytest.mark.parametrize("limited", [False, True])
def test_vectors_and_boundary_windows(dtype, window, limited):
    panel, factor = synthetic(seed=283)
    for key, value in panel.arrays.items():
        if value.dtype.kind == "f":
            panel.arrays[key] = value.astype(dtype)
    vector = np.random.default_rng(57).normal(size=factor.shape)
    vector[266, :7] = np.nan
    # A second regression, ties, and missing vectors exercise compact dot order.
    options = dict(
        universe="ALL",
        group_neutralize="cap+industry",
        n_groups=5,
        vector_neut=[vector, vector[:, ::-1]],
        decay=400,
        weights_portfolio=True,
        trade_limitation=limited,
        window=window,
    )
    compare(
        snapshot(oracle.run(factor, panel, **options)),
        snapshot(groupbt.run(factor, panel, **options)),
    )


def test_failure_rows_match_frozen_main():
    panel, factor = synthetic()
    factor[260] = 1.0
    factor[274] = np.nan
    factor[274, :3] = [1.0, 2.0, 3.0]
    errors = []
    for implementation in (oracle, groupbt):
        with pytest.raises(ValueError) as caught:
            implementation.run(factor, panel, "ALL")
        errors.append(caught.value.rows)
    assert errors[0] == errors[1] == [260, 274]


def test_long_carry_history_and_negative_decay():
    panel, factor = synthetic()
    for key in ("open", "close"):
        panel.arrays[key][1:270, :3] = np.nan
    options = dict(universe="ALL", decay=-3, weights_portfolio=True)
    compare(
        snapshot(oracle.run(factor, panel, **options)),
        snapshot(groupbt.run(factor, panel, **options)),
    )


def test_wide_row_reduction_order():
    panel, factor = synthetic(n=9001)
    options = dict(
        universe="ALL",
        group_neutralize="industry",
        weights_portfolio=True,
        sub_universes=False,
        window=("20241101", "20241105"),
    )
    compare(
        snapshot(oracle.run(factor, panel, **options)),
        snapshot(groupbt.run(factor, panel, **options)),
    )


def test_window_before_history_is_empty():
    panel, factor = synthetic()
    options = dict(universe="ALL", window=("19000101", "19010101"))
    compare(
        snapshot(oracle.run(factor, panel, **options)),
        snapshot(groupbt.run(factor, panel, **options)),
    )


def test_threads_preserve_all_outputs():
    panel, factor = synthetic()
    options = [
        dict(
            universe="ALL",
            group_neutralize=mode,
            weights_portfolio=True,
            decay=3,
            window=("20150101", "20260101"),
        )
        for mode in (None, "industry", "cap", "cap+industry")
    ]
    expected = [snapshot(oracle.run(factor, panel, **opt)) for opt in options]

    def job(i):
        compare(expected[i % 4], snapshot(groupbt.run(factor, panel, **options[i % 4])))

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(job, range(8)))


@pytest.mark.parametrize("mode", [None, "industry", "cap", "cap+industry"])
def test_fortran_inputs(mode):
    panel, factor = synthetic()
    panel.arrays = {k: np.asfortranarray(v) for k, v in panel.arrays.items()}
    factor = np.asfortranarray(factor)
    options = dict(
        universe="ALL", group_neutralize=mode, weights_portfolio=np.bool_(True)
    )
    compare(
        snapshot(oracle.run(factor, panel, **options)),
        snapshot(groupbt.run(factor, panel, **options)),
    )


def test_all_negative_industries_are_missing():
    panel, factor = synthetic()
    panel.arrays["industry"] = np.full(factor.shape, -7, dtype=np.int64)
    options = dict(universe="ALL", group_neutralize="industry")
    compare(
        snapshot(oracle.run(factor, panel, **options)),
        snapshot(groupbt.run(factor, panel, **options)),
    )
