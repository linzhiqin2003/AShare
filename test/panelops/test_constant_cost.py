# ruff: noqa: E402
"""Exactly constant windows must stay on the cheap path (no per-cell window rescans).

Before the run-length proof, a constant W=250 window cost 40-60 times a random one.
The bound below leaves a wide margin for a busy machine: timings alternate between
the two inputs and the minimum of several runs is compared.
"""

import time
from pathlib import Path

import panelops as po

ROOT = Path(__file__).resolve().parents[2]
assert Path(po._api._core.__file__).resolve().is_relative_to(ROOT)

import numpy as np
import pytest

PAIRED = {"corr", "cov", "regression_slope"}
LIMIT = 4.0


def fastest(calls, rounds=5):
    best = [float("inf")] * len(calls)
    for _ in range(rounds):
        for i, call in enumerate(calls):
            start = time.perf_counter()
            call()
            best[i] = min(best[i], time.perf_counter() - start)
    return best


@pytest.mark.parametrize(
    "op", ["std", "var", "zscore", "corr", "cov", "skew", "kurt", "slope", "rsquare"]
)
def test_constant_input_costs_no_more_than_random(op):
    rng = np.random.default_rng(5)
    random = rng.normal(size=(2400, 384))
    constant = np.full_like(random, 0.1)
    other = rng.normal(size=random.shape)
    fn = getattr(po, "ts_" + op)
    window = 250

    def run(x):
        return (
            (lambda: fn(x, other, window)) if op in PAIRED else (lambda: fn(x, window))
        )

    try:
        po.set_num_threads(1)
        run(random)()
        run(constant)()
        slow, fast = fastest([run(constant), run(random)])
    finally:
        po.set_num_threads(0)
    assert slow <= LIMIT * fast, (
        f"{op}: constant {slow * 1e3:.1f} ms, random {fast * 1e3:.1f} ms"
    )
