"""Pure-weight accounting preserves the frozen NumPy simulator bit for bit."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import panelbt
import pytest
from panelbt import reference_weights


def assert_bits(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert a.tobytes() == b.tobytes(), f"max error {np.max(np.abs(a - b), initial=0)}"


@pytest.mark.parametrize("n", [0, 1, 7, 8, 17, 127, 128, 129, 257, 1801, 5923])
@pytest.mark.parametrize("limited", [False, True])
def test_dense_bitwise(n, limited):
    rng = np.random.default_rng(430 + n)
    target = rng.normal(size=(13, n)) / max(n, 1)
    target[rng.random(target.shape) < 0.55] = 0
    target[rng.random(target.shape) < 0.1] = np.nan
    returns = rng.normal(size=target.shape) * 0.03
    returns[rng.random(target.shape) < 0.1] = np.nan
    buy = rng.random(target.shape) > 0.4 if limited else None
    sell = rng.random(target.shape) > 0.4 if limited else None
    expected = reference_weights.simulate(target, returns, buy, sell)
    actual = panelbt.run_weights(target, returns, buy, sell)
    for key, value in expected.items():
        assert_bits(value, getattr(actual, key))


def test_batch_group_labels_and_dense_are_identical():
    rng = np.random.default_rng(77)
    shape = (35, 273)
    labels = rng.integers(-1, 10, shape, dtype=np.int64)
    labels[0] = -1
    ret = rng.normal(size=shape)
    buy, sell = rng.random(shape) > 0.3, rng.random(shape) > 0.3
    plans = [panelbt.GroupWeightTargets(labels, g) for g in range(10)]
    plans.append(panelbt.GroupWeightTargets(labels, 0, 9, 0.5, 0.5))
    dense = []
    for plan in plans:
        w = np.zeros(shape)
        for group, scale in (
            (plan.long_group, plan.long_budget),
            (plan.short_group, -plan.short_budget),
        ):
            mask = labels == group
            counts = mask.sum(axis=1).astype(float)
            has = counts > 0
            part = np.zeros(shape)
            part[has] = mask[has] / counts[has, None]
            w += scale * part
        dense.append(w)
    got = panelbt.run_weights_batch(ret, plans, buy, sell)
    for actual, target in zip(got, dense, strict=True):
        expected = reference_weights.simulate(target, ret, buy, sell)
        for key, value in expected.items():
            assert_bits(value, getattr(actual, key))

    def worker(_):
        return panelbt.run_weights_batch(ret, plans, buy, sell, keep_weights=False)

    with ThreadPoolExecutor(4) as pool:
        for result in pool.map(worker, range(8)):
            for a, b in zip(got, result, strict=True):
                assert b.weights is None
                assert_bits(a.pnl, b.pnl)
                assert_bits(a.turnover, b.turnover)


def test_invalid_weight_inputs():
    a = np.ones((3, 4))
    for call in (
        lambda: panelbt.run_weights(a, a[0]),
        lambda: panelbt.run_weights(a[:2], a),
        lambda: panelbt.run_weights(a, a, can_buy=a[:2]),
        lambda: panelbt.run_weights_batch(a, [a], keep_weights=[]),
        lambda: panelbt.run_weights(panelbt.GroupWeightTargets(a, 0), a),
        lambda: panelbt.run_weights(panelbt.GroupWeightTargets(a.astype(int), -1), a),
    ):
        with pytest.raises(ValueError):
            call()
    assert panelbt.run_weights_batch(a, []) == []
