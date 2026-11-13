"""Weight inputs are protected for the entire native borrow, including errors."""

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import numpy as np
import panelbt
import pytest
from panelbt import _core
from panelbt._readonly import readonly

from ashare.factor import groupbt
from bench.groupbt.common import compare, snapshot, synthetic


def test_shared_array_stays_readonly_until_last_thread_exits():
    a = np.ones((2, 3))
    entered, release = Event(), Event()

    def second():
        with readonly([a, a]):
            entered.set()
            assert release.wait(10)
            assert not a.flags.writeable

    with ThreadPoolExecutor(1) as pool:
        with readonly([a]):
            task = pool.submit(second)
            try:
                assert entered.wait(10)
            except BaseException:
                release.set()
                raise
        try:
            assert not a.flags.writeable
            with pytest.raises(ValueError, match="read-only"):
                a[0, 0] = 2
        finally:
            release.set()
        task.result(timeout=10)
    assert a.flags.writeable


def test_readonly_owner_and_view_restore_in_dependency_order():
    owner = np.ones(4)
    view = owner[1:]
    with readonly([owner]):
        with readonly([view]):
            assert not view.flags.writeable
        assert not view.flags.writeable
    assert owner.flags.writeable and view.flags.writeable
    with readonly([view, owner]):
        assert not owner.flags.writeable and not view.flags.writeable
    assert owner.flags.writeable and view.flags.writeable


def test_writable_view_of_readonly_owner_keeps_original_flags():
    owner = np.ones((3, 4))
    view = owner[1:]
    owner.flags.writeable = False
    panelbt.run_weights(view, np.zeros_like(view))
    assert not owner.flags.writeable
    assert view.flags.writeable


@pytest.mark.parametrize("fail", [False, True])
def test_weight_batch_protects_all_inputs_and_restores_flags(monkeypatch, fail):
    ret = np.zeros((2, 3))
    dense = np.ones_like(ret)
    labels = np.zeros(ret.shape, dtype=np.int64)
    buy = np.ones(ret.shape, bool)
    sell = buy.copy()
    sell.flags.writeable = False
    real = _core.run_weights_batch

    def native(r, b, s, plans):
        for a in [r, b, s, *(a for p in plans for a in p[:2])]:
            assert not a.flags.writeable
        if fail:
            raise RuntimeError("native failure")
        return real(r, b, s, plans)

    monkeypatch.setattr(_core, "run_weights_batch", native)
    args = (ret, [dense, panelbt.GroupWeightTargets(labels, 0)], buy, sell)
    if fail:
        with pytest.raises(RuntimeError, match="native failure"):
            panelbt.run_weights_batch(*args)
    else:
        panelbt.run_weights_batch(*args)
    assert all(a.flags.writeable for a in [ret, dense, labels, buy])
    assert not sell.flags.writeable


@pytest.mark.parametrize("fail", [False, True])
def test_groupbt_copies_factor_window_and_protects_native_views(monkeypatch, fail):
    panel, factor = synthetic()
    vector = np.random.default_rng(31).normal(size=factor.shape)
    panel.arrays["close"].flags.writeable = False
    options = dict(universe="ALL", group_neutralize="industry", vector_neut=vector)
    expected = snapshot(groupbt.run(factor, panel, **options))
    real = _core.run_group_weights
    borrowed, original_flags = [], []

    def native(*args):
        assert args[0].shape[0] == 25
        assert not np.shares_memory(args[0], factor)
        arrays = [*args[:3], *args[3], *args[4], *args[5:10]]
        borrowed.extend(arrays)
        assert all(not a.flags.writeable for a in arrays)
        for a in arrays:
            if a.size:
                with pytest.raises(ValueError, match="read-only"):
                    a.flat[0] = 0
        # A writer acting after the borrow starts cannot change the copied factor.
        with ThreadPoolExecutor(1) as pool:
            pool.submit(factor.fill, 0.0).result(timeout=10)
        if fail:
            raise RuntimeError("native failure")
        return real(*args)

    # Capture flags before the package makes its native arguments read-only.
    from panelbt import group_weights

    real_guard = group_weights.readonly

    def guard(arrays):
        original_flags.extend(a.flags.writeable for a in arrays)
        return real_guard(arrays)

    monkeypatch.setattr(group_weights, "readonly", guard)
    monkeypatch.setattr(_core, "run_group_weights", native)
    if fail:
        with pytest.raises(RuntimeError, match="native failure"):
            groupbt.run(factor, panel, **options)
    else:
        compare(expected, snapshot(groupbt.run(factor, panel, **options)))
    assert [a.flags.writeable for a in borrowed] == original_flags
    assert factor.flags.writeable and vector.flags.writeable
    assert not panel["close"].flags.writeable
