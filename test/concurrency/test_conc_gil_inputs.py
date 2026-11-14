"""内核释放 GIL 期间的输入保护。

闸门插在 `panelbt.group_weights._core` 上：Python 包装层已经把借出的数组设成只读、
即将进入原生内核时停住。此时另一个线程去改输入：
- 借出的数组对象写不进（ValueError）；同一数组被两个调用同时借用时，先结束的调用
  不能提前恢复可写，直到最后一个借用者结束；
- 分组回测的可写因子在进内核前已独立复制，别的线程改原数组不影响结果。
来源：2026-11-14 并发测试补全（GIL 释放期间输入只读）。
"""

import threading

import numpy as np
import panelbt
import pytest
from panelbt import group_weights

from ashare.factor import groupbt
from ashare.factor.groupbt import api
from bench.groupbt.common import compare, snapshot, synthetic

from .sync import Recorder, Worker, wait_for


class GatedCore:
    """转发到真正的 `_core`；`run_group_weights` 进门前按调用次序取一把闸门。"""

    def __init__(self, real, gates):
        self._real, self._gates = real, list(gates)
        self.entered = Recorder()
        self._lock = threading.Lock()
        self._next = 0

    def __getattr__(self, name):
        return getattr(self._real, name)

    def run_group_weights(self, *args):
        with self._lock:
            index = self._next
            self._next += 1
        self.entered.add("enter", index)
        if index < len(self._gates):
            assert self._gates[index].wait(30)
        try:
            return self._real.run_group_weights(*args)
        finally:
            self.entered.add("leave", index)


def _prepared():
    panel, factor = synthetic()
    prepared = api._prepare(factor, panel, "ALL", sub_universes=False)
    return prepared


def _try_write(array):
    try:
        array[0, 0] = array[0, 0]
    except ValueError:
        return False
    return True


@pytest.mark.parametrize("repeat", range(3))
def test_borrowed_views_are_readonly_while_kernel_runs(monkeypatch, repeat):
    prepared = _prepared()
    price = prepared.args[1]
    assert price.flags.writeable and _try_write(price)
    expected = panelbt.run_group_weights(*prepared.args, **prepared.options)
    gate = threading.Event()
    core = GatedCore(group_weights._core, [gate])
    monkeypatch.setattr(group_weights, "_core", core)
    worker = Worker(panelbt.run_group_weights, *prepared.args, **prepared.options)
    worker.start()
    wait_for(lambda: core.entered.count("enter") == 1)
    # 窗口：调用停在原生内核门口，借出的价格视图此刻必须写不进。
    assert not price.flags.writeable
    assert not _try_write(price)
    gate.set()
    actual = worker.get()
    assert price.flags.writeable and _try_write(price)
    for key in ("pnl", "turnover", "exposure"):
        np.testing.assert_array_equal(
            actual[0][key].view(np.uint64), expected[0][key].view(np.uint64)
        )


@pytest.mark.parametrize("repeat", range(3))
def test_overlapping_borrowers_restore_after_last(monkeypatch, repeat):
    """两个调用同时借同一个数组：第一个结束后数组仍只读，第二个结束才恢复可写。"""
    gates = [threading.Event(), threading.Event()]
    core = GatedCore(group_weights._core, gates)
    monkeypatch.setattr(group_weights, "_core", core)
    prepared = _prepared()
    price = prepared.args[1]
    workers = [
        Worker(panelbt.run_group_weights, *prepared.args, **prepared.options)
        for _ in range(2)
    ]
    for worker in workers:
        worker.start()
    wait_for(lambda: core.entered.count("enter") == 2)
    assert not _try_write(price)
    # 先放行先进门的那个；它结束时另一个仍在内核门口借着同一数组。
    gates[0].set()
    wait_for(lambda: any(w.finished.is_set() for w in workers))
    assert sum(w.finished.is_set() for w in workers) == 1
    assert not price.flags.writeable, "还有借用者在内核里，数组提前恢复了可写"
    assert not _try_write(price)
    gates[1].set()
    first, second = (w.get() for w in workers)
    assert price.flags.writeable and _try_write(price)
    np.testing.assert_array_equal(
        first[0]["pnl"].view(np.uint64), second[0]["pnl"].view(np.uint64)
    )


@pytest.mark.parametrize("repeat", range(3))
def test_mutating_factor_during_kernel_does_not_change_result(monkeypatch, repeat):
    """分组回测里可写的因子在进内核前独立复制：内核运行时另一个线程改原数组，结果不变。"""
    panel, factor = synthetic()
    pristine = factor.copy()
    options = dict(universe="ALL", record_holdings=True)
    expected = snapshot(groupbt.run(pristine, panel, **options))
    gate = threading.Event()
    core = GatedCore(group_weights._core, [gate])
    monkeypatch.setattr(group_weights, "_core", core)
    worker = Worker(groupbt.run, factor, panel, **options)
    worker.start()
    wait_for(lambda: core.entered.count("enter") == 1)
    assert factor.flags.writeable  # 调用者的数组本身没有被锁
    factor[:] = -factor  # 正在回测时把因子整体反号
    gate.set()
    compare(snapshot(worker.get()), expected)


def test_platform_signal_and_panel_owners_are_immutable(monkeypatch, tmp_path):
    """平台路径：进内核的因子、价格等数组，连同它们的底层所有者都不可写，别的线程无从改起。"""
    from ashare.platform.runner import PanelCache
    from bench.platform.synthetic import write_panel

    write_panel(tmp_path / "data")
    cache = PanelCache(tmp_path / "data")
    cache.ensure(groupbt.FIELDS)
    signal_path = tmp_path / "signal.npy"
    rng = np.random.default_rng(3)
    np.save(signal_path, rng.normal(size=cache.shape))
    signal = np.load(signal_path, mmap_mode="r")
    seen = []

    class Inspecting:
        def __getattr__(self, name):
            return getattr(real, name)

        def run_group_weights(self, *args):
            seen.append(args)
            return real.run_group_weights(*args)

    real = group_weights._core
    monkeypatch.setattr(group_weights, "_core", Inspecting())
    groupbt.run(signal, cache, "ALL", sub_universes=False)
    (args,) = seen

    def owner_writeable(array):
        while isinstance(array, np.ndarray):
            if array.flags.writeable:
                return True
            array = array.base
        return False

    factor, price, close, adjustment = args[0], args[5], args[6], args[7]
    for name, array in (
        ("factor", factor),
        ("price", price),
        ("close", close),
        ("adjustment", adjustment),
    ):
        assert not owner_writeable(array), f"{name} 的底层可写"
    assert np.shares_memory(factor, signal)  # 只读 mmap 免拷贝，仍然安全
