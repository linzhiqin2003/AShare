"""平台任务内共享的 IC 输入缓存（`analysis._ic_inputs`）：并发变体只算一次、拿到同一份。

两把锁各守一段：全局 `_IC_LOCK` 守「查缓存 + 放占位」，每个键的锁守「算一次」。
两个用例分别把第一个线程卡在这两段里，让第二个线程撞进同一窗口，再放行。
来源：2026-11-14 并发测试补全（平台共享缓存加锁）。
"""

import threading
from types import SimpleNamespace

import numpy as np
import pytest

from ashare.platform import analysis

from .sync import Worker, in_stack, parked_at, wait_for

KEY_ARGS = ("TOP1800", "20160101", "open")


def _fake_inputs():
    pool = np.ones((4, 3), bool)
    return pool, pool.copy(), np.zeros((4, 3))


class GatedDict(dict):
    """第一个 `get` 停在闸门里（此时它持有 `_IC_LOCK`），记录每次 get 的线程。"""

    def __init__(self, release):
        super().__init__()
        self.release, self.callers = release, []
        self.entered = threading.Event()

    def get(self, key, default=None):
        self.callers.append(threading.current_thread().name)
        if len(self.callers) == 1:
            self.entered.set()
            assert self.release.wait(30)
        return super().get(key, default)


@pytest.mark.parametrize("repeat", range(3))
def test_second_variant_waits_for_first_computation(monkeypatch, repeat):
    """第一个线程正在算时，第二个线程必须停在该键的锁上，而不是再算一遍。"""
    release, computing = threading.Event(), threading.Event()
    calls = []

    def compute(cut, universe, start, trade_at):
        calls.append(threading.current_thread().name)
        if len(calls) == 1:
            computing.set()
            assert release.wait(30)
        return _fake_inputs()

    monkeypatch.setattr(analysis, "_compute_ic_inputs", compute)
    panel = SimpleNamespace(diagnostics={})
    cut = SimpleNamespace(stop=4)
    first = Worker(analysis._ic_inputs, panel, cut, *KEY_ARGS)
    first.start()
    wait_for(computing.is_set)
    second = Worker(analysis._ic_inputs, panel, cut, *KEY_ARGS)
    second.start()
    # 窗口：第二个线程停在 `with entry[0]:`（锁被第一个线程拿着），或（锁坏了时）进了计算。
    wait_for(
        lambda: (
            parked_at(second, analysis._ic_inputs, "with entry[0]:")
            or len(calls) > 1
            or second.finished.is_set()
        )
    )
    assert not second.finished.is_set()
    release.set()
    a, b = first.get(), second.get()
    assert len(calls) == 1, f"同一键算了 {len(calls)} 次"
    assert a is b


@pytest.mark.parametrize("repeat", range(3))
def test_lookup_and_placeholder_are_one_critical_section(monkeypatch, repeat):
    """查缓存和放占位之间不能被别的线程插进来，否则两个线程各放一个占位、各算一遍。"""
    release = threading.Event()
    calls = []

    def compute(cut, universe, start, trade_at):
        calls.append(threading.current_thread().name)
        return _fake_inputs()

    monkeypatch.setattr(analysis, "_compute_ic_inputs", compute)
    cache = GatedDict(release)
    panel = SimpleNamespace(diagnostics=cache)
    cut = SimpleNamespace(stop=4)
    first = Worker(analysis._ic_inputs, panel, cut, *KEY_ARGS)
    first.start()
    wait_for(cache.entered.is_set)
    second = Worker(analysis._ic_inputs, panel, cut, *KEY_ARGS)
    second.start()
    # 窗口：第二个线程停在 `with _IC_LOCK:`，或（全局锁坏了时）也进了 get。
    wait_for(
        lambda: (
            parked_at(second, analysis._ic_inputs, "with _IC_LOCK:")
            or len(cache.callers) > 1
            or in_stack(second, "compute")
            or second.finished.is_set()
        )
    )
    assert not second.finished.is_set()
    release.set()
    a, b = first.get(), second.get()
    assert len(calls) == 1, f"同一键算了 {len(calls)} 次"
    assert a is b
    assert len(cache) == 1


def test_distinct_keys_do_not_serialize(monkeypatch):
    """不同的（截断, 股票池, 成交时点）各算各的：一个键算的时候另一个键不用等。"""
    release = threading.Event()
    calls = []

    def compute(cut, universe, start, trade_at):
        calls.append(trade_at)
        if trade_at == "open":
            assert release.wait(30)
        return _fake_inputs()

    monkeypatch.setattr(analysis, "_compute_ic_inputs", compute)
    panel = SimpleNamespace(diagnostics={})
    cut = SimpleNamespace(stop=4)
    slow = Worker(analysis._ic_inputs, panel, cut, "TOP1800", "20160101", "open")
    slow.start()
    wait_for(lambda: calls == ["open"])
    # 闸门还关着时，另一个键必须能独立算完。
    other = analysis._ic_inputs(panel, cut, "TOP1800", "20160101", "close")
    assert calls == ["open", "close"] and not slow.finished.is_set()
    release.set()
    assert slow.get() is not other


def test_concurrent_real_inputs_match_uncached(tmp_path):
    """真实计算：4 个线程同时取同一组键，结果与不缓存现算逐位相同。"""
    from ashare.factor import groupbt
    from ashare.platform.analysis import CutPanel
    from bench.platform.synthetic import panel

    source = panel()

    class Cached:
        diagnostics = {}

        def __getattr__(self, name):
            return getattr(source, name)

        def __getitem__(self, name):
            return source[name]

    cached = Cached()
    cut = CutPanel(cached, groupbt.IS[1])
    barrier = threading.Barrier(4)
    keys = [("TOP1800", "open"), ("ALL", "close")] * 2

    def fetch(universe, trade_at):
        barrier.wait(30)
        return analysis._ic_inputs(cached, cut, universe, groupbt.IS[0], trade_at)

    workers = [Worker(fetch, *key) for key in keys]
    for worker in workers:
        worker.start()
    results = [worker.get() for worker in workers]
    assert results[0] is results[2] and results[1] is results[3]
    plain = CutPanel(source, groupbt.IS[1])
    for (universe, trade_at), got in zip(keys, results, strict=True):
        expected = analysis._compute_ic_inputs(plain, universe, groupbt.IS[0], trade_at)
        for a, b in zip(got, expected, strict=True):
            assert a.dtype == b.dtype and a.tobytes() == b.tobytes()
