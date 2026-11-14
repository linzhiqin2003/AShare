"""分组回测 `run_many`：共享缓存、外部线程池、on_result 回调与并发调用。

闸门插在原生内核门口（`Barrier`）：几个回测同时进了门才一起往下走，证明它们确实
在内核里重叠（共享的风险均值缓存从空开始被同时填充）。所有结果与逐个 `groupbt.run`
串行逐位相同。来源：2026-11-14 并发测试补全（groupbt run_many 与外部 executor）。
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
from panelbt import group_weights

from ashare.factor import groupbt
from bench.groupbt.common import compare, snapshot, synthetic

from .sync import Recorder, Worker


class BarrierCore:
    """前 `parties` 个内核调用在门口会合；之后的直接通过。"""

    def __init__(self, real, parties):
        self._real = real
        self.barrier = threading.Barrier(parties)
        self.parties = parties
        self.calls = Recorder()
        self._lock = threading.Lock()
        self._count = 0

    def __getattr__(self, name):
        return getattr(self._real, name)

    def run_group_weights(self, *args):
        with self._lock:
            index = self._count
            self._count += 1
        self.calls.add("enter", threading.get_ident())
        if index < self.parties:
            self.barrier.wait(60)
        return self._real.run_group_weights(*args)


def _configurations():
    return [
        dict(
            universe=pool,
            group_neutralize=mode,
            rebalance=rebalance,
            weights_portfolio=weighted,
            record_holdings=True,
            window=window,
        )
        for pool in ("TOP800", "ALL")
        for mode in (None, "cap+industry")
        for rebalance, weighted in ((1, False), (3, True))
        for window in (groupbt.IS, groupbt.VALIDATION)
    ]


def _serial(factors, panel, configurations):
    return [
        snapshot(groupbt.run(x, panel, **c))
        for x, c in zip(factors, configurations, strict=True)
    ]


@pytest.fixture(scope="module")
def case():
    panel, factor = synthetic()
    configurations = _configurations()
    factors = [factor if i % 3 else -factor for i in range(len(configurations))]
    return panel, factors, configurations, _serial(factors, panel, configurations)


def _check(results, expected):
    assert len(results) == len(expected)
    for got, want in zip(results, expected, strict=True):
        compare(snapshot(got), want)


@pytest.mark.parametrize("repeat", range(2))
def test_overlapping_kernels_share_exposure_cache_bitwise(monkeypatch, case, repeat):
    panel, factors, configurations, expected = case
    core = BarrierCore(group_weights._core, parties=4)
    monkeypatch.setattr(group_weights, "_core", core)
    results = groupbt.run_many(factors, panel, configurations, workers=4)
    # 窗口：4 个内核调用在门口会合后才放行（Barrier 没有破）。
    assert not core.barrier.broken
    assert len({ident for _, ident in core.calls.events[:4]}) == 4
    _check(results, expected)


@pytest.mark.parametrize("repeat", range(2))
def test_external_executor_shared_with_on_result_tasks(monkeypatch, case, repeat):
    """平台的用法：回测和下游统计共用一个线程池，回调里继续往同一池子提交任务。"""
    panel, factors, configurations, expected = case
    core = BarrierCore(group_weights._core, parties=3)
    monkeypatch.setattr(group_weights, "_core", core)
    order, downstream = [], []
    with ThreadPoolExecutor(max_workers=4) as executor:

        def on_result(index, result):
            assert threading.current_thread() is threading.main_thread()
            order.append(index)
            downstream.append(
                executor.submit(lambda r=result, i=index: (i, snapshot(r)))
            )

        results = groupbt.run_many(
            factors,
            panel,
            configurations,
            workers=3,
            on_result=on_result,
            executor=executor,
        )
        stats = dict(f.result(timeout=60) for f in downstream)
    assert not core.barrier.broken
    assert sorted(order) == list(range(len(configurations)))
    _check(results, expected)
    for index, value in stats.items():
        compare(value, expected[index])


@pytest.mark.parametrize("repeat", range(2))
def test_two_callers_on_same_inputs_concurrently(monkeypatch, case, repeat):
    """两个线程同时对同一因子、同一面板调用 run_many：各自的共享缓存互不串，结果逐位相同。"""
    panel, factors, configurations, expected = case
    core = BarrierCore(group_weights._core, parties=2)
    monkeypatch.setattr(group_weights, "_core", core)
    half = len(configurations) // 2
    jobs = [
        (factors[:half], configurations[:half], expected[:half]),
        (factors[half:], configurations[half:], expected[half:]),
    ]
    workers = [Worker(groupbt.run_many, f, panel, c, workers=1) for f, c, _ in jobs]
    for worker in workers:
        worker.start()
    results = [worker.get(60) for worker in workers]
    # 每个调用 workers=1、在自己线程里串行，会合成功说明两个调用的内核确实重叠。
    assert not core.barrier.broken
    assert len({ident for _, ident in core.calls.events[:2]}) == 2
    for got, (_, _, want) in zip(results, jobs, strict=True):
        _check(got, want)


def test_on_result_failure_propagates_without_hanging(case):
    """回调抛错时 run_many 把错误交给调用者，不会卡住外部线程池。"""
    panel, factors, configurations, _ = case

    def on_result(index, result):
        raise RuntimeError("downstream failed")

    with ThreadPoolExecutor(max_workers=2) as executor:
        with pytest.raises(RuntimeError, match="downstream failed"):
            groupbt.run_many(
                factors[:4],
                panel,
                configurations[:4],
                workers=2,
                on_result=on_result,
                executor=executor,
            )
        # 池子仍可用。
        assert executor.submit(lambda: 7).result(timeout=30) == 7
