"""平台调度器 `Manager`：多 tab 同时跑、运行中取消后同 tab 重跑、排队任务取消。

tab 工作进程是真正的 spawn 子进程，闸门见 `workers.py`（跨进程的 Barrier / Event）。
每个结果与在测试进程里直接调用 `run_variant` 的结果逐位相同。
来源：2026-11-14 并发测试补全（平台多 tab、取消重跑）。
"""

import json

import numpy as np
import pytest

from ashare.platform import runner
from ashare.platform.analysis import run_variant
from ashare.platform.config import expand_settings
from ashare.platform.holdings import archive_arrays
from ashare.platform.language import parse
from ashare.platform.runner import Manager
from ashare.platform.storage import Store
from bench.platform.synthetic import write_panel

from .sync import wait_for
from .workers import GatedContext, Gates

pytestmark = pytest.mark.timeout(240)

A = ("-ts_std(returns, [20, 60])", {"group_neutralize": ["industry", "cap+industry"]})
B = ("rank(close)", {"rebalance": [1, 5]})


@pytest.fixture(scope="module")
def data_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("manager-data")
    write_panel(root)
    return root


@pytest.fixture(scope="module")
def expected(data_root):
    """在测试进程里直接逐个算：{(源码, 配置 JSON): (summary, arrays)}。"""
    cache = runner.PanelCache(data_root)
    cache.ensure(runner.groupbt.FIELDS)
    table = {}
    for source, settings in (A, B):
        variants = parse(source)
        for config in expand_settings(settings, len(variants)):
            for variant in variants:
                cache.ensure(variant.expression.fields)
                table[_key(variant.source, config)] = run_variant(
                    variant.expression, cache, config
                )
    return table


def _manager(tmp_path, data_root, gates, limit=4):
    manager = Manager(
        Store(tmp_path / "results"),
        limit=limit,
        data_root=str(data_root),
        mode="threads",
        concurrency=2,
        idle_seconds=60,
    )
    manager.context = GatedContext(gates)
    return manager


def _key(source, settings):
    """结果库里的配置不含基准，调仓周期 1 省略（条目身份规则）。"""
    settings = {
        k: v
        for k, v in settings.items()
        if k != "benchmark" and not (k == "rebalance" and v == 1)
    }
    return source, json.dumps(settings, sort_keys=True)


def _finished(manager, job):
    return manager.store.job(job)["status"] != "running"


def _check_job(manager, job, expected):
    store = manager.store
    result = store.job(job)
    assert result["status"] == "complete"
    store.wait_holdings()
    with store.connect() as db:
        sources = dict(
            db.execute(
                "SELECT r.id, e.source FROM runs r JOIN entries e ON e.id=r.entry WHERE r.job=?",
                (job,),
            ).fetchall()
        )
    for run in result["runs"]:
        summary, arrays = expected[_key(sources[run["id"]], run["settings"])]
        for part in ("is", "validation"):
            assert run["summary"][part] == summary[part]
        arrays = {k: v.copy() for k, v in arrays.items()}
        holdings = archive_arrays(arrays)
        series = store.series(run["id"])
        assert series.keys() == arrays.keys()
        for key, value in arrays.items():
            assert series[key].tobytes() == value.tobytes(), key
        path = store.root / "holdings" / f"{run['id']}.npz"
        with np.load(path, allow_pickle=False) as archive:
            for key, value in holdings.items():
                assert archive[key].tobytes() == value.tobytes(), key


def test_two_tabs_compute_simultaneously_bitwise(tmp_path, data_root, expected):
    gates = Gates(tmp_path / "gates", parties=2)
    manager = _manager(tmp_path, data_root, gates)
    try:
        tabs = [manager.create_tab(), manager.create_tab()]
        jobs = [
            manager.submit(tab, source, settings, name)
            for tab, (source, settings), name in zip(tabs, (A, B), "ab", strict=True)
        ]
        for job in jobs:
            wait_for(lambda job=job: _finished(manager, job), timeout=120)
        # 窗口：每个进程进入计算后都等到另一个也进来才往下算（两个任务都完成即会合成功）。
        arrivals = gates.arrived()
        assert {job for job, _ in arrivals} == set(jobs)
        assert len({pid for _, pid in arrivals}) == 2
        for job in jobs:
            _check_job(manager, job, expected)
    finally:
        manager.close()


def test_cancel_mid_job_then_rerun_same_tab(tmp_path, data_root, expected):
    """第一个结果落盘后取消：已完成的变体保留，其余标取消；同 tab 重跑得到完整、逐位相同的结果。"""
    gates = Gates(tmp_path / "gates", results_before_hold=1)
    manager = _manager(tmp_path, data_root, gates)
    try:
        tab = manager.create_tab()
        job = manager.submit(tab, *A, "a")
        ((_, pid),) = gates.wait_arrivals(1)
        store = manager.store

        def statuses():
            return [r["status"] for r in store.job(job)["runs"]]

        # 窗口：恰好一个变体已保存，其余结果被扣在子进程里，任务仍在运行。
        wait_for(lambda: statuses().count("complete") == 1, timeout=120)
        assert store.job(job)["status"] == "running"
        assert manager.cancel(job)
        assert store.job(job)["status"] == "cancelled"
        assert sorted(statuses()) == ["cancelled"] * 3 + ["complete"]
        assert manager.tabs[tab]["process"] is None
        assert not (store.root / "work" / job).exists()
        assert not manager.cancel(job)  # 重复取消无效
        # 被扣住的结果永远不会再到达：放行后重跑，旧任务的状态不变。
        gates.release()
        again = manager.submit(tab, *A, "a")
        arrivals = gates.wait_arrivals(2)
        assert {job for job, _ in arrivals} == {job, again}
        new_pid = next(p for j, p in arrivals if j == again)
        assert new_pid != pid
        wait_for(lambda: _finished(manager, again), timeout=120)
        _check_job(manager, again, expected)
        assert store.job(job)["status"] == "cancelled"
        assert sorted(statuses()) == ["cancelled"] * 3 + ["complete"]
        kept = next(r for r in store.job(job)["runs"] if r["status"] == "complete")
        assert store.series(kept["id"]) is not None
    finally:
        manager.close()


def test_cancel_queued_job_never_starts_a_process(tmp_path, data_root, expected):
    gates = Gates(tmp_path / "gates")
    gates.hold()
    manager = _manager(tmp_path, data_root, gates, limit=1)
    try:
        first, second = manager.create_tab(), manager.create_tab()
        running = manager.submit(first, *B, "b")
        gates.wait_arrivals(1)  # 第一个任务占满预算，停在计算闸门里
        queued = manager.submit(second, *A, "a")
        with manager.lock:
            assert (
                manager.tabs[second]["pending"] and not manager.tabs[second]["process"]
            )
        assert manager.cancel(queued)
        assert manager.store.job(queued)["status"] == "cancelled"
        gates.release()
        wait_for(lambda: _finished(manager, running), timeout=120)
        _check_job(manager, running, expected)
        with manager.lock:
            state = manager.tabs[second]
            assert state["process"] is None and state["pending"] is None
        # 只有第一个 tab 起过进程。
        assert len(manager.context.processes) == 1
        assert [job for job, _ in gates.arrived()] == [running]
    finally:
        manager.close()
