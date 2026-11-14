"""平台工作进程内的并发计算、后台写持仓与结果库并发读写（同进程，不起子进程）。

- `calculate_many` 多个变体同时进统计（闸门：4 方会合），结果与串行逐位相同；
- 持仓档案在后台写：读持仓的线程停在等待上，写完才读到完整档案；不等无关的变体；
  关闭结果库时等全部写完；
- 结果库 SQLite WAL：4 个变体同时停在保存中途时，读线程照常读，最终库与串行保存相同。
来源：2026-11-14 并发测试补全（平台调度、后台写持仓、结果库并发读写）。
"""

import copy
import queue
import threading

import numpy as np
import pytest

import ashare.platform.storage as storage
from ashare.platform import runner
from ashare.platform.analysis import run_variant
from ashare.platform.config import DEFAULTS
from ashare.platform.holdings import Holdings, archive_arrays
from ashare.platform.language import parse
from ashare.platform.storage import Store
from bench.platform.synthetic import write_panel

from .sync import Worker, in_stack, parked_at, wait_for

SOURCE = "vol = ts_std(returns, [20, 60])\nalpha = -vol"
NEUTRAL = ("industry", "cap+industry")


def _pairs():
    variants = parse(SOURCE, factors={})
    return [
        (v, dict(DEFAULTS, group_neutralize=mode)) for mode in NEUTRAL for v in variants
    ]


@pytest.fixture(scope="module")
def data_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("platform-data")
    write_panel(root)
    return root


@pytest.fixture(scope="module")
def direct(data_root):
    """每个变体直接调用 run_variant（不共享、不并发）的结果，作为逐位对照。"""
    cache = runner.PanelCache(data_root)
    results = []
    for variant, settings in _pairs():
        cache.ensure(variant.expression.fields)
        cache.ensure(runner.groupbt.FIELDS)
        results.append(run_variant(variant.expression, cache, settings))
    return results


def _bitwise(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in expected.items():
        got = actual[key]
        assert got.dtype == value.dtype and got.shape == value.shape, key
        assert got.tobytes() == value.tobytes(), key


def _compute(data_root, tmp_path, parallel, gate=None):
    store = Store(tmp_path / f"results-{parallel}")
    job, runs = store.begin("tab", "alpha", _pairs(), {})
    cache = runner.PanelCache(data_root)
    runs = runner.prepare_signals(
        runs, cache, store.root / "work" / job, lambda i, p: None
    )
    out = queue.Queue()
    runner.calculate_many(runs, cache, job, out, parallel)
    ordinal = {run["id"]: run["ordinal"] for run in runs}
    results = {}
    while not out.empty():
        _, kind, payload = out.get_nowait()
        if kind == "result":
            results[ordinal[payload["run"]]] = payload
    store.close()
    return [results[i] for i in range(len(runs))]


@pytest.mark.parametrize("repeat", range(2))
def test_parallel_variants_bitwise_equal_serial(
    monkeypatch, data_root, direct, tmp_path, repeat
):
    serial = _compute(data_root, tmp_path, 1)
    barrier = threading.Barrier(4)
    entered = []
    real = runner.run_variant

    def gated(*args, **kwargs):
        entered.append(threading.get_ident())
        barrier.wait(60)
        return real(*args, **kwargs)

    monkeypatch.setattr(runner, "run_variant", gated)
    parallel = _compute(data_root, tmp_path, 4)
    # 窗口：4 个变体同时进了统计（共享的 IC 输入缓存、诊断缓存被并发读写）。
    assert not barrier.broken and len(set(entered)) == 4
    for (summary, arrays), a, b in zip(direct, serial, parallel, strict=True):
        assert a["summary"] == b["summary"] == summary
        _bitwise(a["arrays"], arrays)
        _bitwise(b["arrays"], arrays)


@pytest.fixture(scope="module")
def saved(direct):
    return [(summary, arrays) for summary, arrays in direct]


def _fresh(saved):
    return [(copy.deepcopy(s), {k: v.copy() for k, v in a.items()}) for s, a in saved]


@pytest.mark.parametrize("repeat", range(2))
def test_holdings_reader_waits_for_background_write(
    monkeypatch, saved, tmp_path, repeat
):
    release = threading.Event()
    writing = []
    original = storage.write_archive

    def slow(path, data):
        writing.append(path.name)
        assert release.wait(30)
        original(path, data)

    monkeypatch.setattr(storage, "write_archive", slow)
    store = Store(tmp_path / "results")
    variant, settings = _pairs()[0]
    job, runs = store.begin("tab", "alpha", [(variant, settings)], {})
    ((summary, arrays),) = _fresh(saved[:1])
    expected = archive_arrays({k: v.copy() for k, v in arrays.items()})
    run = runs[0]["id"]
    assert store.save(run, summary, arrays, 0, "synthetic")
    store.finish(job, "complete")
    wait_for(lambda: len(writing) == 1)
    reader = Worker(Holdings(store).read, run)
    reader.start()
    # 窗口：档案还在写（闸门关着），读线程必须停在等待上，不能先读到「没有持仓」。
    wait_for(lambda: in_stack(reader, "wait_holdings") or reader.finished.is_set())
    assert not reader.finished.is_set(), "后台档案没写完，读线程就返回了"
    release.set()
    result = reader.get()
    assert result["dates"]
    path = store.root / "holdings" / f"{run}.npz"
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == set(expected)
        for key, value in expected.items():
            assert archive[key].tobytes() == value.tobytes(), key
    assert not list(path.parent.glob("*.tmp"))
    store.close()


def test_holdings_reader_does_not_wait_for_other_runs(monkeypatch, saved, tmp_path):
    store = Store(tmp_path / "results")
    pairs = _pairs()[:2]
    job, runs = store.begin("tab", "alpha", pairs, {})
    data = _fresh(saved[:2])
    first, second = (r["id"] for r in runs)
    store.save(first, *data[0], 0, "synthetic")
    store.wait_holdings(first)
    release = threading.Event()
    original = storage.write_archive

    def slow(path, arrays):
        assert release.wait(30)
        original(path, arrays)

    monkeypatch.setattr(storage, "write_archive", slow)
    store.save(second, *data[1], 0, "synthetic")
    store.finish(job, "complete")
    # 第二个变体的档案卡在闸门里时，读第一个变体不用等它。
    reader = Worker(Holdings(store).read, first)
    reader.start()
    assert reader.get(20)["dates"]
    assert not (store.root / "holdings" / f"{second}.npz").exists()
    closing = Worker(store.close)
    closing.start()
    # 关闭结果库要等全部后台写完：停在 wait_holdings 里，放行后档案落盘。
    wait_for(lambda: in_stack(closing, "wait_holdings") or closing.finished.is_set())
    assert not closing.finished.is_set()
    release.set()
    closing.get()
    assert (store.root / "holdings" / f"{second}.npz").exists()


def _state(store, job):
    result = store.job(job)
    rows = []
    for run in result["runs"]:
        series = store.series(run["id"])
        rows.append((run["ordinal"], run["status"], run["summary"], series))
    return result["status"], rows


@pytest.mark.parametrize("repeat", range(2))
def test_concurrent_saves_with_readers_match_serial(
    monkeypatch, saved, tmp_path, repeat
):
    pairs = _pairs()
    serial = Store(tmp_path / "serial")
    job, runs = serial.begin("tab", "alpha", pairs, {})
    for run, (summary, arrays) in zip(runs, _fresh(saved), strict=True):
        serial.save(run["id"], summary, arrays, 0, "synthetic")
    serial.finish(job, "complete")
    expected = _state(serial, job)
    serial.close()

    store = Store(tmp_path / "concurrent")
    job, runs = store.begin("tab", "alpha", pairs, {})
    stop = threading.Event()
    reads = [0, 0]
    errors = []

    def reader(slot):
        while not stop.is_set():
            try:
                store.library()
                store.job(job)
                store.events(job, 0)
                store.trials()
            except Exception as error:  # noqa: BLE001
                errors.append(error)
                return
            reads[slot] += 1

    readers = [Worker(reader, i) for i in range(2)]
    for r in readers:
        r.start()
    wait_for(lambda: all(reads))

    def all_parked():
        # 4 个保存都停在会合点上时，读线程各自再完整读一轮：读写确实重叠。
        before = list(reads)
        wait_for(lambda: all(a > b for a, b in zip(reads, before, strict=True)))

    barrier = threading.Barrier(len(runs), action=all_parked)
    real = storage.checks

    def gated(*args, **kwargs):
        barrier.wait(60)
        return real(*args, **kwargs)

    monkeypatch.setattr(storage, "checks", gated)
    savers = [
        Worker(store.save, run["id"], summary, arrays, 0, "synthetic")
        for run, (summary, arrays) in zip(runs, _fresh(saved), strict=True)
    ]
    for s in savers:
        s.start()
    assert all(s.get(60) for s in savers)
    stop.set()
    for r in readers:
        r.get()
    assert not errors and not barrier.broken
    monkeypatch.setattr(storage, "checks", real)
    store.finish(job, "complete")
    actual = _state(store, job)
    assert actual[0] == expected[0] == "complete"
    for (o1, s1, sum1, ser1), (o2, s2, sum2, ser2) in zip(
        actual[1], expected[1], strict=True
    ):
        assert (o1, s1, sum1) == (o2, s2, sum2)
        _bitwise(ser1, ser2)
    store.close()


def test_parked_at_helper_sees_lock_waits():
    """自检：`parked_at` 能认出停在 `with lock:` 上的线程（其他用例依赖它判断窗口）。"""
    lock = threading.Lock()

    def target():
        with lock:
            return 1

    lock.acquire()
    worker = Worker(target)
    worker.start()
    wait_for(lambda: parked_at(worker, target, "with lock:"))
    lock.release()
    assert worker.get() == 1
