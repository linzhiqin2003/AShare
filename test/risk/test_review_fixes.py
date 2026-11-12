"""性能轮独立复核后的修补：交出的日子、fork、写失败规则、证券表校验、参数校验、测试空档。"""

import atexit
import os
import subprocess
import sys
import textwrap
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import ashare.risk.cache as cache_module
import ashare.risk.series as series
from ashare.risk import FundamentalConfig, ModelCache
from ashare.risk.data import MemoryPanel
from ashare.risk.series import model_series
from ashare.risk.specific import SpecificMoments, direct_variances
from test.risk.test_risk_review import config
from test.risk.test_series import source

ROOT = Path(__file__).resolve().parents[2]


def _same(a, b):
    assert a.date == b.date and a.metadata == b.metadata and a.symbols == b.symbols
    for name in ("B", "F", "D"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


def _same_series(actual, expected):
    assert [m.date for m in actual] == [m.date for m in expected]
    for a, b in zip(actual, expected, strict=True):
        _same(a, b)


def _run(code, timeout=90):
    """在全新的解释器里跑一段代码（fork、退出时行为都要隔离在子进程里测）。"""
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )


# P0-1：交出的日子与缓存命中多少无关 ---------------------------------------------


@pytest.fixture(scope="module")
def reference():
    p, c = source(90, 40), config()  # history=20
    return p, c, list(model_series(p, config=c, end=85))


@pytest.mark.parametrize(
    ("cached", "start"),
    [
        (80, 0),  # 部分命中，start 远早于窗口 [60, 80)
        (80, 30),
        (80, 59),  # 窗口前一天
        (80, 60),  # 窗口起点
        (80, 70),  # 窗口内
        (80, 80),  # 等于第一个未命中日
        (80, 85),  # 空区间
        (85, 0),  # 全部命中
        (85, 40),
        (85, 84),
        (0, 0),  # 全部未命中
        (0, 37),
        (10, 3),  # 窗口从第 0 天起
    ],
)
def test_yielded_days_do_not_depend_on_cache_hits(tmp_path, reference, cached, start):
    p, c, full = reference
    if cached:
        list(model_series(p, config=c, cache=tmp_path, end=cached))
    got = list(model_series(p, config=c, cache=tmp_path, start=start, end=85))
    _same_series(got, full[start:])
    # 调用方用 enumerate 当日期偏移也不会错位。
    for offset, model in enumerate(got):
        assert model.date == str(p.dates[start + offset])


@pytest.mark.parametrize(
    ("cached", "start", "broken"), [(80, 0, 40), (80, 65, 65), (15, 10, 10)]
)
def test_damaged_snapshot_before_the_window_still_yields_every_day(
    tmp_path, reference, cached, start, broken
):
    """预读只取了部分字段的前一天、窗口外的日子坏了：都要重新计划，交出的日子不变。"""
    p, c, full = reference
    list(model_series(p, config=c, cache=tmp_path, end=cached))
    target = next(tmp_path.glob(f"{broken:06d}-*.npz"))
    target.write_bytes(b"damaged")
    got = list(model_series(p, config=c, cache=tmp_path, start=start, end=85))
    _same_series(got, full[start:])


# P0-2：fork ------------------------------------------------------------------

_PREAMBLE = """
import os, sys, time, threading
import numpy as np
import ashare.risk.cache as cache_module
from ashare.risk import ModelCache
from ashare.risk.model import RiskModel

def model(day):
    rng = np.random.default_rng(day)
    return RiskModel(("a", "b", "c"), rng.normal(size=(3, 2)), ("x", "y"),
                     np.eye(2), np.full(3, 0.1), f"2020-01-{day + 1:02d}", {})

STATE = {"f": np.zeros(2), "residual": np.zeros(3), "base_f": np.ones(2),
         "base_d": np.ones(3), "bias": np.zeros(2), "estimation": np.ones(3),
         "cap": np.ones(3)}

def wait_child(pid, seconds=20):
    deadline = time.time() + seconds
    while time.time() < deadline:
        done, status = os.waitpid(pid, os.WNOHANG)
        if done:
            return os.waitstatus_to_exitcode(status)
        time.sleep(0.05)
    os.kill(pid, 9)
    os.waitpid(pid, 0)
    return "hung"
"""


def test_forked_child_does_not_hang_and_does_not_take_over_parent_writes(tmp_path):
    code = (
        _PREAMBLE
        + f"""
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
cache = ModelCache({str(tmp_path)!r})
cache.prepare("g", ("a", "b", "c"))
cache.save("0" * 64, model(0), STATE, day=0, latest=False)
cache.flush()
gate, entered = threading.Event(), threading.Event()
original = cache_module._write_atomic
def slow(directory, path, write):
    if path.name.startswith("000005"):
        entered.set()
        gate.wait()
    return original(directory, path, write)
cache_module._write_atomic = slow
cache.save("5" * 64, model(5), STATE, day=5, latest=False)  # 父进程排队中、尚未落盘
entered.wait(10)
pid = os.fork()
if pid == 0:
    try:
        cache_module._write_atomic = original
        cache.save("6" * 64, model(6), STATE, day=6, latest=False)
        ok = cache.load("6" * 64, day=6, with_model=False) is not None
        missing = cache.load("5" * 64, day=5, with_model=False) is None
        cache.flush()
        cache.close()
        os._exit(0 if ok and missing else 1)
    except BaseException:
        os._exit(2)
code = wait_child(pid)
gate.set()
cache.flush()
written = sorted(p.name[:6] for p in cache.directory.glob("*.npz"))
print("child", code, "files", written)
"""
    )
    result = _run(code)
    assert result.returncode == 0, result.stderr
    assert "child 0 files ['000000', '000005', '000006']" in result.stdout


def test_exposure_pool_is_rebuilt_in_a_forked_child():
    code = (
        _PREAMBLE
        + """
import warnings
warnings.simplefilter("ignore", DeprecationWarning)
from ashare.risk.series import _ExposurePool
pool = _ExposurePool(3)
assert pool.map(lambda x: x * 2, range(6)) == [0, 2, 4, 6, 8, 10]
pid = os.fork()
if pid == 0:
    ok = pool.map(lambda x: x + 1, range(6)) == [1, 2, 3, 4, 5, 6]
    pool.shutdown()
    os._exit(0 if ok else 1)
print("child", wait_child(pid))
pool.shutdown()
"""
    )
    result = _run(code)
    assert result.returncode == 0, result.stderr
    assert "child 0" in result.stdout


# P1-1：写失败规则 --------------------------------------------------------------


def _failing_writes(monkeypatch, after=0):
    """第 after 次之后的快照写入都失败。"""
    calls = []
    original = cache_module._write_atomic

    def write(directory, path, job):
        if path.suffix == ".npz" and path.name[0].isdigit():
            calls.append(path.name)
            if len(calls) > after:
                raise OSError("disk full")
        return original(directory, path, job)

    monkeypatch.setattr(cache_module, "_write_atomic", write)
    return calls


def test_write_failure_is_sticky_and_raised_by_flush_close_and_context(
    tmp_path, monkeypatch
):
    p, c = source(30, 40), config()
    models = list(model_series(p, config=c))
    _failing_writes(monkeypatch)
    state = {"f": np.zeros(3)}
    with pytest.raises(RuntimeError, match="写入失败") as caught:
        with ModelCache(tmp_path / "a") as cache:
            cache.prepare("g", p.symbols)
            cache.save("a" * 64, models[-1], state, day=0, latest=False)
    assert isinstance(caught.value.__cause__, OSError)
    for action in (
        lambda: cache.flush(),
        lambda: cache.close(),
        lambda: cache.save("b" * 64, models[-1], state, day=1, latest=False),
        lambda: cache.load("a" * 64, day=0, with_model=False),
    ):
        with pytest.raises(RuntimeError, match="写入失败"):
            action()


def test_queued_writes_are_skipped_after_a_failure(tmp_path, monkeypatch):
    p, c = source(30, 40), config()
    models = list(model_series(p, config=c))
    gate, entered = threading.Event(), threading.Event()
    original = cache_module._write_atomic

    def write(directory, path, job):
        if path.name.startswith("000000"):
            entered.set()
            gate.wait(10)
            raise OSError("disk full")
        return original(directory, path, job)

    monkeypatch.setattr(cache_module, "_write_atomic", write)
    cache = ModelCache(tmp_path)
    cache.prepare("g", p.symbols)
    state = {"f": np.zeros(3)}
    cache.save("0" * 64, models[-1], state, day=0, latest=False)
    entered.wait(10)
    for day in range(1, 6):  # 失败前已排队
        cache.save(f"{day}" * 64, models[-1], state, day=day, latest=False)
    gate.set()
    with pytest.raises(RuntimeError):
        cache.flush()
    assert list(tmp_path.glob("*.npz")) == []


@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
def test_generator_collected_after_break_reports_and_poisons_the_cache(
    tmp_path, monkeypatch, capsys
):
    p, c = source(40, 40), config()
    _failing_writes(monkeypatch, after=10)
    cache = ModelCache(tmp_path)
    for i, _ in enumerate(model_series(p, config=c, cache=cache)):
        if i == 20:
            break  # 生成器在这里被回收关闭，异常无法交给调用方
    assert "生成器关闭时" in capsys.readouterr().err
    with pytest.raises(RuntimeError, match="写入失败"):
        cache.flush()
    with pytest.raises(RuntimeError, match="写入失败"):
        cache.load(cache.entries[0], day=0, with_model=False)


def test_write_failure_at_interpreter_exit_goes_to_stderr(tmp_path):
    code = (
        _PREAMBLE
        + f"""
def broken(directory, path, write):
    raise OSError("disk full")
cache = ModelCache({str(tmp_path)!r})
cache.prepare("g", ("a", "b", "c"))
cache_module._write_atomic = broken
cache.save("0" * 64, model(0), STATE, day=0, latest=False)
print("returned")
"""
    )
    result = _run(code)
    assert "returned" in result.stdout
    assert "解释器退出时" in result.stderr and "写入失败" in result.stderr


def test_atexit_registrations_do_not_grow_with_instances(tmp_path):
    p, c = source(30, 40), config()
    model = list(model_series(p, config=c))[-1]
    before = atexit._ncallbacks()
    for i in range(20):
        cache = ModelCache(tmp_path / str(i))
        cache.prepare("g", p.symbols)
        cache.save("a" * 64, model, {"f": np.zeros(3)}, day=0, latest=False)
        cache.close()
    assert atexit._ncallbacks() == before


def test_writer_queue_is_bounded(tmp_path, monkeypatch):
    p, c = source(30, 40), config()
    model = list(model_series(p, config=c))[-1]
    gate, entered = threading.Event(), threading.Event()
    original = cache_module._write_atomic

    def write(directory, path, job):
        if path.suffix == ".npz":
            entered.set()
            gate.wait(20)
        return original(directory, path, job)

    monkeypatch.setattr(cache_module, "_write_atomic", write)
    cache = ModelCache(tmp_path)
    cache.prepare("g", p.symbols)
    returned = []

    def produce():
        for day in range(200):
            cache.save(f"{day:064x}", model, {"f": np.zeros(3)}, day=day, latest=False)
            returned.append(day)

    worker = threading.Thread(target=produce, daemon=True)
    worker.start()
    entered.wait(10)
    time.sleep(0.5)
    queued = len(returned)
    gate.set()
    worker.join(30)
    cache.flush()
    assert queued <= cache_module._QUEUE_LIMIT + 2
    assert len(returned) == 200


def test_load_waits_for_its_own_pending_write(tmp_path, monkeypatch):
    p, c = source(30, 40), config()
    model = list(model_series(p, config=c))[-1]
    original = cache_module._write_atomic

    def slow(directory, path, job):
        time.sleep(0.3)
        return original(directory, path, job)

    monkeypatch.setattr(cache_module, "_write_atomic", slow)
    cache = ModelCache(tmp_path)
    cache.prepare("g", p.symbols)
    state = {name: np.arange(3.0) for name in cache_module.STATE_FIELDS}
    cache.save("a" * 64, model, state, day=0, latest=False)
    loaded = cache.load("a" * 64, day=0, with_model=False)
    assert loaded is not None
    np.testing.assert_array_equal(loaded[1]["f"], np.arange(3.0))


# P1-2 / P1-3 / P1-4：证券表、fields、翻比特 -----------------------------------------


def test_tampered_or_unreadable_symbol_table_is_a_miss_and_keeps_snapshots(tmp_path):
    p, c = source(30, 40), config()
    expected = list(model_series(p, config=c))
    list(model_series(p, config=c, cache=tmp_path))
    table = tmp_path / ".symbols.npy"
    snapshots = sorted(tmp_path.glob("*.npz"))
    cache = ModelCache(tmp_path)
    day = 29
    np.save(table, np.asarray(["X", *p.symbols[1:]]))
    assert ModelCache(tmp_path).load(cache.entries[day], day=day) is None
    table.write_bytes(b"\x93NUMPY broken")
    fresh = ModelCache(tmp_path)
    assert fresh.load(fresh.entries[day], day=day) is None
    assert day in fresh.entries and sorted(tmp_path.glob("*.npz")) == snapshots
    _same_series(list(model_series(p, config=c, cache=tmp_path)), expected)


@pytest.mark.parametrize("fields", [("residul",), (), "f", ("f", 3)])
def test_bad_fields_raise_before_touching_the_cache(tmp_path, fields):
    p, c = source(30, 40), config()
    list(model_series(p, config=c, cache=tmp_path))
    cache = ModelCache(tmp_path)
    before = sorted(tmp_path.iterdir())
    with pytest.raises(ValueError, match="fields"):
        cache.load(cache.entries[20], day=20, with_model=False, fields=fields)
    assert sorted(tmp_path.iterdir()) == before and 20 in cache.entries


def test_random_bit_flips_are_misses_or_identical(tmp_path):
    p, c = source(30, 40), config()
    list(model_series(p, config=c, cache=tmp_path))
    target = sorted(tmp_path.glob("*.npz"))[20]
    original = target.read_bytes()
    day, key = 20, target.stem.split("-", 1)[1]
    good = ModelCache(tmp_path).load(key, day=day, with_model=False)[1]
    rng = np.random.default_rng(5)
    positions = set(rng.integers(0, len(original), 300).tolist())
    positions |= set(range(64)) | set(range(len(original) - 100, len(original)))
    for position in sorted(positions):
        for bit in (0, 3, 7):
            damaged = bytearray(original)
            damaged[position] ^= 1 << bit
            target.write_bytes(bytes(damaged))
            got = ModelCache(tmp_path).load(key, day=day, with_model=False)
            if got is not None:
                for name, value in good.items():
                    assert got[1][name].dtype == value.dtype
                    np.testing.assert_array_equal(got[1][name], value)
    target.write_bytes(original)


# P1-5：测试空档 ----------------------------------------------------------------


def test_threshold_gate_routes_columns_near_the_floor_to_direct():
    """base、corr0 距 variance_floor 相对 1e-9 以内的列必须逐位等于从头算（走回退）。"""
    c = replace(FundamentalConfig(), history=40, specific_reanchor_days=7)
    rng = np.random.default_rng(29)
    t, n = 70, 600
    x = rng.normal(0, 1, (t, n)) + rng.normal(0, 3, n)
    base, corr0, _ = direct_variances(x[t - 40 :], c)
    which = np.arange(n) % 2 == 0
    offset = rng.uniform(-8e-10, 8e-10, n)
    x = x * np.sqrt(c.variance_floor * (1 + offset) / np.where(which, base, corr0))
    moments = SpecificMoments(n, c)
    for day in range(t):
        moments.push(day, x[day])
    got = moments.variances(t - 1)
    expected = direct_variances(x[t - 40 :], c)
    near = (np.abs(expected[0] / c.variance_floor - 1) <= 9e-10) | (
        np.abs(expected[1] / c.variance_floor - 1) <= 9e-10
    )
    assert near.sum() > n // 2
    for a, b in zip(got[:3], expected, strict=True):
        np.testing.assert_array_equal(a[near], b[near])


def test_exposure_pool_threads_are_shut_down(tmp_path):
    p, c = source(70, 40), config()

    def alive():
        return [t for t in threading.enumerate() if t.name.startswith("risk-exposure")]

    list(model_series(p, config=c, threads=3, batch_size=8))
    assert alive() == []
    generator = model_series(p, config=c, threads=3, batch_size=8)
    for i, _ in enumerate(generator):
        if i == 30:
            break
    generator.close()
    assert alive() == []


def test_row_digest_chunks_cover_every_day_of_long_panels(tmp_path):
    t, n = 600, 4
    rng = np.random.default_rng(3)
    fields = {"close": rng.uniform(1, 2, (t, n)), "is_listed": np.ones((t, n), bool)}
    dates = np.datetime64("2020-01-01") + np.arange(t)
    base = MemoryPanel(dates, ("a", "b", "c", "d"), "v", fields)
    cache = ModelCache(tmp_path)
    series._prefix_chain(base, None, t, cache, 4)
    cache.flush()
    for day in (254, 255, 256, 511, 512, 599):
        changed = {k: v.copy() for k, v in fields.items()}
        changed["close"][day, 1] += 0.5
        panel = MemoryPanel(dates, base.symbols, "v", changed)
        got = series._prefix_chain(panel, None, t, ModelCache(tmp_path), 4)
        assert got == series._prefix_chain(panel, None, t, None, 1), day
        series._prefix_chain(base, None, t, ModelCache(tmp_path), 4)


# P2 --------------------------------------------------------------------------


def test_default_threads_is_capped_and_env_is_strict(monkeypatch):
    monkeypatch.delenv("ASHARE_RISK_THREADS", raising=False)
    monkeypatch.setattr(series.os, "cpu_count", lambda: 128)
    monkeypatch.setattr(
        series.ctypes, "CDLL", lambda *a: (_ for _ in ()).throw(OSError())
    )
    assert series.default_threads() == series.MAX_DEFAULT_THREADS
    monkeypatch.setenv("ASHARE_RISK_THREADS", "")
    assert series.default_threads() == series.MAX_DEFAULT_THREADS
    for value in ("0", "٣", "99999", " 4", "4.0"):
        monkeypatch.setenv("ASHARE_RISK_THREADS", value)
        with pytest.raises(ValueError):
            series.default_threads()
    monkeypatch.setenv("ASHARE_RISK_THREADS", "12")
    assert series.default_threads() == 12
    with pytest.raises(ValueError):
        next(model_series(source(10, 20), config=config(), threads=1000))


def test_stale_temporaries_are_removed(tmp_path):
    cache = ModelCache(tmp_path)
    cache.prepare("g")
    old = tmp_path / f".{'a' * 32}.tmp"
    new = tmp_path / f".{'b' * 32}.tmp"
    other = tmp_path / "notes.tmp"
    for path in (old, new, other):
        path.write_bytes(b"x")
    stale = time.time() - cache_module._STALE_TEMPORARY - 60
    os.utime(old, (stale, stale))
    cache.prepare("g")
    assert not old.exists() and new.exists() and other.exists()
    cache.clear()
    assert not new.exists() and other.exists()


@pytest.mark.parametrize("value", [0, "no", 1])
def test_specific_incremental_must_be_bool(value):
    with pytest.raises(ValueError):
        replace(FundamentalConfig(), specific_incremental=value)
