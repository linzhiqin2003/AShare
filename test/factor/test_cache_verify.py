# ruff: noqa: E402
"""P-012：缓存命中校验改为分块并行 SHA-256，各类损坏和替换仍判为未命中。"""

import hashlib
import json
import os
import shutil
from pathlib import Path

import panelops

import ashare

ROOT = Path(__file__).resolve().parents[2]
assert Path(ashare.__file__).resolve().is_relative_to(ROOT)
assert Path(panelops.__file__).resolve().is_relative_to(ROOT)

import numpy as np
import pytest

import ashare.factor as af
import ashare.factor.engine as engine


class Source:
    def __init__(self):
        rng = np.random.default_rng(3)
        self.fields = {"x": rng.normal(size=(40, 9)), "y": rng.normal(size=(40, 9))}
        self.dates = list(range(40))
        self.symbols = list(range(9))
        self.version = "v"

    def __getitem__(self, name):
        return self.fields[name]


def reference(path, chunk):
    data = path.read_bytes()
    total = hashlib.sha256(f"ashare-factor-cache:2:{len(data)}:".encode())
    for start in range(0, len(data), chunk):
        total.update(hashlib.sha256(data[start : start + chunk]).digest())
    return total.hexdigest()


@pytest.mark.parametrize("size", [0, 1, 63, 64, 65, 640, 1000])
def test_checksum_matches_reference(tmp_path, monkeypatch, size):
    monkeypatch.setattr(engine, "_CHUNK", 64)
    path = tmp_path / "f.bin"
    path.write_bytes(np.random.default_rng(size).bytes(size))
    assert engine._checksum(path) == reference(path, 64)


@pytest.fixture
def cached(tmp_path, monkeypatch):
    monkeypatch.setattr(engine, "_CHUNK", 256)  # 小文件也分成多块
    source = Source()
    expression = af.call("ts_mean", af.field("x"), window=3)
    ev = af.Evaluator(tmp_path)
    expected = np.array(ev.evaluate(expression, source))
    assert not ev.cache_hit
    assert np.array_equal(ev.evaluate(expression, source), expected, equal_nan=True)
    assert ev.cache_hit
    path = next(tmp_path.glob("*.npy"))
    return ev, source, expression, expected, path


def recomputes(ev, source, expression, expected):
    got = ev.evaluate(expression, source)
    assert not ev.cache_hit
    assert got.tobytes() == expected.tobytes()
    ev.evaluate(expression, source)  # 已原子重写，下一次命中
    assert ev.cache_hit


@pytest.mark.parametrize("where", ["header", "first", "middle", "last"])
def test_flipped_byte(cached, where):
    ev, source, expression, expected, path = cached
    data = bytearray(path.read_bytes())
    at = {"header": 20, "first": 128, "middle": len(data) // 2, "last": -1}[where]
    data[at] ^= 0x01
    path.write_bytes(bytes(data))
    recomputes(ev, source, expression, expected)


@pytest.mark.parametrize("keep", [0, 1, 0.5, -1])
def test_truncated_and_zero_length(cached, keep):
    ev, source, expression, expected, path = cached
    data = path.read_bytes()
    cut = {0: 0, 1: 1, 0.5: len(data) // 2, -1: len(data) - 1}[keep]
    path.write_bytes(data[:cut])
    recomputes(ev, source, expression, expected)


def test_deleted_metadata(cached):
    ev, source, expression, expected, path = cached
    path.with_suffix(".json").unlink()
    recomputes(ev, source, expression, expected)


@pytest.mark.parametrize(
    "edit",
    [
        lambda m: "not json",
        lambda m: json.dumps([m]),
        lambda m: json.dumps({k: v for k, v in m.items() if k != "tree_sha256"}),
        lambda m: json.dumps(m | {"tree_sha256": "0" * 64}),
        lambda m: json.dumps(m | {"bytes": m["bytes"] + 1}),
        lambda m: json.dumps(m | {"format": 1}),
        lambda m: json.dumps(m | {"key": "0" * 64}),
        lambda m: json.dumps({"sha256": m["tree_sha256"]}),  # 旧格式
        lambda m: "",
    ],
)
def test_bad_metadata(cached, edit):
    ev, source, expression, expected, path = cached
    meta = path.with_suffix(".json")
    meta.write_text(edit(json.loads(meta.read_text())))
    recomputes(ev, source, expression, expected)


def test_replaced_by_another_entry(cached):
    """把另一个表达式的缓存文件对（数据和元数据都自洽）换进来也判为未命中。"""
    ev, source, expression, expected, path = cached
    other = af.call("ts_mean", af.field("y"), window=3)
    ev.evaluate(other, source)
    other_path = next(p for p in path.parent.glob("*.npy") if p != path)
    shutil.copyfile(other_path, path)
    shutil.copyfile(other_path.with_suffix(".json"), path.with_suffix(".json"))
    recomputes(ev, source, expression, expected)


def test_metadata_fields(cached):
    _, _, _, _, path = cached
    meta = json.loads(path.with_suffix(".json").read_text())
    assert meta == {
        "format": 2,
        "key": path.stem,
        "bytes": path.stat().st_size,
        "tree_sha256": reference(path, 256),
    }


@pytest.mark.parametrize("field", ["format", "key", "bytes", "tree_sha256"])
def test_missing_metadata_field(cached, field):
    ev, source, expression, expected, path = cached
    meta = path.with_suffix(".json")
    data = json.loads(meta.read_text())
    del data[field]
    meta.write_text(json.dumps(data))
    recomputes(ev, source, expression, expected)


def test_short_read_raises_and_counts_as_miss(cached, monkeypatch):
    """校验中文件变短：_checksum 报 EOFError，求值按未命中处理而不是抛出。"""
    ev, source, expression, expected, path = cached
    inode, real = path.stat().st_ino, os.preadv

    def shrunk(fd, buffers, offset):
        # 只让这份缓存文件「变短」；重算后写新文件时照常读
        return 0 if os.fstat(fd).st_ino == inode else real(fd, buffers, offset)

    monkeypatch.setattr(engine.os, "preadv", shrunk)
    with pytest.raises(EOFError):
        engine._checksum(path)
    got = ev.evaluate(expression, source)
    assert not ev.cache_hit and got.tobytes() == expected.tobytes()


@pytest.mark.parametrize("damage", ["flip", "truncate", "metadata"])
def test_batch_entry_rejects_damaged_cache(tmp_path, monkeypatch, damage):
    monkeypatch.setattr(engine, "_CHUNK", 256)
    source = Source()
    exprs = {
        "a": af.call("ts_mean", af.field("x"), window=3),
        "b": af.call("abs", af.field("y")),
    }
    expected = {
        k: np.array(af.Evaluator().evaluate(e, source)) for k, e in exprs.items()
    }
    ev = af.Evaluator(tmp_path)
    list(ev.evaluate_many(exprs, source))
    list(ev.evaluate_many(exprs, source))
    assert ev.cache_hit and ev.last_batch["cache_hits"] == 2
    path = ev._cache_path(exprs["a"], source, None, "all")
    if damage == "flip":
        data = bytearray(path.read_bytes())
        data[len(data) // 2] ^= 0x10
        path.write_bytes(bytes(data))
    elif damage == "truncate":
        path.write_bytes(path.read_bytes()[:-3])
    else:
        meta = path.with_suffix(".json")
        meta.write_text(json.dumps(json.loads(meta.read_text()) | {"bytes": 1}))
    got = dict(ev.evaluate_many(exprs, source))
    assert ev.last_batch["cache_hits"] == 1 and not ev.cache_hit
    for name in exprs:
        assert got[name].tobytes() == expected[name].tobytes()
    list(ev.evaluate_many(exprs, source))
    assert ev.cache_hit


def test_fd_stays_open_until_every_chunk_finishes(tmp_path, monkeypatch):
    """一块读失败时，其余块读完才关 fd，不会读到复用同一 fd 号的别的文件。"""
    import threading
    import time

    monkeypatch.setattr(engine, "_CHUNK", 64)
    path = tmp_path / "f.bin"
    path.write_bytes(bytes(64 * 6))
    inode = path.stat().st_ino
    seen, lock = [], threading.Lock()
    started = threading.Event()
    real = engine._chunk_digest

    def chunk(fd, start, length):
        if start == 0:
            # 等别的块真正开始读再失败，否则它们会被直接取消，测不到关 fd 的时机
            assert started.wait(10)
            raise OSError("injected")
        started.set()
        time.sleep(0.2)
        try:
            observed = os.fstat(fd).st_ino
        except OSError:
            observed = None
        with lock:
            seen.append(observed)
        return real(fd, start, length)

    monkeypatch.setattr(engine, "_chunk_digest", chunk)
    with pytest.raises(OSError, match="injected"):
        engine._checksum(path)
    # 失败后立刻打开别的文件，若 fd 已关就会拿到同一个号
    other = tmp_path / "other.bin"
    other.write_bytes(b"x")
    with other.open("rb"):
        time.sleep(0.3)
    assert seen and all(s == inode for s in seen)


FORK_CHILD = """
import os, signal, sys, numpy as np
import ashare.factor as af
root = sys.argv[1]
class S:
    dates = list(range(50)); symbols = list(range(40)); version = "fork"
    f = {"a": np.random.default_rng(0).normal(size=(50, 40))}
    def __getitem__(self, k): return self.f[k]
e = af.call("ts_mean", af.field("a"), window=5)
af.Evaluator(root).evaluate(e, S())
ev = af.Evaluator(root); ev.evaluate(e, S()); assert ev.cache_hit
pid = os.fork()
if pid == 0:
    signal.alarm(20)
    ev = af.Evaluator(root); ev.evaluate(e, S())
    os._exit(0 if ev.cache_hit else 3)
_, status = os.waitpid(pid, 0)
sys.exit(os.waitstatus_to_exitcode(status))
"""


@pytest.mark.skipif(not hasattr(os, "fork"), reason="需要 fork")
def test_cache_hit_in_forked_child_does_not_hang(tmp_path):
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-c", FORK_CHILD, str(tmp_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (result.returncode, result.stderr[-2000:])
