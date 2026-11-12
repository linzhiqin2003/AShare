# ruff: noqa: E402
"""P-012：缓存命中校验改为分块并行 SHA-256，各类损坏和替换仍判为未命中。"""

import hashlib
import json
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
