"""缓存格式 4（P-004）：后台写入、证券表只存一次、键核对与损坏自愈、旧格式失效。"""

import json
import subprocess
import sys

import numpy as np
import pytest

import ashare.risk.cache as cache_module
from ashare.risk import ModelCache
from ashare.risk.series import model_series
from test.risk.test_risk_review import config
from test.risk.test_series import source


def _same(a, b):
    assert a.date == b.date and a.metadata == b.metadata and a.symbols == b.symbols
    for name in ("B", "F", "D"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


def _snapshots(directory):
    return sorted(directory.glob("*.npz"))


def test_symbols_stored_once_and_snapshot_carries_its_key(tmp_path):
    p, c = source(30, 40), config()
    list(model_series(p, config=c, cache=tmp_path))
    files = _snapshots(tmp_path)
    assert len(files) == 30
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            assert "symbols" not in data.files
            assert str(data["key"]) == path.stem.split("-", 1)[1]
    stored = np.load(tmp_path / ".symbols.npy", allow_pickle=False)
    assert tuple(stored.tolist()) == tuple(p.symbols)
    marker = json.loads((tmp_path / ".generation.json").read_text())
    assert marker["format"] == cache_module.FORMAT == 4
    cache = ModelCache(tmp_path)
    assert cache.stats()["bytes"] >= (tmp_path / ".symbols.npy").stat().st_size
    assert cache.clear() > 0 and not (tmp_path / ".symbols.npy").exists()


@pytest.mark.parametrize("damage", ("truncate", "swap", "garbage", "symbols"))
def test_damaged_snapshot_is_a_miss_and_recomputed(tmp_path, damage):
    p, c = source(30, 40), config()
    expected = list(model_series(p, config=c))
    list(model_series(p, config=c, cache=tmp_path))
    files = _snapshots(tmp_path)
    target = files[27]
    if damage == "truncate":
        target.write_bytes(target.read_bytes()[: target.stat().st_size // 2])
    elif damage == "swap":
        # 换成另一份结构完全有效的快照：键不符必须当未命中。
        target.write_bytes(files[26].read_bytes())
    elif damage == "garbage":
        target.write_bytes(b"not a zip file")
    else:
        (tmp_path / ".symbols.npy").write_bytes(b"\x93NUMPY broken")
    actual = list(model_series(p, config=c, cache=tmp_path, start=25))
    for a, b in zip(actual, expected[25:], strict=True):
        _same(a, b)
    with np.load(target, allow_pickle=False) as data:
        assert str(data["key"]) == target.stem.split("-", 1)[1]


def test_old_format_directory_is_cleared(tmp_path):
    p, c = source(25, 30), config()
    list(model_series(p, config=c, cache=tmp_path))
    marker = tmp_path / ".generation.json"
    content = json.loads(marker.read_text())
    content.pop("format")
    marker.write_text(json.dumps(content))
    # 旧格式目录里伪造一份「看上去能读」的旧式快照（压缩、无键）。
    old = _snapshots(tmp_path)[-1]
    with np.load(old, allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files if k != "key"}
    np.savez_compressed(old, symbols=np.asarray(p.symbols), **arrays)
    cache = ModelCache(tmp_path)
    assert cache.generation is None
    actual = list(model_series(p, config=c, cache=cache))
    for a, b in zip(actual, list(model_series(p, config=c)), strict=True):
        _same(a, b)
    with np.load(_snapshots(tmp_path)[-1], allow_pickle=False) as data:
        assert "key" in data.files and "symbols" not in data.files


def test_write_failure_is_reported(tmp_path, monkeypatch):
    p, c = source(25, 30), config()
    calls = []
    original = cache_module._savez

    def failing(stream, arrays):
        calls.append(1)
        if len(calls) == 10:
            raise OSError("disk full")
        return original(stream, arrays)

    monkeypatch.setattr(cache_module, "_savez", failing)
    with pytest.raises(RuntimeError, match="缓存写入失败") as info:
        list(model_series(p, config=c, cache=tmp_path))
    assert isinstance(info.value.__cause__, OSError)


def test_early_close_flushes_pending_writes(tmp_path):
    p, c = source(25, 30), config()
    generator = model_series(p, config=c, cache=tmp_path)
    for _ in range(12):
        next(generator)
    generator.close()
    assert len(_snapshots(tmp_path)) >= 12
    for path in _snapshots(tmp_path):
        with np.load(path, allow_pickle=False) as data:
            assert str(data["key"]) == path.stem.split("-", 1)[1]


@pytest.mark.parametrize("action", ("stats", "clear"))
def test_cli_does_not_create_missing_directory(tmp_path, action):
    missing = tmp_path / "absent"
    out = subprocess.run(
        [sys.executable, "-m", "ashare.risk.cache", action, str(missing)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    value = json.loads(out)
    assert value.get("bytes", value.get("removed_bytes")) == 0
    assert not missing.exists()
