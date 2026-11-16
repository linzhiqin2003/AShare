"""§7/§9.3，review_d2_r3.md P1-3：同 inode 改写竞态。"""

import importlib
import io
import os
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from ashare.data.spec import open
from test.spec.conftest import resign


@pytest.mark.parametrize("cached", [False, True])
def test_s7_r3_review_mutable_stream_race(package, monkeypatch, cached):
    reader = importlib.import_module("ashare.data.spec.reader")
    reader._VERIFIED.clear()
    ds = open(package)
    path = package / "bars/2024.parquet"
    original = path.read_bytes()
    replacement = io.BytesIO()
    pl.read_parquet(io.BytesIO(original)).with_columns(
        pl.lit(999.0).alias("close")
    ).write_parquet(replacement)
    state = SimpleNamespace(
        **{
            n: getattr(path.stat(), n)
            for n in ("st_size", "st_mtime_ns", "st_dev", "st_ino", "st_ctime_ns")
        }
    )
    if cached:
        ds.read(fields="close")
    actual_open, actual_fstat, actual_verify = (
        Path.open,
        reader.os.fstat,
        reader._verify,
    )

    class MutableFile(io.BytesIO):
        def fileno(self):
            return 123456789

    def opened(p, mode="r", *args, **kwargs):
        return (
            MutableFile(original)
            if p == path
            else actual_open(p, mode, *args, **kwargs)
        )

    def overwrite(p, entry, stream=None):
        result = actual_verify(p, entry, stream)
        if p == path:
            stream.seek(0)
            stream.truncate()
            stream.write(replacement.getvalue())
            stream.seek(0)
            state.st_size = len(replacement.getvalue())
            state.st_ctime_ns += 1
        return result

    monkeypatch.setattr(Path, "open", opened)
    monkeypatch.setattr(
        reader.os, "fstat", lambda fd: state if fd == 123456789 else actual_fstat(fd)
    )
    monkeypatch.setattr(reader, "_verify", overwrite)
    if cached:
        with pytest.raises(ValueError, match="文件.*改变"):
            ds.read(fields="close")
    else:
        # 冷读必须解析已签名快照，不能解析核对后改写的 999。
        assert (
            ds.read(fields="close")["close"].to_list()
            == pl.read_parquet(io.BytesIO(original))["close"].to_list()
        )


@pytest.mark.parametrize("entry", ["read", "panel"])
def test_s7_r3_real_inode_overwrite_rejected(package, monkeypatch, entry):
    reader = importlib.import_module("ashare.data.spec.reader")
    reader._VERIFIED.clear()
    ds = open(package)
    ds.read(fields="close")  # 先建立真实文件缓存
    path = package / "bars/2024.parquet"
    before = path.stat()
    replacement = io.BytesIO()
    pl.read_parquet(path).with_columns(pl.lit(999.0).alias("close")).write_parquet(
        replacement
    )
    actual_verify = reader._verify
    written = []

    def overwrite(p, desc, stream=None):
        result = actual_verify(p, desc, stream)
        if p == path and not written:
            with path.open("r+b") as target:
                target.write(replacement.getvalue())
                target.truncate()
                target.flush()
                os.fsync(target.fileno())
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
            assert path.stat().st_ino == before.st_ino
            written.append(True)
        return result

    monkeypatch.setattr(reader, "_verify", overwrite)
    with pytest.raises(ValueError, match="文件.*改变"):
        getattr(ds, entry)(fields="close")
    assert written


def test_s7_r3_cold_snapshot_is_hashed_and_parsed(package, monkeypatch):
    reader = importlib.import_module("ashare.data.spec.reader")
    reader._VERIFIED.clear()
    ds = open(package)
    path = package / "bars/2024.parquet"
    original = path.read_bytes()
    actual_verify = reader._verify
    changed = []

    def corrupt(p, desc, stream=None):
        result = actual_verify(p, desc, stream)
        if p == path and not changed:
            path.write_bytes(b"corruption after verification")
            changed.append(True)
        return result

    monkeypatch.setattr(reader, "_verify", corrupt)
    assert (
        ds.read(fields="close")["close"].to_list()
        == pl.read_parquet(io.BytesIO(original))["close"].to_list()
    )
    # 下一次入口必须重新识别已经改变的文件，不能复用上一次的快照。
    with pytest.raises(ValueError, match="大小|哈希"):
        ds.read(fields="close")


def test_s7_r3_cached_identity_checked_after_parquet_read(package, monkeypatch):
    """来源：P1-3 要求在实际读取完成之后检查，不能只检查扫描创建时。"""
    ds = open(package)
    ds.read(fields="close")
    path = package / "bars/2024.parquet"
    before = path.stat()
    replacement = io.BytesIO()
    pl.read_parquet(path).with_columns(pl.lit(999.0).alias("close")).write_parquet(
        replacement
    )
    actual_collect = pl.LazyFrame.collect
    changed = []

    def corrupt_after_collect(frame, *args, **kwargs):
        result = actual_collect(frame, *args, **kwargs)
        if not changed:
            with path.open("r+b") as stream:
                stream.write(replacement.getvalue())
                stream.truncate()
                stream.flush()
                os.fsync(stream.fileno())
            os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
            assert path.stat().st_ino == before.st_ino
            changed.append(True)
        return result

    monkeypatch.setattr(pl.LazyFrame, "collect", corrupt_after_collect)
    with pytest.raises(ValueError, match="文件.*改变"):
        ds.read(fields="close")
    assert changed


def test_s9_r3_cold_partitions_are_opened_incrementally(package, monkeypatch):
    """来源：P1-3 快照不得扩大为整包字节常驻，列投影仍下推到每个分区。"""
    reader = importlib.import_module("ashare.data.spec.reader")
    original = pl.read_parquet(package / "bars/2024.parquet")
    for year in (2025, 2026):
        original.with_columns(
            pl.col("date").dt.offset_by(f"{year - 2024}y")
        ).write_parquet(package / f"bars/{year}.parquet")
    resign(package)
    reader._VERIFIED.clear()
    seen = []
    actual_verify = reader._verify

    def verify(path, entry, stream=None):
        seen.append(path.name)
        return actual_verify(path, entry, stream)

    monkeypatch.setattr(reader, "_verify", verify)
    lazy = open(package)._scan("bars", fields="close")
    assert seen == ["2024.parquet"]
    assert "PROJECT 3/7 COLUMNS" in lazy.explain()
    result = lazy.collect(engine="streaming")
    assert result.height == 360 and result.columns == ["symbol", "date", "close"]
    assert seen == ["2024.parquet", "2025.parquet", "2026.parquet"]
