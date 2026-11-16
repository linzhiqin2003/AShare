"""§7、§9.3：独立审查 P1-2，缓存身份和校验/读取之间的路径替换。"""

import hashlib
import importlib
import os

import polars as pl
import pytest

from ashare.data.spec import open
from test.spec.conftest import resign


@pytest.mark.parametrize("replace", [False, True])
def test_s7_same_size_restored_mtime_cannot_bypass_hash(tmp_path, replace):
    reader = importlib.import_module("ashare.data.spec.reader")
    path = tmp_path / "all.parquet"
    other = tmp_path / "changed.parquet"
    for target, value in [(path, 10.0), (other, 20.0)]:
        pl.DataFrame({"value": [value]}).write_parquet(
            target, compression="uncompressed"
        )
    before = path.stat()
    assert before.st_size == other.stat().st_size
    entry = dict(
        bytes=before.st_size,
        hash="sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
    )
    reader._VERIFIED.clear()
    reader._verify(path, entry)
    if replace:
        os.replace(other, path)
    else:
        path.write_bytes(other.read_bytes())
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    with pytest.raises(ValueError, match="哈希"):
        reader._verify(path, entry)


def test_s7_read_uses_verified_file_instance(package, monkeypatch):
    reader = importlib.import_module("ashare.data.spec.reader")
    path = package / "bars/2024.parquet"
    expected = pl.read_parquet(path)["close"].to_list()
    replacement = package.parent / "replacement.parquet"
    pl.read_parquet(path).with_columns(pl.lit(999.0).alias("close")).write_parquet(
        replacement
    )
    resign(package)
    actual = reader._verify

    def swap_after_verification(p, entry, *args, **kwargs):
        result = actual(p, entry, *args, **kwargs)
        if p == path:
            os.replace(replacement, path)
        return result

    monkeypatch.setattr(reader, "_verify", swap_after_verification)
    assert open(package).read(fields="close")["close"].to_list() == expected


def test_s9_as_of_prunes_future_bar_partition(package):
    path = package / "bars/2025.parquet"
    pl.read_parquet(package / "bars/2024.parquet").with_columns(
        pl.col("date").dt.offset_by("1y")
    ).write_parquet(path)
    resign(package)
    path.write_bytes(b"unselected future file")
    ds = open(package)
    assert ds.read(as_of="2024-01-02T07:00:00Z", fields="close").height == 3
    with pytest.raises(ValueError, match="大小|哈希"):
        ds.read(fields="close")
