"""逐分区构建与原子发布。未经独立 check 通过的目录永远不发布。"""

import ctypes
import fcntl
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Protocol

import polars as pl

from .check import check
from .constants import PARTITION, RESOURCE_NAME
from .manifest import load
from .seal import atomic_json, seal


class Adapter(Protocol):
    def describe(self) -> dict: ...
    def partitions(self) -> Iterable[tuple[str, str]]: ...
    def read(self, table: str, partition: str) -> pl.DataFrame: ...


class ValidationError(ValueError):
    def __init__(self, report):
        self.report = report
        failures = [x["code"] for x in report["checks"] if x["status"] == "fail"]
        super().__init__("数据包未通过校验：" + ", ".join(failures))


def _exchange(stage, target):
    """同文件系统目录交换；不使用先删除目标的非原子降级。"""
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin":
        function = libc.renamex_np
        function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        result = function(os.fsencode(stage), os.fsencode(target), 2)
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        function = libc.renameat2
        function.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        result = function(-100, os.fsencode(stage), -100, os.fsencode(target), 2)
    else:
        raise OSError("平台不支持原子目录交换；请发布到新目录")
    if result:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def build(adapter: Adapter, root):
    root = Path(root).absolute()
    if root.is_symlink():
        raise ValueError("包目录不得是符号链接")
    if (
        root.is_dir()
        and any(root.iterdir())
        and not (root / "datapackage.json").is_file()
    ):
        raise ValueError("非空输出目录不是数据包")
    root.parent.mkdir(parents=True, exist_ok=True)
    lock_path = root.parent / f".{root.name}.build.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        stage = Path(tempfile.mkdtemp(prefix=f".{root.name}.stage-", dir=root.parent))
        try:
            manifest = adapter.describe()
            names = {r["name"] for r in manifest["resources"]}
            seen, reused, written = set(), 0, 0
            for table, partition in adapter.partitions():
                if (
                    table not in names
                    or not re.fullmatch(RESOURCE_NAME, table)
                    or not re.fullmatch(PARTITION, partition)
                    or (table, partition) in seen
                ):
                    raise ValueError("非法或重复的分区标识")
                seen.add((table, partition))
                frame = adapter.read(table, partition)
                if not isinstance(frame, pl.DataFrame):
                    raise TypeError("Adapter.read 必须返回 Polars DataFrame")
                destination = stage / table / f"{partition}.parquet"
                destination.parent.mkdir(exist_ok=True)
                old = root / table / destination.name
                old_frame = (
                    pl.read_parquet(old)
                    if old.is_file() and not old.is_symlink()
                    else None
                )
                if (
                    old_frame is not None
                    and frame.schema == old_frame.schema
                    and frame.equals(old_frame, null_equal=True)
                ):
                    os.link(old, destination)
                    reused += 1
                else:
                    frame.write_parquet(
                        destination, compression="zstd", statistics=True
                    )
                    written += 1
            atomic_json(stage / "datapackage.json", manifest)
            sealed = seal(stage)
            report = check(stage)
            if not report["ok"]:
                raise ValidationError(report)
            if (
                (root / "datapackage.json").is_file()
                and load(root).get("version") == sealed["version"]
                and check(root)["ok"]
            ):
                # 无内容变化也不重写清单，created/mtime 同样保留。
                return dict(
                    version=sealed["version"],
                    reused=reused,
                    written=written,
                    published=False,
                )
            if root.exists():
                _exchange(stage, root)
            else:
                os.replace(stage, root)
            return dict(
                version=sealed["version"],
                reused=reused,
                written=written,
                published=True,
            )
        finally:
            if stage.exists():
                shutil.rmtree(stage)
