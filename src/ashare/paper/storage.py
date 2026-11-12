"""逐批不可变 Parquet 录制，保存接口时间和本地接收时间。"""

import hashlib
import io
from pathlib import Path

import numpy as np
import polars as pl
from paperbroker import COLUMNS, QuoteArray
from paperbroker.journal import durable_write


class ParquetStore:
    def __init__(self, directory: Path):
        self.directory = directory
        self.received_at = 0
        self.sent_at = 0
        self.source = "unknown"

    def put(self, sequence: int, quotes: QuoteArray) -> dict:
        table = pl.DataFrame({name: quotes[:, i] for i, name in enumerate(COLUMNS)})
        table = table.with_columns(
            pl.lit(2, dtype=pl.Int64).alias("schema_version"),
            pl.lit(self.sent_at, dtype=pl.Int64).alias("sent_at"),
            pl.lit(self.received_at, dtype=pl.Int64).alias("received_at"),
            pl.lit(self.source).alias("source"),
        )
        buffer = io.BytesIO()
        table.write_parquet(buffer, compression="zstd")
        data = buffer.getvalue()
        digest = hashlib.sha256(data).hexdigest()
        name = f"{sequence:012d}-{digest}.parquet"
        durable_write(self.directory / name, data)
        return dict(file=name, sha256=digest)

    def get(self, reference: dict) -> QuoteArray:
        name = reference["file"]
        if Path(name).name != name:
            raise ValueError("非法录制路径")
        data = (self.directory / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != reference["sha256"]:
            raise ValueError("行情文件校验失败")
        return np.ascontiguousarray(
            pl.read_parquet(io.BytesIO(data)).select(COLUMNS).to_numpy(), dtype=np.int64
        )


def disk_usage(directory: Path) -> dict[str, int]:
    """按分项统计文件逻辑字节；硬链接只计一次，避免日报重复计算。"""
    totals = dict(wal=0, snapshots=0, quotes=0, daily_archives=0, reports=0, other=0)
    seen = set()
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        stat = path.stat()
        key = (stat.st_dev, stat.st_ino)
        if key in seen:
            continue
        seen.add(key)
        relative = path.relative_to(directory)
        if relative.name == "events.jsonl":
            category = "wal"
        elif relative.name.startswith("state-"):
            category = "snapshots"
        else:
            category = {
                "quotes": "quotes",
                "days": "daily_archives",
                "reports": "reports",
            }.get(relative.parts[0], "other")
        totals[category] += stat.st_size
    return totals | {"total": sum(totals.values())}
