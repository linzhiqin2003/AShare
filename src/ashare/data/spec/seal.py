"""只计算封存元数据；不替校验器接受/修正内容。"""

import hashlib
import json
import os
import re
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq

from .constants import PARTITION, RESOURCE_NAME
from .manifest import load, version


def atomic_json(path, value):
    fd, name = tempfile.mkstemp(prefix=".manifest-", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def seal(root):
    root = Path(root)
    manifest = load(root)
    for resource in manifest["resources"]:
        name = resource["name"]
        if not re.fullmatch(RESOURCE_NAME, name):
            raise ValueError(f"非法表名：{name}")
        folder = root / name
        if folder.is_symlink():
            raise ValueError("表目录不得是符号链接")
        paths = sorted(folder.glob("*.parquet"))
        if not paths:
            raise ValueError(f"表 {name} 没有分区；空表也需带 schema 的 all.parquet")
        files, total = [], hashlib.sha256()
        for path in paths:
            if path.is_symlink() or not re.fullmatch(PARTITION, path.stem):
                raise ValueError(f"非法分区：{path.name}")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                while chunk := stream.read(1 << 20):
                    digest.update(chunk)
                    total.update(chunk)
            files.append(
                dict(
                    path=path.relative_to(root).as_posix(),
                    bytes=path.stat().st_size,
                    hash="sha256:" + digest.hexdigest(),
                    rows=pq.read_metadata(path).num_rows,
                )
            )
        resource.update(
            path=[f["path"] for f in files],
            files=files,
            bytes=sum(f["bytes"] for f in files),
            rows=sum(f["rows"] for f in files),
            hash="sha256:" + total.hexdigest(),
        )
    manifest["version"] = version(manifest)
    manifest["created"] = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    atomic_json(root / "datapackage.json", manifest)
    return manifest
