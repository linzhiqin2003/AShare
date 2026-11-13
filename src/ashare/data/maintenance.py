"""清理可重建缓存和旧面板，真实内容修订始终保留。"""

import json
import shutil

from .catalog import FUTURES_DATASETS
from .store import lock


def disk_usage(root):
    result = {}
    futures_dirs = {spec.raw_dir or spec.api for spec in FUTURES_DATASETS.values()}
    for name in ("raw", "panel", "reports", "projections", "futures"):
        if name == "futures":
            # 期货原始库与期货面板单独计，股票部分不随期货变化。
            paths = [
                p for d in sorted(futures_dirs) for p in (root / "raw" / d).rglob("*")
            ] + list((root / "futures_panel").rglob("*"))
            paths += [
                p for p in (root / "reports").glob("futures_verify*") if p.exists()
            ]
        elif (root / name).exists():
            paths = [
                p
                for p in (root / name).rglob("*")
                if not (
                    name == "raw"
                    and p.relative_to(root / "raw").parts[0] in futures_dirs
                )
                and not (name == "reports" and p.name.startswith("futures_verify"))
            ]
        else:
            paths = []
        files = [p for p in paths if p.is_file()]
        if name == "futures" and not files:
            continue
        result[name] = {
            "files": len(files),
            "bytes": sum(p.stat().st_size for p in files),
            "allocated_bytes": sum(p.stat().st_blocks * 512 for p in files),
        }
    result["manifest"] = {
        "bytes": sum(p.stat().st_size for p in root.glob("manifest*") if p.is_file())
    }
    return result


def collect_garbage(store, keep=1, *, locked=False):
    if keep < 1:
        raise ValueError("至少保留一代面板")
    if not locked:
        with lock(store.root / ".build.lock"):
            return collect_garbage(store, keep, locked=True)
    base = store.root / "panel"
    current = (
        json.loads((base / "current.json").read_text())["version"]
        if (base / "current.json").exists()
        else None
    )
    versions = sorted(
        (p for p in base.glob("*") if p.is_dir() and not p.name.startswith(".")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    retained = {current}
    retained.update(
        p.name for p in [p for p in versions if p.name != current][: keep - 1]
    )
    removed = []
    for path in (
        list(base.glob(".building-*"))
        + [p for p in versions if p.name not in retained]
        + [store.root / "projections"]
    ):
        if path.exists():
            shutil.rmtree(path)
            removed.append(str(path.relative_to(store.root)))
    try:
        if (store.root / "raw").is_symlink():
            raise BlockingIOError("共享只读原始库不清理临时文件")
        with lock(store.root / ".raw.lock", blocking=False):
            for path in (store.root / "raw").rglob("*.tmp"):
                path.unlink()
    except BlockingIOError:
        pass  # 活跃写者持共享锁；本次不清原始临时文件。
    return {"removed": removed, "retained": sorted(x for x in retained if x)}
