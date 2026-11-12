"""清理可重建缓存和旧面板，真实内容修订始终保留。"""

import json
import shutil

from .store import lock


def disk_usage(root):
    result = {}
    for name in ("raw", "panel", "reports", "projections"):
        paths = list((root / name).rglob("*")) if (root / name).exists() else []
        files = [p for p in paths if p.is_file()]
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
