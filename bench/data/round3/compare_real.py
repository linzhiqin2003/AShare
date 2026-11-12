"""共享只读原始文件，独立全量构建后与已发布增量面板逐字节比较并清理副本。"""

import argparse
import hashlib
import json
import resource
import shutil
import tempfile
import time
from pathlib import Path

from ashare.data.panel import build
from ashare.data.store import Store, atomic_json


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    source = Store(args.root)
    current = json.loads((source.root / "panel/current.json").read_text())["version"]
    expected = source.root / "panel" / current
    base = Path(tempfile.mkdtemp(prefix=".round3-full-check-", dir=source.root.parent))
    try:
        root = base / "cn"
        root.mkdir()
        (root / "raw").symlink_to(source.root / "raw", target_is_directory=True)
        source.export_manifest(root / "manifest.json")
        begin = time.perf_counter()
        meta = build(Store(base), end="20261111")
        elapsed = time.perf_counter() - begin
        actual = root / "panel" / meta["version"]
        hashes = {}
        for name in meta["fields"]:
            a = digest(expected / f"{name}.npy")
            b = digest(actual / f"{name}.npy")
            assert a == b, name
            hashes[name] = a
        result = dict(
            shape=meta["shape"],
            fields=len(hashes),
            identical=True,
            sha256=hashes,
            version=meta["version"],
            reused_rows=meta["reused_rows"],
            full_build_seconds=elapsed,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            output_bytes=sum(p.stat().st_size for p in actual.iterdir() if p.is_file()),
        )
        assert meta["reused_rows"] == 0
        atomic_json(source.root / "reports/round3/bitwise-real.json", result)
        print({k: v for k, v in result.items() if k != "sha256"})
    finally:
        shutil.rmtree(base)


if __name__ == "__main__":
    main()
