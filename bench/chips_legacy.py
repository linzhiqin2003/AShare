"""D3 合成面板旧字段逐位回归；--baseline-src 指向修改前的 src 副本（只在 tmp/）。"""

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build_and_digest(root):
    import numpy as np

    import ashare
    from ashare.data.fund_panel import build_funds
    from ashare.data.panel import build
    from ashare.data.store import Store

    store = Store(root)
    build(store)
    build_funds(store, kinds=("fund", "cb"))
    out = {}
    for kind in ("panel", "fund_panel/fund", "fund_panel/cb"):
        base = store.root / kind
        path = base / json.loads((base / "current.json").read_text())["version"]
        meta = json.loads((path / "metadata.json").read_text())
        fields = {}
        for name in meta["fields"]:
            value = np.load(path / f"{name}.npy")
            fields[name] = dict(
                dtype=str(value.dtype),
                shape=list(value.shape),
                sha256=hashlib.sha256(value.tobytes()).hexdigest(),
            )
        out[kind] = dict(dates=meta["dates"], symbols=meta["symbols"], fields=fields)
    return dict(
        imported=str(Path(ashare.__file__).resolve().relative_to(ROOT)), panels=out
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-src", type=Path)
    parser.add_argument("--build", type=Path)
    args = parser.parse_args()
    if args.build:
        print(json.dumps(build_and_digest(args.build)))
        return
    baseline = args.baseline_src.resolve()
    if not baseline.is_relative_to(ROOT / "tmp"):
        raise ValueError("基线源码必须在 worktree/tmp 下")
    folder = ROOT / "tmp/d3-legacy"
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    spec = importlib.util.spec_from_file_location(
        "stock_fixture", ROOT / "test/data/conftest.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tables = module.tables.__wrapped__()
    store = module.make_store.__wrapped__(folder)(tables, name="input")
    sys.path.insert(0, str(ROOT / "test/data"))
    from futures_fixtures import DAYS
    from test_fund_fixtures import FakeFundTushare

    from ashare.data.fund_sync import sync_funds

    asyncio.run(sync_funds(FakeFundTushare(), store, DAYS[0], DAYS[-1]))
    # 复制时先关 SQLite，避免两个构建共享 WAL 或锁。
    store._manifest.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    store._manifest.db.close()
    results = []
    for label, source in (("old", baseline), ("new", ROOT / "src")):
        target = folder / label
        shutil.copytree(folder / "input", target)
        env = dict(
            os.environ,
            PYTHONPATH=str(source),
            PYTHONDONTWRITEBYTECODE="1",
            POLARS_MAX_THREADS="2",
        )
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--build", str(target)],
            env=env,
            cwd=ROOT,
            text=True,
            capture_output=True,
            timeout=180,
        )
        if result.returncode:
            raise RuntimeError(result.stderr)
        results.append(json.loads(result.stdout))
    checks = {}
    for name, before in results[0]["panels"].items():
        after = results[1]["panels"][name]
        assert (
            before["dates"] == after["dates"] and before["symbols"] == after["symbols"]
        )
        assert all(after["fields"][k] == v for k, v in before["fields"].items())
        checks[name] = dict(
            unchanged_fields=len(before["fields"]),
            new_fields=sorted(set(after["fields"]) - set(before["fields"])),
            verdict="bitwise_equal",
        )
    output = dict(imported=[r["imported"] for r in results], checks=checks)
    (ROOT / "bench/results/chips_legacy.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
