"""main 冻结基准的存盘与逐位复核。用 --data real 读取 ASHARE_DATA_DIR。"""

import argparse
import hashlib
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np

from ashare.factor import groupbt
from bench.groupbt.common import cases, compare, real, snapshot, synthetic
from bench.groupbt.oracle import groupbt as oracle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("baseline", "compare"))
    p.add_argument("--data", choices=("synthetic", "real"), default="synthetic")
    p.add_argument("--cache", type=Path, default=Path(".cache/groupbt"))
    p.add_argument("--limit", type=int)
    p.add_argument("--implementation", default="current")
    p.add_argument("--wait-baseline", action="store_true")
    p.add_argument("-o", type=Path)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    if not args.cache.resolve().is_relative_to(root / ".cache"):
        p.error("基准只能保存在本 worktree 的 .cache 内")
    folder = args.cache / args.data
    folder.mkdir(parents=True, exist_ok=True)
    panel, factor = real() if args.data == "real" else synthetic()
    implementation = groupbt
    if args.implementation == "simulation":
        implementation = importlib.import_module("bench.groupbt.simulation")
    elif args.implementation != "current":
        sys.path.insert(0, str(root / ".cache/groupbt"))
        implementation = importlib.import_module(args.implementation + ".groupbt")
    if args.action == "compare" and args.implementation in ("current", "simulation"):
        from test.build_guard import assert_current_build

        build = assert_current_build(root)
    else:
        build = None
    all_configs = cases(long_windows=args.data == "synthetic")
    configs = all_configs[: args.limit]
    source = root / "bench/groupbt/oracle"
    manifest = json.loads((source / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        assert (
            hashlib.sha256(
                (source / name.removeprefix("src/ashare/factor/")).read_bytes()
            ).hexdigest()
            == digest
        ), "冻结参考实现被改动"
    identity = dict(
        oracle=manifest,
        numpy=np.__version__,
        panel=getattr(panel, "version", "synthetic-1701"),
        factor=hashlib.sha256(memoryview(factor)).hexdigest(),
        cases=all_configs,
    )
    identity = json.loads(json.dumps(identity))
    metadata = folder / "identity.json"
    if metadata.exists():
        assert json.loads(metadata.read_text()) == identity, (
            "输入或 NumPy 环境已变，须使用新的基准目录"
        )
    else:
        metadata.write_text(json.dumps(identity, indent=2) + "\n")
    started = time.perf_counter()
    seconds = []
    cached = 0
    for i, config in enumerate(configs):
        path = folder / f"{i:03d}.npz"
        if args.action == "baseline" and path.exists():
            # Resumed runs do not claim to have recomputed cached cases.
            with np.load(path, allow_pickle=False) as saved:
                for key in saved:
                    saved[key]  # Verify the archive's CRC and dtype on resume.
            cached += 1
            continue
        if args.action == "compare" and args.wait_baseline:
            while not path.exists():
                time.sleep(0.25)
        tick = time.perf_counter()
        result = (oracle if args.action == "baseline" else implementation).run(
            factor, panel, **config
        )
        seconds.append(time.perf_counter() - tick)
        got = snapshot(result)
        if args.action == "baseline":
            temporary = path.with_suffix(".tmp.npz")
            np.savez_compressed(temporary, **got)
            temporary.replace(path)
        else:
            with np.load(path, allow_pickle=False) as saved:
                compare(saved, got)
        if (i + 1) % 16 == 0:
            print(
                f"{args.action}: {i + 1}/{len(configs)}，{time.perf_counter() - started:.1f} s",
                flush=True,
            )
    report = dict(
        action=args.action,
        data=args.data,
        passed=len(configs),
        total=len(configs),
        elapsed=time.perf_counter() - started,
        run_seconds=seconds,
        cached=cached,
        computed=len(seconds),
        build=build,
    )
    (folder / f"{args.action}-{args.implementation}.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    if args.o:
        args.o.write_text(
            f"逐位对照：{report['passed']}/{report['total']}；耗时 {report['elapsed']:.3f} 秒。\n"
        )
    print(f"{args.action}: {len(configs)}/{len(configs)}，{report['elapsed']:.3f} s")


if __name__ == "__main__":
    main()
