"""用真实原始库重跑构建或校验，记录进程峰值内存。"""

import argparse
import cProfile
import json
import pstats
import resource
import time
from pathlib import Path

from ashare.data.panel import build
from ashare.data.store import Store, atomic_json
from ashare.data.verify import verify


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["build", "verify"])
    parser.add_argument("--root", required=True)
    parser.add_argument("--end")
    parser.add_argument("--incremental", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument(
        "--record", required=True, help="数据目录 reports 下的相对文件名"
    )
    args = parser.parse_args()
    store = Store(args.root)
    begin = time.perf_counter()
    profiler = cProfile.Profile() if args.profile else None
    if profiler:
        profiler.enable()
    if args.action == "build":
        result = build(store, end=args.end, incremental=args.incremental)
        record = {
            k: result[k]
            for k in (
                "version",
                "shape",
                "elapsed_seconds",
                "reused_rows",
                "diagnostics",
            )
        }
        path = store.root / "panel" / result["version"]
        record["output_bytes"] = sum(
            p.stat().st_size for p in path.rglob("*") if p.is_file()
        )
    else:
        result = verify(store)
        record = {
            k: result[k]
            for k in ("version", "elapsed_seconds", "checks", "dividend_statistics")
        }
    record["wall_seconds"] = time.perf_counter() - begin
    if profiler:
        profiler.disable()
        with (store.root / "reports/round3/profile.txt").open("w") as stream:
            pstats.Stats(profiler, stream=stream).sort_stats("cumtime").print_stats(50)
    record["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    atomic_json(store.root / "reports/round3" / Path(args.record).name, record)
    print(
        json.dumps(
            {k: v for k, v in record.items() if k not in ("diagnostics", "checks")},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
