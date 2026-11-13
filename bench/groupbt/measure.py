"""独立进程计时/RSS/内存分配/线程扩展性；只输出性能，不输出行情数字。"""

import argparse
import cProfile
import importlib
import json
import resource
import sys
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ashare.factor import groupbt
from bench.groupbt.common import real, synthetic
from bench.groupbt.oracle import groupbt as oracle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--backend",
        default="current",
        help="main / current / saved stage in .cache/groupbt",
    )
    p.add_argument("--data", choices=("real", "synthetic"), default="real")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--trace", action="store_true")
    p.add_argument("--profile", action="store_true")
    p.add_argument("--repeat", type=int, default=1)
    p.add_argument(
        "--platform",
        action="store_true",
        help="20/60 日波动率 × industry/cap+industry，共四组",
    )
    p.add_argument("-o", type=Path, required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    if not args.o.resolve().is_relative_to(root / ".cache"):
        p.error("原始剖析只能写在 worktree 的 .cache 内")
    panel, factor = real() if args.data == "real" else synthetic()
    api = oracle if args.backend == "main" else groupbt
    if args.backend == "simulation":
        api = importlib.import_module("bench.groupbt.simulation")
    elif args.backend not in ("main", "current"):
        sys.path.insert(0, str(root / ".cache/groupbt"))
        api = importlib.import_module(args.backend + ".groupbt")
    factors = [factor]
    if args.platform:
        import panelops

        close = panel["close"] * panel["adj_factor"]
        returns = close / panelops.ts_delay(close, 1) - 1
        factors.append(panelops.ts_std(returns, 60))
        del close, returns
        args.jobs = 4
    stages = []
    if args.profile and args.backend == "current":
        import panelbt

        native = panelbt.run_group_weights

        def profiled(*a, **kw):
            out = native(*a, **kw, profile=True)
            stages.append(out[0]["stages"])
            return out

        panelbt.run_group_weights = profiled
    steps = []
    memory = []
    if args.trace:
        tracemalloc.start()

        # 返回每个函数时取当前存活分配，另保留整个调用的峰值。
        def profiler(frame, event, arg):
            if event == "return" and "/groupbt/" in frame.f_code.co_filename:
                current, peak = tracemalloc.get_traced_memory()
                memory.append((current, peak, frame.f_code.co_name))

        sys.setprofile(profiler)
    profile = cProfile.Profile() if args.profile else None
    if profile:
        profile.enable()

    def job(i):
        f = factors[i // 2] if args.platform else factor
        neutral = ("industry", "cap+industry")[i % 2] if args.platform else "industry"
        tick = time.perf_counter()
        a = api.run(f, panel, "TOP1800", group_neutralize=neutral)
        middle = time.perf_counter()
        b = api.run(
            f,
            panel,
            "TOP1800",
            group_neutralize=neutral,
            sub_universes=False,
            window=api.VALIDATION,
        )
        end = time.perf_counter()
        steps.append([middle - tick, end - middle])
        return a.group_pnl.shape, b.group_pnl.shape

    elapsed_runs = []
    for _ in range(args.repeat):
        tick = time.perf_counter()
        if args.threads == 1:
            for i in range(args.jobs):
                job(i)
        else:
            with ThreadPoolExecutor(max_workers=args.threads) as pool:
                list(pool.map(job, range(args.jobs)))
        elapsed_runs.append(time.perf_counter() - tick)
    if profile:
        profile.disable()
        profile.dump_stats(str(args.o.with_suffix(".prof")))
    if args.trace:
        sys.setprofile(None)
    info = dict(
        backend=args.backend,
        data=args.data,
        threads=args.threads,
        jobs=args.jobs,
        elapsed=elapsed_runs,
        platform=args.platform,
        native_stages=stages,
        steps=steps,
        peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / (1e9 if sys.platform == "darwin" else 1e6),
    )
    if args.trace:
        info["traced_peak_gb"] = tracemalloc.get_traced_memory()[1] / 1e9
        info["large_live_returns"] = sorted(set(memory), reverse=True)[:30]
        tracemalloc.stop()
    args.o.parent.mkdir(parents=True, exist_ok=True)
    args.o.write_text(json.dumps(info, indent=2) + "\n")
    print(json.dumps(info))


if __name__ == "__main__":
    main()
