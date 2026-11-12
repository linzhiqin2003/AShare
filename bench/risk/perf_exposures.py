"""P-007 探针：风格暴露按天并行时线程池能拿到多少（GIL 的影响），以及各线程数下的耗时。

滚动描述子按日递推，先顺序算好每天的结果，再把各天的 `exposures_at` 分给线程池；
结果与顺序计算逐位比较。每个线程数在全新进程里跑。

    python bench/risk/perf_exposures.py --days 300 --batch 64 --threads 1 5 10 15
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def once(args):
    sys.path.insert(0, str(ROOT / "bench/e2e"))
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np
    from common import default_data
    from risk_adapter import load

    from ashare.risk.data import adjusted_returns
    from ashare.risk.fundamental import exposures_at
    from ashare.risk.rolling import RollingDescriptors

    source, config = load(default_data(), args.days)
    t = len(source.dates)
    returns = adjusted_returns(source, t - 1)
    returns[np.asarray(source["is_suspended"][:t], bool)] = np.nan
    ages = np.cumsum(np.asarray(source["is_listed"][:t], bool), axis=0, dtype=np.int32)
    rolling = RollingDescriptors(source, returns, config, ages)
    days = list(range(t - args.batch, t))
    raws = {}
    for day in range(t):
        raw = rolling.at(day)
        if day in days:
            raws[day] = raw

    class Fixed:
        def at(self, day):
            return raws[day]

    fixed = Fixed()

    def one(day):
        return exposures_at(
            source, returns, day, config, listing_ages=ages, rolling=fixed
        )

    reference = [one(d)[0] for d in days]
    best = float("inf")
    for _ in range(args.repeat):
        start = time.perf_counter()
        if args.threads == 1:
            out = [one(d)[0] for d in days]
        else:
            with ThreadPoolExecutor(args.threads) as pool:
                out = [r[0] for r in pool.map(one, days)]
        best = min(best, time.perf_counter() - start)
    same = all(np.array_equal(a, b) for a, b in zip(out, reference, strict=True))
    print(
        json.dumps(
            {
                "threads": args.threads,
                "seconds": best,
                "per_day_ms": best / len(days) * 1e3,
                "identical": same,
                "load": os.getloadavg()[0],
            }
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=300)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 5, 10, 15])
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.once:
        args.threads = args.threads[0]
        once(args)
        return
    for threads in args.threads:
        subprocess.run(
            [
                sys.executable,
                __file__,
                "--once",
                "--days",
                str(args.days),
                "--batch",
                str(args.batch),
                "--repeat",
                str(args.repeat),
                "--threads",
                str(threads),
            ],
            check=True,
            timeout=1800,
        )


if __name__ == "__main__":
    main()
