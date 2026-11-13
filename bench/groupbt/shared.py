"""交替测量冻结当前版本与共享实现；参考包和所有产物仅保存在 .cache。"""

import argparse
import importlib.util
import json
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

from ashare.factor import groupbt
from bench.groupbt.common import compare, real, snapshot, synthetic


def frozen(root):
    def load(name, folder):
        spec = importlib.util.spec_from_file_location(
            name, folder / "__init__.py", submodule_search_locations=[str(folder)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    native = load("panelbt_a5", root / "panelbt_a5")
    reference = load("ashare.factor.groupbt_a5", root / "groupbt_a5")
    reference.api.panelbt = native
    return reference


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--reference", type=Path, default=Path(".cache/groupbt-fusion/frozen")
    )
    p.add_argument("--data", choices=("synthetic", "real"), default="synthetic")
    p.add_argument("--repeat", type=int, default=7)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--scenario", choices=("single", "signals", "ledgers"), default="single"
    )
    p.add_argument("-o", type=Path, required=True)
    args = p.parse_args()
    if not args.o.resolve().is_relative_to(Path(".cache").resolve()):
        p.error("测量结果只能写入本 worktree 的 .cache")
    reference = frozen(args.reference)
    panel, factor = real() if args.data == "real" else synthetic()
    signal_file = args.o.with_suffix(".signal.npy")
    signal_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(signal_file, factor)
    del factor
    factor = np.load(signal_file, mmap_mode="r")
    configs = [
        dict(
            universe="TOP1800",
            group_neutralize="industry",
            window=window,
            sub_universes=window == groupbt.IS,
            record_holdings=True,
        )
        for window in (groupbt.IS, groupbt.VALIDATION)
    ]
    factors = [factor, factor]
    if args.scenario == "signals":
        configs *= 4
        factors = [factor if i % 4 < 2 else -factor for i in range(8)]
    elif args.scenario == "ledgers":
        configs = [
            c | {"rebalance": n, "weights_portfolio": weighted, "decay": 3}
            for n, weighted in ((1, False), (3, False), (3, True), (5, True))
            for c in configs
        ]
        factors *= 4

    def before():
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            return list(
                executor.map(
                    lambda pair: reference.run(pair[0], panel, **pair[1]),
                    zip(factors, configs, strict=True),
                )
            )

    def after():
        if args.scenario == "single":
            return [
                groupbt.run(x, panel, **c)
                for x, c in zip(factors, configs, strict=True)
            ]
        return groupbt.run_many(factors, panel, configs, workers=args.workers)

    expected, actual = before(), after()
    for a, b in zip(expected, actual, strict=True):
        compare(snapshot(a), snapshot(b))
        compare(a.holdings, b.holdings)
    del expected, actual
    timings = {"before": [], "after": []}
    methods = {"before": before, "after": after}
    for repeat in range(args.repeat):
        for name in ("before", "after") if repeat % 2 == 0 else ("after", "before"):
            tick = time.perf_counter()
            methods[name]()
            timings[name].append(time.perf_counter() - tick)
    peaks = {}
    for name, run in methods.items():
        tracemalloc.start()
        run()
        peaks[name] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    result = {
        "bitwise": len(configs),
        "seconds": timings,
        "median": {k: statistics.median(v) for k, v in timings.items()},
        "peak_allocated_bytes": peaks,
    }
    args.o.write_text(json.dumps(result, indent=2) + "\n")
    signal_file.unlink()
    print(json.dumps({k: v for k, v in result.items() if k != "seconds"}))


if __name__ == "__main__":
    main()
