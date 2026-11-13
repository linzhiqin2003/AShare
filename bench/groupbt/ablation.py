"""独立开关共享行情、暴露均值和批量记账，交替计时并逐位核对原始内核输出。"""

import argparse
import json
import statistics
import time
import tracemalloc
from pathlib import Path

import numpy as np
import panelbt

from ashare.factor.groupbt.api import _prepare
from bench.groupbt.common import compare, real


def measured(before, after, repeat):
    values = {"before": [], "after": []}
    methods = {"before": before, "after": after}
    for i in range(repeat):
        for key in ("before", "after") if i % 2 == 0 else ("after", "before"):
            tick = time.perf_counter()
            methods[key]()
            values[key].append(time.perf_counter() - tick)
    peaks = {}
    for key, call in methods.items():
        tracemalloc.start()
        call()
        peaks[key] = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    return {
        "seconds": values,
        "median": {k: statistics.median(v) for k, v in values.items()},
        "peak_allocated_bytes": peaks,
    }


def outputs(runs):
    result = {}
    for i, pools in enumerate(runs):
        for j, pool in enumerate(pools):
            for key in ("pnl", "turnover", "exposure", "first_live", "bad_rows"):
                result[f"{i}/{j}/{key}"] = np.asarray(pool[key])
            for key, value in pool.get("holdings", {}).items():
                result[f"{i}/{j}/holdings/{key}"] = value
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-o", type=Path, required=True)
    p.add_argument("--repeat", type=int, default=9)
    args = p.parse_args()
    if not args.o.resolve().is_relative_to(Path(".cache").resolve()):
        p.error("测量结果只能写入本 worktree 的 .cache")
    panel, signal = real()
    path = args.o.with_suffix(".signal.npy")
    np.save(path, signal)
    del signal
    factor = np.load(path, mmap_mode="r")

    def prepare(shared):
        cache = {} if shared else None
        return [
            _prepare(factor, panel, "TOP1800", group_neutralize=mode, _shared=cache)
            for mode in (None, "industry", "cap", "cap+industry")
        ]

    report = {
        "market_preparation": measured(
            lambda: prepare(False), lambda: prepare(True), args.repeat
        )
    }
    prepared = prepare(True)

    def native(cached):
        first = prepared[0]
        a = first.args
        cache = panelbt.GroupExposureCache(
            a[4], a[5], a[2], a[3], first.options["signal_start"]
        )
        return [
            panelbt.run_group_weights(
                *p.args,
                **(
                    p.options
                    | {"exposure_cache": cache if cached else None, "profile": True}
                ),
            )
            for p in prepared
        ]

    compare(outputs(native(False)), outputs(native(True)))
    report["exposure_means"] = measured(
        lambda: native(False), lambda: native(True), args.repeat
    )
    report["exposure_means"]["stages"] = {
        key: np.sum([run[0]["stages"] for run in native(flag)], axis=0).tolist()
        for key, flag in (("before", False), ("after", True))
    }
    prepared = prepared[1]
    options = {
        k: v
        for k, v in prepared.options.items()
        if k not in {"rebalance", "weighted", "record_holdings", "exposure_cache"}
    }
    variants = [
        dict(rebalance=n, weighted=w, record_holdings=True)
        for n, w in ((1, False), (1, True), (3, False), (3, True))
    ]

    def ledgers(batch):
        a = prepared.args
        cache = panelbt.GroupExposureCache(
            a[4], a[5], a[2], a[3], options["signal_start"]
        )
        if batch:
            return panelbt.run_group_weights_batch(
                *a, **options, exposure_cache=cache, variants=variants
            )
        return [
            panelbt.run_group_weights(*a, **options, exposure_cache=cache, **v)
            for v in variants
        ]

    compare(outputs(ledgers(False)), outputs(ledgers(True)))
    report["batch_targets"] = measured(
        lambda: ledgers(False), lambda: ledgers(True), args.repeat
    )
    args.o.write_text(json.dumps(report, indent=2) + "\n")
    path.unlink()
    print(json.dumps({k: v["median"] for k, v in report.items()}))


if __name__ == "__main__":
    main()
