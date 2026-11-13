"""比较同一合成面板上的串行、线程与 spawn 子进程，输出耗时与峰值 RSS。"""

import json
import multiprocessing
import resource
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import panelops

from ashare.platform.analysis import run_variant
from ashare.platform.config import DEFAULTS
from ashare.platform.language import parse
from bench.platform.synthetic import panel


def one(window):
    panelops.set_num_threads(1)
    source = panel(n=120, full=True)
    expression = parse(f"-ts_std(returns, {window})")[0].expression
    summary, _ = run_variant(expression, source, DEFAULTS | {"sub_universes": False})
    return summary["is"]["sharpe"], resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def main():
    results = {}
    for mode in ("serial", "threads2", "processes2"):
        started = time.perf_counter()
        if mode == "serial":
            values = [one(w) for w in (10, 20, 30, 40, 50, 60, 70, 80)]
        else:
            factory = ThreadPoolExecutor if mode == "threads2" else ProcessPoolExecutor
            options = (
                {}
                if mode == "threads2"
                else {"mp_context": multiprocessing.get_context("spawn")}
            )
            with factory(max_workers=2, **options) as executor:
                values = list(executor.map(one, (10, 20, 30, 40, 50, 60, 70, 80)))
        results[mode] = {
            "seconds": time.perf_counter() - started,
            "peak_rss": max(v[1] for v in values),
            "values": [v[0] for v in values],
        }
        print(json.dumps({mode: results[mode]}), flush=True)
    assert (
        results["serial"]["values"]
        == results["threads2"]["values"]
        == results["processes2"]["values"]
    )


if __name__ == "__main__":
    main()
