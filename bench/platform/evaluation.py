"""比较独立求值与已有公共子式批量求值；只记录耗时和分配量。"""

import argparse
import json
import statistics
import time
import tracemalloc
from pathlib import Path

import panelops

from ashare.factor.engine import Evaluator
from ashare.factor.groupbt import VALIDATION
from ashare.platform.analysis import CutPanel, evaluate_signal
from ashare.platform.config import ops_threads
from ashare.platform.language import parse
from ashare.platform.runner import PanelCache


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-o", type=Path, required=True)
    p.add_argument("--repeat", type=int, default=9)
    args = p.parse_args()
    if not args.o.resolve().is_relative_to(Path(".cache").resolve()):
        p.error("测量结果只能写入本 worktree 的 .cache")
    panelops.set_num_threads(ops_threads())
    source = PanelCache(None)
    report = {}
    for windows in ([20, 60], [5, 20, 60, 120]):
        expressions = [
            v.expression
            for v in parse(f"vol = ts_std(returns, {windows})\nalpha = -vol")
        ]
        source.ensure(set().union(*(e.fields for e in expressions)))
        cut = CutPanel(source, VALIDATION[1])

        def before(expressions=expressions):
            for e in expressions:
                yield evaluate_signal(e, source)

        def after(expressions=expressions, cut=cut):
            for _, x in Evaluator().evaluate_many(expressions, cut):
                yield x

        for a, b in zip(before(), after(), strict=True):
            assert (
                a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
            )
        values = {"before": [], "after": []}
        methods = {"before": before, "after": after}
        for i in range(args.repeat):
            for key in ("before", "after") if i % 2 == 0 else ("after", "before"):
                tick = time.perf_counter()
                for value in methods[key]():
                    del value
                values[key].append(time.perf_counter() - tick)
        peaks = {}
        for key in methods:
            tracemalloc.start()
            for value in methods[key]():
                del value
            peaks[key] = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
        report[str(len(windows))] = {
            "seconds": values,
            "median": {k: statistics.median(v) for k, v in values.items()},
            "peak_allocated_bytes": peaks,
        }
    args.o.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: {
                    "median": v["median"],
                    "peak_allocated_bytes": v["peak_allocated_bytes"],
                }
                for k, v in report.items()
            }
        )
    )


if __name__ == "__main__":
    main()
