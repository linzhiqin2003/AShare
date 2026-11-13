"""测量实际调度器的冷/热运行、共享求值、进程峰值，全部使用合成数据。"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ashare.platform.runner import Manager
from ashare.platform.storage import Store
from bench.platform.synthetic import panel, write_panel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=".cache/platform/models-r2")
    args = parser.parse_args()
    root = Path(args.output)
    write_panel(root / "data", panel(n=120, full=True))
    results, reference = [], None
    for mode in ("threads", "processes"):
        manager = Manager(
            Store(root / mode),
            data_root=str(root / "data"),
            mode=mode,
            limit=4,
            concurrency=2,
        )
        try:
            tab = manager.create_tab()
            for attempt in ("cold", "warm"):
                start = time.perf_counter()
                job = manager.submit(
                    tab,
                    "-ts_std(returns,[20,60])",
                    {
                        "group_neutralize": [None, "industry", "cap", "cap+industry"],
                        "sub_universes": False,
                    },
                    "synthetic",
                )
                deadline = start + 180
                while (
                    manager.store.job(job)["status"] == "running"
                    and time.perf_counter() < deadline
                ):
                    time.sleep(0.05)
                wall = time.perf_counter() - start
                result = manager.store.job(job)
                assert result["status"] == "complete", result["status"]
                arrays = [
                    manager.store.series(r["id"])["is.pnl.long_short"].view(np.uint64)
                    for r in result["runs"]
                ]
                if reference is None:
                    reference = arrays
                else:
                    for expected, actual in zip(reference, arrays, strict=True):
                        np.testing.assert_array_equal(expected, actual)
                events = manager.store.events(job, 0)
                timing = next(e for e in events if e["kind"] == "timing")
                rss = {e["pid"]: e["rss"] for e in events if e["kind"] == "memory"}
                row = dict(
                    mode=mode,
                    attempt=attempt,
                    wall=wall,
                    worker=timing["elapsed"],
                    evaluation=timing["evaluation_seconds"],
                    coordinator_rss=timing["rss"],
                    child_peak_sum=sum(rss.values()),
                    children=len(rss),
                    expressions=2,
                    variants=8,
                )
                results.append(row)
                print(json.dumps(row), flush=True)
        finally:
            manager.close()
    (root / "metrics.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
