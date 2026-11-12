"""ABAB timing of the rolling-sum family: two panelops builds, fresh process per run.

Usage:
    python bench/panelops/rolling_sum_ab.py --old /path/to/dir_containing_old_panelops
    python bench/panelops/rolling_sum_ab.py --exact   # exact=True cost vs the default

``--old`` names a directory holding a copy of an earlier ``panelops`` package
(``panelops/__init__.py`` plus its compiled ``_core``); the current build is whatever
``import panelops`` resolves to. Each measurement is the minimum of several calls in a
fresh single-threaded process; old and new alternate within every round.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys

CHILD = r"""
import json, sys, time, numpy as np, panelops as po
po.set_num_threads(1)
op, window, sparse, exact, reps, data = sys.argv[1], int(sys.argv[2]), sys.argv[3] == "1", sys.argv[4] == "1", int(sys.argv[5]), sys.argv[6]
rng = np.random.default_rng(7)
x = rng.normal(10, 3, (1500, 2000)).round(4)
if data == "events":  # mostly zero: long all-zero windows, the constant-window path
    x = np.where(rng.random(x.shape) < 0.02, x, 0.0)
c = (rng.random(x.shape) < 0.5).astype(float)
if sparse:
    x[rng.random(x.shape) < 0.05] = np.nan
kw = {"exact": True} if exact else {}
f = {
    "ts_sum": lambda: po.ts_sum(x, window, **kw),
    "ts_mean": lambda: po.ts_mean(x, window, **kw),
    "ts_sumif": lambda: po.ts_sumif(x, c, window, **kw),
    "ts_count": lambda: po.ts_count(x, window),
    "ts_decay_linear": lambda: po.ts_decay_linear(x, window),
}[op]
f()
best = float("inf")
for _ in range(reps):
    s = time.perf_counter(); f(); best = min(best, time.perf_counter() - s)
print(json.dumps({"seconds": best, "file": po.__file__}))
"""


def measure(path, op, window, sparse, exact, reps, data="normal"):
    env = dict(os.environ)
    if path:
        env["PYTHONPATH"] = path
    out = subprocess.run(
        [
            sys.executable,
            "-c",
            CHILD,
            op,
            str(window),
            str(int(sparse)),
            str(int(exact)),
            str(reps),
            data,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=900,
        check=True,
    )
    return json.loads(out.stdout)


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--old")
    a.add_argument("--exact", action="store_true")
    a.add_argument("--rounds", type=int, default=5)
    a.add_argument("--reps", type=int, default=5)
    a.add_argument("--ops", default="ts_sum,ts_mean,ts_sumif,ts_count,ts_decay_linear")
    a.add_argument("--windows", default="5,20,60,250")
    a.add_argument("--data", choices=("normal", "events"), default="normal")
    args = a.parse_args()
    print("load", os.getloadavg(), flush=True)
    for op in args.ops.split(","):
        for window in map(int, args.windows.split(",")):
            for sparse in (False, True):
                if args.exact and op not in ("ts_sum", "ts_mean", "ts_sumif"):
                    continue
                pair = {"A": [], "B": []}
                for _ in range(args.rounds):
                    if args.exact:
                        pair["A"].append(
                            measure(
                                None, op, window, sparse, False, args.reps, args.data
                            )["seconds"]
                        )
                        pair["B"].append(
                            measure(
                                None, op, window, sparse, True, args.reps, args.data
                            )["seconds"]
                        )
                    else:
                        pair["A"].append(
                            measure(
                                args.old,
                                op,
                                window,
                                sparse,
                                False,
                                args.reps,
                                args.data,
                            )["seconds"]
                        )
                        pair["B"].append(
                            measure(
                                None, op, window, sparse, False, args.reps, args.data
                            )["seconds"]
                        )
                a_med, b_med = (
                    statistics.median(pair["A"]),
                    statistics.median(pair["B"]),
                )
                label = "default ms, exact ms" if args.exact else "old ms, new ms"
                print(
                    f"{op:16s} w={window:3d} {'nan5%' if sparse else 'dense'} "
                    f"{label}: {a_med * 1e3:8.2f} {b_med * 1e3:8.2f} ratio {b_med / a_med:5.3f}",
                    flush=True,
                )
    print("load", os.getloadavg(), flush=True)


if __name__ == "__main__":
    main()
