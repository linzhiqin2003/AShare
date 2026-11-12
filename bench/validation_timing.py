"""Paired, alternating timing of input validation before/after the kernel move.

Usage: python bench/validation_timing.py --before DIR [--rounds 8 --repeats 5]

DIR holds a copy of an older `panelbt` package (with its built extension); the
current checkout is the other side. Two warm worker processes receive the same
requests in AB/BA order each round. Raw samples go to .cache/fix/ (ignored);
the compact summary is printed and written to bench/results/step1c-validation.json.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from statistics import median
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]


def worker():
    import numpy as np
    import panelbt as bt

    sys.path.insert(0, str(ROOT / "bench"))
    from compare.common import plan
    from pair import inputs

    big = inputs(4000, 5500)
    big["can_short"] = np.ones((4000, 5500), bool)
    days = np.arange(0, 4000, 21, dtype=np.int64)
    big_plan = dict(
        days=days,
        indptr=np.arange(len(days) + 1, dtype=np.int64) * 100,
        cols=((np.arange(100)[None, :] + np.arange(len(days))[:, None] * 37) % 5500)
        .ravel()
        .astype(np.int64),
        weights=np.full(len(days) * 100, 0.96 / 100),
    )
    small = inputs(1000, 100)
    plans = [plan(1000, 100, 21, i, 1) for i in range(1000)]
    raw_plans = [(p.days, p.indptr, p.cols, p.weights) for p in plans]
    costs = bt.Costs(tax_sell=0.0005)
    config = bt.Config(initial_cash=1e7, max_participation=0.1)
    has_switch = "validate" in {
        f.name for f in __import__("dataclasses").fields(bt.Market)
    }

    def market_check():
        bt.Market(**big)

    def batch_check_and_run():
        m = bt.Market(**small)
        ts = [bt.Targets(*x) for x in raw_plans]
        return bt.run_batch(m, ts, costs, bt.Config(initial_cash=1e7), threads=0)

    def single(validate):
        kw = {"validate": validate} if has_switch else {}
        m = bt.Market(**big, **kw)
        t = bt.Targets(**big_plan, **kw)
        return bt.run(m, t, costs, config)

    scenarios = {
        "market_4000x5500": market_check,
        "batch_1000_check_and_run": batch_check_and_run,
        "single_total_validated": lambda: single(True),
    }
    if has_switch:
        scenarios["single_total_unchecked"] = lambda: single(False)
    nav = {
        "batch": float(batch_check_and_run().nav[:, -1].sum()),
        "single": float(single(True).nav[-1]),
    }
    print(
        json.dumps(
            {
                "ready": True,
                "file": bt.__file__,
                "scenarios": list(scenarios),
                "nav": nav,
            }
        ),
        flush=True,
    )
    for line in sys.stdin:
        request = json.loads(line)
        fn = scenarios.get(request["scenario"])
        samples = []
        if fn is not None:
            for _ in range(request["repeats"]):
                start = perf_counter()
                fn()
                samples.append(perf_counter() - start)
        print(json.dumps({"samples_s": samples}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--before", type=Path)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.worker:
        return worker()
    procs = {}
    for side, path in (("before", args.before.resolve()), ("after", None)):
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        if path is not None:
            env["PYTHONPATH"] = str(path)
        procs[side] = subprocess.Popen(
            [sys.executable, __file__, "--worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            env=env,
        )
    info = {side: json.loads(p.stdout.readline()) for side, p in procs.items()}
    assert info["before"]["file"] != info["after"]["file"], info
    assert info["before"]["nav"] == info["after"]["nav"], info
    scenarios = info["after"]["scenarios"]
    samples = {side: {s: [] for s in scenarios} for side in procs}
    load = [os.getloadavg()]
    for r in range(args.rounds):
        order = ("before", "after") if r % 2 == 0 else ("after", "before")
        for scenario in scenarios:
            for side in order:
                p = procs[side]
                p.stdin.write(
                    json.dumps({"scenario": scenario, "repeats": args.repeats}) + "\n"
                )
                p.stdin.flush()
                samples[side][scenario] += json.loads(p.stdout.readline())["samples_s"]
        load.append(os.getloadavg())
    for p in procs.values():
        p.stdin.close()
        p.wait()
    summary = {
        "method": f"two warm processes, AB/BA alternating, {args.rounds} rounds x "
        f"{args.repeats} repeats per scenario",
        "load_1_5_15": [
            list(map(lambda x: round(x, 2), load[0])),
            list(map(lambda x: round(x, 2), load[-1])),
        ],
        "max_load_1": round(max(x[0] for x in load), 2),
        "ms": {
            s: {
                side: [
                    round(min(samples[side][s]) * 1e3, 3),
                    round(median(samples[side][s]) * 1e3, 3),
                ]
                for side in procs
                if samples[side][s]
            }
            for s in scenarios
        },
    }
    raw = ROOT / ".cache/fix/validation_timing_raw.json"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(json.dumps({"info": info, "samples_s": samples, "load": load}))
    (ROOT / "bench/results/step1c-validation.json").write_text(
        json.dumps(summary, indent=1) + "\n"
    )
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
