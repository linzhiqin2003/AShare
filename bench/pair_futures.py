"""Alternating old/current timing of the futures and exit-rule paths.

    python bench/pair.py --base-ref <rev> --keep     # builds .cache/panelbt/pair-base
    python bench/pair_futures.py [--t 2000 --n 2000]

Uses the historical build that `bench/pair.py --keep` left in
.cache/panelbt/pair-base. Scenarios: a spot portfolio with an index-futures
hedge leg (bench/futures.py inputs), a pure futures market of 60 contracts,
and spot exit rules. Results must be identical in both builds; the report
(.cache/panelbt/pair_futures.json) has min and median ratios.
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
sys.path.insert(0, str(ROOT))


def worker(t, n):
    import numpy as np
    import panelbt as bt

    from bench.benchmark import market_inputs
    from bench.futures import add_futures, plans

    raw = market_inputs(t, n)
    _, mixed_plan, expiry = plans(t, n)
    mixed = add_futures(raw, expiry)
    costs = bt.Costs(tax_sell=0.0005)
    config = bt.Config(initial_cash=1e7, max_participation=0.1)
    # Pure futures: 60 contracts, each listed for 120 rows, rolled by notional.
    k = 60
    rows = np.arange(t)[:, None]
    price = 3000 * np.exp(0.0004 * rows + 0.02 * np.sin(rows / 17 + np.arange(k)))
    settle = price * (1 + 0.001 * np.cos(rows / 5 + np.arange(k)))
    expiry_k = np.minimum((np.arange(k) + 1) * (t // k), t)
    pure = bt.Market(
        price,
        price,
        np.ones((t, k), bool),
        np.ones((t, k), bool),
        volume=np.full((t, k), 1e5),
        futures=bt.Futures(
            is_future=np.ones(k, bool),
            multiplier=np.full(k, 10.0),
            margin_rate=np.full((t, k), 0.12),
            settlement=settle,
            expiry=expiry_k,
            fee_rate=1e-4,
            fee_per_contract=1.0,
        ),
    )
    days = np.arange(0, t, 5, dtype=np.int64)
    col = np.searchsorted(expiry_k, days + 3, side="right").clip(0, k - 1)
    pure_plan = bt.Targets(
        days,
        np.arange(len(days) + 1, dtype=np.int64),
        col,
        np.where((days // 21) % 2 == 0, 3.0, -3.0),
    )
    # Spot exit rules on the spot market of the hedge scenario.
    spot = bt.Market(
        **raw,
        open=raw["exec_price"],
        high=raw["exec_price"] * 1.01,
        low=raw["exec_price"] * 0.99,
    )
    stock_plan, _, _ = plans(t, n)
    rules = bt.Config(
        initial_cash=1e7,
        max_participation=0.1,
        rules=bt.Rules(fill_time="open", stop_loss=0.05, trailing_stop=0.08),
    )
    scenarios = {
        "hedge_leg": (mixed, mixed_plan, config),
        "pure_futures": (pure, pure_plan, bt.Config(initial_cash=1e7)),
        "spot_rules": (spot, stock_plan, rules),
    }
    print(json.dumps({"ready": True}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        market, plan, cfg = scenarios[request["scenario"]]
        samples = []
        for _ in range(request["repeats"]):
            start = perf_counter()
            r = bt.run(market, plan, costs, cfg)
            samples.append(perf_counter() - start)
        digest = float(np.sum(r.equity)) + float(np.sum(r.commission))
        print(json.dumps(dict(samples_s=samples, digest=digest)), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--t", type=int, default=2000)
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--rounds", type=int, default=8)
    args = parser.parse_args()
    if args.worker:
        return worker(args.t, args.n)
    base = ROOT / ".cache/panelbt/pair-base/.venv/bin/python"
    assert base.exists(), "run bench/pair.py --base-ref <rev> --keep first"
    procs = []
    for python in (base, ROOT / ".venv/bin/python"):
        p = subprocess.Popen(
            [str(python), __file__, "--worker", "--t", str(args.t), "--n", str(args.n)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            cwd=ROOT,
        )
        assert json.loads(p.stdout.readline())["ready"]
        procs.append(p)
    report = dict(T=args.t, N=args.n, load_start=os.getloadavg(), scenarios={})
    try:
        for name in ("hedge_leg", "pure_futures", "spot_rules"):
            values, digests = [[], []], [None, None]
            for r in range(args.rounds):
                for i in (0, 1) if r % 2 == 0 else (1, 0):
                    procs[i].stdin.write(
                        json.dumps(dict(scenario=name, repeats=20)) + "\n"
                    )
                    procs[i].stdin.flush()
                    out = json.loads(procs[i].stdout.readline())
                    values[i].extend(out["samples_s"])
                    digests[i] = out["digest"]
            assert digests[0] == digests[1], (name, digests)
            row = dict(
                base_median_ms=1000 * median(values[0]),
                current_median_ms=1000 * median(values[1]),
                median_ratio=median(values[1]) / median(values[0]),
                min_ratio=min(values[1]) / min(values[0]),
                identical=True,
            )
            report["scenarios"][name] = row
            print(name, json.dumps(row), flush=True)
    finally:
        for p in procs:
            p.stdin.close()
            p.wait()
    report["load_end"] = os.getloadavg()
    out = ROOT / ".cache/panelbt/pair_futures.json"
    out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
