"""Alternating old/current LTO timing on identical deterministic inputs.

The historical checkout is created under .cache by the documented git archive
command. Both workers remain warm; AB/BA order alternates each round.
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from statistics import median
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from test.build_guard import assert_current_build  # noqa: E402


def inputs(t, n, actions=True, delist=True):
    import numpy as np

    rng = np.random.default_rng(801)
    p = 25 * np.exp(np.cumsum(rng.normal(0, 0.008, (t, n)), axis=0))
    close = p * np.exp(rng.normal(0, 0.002, (t, n)))
    buy, sell = np.ones((t, n), bool), np.ones((t, n), bool)
    buy[::13, ::11] = False
    sell[::17, ::7] = False
    volume = np.full((t, n), 1e7)
    volume[::19, ::9] = 500
    split = np.ones((t, n))
    split[251::500, ::23] = 1.5
    dividend = np.zeros((t, n))
    dividend[250::500, ::17] = 0.1
    d = np.zeros((t, n), bool)
    d[t // 2, ::101] = True
    p[::31, ::29] = np.nan
    close[::37, ::31] = np.nan
    out = dict(exec_price=p, close=close, can_buy=buy, can_sell=sell, volume=volume)
    if actions:
        out.update(split=split, cash_div=dividend)
    if delist:
        out["delist"] = d
    return out


def worker(t, n):
    from dataclasses import fields

    import numpy as np
    import panelbt as bt

    days = np.arange(0, t, 21, dtype=np.int64)
    target = bt.Targets(
        days,
        np.arange(len(days) + 1, dtype=np.int64) * 100,
        ((np.arange(100)[None, :] + np.arange(len(days))[:, None] * 37) % n).ravel(),
        np.full(len(days) * 100, 0.96 / 100),
    )
    costs = bt.Costs(tax_sell=0.0005)
    scenarios = {}
    for name, actions, delist, part in [
        ("full_rules", True, True, 0.1),
        ("no_delist_panel", True, False, 0.1),
        ("prices_only", False, False, 0),
    ]:
        market = bt.Market(**inputs(t, n, actions, delist))
        cfg = bt.Config(initial_cash=1e7, max_participation=part)
        bt.run(market, target, costs, cfg)
        scenarios[name] = (market, cfg)
    print(
        json.dumps(
            {
                "ready": True,
                "python": platform.python_version(),
                "legacy": "commission" in {x.name for x in fields(bt.Costs)},
            }
        ),
        flush=True,
    )
    for line in sys.stdin:
        request = json.loads(line)
        name = request["scenario"]
        market, cfg = scenarios[name]
        samples = []
        for _ in range(request["repeats"]):
            start = perf_counter()
            r = bt.run(market, target, costs, cfg)
            samples.append(perf_counter() - start)
        print(
            json.dumps(
                dict(
                    samples_s=samples,
                    nav_last=float(r.nav[-1]),
                    trades=r.diagnostics["trades"],
                )
            ),
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument(
        "--base-ref", help="Explicit revision to compare; no historical default"
    )
    parser.add_argument("--t", type=int, default=2000)
    parser.add_argument("--n", type=int, default=2000)
    args = parser.parse_args()
    if args.worker:
        return worker(args.t, args.n)
    if not args.base_ref:
        parser.error("--base-ref is required for paired timing")
    assert_current_build()
    historical = ROOT / ".cache/panelbt/pair-base"
    if historical.exists():
        shutil.rmtree(historical)
    if not (historical / ".venv/bin/python").exists():
        historical.mkdir(parents=True, exist_ok=True)
        archive = ROOT / ".cache/panelbt/pair-base.tar"
        with archive.open("wb") as stream:
            subprocess.run(
                ["git", "archive", args.base_ref], cwd=ROOT, stdout=stream, check=True
            )
        subprocess.run(["tar", "-xf", str(archive), "-C", str(historical)], check=True)
        archive.unlink()
        logdir = ROOT / ".cache/panelbt"
        logdir.mkdir(parents=True, exist_ok=True)
        with (logdir / "pair-base-build.log").open("w") as stream:
            subprocess.run(
                ["uv", "sync", "--all-packages"],
                cwd=historical,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
    paths = [historical / ".venv/bin/python", ROOT / ".venv/bin/python"]
    processes = []
    for p in paths:
        process = subprocess.Popen(
            [str(p), __file__, "--worker", "--t", str(args.t), "--n", str(args.n)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        )
        assert json.loads(process.stdout.readline())["ready"]
        processes.append(process)
    report = dict(
        platform=platform.platform(),
        cpus=os.cpu_count(),
        T=args.t,
        N=args.n,
        base_ref=args.base_ref,
        method="LTO in both builds; 6 alternating AB/BA rounds, 40 calls per scenario each round; imports/data construction excluded",
        load_start=os.getloadavg(),
        scenarios={},
    )
    try:
        for name in ["full_rules", "no_delist_panel", "prices_only"]:
            values = [[], []]
            results = [None, None]
            for round_index in range(6):
                for index in [0, 1] if round_index % 2 == 0 else [1, 0]:
                    p = processes[index]
                    p.stdin.write(json.dumps(dict(scenario=name, repeats=40)) + "\n")
                    p.stdin.flush()
                    result = json.loads(p.stdout.readline())
                    values[index].extend(result["samples_s"])
                    results[index] = result
            row = {
                label: dict(
                    min_s=min(v),
                    median_s=median(v),
                    samples_s=v,
                    nav_last=r["nav_last"],
                    trades=r["trades"],
                )
                for label, v, r in zip(
                    ["base", "current"], values, results, strict=True
                )
            }
            row["minimum_ratio"] = row["current"]["min_s"] / row["base"]["min_s"]
            row["passes_5pct"] = row["minimum_ratio"] <= 1.05
            report["scenarios"][name] = row
            print(name, row["minimum_ratio"], flush=True)
    finally:
        for p in processes:
            p.stdin.close()
            p.wait()
    report["load_end"] = os.getloadavg()
    (ROOT / ".cache/panelbt/pair.json").write_text(json.dumps(report, indent=2) + "\n")

    shutil.rmtree(historical)


if __name__ == "__main__":
    main()
