"""Alternating warm AB/BA workers from pair.py and pair_futures.py, plus C1 scale."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from statistics import median
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def crypto_scale():
    import numpy as np
    import panelbt as bt

    from test.build_guard import assert_current_build

    assert_current_build()
    t, n = 70000, 100
    shape = (t, n)
    p = np.full(shape, 100.0)
    f = bt.Futures(
        np.ones(n, bool),
        np.full(n, 0.001),
        0.1,
        p,
        np.full(n, t),
        funding=np.full(shape, 0.0000001),
        maint_rate=0.004,
        liquidation=bt.Liquidation(p, p * 1.02, p * 0.98, 0.0125, 50000),
        exec_at_open=True,
        qty_step=np.full(shape, 2.0),
        min_qty=np.full(shape, 2.0),
        min_notional=np.full(shape, 5.0),
        fee_rate=0.0005,
    )
    m = bt.Market(p, p, np.ones(shape, bool), np.ones(shape, bool), futures=f)
    days = np.arange(0, t, 24, dtype=np.int64)
    targets = bt.Targets(
        days,
        np.arange(len(days) + 1, dtype=np.int64) * n,
        np.tile(np.arange(n), len(days)),
        np.full(len(days) * n, 0.02),
    )
    config = bt.Config(initial_cash=100000)
    costs = bt.Costs(slippage=0, commission_rate=0, commission_min=0)
    bt.run(m, targets, costs, config)
    samples = []
    for _ in range(7):
        start = perf_counter()
        result = bt.run(m, targets, costs, config)
        samples.append(perf_counter() - start)
    assert result.coverage["funding_paid"] > 0
    return dict(
        T=t,
        N=n,
        held_columns=n,
        rebalance_rows=24,
        record_trades=False,
        construction_included=False,
        median_s=median(samples),
        samples_s=samples,
        events={
            k: result.coverage[k]
            for k in ("funding_paid", "futures_open", "liquidation_intraday")
        },
    )


def pairs(script, names, t, n, rounds, *, same_build=False, reverse_launch=False):
    procs = [None, None]
    try:
        for old in (False, True) if reverse_launch else (True, False):
            source = ROOT / (
                ".cache/crypto-baseline"
                if old or same_build
                else "packages/panelbt/python"
            )
            env = dict(os.environ, PYTHONPATH=str(source))
            p = subprocess.Popen(
                [
                    sys.executable,
                    str(ROOT / "bench" / script),
                    "--worker",
                    "--t",
                    str(t),
                    "--n",
                    str(n),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=ROOT,
                env=env,
            )
            line = p.stdout.readline()
            assert line, p.stderr.read()
            assert json.loads(line)["ready"]
            procs[0 if old else 1] = p
        results = {}
        for name in names:
            values = [[], []]
            round_medians = [[], []]
            digests = [None, None]
            for round_index in range(rounds):
                for k in (0, 1) if round_index % 2 == 0 else (1, 0):
                    p = procs[k]
                    p.stdin.write(json.dumps(dict(scenario=name, repeats=20)) + "\n")
                    p.stdin.flush()
                    record = json.loads(p.stdout.readline())
                    values[k].extend(record["samples_s"])
                    round_medians[k].append(median(record["samples_s"]))
                    digests[k] = record.get(
                        "digest", (record.get("nav_last"), record.get("trades"))
                    )
            assert digests[0] == digests[1], (name, digests)
            results[name] = dict(
                base_median_ms=1000 * median(values[0]),
                new_median_ms=1000 * median(values[1]),
                ratio=median(values[1]) / median(values[0]),
                min_ratio=min(values[1]) / min(values[0]),
                round_ratios=[
                    new / base for base, new in zip(*round_medians, strict=True)
                ],
            )
            print(name, json.dumps(results[name]), flush=True)
        return results
    finally:
        for p in procs:
            if p is not None:
                p.stdin.close()
                p.wait(timeout=30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--forward-launch", action="store_true")
    args = parser.parse_args()
    reverse_launch = args.audit and not args.forward_launch
    from test.build_guard import assert_current_build

    report = ROOT / "bench/results/crypto_performance.json"
    result = dict(
        source=assert_current_build(),
        python_sha256={
            label: hashlib.sha256(
                b"".join((path / name).read_bytes() for name in ("api.py", "models.py"))
            ).hexdigest()
            for label, path in (
                ("base", ROOT / ".cache/crypto-baseline/panelbt"),
                ("new", ROOT / "packages/panelbt/python/panelbt"),
            )
        },
        method="Warm workers, alternating AB/BA, identical inputs and result checks",
        rounds=args.rounds,
        repeats_per_round=20,
        reverse_launch=reverse_launch,
        control_reverse_launch=reverse_launch,
        load_start=os.getloadavg(),
    )
    if args.audit:
        result["previous_measurement"] = json.loads(report.read_text())
        result["previous_measurement"].setdefault("control_reverse_launch", False)
        result["same_build_control"] = pairs(
            "pair.py",
            ["full_rules", "prices_only", "long_short", "batch16"],
            2000,
            2000,
            args.rounds,
            same_build=True,
            reverse_launch=reverse_launch,
        ) | pairs(
            "pair_futures.py",
            ["hedge_leg", "pure_futures", "spot_rules"],
            2000,
            2000,
            args.rounds,
            same_build=True,
            reverse_launch=reverse_launch,
        )
    result["legacy_spot"] = pairs(
        "pair.py",
        ["full_rules", "prices_only", "long_short", "batch16"],
        2000,
        2000,
        args.rounds,
        reverse_launch=reverse_launch,
    )
    result["legacy_futures"] = pairs(
        "pair_futures.py",
        ["hedge_leg", "pure_futures", "spot_rules"],
        2000,
        2000,
        args.rounds,
        reverse_launch=reverse_launch,
    )
    result["enabled"] = crypto_scale()
    result["load_end"] = os.getloadavg()
    import numpy as np

    rng = np.random.default_rng(20261115)
    for group in ("legacy_spot", "legacy_futures"):
        for value in result[group].values():
            ratios = np.asarray(value["round_ratios"])
            medians = np.median(rng.choice(ratios, (20000, len(ratios))), axis=1)
            value["paired_round_median_ratio"] = float(np.median(ratios))
            value["paired_round_median_bootstrap_95"] = np.quantile(
                medians, [0.025, 0.975]
            ).tolist()
    result["uncertainty"] = (
        f"Descriptive bootstrap of {args.rounds} paired round medians; shared-machine "
        "measurements do not prove exactly zero regression. "
        "All earlier measurements retained."
    )
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["enabled"]), flush=True)


if __name__ == "__main__":
    main()
