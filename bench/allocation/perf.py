"""资产配置的性能基线：30 个资产、2500 个调仓日（每天调仓，回看 252 日）。

    python bench/allocation/perf.py [--repeat 3]

分开计时：协方差估计（`ashare.risk` 的收缩估计）、风险平价求解（牛顿法）、整条 `allocate`。
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import numpy as np

import ashare.allocation as aa
from ashare.allocation.riskparity import solve_risk_parity


def data(t=2752, n=30, seed=0):
    rng = np.random.default_rng(seed)
    loadings = rng.normal(size=(n, 3)) * 0.006
    vols = rng.uniform(0.003, 0.02, n)
    r = rng.normal(size=(t, 3)) @ loadings.T + rng.normal(size=(t, n)) * vols
    r[0] = np.nan
    return r


def timed(fn, repeat):
    out = []
    for _ in range(repeat):
        started = time.perf_counter()
        fn()
        out.append(time.perf_counter() - started)
    return statistics.median(out), out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args(argv)
    r = data()
    rows = np.arange(252, len(r))
    assert len(rows) == 2500
    cov = aa.Covariance()
    sigmas = [cov.estimate(r[t - 252 : t])[0] for t in rows[:200]]
    results = {"load": os.getloadavg()[0], "assets": 30, "rebalances": len(rows)}

    def covariances():
        for t in rows:
            cov.estimate(r[t - 252 : t])

    def solves():
        for s in sigmas:
            solve_risk_parity(s)

    def full():
        aa.allocate(
            r, aa.RiskParity(), rebalance=list(rows), lookback=252, min_history=60
        )

    results["covariance_seconds"], results["covariance_runs"] = timed(
        covariances, args.repeat
    )
    m, runs = timed(solves, args.repeat)
    results["risk_parity_seconds_per_solve"] = m / len(sigmas)
    results["risk_parity_runs_200"] = runs
    results["allocate_seconds"], results["allocate_runs"] = timed(full, args.repeat)
    results["load_after"] = os.getloadavg()[0]
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
