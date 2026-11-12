"""归因与容量的性能基线：python -m ashare.analysis.bench [--repeat 3]

规模取十五年日频（T=3700）、K=42 个因子：
- 归因数值核心 `decompose`：n=1000（组合与沪深 300 出现过的证券的量级，整段放内存）；
  以及全市场 n=5923——全市场十五年的暴露有 7 GB，这里用一个按块返回同一块暴露的
  对象代替（计算量和访存模式与真实数据相同，不含从风险模型读暴露的时间）。
- 多期链接、Brinson（全市场 T×N）、逐日风险拆分（n=1000）、容量的一次解析模拟。
每项重复 --repeat 次取中位数，并记录当时的 uptime 负载。
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time

import numpy as np

from .attribution import decompose
from .brinson import brinson_effects, sector_aggregate
from .linking import link

T, K = 3700, 42


class _Repeat:
    """暴露提供者：任意切片都返回同一块预生成的暴露（只为测计算量）。"""

    def __init__(self, n, k, chunk=128, seed=0):
        rng = np.random.default_rng(seed)
        self.block = rng.normal(0, 1, (chunk, n, k))
        self.shape = (T, n, k)

    def __getitem__(self, s):
        start, stop, _ = s.indices(T)
        return self.block[: stop - start]


def _time(fn, repeat):
    out = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return statistics.median(out), out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)
    rng = np.random.default_rng(1)
    results = {"load": os.getloadavg(), "T": T, "K": K}

    for n, label in ((1000, "decompose_n1000"), (5923, "decompose_full_market")):
        held = rng.uniform(0, 1, (T, n)) / n
        bench = rng.uniform(0, 1, (T, n)) / n
        r = rng.normal(0, 0.02, (T, n))
        f = rng.normal(0, 0.005, (T, K))
        b = rng.normal(0, 1, (T, n, K)) if n == 1000 else _Repeat(n, K)
        med, runs = _time(
            lambda h=held, w=bench, x=r, e=b, g=f: decompose(h, w, x, e, g),
            args.repeat,
        )
        results[label] = {"median": med, "runs": runs, "n": n}

    a = rng.normal(0, 1e-4, (T, K + 3))
    rp, rb = rng.normal(0, 0.01, T), rng.normal(0, 0.01, T)
    for method in ("grap", "carino"):
        med, runs = _time(lambda m=method: link(a, rp, rb, m), args.repeat)
        results[f"link_{method}"] = {"median": med, "runs": runs}

    n = 5923
    held = rng.uniform(0, 1, (T, n)) / n
    bench = rng.uniform(0, 1, (T, n)) / n
    r = rng.normal(0, 0.02, (T, n))
    sector = rng.integers(0, 31, (T, n))

    def run_brinson():
        wp, cp = sector_aggregate(held, r, sector, 31)
        wb, cb = sector_aggregate(bench, r, sector, 31)
        with np.errstate(invalid="ignore", divide="ignore"):
            effects = brinson_effects(wp, wb, cp / wp, cb / wb)
        daily = np.column_stack([e.sum(1) for e in effects])
        link(daily, cp.sum(1), cb.sum(1))

    med, runs = _time(run_brinson, args.repeat)
    results["brinson_full_market"] = {"median": med, "runs": runs}

    from .history import FactorHistory
    from .risk import risk_series

    n = 1000
    dates = np.array([f"{20000000 + i:08d}" for i in range(T)])
    symbols = tuple(f"S{i:05d}" for i in range(n))
    cov = np.broadcast_to(np.eye(K) * 1e-4, (T, K, K))
    hist = FactorHistory(
        dates,
        symbols,
        ("country",) + tuple(f"style_{i}" for i in range(K - 1)),
        rng.normal(0, 1, (T, n, K)),
        rng.normal(0, 0.005, (T, K)),
        cov,
        np.full((T, n), 4e-4),
    )

    close = 10 * np.cumprod(1 + rng.normal(0, 0.02, (T, n)), axis=0)
    fields = {"close": close, "adj_factor": np.ones((T, n))}

    class Panel:
        def __getitem__(self, k):
            return fields[k]

    p = Panel()
    p.dates, p.symbols = dates, symbols
    weights = rng.uniform(0, 1, (T, n)) / n
    med, runs = _time(lambda: risk_series(weights, p, hist), args.repeat)
    results["risk_series_n1000"] = {"median": med, "runs": runs}
    print(json.dumps(results, indent=1, default=float))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(results, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
