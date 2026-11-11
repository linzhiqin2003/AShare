"""最小核心的三方对拍和摸底；完整批量实测，不按小样本外推。"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import multiprocessing as mp
import os
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / ".cache" / "spike"
MARKET = None


def simulate(p, close, h, interval, variant=0, sparse=True):
    t, n = p.shape
    cash = 1e7
    shares = np.zeros(n)
    weights = np.zeros(n)
    nav = np.empty(t)
    held = np.empty(0, dtype=np.int64)
    all_cols = np.arange(n)
    template = np.arange(h)
    exposure = 0.96 + 0.039 * (variant % 101) / 100
    for day in range(t):
        if day % interval == 0:
            value = (
                cash + np.dot(shares[held], p[day, held])
                if sparse
                else cash + np.dot(shares, p[day])
            )
            cols = (day // interval * 37 + template + variant * 11) % n
            if sparse:
                weights[held] = 0
            else:
                weights.fill(0)
            weights[cols] = exposure / h
            active = np.union1d(held, cols) if sparse else all_cols
            prices = p[day, active] if sparse else p[day]
            q = shares[active] if sparse else shares.copy()
            target = np.floor(weights[active] * value / prices / 100) * 100
            sold = np.maximum(q - target, 0)
            amounts = sold * prices * (1 - 0.0005)
            fees = np.where(sold > 0, np.maximum(amounts * 0.00025, 5), 0)
            cash += np.sum(amounts - fees - amounts * 0.001)
            q -= sold
            buy = np.maximum(target - q, 0)
            bp = prices * (1 + 0.0005)
            amounts = buy * bp
            need = amounts + np.where(buy > 0, np.maximum(amounts * 0.00025, 5), 0)
            total = need.sum()
            if total > cash:
                buy = np.floor(buy * (cash / total) / 100) * 100
                amounts = buy * bp
                need = amounts + np.where(buy > 0, np.maximum(amounts * 0.00025, 5), 0)
            # 绝大多数订单可整体入账；最低佣金导致仍超额时才逐笔处理。
            if need.sum() <= cash:
                cash -= need.sum()
                q += buy
            else:
                for j in np.flatnonzero(buy):
                    x = buy[j]
                    if need[j] > cash:
                        x = (
                            np.floor(
                                max(
                                    0, min(cash / (bp[j] * 1.00025), (cash - 5) / bp[j])
                                )
                                / 100
                            )
                            * 100
                        )
                    if x:
                        v = x * bp[j]
                        cash -= v + max(v * 0.00025, 5)
                        q[j] += x
            shares[active] = q
            held = active[q > 0]
        nav[day] = (
            cash
            + (
                np.dot(shares[held], close[day, held])
                if sparse
                else np.dot(shares, close[day])
            )
        ) / 1e7
    return nav


def initialize():
    global MARKET
    MARKET = (
        np.load(CACHE / "exec.npy", mmap_mode="r"),
        np.load(CACHE / "close.npy", mmap_mode="r"),
    )


def job(args):
    h, interval, variant, sparse = args
    return simulate(*MARKET, h, interval, variant, sparse)


def timed(fn, repeat):
    times = []
    value = None
    for _ in range(repeat):
        start = perf_counter()
        value = fn()
        times.append(perf_counter() - start)
    return median(times), times, value


def main():
    import panelbt_spike

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--k", type=int, default=1000)
    args = parser.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    if not (CACHE / "exec.npy").exists():
        rng = np.random.default_rng(701)
        p = 25 * np.exp(np.cumsum(rng.normal(0, 0.008, (4000, 5500)), axis=0))
        np.save(CACHE / "exec.npy", p)
        np.save(CACHE / "close.npy", p * np.exp(rng.normal(0, 0.002, p.shape)))
    initialize()
    p, close = MARKET
    records = []
    for h, interval in [(100, 21), (1000, 21), (100, 1), (1000, 1)]:
        row = {"holdings": h, "interval": interval}
        # 先三方逐日净值对拍，然后才允许计时。
        a = simulate(p, close, h, interval, sparse=False)
        b = simulate(p, close, h, interval, sparse=True)
        c = np.asarray(
            panelbt_spike.run(p, close, h, interval, np.array([0], dtype=np.int64), 1)
        )[0]
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(a, c, rtol=1e-9, atol=1e-9)
        row["max_nav_error"] = float(max(np.max(np.abs(a - b)), np.max(np.abs(a - c))))
        for name, fn in [
            (
                "numpy_full",
                lambda h=h, interval=interval: simulate(
                    p, close, h, interval, sparse=False
                ),
            ),
            (
                "numpy_sparse",
                lambda h=h, interval=interval: simulate(
                    p, close, h, interval, sparse=True
                ),
            ),
            (
                "rust",
                lambda h=h, interval=interval: panelbt_spike.run(
                    p, close, h, interval, np.array([0], dtype=np.int64), 1
                ),
            ),
        ]:
            row[name], row[name + "_samples"], _ = timed(fn, args.repeat)
        print(json.dumps(row), flush=True)
        records.append(row)
    (CACHE / "single.json").write_text(json.dumps(records, indent=2))
    if not args.batch:
        return
    # 先用同样 T/N 的 30 组筛选并行方式，再对胜者实测完整 1000 组。
    # 筛选值单独列出，不把小批量按比例外推为正式吞吐。
    for row in records:
        h, interval = row["holdings"], row["interval"]
        sparse = row["numpy_sparse"] < row["numpy_full"]
        tasks = [(h, interval, i, sparse) for i in range(args.k)]
        pilot = tasks[:30]
        selector = []
        candidates = [
            ("serial", 1),
            ("threads", 1),
            ("threads", 2),
            ("threads", 4),
            ("processes", os.cpu_count()),
        ]
        expected = None
        for name, workers in candidates:
            if name == "serial":
                seconds, _, result = timed(lambda pilot=pilot: list(map(job, pilot)), 1)
            else:
                cls = (
                    cf.ThreadPoolExecutor
                    if name == "threads"
                    else cf.ProcessPoolExecutor
                )
                options = {"max_workers": workers, "initializer": initialize}
                if name == "processes":
                    options["mp_context"] = mp.get_context("spawn")
                with cls(**options) as pool:
                    list(pool.map(job, pilot, chunksize=1))
                    seconds, _, result = timed(
                        lambda pool=pool, pilot=pilot: list(
                            pool.map(job, pilot, chunksize=1)
                        ),
                        1,
                    )
            result = np.asarray(result)
            if expected is None:
                expected = result
            else:
                np.testing.assert_array_equal(expected, result)
            selector.append(
                {
                    "method": name,
                    "workers": workers,
                    "K": len(pilot),
                    "seconds": seconds,
                }
            )
            print(
                json.dumps(
                    {"holdings": h, "interval": interval, "selector": selector[-1]}
                ),
                flush=True,
            )
        best = min(selector, key=lambda item: item["seconds"])
        row["selection"] = selector
        row["numpy_batch_method"] = best["method"]
        row["numpy_batch_workers"] = best["workers"]
        if best["method"] == "serial":
            seconds, samples, result = timed(
                lambda tasks=tasks: list(map(job, tasks)), args.repeat
            )
        else:
            cls = (
                cf.ThreadPoolExecutor
                if best["method"] == "threads"
                else cf.ProcessPoolExecutor
            )
            options = {"max_workers": best["workers"], "initializer": initialize}
            if best["method"] == "processes":
                options["mp_context"] = mp.get_context("spawn")
            with cls(**options) as pool:
                list(pool.map(job, pilot, chunksize=1))
                seconds, samples, result = timed(
                    lambda pool=pool, tasks=tasks: list(
                        pool.map(job, tasks, chunksize=8)
                    ),
                    args.repeat,
                )
        row["batch_numpy"] = seconds
        row["batch_numpy_samples"] = samples
        expected = np.asarray(result)
        print(
            json.dumps(
                {
                    "holdings": h,
                    "interval": interval,
                    "K": args.k,
                    "method": best["method"],
                    "seconds": seconds,
                }
            ),
            flush=True,
        )
        seconds, samples, result = timed(
            lambda h=h, interval=interval: panelbt_spike.run(
                p, close, h, interval, np.arange(args.k, dtype=np.int64), 0
            ),
            args.repeat,
        )
        np.testing.assert_allclose(expected, result, rtol=1e-9, atol=1e-9)
        row["batch_rust"] = seconds
        row["batch_rust_samples"] = samples
        print(
            json.dumps(
                {
                    "holdings": h,
                    "interval": interval,
                    "K": args.k,
                    "method": "rayon",
                    "seconds": seconds,
                }
            ),
            flush=True,
        )
        (CACHE / "results.json").write_text(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
