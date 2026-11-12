"""Paired full-market spot/hedge timing and an equivalent NumPy hedge baseline."""

import argparse
import json
import os
import platform
import resource
import sys
from dataclasses import replace
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import panelbt as bt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.benchmark import market_inputs  # noqa: E402
from test.build_guard import assert_current_build  # noqa: E402


def plans(t, n):
    days = np.arange(0, t, 21, dtype=np.int64)
    h = 100
    stock_cols = (np.arange(h)[None, :] + np.arange(len(days))[:, None] * 37) % n
    stock_weights = np.full((len(days), h), 0.9 / h)
    stock = bt.Targets(
        days,
        np.arange(len(days) + 1) * h,
        stock_cols.ravel(),
        stock_weights.ravel(),
    )
    expiry = np.minimum((np.arange(20) + 1) * (t // 20) - 1, t - 1)
    future = np.array([np.searchsorted(expiry, d + 5, side="right") for d in days])
    future = np.minimum(future, 19)
    mixed_cols = np.column_stack((stock_cols, n + future)).ravel()
    mixed_weights = np.column_stack((stock_weights, np.full(len(days), -0.5))).ravel()
    mixed = bt.Targets(
        days,
        np.arange(len(days) + 1) * (h + 1),
        mixed_cols,
        mixed_weights,
    )
    return stock, mixed, expiry


def add_futures(raw, expiry):
    t, n = raw["exec_price"].shape
    rows = np.arange(t)[:, None]
    cols = np.arange(20)[None, :]
    price = 2500 + 40 * np.sin(rows / 30 + cols / 4)
    settle = price + 3 * np.sin(rows / 7 + cols)
    additions = {
        "exec_price": price,
        "close": price,
        "can_buy": np.ones((t, 20), bool),
        "can_sell": np.ones((t, 20), bool),
        "volume": np.full((t, 20), 100_000.0),
        "split": np.ones((t, 20)),
        "cash_div": np.zeros((t, 20)),
        "delist": np.zeros((t, 20), bool),
    }
    combined = {
        key: np.concatenate((value, additions[key]), axis=1)
        for key, value in raw.items()
    }
    settlement = np.empty((t, n + 20))
    settlement[:, :n] = np.nan
    settlement[:, n:] = settle
    futures = bt.Futures(
        is_future=np.r_[np.zeros(n, bool), np.ones(20, bool)],
        multiplier=np.r_[np.ones(n), np.full(20, 200.0)],
        margin_rate=0.12,
        settlement=settlement,
        expiry=np.r_[np.full(n, t), expiry],
        fee_rate=0.0002,
        delivery_fee_rate=0.0001,
    )
    return bt.Market(**combined, futures=futures)


def timing(fn, repeats):
    samples = []
    result = None
    for _ in range(repeats):
        start = perf_counter()
        result = fn()
        samples.append(perf_counter() - start)
    return dict(min_s=min(samples), median_s=median(samples), samples_s=samples), result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--t", type=int, default=4000)
    parser.add_argument("--n", type=int, default=5500)
    parser.add_argument("--repeats", type=int, default=9)
    args = parser.parse_args()
    assert_current_build()
    raw = market_inputs(args.t, args.n)
    stock_plan, mixed_plan, expiry = plans(args.t, args.n)
    stock = bt.Market(**raw)
    mixed = add_futures(raw, expiry)
    costs = bt.Costs(tax_sell=0.0005)
    config = bt.Config(initial_cash=1e7, max_participation=0.1)
    jobs = {
        "spot": lambda: bt.run(stock, stock_plan, costs, config),
        "mixed": lambda: bt.run(mixed, mixed_plan, costs, config),
    }
    for fn in jobs.values():
        fn()
    samples = {key: [] for key in jobs}
    results = {}
    load_before = os.getloadavg()
    for repeat in range(args.repeats):
        for key in ("spot", "mixed") if repeat % 2 == 0 else ("mixed", "spot"):
            start = perf_counter()
            results[key] = jobs[key]()
            samples[key].append(perf_counter() - start)
    pair = {
        key: dict(min_s=min(row), median_s=median(row), samples_s=row)
        for key, row in samples.items()
    }
    pair["mixed_over_spot_min"] = pair["mixed"]["min_s"] / pair["spot"]["min_s"]
    pair["mixed_over_spot_median"] = (
        pair["mixed"]["median_s"] / pair["spot"]["median_s"]
    )
    pair["spot_nav_last"] = float(results["spot"].nav[-1])
    pair["mixed_nav_last"] = float(results["mixed"].nav[-1])

    # A sparse, fully vectorized no-friction hedge with identical dated fills.
    selected = np.array(
        [
            i
            for i in range(args.n)
            if all(i % divisor for divisor in (11, 17, 23, 29, 31, 101))
        ]
    )[:100]
    qty = np.full(len(selected), 100.0)
    future_col = args.n + 19
    future_qty = -2.0
    simple_weights = np.r_[qty, future_qty]
    simple_cols = np.r_[selected, future_col]
    simple_plan = bt.Targets(
        [0], [0, len(simple_cols)], simple_cols, simple_weights, kind="shares"
    )
    simple_f = replace(
        mixed.futures,
        fee_rate=0.0,
        fee_per_contract=0.0,
        delivery_fee_rate=0.0,
        delivery_fee_per_contract=0.0,
        expiry=np.r_[np.full(args.n, args.t), np.full(20, args.t)],
    )
    simple_market = bt.Market(
        mixed.exec_price,
        mixed.close,
        mixed.can_buy,
        mixed.can_sell,
        futures=simple_f,
    )
    zero = bt.Costs(
        commission_rate=0,
        commission_min=0,
        tax_sell=0,
        fee_rate_buy=0,
        fee_rate_sell=0,
        slippage=0,
    )
    simple_config = bt.Config(initial_cash=1e7, min_qty=0, qty_step=0)

    def numpy_hedge():
        spot_value = np.sum(simple_market.close[:, selected] * qty, axis=1)
        cash = 1e7 - np.sum(qty * simple_market.exec_price[0, selected])
        futures_pnl = (
            future_qty
            * 200
            * (
                simple_market.futures.settlement[:, future_col]
                - simple_market.exec_price[0, future_col]
            )
        )
        return (cash + spot_value + futures_pnl) / 1e7

    def simple_fn():
        return bt.run(simple_market, simple_plan, zero, simple_config)

    simple_fn()
    numpy_hedge()
    simple_r, simple_result = timing(simple_fn, args.repeats)
    numpy_r, numpy_result = timing(numpy_hedge, args.repeats)
    np.testing.assert_allclose(simple_result.nav, numpy_result, atol=1e-12, rtol=1e-12)
    report = dict(
        machine=dict(
            platform=platform.platform(),
            python=platform.python_version(),
            numpy=np.__version__,
            cpus=os.cpu_count(),
            rss_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            load_start=load_before,
            load_end=os.getloadavg(),
        ),
        parameters=dict(
            T=args.t,
            spot_N=args.n,
            futures_N=20,
            holdings=100,
            rebalance_interval=21,
            repeats=args.repeats,
            threads=1,
        ),
        paired_full_rules=pair,
        simple_exact_numpy=dict(
            panelbt=simple_r,
            numpy=numpy_r,
            max_nav_error=float(np.max(abs(simple_result.nav - numpy_result))),
        ),
    )
    output = ROOT / ".cache/panelbt/futures-current.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
