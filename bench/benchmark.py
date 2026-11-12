"""完整引擎基准：准备、校验、内核、后处理分别计时，不含数据生成。"""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import panelbt as bt
from panelbt import _core
from panelbt.api import _batch_result, _prepare, _result
from panelbt.reference import run as reference_run

ROOT = Path(__file__).resolve().parents[1]
assert Path(bt.__file__).resolve().is_relative_to(ROOT)


def timed(fn, repeats=3):
    samples = []
    output = None
    for _ in range(repeats):
        start = perf_counter()
        output = fn()
        samples.append(perf_counter() - start)
    return median(samples), samples, output


def market_inputs(t=4000, n=5500):
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
    delist = np.zeros((t, n), bool)
    delist[t // 2, ::101] = True
    p[::31, ::29] = np.nan
    close[::37, ::31] = np.nan
    return dict(
        exec_price=p,
        close=close,
        can_buy=buy,
        can_sell=sell,
        volume=volume,
        split=split,
        cash_div=dividend,
        delist=delist,
    )


def plans(t, n, k=1000, h=100):
    days = np.arange(0, t, 21, dtype=np.int64)
    indptr = np.arange(len(days) + 1, dtype=np.int64) * h
    template = np.arange(h)[None, :] + np.arange(len(days))[:, None] * 37
    return [
        bt.Targets(
            days,
            indptr,
            ((template + i * 11) % n).ravel(),
            np.full(len(days) * h, (0.96 + 0.039 * (i % 101) / 100) / h),
        )
        for i in range(k)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    (ROOT / ".cache/panelbt").mkdir(parents=True, exist_ok=True)
    import sys

    sys.path.insert(0, str(ROOT))
    from test.build_guard import assert_current_build

    assert_current_build()
    raw = market_inputs()
    setup, samples, market = timed(lambda: bt.Market(**raw), args.repeat)
    targets = plans(*market.shape, 1 if args.single else 1000)
    costs = bt.Costs(tax_sell=0.0005)
    report = {
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "cpus": os.cpu_count(),
            "platform": platform.platform(),
            "T": 4000,
            "N": 5500,
            "K": len(targets),
            "repeats": args.repeat,
            "features": "含全部可选面板，限制/缺失/分红/送转/退市",
        },
        "market_validation_once_s": setup,
        "market_validation_samples": samples,
    }
    print(json.dumps(report, ensure_ascii=False), flush=True)
    for label, step in [("lots", 100.0), ("fractional", 0.0)]:
        config = bt.Config(
            initial_cash=1e7, min_qty=100, qty_step=step, max_participation=0.1
        )
        bt.run(market, targets[0], costs, config)
        total, samples, _ = timed(
            lambda config=config: bt.run(market, targets[0], costs, config), args.repeat
        )
        prepare, prep_samples, prepared = timed(
            lambda config=config: _prepare(market, [targets[0]], costs, config),
            args.repeat,
        )
        kernel, kernel_samples, raw_result = timed(
            lambda prepared=prepared: _core.run(*prepared, 1), args.repeat
        )
        post, post_samples, _ = timed(
            lambda raw_result=raw_result: _result(raw_result[0]), args.repeat
        )
        report[label] = {
            "total_s": total,
            "total_samples": samples,
            "prepare_s": prepare,
            "prepare_samples": prep_samples,
            "kernel_and_binding_s": kernel,
            "kernel_samples": kernel_samples,
            "post_s": post,
            "post_samples": post_samples,
        }
        print(label, json.dumps(report[label]), flush=True)
    if not args.single:
        config = bt.Config(initial_cash=1e7, max_participation=0.1)
        total, samples, result = timed(
            lambda: bt.run_batch(market, targets, costs, config), args.repeat
        )
        assert result.nav.shape == (1000, 4000)
        prepare, prep_samples, prepared = timed(
            lambda: _prepare(market, targets, costs, config), args.repeat
        )
        kernel, kernel_samples, raw_batch = timed(
            lambda: _core.run(*prepared, 0), args.repeat
        )
        post, post_samples, _ = timed(
            lambda: _batch_result(raw_batch, *market.shape, False), args.repeat
        )
        report["batch"] = {
            "total_s": total,
            "samples": samples,
            "prepare_s": prepare,
            "prepare_samples": prep_samples,
            "kernel_and_binding_s": kernel,
            "kernel_samples": kernel_samples,
            "post_s": post,
            "post_samples": post_samples,
        }
        print("batch", json.dumps(report["batch"]), flush=True)
    small = bt.Market(**market_inputs(120, 80))
    target = plans(120, 80, 1, 20)[0]
    config = bt.Config(initial_cash=1e5, record_trades=True, max_participation=0.1)
    rust, _, a = timed(lambda: bt.run(small, target, costs, config), args.repeat)
    reference, _, b = timed(
        lambda: reference_run(small, target, costs, config), args.repeat
    )
    np.testing.assert_allclose(a.nav, b.nav, rtol=1e-9, atol=1e-9)
    report["reference"] = {"T": 120, "N": 80, "rust_s": rust, "python_s": reference}
    print("reference", json.dumps(report["reference"]), flush=True)
    out = (
        ROOT
        / ".cache/panelbt"
        / ("benchmark-single.json" if args.single else "benchmark.json")
    )
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
