"""One isolated timed job; stdout contains progress and JSON is written atomically."""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import importlib
import importlib.metadata
import json
import multiprocessing as mp
import os
import resource
import sys
from pathlib import Path
from statistics import median
from time import perf_counter, sleep, time

import numpy as np
import panelbt as panel
from common import inputs, plan
from panelbt.reference import run as reference
from prepared import clone_inputs, execute_block, initialize_block, prepare, ready

MARKET = None
ENGINE = None
TASK = None
PLANS = None


def initialize(engine, t, n, interval, task, k):
    global MARKET, ENGINE, TASK, PLANS
    MARKET = inputs(t, n, interval, task=task)
    ENGINE = engine
    TASK = task
    # Synthetic target generation is outside both timing scopes, just like prices.
    PLANS = [plan(t, n, interval, i, task) for i in range(k)]
    if engine not in ("panelbt", "reference"):
        importlib.import_module(
            engine if engine not in ("numpy_full", "numpy_sparse") else "numpy_daily"
        )


def evaluate(engine, market, target, costs, config, task):
    if engine == "panelbt":
        return panel.run(market, target, costs, config).nav
    if engine == "reference":
        return reference(market, target, costs, config).nav
    if engine in ("numpy_full", "numpy_sparse"):
        return importlib.import_module("numpy_daily").run(
            market, target, task, sparse=engine == "numpy_sparse"
        )
    module = importlib.import_module(engine)
    if engine in ("backtrader_strategy", "bt_strategy"):
        return module.run(market, target, task)
    if engine == "vectorbt_strategy" and task == 2:
        return module.run_lots(market, target)
    return module.run(market, target)


def batch_job(args):
    variant, _interval = args
    m, _, c, g = MARKET
    return evaluate(ENGINE, m, PLANS[variant], c, g, TASK)


def batch_zero_fee_job(args):
    """Independent zero-fee control for every approximate batch portfolio."""
    from dataclasses import replace

    variant, _interval = args
    m, _, c, g = MARKET
    module = importlib.import_module(ENGINE)
    original = module.FEE
    try:
        module.FEE = 0.0
        return evaluate(
            ENGINE, m, PLANS[variant], replace(c, commission_rate=0), g, TASK
        )
    finally:
        module.FEE = original


def main():
    parser = argparse.ArgumentParser()
    for key in ("t", "n", "interval", "task", "k"):
        parser.add_argument("--" + key, type=int, required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    pause = Path(__file__).resolve().parents[2] / ".cache/benchmark-pause"
    while pause.exists():
        sleep(0.5)

    def stage(name):
        heartbeat = Path(args.output + ".running")
        temporary = Path(args.output + ".running.tmp")
        temporary.write_text(json.dumps({"started_at": time(), "phase": name}))
        temporary.replace(heartbeat)

    stage("setup")
    start = perf_counter()
    m, target, c, g = inputs(args.t, args.n, args.interval, task=args.task)
    engine = args.engine
    if engine not in ("panelbt", "reference"):
        module = importlib.import_module(
            engine if engine not in ("numpy_full", "numpy_sparse") else "numpy_daily"
        )
    report = dict(
        engine=engine,
        T=args.t,
        N=args.n,
        interval=args.interval,
        task=args.task,
        K=args.k,
        setup_s=perf_counter() - start,
        threads=1,
        status="ok",
        initial_cash=g.initial_cash,
    )
    expected_result = panel.run(m, target, c, g)
    expected = expected_result.nav
    report["panelbt_diagnostics"] = expected_result.diagnostics
    report["adapter_sha256"] = (
        hashlib.sha256(
            Path(__file__).with_name(engine + ".py").read_bytes()
        ).hexdigest()
        if Path(__file__).with_name(engine + ".py").exists()
        else None
    )
    stage("first_call")
    start = perf_counter()
    actual = evaluate(engine, m, target, c, g, args.task)
    first = perf_counter() - start
    report["first_call_s"] = first
    report["jit_first_call_s"] = first if engine == "vectorbt_strategy" else None
    report["max_nav_error"] = float(np.max(np.abs(actual - expected)))
    approximate = args.task == 1 and engine in (
        "numpy_approx",
        "polars_approx",
        "bt_strategy",
    )
    if approximate:
        from dataclasses import replace

        original = module.FEE
        try:
            module.FEE = 0.0
            stage("zero_fee_control")
            free = evaluate(
                engine, m, target, replace(c, commission_rate=0), g, args.task
            )
            free_expected = panel.run(m, target, replace(c, commission_rate=0), g).nav
            np.testing.assert_allclose(free, free_expected, atol=1e-9, rtol=1e-9)
            report["zero_fee_error"] = float(np.max(np.abs(free - free_expected)))
        finally:
            module.FEE = original
        report["semantics"] = (
            "post-cost allocation / frictionless drift fee approximation; zero-fee control matches"
        )
    else:
        np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=1e-9)
        report["semantics"] = "same daily NAV within 1e-9"
    pool = None
    if args.k > 1:
        targets = [
            plan(args.t, args.n, args.interval, i, args.task) for i in range(args.k)
        ]
        fingerprints = {
            hashlib.sha256(
                x.days.tobytes()
                + x.indptr.tobytes()
                + x.cols.tobytes()
                + x.weights.tobytes()
            ).digest()
            for x in targets
        }
        assert len(fingerprints) == args.k, "batch target plans must be distinct"
        report["unique_targets"] = len(fingerprints)
        if engine == "panelbt":

            def fn():
                return panel.run_batch(m, targets, c, g, threads=0).nav

            report["threads"] = os.cpu_count()
            report["batch_method"] = "rayon run_batch"
        else:
            # Compare native batching, serial calls, and warmed spawned workers.
            workers = os.cpu_count() or 1
            pool = cf.ProcessPoolExecutor(
                max_workers=workers,
                mp_context=mp.get_context("spawn"),
                initializer=initialize,
                initargs=(engine, args.t, args.n, args.interval, args.task, args.k),
            )
            pilot = [(i, args.interval) for i in range(min(16, args.k))]
            stage("batch_warmup")
            list(pool.map(batch_job, pilot))
            begin = perf_counter()
            parallel = np.asarray(list(pool.map(batch_job, pilot)))
            parallel_s = perf_counter() - begin
            begin = perf_counter()
            serial = np.asarray(
                [evaluate(engine, m, targets[i], c, g, args.task) for i, _ in pilot]
            )
            serial_s = perf_counter() - begin
            np.testing.assert_allclose(parallel, serial, atol=1e-9, rtol=1e-9)
            timings = {"serial_s": serial_s, "processes_s": parallel_s}
            if engine in ("vectorbt_strategy", "vectorbt_rust_strategy"):
                stage("batch_compile_warmup")
                begin = perf_counter()
                module.batch(m, targets[: len(pilot)])
                report["batch_first_call_s"] = perf_counter() - begin
                begin = perf_counter()
                native = module.batch(m, targets[: len(pilot)])
                timings["native_grouped_s"] = perf_counter() - begin
                np.testing.assert_allclose(native, serial, atol=1e-9, rtol=1e-9)
            report["batch_pilot"] = {"K": len(pilot), **timings, "workers": workers}
            winner = min(timings, key=timings.get)
            if winner == "processes_s":

                def fn():
                    return np.asarray(
                        list(
                            pool.map(
                                batch_job,
                                [(i, args.interval) for i in range(args.k)],
                                chunksize=4,
                            )
                        )
                    )

                report["threads"] = workers
                report["batch_method"] = (
                    "spawn workers; shared market per worker (pilot winner)"
                )
            elif winner == "native_grouped_s":

                def fn():
                    return module.batch(m, targets)

                report["batch_method"] = (
                    "grouped native portfolios in chunks of 20 (pilot winner)"
                )
            else:

                def fn():
                    return np.asarray(
                        [evaluate(engine, m, x, c, g, args.task) for x in targets]
                    )

                report["batch_method"] = "serial (pilot winner)"
        # Check every strategy, not merely the first row or a summary statistic.
        # Copy only NAV so the correctness oracle does not retain K full results
        # and dominate a competitor's measured process high-water mark.
        expected_batch = np.asarray([panel.run(m, x, c, g).nav.copy() for x in targets])
        if approximate:
            stage("batch_zero_fee_control")
            free_batch = np.asarray(
                list(
                    pool.map(
                        batch_zero_fee_job,
                        [(i, args.interval) for i in range(args.k)],
                        chunksize=4,
                    )
                )
            )
            free_costs = replace(c, commission_rate=0)
            free_expected = np.asarray(
                [panel.run(m, x, free_costs, g).nav.copy() for x in targets]
            )
            np.testing.assert_allclose(free_batch, free_expected, atol=1e-9, rtol=1e-9)
            report["batch_zero_fee_error"] = float(
                np.max(np.abs(free_batch - free_expected))
            )
            report["zero_fee_portfolios_checked"] = args.k
    else:

        def fn():
            return evaluate(engine, m, target, c, g, args.task)

    # Both scopes are measured directly, never estimated by subtracting medians.
    # Synthetic data generation and imports are outside both scopes.
    base_fn = fn

    def inclusive():
        if engine in ("panelbt", "reference"):
            fresh_m, fresh_targets = clone_inputs(
                m, targets if args.k > 1 else [target]
            )
        else:
            # Canonical NumPy panels/CSR are already the synthetic raw inputs;
            # competitors pay their own format conversion, not panelbt validation.
            fresh_m, fresh_targets = m, targets if args.k > 1 else [target]
        if args.k == 1:
            return evaluate(engine, fresh_m, fresh_targets[0], c, g, args.task)
        if engine == "panelbt":
            return panel.run_batch(fresh_m, fresh_targets, c, g, threads=0).nav
        return base_fn()

    canonical_costed_batch = None

    def sample(function, prefix):
        nonlocal canonical_costed_batch
        values = []
        for repeat in range(3):
            stage(f"{prefix}_{repeat}")
            begin = perf_counter()
            value = function()
            elapsed = perf_counter() - begin
            values.append(elapsed)
            expected_value = expected_batch if args.k > 1 else expected
            if not approximate:
                np.testing.assert_allclose(value, expected_value, atol=1e-9, rtol=1e-9)
            elif args.k == 1:
                np.testing.assert_allclose(value, actual, atol=1e-9, rtol=1e-9)
            elif canonical_costed_batch is None:
                canonical_costed_batch = np.asarray(value).copy()
            else:
                np.testing.assert_allclose(
                    value, canonical_costed_batch, atol=1e-9, rtol=1e-9
                )
            if args.k > 1:
                report["max_batch_nav_error"] = max(
                    report.get("max_batch_nav_error", 0.0),
                    float(np.max(np.abs(value - expected_value))),
                )
                report["max_nav_error"] = report["max_batch_nav_error"]
            if elapsed > 60:
                report["single_sample_reason"] = "single run > 60 seconds"
                break
        return values

    if (
        args.k == 1
        and first > 60
        and engine not in ("panelbt", "reference", "vectorbt_strategy")
    ):
        inclusive_samples = [first]
        report["inclusive_uses_validated_first_call"] = True
        report["single_sample_reason"] = (
            "validated non-JIT first call > 60 seconds; identical inclusive boundary"
        )
    else:
        inclusive_samples = sample(inclusive, "including_preparation")
    if pool:
        pool.shutdown()
    stage("prepare_execution_only")
    prepared_pools = []
    if args.k == 1:
        executor = prepare(engine, m, target, c, g, args.task)
    elif engine == "panelbt":
        from panelbt import _core
        from panelbt.api import _batch_result, _prepare

        arguments = _prepare(m, targets, c, g)

        def executor():
            return _batch_result(_core.run(*arguments, 0), *m.shape, False).nav
    elif "native" in report["batch_method"]:
        prepared_batch = module.prepare_batch(m, targets)

        def executor():
            return module.execute_batch(prepared_batch)
    elif "spawn" in report["batch_method"]:
        # Dedicated workers keep fixed disjoint prepared inputs across repeats.
        workers = report["threads"]
        for worker in range(workers):
            lo = args.k * worker // workers
            hi = args.k * (worker + 1) // workers
            prepared_pools.append(
                cf.ProcessPoolExecutor(
                    max_workers=1,
                    mp_context=mp.get_context("spawn"),
                    initializer=initialize_block,
                    initargs=(
                        engine,
                        args.t,
                        args.n,
                        args.interval,
                        args.task,
                        list(range(lo, hi)),
                    ),
                )
            )

        def executor():
            futures = [p.submit(execute_block) for p in prepared_pools]
            return np.asarray([row for f in futures for row in f.result()])
    else:
        functions = [prepare(engine, m, x, c, g, args.task) for x in targets]

        def executor():
            return np.asarray([f() for f in functions])

    # Force process initialization and JIT without an extra full slow batch.
    stage("prepared_warmup")
    futures = [p.submit(ready) for p in prepared_pools]
    for future in futures:
        future.result()
    execution_samples = sample(executor, "execution_only")
    for p in prepared_pools:
        p.shutdown()
    report["samples_s"] = execution_samples
    report["median_s"] = median(execution_samples)
    report["execution_samples_s"] = execution_samples
    report["including_preparation_samples_s"] = inclusive_samples
    report["including_preparation_median_s"] = median(inclusive_samples)
    report["timing_boundary"] = (
        "execution-only: immutable adapter inputs prepared; engine instantiation and result included. Inclusive: Market/Targets validation plus adapter conversion. Synthetic RNG and imports excluded. Reference retains its independent expansion in execution."
    )
    rss_unit = 1 if sys.platform == "darwin" else 1024
    report["peak_rss_bytes"] = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * rss_unit
    )
    report["child_peak_rss_bytes"] = (
        resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * rss_unit
    )
    report["rss_scope"] = (
        "process high-water mark including input setup, warmup and batch method pilot; child field is largest child, not sum"
    )
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
