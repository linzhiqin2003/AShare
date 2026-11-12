"""Isolated representative/full-grid benchmarks, with observed CPU threads and RSS."""

import argparse
import ctypes
import importlib.metadata
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / ".cache/panelops/latest.json"
CONDITIONAL = {"ts_sumif", "ts_count"}


def cpu_threads():
    """Read CPU time for this process's OS threads (Darwin libproc, no sampling thread)."""
    if sys.platform != "darwin":
        return {}
    lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    info = lib.proc_pidinfo
    info.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint64,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    info.restype = ctypes.c_int
    ids = (ctypes.c_uint64 * 4096)()
    size = info(os.getpid(), 6, 0, ids, ctypes.sizeof(ids))
    times = {}
    for thread in ids[: size // 8]:
        buffer = ctypes.create_string_buffer(112)
        if info(os.getpid(), 5, thread, buffer, 112) == 112:
            values = (ctypes.c_uint64 * 2).from_buffer(buffer)
            times[str(thread)] = values[0] + values[1]
    return times


def output_bytes(result):
    if hasattr(result, "nbytes"):
        return result.nbytes
    if hasattr(result, "indptr"):
        return sum(
            getattr(result, k).nbytes for k in ("days", "indptr", "cols", "weights")
        )
    return sum(output_bytes(a) for a in result)


def candidate(config, x, y, groups=None):
    import numpy as np
    import panelops as po
    from additional_baselines import ELEMENTS, PAIRED, element, ts
    from cross_baselines import bottleneck_op, pandas_op, polars_op
    from panelops import _numpy

    op, method = config["op"], config["method"]
    w, mp = config["window"], max(1, config["window"] // 2)
    rust = method.startswith("rust")
    bare_method = method.replace("_default", "").replace("_1", "")
    if rust:
        po.set_num_threads(1 if method.endswith("_1") else 0)
    if op in ELEMENTS:
        z = -y if op == "where" else None
        if not rust:
            return lambda: element(x, y, op, bare_method, z)
        count = (
            3
            if op == "where"
            else 2
            if op
            in {
                "gt",
                "ge",
                "lt",
                "le",
                "eq",
                "ne",
                "and_",
                "or_",
                "maximum",
                "minimum",
                "power",
                "signed_power",
                "fill_nan",
            }
            else 1
        )
        args = (x, y, z)[:count]
        return lambda: po.elementwise(op, *args, backend="rust")
    if op.startswith("ts_"):
        name = op[3:]
        if not rust:
            return lambda: ts(
                x,
                name,
                bare_method,
                window=w,
                min_periods=mp,
                y=y if name in PAIRED else None,
            )
        args = [x, y] if name in PAIRED else [x]
        kw = (
            {"periods": w}
            if name in {"delay", "delta"}
            else {}
            if name == "ffill"
            else {"window": w, "min_periods": mp}
        )
        return lambda: getattr(po, op)(*args, **kw, backend="rust")
    if op.startswith("cs_"):
        if rust:
            kw = (
                {"groups": groups, "exposures": [y]}
                if op == "cs_neutralize"
                else {"groups": groups}
                if op == "cs_group_demean"
                else {}
            )
            return lambda: getattr(po, op)(x, **kw, backend="rust")
        if bare_method == "numpy":
            return lambda: _numpy.cs(
                x, op[3:], groups=groups, exposures=[y] if op == "cs_neutralize" else []
            )
        if op in {"cs_demean", "cs_scale"}:
            if bare_method == "bottleneck":
                import bottleneck as bn

                return lambda: (
                    x - bn.nanmean(x, axis=1)[:, None]
                    if op == "cs_demean"
                    else x / bn.nansum(abs(x), axis=1)[:, None]
                )
            if bare_method == "pandas":
                import pandas as pd

                def cs_pandas():
                    a = pd.DataFrame(x)
                    return (
                        a.sub(a.mean(axis=1), axis=0).to_numpy()
                        if op == "cs_demean"
                        else a.div(a.abs().sum(axis=1), axis=0).to_numpy()
                    )

                return cs_pandas
            import polars as pl

            def cs_polars():
                f = pl.DataFrame(
                    {"day": np.repeat(np.arange(len(x)), x.shape[1]), "x": x.ravel()}
                ).with_columns(pl.col("x").fill_nan(None))
                a = pl.col("x")
                return (
                    f.select(
                        a - a.mean().over("day")
                        if op == "cs_demean"
                        else a / a.abs().sum().over("day")
                    )
                    .to_numpy()
                    .reshape(x.shape)
                )

            return cs_polars
    if rust:
        args = [x, y] if op in {"ic", "rank_ic", "group_returns"} else [x]
        return lambda: getattr(po, op)(*args, backend="rust")
    if bare_method == "numpy":
        if op in {"ic", "rank_ic"}:
            return lambda: _numpy.ic(x, y, op == "rank_ic")
        if op == "factor_autocorr":
            return lambda: _numpy.ic(
                x, np.vstack((np.full((1, x.shape[1]), np.nan), x[:-1])), True
            )
        if op == "group_returns":
            return lambda: _numpy.group_returns(x, y)
        if op == "group_turnover":
            return lambda: _numpy.group_turnover(x)
        if op == "targets":
            return lambda: _numpy.targets(x, np.arange(len(x)), 0, 0.2)
    if bare_method == "bottleneck":
        return lambda: bottleneck_op(x, op, y=y)
    return lambda: {"pandas": pandas_op, "polars": polars_op}[bare_method](
        x, op, y=y, groups=groups
    )


def worker(config):
    import bottleneck  # noqa: F401
    import numpy as np
    import pandas  # noqa: F401
    import panelops as po
    import polars as pl

    assert Path(po.__file__).resolve().is_relative_to(ROOT)
    t, n = config["shape"]
    rng = np.random.default_rng(732)
    # Generate in-place and mask small row chunks: input preparation has no panel-sized temporary.
    x, y = np.empty((t, n)), np.empty((t, n))
    for a in (x, y):
        rng.standard_normal(out=a)
        for i in range(0, t, 16):
            view = a[i : i + 16]
            view[rng.random(view.shape) < config["missing"]] = np.nan
    conditions = {
        "and_": (x, y),
        "or_": (x, y),
        "not_": (x,),
        "where": (x,),
        "ts_sumif": (y,),
        "ts_count": (x,),
    }
    if (
        config["op"] in {"and_", "or_", "not_", "where"}
        or config.get("condition") == "half"
    ):
        # Logical conditions exercise both branches; Gaussian nonzero values would all be true.
        for panel in conditions[config["op"]]:
            for i in range(0, t, 16):
                view = panel[i : i + 16]
                view[view < 0] = 0.0
                view[view > 0] = 1.0
    groups = (
        np.broadcast_to(np.arange(n, dtype=np.int64) % 20, x.shape).copy()
        if config["op"] in {"cs_neutralize", "cs_group_demean"}
        else None
    )
    fn = candidate(config, x, y, groups)
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    memory_result = fn()
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    out_bytes = output_bytes(memory_result)
    del memory_result
    factor = 1 if sys.platform == "darwin" else 1024
    samples = []
    active = []
    for _ in range(config["repeat"]):
        counters = cpu_threads()
        start = time.perf_counter()
        result = fn()
        elapsed = time.perf_counter() - start
        end = cpu_threads()
        active.append(sum(v - counters.get(k, 0) > 250_000 for k, v in end.items()))
        samples.append(elapsed)
        del result
    return dict(
        status="ok",
        median_ms=statistics.median(samples) * 1000,
        incremental_mib=max(0, (after - before) * factor - out_bytes) / 2**20,
        output_mib=out_bytes / 2**20,
        peak_rss_mib=after * factor / 2**20,
        observed_cpu_threads=active,
        polars_pool=pl.thread_pool_size(),
        accelerate_limit=os.environ.get("VECLIB_MAXIMUM_THREADS", "default"),
        rust_pool=po._api._core.actual_threads(po.get_num_threads())
        if config["method"].startswith("rust")
        else None,
        build_id=po.build_id,
        measured_at=datetime.now().astimezone().isoformat(),
    )


def methods(op):
    from additional_baselines import ELEMENTS

    result = [
        "numpy",
        "bottleneck",
        "pandas",
        "polars_1",
        "polars_default",
        "rust_1",
        "rust_default",
    ]
    if op.startswith("ts_") and op not in ("ts_delay", "ts_delta"):
        result[5:5] = ["polars_long_1", "polars_long_default"]
    if op == "cs_neutralize":
        return ["numpy_1", "numpy", "rust_1", "rust_default"]
    if op in ELEMENTS:
        result.remove("bottleneck")
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worker")
    p.add_argument("--full", action="store_true")
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--timeout", type=float, default=120)
    p.add_argument("--shape", nargs=2, type=int)
    p.add_argument("--ops", nargs="*")
    p.add_argument("--methods", nargs="*")
    p.add_argument("--ignore-pause", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--replace", action="store_true")
    p.add_argument("--compare", type=Path)
    a = p.parse_args()
    if a.worker:
        try:
            with warnings.catch_warnings(), __import__("numpy").errstate(all="ignore"):
                warnings.simplefilter("ignore")
                result = worker(json.loads(a.worker))
        except NotImplementedError as error:
            result = {"status": "unsupported", "error": str(error)}
        except Exception as error:
            result = {"status": "error", "error": f"{type(error).__name__}: {error}"}
        print(json.dumps(result), flush=True)
        return
    from additional_baselines import ELEMENTS
    from panelops.api import CS_OPS, TS_OPS

    operations = a.ops or [
        *("ts_" + o for o in TS_OPS),
        *("cs_" + o for o in CS_OPS),
        "ic",
        "rank_ic",
        "group_returns",
        "group_turnover",
        "factor_autocorr",
        "targets",
        *ELEMENTS,
    ]
    report = dict(
        schema=2,
        date=datetime.now().astimezone().isoformat(),
        machine=platform.platform(),
        cpu=platform.processor(),
        cpus=os.cpu_count(),
        python=platform.python_version(),
        versions={
            n: importlib.metadata.version(n)
            for n in ["numpy", "pandas", "polars", "bottleneck", "panelops"]
        },
        note="Concurrent external tasks may interfere. Each method runs in a fresh process. Incremental RSS = max(0, first-call RSS high-water growth minus retained output bytes); allocator/page granularity makes this a lower-bound estimate, not an allocation profiler. Observed CPU threads count per-thread CPU increments >0.25 ms around each timed call (Darwin libproc); it includes caller work. Non-BLAS NumPy kernels use one thread. Polars single-thread workers set POLARS_MAX_THREADS=1; NumPy single-thread workers set VECLIB_MAXIMUM_THREADS=1. All other libraries keep their defaults.",
        measurements=[],
    )
    if a.resume and a.output.exists():
        report = json.loads(a.output.read_text())

    def key(c):
        return (
            c["op"],
            c["method"],
            tuple(c["shape"]),
            c["window"],
            c["missing"],
            c.get("condition"),
        )

    if a.replace:
        report["measurements"] = [
            c
            for c in report["measurements"]
            if not (
                c["op"] in operations and (not a.methods or c["method"] in a.methods)
            )
        ]
    seen = {key(c) for c in report["measurements"]}
    configs = []
    for shape in (
        [a.shape]
        if a.shape
        else [(1000, 1000), (4000, 5500)]
        if a.full
        else [(4000, 5500)]
    ):
        for missing in [0.0, 0.05, 0.3] if a.full else [0.05]:
            for op in operations:
                for w in [5, 20, 60, 250] if a.full and op.startswith("ts_") else [20]:
                    for method in methods(op):
                        if a.methods and method not in a.methods:
                            continue
                        # Conditional reductions also run with a balanced 0/1 condition:
                        # Gaussian conditions are almost all true and hide branch costs.
                        for condition in (
                            [None, "half"] if op in CONDITIONAL else [None]
                        ):
                            config = dict(
                                op=op,
                                method=method,
                                shape=list(shape),
                                window=w,
                                missing=missing,
                                repeat=a.repeat,
                            )
                            if condition:
                                config["condition"] = condition
                            configs.append(config)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    for i, c in enumerate(configs):
        if key(c) in seen:
            continue
        while (
            not a.ignore_pause and (ROOT / ".cache/panelops-benchmark-pause").exists()
        ):
            time.sleep(0.1)
        env = os.environ.copy()
        if c["method"].startswith("polars") and c["method"].endswith("_1"):
            env["POLARS_MAX_THREADS"] = "1"
        if c["method"] == "numpy_1":
            env["VECLIB_MAXIMUM_THREADS"] = "1"
        try:
            proc = subprocess.run(
                [sys.executable, __file__, "--worker", json.dumps(c)],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=a.timeout,
            )
            entry = json.loads(proc.stdout.strip().splitlines()[-1])
        except subprocess.TimeoutExpired:
            entry = dict(
                status="timeout",
                error=f"process exceeded {a.timeout}s including setup, memory run and timed repetitions",
            )
        except (ValueError, IndexError):
            entry = dict(
                status="error", error=proc.stderr[-1200:] or f"exit {proc.returncode}"
            )
        entry = c | entry
        report["measurements"].append(entry)
        temporary = a.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(a.output)
        print(
            f"{i + 1}/{len(configs)} {c['op']} {c['method']}: "
            + (
                f"{entry['median_ms']:.3f} ms +{entry['incremental_mib']:.1f} MiB"
                if entry["status"] == "ok"
                else entry["status"] + " " + entry["error"]
            ),
            flush=True,
        )
    if a.compare:
        old = {
            key(c): c
            for c in json.loads(a.compare.read_text())["measurements"]
            if c["status"] == "ok"
        }
        changes = [
            {
                "op": c["op"],
                "method": c["method"],
                "ratio": c["median_ms"] / old[key(c)]["median_ms"],
            }
            for c in report["measurements"]
            if c["status"] == "ok" and key(c) in old
        ]
        a.output.with_suffix(".comparison.json").write_text(
            json.dumps(changes, indent=2)
        )
        print("Regressions >10%:", sum(c["ratio"] > 1.1 for c in changes))


if __name__ == "__main__":
    main()
