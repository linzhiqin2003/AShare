"""先测量 numpy 参考内核，再决定 Rust 的投入。"""

import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import portopt
from threadpoolctl import threadpool_info, threadpool_limits

from ashare.risk.kernels import (
    cross_section_batch,
    ew_covariance,
    ew_variance,
    factor_covariance,
    rolling_bias,
)

ROOT = Path(__file__).resolve().parents[2]


def metadata():
    blas = np.show_config(mode="dicts")["Build Dependencies"]["blas"]
    return {
        "date": datetime.now().astimezone().isoformat(),
        "pid": os.getpid(),
        "machine": platform.platform(),
        "processor": platform.processor(),
        "cpus": os.cpu_count(),
        "rust_auto_threads": portopt.available_threads(),
        "python": sys.version,
        "numpy": np.__version__,
        "rust": subprocess.check_output(["rustc", "--version"], text=True).strip(),
        "blas": blas,
        "requested_blas_threads": 1,
        "timed_blas_threads": None if blas["name"] == "accelerate" else 1,
        "veclib_maximum_threads": os.environ.get("VECLIB_MAXIMUM_THREADS"),
        "blas_thread_note": "Accelerate thread count is not exposed by threadpoolctl; None means unverified/system managed.",
        "threadpools": threadpool_info(),
        "interference": "Other worktree sessions and system tasks may run concurrently.",
    }


def run(t=4000, n=5500, k=45):
    rng = np.random.default_rng(8241)
    b = np.column_stack(
        (np.ones(n), np.eye(31)[np.arange(n) % 31], rng.normal(size=(n, k - 32)))
    )
    f = rng.normal(0, 0.008, (t, k))
    r = f @ b.T + rng.normal(0, 0.015, (t, n))
    cap = np.exp(rng.normal(10, 1, (t, n)))
    est = rng.random((t, n)) > 0.02
    r[rng.random((t, n)) < 0.01] = np.nan
    cases = {
        "regression": lambda: cross_section_batch(r, b, cap, est, np.arange(1, 32)),
        "ew_covariance": lambda: ew_covariance(f, 504),
        "newey_west": lambda: factor_covariance(f),
        "specific_risk": lambda: ew_variance(r, 84, 5),
        "bias": lambda: rolling_bias(r, np.full_like(r, 0.02), 252),
    }
    result = {
        "metadata": metadata(),
        "shape": [t, n, k],
        "fixture": "Time-invariant B; daily returns/cap/ESTU vary. No T*N*K allocation.",
        "seconds": {},
    }
    with threadpool_limits(limits=1):
        for name, fn in cases.items():
            start = perf_counter()
            value = fn()
            elapsed = perf_counter() - start
            result["seconds"][name] = elapsed
            del value
            print(name, elapsed, flush=True)
    path = ROOT / ".cache/risk-bench/numpy-initial.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run()
