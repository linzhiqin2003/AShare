"""完整规模、逐项对拍并记录 numpy/Rust 的真实耗时。"""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import portopt
from baseline import metadata
from threadpoolctl import threadpool_limits

from ashare.risk.kernels import cross_section_batch, ew_variance, rolling_bias


def run(output=None):
    rng = np.random.default_rng(8241)
    t, n, k = 4000, 5500, 45
    b = np.column_stack(
        (np.ones(n), np.eye(31)[np.arange(n) % 31], rng.normal(size=(n, k - 32)))
    )
    f = rng.normal(0, 0.008, (t, k))
    r = f @ b.T + rng.normal(0, 0.015, (t, n))
    cap = np.exp(rng.normal(10, 1, (t, n)))
    est = rng.random((t, n)) > 0.02
    r[rng.random((t, n)) < 0.01] = np.nan
    p = np.full_like(r, 0.02)
    results = {
        "metadata": metadata(),
        "shape": [t, n, k],
        "repeats": 3,
        "seconds": {},
        "max_errors": {},
    }
    cases = {
        "regression": (
            lambda: cross_section_batch(r, b, cap, est, np.arange(1, 32)),
            lambda threads: portopt.regress(r, b, cap, est, np.arange(1, 32), threads),
        ),
        "specific_risk": (
            lambda: ew_variance(r, 84, 5),
            lambda threads: portopt.ew_variance(r, 84, 5, threads),
        ),
        "bias": (
            lambda: rolling_bias(r, p, 252),
            lambda threads: portopt.rolling_bias(r, p, 252, threads),
        ),
    }
    path = Path(output) if output else Path(".cache/risk-bench") / "kernels.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        for name, (reference, optimized) in cases.items():
            ref = reference()
            results["seconds"][name] = {}
            results["max_errors"][name] = {}
            for method, fn in [("numpy", reference)] + [
                (f"rust_{th}", lambda th=th, optimized=optimized: optimized(th))
                for th in (1, 4, 0)
            ]:
                samples = []
                for _ in range(3):
                    start = perf_counter()
                    value = fn()
                    samples.append(perf_counter() - start)
                arrays = (
                    zip(ref, value, strict=True)
                    if isinstance(ref, tuple)
                    else [(ref, value)]
                )
                error = 0.0
                for a, z in arrays:
                    np.testing.assert_allclose(
                        a, z, rtol=1e-9, atol=1e-11, equal_nan=True
                    )
                    error = max(error, float(np.nanmax(np.abs(a - z))))
                results["seconds"][name][method] = {
                    "samples": samples,
                    "median": float(np.median(samples)),
                }
                results["max_errors"][name][method] = error
                path.write_text(json.dumps(results, indent=2))
                print(name, method, np.median(samples), "error", error, flush=True)
    return results


if __name__ == "__main__":
    run()
