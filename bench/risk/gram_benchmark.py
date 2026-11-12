"""避免只拿 SVD 与 Rust 正规方程比较：补测 BLAS Gram 和 Python 线程池。"""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
import portopt
from baseline import metadata
from threadpoolctl import threadpool_limits

from ashare.risk.kernels import cross_section_gram


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
    industries = np.arange(1, 32)

    def day(i):
        return cross_section_gram(r[i], b, cap[i], est[i], industries)

    def evaluate(threads):
        if threads == 1:
            rows = [day(i) for i in range(t)]
        else:
            with ThreadPoolExecutor(threads) as pool:
                rows = list(pool.map(day, range(t)))
        return np.array([row[0] for row in rows]), np.array([row[1] for row in rows])

    result = {"metadata": metadata(), "shape": [t, n, k], "seconds": {}}
    with threadpool_limits(limits=1):
        reference = portopt.regress(r, b, cap, est, industries)
        for threads in (1, 4, 15):
            samples = []
            for _ in range(3):
                begin = perf_counter()
                value = evaluate(threads)
                samples.append(perf_counter() - begin)
            for a, z in zip(reference, value, strict=True):
                np.testing.assert_allclose(a, z, rtol=1e-9, atol=1e-11, equal_nan=True)
            result["seconds"][f"numpy_gram_{threads}"] = {
                "median": float(np.median(samples)),
                "samples": samples,
            }
            print(threads, result["seconds"][f"numpy_gram_{threads}"], flush=True)
    path = Path(output) if output else Path(".cache/risk-bench") / "gram.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run()
