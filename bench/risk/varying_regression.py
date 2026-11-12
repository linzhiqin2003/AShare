"""以小块生成真实变化的 T×N×K 暴露，避免同时持有约 8 GB 的面板。"""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
import portopt
from baseline import metadata
from threadpoolctl import threadpool_limits

from ashare.risk.kernels import cross_section_batch, cross_section_gram


def run(output=None):
    rng = np.random.default_rng(113)
    t, n, k = 4000, 5500, 45
    base = np.column_stack(
        (np.ones(n), np.eye(31)[np.arange(n) % 31], rng.normal(size=(n, 13)))
    )
    r = rng.normal(0, 0.01, (t, n))
    cap = np.exp(rng.normal(10, 1, (t, n)))
    est = rng.random((t, n)) > 0.02

    def evaluate(backend):
        fs, us = [], []
        for start in range(0, t, 64):
            stop = min(t, start + 64)
            days = np.arange(start, stop)
            b = np.broadcast_to(base, (stop - start, n, k)).copy()
            b[:, :, 32:] *= 1 + 0.05 * np.sin(
                days[:, None, None] * 0.1 + np.arange(n)[None, :, None] * 0.01
            )
            b[:, :, 1:32] = np.eye(31)[
                (np.arange(n)[None, :] + days[:, None] // 252) % 31
            ]
            if backend == "numpy_gram_4":

                def one_day(local, start=start, b=b):
                    day = start + local
                    return cross_section_gram(
                        r[day], b[local], cap[day], est[day], np.arange(1, 32)
                    )

                with ThreadPoolExecutor(4) as pool:
                    rows = list(pool.map(one_day, range(stop - start)))
                f, u = np.array([v[0] for v in rows]), np.array([v[1] for v in rows])
            else:
                fn = cross_section_batch if backend == "numpy" else portopt.regress
                f, u = fn(
                    r[start:stop], b, cap[start:stop], est[start:stop], np.arange(1, 32)
                )
            fs.append(f)
            us.append(u)
        return np.concatenate(fs), np.concatenate(us)

    result = {
        "metadata": metadata(),
        "shape": [t, n, k],
        "chunk_days": 64,
        "includes_exposure_generation": True,
        "seconds": {},
    }
    values = {}
    with threadpool_limits(limits=1):
        for backend in ("numpy", "numpy_gram_4", "rust"):
            samples = []
            for _ in range(3):
                begin = perf_counter()
                values[backend] = evaluate(backend)
                samples.append(perf_counter() - begin)
            result["seconds"][backend] = {
                "samples": samples,
                "median": float(np.median(samples)),
            }
            print(backend, result["seconds"][backend], flush=True)
    error = 0.0
    for backend in ("numpy_gram_4", "rust"):
        for a, b in zip(values["numpy"], values[backend], strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-11)
            error = max(error, float(np.max(np.abs(a - b))))
    result["max_error"] = error
    path = (
        Path(output)
        if output
        else Path(".cache/risk-bench") / "varying-regression.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run()
