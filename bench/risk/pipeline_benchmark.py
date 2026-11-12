"""完整特质估计、描述子构建和统计 PCA 的补充基准。"""

import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
from baseline import metadata
from threadpoolctl import threadpool_limits

import ashare.risk.fundamental as fundamental
from ashare.risk.kernels import ew_variance
from ashare.risk.statistical import statistical_model


class Source:
    def __init__(self, t, n, rng):
        self.dates = np.arange(t)
        self.symbols = tuple(map(str, range(n)))
        self.version = "benchmark"
        base = {
            "total_mv": np.exp(rng.normal(20, 1, n)),
            "industry": np.arange(n) % 31,
            "is_listed": np.ones(n, bool),
            "is_suspended": np.zeros(n, bool),
            "turnover_rate": rng.uniform(0.001, 0.03, n),
            "pb": rng.uniform(0.5, 5, n),
            "pe_ttm": rng.uniform(5, 50, n),
        }
        self.fields = {
            name: np.broadcast_to(value, (t, n)) for name, value in base.items()
        }

    def __getitem__(self, key):
        return self.fields[key]


def run(output=None):
    rng = np.random.default_rng(992)
    t, n, k = 4000, 5500, 45
    source = Source(t, n, rng)
    b = rng.normal(size=(n, k))
    r = rng.normal(0, 0.002, (t, k)) @ b.T + rng.normal(0, 0.015, (t, n))
    result = {
        "metadata": metadata(),
        "shape": [t, n, k],
        "seconds": {},
        "max_errors": {},
    }
    config = fundamental.FundamentalConfig()
    begin = perf_counter()
    ages = np.cumsum(source["is_listed"], axis=0, dtype=np.int32)
    result["listing_prefix_seconds"] = perf_counter() - begin
    with threadpool_limits(limits=1):
        for name, fn in (
            (
                "specific_complete",
                lambda: fundamental.specific_variance(
                    r, b, source["total_mv"][-1], config
                )[0],
            ),
            (
                "exposures",
                lambda: fundamental.exposures_at(source, r, t - 1, config)[0],
            ),
            (
                "exposures_precomputed_age",
                lambda: fundamental.exposures_at(
                    source, r, t - 1, config, listing_ages=ages
                )[0],
            ),
        ):
            variants = {}
            arrays = {}
            for backend in ("numpy", "rust"):
                variance = (
                    ew_variance if backend == "numpy" else fundamental.ew_variance
                )
                with patch.object(fundamental, "ew_variance", variance):
                    times = []
                    for _ in range(3):
                        begin = perf_counter()
                        value = fn()
                        times.append(perf_counter() - begin)
                    arrays[backend] = value
                variants[backend] = {
                    "samples": times,
                    "median": float(np.median(times)),
                }
            np.testing.assert_allclose(
                arrays["numpy"], arrays["rust"], atol=1e-10, rtol=1e-9
            )
            result["seconds"][name] = variants
            result["max_errors"][name] = float(
                np.max(np.abs(arrays["numpy"] - arrays["rust"]))
            )
            print(name, variants, flush=True)
        begin = perf_counter()
        model = statistical_model(r, components=45)
        result["statistical_pca_lw_seconds"] = perf_counter() - begin
        result["statistical_shrinkage"] = model.metadata["shrinkage"]
        print("PCA+LW", result["statistical_pca_lw_seconds"], flush=True)
    path = Path(output) if output else Path(".cache/risk-bench") / "pipeline.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run()
