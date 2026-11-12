"""补测默认 1008 日及短窗口，避免把 4000 日的线程收益机械推广。"""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import portopt
from baseline import metadata
from factor_kernels import rust_factor
from threadpoolctl import threadpool_limits

from ashare.risk.kernels import ew_covariance, ew_variance, factor_covariance


def run(output=None):
    rng = np.random.default_rng(614)
    result = {"metadata": metadata(), "seconds": {}}
    with threadpool_limits(limits=1):
        for t in (252, 1008, 4000):
            x = rng.normal(0, 0.01, (t, 45))
            for name, numpy_fn, rust_fn in (
                (
                    "factor_variance",
                    lambda x=x: ew_variance(x, 84),
                    lambda x=x: portopt.ew_variance(x, 84),
                ),
                (
                    "covariance",
                    lambda x=x: ew_covariance(x, 504),
                    lambda x=x: portopt.ew_covariance(x, 504),
                ),
                (
                    "factor_complete",
                    lambda x=x: factor_covariance(x),
                    lambda x=x: rust_factor(x, 0),
                ),
            ):
                variants, arrays = {}, {}
                for method, fn in (("numpy", numpy_fn), ("rust", rust_fn)):
                    samples = []
                    for _ in range(20):
                        begin = perf_counter()
                        value = fn()
                        samples.append(perf_counter() - begin)
                    arrays[method] = value
                    variants[method] = {
                        "median": float(np.median(samples)),
                        "samples": samples,
                    }
                np.testing.assert_allclose(
                    arrays["numpy"], arrays["rust"], rtol=1e-11, atol=1e-14
                )
                result["seconds"][f"{t}/{name}"] = variants
                print(
                    t, name, {k: v["median"] for k, v in variants.items()}, flush=True
                )
    path = Path(output) if output else Path(".cache/risk-bench") / "windows.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run()
