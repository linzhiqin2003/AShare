"""对 BLAS 已经很快的 K 维协方差环节也实际测量 Rust 候选。"""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import portopt
from baseline import metadata
from threadpoolctl import threadpool_limits

from ashare.risk.fundamental import FundamentalConfig, specific_variance
from ashare.risk.kernels import ew_covariance, factor_covariance, psd


def rust_factor(x, threads):
    corr = portopt.ew_covariance(x, 504, 2, threads)
    sd = np.sqrt(np.maximum(np.diag(corr), 1e-30))
    corr /= np.outer(sd, sd)
    vol = np.sqrt(portopt.ew_variance(x, 84, 5, threads))
    return psd(corr * np.outer(vol, vol))


def run(output=None):
    rng = np.random.default_rng(442)
    x = rng.normal(0, 0.01, (4000, 45))
    result = {
        "metadata": metadata(),
        "shape": [4000, 5500, 45],
        "repeats": 20,
        "seconds": {},
    }
    with threadpool_limits(limits=1):
        for name, ref, rust in (
            (
                "ew_covariance",
                lambda: ew_covariance(x, 504),
                lambda th: portopt.ew_covariance(x, 504, 0, th),
            ),
            ("newey_west", lambda: factor_covariance(x), lambda th: rust_factor(x, th)),
        ):
            reference = ref()
            result["seconds"][name] = {}
            for method, fn in [("numpy", ref)] + [
                (f"rust_{th}", lambda th=th, func=rust: func(th)) for th in (1, 4, 0)
            ]:
                samples = []
                for _ in range(20):
                    begin = perf_counter()
                    value = fn()
                    samples.append(perf_counter() - begin)
                np.testing.assert_allclose(value, reference, rtol=1e-11, atol=1e-14)
                result["seconds"][name][method] = {
                    "median": float(np.median(samples)),
                    "samples": samples,
                }
                print(name, method, np.median(samples), flush=True)
        # 特质模型完整阶段（含结构化回归和贝叶斯收缩），不只报 NW 内核。
        r = rng.normal(0, 0.02, (4000, 5500))
        r[:3900, :100] = np.nan
        b = np.column_stack((np.ones(5500), rng.normal(size=(5500, 44))))
        cap = np.exp(rng.normal(20, 1, 5500))
        config = FundamentalConfig()
        begin = perf_counter()
        specific_variance(r, b, cap, config)
        elapsed = perf_counter() - begin
        result["specific_complete_seconds"] = elapsed
        print("specific complete", elapsed, flush=True)
    output = (
        Path(output) if output else Path(".cache/risk-bench") / "factor-kernels.json"
    )
    output.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run()
