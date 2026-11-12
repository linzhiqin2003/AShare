"""一键运行全历史规模散布/透视对比；每种方法在独立进程量峰值 RSS。"""

import argparse
import json
import resource
import subprocess
import sys
import time

import numpy as np
import polars as pl


def trial(method):
    t, n = 4800, 5900
    rows = np.repeat(np.arange(t, dtype=np.int32), n)
    cols = np.tile(np.arange(n, dtype=np.int32), t)
    values = np.arange(t * n, dtype=np.float64) % 1000 / 100 + 10
    start = time.perf_counter()
    if method == "numpy":
        result = np.full((t, n), np.nan)
        result[rows, cols] = values
    else:
        table = pl.DataFrame(dict(row=rows, col=cols, value=values))
        result = (
            table.pivot(on="col", index="row", values="value", maintain_order=True)
            .drop("row")
            .to_numpy(order="c")
        )
    elapsed = time.perf_counter() - start
    assert result.shape == (t, n) and result.flags.c_contiguous
    assert result[4799, 5899] == values[-1]
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return dict(
        method=method,
        t=t,
        n=n,
        seconds=elapsed,
        peak_rss_mib=peak / (1024**2 if sys.platform == "darwin" else 1024),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["numpy", "polars"])
    args = parser.parse_args()
    if args.method:
        print(json.dumps(trial(args.method)))
    else:
        results = []
        for method in ("numpy", "polars"):
            result = subprocess.run(
                [sys.executable, __file__, "--method", method],
                check=True,
                text=True,
                capture_output=True,
            )
            results.append(json.loads(result.stdout))
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
