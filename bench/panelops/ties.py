"""Benchmark the ties="earliest" option of ts_argmax/ts_idxmax against a NumPy baseline.

    uv run --no-sync python bench/panelops/ties.py [--rows 4000] [--cols 5500] [--repeat 3]

Representative configuration of docs/PANELOPS_BENCHMARKS.md: W=20, min_periods=10,
5% missing, plus a heavily tied integer panel (values 0..4) that exercises the tie path.
Runs alternate between implementations; the median of `repeat` timings is reported.
"""

import argparse
import os
import statistics
import time

import numpy as np
import panelops as po


def numpy_earliest(x, window, min_periods, op):
    """Plain NumPy: sliding windows, first tied extremum, padded leading rows."""
    t, n = x.shape
    valid = np.isfinite(x)
    fill = -np.inf if op == "idxmax" else np.inf
    padded = np.vstack([np.full((window - 1, n), fill), np.where(valid, x, fill)])
    counts = np.vstack([np.zeros((window - 1, n)), valid.astype(float)])
    out = np.empty((t, n))
    step = 256  # bound the temporary window copy
    for start in range(0, t, step):
        stop = min(t, start + step)
        view = np.lib.stride_tricks.sliding_window_view(
            padded[start : stop + window - 1], window, axis=0
        )
        pos = view.argmax(axis=-1) if op == "idxmax" else view.argmin(axis=-1)
        count = np.lib.stride_tricks.sliding_window_view(
            counts[start : stop + window - 1], window, axis=0
        ).sum(-1)
        out[start:stop] = np.where(count >= min_periods, pos + 1, np.nan)
    return out


def timed(fn, repeat):
    fn()  # warm-up
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        times.append(1000 * (time.perf_counter() - start))
    return statistics.median(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=4000)
    parser.add_argument("--cols", type=int, default=5500)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    rng = np.random.default_rng(0)
    panels = {
        "normal": rng.normal(size=(args.rows, args.cols)),
        "tied 0..4": rng.integers(0, 5, (args.rows, args.cols)).astype(float),
    }
    for x in panels.values():
        x[rng.uniform(size=x.shape) < 0.05] = np.nan
    w, mp = 20, 10
    print(f"T={args.rows} N={args.cols} W={w} min_periods={mp}; load {os.getloadavg()}")
    for label, x in panels.items():
        corner = np.ascontiguousarray(x[:300, :300])
        check = numpy_earliest(corner, w, mp, "idxmax")
        np.testing.assert_array_equal(
            check, po.ts_idxmax(corner, w, min_periods=mp, ties="earliest")
        )
        for op in ("idxmax", "argmax"):
            fn = getattr(po, f"ts_{op}")
            row = {}
            for threads in (1, 0):
                po.set_num_threads(threads)
                tag = "1" if threads else "default"
                row[f"rust latest {tag}"] = timed(
                    lambda fn=fn, x=x: fn(x, w, min_periods=mp), args.repeat
                )
                row[f"rust earliest {tag}"] = timed(
                    lambda fn=fn, x=x: fn(x, w, min_periods=mp, ties="earliest"),
                    args.repeat,
                )
            po.set_num_threads(0)
            if op == "idxmax":
                row["numpy earliest 1"] = timed(
                    lambda x=x: numpy_earliest(x, w, mp, "idxmax"), args.repeat
                )
            print(label, op, {k: round(v, 1) for k, v in row.items()})
    print(f"load after {os.getloadavg()}")


if __name__ == "__main__":
    main()
