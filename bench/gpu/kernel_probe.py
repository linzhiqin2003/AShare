"""附带一项：滑动均值写成单个 Metal 自定义内核（mx.fast.metal_kernel）与「W 个切片逐项相加」的对比。

    nice -n 15 uv run --no-sync python bench/gpu/kernel_probe.py

只用随机数据（形状 2430 × 5923，含 3% 缺失），只记耗时与两种写法的最大差。
正式实现若要把滑动窗口一族压成一趟读写，这是 Metal 上的做法；CUDA 上对应 mx.fast.cuda_kernel 或 Triton。
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402
import gpu_ops  # noqa: E402

SOURCE = """
    uint col = thread_position_in_grid.x;
    if (col >= N) return;
    for (int t = 0; t < T; ++t) {
        float acc = NAN;
        if (t >= W - 1) {
            acc = 0.0f;
            for (int k = 0; k < W; ++k) acc += x[(t - k) * N + col];
        }
        out[t * N + col] = isfinite(acc) ? acc / W : NAN;
    }
"""


def main():
    import mlx.core as mx

    xp = gpu_ops.backend("mlx")
    rng = np.random.default_rng(0)
    T, N = 2430, 5923
    a = rng.lognormal(2, 0.5, (T, N)).astype(np.float32)
    a[rng.random((T, N)) < 0.03] = np.nan
    x = xp.asarray(a)
    kernel = mx.fast.metal_kernel(name="rolling_mean", input_names=["x"],
                                  output_names=["out"], source=SOURCE)  # fmt: skip
    out = {"shape": [T, N], "load_before": common.load_avg(), "windows": {}}
    for w in (5, 20, 60):

        def fused(w=w):
            return kernel(inputs=[x], template=[("W", w), ("T", T), ("N", N)],
                          grid=(N, 1, 1), threadgroup=(256, 1, 1),
                          output_shapes=[(T, N)], output_dtypes=[mx.float32])[0]  # fmt: skip

        def sliced(w=w):
            return gpu_ops.ts(xp, "mean", x, w)

        times = {"fused": [], "sliced": []}
        for _ in range(7):
            for name, f in (("fused", fused), ("sliced", sliced)):
                t0 = time.perf_counter()
                r = f()
                mx.eval(r)
                times[name].append(time.perf_counter() - t0)
        diff = np.nanmax(np.abs(np.array(fused()) - np.array(sliced())))
        out["windows"][w] = {
            k: statistics.median(v[2:]) * 1000 for k, v in times.items()
        }
        out["windows"][w]["max_abs_diff"] = float(diff)
        print(w, out["windows"][w], flush=True)
    out["load_after"] = common.load_avg()
    common.write_json(common.RESULTS / "kernel_probe_mlx_metal.json", out)


if __name__ == "__main__":
    main()
