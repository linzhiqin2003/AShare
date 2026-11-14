"""差分检查：GPU float32 算子对 panelops float64 逐项比较（缺失位置是否一致、数值误差、并列秩）。

    uv run --no-sync python bench/gpu/check_ops.py --backend mlx  [--synthetic | --root <数据根目录>]
    uv run --no-sync python bench/gpu/check_ops.py --backend torch --device cuda --synthetic

默认取面板的最后 ``--rows`` 行（真实数据只读）。输出只有误差统计，写到 results/check_ops_<后端>_<设备>_<数据>.json。

判据：缺失位置必须完全一致（只允许 float32 下溢、上溢或「常数窗口」判定引起的个别格子，逐项报告数目）；
数值误差用「相对误差 = |gpu − cpu| / max(|cpu|, 尺度)」的 99.9% 分位和最大值报告，尺度取该输出
有限值绝对值的中位数（避免接近 0 的值放大相对误差）。秩类输出（ts_rank、cs_rank）报告与 CPU 不同的格子比例。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import panelops

sys.path.insert(0, str(Path(__file__).parent))
import common  # noqa: E402
import gpu_ops  # noqa: E402

CASES = [
    ("ts_mean", 1, {"window": 5}), ("ts_mean", 1, {"window": 60}),
    ("ts_std", 1, {"window": 5}), ("ts_std", 1, {"window": 60}),
    ("ts_zscore", 1, {"window": 20}), ("ts_zscore", 1, {"window": 60}),
    ("ts_max", 1, {"window": 20}), ("ts_min", 1, {"window": 5}),
    ("ts_rank", 1, {"window": 5}), ("ts_rank", 1, {"window": 20}),
    ("ts_decay_linear", 1, {"window": 10}),
    ("ts_corr", 2, {"window": 10}), ("ts_corr", 2, {"window": 20}),
    ("ts_delta", 1, {"periods": 5}), ("ts_delay", 1, {"periods": 1}),
    ("cs_rank", 1, {}), ("cs_zscore", 1, {}),
    ("divide", 2, {}), ("log", 1, {}), ("multiply", 2, {}),
]  # fmt: skip
INPUTS = [("close", "volume"), ("turnover", "amount"), ("vwap", "close")]


def compare(gpu, cpu, ranks=False):
    g, c = np.asarray(gpu, np.float64), np.asarray(cpu, np.float64)
    gf, cf = np.isfinite(g), np.isfinite(c)
    both = gf & cf
    out = {"cells": int(c.size), "finite_cpu": int(cf.sum()),
           "missing_mismatch": int((gf != cf).sum()),
           "gpu_only_missing": int((cf & ~gf).sum()),
           "cpu_only_missing": int((gf & ~cf).sum())}  # fmt: skip
    if both.any():
        d = np.abs(g[both] - c[both])
        scale = float(np.median(np.abs(c[both]))) or 1.0
        rel = d / np.maximum(np.abs(c[both]), scale)
        out.update(rel_p999=float(np.quantile(rel, 0.999)), rel_max=float(rel.max()),
                   abs_max=float(d.max()))  # fmt: skip
        if ranks:
            out["rank_cells_differ"] = float((d > 1e-3).mean())
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    common.add_data_args(ap)
    ap.add_argument(
        "--backend", choices=["mlx", "mlx-fused", "torch", "torch-fused"], required=True
    )
    ap.add_argument("--device", default=None)
    ap.add_argument("--rows", type=int, default=400)
    args = ap.parse_args(argv)
    panelops.set_num_threads(args.threads)
    panel, source, _ = common.load_panel(args)
    xp = gpu_ops.backend(args.backend, args.device)
    rows = slice(-args.rows, None)
    report = {"source": source, "backend": args.backend, "device": xp.device,
              "rows": args.rows, "cases": []}  # fmt: skip
    for a_name, b_name in INPUTS:
        a64 = np.ascontiguousarray(np.asarray(panel[a_name], np.float64)[rows])
        b64 = np.ascontiguousarray(np.asarray(panel[b_name], np.float64)[rows])
        a, b = xp.asarray(a64), xp.asarray(b64)
        for op, arity, params in CASES:
            args_cpu = [a64, b64][:arity]
            args_gpu = [a, b][:arity]
            if op in ("divide", "log", "multiply"):
                cpu = panelops.elementwise(op, *args_cpu)
            else:
                cpu = getattr(panelops, op)(*args_cpu, **params)
            gpu = xp.numpy(gpu_ops.compute(xp, op, params, args_gpu))
            r = compare(gpu, cpu, ranks=op in ("ts_rank", "cs_rank"))
            r.update(op=op, params=params, inputs=[a_name, b_name][:arity])
            report["cases"].append(r)
            print(op, params, r["inputs"], {k: r.get(k) for k in
                  ("missing_mismatch", "rel_p999", "rel_max", "rank_cells_differ")})  # fmt: skip
        # Rank IC：用 close 的 5 日变化当因子、对 5 日后的收益（这里只要两份面板，语义检查）
        score = panelops.ts_delta(a64, periods=5)
        ret = np.ascontiguousarray(
            np.vstack([b64[5:] / b64[:-5] - 1, np.full((5, b64.shape[1]), np.nan)])
        )
        rank_ic = getattr(xp, "rank_ic_fused", None) or (
            lambda a, b: gpu_ops.rank_ic(xp, a, b)
        )
        # 第二组是并列很多的情形：因子取 5 日时序排名（只有 1..5 几个值），收益保留两位小数
        tied = (panelops.ts_rank(a64, window=5), np.round(ret, 2))
        for label, (sc, rt) in (("plain", (score, ret)), ("ties", tied)):
            sc, rt = np.ascontiguousarray(sc), np.ascontiguousarray(rt)
            c = panelops.rank_ic(sc, rt)
            g = np.asarray(
                xp.numpy(rank_ic(xp.asarray(sc), xp.asarray(rt))), np.float64
            )
            ok = np.isfinite(g) & np.isfinite(c)
            r = {"op": "rank_ic", "case": label, "inputs": [a_name, b_name],
                 "missing_mismatch": int((np.isfinite(g) != np.isfinite(c)).sum()),
                 "abs_max": float(np.abs(g[ok] - c[ok]).max()) if ok.any() else None}  # fmt: skip
            report["cases"].append(r)
            print(r)
    name = f"check_ops_{args.backend}_{xp.device}_{source['kind']}.json"
    common.write_json(common.RESULTS / name, report)
    xp.release()


if __name__ == "__main__":
    main()
