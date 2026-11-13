"""模拟检验：重叠持有期的日 Rank IC 在零假设下，Holm 与 BH 的族错误率是否守住名义水平。

网格：持有期 h ∈ {1, 5, 20} × 样本天数 n ∈ {250, 750, 1500}。每格重复 --reps 次，每次
m 个互相独立的零假设因子，各自对同一份收益面板算日 Rank IC，再用 `ic_summary` 的 p 值做
Holm 和 BH（BH 在全零假设下的错误发现率就是族错误率）。因子两种持续性：

- fixed：每只股票一个固定分数（因子完全不变，重叠最严重，零假设下日 IC 的自相关最高）；
- ar：每只股票的分数按 AR(1) φ = 0.9 演化（更像常见的量价因子）。

收益：每只股票日收益独立同分布，h 日持有收益 = 未来 h 天之和（与真实标签同样重叠）。
不重叠的块少于 10 个的格子不给 p 值（`ic_summary` 的规则），这里记为「不给 p 值」。

用法：uv run --no-sync python bench/mining/ic_grid.py --out bench/mining/results/ic_grid.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import panelops

from ashare.factor import stats
from ashare.factor.evaluate import ic_summary


def _scores(rng, kind, t, n, m):
    if kind == "fixed":
        return [np.repeat(rng.standard_normal((1, n)), t, axis=0) for _ in range(m)]
    out = []
    for _ in range(m):
        x = np.empty((t, n))
        x[0] = rng.standard_normal(n)
        z = rng.standard_normal((t, n)) * math.sqrt(1 - 0.9**2)
        for i in range(1, t):
            x[i] = 0.9 * x[i - 1] + z[i]
        out.append(x)
    return out


def cell(h, n_days, kind, reps, m, stocks, alpha, seed):
    rng = np.random.default_rng(seed)
    hits = {"holm": 0, "bh": 0, "raw_any": 0}
    no_p = 0
    for _ in range(reps):
        daily = rng.standard_normal((n_days + h, stocks)) * 0.02
        fwd = np.lib.stride_tricks.sliding_window_view(daily, h, axis=0).sum(-1)
        fwd = np.ascontiguousarray(fwd[1 : n_days + 1])  # 信号日 d 之后的 h 天
        p = []
        for x in _scores(rng, kind, n_days, stocks, m):
            s = ic_summary(panelops.rank_ic(np.ascontiguousarray(x), fwd), h)
            p.append(s["p"])
        if all(math.isnan(v) for v in p):
            no_p += 1
            continue
        hits["holm"] += bool(stats.reject(p, alpha, "holm").any())
        hits["bh"] += bool(stats.reject(p, alpha, "bh").any())
        hits["raw_any"] += any(v <= alpha for v in p)
    done = reps - no_p
    out = {"reps": reps, "no_p_value": no_p}
    if done:
        se = math.sqrt(alpha * (1 - alpha) / done)
        out.update({k: v / done for k, v in hits.items()})
        out["upper_3se"] = alpha + 3 * se
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--m", type=int, default=20)
    ap.add_argument("--stocks", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args(argv)
    started = time.time()
    result = {"setup": vars(args) | {"out": None}, "cells": {}}
    for kind in ("fixed", "ar"):
        for h in (1, 5, 20):
            for n_days in (250, 750, 1500):
                key = f"{kind} h={h} n={n_days}"
                result["cells"][key] = cell(h, n_days, kind, args.reps, args.m, args.stocks,
                                            args.alpha, 1000 * h + n_days + (7 if kind == "ar" else 0))  # fmt: skip
                print(f"[{time.time() - started:7.1f}s] {key} {result['cells'][key]}",
                      flush=True)  # fmt: skip
    result["seconds"] = time.time() - started
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
