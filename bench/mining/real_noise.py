"""真实收益上的噪声检验：与未来收益无关的因子，折算后会不会被判显著。

在真实 A 股面板（只读）上构造四类与未来收益无关的「因子」，每次 N 个候选全部跑完整评估
（Rank IC + 多头端回测，默认基准与费用），再按搜索里同样的口径折算：

- redraw：每天每只股票重新抽一个随机数；
- ar：每只股票的分数按 AR(1) φ = 0.98 演化（持续性接近常见的慢变量因子）；
- fixed：每只股票一个固定随机数，整段不变；
- permuted：真实的 20 日收益（复权收盘价算），每次把股票列随机置换（保留时间结构，
  切断和本股票未来收益的联系）。

每次记录：DSR（V 取零假设、按序列相关修正）大于 1 − α 的个数、不修正序列相关的个数、
V 取经验方差的个数、PSR 单个拒绝率、经验夏普方差 ÷ 1/(T−1)、方差比的分布、训练 IC 的
Holm / BH 拒绝个数。只写汇总，不含任何个股或行情数值。

    uv run --no-sync python bench/mining/real_noise.py --root <数据根目录> \\
        --out bench/mining/results/real_noise.json
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np

KINDS = ("redraw", "ar", "fixed", "permuted")


def _load():
    out = subprocess.run(["uptime"], capture_output=True, text=True).stdout
    return out.split("load averages:")[-1].strip()


def _factors(kind, rng, shape, feature, count):
    t, n = shape
    for k in range(count):
        if kind == "redraw":
            x = rng.standard_normal(shape)
        elif kind == "fixed":
            x = np.repeat(rng.standard_normal((1, n)), t, axis=0)
        elif kind == "ar":
            x = np.empty(shape)
            x[0] = rng.standard_normal(n)
            s = math.sqrt(1 - 0.98**2)
            for i in range(1, t):
                x[i] = 0.98 * x[i - 1] + s * rng.standard_normal(n)
        else:
            x = np.ascontiguousarray(feature[:, rng.permutation(n)])
        yield f"{kind}{k}", x


def run(fe, kind, seed, n_trials, alpha, feature, batch):
    from ashare.factor import stats

    rng = np.random.default_rng(seed)
    rows = []
    gen = _factors(kind, rng, fe.shape, feature, n_trials)
    while True:
        chunk = dict(next(gen, (None, None)) for _ in range(batch))
        chunk.pop(None, None)
        if not chunk:
            break
        for rep in fe.evaluate_many(chunk).values():
            m = rep.segments["all"]
            lo = m["long"]
            rows.append({"p": m["ic"][5]["p"], "sr": lo["sr_period"], "t": lo["days"],
                         "g3": lo["skew"] or 0.0, "g4": lo["kurt"] or 3.0,
                         "q": lo["var_ratio"] or 1.0})  # fmt: skip
        del chunk
    t = max(r["t"] for r in rows)
    v_null = 1.0 / (t - 1)
    srs = [r["sr"] for r in rows if r["sr"] is not None]
    v_emp = stats.trial_sharpe_variance(srs, floor=False)
    out = {
        "dsr_null_q": 0,
        "dsr_null_iid": 0,
        "dsr_empirical": 0,
        "psr_q": 0,
        "psr_iid": 0,
    }
    for r in rows:
        if r["sr"] is None:
            continue
        a = (r["sr"], r["t"], r["g3"], r["g4"])
        out["dsr_null_q"] += stats.deflated_sharpe(*a, n_trials=n_trials,
                                                   sr_var=v_null * r["q"], q=r["q"]) > 1 - alpha  # fmt: skip
        out["dsr_null_iid"] += stats.deflated_sharpe(*a, n_trials=n_trials,
                                                     sr_var=v_null) > 1 - alpha  # fmt: skip
        out["dsr_empirical"] += stats.deflated_sharpe(*a, n_trials=n_trials,
                                                      sr_var=v_emp) > 1 - alpha  # fmt: skip
        out["psr_q"] += stats.probabilistic_sharpe(*a, q=r["q"]) > 1 - alpha
        out["psr_iid"] += stats.probabilistic_sharpe(*a) > 1 - alpha
    out = {k: int(v) for k, v in out.items()}
    p = [r["p"] if r["p"] is not None else math.nan for r in rows]
    out.update({
        "n": len(rows),
        "psr_q_rate": out["psr_q"] / len(rows),
        "psr_iid_rate": out["psr_iid"] / len(rows),
        "v_empirical_over_null": v_emp / v_null,
        "q_quantiles": [float(x) for x in np.quantile([r["q"] for r in rows], [0.1, 0.5, 0.9])],
        "mean_sr_annual": float(np.mean(srs) * math.sqrt(252)),
        "ic_raw": int(sum(1 for x in p if x <= alpha)),
        "ic_holm": int(stats.reject(p, alpha, "holm").sum()),
        "ic_bh": int(stats.reject(p, alpha, "bh").sum()),
    })  # fmt: skip
    return out


def main(argv=None):
    import ashare.data as ad
    from ashare.backtest import market_from_panel
    from ashare.factor.evaluate import FactorEvaluator

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", default="20160101")
    ap.add_argument("--end", default="20211231")
    ap.add_argument("--trials", type=int, default=1000)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--kinds", default=",".join(KINDS))
    args = ap.parse_args(argv)
    started = time.time()
    data = ad.open(root=args.root)
    panel = data.panel(
        list(FactorEvaluator.required_fields()), start=args.start, end=args.end
    )
    market = market_from_panel(data, panel=panel, validate=False)
    fe = FactorEvaluator(panel, horizons=(5,), market=market, size_buckets=1)
    price = np.asarray(panel["close"], dtype=np.float64) * np.asarray(
        panel["adj_factor"]
    )
    feature = np.full(price.shape, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        feature[20:] = price[20:] / price[:-20] - 1
    result = {
        "settings": {"start": args.start[:4], "end": args.end[:4], "trials": args.trials,
                     "reps": args.reps, "top_fraction": 0.1, "rebalance": 5,
                     "benchmark": "universe", "costs": "A 股默认", "alpha": args.alpha},
        "load_before": _load(), "runs": {},
    }  # fmt: skip
    for kind in args.kinds.split(","):
        for r in range(args.reps):
            t0 = time.time()
            res = run(fe, kind, 1000 * r + KINDS.index(kind), args.trials, args.alpha,
                      feature, args.batch)  # fmt: skip
            res["seconds"] = time.time() - t0
            result["runs"][f"{kind}/{r}"] = res
            print(kind, r, json.dumps(res, ensure_ascii=False), flush=True)
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1))
    result["seconds"] = time.time() - started
    result["load_after"] = _load()
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
