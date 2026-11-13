"""模拟检验：纯噪声上的完整搜索（折算后「显著」的比例不应超过名义水平）与埋信号的找回率。

用法（规模可调，默认约十分钟；结果只含汇总比例，写到 --out）：

    uv run --no-sync python bench/mining/simulate.py --out bench/mining/results/simulation.json

纯噪声：合成行情里 t+1 开盘以后的收益与 t 日及以前的一切独立，任何量价因子都没有预测力。
默认带摩擦（`Frictions.typical()`：停牌、退市、ST、中途上市），`--no-frictions` 关掉。
每次重复用不同种子的行情、在默认空间上随机采样 `--candidates` 个候选跑完整搜索，记录
「这次搜索有没有至少一个候选被判显著」——这是族错误率（FWER）。分两种成本：A 股默认
费用（多头端超额被成本拉低，检验天然偏保守）和零成本（对低换手的候选零假设近似成立；高换手
的候选即使零成本也有涨跌停买卖不了带来的执行损失，超额偏负，方向仍偏保守）。

埋信号：在合成行情里埋一个已知强度的短期反转，在一个包含它的小空间上穷举搜索，记录
埋入信号（或数值上等价的写法，指纹相关 > 0.95）是否被回测、是否在折算后仍显著、
验证段第一名是否与它高度相关（> 0.6）。
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import panelbt

from ashare.factor import Evaluator
from ashare.factor.mining import Enumerate, Random, Space, Split, mine
from ashare.factor.mining.search import fingerprint, fingerprint_corr
from ashare.factor.mining.synthetic import Frictions, market, planted_expression

ZERO = panelbt.Costs(
    commission_rate=0.0,
    commission_min=None,
    fee_rate_buy=0.0,
    fee_rate_sell=0.0,
    slippage=0.0,
)
TINY = Space(
    fields={"close": "price", "open": "price", "volume": "volume"},
    ops={
        "divide": {},
        "negate": {},
        "ts_delay": {"periods": (1, 5)},
        "ts_mean": {"window": (5, 20)},
    },
    max_depth=3,
    max_nodes=5,
)


def _flags(res, alpha):
    s = res.summary["significant"]
    cands = [
        r
        for r in res.candidates
        if r["stage"] == "backtest" and r["duplicate_of"] is None
    ]
    emp = []
    # 同一次搜索也按 Bailey 原文的经验方差算一遍（V = 各试验夏普的截面方差）
    from ashare.factor import stats

    srs = [
        r["train"]["long"]["sr_period"]
        for r in res.candidates
        if r["stage"] == "backtest"
    ]
    v = stats.trial_sharpe_variance(srs, floor=False)
    for r in cands:
        lo = r["train"]["long"]
        if lo["sr_period"] is not None and math.isfinite(v):
            emp.append(stats.deflated_sharpe(lo["sr_period"], lo["days"], lo["skew"] or 0,
                                             lo["kurt"] or 3, n_trials=res.summary["trials_total"],
                                             sr_var=v))  # fmt: skip
    return {
        "ic_raw_any": s["ic_raw"] > 0,
        "ic_raw_count": s["ic_raw"],
        "ic_holm": s["ic_holm"] > 0,
        "ic_bh": s["ic_bh"] > 0,
        "ic_by": s["ic_by"] > 0,
        "psr_any": s["long_psr"] > 0,
        "psr_count": s["long_psr"],
        "dsr_null": s["long_dsr_null"] > 0,
        "dsr_empirical": any(x > 1 - alpha for x in emp),
        "train_and_valid": s["valid_dsr"] > 0,
        "unique": res.summary["unique"],
    }


FRICTIONS = Frictions.typical()


def noise(reps, candidates, costs, alpha, t, n, seed0, log):
    rows = []
    for k in range(reps):
        root = Path(tempfile.mkdtemp(prefix="mining-noise-"))
        try:
            p = market(t=t, n=n, seed=seed0 + k, frictions=FRICTIONS)
            split = Split.recent(p.dates, holdout_years=1, valid_years=1)
            res = mine(p, split=split, root=root, strategy=Random(n=candidates, seed=k),
                       costs=costs, alpha=alpha)  # fmt: skip
            rows.append(_flags(res, alpha))
        finally:
            shutil.rmtree(root, ignore_errors=True)
        log(f"噪声 {k + 1}/{reps}")
    return _aggregate(rows, reps, alpha)


def _aggregate(rows, reps, alpha):
    out = {"reps": reps, "alpha": alpha}
    for key in ("ic_raw_any", "ic_holm", "ic_bh", "ic_by", "psr_any", "dsr_null", "dsr_empirical",
                "train_and_valid"):  # fmt: skip
        hits = sum(r[key] for r in rows)
        out[key] = {"hits": hits, "rate": hits / reps}
    total = sum(r["unique"] for r in rows)
    out["per_candidate_ic_raw_rate"] = sum(r["ic_raw_count"] for r in rows) / max(
        total, 1
    )
    out["per_candidate_psr_rate"] = sum(r["psr_count"] for r in rows) / max(total, 1)
    out["mean_candidates"] = total / max(reps, 1)
    out["binomial_upper_3sd"] = alpha + 3 * math.sqrt(alpha * (1 - alpha) / reps)
    return out


def planted(reps, strengths, alpha, t, n, seed0, log):
    out = {}
    for strength in strengths:
        found = {
            "backtested": 0,
            "dsr_null": 0,
            "dsr_empirical": 0,
            "ic_holm": 0,
            "top_corr": 0,
        }
        for k in range(reps):
            root = Path(tempfile.mkdtemp(prefix="mining-plant-"))
            try:
                p = market(t=t, n=n, seed=seed0 + k, strength=strength,
                           frictions=FRICTIONS)  # fmt: skip
                split = Split.recent(p.dates, holdout_years=1, valid_years=1)
                res = mine(p, split=split, space=TINY, root=root, strategy=Enumerate(limit=5000),
                           alpha=alpha)  # fmt: skip
                a, v, h = split.rows(p.dates)
                rows = np.unique(
                    np.linspace(a + (v - a) // 3, v - 1, 16).astype(np.int64)
                )
                target = fingerprint(
                    Evaluator().evaluate(planted_expression(), p), rows
                )

                def corr(r, root=root, target=target):
                    path = root / "fingerprints" / f"{r['trial_id']}.npy"
                    return (
                        fingerprint_corr(target, np.load(path))
                        if path.exists()
                        else -1.0
                    )

                twins = [
                    r
                    for r in res.candidates
                    if r["stage"] == "backtest" and corr(r) > 0.95
                ]
                flags = _flags(res, alpha)
                found["backtested"] += bool(twins)
                found["dsr_null"] += any(
                    (r["dsr_train_null"] or 0) > 1 - alpha for r in twins
                )
                found["ic_holm"] += flags["ic_holm"]
                found["dsr_empirical"] += flags["dsr_empirical"]
                found["top_corr"] += bool(res.selected) and corr(res.selected[0]) > 0.6
            finally:
                shutil.rmtree(root, ignore_errors=True)
            log(f"埋信号 strength={strength} {k + 1}/{reps}")
        out[str(strength)] = {
            k: {"hits": v, "rate": v / reps} for k, v in found.items()
        }
        out[str(strength)]["reps"] = reps
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--noise-reps", type=int, default=60)
    ap.add_argument("--candidates", type=int, default=80)
    ap.add_argument("--plant-reps", type=int, default=8)
    ap.add_argument("--strengths", default="0.001,0.002,0.003")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--t", type=int, default=760)
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--no-frictions", action="store_true")
    args = ap.parse_args(argv)
    global FRICTIONS
    if args.no_frictions:
        FRICTIONS = None
    started = time.time()

    def log(msg):
        print(f"[{time.time() - started:7.1f}s] {msg}", flush=True)

    result = {
        "setup": {
            "t": args.t,
            "n": args.n,
            "candidates": args.candidates,
            "alpha": args.alpha,
            "split": "最后 1 年留出、之前 1 年验证、其余训练",
            "frictions": None if FRICTIONS is None else FRICTIONS.__dict__,
        },  # fmt: skip
        "noise_default_costs": noise(
            args.noise_reps,
            args.candidates,
            None,
            args.alpha,
            args.t,
            args.n,
            1000,
            log,
        ),  # fmt: skip
        "noise_zero_costs": noise(
            args.noise_reps,
            args.candidates,
            ZERO,
            args.alpha,
            args.t,
            args.n,
            5000,
            log,
        ),  # fmt: skip
        "planted": planted(
            args.plant_reps,
            [float(x) for x in args.strengths.split(",")],
            args.alpha,
            args.t,
            args.n,
            9000,
            log,
        ),  # fmt: skip
    }
    result["seconds"] = time.time() - started
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1))
    log("完成")


if __name__ == "__main__":
    main()
