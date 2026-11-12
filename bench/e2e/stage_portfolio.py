"""阶段 5：组合优化——单期求解与连续 60 期月度调仓，股票数 1000、3000 和全部可用。

风险模型取风险阶段交接的最近 61 个月末快照（没有则自建较短窗口）；alpha 用
reversal_20 的截面 z 分数乘 1e-3（日收益量级）。只量默认路径（backend="auto"），
它也是 docs/RISK_BENCHMARKS.md 对依赖前期持仓的序列推荐的路径。
"""

import warnings

import numpy as np
from common import digest, measure
from stage_risk import MONTH, PERIODS, build_months

import ashare.data
from ashare.factor import Evaluator, default_registry
from ashare.portfolio import PortfolioProblem, optimize, optimize_sequence
from ashare.risk import RiskModel

SIZES = (1000, 3000, None)


def models(ctx):
    path = ctx["work"] / "handoff" / "risk_months.npz"
    if path.exists():
        with np.load(path) as f:
            return (
                f["days"] + int(f["offset"]),
                f["B"],
                f["F"],
                f["D"],
                tuple(f["symbols"].tolist()),
                "handoff",
            )
    days = MONTH * (8 if ctx["quick"] else PERIODS) + 300
    source, months, _ = build_months(ctx, days)
    keep = sorted(months)[-PERIODS:]
    offset = len(ashare.data.open(ctx["data"]).panel("close").dates) - len(source.dates)
    return (
        np.array(keep) + offset,
        np.array([months[d][0] for d in keep]),
        np.array([months[d][1] for d in keep]),
        np.array([months[d][2] for d in keep]),
        tuple(source.symbols),
        f"built_{days}_days",
    )


def problems(days, b, f, d, symbols, alpha, cap, size):
    good = np.isfinite(b).all(axis=(0, 2)) & np.isfinite(d).all(axis=0)
    good &= np.isfinite(cap[days]).all(axis=0) & (cap[days] > 0).all(axis=0)
    candidates = np.flatnonzero(good)
    order = candidates[np.argsort(-cap[days[-1], candidates], kind="stable")]
    universe = np.sort(order if size is None else order[:size])
    names = tuple(f"f{i}" for i in range(b.shape[2]))
    out = []
    for j, day in enumerate(days):
        weights = cap[day, universe]
        risk = RiskModel(
            tuple(symbols[i] for i in universe),
            b[j][universe],
            names,
            f[j],
            d[j][universe],
        )
        out.append(
            PortfolioProblem(
                alpha=np.nan_to_num(alpha[day, universe]) * 1e-3,
                risk=risk,
                benchmark=weights / weights.sum(),
                current=np.zeros(len(universe)),
            )
        )
    return out, len(candidates)


def run(ctx):
    days, b, f, d, symbols, origin = models(ctx)
    data = ashare.data.open(ctx["data"])
    panel = data.panel(["total_market_cap", "close", "adj_factor"])
    cap = np.asarray(panel["total_market_cap"])
    factor = default_registry()["reversal_20"].expression
    score = np.asarray(Evaluator().evaluate(factor, panel))
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    mean = np.nanmean(score, axis=1, keepdims=True)
    sd = np.nanstd(score, axis=1, keepdims=True)
    alpha = (score - mean) / np.where(sd > 0, sd, 1)
    cases = []
    for size in (1000,) if ctx["quick"] else SIZES:
        plist, available = problems(days, b, f, d, symbols, alpha, cap, size)
        n = len(plist[0].alpha)
        scale = {
            "symbols": n,
            "available": available,
            "periods": len(plist),
            "models": origin,
        }
        stats, result = measure(lambda p=plist[-1]: optimize(p), ctx["repeat"](3))
        cases.append(
            {
                "case": f"single_auto_{size or 'all'}",
                "scale": scale,
                **stats,
                "summary": {"weights": digest(result.weights), "status": result.status},
            }
        )
        stats, (results, targets) = measure(
            lambda pl=plist: optimize_sequence(np.arange(len(pl)), pl),
            ctx["repeat"](1),
        )
        cases.append(
            {
                "case": f"sequence_auto_{size or 'all'}",
                "scale": scale,
                **stats,
                "summary": {
                    "weights": digest(targets.weights, targets.cols),
                    "failed": sum(not r.success for r in results),
                },
            }
        )
        del plist, results
    return cases
