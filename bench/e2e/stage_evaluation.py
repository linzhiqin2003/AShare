"""阶段 3：因子检验——秩 IC、IC、分组收益、分组换手、自相关、目标权重。

对首批因子逐个求值（不计时），再对每个因子做全套检验并按检验类别累计时间；
另对一个代表性因子重复测量整套检验。
"""

import time

import numpy as np
import panelops
from common import Phases, digest, measure
from stage_factors import open_source

from ashare.factor import Evaluator, default_registry, returns

REBALANCE_EVERY = 21


def forward_returns(evaluator, source):
    daily = np.asarray(evaluator.evaluate(returns(), source))
    forward = np.full_like(daily, np.nan)
    forward[:-1] = daily[1:]
    return forward


def suite(score, forward, days, phases=None):
    calls = {
        "rank_ic": lambda: panelops.rank_ic(score, forward),
        "ic": lambda: panelops.ic(score, forward),
        "group_returns": lambda: panelops.group_returns(score, forward, 5),
        "group_turnover": lambda: panelops.group_turnover(score, 5, REBALANCE_EVERY),
        "factor_autocorr": lambda: panelops.factor_autocorr(score, REBALANCE_EVERY),
        "targets": lambda: panelops.targets(score, fraction=0.1, days=days),
    }
    out = {}
    for name, fn in calls.items():
        start = time.perf_counter()
        out[name] = fn()
        if phases is not None:
            phases.add(name, time.perf_counter() - start)
    targets = out.pop("targets")
    return {
        **{
            k: digest(*v) if isinstance(v, tuple) else digest(v) for k, v in out.items()
        },
        "targets": digest(targets.indptr, targets.cols, targets.weights),
        "mean_rank_ic": float(np.nanmean(out["rank_ic"])),
    }


def run(ctx):
    panelops.set_num_threads(ctx["threads"])
    source = open_source(ctx)
    evaluator = Evaluator()
    forward = forward_returns(evaluator, source)
    days = np.arange(252, len(forward) - 1, REBALANCE_EVERY, dtype=np.int64)
    phases, summary = Phases(), {}
    factors = list(default_registry())
    if ctx["quick"]:
        factors = factors[:3]
    start = time.perf_counter()
    for factor in factors:
        score = np.asarray(evaluator.evaluate(factor.expression, source))
        summary[factor.name] = suite(score, forward, days, phases)
    total = time.perf_counter() - start
    cases = [
        {
            "case": "suite_all_factors_one_pass",
            "scale": {
                "factors": len(factors),
                "shape": source.shape,
                "rebalances": len(days),
            },
            "repeat": 1,
            "wall_min": sum(v["seconds"] for v in phases.report().values()),
            "wall_median": sum(v["seconds"] for v in phases.report().values()),
            "summary": summary,
            "detail": {"by_test": phases.report(), "including_factor_eval": total},
        }
    ]
    # 多个因子对同一远期收益的秩 IC：逐个 rank_ic 与（P-011 起的）rank_ic_many。
    scores = [np.asarray(evaluator.evaluate(f.expression, source)) for f in factors[:8]]
    variants = {"rank_ic_loop": lambda: [panelops.rank_ic(x, forward) for x in scores]}
    if hasattr(panelops, "rank_ic_many"):
        variants["rank_ic_many"] = lambda: panelops.rank_ic_many(scores, forward)
    for name, fn in variants.items():
        stats, value = measure(lambda fn=fn: digest(*fn()), ctx["repeat"](3))
        cases.append(
            {
                "case": f"{name}_{len(scores)}_factors",
                "scale": {"factors": len(scores), "shape": source.shape},
                **stats,
                "summary": value,
            }
        )
    scores.clear()
    score = np.asarray(evaluator.evaluate(factors[1].expression, source))
    stats, value = measure(lambda: suite(score, forward, days), ctx["repeat"](3))
    cases.append(
        {
            "case": f"suite_one_factor_{factors[1].name}",
            "scale": {"shape": source.shape},
            **stats,
            "summary": value,
        }
    )
    return cases
