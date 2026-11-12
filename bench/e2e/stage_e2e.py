"""阶段 7：端到端一次完整研究，按步骤计时形成瀑布。

算一批因子（首批 + 代表性表达式）并逐个做秩 IC → 选 |IC| 最大的一个做全套检验 →
全历史全市场风险模型（写缓存）→ 用该因子做 alpha，最近 60 个月末连续优化 →
目标权重下一交易日开盘成交的完整 A 股规则回测。同一进程串行，只跑一次。

三个用例：
- `research_once`：全部走默认接口和默认配置（逐个 `evaluate` 加 `rank_ic`、串行优化）。
  风险模型的特质风险默认走增量算法（P-006，相对从头算 ≤ 1e-12），所以风险、组合、回测
  的摘要与优化前的基线不逐位相同，差在这一处容差上。
- `research_once_exact`：同上，只把特质风险切回从头算（`specific_incremental=False`）。
  这是逐位对照：所有摘要应与优化前的基线完全相同。
- `research_once_fast`：改用这一轮新增的批量和并行入口（`evaluate_many`、`rank_ic_many`、
  `optimize_sequence(parallel=True)`）。前两个入口与默认逐位相同；并行优化与串行在求解
  容差内一致但不逐位（见 docs/perf_log.d/P-014）。
"""

import shutil
import time
import warnings
from dataclasses import replace

import numpy as np
import panelbt
import panelops
from common import digest, task_info
from expressions import expressions
from risk_adapter import load
from stage_backtest import config, costs, market_inputs
from stage_evaluation import forward_returns, suite
from stage_portfolio import problems
from stage_risk import MONTH, PERIODS

import ashare.data
from ashare.backtest import execution_days
from ashare.factor import Evaluator, default_registry
from ashare.portfolio import optimize_sequence
from ashare.risk import FundamentalConfig, ModelCache, model_series


def run(ctx):
    cases = [research(ctx)]
    if hasattr(FundamentalConfig, "specific_incremental"):
        cases.append(research(ctx, exact=True))
    if hasattr(Evaluator, "evaluate_many") and hasattr(panelops, "rank_ic_many"):
        cases.append(research(ctx, fast=True))
    return cases


def research(ctx, fast=False, exact=False):
    panelops.set_num_threads(ctx["threads"])
    steps, summary, peak = {}, {}, [0]
    clock = [time.perf_counter()]

    def mark(name):
        now = time.perf_counter()
        steps[name] = steps.get(name, 0.0) + now - clock[0]
        clock[0] = now
        peak[0] = max(peak[0], task_info()[0])

    start = time.perf_counter()
    cpu = time.process_time()
    data = ashare.data.open(ctx["data"])
    panel = data.panel(list(data._meta["fields"]))
    evaluator = Evaluator()
    forward = forward_returns(evaluator, panel)
    mark("data_open_and_forward_returns")

    batch = {f.name: f.expression for f in default_registry()}
    batch.update(expressions())
    if ctx["quick"]:
        batch = dict(list(batch.items())[:12])
    best, best_ic, best_name, ics = None, -1.0, None, {}
    if fast:

        def scores():
            # 生成器里的时间记到求值，生成器外（秩 IC）的时间记到 IC。
            for _, score in evaluator.evaluate_many(batch, panel):
                mark("factor_evaluate")
                yield score
                mark("factor_rank_ic")

        mark("factor_rank_ic")
        series = panelops.rank_ic_many(scores(), forward)
        mark("factor_rank_ic")
        for name, values in zip(batch, series, strict=True):
            ic = float(np.nanmean(values))
            ics[name] = ic
            if np.isfinite(ic) and abs(ic) > best_ic:
                best_ic, best_name = abs(ic), name
        best = np.array(evaluator.evaluate(batch[best_name], panel))
        mark("factor_evaluate")
    else:
        for name, expression in batch.items():
            score = evaluator.evaluate(expression, panel)
            mark("factor_evaluate")
            ic = float(np.nanmean(panelops.rank_ic(score, forward)))
            ics[name] = ic
            if np.isfinite(ic) and abs(ic) > best_ic:
                best, best_ic, best_name = np.array(score), abs(ic), name
            del score
            mark("factor_rank_ic")
    sign = np.sign(ics[best_name])
    days = np.arange(252, len(forward) - 1, MONTH, dtype=np.int64)
    summary["factor"] = {"chosen": best_name, "rank_ic": ics[best_name]}
    summary["tests"] = suite(best, forward, days)
    mark("factor_full_tests")

    source, risk_config = load(ctx["data"], 400 if ctx["quick"] else None)
    if exact:
        risk_config = replace(risk_config, specific_incremental=False)
    mark("risk_adapter_load")
    t = len(source.dates)
    cache_dir = ctx["work"] / "e2e-risk-cache"
    shutil.rmtree(cache_dir, ignore_errors=True)
    months = {}
    for day, model in enumerate(
        model_series(source, config=risk_config, cache=ModelCache(cache_dir))
    ):
        if day % MONTH == (t - 2) % MONTH:
            months[day] = (model.B, model.F, model.D)
            if len(months) > PERIODS:
                months.pop(min(months))
    symbols = tuple(source.symbols)
    offset = len(panel.dates) - t
    summary["risk_last"] = digest(*months[max(months)])
    del source
    shutil.rmtree(cache_dir, ignore_errors=True)
    mark("risk_model_full")

    keep = sorted(months)[-PERIODS:]
    rows = np.array(keep) + offset
    b = np.array([months[d][0] for d in keep])
    f = np.array([months[d][1] for d in keep])
    dd = np.array([months[d][2] for d in keep])
    cap = np.asarray(panel["total_market_cap"])
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    mean = np.nanmean(best, axis=1, keepdims=True)
    sd = np.nanstd(best, axis=1, keepdims=True)
    alpha = sign * (best - mean) / np.where(sd > 0, sd, 1)
    plist, _ = problems(rows, b, f, dd, symbols, alpha, cap, None)
    mark("portfolio_setup")
    universe = np.flatnonzero(np.isin(panel.symbols, plist[0].risk.symbols))
    results, sparse = optimize_sequence(
        execution_days(rows, len(panel.dates), tail="error"),
        plist,
        columns=universe.astype(np.int64),
        **({"parallel": True} if fast else {}),
    )
    summary["portfolio"] = {
        "weights": digest(sparse.weights, sparse.cols),
        "failed": sum(not r.success for r in results),
        "symbols": len(universe),
    }
    mark("portfolio_optimize_sequence")

    market = panelbt.Market(**market_inputs(panel))
    mark("backtest_market_build")
    targets = panelbt.Targets(sparse.days, sparse.indptr, sparse.cols, sparse.weights)
    result = panelbt.run(
        market, targets, costs(panel.dates), config(ctx["data"], panel.symbols)
    )
    summary["backtest"] = {
        "nav": digest(result.nav),
        "final_nav": float(result.nav[-1]),
    }
    mark("backtest_run")
    wall = time.perf_counter() - start
    return {
        "case": "research_once" + ("_fast" if fast else "_exact" if exact else ""),
        "scale": {
            "expressions": len(batch),
            "shape": panel.shape,
            "risk_days": t,
            "periods": len(plist),
        },
        "repeat": 1,
        "wall_min": wall,
        "wall_median": wall,
        "cpu_ratio": (time.process_time() - cpu) / wall,
        "peak_rss_mb": peak[0] / 2**20,
        "summary": summary,
        "detail": {"steps": {k: round(v, 4) for k, v in steps.items()}},
    }
