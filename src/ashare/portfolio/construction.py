"""简单对照组合和不依赖 panelbt 的稀疏目标数组。"""

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace

import numpy as np

from .optimize import optimize
from .solvers import SolverSession


def top_equal(scores, count, eligible=None):
    x = np.asarray(scores, dtype=float)
    if x.ndim != 1 or not isinstance(count, int) or count < 1:
        raise ValueError("scores/count: 需要向量与正整数")
    valid = np.isfinite(x)
    if eligible is not None:
        valid &= np.asarray(eligible, dtype=bool)
    candidates = np.flatnonzero(valid)
    chosen = candidates[np.argsort(-x[candidates], kind="stable")[:count]]
    w = np.zeros(len(x))
    if len(chosen):
        w[chosen] = 1 / len(chosen)
    return w


def score_weighted(scores, eligible=None):
    x = np.asarray(scores, dtype=float)
    valid = np.isfinite(x)
    if eligible is not None:
        valid &= np.asarray(eligible, dtype=bool)
    weights = np.where(valid, np.maximum(x, 0), 0)
    return weights / weights.sum() if weights.sum() > 0 else np.zeros_like(weights)


def industry_proportional(scores, industries, benchmark, fraction=0.2):
    if not 0 < fraction <= 1:
        raise ValueError("fraction: 必须在 (0,1]")
    groups = np.asarray(industries)
    bench = np.asarray(benchmark, dtype=float)
    if bench.shape != groups.shape or np.any(bench < 0) or not np.isfinite(bench).all():
        raise ValueError("benchmark: 非负有限且与行业同形状")
    w = np.zeros(len(bench))
    for code in np.unique(groups):
        ix = np.flatnonzero(groups == code)
        selected = top_equal(
            np.asarray(scores)[ix], max(1, int(np.ceil(len(ix) * fraction)))
        )
        w[ix] = selected * bench[ix].sum()
    return w


@dataclass(frozen=True)
class SparseTargets:
    days: np.ndarray
    indptr: np.ndarray
    cols: np.ndarray
    weights: np.ndarray

    def arrays(self):
        return self.days, self.indptr, self.cols, self.weights


def to_targets(days, weights, columns=None):
    d = np.asarray(days)
    w = np.asarray(weights, dtype=float).copy()
    if (
        d.ndim != 1
        or d.dtype.kind not in "iu"
        or np.any(d < 0)
        or np.any(np.diff(d) <= 0)
    ):
        raise ValueError("days: 需要非负严格升序整数")
    if (
        w.ndim != 2
        or len(w) != len(d)
        or not np.isfinite(w).all()
        or np.any(w < 0)
        or np.any(w.sum(axis=1) > 1 + 1e-7)
    ):
        raise ValueError("weights: 需要非负、总和不超过 1 的调仓面板")
    # 消除求解容差带来的极小超额，使 Targets 满足严格的现金约束。
    for row in w:
        if row.sum() > 1:
            row[np.argmax(row)] -= row.sum() - 1
    mapping = (
        np.arange(w.shape[1], dtype=np.int64)
        if columns is None
        else np.asarray(columns)
    )
    if (
        mapping.shape != (w.shape[1],)
        or mapping.dtype.kind not in "iu"
        or np.any(mapping < 0)
        or len(np.unique(mapping)) != len(mapping)
    ):
        raise ValueError("columns: 必须是长度 N、唯一且非负的全市场整数下标")
    rows, cols = np.nonzero(w)
    values = w[rows, cols]
    cols = mapping[cols]
    order = np.lexsort((cols, rows))
    rows, cols, values = rows[order], cols[order], values[order]
    counts = np.bincount(rows, minlength=len(d))
    return SparseTargets(
        d.astype(np.int64),
        np.r_[0, np.cumsum(counts)].astype(np.int64),
        cols.astype(np.int64),
        values,
    )


def independent_periods(problems):
    """各期能否互不依赖地求解：只在确定上一期持仓不进入问题时返回 True。

    上一期持仓进入问题的途径：交易成本和冲击成本（目标函数）、换手约束、不能买或
    不能卖的冻结股票（个股边界取当前持仓）。都没有时，前期持仓只出现在
    「主动权重 − 买入 + 卖出 = 持仓 − 基准」这条等式的右端，买卖变量没有代价也没有
    上限，最优权重与持仓无关；但求解器看到的右端不同，数值解不逐位相同（差异见
    docs/perf_log.d/P-014）。拿不准一律返回 False。
    """
    for p in problems:
        c = p.constraints
        if (
            c.turnover is not None
            or np.any(p.buy_cost)
            or np.any(p.sell_cost)
            or np.any(p.impact)
            or not np.all(p.can_buy)
            or not np.all(p.can_sell)
        ):
            return False
    return True


def optimize_sequence(
    days,
    problems,
    *,
    initial=None,
    backend="auto",
    relaxations=(),
    tolerance=1e-7,
    columns=None,
    parallel=False,
    threads=None,
):
    """逐期递推优化，上一期结果作为下一期的当前持仓。

    parallel=True 时，若各期互不依赖（见 `independent_periods`），各期用各自的
    会话在线程池里独立求解（求解器释放 GIL）；每期的当前持仓取该问题自带的
    `current`。结果与串行在求解容差内一致但不逐位相同，所以默认仍串行。有依赖时
    忽略 parallel、照常串行。任一期失败时从第一个失败期起改回串行递推，使
    fallback_previous 的语义与串行相同。threads 默认取 CPU 核数与期数的较小值。
    """
    problems = list(problems)
    if len(days) != len(problems) or not problems:
        raise ValueError("需要等长且非空的日期与问题序列")
    if threads is not None and (not isinstance(threads, int) or threads < 1):
        raise ValueError("threads: 需要正整数或 None")
    current = (
        problems[0].current if initial is None else np.asarray(initial, dtype=float)
    )
    symbols = problems[0].risk.symbols
    for p in problems:
        if p.risk.symbols != symbols:
            raise ValueError("批量模型必须按同一主证券表排列")
    options = dict(backend=backend, relaxations=relaxations, tolerance=tolerance)
    results = []
    if parallel and len(problems) > 1 and independent_periods(problems):
        first = replace(problems[0], current=current)
        workers = min(len(problems), threads or os.cpu_count() or 1)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            solved = list(
                pool.map(
                    lambda p: optimize(p, session=SolverSession(backend), **options),
                    [first, *problems[1:]],
                )
            )
        for result in solved:
            if not result.success:
                break
            results.append(result)
        if results:
            current = results[-1].weights
    session = SolverSession(backend)
    for p in problems[len(results) :]:
        result = optimize(
            replace(p, current=current),
            session=session,
            **options,
        )
        results.append(result)
        current = result.weights
    return results, to_targets(days, np.array([r.weights for r in results]), columns)
