"""简单对照组合和不依赖 panelbt 的稀疏目标数组。"""

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


def optimize_sequence(
    days,
    problems,
    *,
    initial=None,
    backend="auto",
    relaxations=(),
    tolerance=1e-7,
    columns=None,
):
    problems = list(problems)
    if len(days) != len(problems) or not problems:
        raise ValueError("需要等长且非空的日期与问题序列")
    current = (
        problems[0].current if initial is None else np.asarray(initial, dtype=float)
    )
    symbols = problems[0].risk.symbols
    session = SolverSession(backend)
    results = []
    for p in problems:
        if p.risk.symbols != symbols:
            raise ValueError("批量模型必须按同一主证券表排列")
        result = optimize(
            replace(p, current=current),
            backend=backend,
            session=session,
            relaxations=relaxations,
            tolerance=tolerance,
        )
        results.append(result)
        current = result.weights
    return results, to_targets(days, np.array([r.weights for r in results]), columns)
