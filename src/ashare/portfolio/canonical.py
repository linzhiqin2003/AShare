"""因子结构的稀疏锥表示，变量顺序为主动权重、因子权重、买入、卖出。"""

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

from .problem import factor_indices, vector, weight_bounds


@dataclass
class ConicData:
    P: sparse.csc_matrix
    q: np.ndarray
    A: sparse.csc_matrix
    b: np.ndarray
    cones: list[tuple[int, int]]
    n: int
    scale: float = 1000.0
    vector_blocks: list = field(default_factory=list, repr=False)

    def update_vectors(self, p):
        lower, upper = weight_bounds(p)
        lower, upper = lower - p.benchmark, upper - p.benchmark
        self.q = linear_objective(p)
        self.b = (
            np.concatenate([fn(p, lower, upper) for fn in self.vector_blocks]) * self.n
        )


def linear_objective(p):
    return (
        np.concatenate(
            (
                -p.alpha + 2 * p.impact * (p.benchmark - p.current),
                np.zeros(p.risk.B.shape[1]),
                p.buy_cost,
                p.sell_cost,
            )
        )
        * 1000
    )


def structure_key(p):
    # P/A 不变时只更新 q/b；任何会改变行布局或系数的输入均进入键。
    low, high = weight_bounds(p)
    fixed = np.isfinite(low) & np.isfinite(high) & (low == high)
    c = p.constraints
    return (
        id(p.risk),
        p.risk_aversion,
        p.impact.tobytes(),
        c.long_only,
        tuple(
            getattr(c, name) is not None
            for name in (
                "invested",
                "max_weight",
                "active_bound",
                "industry_bound",
                "style_bound",
                "turnover",
                "tracking_error",
            )
        ),
        np.isfinite(low).tobytes(),
        np.isfinite(high).tobytes(),
        fixed.tobytes(),
        None if p.industries is None else (p.industries.shape, p.industries.tobytes()),
        None if p.styles is None else (p.styles.shape, p.styles.tobytes()),
    )


def canonicalize(problem, *, factor_constraints=True):
    p, c = problem, problem.constraints
    n, k = p.risk.B.shape
    m = 3 * n + k
    identity = sparse.eye(n, format="csc")

    def Z(rows, cols):
        return sparse.csc_matrix((rows, cols))

    # q 的常数项无关解，报告目标值始终用原始定义独立重算。
    P = sparse.block_diag(
        (
            sparse.diags(2 * p.risk_aversion * p.risk.D + 2 * p.impact),
            2 * p.risk_aversion * sparse.csc_matrix(p.risk.F),
            Z(2 * n, 2 * n),
        ),
        format="csc",
    )
    q = linear_objective(p)
    eq = [
        sparse.hstack(
            (-sparse.csc_matrix(p.risk.B.T), sparse.eye(k), Z(k, 2 * n)), format="csc"
        ),
        sparse.hstack((identity, Z(n, k), -identity, identity), format="csc"),
    ]
    rhs = [np.zeros(k), p.current - p.benchmark]
    eq_updates = [
        lambda p, lo, hi: np.zeros(k),
        lambda p, lo, hi: p.current - p.benchmark,
    ]
    ineq_updates = []
    if c.invested is not None:
        eq.append(
            sparse.csc_matrix(
                np.concatenate((np.ones(n), np.zeros(k + 2 * n)))[None, :]
            )
        )
        rhs.append(np.array([c.invested - p.benchmark.sum()]))
        eq_updates.append(
            lambda p, lo, hi: np.array([p.constraints.invested - p.benchmark.sum()])
        )
    inequalities, limits = [], []

    def add_weights(matrix, bound):
        matrix = sparse.csc_matrix(matrix)
        inequalities.append(
            sparse.hstack((matrix, Z(matrix.shape[0], k + 2 * n)), format="csc")
        )
        limits.append(np.atleast_1d(bound))

    lower, upper = weight_bounds(p)
    lower, upper = lower - p.benchmark, upper - p.benchmark
    # 相切的上下界直接进入等式锥，避免内点法在零宽度面上退化。
    fixed = np.isfinite(lower) & np.isfinite(upper) & (lower == upper)
    if fixed.any():
        eq.append(
            sparse.hstack(
                (identity[fixed], Z(int(fixed.sum()), k + 2 * n)), format="csc"
            )
        )
        rhs.append(lower[fixed])
        eq_updates.append(lambda p, lo, hi: lo[fixed])
    valid = np.isfinite(upper) & ~fixed
    add_weights(identity[valid], upper[valid])
    ineq_updates.append(lambda p, lo, hi, ix=valid: hi[ix])
    valid = np.isfinite(lower) & ~fixed
    add_weights(-identity[valid], -lower[valid])
    ineq_updates.append(lambda p, lo, hi, ix=valid: -lo[ix])
    inequalities.append(
        sparse.hstack((Z(2 * n, n + k), -sparse.eye(2 * n)), format="csc")
    )
    limits.append(np.zeros(2 * n))
    ineq_updates.append(lambda p, lo, hi: np.zeros(2 * n))
    for name, matrix, bound in (
        ("industry_bound", p.industries, c.industry_bound),
        ("style_bound", p.styles, c.style_bound),
    ):
        if bound is not None:
            limit = vector(bound, matrix.shape[1], "exposure_bound")
            ix = factor_indices(p, matrix) if factor_constraints else None
            if ix is None:
                add_weights(matrix.T, limit)
                add_weights(-matrix.T, limit)
            else:
                rows = np.arange(len(ix))
                selection = sparse.csc_matrix(
                    (np.ones(len(ix)), (rows, n + ix)), shape=(len(ix), m)
                )
                inequalities.extend((selection, -selection))
                limits.extend((limit, limit))
            for _ in range(2):
                ineq_updates.append(
                    lambda p, lo, hi, name=name, size=matrix.shape[1]: vector(
                        getattr(p.constraints, name), size, name
                    )
                )
    if c.turnover is not None:
        row = np.zeros((2, m))
        row[0, n + k : n + k + n] = 1
        row[1, n + k + n :] = 1
        inequalities.append(sparse.csc_matrix(row))
        limits.append(np.full(2, c.turnover))
        ineq_updates.append(lambda p, lo, hi: np.full(2, p.constraints.turnover))
    Aeq, Aineq = (
        sparse.vstack(eq, format="csc"),
        sparse.vstack(inequalities, format="csc"),
    )
    b = np.concatenate((*rhs, *limits))
    A = sparse.vstack((Aeq, Aineq), format="csc")
    cones = [(0, Aeq.shape[0]), (1, Aineq.shape[0])]
    soc_updates = []
    if c.tracking_error is not None:
        vals, vec = np.linalg.eigh(p.risk.F)
        root = np.sqrt(np.maximum(vals, 0))[:, None] * vec.T
        risk = sparse.vstack(
            (
                sparse.hstack((Z(k, n), sparse.csc_matrix(root), Z(k, 2 * n))),
                sparse.hstack((sparse.diags(np.sqrt(p.risk.D)), Z(n, k + 2 * n))),
            ),
            format="csc",
        )
        soc = sparse.vstack((Z(1, m), -risk), format="csc")
        A = sparse.vstack((A, soc), format="csc")
        b = np.concatenate((b, [c.tracking_error], np.zeros(k + n)))
        cones.append((2, k + n + 1))
        soc_updates.append(
            lambda p, lo, hi: np.r_[p.constraints.tracking_error, np.zeros(k + n)]
        )
    A.sort_indices()
    P = sparse.triu(P, format="csc")
    return ConicData(
        P * (1000 / n),
        q,
        A,
        b * n,
        cones,
        n,
        1000.0 * n,
        eq_updates + ineq_updates + soc_updates,
    )
