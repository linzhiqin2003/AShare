"""优化问题、约束与求解诊断的数据契约；收益和风险均为日频。"""

from dataclasses import dataclass, field

import numpy as np

from ashare.risk import RiskModel


def vector(value, n, name, nonnegative=False):
    a = np.broadcast_to(np.asarray(value, dtype=float), (n,)).copy()
    if not np.isfinite(a).all() or (nonnegative and (a < 0).any()):
        raise ValueError(f"{name}: 非有限数或非法负数")
    return a


@dataclass(frozen=True)
class Constraints:
    invested: float | None = 1.0
    long_only: bool = True
    max_weight: float | np.ndarray | None = 0.02
    active_bound: float | np.ndarray | None = None
    industry_bound: float | np.ndarray | None = None
    style_bound: float | np.ndarray | None = None
    turnover: float | None = None
    tracking_error: float | None = None

    def __post_init__(self):
        if self.invested is not None and (
            not np.isfinite(self.invested) or not 0 <= self.invested <= 1
        ):
            raise ValueError("invested: 必须在 [0,1]")
        for name in (
            "max_weight",
            "active_bound",
            "industry_bound",
            "style_bound",
            "turnover",
            "tracking_error",
        ):
            v = getattr(self, name)
            if v is not None and (
                not np.isfinite(v).all() or (np.asarray(v) < 0).any()
            ):
                raise ValueError(f"{name}: 必须非负有限")


@dataclass(frozen=True)
class Relaxation:
    constraint: str
    value: float | None

    def __post_init__(self):
        if self.constraint not in (
            "max_weight",
            "active_bound",
            "industry_bound",
            "style_bound",
            "turnover",
            "tracking_error",
        ):
            raise ValueError("只能放松显式软约束；不可放松预算、只做多或方向冻结")
        if self.value is not None and (not np.isfinite(self.value) or self.value < 0):
            raise ValueError("放松值必须非负有限或 None（关闭）")


@dataclass
class PortfolioProblem:
    alpha: np.ndarray
    risk: RiskModel
    benchmark: np.ndarray
    current: np.ndarray
    risk_aversion: float = 10.0
    constraints: Constraints = field(default_factory=Constraints)
    buy_cost: float | np.ndarray = 0.0
    sell_cost: float | np.ndarray = 0.0
    can_buy: np.ndarray | None = None
    can_sell: np.ndarray | None = None
    industries: np.ndarray | None = None
    styles: np.ndarray | None = None
    impact: float | np.ndarray = 0.0
    industry_factors: tuple[int, ...] | None = None
    style_factors: tuple[int, ...] | None = None

    def __post_init__(self):
        n = len(self.risk.D)
        for name in (
            "alpha",
            "benchmark",
            "current",
            "buy_cost",
            "sell_cost",
            "impact",
        ):
            setattr(
                self,
                name,
                vector(
                    getattr(self, name),
                    n,
                    name,
                    name in ("buy_cost", "sell_cost", "impact"),
                ),
            )
        if not np.isfinite(self.risk_aversion) or self.risk_aversion <= 0:
            raise ValueError("risk_aversion: 必须为正有限数")
        for name in ("can_buy", "can_sell"):
            value = getattr(self, name)
            if value is None:
                value = np.ones(n, dtype=bool)
            value = np.asarray(value)
            if value.shape != (n,) or value.dtype != bool:
                raise ValueError(f"{name}: 必须为长度 N 的 bool 数组")
            setattr(self, name, value.copy())
        for name, bound in (
            ("industries", self.constraints.industry_bound),
            ("styles", self.constraints.style_bound),
        ):
            value = getattr(self, name)
            indices = getattr(
                self, "industry_factors" if name == "industries" else "style_factors"
            )
            if indices is not None:
                indices = np.asarray(indices)
                if (
                    indices.ndim != 1
                    or indices.dtype.kind not in "iu"
                    or np.any(indices < 0)
                    or np.any(indices >= self.risk.B.shape[1])
                ):
                    raise ValueError(f"{name}: 因子下标非法")
                selected = self.risk.B[:, indices]
                if value is not None and not np.array_equal(value, selected):
                    raise ValueError(f"{name}: 暴露与所给因子下标不一致")
                value = selected
            if value is None:
                if bound is not None:
                    raise ValueError(f"{name}: 启用对应约束时必须提供 N×J 暴露")
            else:
                value = np.asarray(value, dtype=float)
                if (
                    value.ndim != 2
                    or value.shape[0] != n
                    or not np.isfinite(value).all()
                ):
                    raise ValueError(f"{name}: 必须为有限 N×J 数组")
                setattr(self, name, value.copy())
                if bound is not None:
                    vector(bound, value.shape[1], name + "_bound", True)
        for name in ("max_weight", "active_bound"):
            value = getattr(self.constraints, name)
            if value is not None:
                vector(value, n, name, True)


def box_limits(p):
    """只放宽与交易方向冲突的个股边界；不放宽预算和组合约束。"""
    n, c = len(p.alpha), p.constraints
    upper = (
        np.full(n, np.inf)
        if c.max_weight is None
        else vector(c.max_weight, n, "max_weight")
    )
    active_lower = (
        np.full(n, -np.inf) if c.active_bound is None else p.benchmark - c.active_bound
    )
    active_upper = (
        np.full(n, np.inf) if c.active_bound is None else p.benchmark + c.active_bound
    )
    records = []
    for name, side, values, blocked in (
        ("max_weight", "upper", upper, ~p.can_sell),
        ("active_bound", "upper", active_upper, ~p.can_sell),
        ("active_bound", "lower", active_lower, ~p.can_buy),
    ):
        conflict = blocked & (
            (values < p.current) if side == "upper" else (values > p.current)
        )
        for i in np.flatnonzero(conflict):
            records.append(
                {
                    "index": int(i),
                    "symbol": p.risk.symbols[i],
                    "constraint": name,
                    "side": side,
                    "before": float(values[i]),
                    "after": float(p.current[i]),
                    "reason": "cannot_sell" if side == "upper" else "cannot_buy",
                }
            )
        values[conflict] = p.current[conflict]
    return upper, active_lower, active_upper, records


def weight_bounds(p):
    upper, active_lower, active_upper, _ = box_limits(p)
    lower = np.maximum(0.0 if p.constraints.long_only else -np.inf, active_lower)
    upper = np.minimum(upper, active_upper)
    lower[~p.can_sell] = np.maximum(lower[~p.can_sell], p.current[~p.can_sell])
    upper[~p.can_buy] = np.minimum(upper[~p.can_buy], p.current[~p.can_buy])
    return lower, upper


def factor_indices(p, matrix):
    """精确匹配 B 的列；任意外部约束矩阵仍走原有 N 维表达。"""
    columns = {p.risk.B[:, j].tobytes(): j for j in range(p.risk.B.shape[1])}
    indices = [columns.get(matrix[:, j].tobytes()) for j in range(matrix.shape[1])]
    return None if any(i is None for i in indices) else np.asarray(indices, dtype=int)


def objective(problem, weights):
    delta = weights - problem.current
    cost = np.dot(problem.buy_cost, np.maximum(delta, 0)) + np.dot(
        problem.sell_cost, np.maximum(-delta, 0)
    )
    cost += np.dot(problem.impact, delta * delta)
    return float(
        problem.alpha @ weights
        - problem.risk_aversion * problem.risk.variance(weights, problem.benchmark)
        - cost
    )


def violations(problem, weights):
    p, c = problem, problem.constraints
    w = np.asarray(weights, dtype=float)
    a, trade = w - p.benchmark, w - p.current

    def over(x):
        return float(np.maximum(x, 0).max(initial=0))

    result = {
        "finite": 0.0 if np.isfinite(w).all() else float("inf"),
        "cannot_buy": over(trade[~p.can_buy]),
        "cannot_sell": over(-trade[~p.can_sell]),
    }
    if c.invested is not None:
        result["invested"] = abs(float(w.sum() - c.invested))
    if c.long_only:
        result["long_only"] = over(-w)
    if c.max_weight is not None:
        result["max_weight"] = over(w - box_limits(p)[0])
    if c.active_bound is not None:
        _, low, high, _ = box_limits(p)
        result["active_bound"] = max(over(low - w), over(w - high))
    for name, b, limit in (
        ("industry_bound", p.industries, c.industry_bound),
        ("style_bound", p.styles, c.style_bound),
    ):
        if limit is not None:
            result[name] = over(np.abs(b.T @ a) - limit)
    if c.turnover is not None:
        # 单边换手取买入和卖出两者较大值；现金变化也消耗额度。
        result["turnover"] = max(
            0.0,
            float(
                max(np.maximum(trade, 0).sum(), np.maximum(-trade, 0).sum())
                - c.turnover
            ),
        )
    if c.tracking_error is not None:
        result["tracking_error"] = max(
            0.0, p.risk.volatility(w, p.benchmark) - c.tracking_error
        )
    return result


@dataclass
class OptimizationResult:
    weights: np.ndarray
    success: bool
    status: str
    solver: str
    objective: float
    violations: dict[str, float]
    diagnostics: dict = field(default_factory=dict)
    relaxations: list[dict] = field(default_factory=list)
    attempts: list[dict] = field(default_factory=list)
    bound_relaxations: list[dict] = field(default_factory=list)


class VerificationError(RuntimeError):
    """求解器声称最优，但原始问题的独立核验失败。"""
