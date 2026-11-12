"""带独立核验、显式放松和失败标记的组合构建。"""

from dataclasses import replace

import numpy as np

from .problem import (
    OptimizationResult,
    VerificationError,
    box_limits,
    objective,
    violations,
    weight_bounds,
)
from .solvers import SolverSession


def _clean_boundaries(p, weights, tolerance):
    raw = violations(p, weights)
    if max(raw.values(), default=0) > tolerance:
        raise VerificationError(f"最优状态下原解约束超限: {raw}")
    n, c = len(weights), p.constraints
    lower, upper = weight_bounds(p)
    if np.any(lower > upper):
        raise VerificationError("当前持仓与硬边界冲突，无法安全消除数值噪声")
    result = np.clip(weights, lower, upper)
    tiny = tolerance / (10 * n)
    near_lower = np.isfinite(lower) & (result - lower <= tiny)
    near_upper = np.isfinite(upper) & (upper - result <= tiny)
    result[near_lower] = lower[near_lower]
    result[near_upper] = upper[near_upper]
    if c.invested is not None:
        delta = c.invested - result.sum()
        slack = upper - result if delta > 0 else result - lower
        for index in np.argsort(-slack, kind="stable"):
            if abs(delta) < 1e-16:
                break
            adjustment = np.sign(delta) * min(abs(delta), slack[index])
            result[index] += adjustment
            delta -= adjustment
    return result


def optimize(problem, *, backend="auto", relaxations=(), tolerance=1e-7, session=None):
    """只接受 optimal；有限容差内的小负权重不作为可执行目标输出。"""
    if tolerance <= 0 or not np.isfinite(tolerance):
        raise ValueError("tolerance: 必须为正有限值")
    session = SolverSession(backend) if session is None else session
    attempts, changes = [], []
    current = problem
    for step in range(len(relaxations) + 1):
        weights, status, info = session.solve(current, min(tolerance * 0.01, 1e-9))
        attempts.append({"status": status, "solver": session.backend})
        if status not in (
            "optimal",
            "PrimalInfeasible",
            "infeasible",
            "DualInfeasible",
            "unbounded",
        ):
            # 重试保持经济问题不变；不能借放松约束掩盖数值失败。
            weights, status, info = session.solve(
                current, min(tolerance * 0.001, 1e-10)
            )
            attempts.append(
                {"status": status, "solver": session.backend, "kind": "numerical_retry"}
            )
        if status == "optimal":
            if weights is None or not np.isfinite(weights).all():
                raise VerificationError("最优解为空或非有限")
            weights = np.asarray(weights, dtype=float).copy()
            weights = _clean_boundaries(current, weights, tolerance)
            check = violations(current, weights)
            if max(check.values(), default=0) > tolerance:
                raise VerificationError(f"最优状态下约束超限: {check}")
            if not any(
                name in info for name in ("primal_residual", "dual_residual", "gap")
            ):
                raise VerificationError("求解器没有提供残差或对偶间隙")
            for name in ("primal_residual", "dual_residual", "gap"):
                value = info.get(name)
                if value is not None and (not np.isfinite(value) or value > tolerance):
                    raise VerificationError(f"{name} 超限: {value}")
            return OptimizationResult(
                weights,
                True,
                status,
                session.backend,
                objective(current, weights),
                check,
                info,
                changes.copy(),
                attempts,
                box_limits(current)[3],
            )
        if status not in ("PrimalInfeasible", "infeasible") or step == len(relaxations):
            break
        rule = relaxations[step]
        before = getattr(current.constraints, rule.constraint)
        after = rule.value
        if before is None or (after is not None and np.any(after < np.asarray(before))):
            raise ValueError(f"{rule.constraint}: 放松规则不能收紧或重新启用约束")
        changes.append(
            {
                "constraint": rule.constraint,
                "before": np.asarray(before).tolist(),
                "after": after,
                "increase": None
                if after is None
                else (after - np.asarray(before)).tolist(),
            }
        )
        current = replace(
            current,
            constraints=replace(current.constraints, **{rule.constraint: after}),
        )
    weights = problem.current.copy()
    return OptimizationResult(
        weights,
        False,
        "fallback_previous",
        session.backend,
        objective(problem, weights),
        violations(problem, weights),
        {"reason": attempts[-1]["status"], **info},
        changes,
        attempts,
        box_limits(problem)[3],
    )


def woodbury_solution(problem):
    """无不等式且无成本时的闭式解；只反演 K×K 矩阵。"""
    p = problem
    if (
        (p.risk.D <= 0).any()
        or np.any(p.buy_cost)
        or np.any(p.sell_cost)
        or np.any(p.impact)
    ):
        raise ValueError("闭式解要求 D>0 且无成本")
    eigen, vectors = np.linalg.eigh(p.risk.F)
    low = p.risk.B @ (vectors * np.sqrt(np.maximum(eigen, 0)))
    invd = 1 / p.risk.D
    middle = np.eye(low.shape[1]) + (low.T * invd) @ low

    def inverse(v):
        z = invd * v
        return z - invd * (low @ np.linalg.solve(middle, low.T @ z))

    one = inverse(np.ones(len(invd)))
    tilt = inverse(p.alpha) / (2 * p.risk_aversion)
    if p.constraints.invested is not None:
        tilt += (
            one * (p.constraints.invested - p.benchmark.sum() - tilt.sum()) / one.sum()
        )
    return p.benchmark + tilt


def prune_and_resolve(problem, result, threshold=1e-4, **kwargs):
    """去掉极小权重后重解；不能卖出的非零持仓不会被删除。"""
    if not result.success or threshold < 0:
        raise ValueError("需要成功的初始解和非负阈值")
    remove = (result.weights < threshold) & problem.can_sell
    upper = (
        np.full(len(result.weights), 1.0)
        if problem.constraints.max_weight is None
        else np.broadcast_to(
            problem.constraints.max_weight, result.weights.shape
        ).copy()
    )
    upper[remove] = 0
    refined = optimize(
        replace(problem, constraints=replace(problem.constraints, max_weight=upper)),
        **kwargs,
    )
    if not refined.success:
        return result, {
            "success": False,
            "removed": int(remove.sum()),
            "objective_difference": 0.0,
            "status": refined.status,
        }
    return refined, {
        "success": True,
        "removed": int(remove.sum()),
        "objective_difference": refined.objective - result.objective,
        "status": refined.status,
    }
