"""风险平价（等风险贡献，可给风险预算）的求解。

正式实现：对凸目标 ``f(y) = ½ yᵀΣy − Σ bᵢ log yᵢ``（y > 0）做带回溯的牛顿法（Spinu 2013）。
一阶条件 ``yᵢ (Σy)ᵢ = bᵢ`` 正是「风险贡献与预算成比例」，归一化 ``w = y / Σy`` 即得权重。
Hessian ``Σ + diag(b / y²)`` 在 y > 0 时正定，所以协方差奇异（完全相关、样本短于资产数）时
仍有唯一解。独立的参考实现见 `ashare.allocation.reference.risk_parity_reference`。
"""

from __future__ import annotations

import numpy as np


def risk_contributions(weights, sigma):
    """各资产的风险贡献占比 ``wᵢ(Σw)ᵢ / wᵀΣw``；组合方差为 0 时返回 NaN。"""
    w = np.asarray(weights, dtype=float)
    s = np.asarray(sigma, dtype=float)
    rc = w * (s @ w)
    total = rc.sum()
    return rc / total if total > 0 else np.full_like(w, np.nan)


def _newton(s_a, b_a, tol, max_iter):
    """在 y > 0 上最小化 ½yᵀΣy − bᵀlog y；返回 (y, 迭代数, 状态)。

    状态：``"converged"``；``"unbounded"``（目标无下界：存在方差为 0 的长仓组合）；``"max_iter"``（迭代
    次数用完）。
    """
    y = b_a / np.sqrt(np.diag(s_a))
    variance = float(y @ s_a @ y)
    status = "max_iter"
    iterations = 0
    if not variance > 1e-12 * float(np.mean(np.diag(s_a))) * float(y @ y):
        return y, iterations, "unbounded"  # 起点就是零方差组合：无解
    y *= np.sqrt(b_a.sum() / variance)

    def value(v):
        return 0.5 * float(v @ s_a @ v) - float(b_a @ np.log(v))

    for step_number in range(1, max_iter + 1):
        iterations = step_number
        sy = s_a @ y
        grad = sy - b_a / y
        if np.max(np.abs(grad * y)) <= tol:
            status = "converged"
            break
        hess = s_a + np.diag(b_a / (y * y))
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            status = "unbounded"
            break
        slope = float(grad @ step)
        # 牛顿减量已到机器精度：再迭代只是在舍入误差里打转。
        if slope <= 1e-26 * max(1.0, float(b_a.sum())):
            status = "converged"
            break
        # 回溯：保持 y > 0 且目标下降（Armijo）。
        alpha = 1.0
        negative = step > 0
        if negative.any():
            alpha = min(1.0, 0.99 * float(np.min(y[negative] / step[negative])))
        current = value(y)
        # 减量很小时已在二次收敛区：目标值的差会被舍入误差淹没，不做回溯（y > 0 仍由 alpha 保证）。
        while alpha > 1e-16 and slope > 1e-10:
            trial = y - alpha * step
            if (trial > 0).all() and value(trial) <= current - 1e-4 * alpha * slope:
                break
            alpha *= 0.5
        y = y - alpha * step
        if not np.isfinite(y).all() or y.max() > 1e12:
            status = "unbounded"  # 目标无下界：存在零方差的长仓组合（协方差奇异）
            break
    return y, iterations, status


def solve_risk_parity(sigma, budget=None, *, tol=1e-12, max_iter=100):
    """返回 (权重, 诊断)。

    sigma：n×n 半正定协方差，对角线必须为正（零方差资产由调用方先剔除）。
    budget：风险预算，非负、和为 1；None 为等预算。预算为 0 的资产权重为 0。
    诊断：
    - ``converged``：在**原协方差**上解到了等风险贡献（残差 ≤ 1e-8）；
    - ``solved_with_ridge``：原协方差上无解，加了 ridge 才解到（此时 ``converged`` 为 False）；
    - ``ridge``：用到的 ridge（× 平均方差），0 表示没加；
    - ``residual``：在实际求解的矩阵（加了 ridge 时是加过的）上，风险贡献占比与预算的最大绝对差；
    - ``residual_original``：同样的权重在原协方差上的残差（加了 ridge 时可能很大，算不出时为 None）；
    - ``status``：``converged`` / ``ridge`` / ``max_iter`` / ``unsolved``；``iterations``。

    协方差奇异且存在**方差为 0 的长仓组合**时，等风险贡献的解不存在（目标无下界）。此时依次在
    对角线加 ``ridge × 平均方差``（1e-10、1e-8、1e-6、1e-4）重解，小 ridge 下病态到迭代不够就换更大的。
    只有原协方差上「无下界」才加 ridge；原协方差上迭代次数用完（``max_iter`` 太小）直接报 ``max_iter``。其余奇异情形（完全相同的两只资产等）
    不需要 ridge。
    """
    s = np.asarray(sigma, dtype=float)
    n = s.shape[0]
    if s.ndim != 2 or s.shape != (n, n) or n == 0 or not np.isfinite(s).all():
        raise ValueError("sigma: 需要有限的非空方阵")
    diag = np.diag(s)
    if (diag <= 0).any():
        raise ValueError("sigma: 对角线必须为正（零方差资产请先剔除）")
    b = np.full(n, 1.0 / n) if budget is None else np.asarray(budget, dtype=float)
    if b.shape != (n,) or not np.isfinite(b).all() or (b < 0).any() or b.sum() <= 0:
        raise ValueError("budget: 需要长度 n 的非负有限向量，且不全为 0")
    b = b / b.sum()
    active = b > 0
    weights = np.zeros(n)
    if active.sum() == 1:
        weights[active] = 1.0
        return weights, dict(
            iterations=0,
            converged=True,
            solved_with_ridge=False,
            status="converged",
            residual=0.0,
            residual_original=0.0,
            ridge=0.0,
        )
    s_a, b_a = s[np.ix_(active, active)], b[active]
    scale = float(np.mean(np.diag(s_a)))
    total_iterations = 0
    solved = False
    for ridge in (0.0, 1e-10, 1e-8, 1e-6, 1e-4):
        used = s_a + ridge * scale * np.eye(len(s_a)) if ridge else s_a
        y, iterations, status = _newton(used, b_a, tol, max_iter)
        total_iterations += iterations
        if status == "converged" and np.isfinite(y).all():
            share = risk_contributions(y, used)
            solved = bool(
                np.isfinite(share).all() and np.max(np.abs(share - b_a)) <= 1e-8
            )
            if solved:
                break
            status = "unbounded"  # 停在残差不达标的点上：按奇异处理
        if status == "max_iter" and ridge == 0:
            break  # 原协方差上只是迭代不够：不加 ridge（加了 ridge 后迭代不够就换更大的 ridge）
    weights[active] = y / y.sum()
    full = s.copy()
    full[np.ix_(active, active)] = used
    with np.errstate(all="ignore"):
        residual = float(np.max(np.abs(risk_contributions(weights, full) - b)))
        original = float(np.max(np.abs(risk_contributions(weights, s) - b)))
    if solved:
        status = "converged" if ridge == 0 else "ridge"
    elif status != "max_iter":
        status = "unsolved"
    return weights, dict(
        iterations=int(total_iterations),
        converged=bool(solved and ridge == 0),
        solved_with_ridge=bool(solved and ridge > 0),
        status=status,
        residual=residual,
        residual_original=original if np.isfinite(original) else None,
        ridge=float(ridge),
    )
