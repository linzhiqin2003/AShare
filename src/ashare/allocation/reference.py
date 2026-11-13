"""朴素的参考实现：只用 scipy 的通用优化器和直白的循环，与正式实现不共用代码，供测试对拍。

不在正式路径上，慢，精度到 1e-7 左右。
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


def risk_parity_reference(sigma, budget=None, sweeps=200000):
    """循环坐标下降（Griveau-Billion 等 2013）：逐个资产解一元二次方程 Σᵢᵢyᵢ² + cᵢyᵢ − bᵢ = 0。

    纯 Python 双重循环，与正式实现的牛顿法不共用代码；收敛到相邻两轮最大变化 < 1e-15。
    （试过 scipy 的 SLSQP 和非线性最小二乘直接解方程，负相关时会停在某个资产权重为 0 的假解上。）
    """
    s = np.asarray(sigma, dtype=float)
    n = len(s)
    b = (
        [1.0 / n] * n
        if budget is None
        else list(np.asarray(budget, dtype=float) / np.sum(budget))
    )
    y = [1.0 / float(np.sqrt(s[i, i])) if b[i] > 0 else 0.0 for i in range(n)]
    for _ in range(sweeps):
        change = 0.0
        for i in range(n):
            if b[i] == 0:
                continue
            c = sum(float(s[i, j]) * y[j] for j in range(n) if j != i)
            a = float(s[i, i])
            new = (-c + (c * c + 4 * a * b[i]) ** 0.5) / (2 * a)
            change = max(change, abs(new - y[i]) / max(abs(new), 1e-300))
            y[i] = new
        if change < 1e-15:
            break
    total = sum(y)
    return np.array([v / total for v in y])


def min_variance_reference(sigma, max_weight=None):
    """SLSQP 最小化 wᵀΣw，满仓长仓，可选单资产上限。"""
    s = np.asarray(sigma, dtype=float)
    n = len(s)
    cap = 1.0 if max_weight is None else max(max_weight, 1.0 / n)
    best = minimize(
        lambda w: float(w @ s @ w) * 1e4,
        np.full(n, 1.0 / n),
        jac=lambda w: 2e4 * (s @ w),
        method="SLSQP",
        bounds=[(0.0, cap)] * n,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
        options={"ftol": 1e-15, "maxiter": 2000},
    )
    w = np.clip(best.x, 0.0, None)
    return w / w.sum()


def mean_variance_reference(sigma, mu, risk_aversion, max_weight=None):
    """SLSQP 最大化 μᵀw − λ wᵀΣw。"""
    s = np.asarray(sigma, dtype=float)
    m = np.asarray(mu, dtype=float)
    n = len(s)
    cap = 1.0 if max_weight is None else max(max_weight, 1.0 / n)
    best = minimize(
        lambda w: -(float(m @ w) - risk_aversion * float(w @ s @ w)) * 1e4,
        np.full(n, 1.0 / n),
        method="SLSQP",
        bounds=[(0.0, cap)] * n,
        constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}],
        options={"ftol": 1e-15, "maxiter": 2000},
    )
    w = np.clip(best.x, 0.0, None)
    return w / w.sum()


def rebalance_rows_reference(dates):
    """每月第一个交易日的行号（第 0 行除外），日期为 YYYYMMDD 或 YYYY-MM-DD。"""
    rows = []
    for i in range(1, len(dates)):
        a = dates[i - 1].replace("-", "")[:6]
        b = dates[i].replace("-", "")[:6]
        if a != b:
            rows.append(i)
    return rows
