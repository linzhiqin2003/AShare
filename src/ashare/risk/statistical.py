"""矩阵自由的样本收缩和 PCA；不生成股票协方差矩阵。"""

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

from .kernels import centered, exp_weights
from .model import RiskModel


def shrinkage_components(returns, half_life=np.inf, target="identity", shrinkage=None):
    """返回 U、对角项及强度，使协方差等于 U U' + diag(d)。

    完整等权样本的 identity 强度使用 Ledoit–Wolf 公式。指数权重和缺失
    使用加权影响函数推广；常相关目标将平均相关系数裁剪到 [0,1]。
    """
    z, w, valid = centered(returns, half_life)
    p = exp_weights(len(z), half_life)
    x = z * np.sqrt(
        np.divide(w, p[:, None], out=np.zeros_like(w), where=p[:, None] > 0)
    )
    u = x.T * np.sqrt(p)
    diagonal = np.sum(u * u, axis=1)
    n = len(diagonal)
    floor = max(
        float(np.median(diagonal[diagonal > 0])) if (diagonal > 0).any() else 0.02**2,
        1e-12,
    )
    missing = valid.sum(axis=0) < 2
    diagonal[missing] = floor
    s = np.sqrt(np.maximum(diagonal, 1e-30))
    invs = 1 / s
    row_corr = u @ (u.T @ invs) * invs
    corr = float(np.clip((row_corr.sum() - n) / max(n * (n - 1), 1), 0, 1))
    if target == "identity":
        target_diag = np.full(n, diagonal.mean())
        target_rank = np.zeros(n)
    elif target == "constant_correlation":
        target_diag = (1 - corr) * diagonal
        target_rank = np.sqrt(corr) * s
    else:
        raise ValueError("target: identity 或 constant_correlation")
    if shrinkage is None:
        # 时间 Gram 按块扫描，既不展开 N×N，也不常驻 T×T。
        frob = 0.0
        xsx = np.empty(len(z))
        for start in range(0, len(z), 128):
            stop = min(len(z), start + 128)
            gram = u.T @ u[:, start:stop]
            frob += float(np.sum(gram * gram))
            xsx[start:stop] = np.divide(
                np.sum(gram * gram, axis=0),
                p[start:stop],
                out=np.zeros(stop - start),
                where=p[start:stop] > 0,
            )
        norm2 = np.sum(x * x, axis=1)
        pi = float(np.dot(p * p, norm2 * norm2 - 2 * xsx + frob))
        if target == "identity":
            # sklearn 对 scaled-identity 目标使用固定目标渐近公式。
            rho = 0.0
        else:
            v = x * x - diagonal
            # dT 的对角影响及 sqrt(Sii Sjj) 的一阶导数。
            # 平均相关系数的导数也计入有限 N 的自定义修正。
            xs = x @ s
            sx = u @ (u.T @ s)
            off_inner = (
                xs * xs
                - np.sum(x * x * diagonal, axis=1)
                - (s @ sx - diagonal @ diagonal)
            )
            standardized = x * invs
            dr = (
                (standardized.sum(axis=1) ** 2 - np.sum(standardized**2, axis=1))
                - (row_corr.sum() - n)
                - (v / np.maximum(diagonal, 1e-30)) @ (row_corr - 1)
            ) / max(n * (n - 1), 1)
            term = (x * v * invs).sum(axis=1) * xs - (v * invs) @ sx
            rho_per = (
                np.sum(v * v, axis=1)
                + corr * (term - np.sum(v * v, axis=1))
                + dr * off_inner
            )
            rho = float(np.dot(p * p, rho_per))
        target_norm = float(
            target_diag @ target_diag
            + 2 * np.dot(target_diag, target_rank**2)
            + np.dot(target_rank, target_rank) ** 2
        )
        cross = float(diagonal @ target_diag + np.sum((u.T @ target_rank) ** 2))
        delta = max(frob + target_norm - 2 * cross, 0)
        shrinkage = float(np.clip((pi - rho) / delta, 0, 1)) if delta > 1e-30 else 1.0
    if not np.isfinite(shrinkage) or not 0 <= shrinkage <= 1:
        raise ValueError("shrinkage: 必须在 [0,1]")
    d = shrinkage * target_diag
    d[missing] += (1 - shrinkage) * floor
    u = np.column_stack((np.sqrt(1 - shrinkage) * u, np.sqrt(shrinkage) * target_rank))
    return (
        u,
        d,
        {
            "shrinkage": shrinkage,
            "target": target,
            "average_correlation": corr,
            "missing_policy": "weighted-mean imputation with per-column weight normalization",
        },
    )


def statistical_model(
    returns,
    symbols=None,
    *,
    half_life=252.0,
    target="identity",
    components=45,
    shrinkage=None,
    date="",
    min_history=60,
):
    """短史向截面方差先验收缩；None 主成分保留 T+1 因子，仅适合小样本研究。"""
    if not isinstance(min_history, int) or min_history < 2:
        raise ValueError("min_history: 必须至少为 2")
    u, d, meta = shrinkage_components(returns, half_life, target, shrinkage)
    count = np.isfinite(returns).sum(axis=0)
    threshold = min(min_history, len(returns))
    reliability = np.minimum(count / max(threshold, 1), 1.0)
    reliability[count < 2] = 0.0
    diagonal = np.sum(u * u, axis=1) + d
    reliable = (count >= threshold) & (diagonal > 0)
    prior = float(np.median(diagonal[reliable])) if reliable.any() else 0.02**2
    u *= np.sqrt(reliability[:, None])
    d = reliability * d + (1 - reliability) * max(prior, 1e-12)
    meta.update(
        min_history=min_history, short_history_count=int((reliability < 1).sum())
    )
    n = len(d)
    symbols = tuple(str(i) for i in range(n)) if symbols is None else tuple(symbols)
    if components is None:
        b = u
        f = np.eye(u.shape[1])
    else:
        if not isinstance(components, int) or not 1 <= components < n:
            raise ValueError("components: 必须是 [1,N) 内整数；None 表示完整收缩模型")
        op = LinearOperator(
            (n, n),
            matvec=lambda v: u @ (u.T @ v) + d * v,
            matmat=lambda v: u @ (u.T @ v) + d[:, None] * v,
        )
        # 固定起点、固定排序和特征向量符号，确保追加前缀可重复。
        values, vectors = eigsh(
            op, k=components, which="LA", v0=np.linspace(1, 2, n), tol=1e-11
        )
        order = np.argsort(values)[::-1]
        values, b = np.maximum(values[order], 0), vectors[:, order]
        signs = np.sign(b[np.argmax(np.abs(b), axis=0), np.arange(components)])
        b *= signs
        f = np.diag(values)
        d = np.maximum(
            np.sum(u * u, axis=1) + d - np.sum(b * b * values, axis=1), 1e-12
        )
    return RiskModel(
        symbols, b, tuple(f"pc_{i}" for i in range(b.shape[1])), f, d, date, meta
    )
