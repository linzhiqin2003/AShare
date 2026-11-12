"""numpy 参考内核；缺失数据以逐列有效权重归一化。"""

import numpy as np
from scipy.linalg import null_space


def exp_weights(length, half_life):
    if length < 1 or half_life <= 0 or np.isnan(half_life):
        raise ValueError("样本长度和半衰期必须为正")
    w = np.exp2(-np.arange(length - 1, -1, -1, dtype=float) / half_life)
    return w / w.sum()


def centered(returns, half_life):
    x = np.asarray(returns, dtype=float)
    if x.ndim != 2 or not len(x) or np.isinf(x).any():
        raise ValueError("returns: 需要非空 T×N 数组，缺失用 NaN")
    valid = np.isfinite(x)
    w = exp_weights(len(x), half_life)[:, None] * valid
    sums = w.sum(axis=0)
    w /= np.where(sums > 0, sums, 1)
    mean = (np.nan_to_num(x) * w).sum(axis=0)
    z = np.where(valid, x - mean, 0)
    return z, w, valid


def ew_covariance(returns, half_life=252.0, lags=0):
    """PSD Gram 估计量；仅用于 K 维因子序列，不用于股票协方差。"""
    z, w, _ = centered(returns, half_life)
    v = z * np.sqrt(w)
    result = v.T @ v
    if lags < 0 or lags >= len(z):
        raise ValueError("lags: 必须在 [0,T) 内")
    for lag in range(1, lags + 1):
        cross = v[lag:].T @ v[:-lag]
        result += (1 - lag / (lags + 1)) * (cross + cross.T)
    return (result + result.T) * 0.5


def ew_variance(returns, half_life=252.0, lags=0):
    z, w, _ = centered(returns, half_life)
    v = z * np.sqrt(w)
    result = np.sum(v * v, axis=0)
    if lags < 0 or lags >= len(z):
        raise ValueError("lags: 必须在 [0,T) 内")
    for lag in range(1, lags + 1):
        result += 2 * (1 - lag / (lags + 1)) * np.sum(v[lag:] * v[:-lag], axis=0)
    return np.maximum(result, 0)


def psd(matrix, floor=0.0):
    a = (matrix + matrix.T) * 0.5
    values, vectors = np.linalg.eigh(a)
    result = (vectors * np.maximum(values, floor)) @ vectors.T
    return (result + result.T) * 0.5


def factor_covariance(
    returns, vol_half_life=84.0, corr_half_life=504.0, vol_lags=5, corr_lags=2
):
    corr = ew_covariance(returns, corr_half_life, corr_lags)
    sd = np.sqrt(np.maximum(np.diag(corr), 1e-30))
    corr = corr / np.outer(sd, sd)
    vol = np.sqrt(ew_variance(returns, vol_half_life, vol_lags))
    return psd(corr * np.outer(vol, vol))


def cross_section(returns, exposures, market_cap, estimation, industry_indices=()):
    """最小化 Σ sqrt(cap)·残差²；行业收益满足 cap 加权和为零。"""
    r, b, cap = map(
        lambda a: np.asarray(a, dtype=float), (returns, exposures, market_cap)
    )
    est = np.asarray(estimation, dtype=bool)
    if (
        b.ndim != 2
        or r.shape != cap.shape
        or est.shape != r.shape
        or b.shape[0] != len(r)
    ):
        raise ValueError("截面输入形状不一致")
    valid = (
        est & np.isfinite(r) & np.isfinite(cap) & (cap > 0) & np.isfinite(b).all(axis=1)
    )
    if valid.sum() < 2:
        raise ValueError("估计池有效股票不足")
    x, y, c = b[valid], r[valid], cap[valid]
    weights = np.sqrt(c)
    weights /= weights.mean()
    transform = np.eye(b.shape[1])
    if len(industry_indices):
        constraint = np.zeros(b.shape[1])
        constraint[np.asarray(industry_indices)] = c @ x[:, industry_indices]
        if np.linalg.norm(constraint):
            transform = null_space(constraint[None, :])
    design = (x @ transform) * np.sqrt(weights[:, None])
    coef = transform @ np.linalg.lstsq(design, y * np.sqrt(weights), rcond=1e-12)[0]
    residual = r - b @ coef
    residual[~np.isfinite(r)] = np.nan
    return coef, residual


def cross_section_batch(
    returns, exposures, market_cap, estimation, industry_indices=()
):
    b = np.asarray(exposures)
    out = [
        cross_section(
            r,
            b if b.ndim == 2 else b[t],
            market_cap[t],
            estimation[t],
            industry_indices,
        )
        for t, r in enumerate(returns)
    ]
    return np.array([v[0] for v in out]), np.array([v[1] for v in out])


def rolling_bias(realized, predicted, window=252):
    """预测量为标准差；返回样本标准差，缺失窗口保留 NaN。"""
    r, p = np.asarray(realized, dtype=float), np.asarray(predicted, dtype=float)
    if r.shape != p.shape or r.ndim != 2 or window < 2 or len(r) < window:
        raise ValueError("偏差统计输入形状或窗口错误")
    z = np.divide(r, p, out=np.full_like(r, np.nan), where=p > 0)
    valid = np.isfinite(z)
    x = np.where(valid, z, 0)

    def roll(v):
        c = np.vstack((np.zeros((1, v.shape[1])), np.cumsum(v, axis=0)))
        return c[window:] - c[:-window]

    count = roll(valid.astype(float))
    total, square = roll(x), roll(x * x)
    var = (square - total * total / np.maximum(count, 1)) / np.maximum(count - 1, 1)
    return np.where(count == window, np.sqrt(np.maximum(var, 0)), np.nan)


def cross_section_gram(returns, exposures, market_cap, estimation, industry_indices=()):
    """BLAS 正规方程基线；秩亏时回到独立的白化 SVD 参考。"""
    r, b, cap = map(
        lambda a: np.asarray(a, dtype=float), (returns, exposures, market_cap)
    )
    if b.ndim != 2 or r.ndim != 1 or cap.shape != r.shape or b.shape[0] != len(r):
        raise ValueError("截面输入形状不一致")
    valid = (
        np.asarray(estimation, bool)
        & np.isfinite(r)
        & np.isfinite(cap)
        & (cap > 0)
        & np.isfinite(b).all(axis=1)
    )
    if valid.sum() < 2:
        raise ValueError("估计池有效股票不足")
    x, y, c = b[valid], r[valid], cap[valid]
    weights = np.sqrt(c)
    weights /= weights.mean()
    constraint = np.zeros(b.shape[1])
    if len(industry_indices):
        constraint[np.asarray(industry_indices)] = c @ x[:, industry_indices]
    # 估计池没有某行业、或冷启动描述子全零时，显式剔除零列再解。
    columns = np.flatnonzero(np.any(x != 0, axis=0))
    if not len(columns):
        return np.zeros(b.shape[1]), r.copy()
    x, constraint = x[:, columns], constraint[columns]
    gram = (x.T * weights) @ x
    rhs = x.T @ (weights * y)
    length = np.linalg.norm(constraint)
    if length > 0:
        constraint /= length
        system = np.block(
            [[gram, constraint[:, None]], [constraint[None, :], np.zeros((1, 1))]]
        )
        target = np.r_[rhs, 0]
    else:
        system, target = gram, rhs
    try:
        if np.linalg.cond(system) > 1e12:
            return cross_section(r, b, cap, estimation, industry_indices)
        coef = np.zeros(b.shape[1])
        coef[columns] = np.linalg.solve(system, target)[: len(columns)]
    except np.linalg.LinAlgError:
        return cross_section(r, b, cap, estimation, industry_indices)
    return coef, r - b @ coef
