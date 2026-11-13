"""分组回测的截面算子与权重构造。

面板一律是 (T, N) 的 float64 数组，行是交易日、列是股票，NaN 表示缺失。
所有函数都不修改传入的数组，结果形状与输入一致。
"""

from __future__ import annotations

import numpy as np


def cs_demean(x):
    """每行减去该行有效值的均值；NaN 与整行 NaN 保持 NaN。"""
    x = np.asarray(x, dtype=np.float64)
    valid = ~np.isnan(x)
    cnt = valid.sum(axis=1)
    total = np.where(valid, x, 0.0).sum(axis=1)
    mean = np.divide(total, cnt, out=np.full(x.shape[0], np.nan), where=cnt > 0)
    return np.where(valid, x - mean[:, None], np.nan)


def group_demean(x, groups):
    """行内按组去均值；groups < 0 或 x 缺失处为 NaN。

    groups 形状为 (T, N) 或 (N,)，后者对每一行相同。
    """
    x = np.asarray(x, dtype=np.float64)
    g = np.asarray(groups)
    if g.ndim == 1:
        g = np.broadcast_to(g, x.shape)
    if g.shape != x.shape:
        raise ValueError("groups 形状必须是 (T, N) 或 (N,)")
    if len(x) > 32:
        out = np.empty_like(x)
        for first in range(0, len(x), 32):
            s = slice(first, first + 32)
            out[s] = group_demean(x[s], g[s])
        return out
    valid = (~np.isnan(x)) & (g >= 0)
    out = np.full(x.shape, np.nan)
    if not valid.any():
        return out
    # 把（行，组）编成一个号，一次分组求和。
    width = int(g[valid].max()) + 1
    rows = np.nonzero(valid)[0]
    key = rows * width + g[valid].astype(np.int64)
    size = x.shape[0] * width
    total = np.bincount(key, weights=x[valid], minlength=size)
    count = np.bincount(key, minlength=size)
    out[valid] = x[valid] - total[key] / count[key]
    return out


def cap_buckets(logcap, n=10):
    """每行按 logcap 升序稳定排名，桶号为 r*n//c；NaN 为 -1。"""
    lc = np.asarray(logcap, dtype=np.float64)
    out = np.full(lc.shape, -1, dtype=np.int64)
    for i in range(lc.shape[0]):
        cols = np.flatnonzero(~np.isnan(lc[i]))
        if cols.size == 0:
            continue
        order = cols[np.argsort(lc[i, cols], kind="stable")]
        out[i, order] = np.arange(order.size) * n // order.size
    return out


def vector_neut(x, y):
    """逐行用带截距的 y 回归 x，返回残差；共同有效值少于 2 个时整行 NaN。"""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("x 与 y 形状必须一致")
    out = np.full(x.shape, np.nan)
    valid = np.isfinite(x) & np.isfinite(y)
    for i in range(x.shape[0]):
        m = valid[i]
        if m.sum() < 2:
            continue
        xv = x[i, m]
        yv = y[i, m]
        cx = xv - xv.mean()
        cy = yv - yv.mean()
        den = np.dot(cy, cy)
        beta = np.dot(cx, cy) / den if den != 0 else 0.0
        r = cx - beta * cy
        out[i, m] = np.where(np.isfinite(r), r, np.nan)
    return out


def normalize(x):
    """每行先去均值再除以绝对值之和，使每行绝对值之和为 1。"""
    d = cs_demean(x)
    valid = ~np.isnan(d)
    cnt = valid.sum(axis=1)
    total = np.where(valid, np.abs(d), 0.0).sum(axis=1)
    out = np.full(d.shape, np.nan)
    ok = (cnt >= 2) & (total > 0)
    out[ok] = d[ok] / total[ok, None]
    return out


def decay_linear(w, n):
    """线性衰减的加权移动平均；NaN 当 0，但 w 为 NaN 处输出仍为 NaN。"""
    w = np.asarray(w, dtype=np.float64)
    if n <= 1:
        return w.copy()
    t = w.shape[0]
    w0 = np.where(np.isnan(w), 0.0, w)
    out = np.zeros_like(w0)
    for k in range(min(n, t)):
        out[k:] += (n - k) * w0[: t - k]
    rows = np.arange(t)
    kk = np.minimum(n - 1, rows)
    den = (kk + 1) * n - kk * (kk + 1) / 2
    out /= den[:, None]
    out[np.isnan(w)] = np.nan
    return out


def build_weights(
    factor,
    universe,
    *,
    industry=None,
    logcap=None,
    group_neutralize=None,
    n_cap_buckets=10,
    vector=None,
    decay=0,
):
    """按固定顺序构造池内权重：全市场中性化、抽池、池内去均值、归一化、衰减。"""
    alpha = build_alpha(
        factor,
        universe,
        industry=industry,
        logcap=logcap,
        group_neutralize=group_neutralize,
        n_cap_buckets=n_cap_buckets,
        vector=vector,
    )
    return {"alpha": alpha, "weights": decay_linear(normalize(alpha), decay)}


def build_alpha(
    factor,
    universe=None,
    *,
    industry=None,
    logcap=None,
    group_neutralize=None,
    n_cap_buckets=10,
    vector=None,
):
    """只计算分组需要的因子；不申请随后会被丢弃的归一化权重。"""
    x = np.asarray(factor, dtype=np.float64)
    if group_neutralize is None:
        pass
    elif group_neutralize == "industry":
        if industry is None:
            raise ValueError("group_neutralize='industry' 需要提供 industry")
        x = group_demean(x, industry)
    elif group_neutralize == "cap":
        if logcap is None:
            raise ValueError("group_neutralize='cap' 需要提供 logcap")
        x = group_demean(x, cap_buckets(logcap, n_cap_buckets))
    elif group_neutralize == "cap+industry":
        if industry is None or logcap is None:
            raise ValueError(
                "group_neutralize='cap+industry' 需要同时提供 industry 和 logcap"
            )
        x = vector_neut(group_demean(x, industry), logcap)
    else:
        raise ValueError(f"无法识别的 group_neutralize 取值：{group_neutralize!r}")

    if vector is not None:
        ys = vector if isinstance(vector, (list, tuple)) else [vector]
        for y in ys:
            x = vector_neut(x, y)

    if universe is not None:
        x = np.where(np.asarray(universe, dtype=bool), x, np.nan)
    return cs_demean(x)
