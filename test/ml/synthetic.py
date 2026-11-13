"""合成面板：已知答案的特征与未来收益，形状和字段与 ashare.data.Panel 一致。"""

from __future__ import annotations

import hashlib

import numpy as np


class SynthPanel:
    """dates / symbols / version / __getitem__，可直接交给 walk_forward。"""

    def __init__(self, arrays, dates, symbols, version=None):
        self._arrays = arrays
        self.dates = np.asarray(dates)
        self.symbols = np.asarray(symbols)
        self.version = (
            version
            or hashlib.sha256(
                b"".join(np.ascontiguousarray(a).tobytes() for a in arrays.values())
            ).hexdigest()
        )

    def __getitem__(self, name):
        return self._arrays[name]

    @property
    def shape(self):
        return len(self.dates), len(self.symbols)

    def replace(self, **arrays):
        return SynthPanel({**self._arrays, **arrays}, self.dates, self.symbols)


def dates(t, start=2010):
    """t 个「交易日」：每月 21 天，日期字符串 YYYYMMDD（月份和季度边界真实）。"""
    out = []
    y, m, d = start, 1, 1
    while len(out) < t:
        out.append(f"{y:04d}{m:02d}{d:02d}")
        d += 1
        if d > 21:
            d, m = 1, m + 1
            if m > 12:
                m, y = 1, y + 1
    return np.array(out)


def make(
    t=600,
    n=80,
    k=4,
    beta=None,
    noise=1.0,
    scale=0.01,
    seed=0,
    lag=1,
    horizon=1,
    missing=0.0,
    blocked=0.0,
    independent=False,
):
    """特征 f0..f{k−1} ~ N(0,1) 独立；从开盘 t+lag 到 t+lag+horizon 的对数收益
    = scale·(beta·f[t] + noise·ε)（independent=True 时 beta 置 0）。

    返回 (panel, features{名字: T×N}, beta)。blocked：随机让一部分格子开盘不可买。
    """
    rng = np.random.default_rng(seed)
    beta = np.asarray(beta if beta is not None else np.linspace(1.0, -0.5, k), float)
    if independent:
        beta = np.zeros_like(beta)
    feats = rng.standard_normal((k, t, n))
    signal = np.tensordot(beta, feats, axes=1)
    # 用「每天一个不重叠的 horizon 步收益」构造：把 horizon 日收益平均分到各天时，相邻
    # 信号日的标签会重叠，这里直接造一条对数价格，使 log P[t+lag+h] − log P[t+lag]
    # = scale·(signal[t] + noise·ε[t]) 只在 horizon = 1 时严格成立；h > 1 时近似。
    eps = rng.standard_normal((t, n))
    step = scale * (signal + noise * eps) / horizon
    logp = np.zeros((t, n))
    # 开盘价 t+lag → t+lag+1 的增量来自信号日 t
    for s in range(t - lag - 1):
        logp[s + lag + 1] = logp[s + lag] + step[s]
    open_ = 10 * np.exp(logp)
    can = np.ones((t, n), bool)
    if blocked:
        can &= rng.random((t, n)) >= blocked
    arrays = {
        "open": open_,
        "close": open_ * 1.001,
        "adj_factor": np.ones((t, n)),
        "can_buy_open": can,
        "can_sell_open": np.ones((t, n), bool),
        "is_listed": np.ones((t, n), bool),
        "is_st": np.zeros((t, n), bool),
        "listing_age": np.full((t, n), 1000, np.uint16),
        "has_quote": np.ones((t, n), bool),
        "industry": np.tile(np.arange(n) % 5, (t, 1)).astype(np.uint32),
        "total_market_cap": np.exp(rng.standard_normal((t, n)) + 20),
    }
    features = {}
    for j in range(k):
        x = feats[j].copy()
        if missing:
            x[rng.random((t, n)) < missing] = np.nan
        features[f"f{j}"] = np.ascontiguousarray(x)
    symbols = np.array([f"{i:06d}.SZ" for i in range(n)])
    return SynthPanel(arrays, dates(t), symbols), features, beta
