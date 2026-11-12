"""合成数据：带停牌、除权、行业变更、月度基准快照的小面板，以及随机风险历史。"""

from __future__ import annotations

import numpy as np

from ashare.analysis import FactorHistory


class FakePanel:
    def __init__(self, dates, symbols, fields, version="synthetic"):
        self.dates = np.asarray(dates)
        self.symbols = tuple(symbols)
        self.fields = fields
        self.version = version

    def __getitem__(self, name):
        return self.fields[name]


def make_panel(t=120, n=12, sectors=4, seed=0, suspend=True, monthly=20):
    rng = np.random.default_rng(seed)
    dates = np.array(
        [
            str(np.datetime64("2019-01-01") + np.timedelta64(i, "D")).replace("-", "")
            for i in range(t)
        ]
    )
    symbols = tuple(f"{600000 + i:06d}.SH" for i in range(n))
    r = rng.normal(0.0005, 0.02, (t, n))
    close = 10 * np.exp(np.cumsum(np.log1p(r), axis=0))
    adj = np.ones((t, n))
    # 除权：第 30 日起某股复权因子跳升（价格同步下跌），总收益不受影响。
    adj[30:, 1] *= 1.25
    close[30:, 1] /= 1.25
    if suspend:
        close[40:43, 2] = np.nan  # 停牌三天
    opn = np.vstack([close[:1], close[:-1]]) * (1 + rng.normal(0, 0.005, (t, n)))
    opn = np.where(np.isfinite(close), opn, np.nan)
    volume = rng.uniform(1e5, 1e6, (t, n))
    volume[40:43, 2] = 0
    amount = volume * np.nan_to_num(close, nan=10)
    industry = np.tile(801010 + 10 * (np.arange(n) % sectors), (t, 1)).astype(float)
    industry[60:, 3] = 801010 + 10 * ((3 + 1) % sectors)  # 行业变更
    cap = rng.uniform(1e9, 1e10, n) * np.ones((t, 1))
    weight = np.zeros((t, n))
    for start in range(0, t, monthly):
        snap = rng.uniform(0.5, 1.5, n)
        snap[n // 2 :] = 0  # 基准只含前一半
        weight[start : start + monthly] = snap / snap.sum()
    fields = {
        "close": close,
        "open": opn,
        "high": np.fmax(opn, close),
        "low": np.fmin(opn, close),
        "upper_limit": np.nan_to_num(close, nan=10) * 1.5,
        "lower_limit": np.nan_to_num(close, nan=10) * 0.5,
        "adj_factor": adj,
        "volume": volume,
        "amount": amount,
        "industry": industry,
        "weight_bench": weight,
        "total_market_cap": cap,
        "float_market_cap": cap * 0.8,
        "turnover": rng.uniform(0.005, 0.03, (t, n)),
        "pb": rng.uniform(0.5, 5, (t, n)),
        "pe_ttm": rng.uniform(5, 50, (t, n)),
        "is_listed": np.ones((t, n), bool),
        "is_suspended": ~np.isfinite(close),
        "has_quote": np.isfinite(close),
        "split": np.ones((t, n)),
        "cash_dividend": np.zeros((t, n)),
        "delist": np.zeros((t, n), bool),
    }
    return FakePanel(dates, symbols, fields)


def random_weights(panel, seed=1, long_only=True, cash=0.05, every=10):
    """收盘后权重：每 every 天换一次，期间按收益漂移；留少量现金。"""
    rng = np.random.default_rng(seed)
    t, n = len(panel.dates), len(panel.symbols)
    from ashare.analysis import total_returns

    r = np.nan_to_num(total_returns(panel))
    w = np.zeros((t, n))
    current = np.zeros(n)
    cash_w = 1.0
    for day in range(t):
        if day:
            grown = current * (1 + r[day])
            total = grown.sum() + cash_w
            current, cash_w = grown / total, cash_w / total
        if day % every == 0:
            raw = rng.uniform(0, 1, n) if long_only else rng.normal(0, 1, n)
            current = raw / np.abs(raw).sum() * (1 - cash)
            cash_w = 1 - current.sum()
        w[day] = current
    return w


def random_history(panel, k_style=3, seed=2, drop=None):
    rng = np.random.default_rng(seed)
    t, n = len(panel.dates), len(panel.symbols)
    sectors = np.unique(panel["industry"])
    factors = (
        ("country",)
        + tuple(f"industry_{i}" for i in range(len(sectors)))
        + tuple(f"style_{i}" for i in range(k_style))
    )
    k = len(factors)
    b = np.zeros((t, n, k))
    b[:, :, 0] = 1
    for j, s in enumerate(sectors):
        b[:, :, 1 + j] = panel["industry"] == s
    b[:, :, 1 + len(sectors) :] = rng.normal(0, 1, (t, n, k_style))
    f = rng.normal(0, 0.005, (t, k))
    f[0] = np.nan
    a = rng.normal(0, 0.01, (t, k, k))
    cov = np.einsum("tij,tkj->tik", a, a) + 1e-6 * np.eye(k)
    d = rng.uniform(1e-4, 4e-4, (t, n))
    if drop is not None:
        b[drop] = np.nan
    return FactorHistory(panel.dates, panel.symbols, factors, b, f, cov, d)
