"""合成行情：已知答案的量价面板，给例子、测试和模拟检验用。

价格按日内、隔夜两段独立的随机游走生成：开盘 = 前收 × e^{隔夜}，收盘 = 开盘 × e^{日内}。
没有埋信号时，t+1 开盘以后的收益与 t 日收盘及以前的一切独立——任何量价因子都没有
预测力（纯噪声）。``strength > 0`` 时埋一个已知信号：t 日收盘后算出的
「过去 5 日收益的相反数」（自编的短期反转）的截面标准分乘以 strength 加到 t+1 日内收益上，
即 ``negate(close / ts_delay(close, periods=5))`` 能预测 t+1 开盘到 t+2 开盘的收益。

```python
from ashare.factor.mining.synthetic import market
panel = market(t=500, n=100, seed=0)          # 纯噪声
panel = market(t=500, n=100, strength=0.004)  # 埋信号
```
"""

from __future__ import annotations

import hashlib

import numpy as np

from ..expression import call, field


def planted_expression():
    """埋进去的信号的表达式（在默认搜索空间里）。"""
    close = field("close")
    return call("negate", close / call("ts_delay", close, periods=5))


def dates(t, start=2012):
    """t 个「交易日」YYYYMMDD：每年 12 个月、每月 21 天（年份边界真实）。"""
    out, y, m, d = [], start, 1, 1
    while len(out) < t:
        out.append(f"{y:04d}{m:02d}{d:02d}")
        d += 1
        if d > 21:
            d, m = 1, m + 1
            if m > 12:
                m, y = 1, y + 1
    return np.array(out)


class SyntheticPanel:
    """合成面板：dates、symbols、version、按字段名取数组（与 `ashare.data` 的面板同形）。"""

    def __init__(self, arrays, dates, symbols, version=None):
        self._arrays = arrays
        self.dates = np.asarray(dates)
        self.symbols = np.asarray(symbols)
        self.version = (
            version
            or hashlib.sha256(
                b"".join(
                    np.ascontiguousarray(arrays[k]).tobytes() for k in sorted(arrays)
                )
            ).hexdigest()
        )

    def __getitem__(self, name):
        return self._arrays[name]

    def fields(self):
        return tuple(self._arrays)

    @property
    def shape(self):
        return len(self.dates), len(self.symbols)

    def replace(self, **arrays):
        """换掉部分字段，版本按内容重算。"""
        return SyntheticPanel({**self._arrays, **arrays}, self.dates, self.symbols)


def market(t=500, n=100, *, seed=0, strength=0.0, vol=0.02, start=2012, blocked=0.0):
    """生成 T×N 的合成量价面板。

    strength：埋入信号的强度（每标准差的日收益，0 为纯噪声）；vol：日内、隔夜各自的
    日波动；blocked：随机让一部分格子开盘不可买、不可卖（模拟停牌与涨跌停）。
    """
    rng = np.random.default_rng(seed)
    night = rng.standard_normal((t, n)) * vol * 0.5
    day = rng.standard_normal((t, n)) * vol
    open_ = np.empty((t, n))
    close = np.empty((t, n))
    level = 10 * np.exp(rng.standard_normal(n) * 0.5)
    for s in range(t):
        prev = level if s == 0 else close[s - 1]
        open_[s] = prev * np.exp(night[s])
        drift = 0.0
        if strength and s >= 6:
            # t = s − 1 日收盘后可知的信号，作用于 s 日日内收益
            f = -(close[s - 1] / close[s - 6])
            z = (f - f.mean()) / f.std()
            drift = strength * z
        close[s] = open_[s] * np.exp(day[s] + drift)
    spread = np.abs(rng.standard_normal((t, n))) * vol * 0.3
    high = np.maximum(open_, close) * np.exp(spread)
    low = np.minimum(open_, close) * np.exp(-spread)
    vwap = (open_ + close + high + low) / 4
    shares = np.exp(rng.standard_normal(n) * 0.5 + 19)
    volume = np.round(shares * np.exp(rng.standard_normal((t, n)) * 0.4 - 4.5))
    can = np.ones((t, n), bool)
    if blocked:
        can &= rng.random((t, n)) >= blocked
    arrays = {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "vwap": vwap,
        "volume": volume,
        "amount": volume * vwap,
        "turnover": volume / shares,
        "adj_factor": np.ones((t, n)),
        "previous_close": np.vstack([level[None], close[:-1]]),
        "total_market_cap": close * shares,
        "can_buy_open": can,
        "can_sell_open": can.copy(),
        "split": np.full((t, n), np.nan),
        "cash_dividend": np.full((t, n), np.nan),
        "delist": np.zeros((t, n), bool),
        "is_listed": np.ones((t, n), bool),
        "is_st": np.zeros((t, n), bool),
        "listing_age": np.tile(
            np.arange(1000, 1000 + t, dtype=np.int64)[:, None], (1, n)
        ),
        "has_quote": np.ones((t, n), bool),
    }
    symbols = np.array([f"{600000 + i:06d}.SH" for i in range(n)])
    return SyntheticPanel(arrays, dates(t, start), symbols)
