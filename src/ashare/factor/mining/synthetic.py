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
panel = market(t=500, n=100, frictions=Frictions.typical())  # 带停牌、退市、ST、上市
```

摩擦（`Frictions`，默认全关，此时与不带摩擦的旧版本逐位相同）：停牌期间没有行情
（价格为缺失、不可买卖），潜在价格照常游走并带一个漂移，复牌时一次性体现；退市前先戴帽
（ST）、再停牌一段，然后除牌（`delist` 标记在第一个不上市的行，之后不上市、没有行情）；
ST 期间带漂移；一部分股票在样本中途上市（之前不上市、`listing_age` 从上市日起算）。
这些正是「股票池等权、收盘到收盘」的旧基准与实际成交组合口径不同的地方。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

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


@dataclass(frozen=True)
class Frictions:
    """合成行情的摩擦参数（概率按每只股票每天、比例按股票数）。

    suspend：每天开始停牌的概率；suspend_days：停牌平均天数（几何分布）；suspend_drift：
    停牌期间潜在价格每天的对数漂移（复牌时一次性体现）。delist：样本内退市的股票比例；
    delist_st、delist_halt：退市前戴帽、停牌的天数；st：另外戴帽一段的股票比例；st_days：
    戴帽天数；st_drift：戴帽期间每天的对数漂移。ipo：样本中途上市的股票比例（上市日在
    前一半里均匀）。"""

    suspend: float = 0.0
    suspend_days: float = 10.0
    suspend_drift: float = 0.0
    delist: float = 0.0
    delist_st: int = 60
    delist_halt: int = 20
    st: float = 0.0
    st_days: int = 120
    st_drift: float = 0.0
    ipo: float = 0.0

    @classmethod
    def typical(cls, **overrides):
        """量级与 A 股相近的一组：停牌约 0.2%/天、平均 10 天、停牌期每天 −0.3%；5 年左右
        约 3% 的股票退市；8% 的股票戴帽一段、期间每天 −0.1%；20% 的股票中途上市。"""
        base = dict(suspend=0.002, suspend_drift=-0.003, delist=0.03, st=0.08,
                    st_drift=-0.001, ipo=0.2)  # fmt: skip
        return cls(**{**base, **overrides})

    @property
    def active(self):
        return bool(self.suspend or self.delist or self.st or self.ipo)


def _frictions(fr, t, n, seed):
    """停牌、ST、上市、退市的掩码与潜在价格的额外漂移（与价格的随机数分开抽）。"""
    rng = np.random.default_rng([seed, 7919])
    listed = np.ones((t, n), bool)
    halt = np.zeros((t, n), bool)
    st = np.zeros((t, n), bool)
    delist = np.zeros((t, n), bool)
    first = np.full(n, -1000, np.int64)  # 上市行（样本开始前已上市的记作 −1000）
    if fr.ipo:
        late = rng.random(n) < fr.ipo
        first[late] = rng.integers(1, max(2, t // 2), late.sum())
        for j in np.flatnonzero(late):
            listed[: first[j], j] = False
    if fr.delist:
        gone = np.flatnonzero(rng.random(n) < fr.delist)
        for j in gone:
            lo = max(first[j], 0) + fr.delist_st + fr.delist_halt + 20
            if lo >= t - 1:
                continue
            d = int(rng.integers(lo, t))
            delist[d, j] = True
            listed[d:, j] = False
            halt[d - fr.delist_halt : d, j] = True
            st[d - fr.delist_halt - fr.delist_st : d - fr.delist_halt, j] = True
    if fr.st:
        for j in np.flatnonzero(rng.random(n) < fr.st):
            a = int(rng.integers(max(first[j], 0), t))
            st[a : a + fr.st_days, j] = True
    if fr.suspend:
        starts = rng.random((t, n)) < fr.suspend
        p = 1.0 / max(fr.suspend_days, 1.0)
        for s, j in zip(*np.nonzero(starts), strict=True):
            halt[s : s + int(rng.geometric(p)), j] = True
    halt &= listed
    st &= listed
    extra = np.where(halt, fr.suspend_drift, 0.0) + np.where(st, fr.st_drift, 0.0)
    age = np.arange(t)[:, None] - first[None, :]
    return listed, halt, st, delist, np.where(listed, age, 0).astype(np.int64), extra


def market(
    t=500, n=100, *, seed=0, strength=0.0, vol=0.02, start=2012, blocked=0.0,
    frictions=None,
):  # fmt: skip
    """生成 T×N 的合成量价面板。

    strength：埋入信号的强度（每标准差的日收益，0 为纯噪声）；vol：日内、隔夜各自的
    日波动；blocked：随机让一部分格子开盘不可买、不可卖（模拟涨跌停）；frictions：
    `Frictions`（停牌、退市、ST、中途上市），默认没有。
    """
    fr = frictions or Frictions()
    rng = np.random.default_rng(seed)
    night = rng.standard_normal((t, n)) * vol * 0.5
    day = rng.standard_normal((t, n)) * vol
    open_ = np.empty((t, n))
    close = np.empty((t, n))
    level = 10 * np.exp(rng.standard_normal(n) * 0.5)
    if fr.active:
        listed, halt, st, delist, age, extra = _frictions(fr, t, n, seed)
    for s in range(t):
        prev = level if s == 0 else close[s - 1]
        open_[s] = prev * np.exp(night[s])
        drift = 0.0
        if strength and s >= 6:
            # t = s − 1 日收盘后可知的信号，作用于 s 日日内收益
            f = -(close[s - 1] / close[s - 6])
            z = (f - f.mean()) / f.std()
            drift = strength * z
        if fr.active:
            drift = drift + extra[s]
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
    if fr.active:
        quote = listed & ~halt
        prev = np.vstack([level[None], close[:-1]])
        for k in ("open", "high", "low", "close", "vwap", "amount", "turnover",
                  "total_market_cap"):  # fmt: skip
            arrays[k] = np.where(quote, arrays[k], np.nan)
        arrays["volume"] = np.where(quote, volume, np.nan)
        arrays["amount"] = np.where(quote, arrays["amount"], np.nan)
        # 前收：最近一个有行情的收盘（上市前缺失）
        last = np.where(quote, close, np.nan)
        filled = last.copy()
        for s in range(1, t):
            filled[s] = np.where(np.isfinite(filled[s]), filled[s], filled[s - 1])
        prev = np.vstack([np.full((1, n), np.nan), filled[:-1]])
        arrays["previous_close"] = np.where(listed, prev, np.nan)
        arrays["can_buy_open"] = can & quote
        arrays["can_sell_open"] = can & quote
        arrays["delist"] = delist
        arrays["is_listed"] = listed
        arrays["is_st"] = st
        arrays["listing_age"] = age
        arrays["has_quote"] = quote
    # 涨跌停价（±20%，合成行情里很少触及）与停牌标记：开盘以外的成交价建行情时要用
    arrays["upper_limit"] = np.round(arrays["previous_close"] * 1.2, 2)
    arrays["lower_limit"] = np.round(arrays["previous_close"] * 0.8, 2)
    arrays["is_suspended"] = ~arrays["has_quote"] & arrays["is_listed"]
    symbols = np.array([f"{600000 + i:06d}.SH" for i in range(n)])
    return SyntheticPanel(arrays, dates(t, start), symbols)
