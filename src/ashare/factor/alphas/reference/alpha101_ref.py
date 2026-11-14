"""WorldQuant Alpha101 的独立参考实现：直接用 NumPy / pandas 照原文公式一层一层写。

不经过 ``Expr``、``dsl``、``ops`` 和 ``panelops``；算子在 ``alpha101_ref_ops``，口径照
``docs/CONVENTIONS.md`` 与负责人裁决（2026-11-13）。每个函数上方的注释是
``docs/factors/sources/alpha101.jsonl`` 的 ``formula`` 字段原文（测试逐字核对）。

写法约定：

- 照公式的结构逐层写，不化简、不合并、不调换运算顺序；同一个子式在公式里出现两次时只算
  一次（同样的输入、同样的运算，结果逐位相同）。
- ``x / y`` 写成 ``div(x, y)``（非有限结果为 NaN）；``x ^ y`` 写成 ``power(x, y)``；比较写成
  ``lt/gt/eq``（结果 1.0 / 0.0，遇 NaN 为 NaN）；``||`` 写成 ``or_``；``c ? a : b`` 写成
  ``where(c, a, b)``；``+ - *`` 直接用 NumPy 运算符。
- ``max(x, y)`` / ``min(x, y)`` 第二个参数是数字时是时序 ``ts_max/ts_min``，两个都是序列时是
  逐元素 ``maximum/minimum``（Q-A-01）。
- ``IndClass.sector / industry / subindustry`` 三个层级都用申万一级 ``industry``（D-06）。
- ``adv{d}`` 是成交额的 d 日均值（D-05）。
- ``ts_argmax/ts_argmin`` 的读法由参数 ``argmax`` 选：``"days_ago"``（默认，读法 A）或
  ``"position"``（读法 B），见 D-13；只影响 001、057、060、096、098、100。

输入是 ``reference_frames`` 给出的 DataFrame 字典（``docs/REFERENCE_INPUTS.md``）。
"""

from functools import partial

import numpy as np
import pandas as pd

from .alpha101_ref_ops import (
    abs_,
    adv,
    clean,
    correlation,
    covariance,
    decay_linear,
    delay,
    delta,
    div,
    eq,
    indneutralize,
    log,
    lt,
    maximum,
    minimum,
    or_,
    power,
    rank,
    scale,
    sign,
    signedpower,
    stddev,
    ts_argmax,
    ts_argmin,
    ts_max,
    ts_min,
    ts_product,
    ts_rank,
    ts_sum,
    where,
)

DEFAULT_ARGMAX = "days_ago"

# 受 D-13 两种读法影响的条目。
ARGMAX_DEPENDENT = (
    "alpha101_001",
    "alpha101_057",
    "alpha101_060",
    "alpha101_096",
    "alpha101_098",
    "alpha101_100",
)


class _Fields:
    """``data`` 字典的 NumPy 视图；``adv(d)`` 按需计算并缓存。"""

    def __init__(self, data):
        def arr(name):
            return np.asarray(data[name], dtype=float)

        self.open = arr("open")
        self.high = arr("high")
        self.low = arr("low")
        self.close = arr("close")
        self.volume = arr("volume")
        self.vwap = arr("vwap")
        self.returns = arr("returns")
        self.amount = arr("amount")
        self._data = data
        self._adv = {}

    @property
    def cap(self):
        return np.asarray(self._data["cap"], dtype=float)

    @property
    def industry(self):
        """IndClass 的三个层级都退到申万一级（D-06）。"""
        return np.asarray(self._data["industry"], dtype=float)

    def adv(self, d):
        if d not in self._adv:
            self._adv[d] = adv(self.amount, d)
        return self._adv[d]


def _fields(data):
    return data if isinstance(data, _Fields) else _Fields(data)


def _frame(values, data):
    like = data._data["close"] if isinstance(data, _Fields) else data["close"]
    return pd.DataFrame(clean(values), index=like.index, columns=like.columns)


# 原文：(rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)
def alpha101_001(data, argmax=DEFAULT_ARGMAX, **params):
    f = _fields(data)
    inner = where(lt(f.returns, 0), stddev(f.returns, 20), f.close)
    out = rank(ts_argmax(signedpower(inner, 2.0), 5, how=argmax)) - 0.5
    return _frame(out, data)


# 原文：(-1 * correlation(rank(delta(log(volume), 2)), rank(((close - open) / open)), 6))
def alpha101_002(data, **params):
    f = _fields(data)
    out = -1 * correlation(
        rank(delta(log(f.volume), 2)), rank(div(f.close - f.open, f.open)), 6
    )
    return _frame(out, data)


# 原文：(-1 * correlation(rank(open), rank(volume), 10))
def alpha101_003(data, **params):
    f = _fields(data)
    out = -1 * correlation(rank(f.open), rank(f.volume), 10)
    return _frame(out, data)


# 原文：(-1 * Ts_Rank(rank(low), 9))
def alpha101_004(data, **params):
    f = _fields(data)
    out = -1 * ts_rank(rank(f.low), 9)
    return _frame(out, data)


# 原文：(rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))
def alpha101_005(data, **params):
    f = _fields(data)
    out = rank(f.open - div(ts_sum(f.vwap, 10), 10)) * (
        -1 * abs_(rank(f.close - f.vwap))
    )
    return _frame(out, data)


# 原文：(-1 * correlation(open, volume, 10))
def alpha101_006(data, **params):
    f = _fields(data)
    out = -1 * correlation(f.open, f.volume, 10)
    return _frame(out, data)


# 原文：((adv20 < volume) ? ((-1 * ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : (-1 * 1))
def alpha101_007(data, **params):
    f = _fields(data)
    out = where(
        lt(f.adv(20), f.volume),
        (-1 * ts_rank(abs_(delta(f.close, 7)), 60)) * sign(delta(f.close, 7)),
        -1 * 1,
    )
    return _frame(out, data)


# 原文：(-1 * rank(((sum(open, 5) * sum(returns, 5)) - delay((sum(open, 5) * sum(returns, 5)), 10))))
def alpha101_008(data, **params):
    f = _fields(data)
    x = ts_sum(f.open, 5) * ts_sum(f.returns, 5)
    out = -1 * rank(x - delay(x, 10))
    return _frame(out, data)


# 原文：((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))
def alpha101_009(data, **params):
    f = _fields(data)
    d1 = delta(f.close, 1)
    out = where(
        lt(0, ts_min(d1, 5)),
        d1,
        where(lt(ts_max(d1, 5), 0), d1, -1 * d1),
    )
    return _frame(out, data)


# 原文：rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1) : ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : (-1 * delta(close, 1)))))
def alpha101_010(data, **params):
    f = _fields(data)
    d1 = delta(f.close, 1)
    out = rank(
        where(
            lt(0, ts_min(d1, 4)),
            d1,
            where(lt(ts_max(d1, 4), 0), d1, -1 * d1),
        )
    )
    return _frame(out, data)


# 原文：((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) * rank(delta(volume, 3)))
def alpha101_011(data, **params):
    f = _fields(data)
    out = (
        rank(ts_max(f.vwap - f.close, 3)) + rank(ts_min(f.vwap - f.close, 3))
    ) * rank(delta(f.volume, 3))
    return _frame(out, data)


# 原文：(sign(delta(volume, 1)) * (-1 * delta(close, 1)))
def alpha101_012(data, **params):
    f = _fields(data)
    out = sign(delta(f.volume, 1)) * (-1 * delta(f.close, 1))
    return _frame(out, data)


# 原文：(-1 * rank(covariance(rank(close), rank(volume), 5)))
def alpha101_013(data, **params):
    f = _fields(data)
    out = -1 * rank(covariance(rank(f.close), rank(f.volume), 5))
    return _frame(out, data)


# 原文：((-1 * rank(delta(returns, 3))) * correlation(open, volume, 10))
def alpha101_014(data, **params):
    f = _fields(data)
    out = (-1 * rank(delta(f.returns, 3))) * correlation(f.open, f.volume, 10)
    return _frame(out, data)


# 原文：(-1 * sum(rank(correlation(rank(high), rank(volume), 3)), 3))
def alpha101_015(data, **params):
    f = _fields(data)
    out = -1 * ts_sum(rank(correlation(rank(f.high), rank(f.volume), 3)), 3)
    return _frame(out, data)


# 原文：(-1 * rank(covariance(rank(high), rank(volume), 5)))
def alpha101_016(data, **params):
    f = _fields(data)
    out = -1 * rank(covariance(rank(f.high), rank(f.volume), 5))
    return _frame(out, data)


# 原文：(((-1 * rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) * rank(ts_rank((volume / adv20), 5)))
def alpha101_017(data, **params):
    f = _fields(data)
    out = (
        (-1 * rank(ts_rank(f.close, 10))) * rank(delta(delta(f.close, 1), 1))
    ) * rank(ts_rank(div(f.volume, f.adv(20)), 5))
    return _frame(out, data)


# 原文：(-1 * rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open, 10))))
def alpha101_018(data, **params):
    f = _fields(data)
    out = -1 * rank(
        (stddev(abs_(f.close - f.open), 5) + (f.close - f.open))
        + correlation(f.close, f.open, 10)
    )
    return _frame(out, data)


# 原文：((-1 * sign(((close - delay(close, 7)) + delta(close, 7)))) * (1 + rank((1 + sum(returns, 250)))))
def alpha101_019(data, **params):
    f = _fields(data)
    out = (-1 * sign((f.close - delay(f.close, 7)) + delta(f.close, 7))) * (
        1 + rank(1 + ts_sum(f.returns, 250))
    )
    return _frame(out, data)


# 原文：(((-1 * rank((open - delay(high, 1)))) * rank((open - delay(close, 1)))) * rank((open - delay(low, 1))))
def alpha101_020(data, **params):
    f = _fields(data)
    out = (
        (-1 * rank(f.open - delay(f.high, 1))) * rank(f.open - delay(f.close, 1))
    ) * rank(f.open - delay(f.low, 1))
    return _frame(out, data)


# 原文：((((sum(close, 8) / 8) + stddev(close, 8)) < (sum(close, 2) / 2)) ? (-1 * 1) : (((sum(close, 2) / 2) < ((sum(close, 8) / 8) - stddev(close, 8))) ? 1 : (((1 < (volume / adv20)) || ((volume / adv20) == 1)) ? 1 : (-1 * 1))))
def alpha101_021(data, **params):
    f = _fields(data)
    mean8 = div(ts_sum(f.close, 8), 8)
    std8 = stddev(f.close, 8)
    mean2 = div(ts_sum(f.close, 2), 2)
    ratio = div(f.volume, f.adv(20))
    out = where(
        lt(mean8 + std8, mean2),
        -1 * 1,
        where(
            lt(mean2, mean8 - std8),
            1,
            where(or_(lt(1, ratio), eq(ratio, 1)), 1, -1 * 1),
        ),
    )
    return _frame(out, data)


# 原文：(-1 * (delta(correlation(high, volume, 5), 5) * rank(stddev(close, 20))))
def alpha101_022(data, **params):
    f = _fields(data)
    out = -1 * (delta(correlation(f.high, f.volume, 5), 5) * rank(stddev(f.close, 20)))
    return _frame(out, data)


# 原文：(((sum(high, 20) / 20) < high) ? (-1 * delta(high, 2)) : 0)
def alpha101_023(data, **params):
    f = _fields(data)
    out = where(lt(div(ts_sum(f.high, 20), 20), f.high), -1 * delta(f.high, 2), 0)
    return _frame(out, data)


# 原文：((((delta((sum(close, 100) / 100), 100) / delay(close, 100)) < 0.05) || ((delta((sum(close, 100) / 100), 100) / delay(close, 100)) == 0.05)) ? (-1 * (close - ts_min(close, 100))) : (-1 * delta(close, 3)))
def alpha101_024(data, **params):
    f = _fields(data)
    x = div(delta(div(ts_sum(f.close, 100), 100), 100), delay(f.close, 100))
    out = where(
        or_(lt(x, 0.05), eq(x, 0.05)),
        -1 * (f.close - ts_min(f.close, 100)),
        -1 * delta(f.close, 3),
    )
    return _frame(out, data)


# 原文：rank(((((-1 * returns) * adv20) * vwap) * (high - close)))
def alpha101_025(data, **params):
    f = _fields(data)
    out = rank((((-1 * f.returns) * f.adv(20)) * f.vwap) * (f.high - f.close))
    return _frame(out, data)


# 原文：(-1 * ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))
def alpha101_026(data, **params):
    f = _fields(data)
    out = -1 * ts_max(correlation(ts_rank(f.volume, 5), ts_rank(f.high, 5), 5), 3)
    return _frame(out, data)


# 原文：((0.5 < rank((sum(correlation(rank(volume), rank(vwap), 6), 2) / 2.0))) ? (-1 * 1) : 1)
def alpha101_027(data, **params):
    f = _fields(data)
    inner = div(ts_sum(correlation(rank(f.volume), rank(f.vwap), 6), 2), 2.0)
    out = where(lt(0.5, rank(inner)), -1 * 1, 1)
    return _frame(out, data)


# 原文：scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))
def alpha101_028(data, **params):
    f = _fields(data)
    out = scale((correlation(f.adv(20), f.low, 5) + div(f.high + f.low, 2)) - f.close)
    return _frame(out, data)


# 原文：(min(product(rank(rank(scale(log(sum(ts_min(rank(rank((-1 * rank(delta((close - 1), 5))))), 2), 1))))), 1), 5) + ts_rank(delay((-1 * returns), 6), 5))
def alpha101_029(data, **params):
    f = _fields(data)
    inner = rank(rank(-1 * rank(delta(f.close - 1, 5))))
    inner = log(ts_sum(ts_min(inner, 2), 1))
    inner = ts_product(rank(rank(scale(inner))), 1)
    out = ts_min(inner, 5) + ts_rank(delay(-1 * f.returns, 6), 5)
    return _frame(out, data)


# 原文：(((1.0 - rank(((sign((close - delay(close, 1))) + sign((delay(close, 1) - delay(close, 2)))) + sign((delay(close, 2) - delay(close, 3)))))) * sum(volume, 5)) / sum(volume, 20))
def alpha101_030(data, **params):
    f = _fields(data)
    signs = (
        sign(f.close - delay(f.close, 1)) + sign(delay(f.close, 1) - delay(f.close, 2))
    ) + sign(delay(f.close, 2) - delay(f.close, 3))
    out = div((1.0 - rank(signs)) * ts_sum(f.volume, 5), ts_sum(f.volume, 20))
    return _frame(out, data)


# 原文：((rank(rank(rank(decay_linear((-1 * rank(rank(delta(close, 10)))), 10)))) + rank((-1 * delta(close, 3)))) + sign(scale(correlation(adv20, low, 12))))
def alpha101_031(data, **params):
    f = _fields(data)
    out = (
        rank(rank(rank(decay_linear(-1 * rank(rank(delta(f.close, 10))), 10))))
        + rank(-1 * delta(f.close, 3))
    ) + sign(scale(correlation(f.adv(20), f.low, 12)))
    return _frame(out, data)


# 原文：(scale(((sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5), 230))))
def alpha101_032(data, **params):
    f = _fields(data)
    out = scale(div(ts_sum(f.close, 7), 7) - f.close) + (
        20 * scale(correlation(f.vwap, delay(f.close, 5), 230))
    )
    return _frame(out, data)


# 原文：rank((-1 * ((1 - (open / close))^1)))
def alpha101_033(data, **params):
    f = _fields(data)
    out = rank(-1 * power(1 - div(f.open, f.close), 1))
    return _frame(out, data)


# 原文：rank(((1 - rank((stddev(returns, 2) / stddev(returns, 5)))) + (1 - rank(delta(close, 1)))))
def alpha101_034(data, **params):
    f = _fields(data)
    out = rank(
        (1 - rank(div(stddev(f.returns, 2), stddev(f.returns, 5))))
        + (1 - rank(delta(f.close, 1)))
    )
    return _frame(out, data)


# 原文：((Ts_Rank(volume, 32) * (1 - Ts_Rank(((close + high) - low), 16))) * (1 - Ts_Rank(returns, 32)))
def alpha101_035(data, **params):
    f = _fields(data)
    out = (ts_rank(f.volume, 32) * (1 - ts_rank((f.close + f.high) - f.low, 16))) * (
        1 - ts_rank(f.returns, 32)
    )
    return _frame(out, data)


# 原文：(((((2.21 * rank(correlation((close - open), delay(volume, 1), 15))) + (0.7 * rank((open - close)))) + (0.73 * rank(Ts_Rank(delay((-1 * returns), 6), 5)))) + rank(abs(correlation(vwap, adv20, 6)))) + (0.6 * rank((((sum(close, 200) / 200) - open) * (close - open)))))
def alpha101_036(data, **params):
    f = _fields(data)
    out = (
        (
            (
                (2.21 * rank(correlation(f.close - f.open, delay(f.volume, 1), 15)))
                + (0.7 * rank(f.open - f.close))
            )
            + (0.73 * rank(ts_rank(delay(-1 * f.returns, 6), 5)))
        )
        + rank(abs_(correlation(f.vwap, f.adv(20), 6)))
    ) + (0.6 * rank((div(ts_sum(f.close, 200), 200) - f.open) * (f.close - f.open)))
    return _frame(out, data)


# 原文：(rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))
def alpha101_037(data, **params):
    f = _fields(data)
    out = rank(correlation(delay(f.open - f.close, 1), f.close, 200)) + rank(
        f.open - f.close
    )
    return _frame(out, data)


# 原文：((-1 * rank(Ts_Rank(close, 10))) * rank((close / open)))
def alpha101_038(data, **params):
    f = _fields(data)
    out = (-1 * rank(ts_rank(f.close, 10))) * rank(div(f.close, f.open))
    return _frame(out, data)


# 原文：((-1 * rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9)))))) * (1 + rank(sum(returns, 250))))
def alpha101_039(data, **params):
    f = _fields(data)
    out = (
        -1
        * rank(
            delta(f.close, 7) * (1 - rank(decay_linear(div(f.volume, f.adv(20)), 9)))
        )
    ) * (1 + rank(ts_sum(f.returns, 250)))
    return _frame(out, data)


# 原文：((-1 * rank(stddev(high, 10))) * correlation(high, volume, 10))
def alpha101_040(data, **params):
    f = _fields(data)
    out = (-1 * rank(stddev(f.high, 10))) * correlation(f.high, f.volume, 10)
    return _frame(out, data)


# 原文：(((high * low)^0.5) - vwap)
def alpha101_041(data, **params):
    f = _fields(data)
    out = power(f.high * f.low, 0.5) - f.vwap
    return _frame(out, data)


# 原文：(rank((vwap - close)) / rank((vwap + close)))
def alpha101_042(data, **params):
    f = _fields(data)
    out = div(rank(f.vwap - f.close), rank(f.vwap + f.close))
    return _frame(out, data)


# 原文：(ts_rank((volume / adv20), 20) * ts_rank((-1 * delta(close, 7)), 8))
def alpha101_043(data, **params):
    f = _fields(data)
    out = ts_rank(div(f.volume, f.adv(20)), 20) * ts_rank(-1 * delta(f.close, 7), 8)
    return _frame(out, data)


# 原文：(-1 * correlation(high, rank(volume), 5))
def alpha101_044(data, **params):
    f = _fields(data)
    out = -1 * correlation(f.high, rank(f.volume), 5)
    return _frame(out, data)


# 原文：(-1 * ((rank((sum(delay(close, 5), 20) / 20)) * correlation(close, volume, 2)) * rank(correlation(sum(close, 5), sum(close, 20), 2))))
def alpha101_045(data, **params):
    f = _fields(data)
    out = -1 * (
        (
            rank(div(ts_sum(delay(f.close, 5), 20), 20))
            * correlation(f.close, f.volume, 2)
        )
        * rank(correlation(ts_sum(f.close, 5), ts_sum(f.close, 20), 2))
    )
    return _frame(out, data)


def _slope_gap(f):
    """046、049、051 共用的 ``(((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))``。"""
    return div(delay(f.close, 20) - delay(f.close, 10), 10) - div(
        delay(f.close, 10) - f.close, 10
    )


# 原文：((0.25 < (((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10))) ? (-1 * 1) : (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < 0) ? 1 : ((-1 * 1) * (close - delay(close, 1)))))
def alpha101_046(data, **params):
    f = _fields(data)
    x = _slope_gap(f)
    out = where(
        lt(0.25, x),
        -1 * 1,
        where(lt(x, 0), 1, (-1 * 1) * (f.close - delay(f.close, 1))),
    )
    return _frame(out, data)


# 原文：((((rank((1 / close)) * volume) / adv20) * ((high * rank((high - close))) / (sum(high, 5) / 5))) - rank((vwap - delay(vwap, 5))))
def alpha101_047(data, **params):
    f = _fields(data)
    out = (
        div(rank(div(1, f.close)) * f.volume, f.adv(20))
        * div(f.high * rank(f.high - f.close), div(ts_sum(f.high, 5), 5))
    ) - rank(f.vwap - delay(f.vwap, 5))
    return _frame(out, data)


# 原文：(indneutralize(((correlation(delta(close, 1), delta(delay(close, 1), 1), 250) * delta(close, 1)) / close), IndClass.subindustry) / sum(((delta(close, 1) / delay(close, 1))^2), 250))
def alpha101_048(data, **params):
    f = _fields(data)
    d1 = delta(f.close, 1)
    numerator = indneutralize(
        div(correlation(d1, delta(delay(f.close, 1), 1), 250) * d1, f.close),
        f.industry,
    )
    denominator = ts_sum(power(div(d1, delay(f.close, 1)), 2), 250)
    out = div(numerator, denominator)
    return _frame(out, data)


# 原文：(((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.1)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
def alpha101_049(data, **params):
    f = _fields(data)
    out = where(
        lt(_slope_gap(f), -1 * 0.1), 1, (-1 * 1) * (f.close - delay(f.close, 1))
    )
    return _frame(out, data)


# 原文：(-1 * ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))
def alpha101_050(data, **params):
    f = _fields(data)
    out = -1 * ts_max(rank(correlation(rank(f.volume), rank(f.vwap), 5)), 5)
    return _frame(out, data)


# 原文：(((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < (-1 * 0.05)) ? 1 : ((-1 * 1) * (close - delay(close, 1))))
def alpha101_051(data, **params):
    f = _fields(data)
    out = where(
        lt(_slope_gap(f), -1 * 0.05), 1, (-1 * 1) * (f.close - delay(f.close, 1))
    )
    return _frame(out, data)


# 原文：((((-1 * ts_min(low, 5)) + delay(ts_min(low, 5), 5)) * rank(((sum(returns, 240) - sum(returns, 20)) / 220))) * ts_rank(volume, 5))
def alpha101_052(data, **params):
    f = _fields(data)
    out = (
        ((-1 * ts_min(f.low, 5)) + delay(ts_min(f.low, 5), 5))
        * rank(div(ts_sum(f.returns, 240) - ts_sum(f.returns, 20), 220))
    ) * ts_rank(f.volume, 5)
    return _frame(out, data)


# 原文：(-1 * delta((((close - low) - (high - close)) / (close - low)), 9))
def alpha101_053(data, **params):
    f = _fields(data)
    out = -1 * delta(div((f.close - f.low) - (f.high - f.close), f.close - f.low), 9)
    return _frame(out, data)


# 原文：((-1 * ((low - close) * (open^5))) / ((low - high) * (close^5)))
def alpha101_054(data, **params):
    f = _fields(data)
    out = div(
        -1 * ((f.low - f.close) * power(f.open, 5)),
        (f.low - f.high) * power(f.close, 5),
    )
    return _frame(out, data)


# 原文：(-1 * correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12)))), rank(volume), 6))
def alpha101_055(data, **params):
    f = _fields(data)
    x = div(f.close - ts_min(f.low, 12), ts_max(f.high, 12) - ts_min(f.low, 12))
    out = -1 * correlation(rank(x), rank(f.volume), 6)
    return _frame(out, data)


# 原文：(0 - (1 * (rank((sum(returns, 10) / sum(sum(returns, 2), 3))) * rank((returns * cap)))))
def alpha101_056(data, **params):
    f = _fields(data)
    out = 0 - (
        1
        * (
            rank(div(ts_sum(f.returns, 10), ts_sum(ts_sum(f.returns, 2), 3)))
            * rank(f.returns * f.cap)
        )
    )
    return _frame(out, data)


# 原文：(0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2))))
def alpha101_057(data, argmax=DEFAULT_ARGMAX, **params):
    f = _fields(data)
    out = 0 - (
        1
        * div(
            f.close - f.vwap,
            decay_linear(rank(ts_argmax(f.close, 30, how=argmax)), 2),
        )
    )
    return _frame(out, data)


# 原文：(-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.sector), volume, 3.92795), 7.89291), 5.50322))
def alpha101_058(data, **params):
    f = _fields(data)
    out = -1 * ts_rank(
        decay_linear(
            correlation(indneutralize(f.vwap, f.industry), f.volume, 3.92795),
            7.89291,
        ),
        5.50322,
    )
    return _frame(out, data)


# 原文：(-1 * Ts_Rank(decay_linear(correlation(IndNeutralize(((vwap * 0.728317) + (vwap * (1 - 0.728317))), IndClass.industry), volume, 4.25197), 16.2289), 8.19648))
def alpha101_059(data, **params):
    f = _fields(data)
    mixed = (f.vwap * 0.728317) + (f.vwap * (1 - 0.728317))
    out = -1 * ts_rank(
        decay_linear(
            correlation(indneutralize(mixed, f.industry), f.volume, 4.25197),
            16.2289,
        ),
        8.19648,
    )
    return _frame(out, data)


# 原文：(0 - (1 * ((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume)))) - scale(rank(ts_argmax(close, 10))))))
def alpha101_060(data, argmax=DEFAULT_ARGMAX, **params):
    f = _fields(data)
    location = div((f.close - f.low) - (f.high - f.close), f.high - f.low) * f.volume
    out = 0 - (
        1
        * (
            (2 * scale(rank(location)))
            - scale(rank(ts_argmax(f.close, 10, how=argmax)))
        )
    )
    return _frame(out, data)


# 原文：(rank((vwap - ts_min(vwap, 16.1219))) < rank(correlation(vwap, adv180, 17.9282)))
def alpha101_061(data, **params):
    f = _fields(data)
    out = lt(
        rank(f.vwap - ts_min(f.vwap, 16.1219)),
        rank(correlation(f.vwap, f.adv(180), 17.9282)),
    )
    return _frame(out, data)


# 原文：((rank(correlation(vwap, sum(adv20, 22.4101), 9.91009)) < rank(((rank(open) + rank(open)) < (rank(((high + low) / 2)) + rank(high))))) * -1)
def alpha101_062(data, **params):
    f = _fields(data)
    out = (
        lt(
            rank(correlation(f.vwap, ts_sum(f.adv(20), 22.4101), 9.91009)),
            rank(
                lt(
                    rank(f.open) + rank(f.open),
                    rank(div(f.high + f.low, 2)) + rank(f.high),
                )
            ),
        )
        * -1
    )
    return _frame(out, data)


# 原文：((rank(decay_linear(delta(IndNeutralize(close, IndClass.industry), 2.25164), 8.22237)) - rank(decay_linear(correlation(((vwap * 0.318108) + (open * (1 - 0.318108))), sum(adv180, 37.2467), 13.557), 12.2883))) * -1)
def alpha101_063(data, **params):
    f = _fields(data)
    mixed = (f.vwap * 0.318108) + (f.open * (1 - 0.318108))
    out = (
        rank(decay_linear(delta(indneutralize(f.close, f.industry), 2.25164), 8.22237))
        - rank(
            decay_linear(
                correlation(mixed, ts_sum(f.adv(180), 37.2467), 13.557), 12.2883
            )
        )
    ) * -1
    return _frame(out, data)


# 原文：((rank(correlation(sum(((open * 0.178404) + (low * (1 - 0.178404))), 12.7054), sum(adv120, 12.7054), 16.6208)) < rank(delta(((((high + low) / 2) * 0.178404) + (vwap * (1 - 0.178404))), 3.69741))) * -1)
def alpha101_064(data, **params):
    f = _fields(data)
    mixed_ol = (f.open * 0.178404) + (f.low * (1 - 0.178404))
    mixed_mv = (div(f.high + f.low, 2) * 0.178404) + (f.vwap * (1 - 0.178404))
    out = (
        lt(
            rank(
                correlation(
                    ts_sum(mixed_ol, 12.7054), ts_sum(f.adv(120), 12.7054), 16.6208
                )
            ),
            rank(delta(mixed_mv, 3.69741)),
        )
        * -1
    )
    return _frame(out, data)


# 原文：((rank(correlation(((open * 0.00817205) + (vwap * (1 - 0.00817205))), sum(adv60, 8.6911), 6.40374)) < rank((open - ts_min(open, 13.635)))) * -1)
def alpha101_065(data, **params):
    f = _fields(data)
    mixed = (f.open * 0.00817205) + (f.vwap * (1 - 0.00817205))
    out = (
        lt(
            rank(correlation(mixed, ts_sum(f.adv(60), 8.6911), 6.40374)),
            rank(f.open - ts_min(f.open, 13.635)),
        )
        * -1
    )
    return _frame(out, data)


# 原文：((rank(decay_linear(delta(vwap, 3.51013), 7.23052)) + Ts_Rank(decay_linear(((((low * 0.96633) + (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2))), 11.4157), 6.72611)) * -1)
def alpha101_066(data, **params):
    f = _fields(data)
    mixed = (f.low * 0.96633) + (f.low * (1 - 0.96633))
    out = (
        rank(decay_linear(delta(f.vwap, 3.51013), 7.23052))
        + ts_rank(
            decay_linear(div(mixed - f.vwap, f.open - div(f.high + f.low, 2)), 11.4157),
            6.72611,
        )
    ) * -1
    return _frame(out, data)


# 原文：((rank((high - ts_min(high, 2.14593)))^rank(correlation(IndNeutralize(vwap, IndClass.sector), IndNeutralize(adv20, IndClass.subindustry), 6.02936))) * -1)
def alpha101_067(data, **params):
    f = _fields(data)
    out = (
        power(
            rank(f.high - ts_min(f.high, 2.14593)),
            rank(
                correlation(
                    indneutralize(f.vwap, f.industry),
                    indneutralize(f.adv(20), f.industry),
                    6.02936,
                )
            ),
        )
        * -1
    )
    return _frame(out, data)


# 原文：((Ts_Rank(correlation(rank(high), rank(adv15), 8.91644), 13.9333) < rank(delta(((close * 0.518371) + (low * (1 - 0.518371))), 1.06157))) * -1)
def alpha101_068(data, **params):
    f = _fields(data)
    mixed = (f.close * 0.518371) + (f.low * (1 - 0.518371))
    out = (
        lt(
            ts_rank(correlation(rank(f.high), rank(f.adv(15)), 8.91644), 13.9333),
            rank(delta(mixed, 1.06157)),
        )
        * -1
    )
    return _frame(out, data)


# 原文：((rank(ts_max(delta(IndNeutralize(vwap, IndClass.industry), 2.72412), 4.79344))^Ts_Rank(correlation(((close * 0.490655) + (vwap * (1 - 0.490655))), adv20, 4.92416), 9.0615)) * -1)
def alpha101_069(data, **params):
    f = _fields(data)
    mixed = (f.close * 0.490655) + (f.vwap * (1 - 0.490655))
    out = (
        power(
            rank(ts_max(delta(indneutralize(f.vwap, f.industry), 2.72412), 4.79344)),
            ts_rank(correlation(mixed, f.adv(20), 4.92416), 9.0615),
        )
        * -1
    )
    return _frame(out, data)


# 原文：((rank(delta(vwap, 1.29456))^Ts_Rank(correlation(IndNeutralize(close, IndClass.industry), adv50, 17.8256), 17.9171)) * -1)
def alpha101_070(data, **params):
    f = _fields(data)
    out = (
        power(
            rank(delta(f.vwap, 1.29456)),
            ts_rank(
                correlation(indneutralize(f.close, f.industry), f.adv(50), 17.8256),
                17.9171,
            ),
        )
        * -1
    )
    return _frame(out, data)


# 原文：max(Ts_Rank(decay_linear(correlation(Ts_Rank(close, 3.43976), Ts_Rank(adv180, 12.0647), 18.0175), 4.20501), 15.6948), Ts_Rank(decay_linear((rank(((low + open) - (vwap + vwap)))^2), 16.4662), 4.4388))
def alpha101_071(data, **params):
    f = _fields(data)
    out = maximum(
        ts_rank(
            decay_linear(
                correlation(
                    ts_rank(f.close, 3.43976), ts_rank(f.adv(180), 12.0647), 18.0175
                ),
                4.20501,
            ),
            15.6948,
        ),
        ts_rank(
            decay_linear(power(rank((f.low + f.open) - (f.vwap + f.vwap)), 2), 16.4662),
            4.4388,
        ),
    )
    return _frame(out, data)


# 原文：(rank(decay_linear(correlation(((high + low) / 2), adv40, 8.93345), 10.1519)) / rank(decay_linear(correlation(Ts_Rank(vwap, 3.72469), Ts_Rank(volume, 18.5188), 6.86671), 2.95011)))
def alpha101_072(data, **params):
    f = _fields(data)
    out = div(
        rank(
            decay_linear(
                correlation(div(f.high + f.low, 2), f.adv(40), 8.93345), 10.1519
            )
        ),
        rank(
            decay_linear(
                correlation(
                    ts_rank(f.vwap, 3.72469), ts_rank(f.volume, 18.5188), 6.86671
                ),
                2.95011,
            )
        ),
    )
    return _frame(out, data)


# 原文：(max(rank(decay_linear(delta(vwap, 4.72775), 2.91864)), Ts_Rank(decay_linear(((delta(((open * 0.147155) + (low * (1 - 0.147155))), 2.03608) / ((open * 0.147155) + (low * (1 - 0.147155)))) * -1), 3.33829), 16.7411)) * -1)
def alpha101_073(data, **params):
    f = _fields(data)
    mixed = (f.open * 0.147155) + (f.low * (1 - 0.147155))
    out = (
        maximum(
            rank(decay_linear(delta(f.vwap, 4.72775), 2.91864)),
            ts_rank(
                decay_linear(div(delta(mixed, 2.03608), mixed) * -1, 3.33829),
                16.7411,
            ),
        )
        * -1
    )
    return _frame(out, data)


# 原文：((rank(correlation(close, sum(adv30, 37.4843), 15.1365)) < rank(correlation(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(volume), 11.4791))) * -1)
def alpha101_074(data, **params):
    f = _fields(data)
    mixed = (f.high * 0.0261661) + (f.vwap * (1 - 0.0261661))
    out = (
        lt(
            rank(correlation(f.close, ts_sum(f.adv(30), 37.4843), 15.1365)),
            rank(correlation(rank(mixed), rank(f.volume), 11.4791)),
        )
        * -1
    )
    return _frame(out, data)


# 原文：(rank(correlation(vwap, volume, 4.24304)) < rank(correlation(rank(low), rank(adv50), 12.4413)))
def alpha101_075(data, **params):
    f = _fields(data)
    out = lt(
        rank(correlation(f.vwap, f.volume, 4.24304)),
        rank(correlation(rank(f.low), rank(f.adv(50)), 12.4413)),
    )
    return _frame(out, data)


# 原文：(max(rank(decay_linear(delta(vwap, 1.24383), 11.8259)), Ts_Rank(decay_linear(Ts_Rank(correlation(IndNeutralize(low, IndClass.sector), adv81, 8.14941), 19.569), 17.1543), 19.383)) * -1)
def alpha101_076(data, **params):
    f = _fields(data)
    out = (
        maximum(
            rank(decay_linear(delta(f.vwap, 1.24383), 11.8259)),
            ts_rank(
                decay_linear(
                    ts_rank(
                        correlation(
                            indneutralize(f.low, f.industry), f.adv(81), 8.14941
                        ),
                        19.569,
                    ),
                    17.1543,
                ),
                19.383,
            ),
        )
        * -1
    )
    return _frame(out, data)


# 原文：min(rank(decay_linear(((((high + low) / 2) + high) - (vwap + high)), 20.0451)), rank(decay_linear(correlation(((high + low) / 2), adv40, 3.1614), 5.64125)))
def alpha101_077(data, **params):
    f = _fields(data)
    out = minimum(
        rank(
            decay_linear((div(f.high + f.low, 2) + f.high) - (f.vwap + f.high), 20.0451)
        ),
        rank(
            decay_linear(
                correlation(div(f.high + f.low, 2), f.adv(40), 3.1614), 5.64125
            )
        ),
    )
    return _frame(out, data)


# 原文：(rank(correlation(sum(((low * 0.352233) + (vwap * (1 - 0.352233))), 19.7428), sum(adv40, 19.7428), 6.83313))^rank(correlation(rank(vwap), rank(volume), 5.77492)))
def alpha101_078(data, **params):
    f = _fields(data)
    mixed = (f.low * 0.352233) + (f.vwap * (1 - 0.352233))
    out = power(
        rank(correlation(ts_sum(mixed, 19.7428), ts_sum(f.adv(40), 19.7428), 6.83313)),
        rank(correlation(rank(f.vwap), rank(f.volume), 5.77492)),
    )
    return _frame(out, data)


# 原文：(rank(delta(IndNeutralize(((close * 0.60733) + (open * (1 - 0.60733))), IndClass.sector), 1.23438)) < rank(correlation(Ts_Rank(vwap, 3.60973), Ts_Rank(adv150, 9.18637), 14.6644)))
def alpha101_079(data, **params):
    f = _fields(data)
    mixed = (f.close * 0.60733) + (f.open * (1 - 0.60733))
    out = lt(
        rank(delta(indneutralize(mixed, f.industry), 1.23438)),
        rank(
            correlation(ts_rank(f.vwap, 3.60973), ts_rank(f.adv(150), 9.18637), 14.6644)
        ),
    )
    return _frame(out, data)


# 原文：((rank(Sign(delta(IndNeutralize(((open * 0.868128) + (high * (1 - 0.868128))), IndClass.industry), 4.04545)))^Ts_Rank(correlation(high, adv10, 5.11456), 5.53756)) * -1)
def alpha101_080(data, **params):
    f = _fields(data)
    mixed = (f.open * 0.868128) + (f.high * (1 - 0.868128))
    out = (
        power(
            rank(sign(delta(indneutralize(mixed, f.industry), 4.04545))),
            ts_rank(correlation(f.high, f.adv(10), 5.11456), 5.53756),
        )
        * -1
    )
    return _frame(out, data)


# 原文：((rank(Log(product(rank((rank(correlation(vwap, sum(adv10, 49.6054), 8.47743))^4)), 14.9655))) < rank(correlation(rank(vwap), rank(volume), 5.07914))) * -1)
def alpha101_081(data, **params):
    f = _fields(data)
    inner = rank(correlation(f.vwap, ts_sum(f.adv(10), 49.6054), 8.47743))
    out = (
        lt(
            rank(log(ts_product(rank(power(inner, 4)), 14.9655))),
            rank(correlation(rank(f.vwap), rank(f.volume), 5.07914)),
        )
        * -1
    )
    return _frame(out, data)


# 原文：(min(rank(decay_linear(delta(open, 1.46063), 14.8717)), Ts_Rank(decay_linear(correlation(IndNeutralize(volume, IndClass.sector), ((open * 0.634196) + (open * (1 - 0.634196))), 17.4842), 6.92131), 13.4283)) * -1)
def alpha101_082(data, **params):
    f = _fields(data)
    mixed = (f.open * 0.634196) + (f.open * (1 - 0.634196))
    out = (
        minimum(
            rank(decay_linear(delta(f.open, 1.46063), 14.8717)),
            ts_rank(
                decay_linear(
                    correlation(indneutralize(f.volume, f.industry), mixed, 17.4842),
                    6.92131,
                ),
                13.4283,
            ),
        )
        * -1
    )
    return _frame(out, data)


# 原文：((rank(delay(((high - low) / (sum(close, 5) / 5)), 2)) * rank(rank(volume))) / (((high - low) / (sum(close, 5) / 5)) / (vwap - close)))
def alpha101_083(data, **params):
    f = _fields(data)
    spread = div(f.high - f.low, div(ts_sum(f.close, 5), 5))
    out = div(
        rank(delay(spread, 2)) * rank(rank(f.volume)),
        div(spread, f.vwap - f.close),
    )
    return _frame(out, data)


# 原文：SignedPower(Ts_Rank((vwap - ts_max(vwap, 15.3217)), 20.7127), delta(close, 4.96796))
def alpha101_084(data, **params):
    f = _fields(data)
    out = signedpower(
        ts_rank(f.vwap - ts_max(f.vwap, 15.3217), 20.7127), delta(f.close, 4.96796)
    )
    return _frame(out, data)


# 原文：(rank(correlation(((high * 0.876703) + (close * (1 - 0.876703))), adv30, 9.61331))^rank(correlation(Ts_Rank(((high + low) / 2), 3.70596), Ts_Rank(volume, 10.1595), 7.11408)))
def alpha101_085(data, **params):
    f = _fields(data)
    mixed = (f.high * 0.876703) + (f.close * (1 - 0.876703))
    out = power(
        rank(correlation(mixed, f.adv(30), 9.61331)),
        rank(
            correlation(
                ts_rank(div(f.high + f.low, 2), 3.70596),
                ts_rank(f.volume, 10.1595),
                7.11408,
            )
        ),
    )
    return _frame(out, data)


# 原文：((Ts_Rank(correlation(close, sum(adv20, 14.7444), 6.00049), 20.4195) < rank(((open + close) - (vwap + open)))) * -1)
def alpha101_086(data, **params):
    f = _fields(data)
    out = (
        lt(
            ts_rank(correlation(f.close, ts_sum(f.adv(20), 14.7444), 6.00049), 20.4195),
            rank((f.open + f.close) - (f.vwap + f.open)),
        )
        * -1
    )
    return _frame(out, data)


# 原文：(max(rank(decay_linear(delta(((close * 0.369701) + (vwap * (1 - 0.369701))), 1.91233), 2.65461)), Ts_Rank(decay_linear(abs(correlation(IndNeutralize(adv81, IndClass.industry), close, 13.4132)), 4.89768), 14.4535)) * -1)
def alpha101_087(data, **params):
    f = _fields(data)
    mixed = (f.close * 0.369701) + (f.vwap * (1 - 0.369701))
    out = (
        maximum(
            rank(decay_linear(delta(mixed, 1.91233), 2.65461)),
            ts_rank(
                decay_linear(
                    abs_(
                        correlation(
                            indneutralize(f.adv(81), f.industry), f.close, 13.4132
                        )
                    ),
                    4.89768,
                ),
                14.4535,
            ),
        )
        * -1
    )
    return _frame(out, data)


# 原文：min(rank(decay_linear(((rank(open) + rank(low)) - (rank(high) + rank(close))), 8.06882)), Ts_Rank(decay_linear(correlation(Ts_Rank(close, 8.44728), Ts_Rank(adv60, 20.6966), 8.01266), 6.65053), 2.61957))
def alpha101_088(data, **params):
    f = _fields(data)
    out = minimum(
        rank(
            decay_linear(
                (rank(f.open) + rank(f.low)) - (rank(f.high) + rank(f.close)), 8.06882
            )
        ),
        ts_rank(
            decay_linear(
                correlation(
                    ts_rank(f.close, 8.44728), ts_rank(f.adv(60), 20.6966), 8.01266
                ),
                6.65053,
            ),
            2.61957,
        ),
    )
    return _frame(out, data)


# 原文：(Ts_Rank(decay_linear(correlation(((low * 0.967285) + (low * (1 - 0.967285))), adv10, 6.94279), 5.51607), 3.79744) - Ts_Rank(decay_linear(delta(IndNeutralize(vwap, IndClass.industry), 3.48158), 10.1466), 15.3012))
def alpha101_089(data, **params):
    f = _fields(data)
    mixed = (f.low * 0.967285) + (f.low * (1 - 0.967285))
    out = ts_rank(
        decay_linear(correlation(mixed, f.adv(10), 6.94279), 5.51607), 3.79744
    ) - ts_rank(
        decay_linear(delta(indneutralize(f.vwap, f.industry), 3.48158), 10.1466),
        15.3012,
    )
    return _frame(out, data)


# 原文：((rank((close - ts_max(close, 4.66719)))^Ts_Rank(correlation(IndNeutralize(adv40, IndClass.subindustry), low, 5.38375), 3.21856)) * -1)
def alpha101_090(data, **params):
    f = _fields(data)
    out = (
        power(
            rank(f.close - ts_max(f.close, 4.66719)),
            ts_rank(
                correlation(indneutralize(f.adv(40), f.industry), f.low, 5.38375),
                3.21856,
            ),
        )
        * -1
    )
    return _frame(out, data)


# 原文：((Ts_Rank(decay_linear(decay_linear(correlation(IndNeutralize(close, IndClass.industry), volume, 9.74928), 16.398), 3.83219), 4.8667) - rank(decay_linear(correlation(vwap, adv30, 4.01303), 2.6809))) * -1)
def alpha101_091(data, **params):
    f = _fields(data)
    out = (
        ts_rank(
            decay_linear(
                decay_linear(
                    correlation(indneutralize(f.close, f.industry), f.volume, 9.74928),
                    16.398,
                ),
                3.83219,
            ),
            4.8667,
        )
        - rank(decay_linear(correlation(f.vwap, f.adv(30), 4.01303), 2.6809))
    ) * -1
    return _frame(out, data)


# 原文：min(Ts_Rank(decay_linear(((((high + low) / 2) + close) < (low + open)), 14.7221), 18.8683), Ts_Rank(decay_linear(correlation(rank(low), rank(adv30), 7.58555), 6.94024), 6.80584))
def alpha101_092(data, **params):
    f = _fields(data)
    out = minimum(
        ts_rank(
            decay_linear(lt(div(f.high + f.low, 2) + f.close, f.low + f.open), 14.7221),
            18.8683,
        ),
        ts_rank(
            decay_linear(correlation(rank(f.low), rank(f.adv(30)), 7.58555), 6.94024),
            6.80584,
        ),
    )
    return _frame(out, data)


# 原文：(Ts_Rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.industry), adv81, 17.4193), 19.848), 7.54455) / rank(decay_linear(delta(((close * 0.524434) + (vwap * (1 - 0.524434))), 2.77377), 16.2664)))
def alpha101_093(data, **params):
    f = _fields(data)
    mixed = (f.close * 0.524434) + (f.vwap * (1 - 0.524434))
    out = div(
        ts_rank(
            decay_linear(
                correlation(indneutralize(f.vwap, f.industry), f.adv(81), 17.4193),
                19.848,
            ),
            7.54455,
        ),
        rank(decay_linear(delta(mixed, 2.77377), 16.2664)),
    )
    return _frame(out, data)


# 原文：((rank((vwap - ts_min(vwap, 11.5783)))^Ts_Rank(correlation(Ts_Rank(vwap, 19.6462), Ts_Rank(adv60, 4.02992), 18.0926), 2.70756)) * -1)
def alpha101_094(data, **params):
    f = _fields(data)
    out = (
        power(
            rank(f.vwap - ts_min(f.vwap, 11.5783)),
            ts_rank(
                correlation(
                    ts_rank(f.vwap, 19.6462), ts_rank(f.adv(60), 4.02992), 18.0926
                ),
                2.70756,
            ),
        )
        * -1
    )
    return _frame(out, data)


# 原文：(rank((open - ts_min(open, 12.4105))) < Ts_Rank((rank(correlation(sum(((high + low) / 2), 19.1351), sum(adv40, 19.1351), 12.8742))^5), 11.7584))
def alpha101_095(data, **params):
    f = _fields(data)
    out = lt(
        rank(f.open - ts_min(f.open, 12.4105)),
        ts_rank(
            power(
                rank(
                    correlation(
                        ts_sum(div(f.high + f.low, 2), 19.1351),
                        ts_sum(f.adv(40), 19.1351),
                        12.8742,
                    )
                ),
                5,
            ),
            11.7584,
        ),
    )
    return _frame(out, data)


# 原文：(max(Ts_Rank(decay_linear(correlation(rank(vwap), rank(volume), 3.83878), 4.16783), 8.38151), Ts_Rank(decay_linear(Ts_ArgMax(correlation(Ts_Rank(close, 7.45404), Ts_Rank(adv60, 4.13242), 3.65459), 12.6556), 14.0365), 13.4143)) * -1)
def alpha101_096(data, argmax=DEFAULT_ARGMAX, **params):
    f = _fields(data)
    out = (
        maximum(
            ts_rank(
                decay_linear(
                    correlation(rank(f.vwap), rank(f.volume), 3.83878), 4.16783
                ),
                8.38151,
            ),
            ts_rank(
                decay_linear(
                    ts_argmax(
                        correlation(
                            ts_rank(f.close, 7.45404),
                            ts_rank(f.adv(60), 4.13242),
                            3.65459,
                        ),
                        12.6556,
                        how=argmax,
                    ),
                    14.0365,
                ),
                13.4143,
            ),
        )
        * -1
    )
    return _frame(out, data)


# 原文：((rank(decay_linear(delta(IndNeutralize(((low * 0.721001) + (vwap * (1 - 0.721001))), IndClass.industry), 3.3705), 20.4523)) - Ts_Rank(decay_linear(Ts_Rank(correlation(Ts_Rank(low, 7.87871), Ts_Rank(adv60, 17.255), 4.97547), 18.5925), 15.7152), 6.71659)) * -1)
def alpha101_097(data, **params):
    f = _fields(data)
    mixed = (f.low * 0.721001) + (f.vwap * (1 - 0.721001))
    out = (
        rank(decay_linear(delta(indneutralize(mixed, f.industry), 3.3705), 20.4523))
        - ts_rank(
            decay_linear(
                ts_rank(
                    correlation(
                        ts_rank(f.low, 7.87871), ts_rank(f.adv(60), 17.255), 4.97547
                    ),
                    18.5925,
                ),
                15.7152,
            ),
            6.71659,
        )
    ) * -1
    return _frame(out, data)


# 原文：(rank(decay_linear(correlation(vwap, sum(adv5, 26.4719), 4.58418), 7.18088)) - rank(decay_linear(Ts_Rank(Ts_ArgMin(correlation(rank(open), rank(adv15), 20.8187), 8.62571), 6.95668), 8.07206)))
def alpha101_098(data, argmax=DEFAULT_ARGMAX, **params):
    f = _fields(data)
    out = rank(
        decay_linear(correlation(f.vwap, ts_sum(f.adv(5), 26.4719), 4.58418), 7.18088)
    ) - rank(
        decay_linear(
            ts_rank(
                ts_argmin(
                    correlation(rank(f.open), rank(f.adv(15)), 20.8187),
                    8.62571,
                    how=argmax,
                ),
                6.95668,
            ),
            8.07206,
        )
    )
    return _frame(out, data)


# 原文：((rank(correlation(sum(((high + low) / 2), 19.8975), sum(adv60, 19.8975), 8.8136)) < rank(correlation(low, volume, 6.28259))) * -1)
def alpha101_099(data, **params):
    f = _fields(data)
    out = (
        lt(
            rank(
                correlation(
                    ts_sum(div(f.high + f.low, 2), 19.8975),
                    ts_sum(f.adv(60), 19.8975),
                    8.8136,
                )
            ),
            rank(correlation(f.low, f.volume, 6.28259)),
        )
        * -1
    )
    return _frame(out, data)


# 原文：(0 - (1 * (((1.5 * scale(indneutralize(indneutralize(rank(((((close - low) - (high - close)) / (high - low)) * volume)), IndClass.subindustry), IndClass.subindustry))) - scale(indneutralize((correlation(close, rank(adv20), 5) - rank(ts_argmin(close, 30))), IndClass.subindustry))) * (volume / adv20))))
def alpha101_100(data, argmax=DEFAULT_ARGMAX, **params):
    f = _fields(data)
    location = rank(
        div((f.close - f.low) - (f.high - f.close), f.high - f.low) * f.volume
    )
    first = indneutralize(indneutralize(location, f.industry), f.industry)
    second = indneutralize(
        correlation(f.close, rank(f.adv(20)), 5)
        - rank(ts_argmin(f.close, 30, how=argmax)),
        f.industry,
    )
    out = 0 - (1 * (((1.5 * scale(first)) - scale(second)) * div(f.volume, f.adv(20))))
    return _frame(out, data)


# 原文：((close - open) / ((high - low) + .001))
def alpha101_101(data, **params):
    f = _fields(data)
    out = div(f.close - f.open, (f.high - f.low) + 0.001)
    return _frame(out, data)


NAMES = tuple(f"alpha101_{i:03d}" for i in range(1, 102))


def references(argmax=DEFAULT_ARGMAX):
    """``{名字: 函数(data)}``；``argmax`` 选 D-13 的读法，作用于 ``ARGMAX_DEPENDENT``。"""
    table = {}
    for name in NAMES:
        func = globals()[name]
        table[name] = partial(func, argmax=argmax) if name in ARGMAX_DEPENDENT else func
    return table


REFERENCES = references(DEFAULT_ARGMAX)


def compute_many(frames, names, argmax=DEFAULT_ARGMAX):
    """一次算一批（共享 ``adv`` 的缓存）；返回 ``{名字: ndarray[T, N]}``。"""
    table = references(argmax)
    fields = _Fields(frames)
    out = {}
    for name in names:
        out[name] = np.asarray(table[name](fields), dtype=float)
    return out
