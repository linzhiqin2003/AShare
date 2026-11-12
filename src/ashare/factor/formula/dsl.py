"""WorldQuant 式、国君式公式的翻译用语言：每个原文函数一个同语义的构造函数。

口径（复权、缺失、排名、取整）全部集中在这一个文件里，改口径时只改这里，翻译不用重做。
括号里的 D-xx 是因子包约定文件里的决定编号。用法（自编的例子）：

    from ashare.factor.formula import dsl as d

    def my_factor_001():
        return -1 * d.correlation(d.rank(d.high), d.rank(d.amount), 7)

约定：带天数的参数都先向下取整（D-15）；滚动窗口必须满（D-09，``min_periods = 天数``）。
"""

import contextlib
import contextvars
import math

from ..expression import call, field
from . import ops

# ---- 取整与窗口 -----------------------------------------------------------------


def days(d):
    """天数参数：向下取整为整数（原文附录：non-integer number of days d is converted to floor(d)）。"""
    if isinstance(d, bool) or not isinstance(d, (int, float)) or not math.isfinite(d):
        raise TypeError(f"天数必须是有限数字：{d!r}")
    n = math.floor(d)
    if n < 0:
        raise ValueError(f"天数不能为负：{d!r}")
    return n


def _window(d):
    n = days(d)
    if n < 1:
        raise ValueError(f"窗口至少 1 天：{d!r}")
    return n


def _rolling(op, x, d, **params):
    n = _window(d)
    return call(op, x, window=n, min_periods=n, **params)


# ---- 输入数据（CONVENTIONS 1.1） -------------------------------------------------

_factor = field("adj_factor")
open = field("open") * _factor
high = field("high") * _factor
low = field("low") * _factor
close = field("close") * _factor
vwap = field("vwap") * _factor
volume = field("volume") / _factor
amount = field("amount")
# 上市首日的 previous_close 是发行价，不是前一天的收盘价：收益为 NaN（D-03）。
returns = ops.keep_where(
    call("ne", field("listing_age"), 1),
    field("close") / field("previous_close") - 1,
)
cap = field("total_market_cap")
benchmark_open = field("benchmark_open", broadcast=True)
benchmark_close = field("benchmark_close", broadcast=True)

INDUSTRY_LEVELS = {
    "sector": "industry_l1",
    "industry": "industry_l1",
    "subindustry": "industry_l1",
}


def adv(d):
    """adv{d}：d 日平均成交额（元），窗口必须满。"""
    return _rolling("ts_mean", amount, d)


def industry(level):
    """IndClass.level 的行业组号（int64，缺失为 −1）；现在三个层级都是申万一级（D-06）。"""
    if level not in INDUSTRY_LEVELS:
        raise ValueError(f"未知行业层级：{level!r}，可选 {sorted(INDUSTRY_LEVELS)}")
    return field(INDUSTRY_LEVELS[level], dtype="int64")


# ---- 逐元素 ---------------------------------------------------------------------


def abs(x):
    return call("abs", x)


def log(x):
    return call("log", x)


def sign(x):
    return call("sign", x)


def sqrt(x):
    return call("sqrt", x)


def power(x, y):
    """原文 ``x ^ y``：实数幂，负数的非整数次方为 NaN。"""
    return call("power", x, y)


def signed_power(x, a):
    """SignedPower(x, a) = sign(x)·|x|^a；a 可以是序列。"""
    return call("signed_power", x, a)


def maximum(x, y):
    """逐元素取大（两个序列参数的 max/MAX）。"""
    return call("maximum", x, y)


def minimum(x, y):
    """逐元素取小。"""
    return call("minimum", x, y)


def lt(x, y):
    return call("lt", x, y)


def gt(x, y):
    return call("gt", x, y)


def le(x, y):
    return call("le", x, y)


def ge(x, y):
    return call("ge", x, y)


def eq(x, y):
    return call("eq", x, y)


def ne(x, y):
    return call("ne", x, y)


def and_(x, y):
    return call("and_", x, y)


def or_(x, y):
    return call("or_", x, y)


def not_(x):
    return call("not_", x)


def where(condition, if_true, if_false):
    """原文 ``cond ? a : b``；条件 NaN 时为 NaN。"""
    return call("where", condition, if_true, if_false)


# ---- 时间序列（窗口必须满） ------------------------------------------------------


def delay(x, d):
    """d 行之前的值；d 向下取整，可以为 0。"""
    return call("ts_delay", x, periods=days(d))


def delta(x, d):
    """x − delay(x, d)。"""
    return call("ts_delta", x, periods=days(d))


def ts_sum(x, d):
    return _rolling("ts_sum", x, d)


def ts_mean(x, d):
    return _rolling("ts_mean", x, d)


def stddev(x, d):
    """样本标准差（分母 d−1）。"""
    return _rolling("ts_std", x, d, ddof=1)


def ts_min(x, d):
    return _rolling("ts_min", x, d)


def ts_max(x, d):
    return _rolling("ts_max", x, d)


def ts_rank(x, d):
    """时序百分位：当天值在 d 日窗口内的平均秩 / d，取值 (0, 1]（D-12）。"""
    n = _window(d)
    return ops.ts_pct_rank(x, n, n)


# D-13：ts_argmax/ts_argmin 的两种读法。翻译只写 d.ts_argmax，读法由注册表选。
ARGMAX_READINGS = ("days_ago", "position")
_argmax_reading = contextvars.ContextVar("argmax_reading", default="days_ago")


@contextlib.contextmanager
def argmax_reading(reading):
    """在 with 块内构造的 ts_argmax/ts_argmin 采用指定读法（注册表构造时用）。"""
    if reading not in ARGMAX_READINGS:
        raise ValueError(f"未知读法：{reading!r}，可选 {ARGMAX_READINGS}")
    token = _argmax_reading.set(reading)
    try:
        yield
    finally:
        _argmax_reading.reset(token)


def _arg(kind, x, d):
    if _argmax_reading.get() == "days_ago":
        return _rolling(f"ts_arg{kind}", x, d)
    return _rolling(f"ts_idx{kind}", x, d, ties="earliest")


def ts_argmax(x, d):
    """WorldQuant 式 ts_argmax（D-13）。

    读法 A（默认，``days_ago``）：最大值出现在几天前，今天为 0，并列取最近一次。
    读法 B（``position``）：窗口内位置，最早一天为 1，并列取最早一次。
    """
    return _arg("max", x, d)


def ts_argmin(x, d):
    """WorldQuant 式 ts_argmin，读法同 ``ts_argmax``。"""
    return _arg("min", x, d)


def ts_product(x, d):
    return _rolling("ts_product", x, d)


def correlation(x, y, d):
    """成对 Pearson 相关；任一序列窗口内方差为 0 时为 NaN。"""
    n = _window(d)
    return call("ts_corr", x, y, window=n, min_periods=n)


def covariance(x, y, d):
    """样本协方差（分母 d−1）。"""
    n = _window(d)
    return call("ts_cov", x, y, window=n, min_periods=n, ddof=1)


def decay_linear(x, d):
    """线性衰减加权平均：今天权重 d，最早一天权重 1，除以权重和。"""
    return _rolling("ts_decay_linear", x, d)


def ts_count(condition, d):
    """国君 COUNT(cond, n)：窗口内条件为真的天数。"""
    return _rolling("ts_count", condition, d)


def ts_sumif(x, condition, d):
    """国君 SUMIF(A, n, cond)：注意参数顺序与原文不同（原文是 A, n, cond）。"""
    n = _window(d)
    return call("ts_sumif", x, condition, window=n, min_periods=n)


def sma(x, n, m):
    """国君 SMA(A, n, m)：Y_t = (m·A_t + (n−m)·Y_{t−1}) / n（D-10）。"""
    n_, m_ = _window(n), m
    if isinstance(m_, bool) or not isinstance(m_, (int, float)) or not 0 < m_ <= n_:
        raise ValueError(f"SMA 要求 0 < m ≤ n：n={n!r}, m={m!r}")
    return ops.ewm(x, n_, m_, n_)


def wma(x, n):
    """国君 WMA(A, n)：权重 0.9^i（今天 i=0），除以权重和。"""
    return ops.decayed_mean(x, _window(n))


def highday(x, n):
    """国君 HIGHDAY：最大值距今天数（今天 0，并列取最近一次，D-14）。"""
    return _rolling("ts_argmax", x, n)


def lowday(x, n):
    """国君 LOWDAY：最小值距今天数（今天 0，并列取最近一次，D-14）。"""
    return _rolling("ts_argmin", x, n)


def regbeta(y, x, n, *, min_pairs=None):
    """国君 REGBETA(A, B, n)：A 对 B 加截距回归的斜率。

    默认窗口必须满；``min_pairs`` 只给 149 用（Q-G-149：窗口内有效配对不少于 20 对）。
    """
    w = _window(n)
    if min_pairs is None:
        need = w
    elif (
        isinstance(min_pairs, bool)
        or not isinstance(min_pairs, int)
        or not 2 <= min_pairs <= w
    ):
        raise ValueError(f"min_pairs 须是 2 到 {w} 之间的整数：{min_pairs!r}")
    else:
        need = min_pairs
    return call("ts_regression_slope", y, x, window=w, min_periods=need)


def regresi(y, x, n):
    """国君 REGRESI(A, B, n)：A 对 B 加截距回归在当天的残差。"""
    w = _window(n)
    return call("ts_regression_residual", y, x, window=w, min_periods=w)


def ts_slope(x, n):
    """国君 REGBETA(A, SEQUENCE(n), n)：对时间回归的斜率（每行）。"""
    return _rolling("ts_slope", x, n)


def ts_residual(x, n):
    """国君 REGRESI(A, SEQUENCE(n), n)：对时间回归在当天的残差。"""
    return _rolling("ts_residual", x, n)


def mask_missing(x, ref):
    """``ref`` 缺失的格子 ``x`` 也缺失，其余原样（逐位不变）。

    给「当天某个输入缺失时整条因子输出缺失」这类裁决用，例如股票当天停牌时
    回归斜率也不出数：``d.mask_missing(d.regbeta(...), d.returns)``。
    """
    return ops.keep_where(ops.valid(ref), x)


def filter_(x, condition):
    """国君 FILTER(A, cond)：条件不成立的日子为 NaN。"""
    return ops.filter_out(x, condition)


# ---- 截面 -----------------------------------------------------------------------


def rank(x):
    """截面百分位：平均秩 / 当天有效个数，取值 (0, 1]（D-11）。"""
    return ops.cs_pct_rank(x)


def scale(x, a=1):
    """截面上除以有效值绝对值之和再乘 a。"""
    scaled = call("cs_scale", x)
    return scaled if a == 1 else scaled * a


def indneutralize(x, level):
    """行业内去均值；level 为 "sector"、"industry"、"subindustry"。"""
    return call("cs_group_demean", x, industry(level))


# ---- 国君的派生变量（研报定义表） --------------------------------------------------

dtm = where(
    le(open, delay(open, 1)),
    0,
    maximum(high - open, open - delay(open, 1)),
)
dbm = where(
    ge(open, delay(open, 1)),
    0,
    maximum(open - low, open - delay(open, 1)),
)
tr = maximum(
    maximum(high - low, abs(high - delay(close, 1))),
    abs(low - delay(close, 1)),
)
hd = high - delay(high, 1)
ld = delay(low, 1) - low
