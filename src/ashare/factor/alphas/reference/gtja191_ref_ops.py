"""国泰君安 191 参考实现的辅助函数（NumPy 写法，独立于 ``panelops`` 和表达式引擎）。

口径照 ``docs/CONVENTIONS.md``（国君 191 一列）和负责人裁决：

- 输入输出都是 T × N 的 ``float64`` 数组（行为交易日，列为证券）；基准指数是 T × 1，
  靠广播与股票数组运算。
- 逐元素运算：任一操作数 NaN 结果为 NaN；结果不是有限数（除以零、对数非正数、负数开
  非整数次方）一律为 NaN（``div``、``pw``、``log`` 负责清洗）。
- 比较、逻辑运算结果为 1.0 / 0.0，任一侧 NaN 结果为 NaN；三目运算条件为 NaN 时结果为
  NaN，只看被选中的分支。
- 滚动窗口必须满：窗口内 n 行都有效才出数，窗口含当天，按行数计；最前面 n−1 行为 NaN。
  滚动统计一律用 ``sliding_window_view`` 对每个窗口直接计算（两遍法），不用在线递推，
  按列分块控制内存。
- ``RANK``：截面平均秩 / 当天有效个数；``TSRANK``：当天值在窗口内的平均秩 / n。
- ``HIGHDAY`` / ``LOWDAY``：距今天数，今天为 0，并列取最近一次。
- ``CORR``：任一侧窗口内数值逐个完全相等时为 NaN；``STD`` 对这样的窗口给精确的 0。
- ``SMA(A, n, m)``：``Y = (m·A + (n−m)·Y_prev) / n``，首个有效值作初值；当天 NaN 时输出
  NaN、状态不变；自开头起累计不少于 n 个有效值才出数。
"""

import math

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

# 每个滑动窗口块的元素上限（约 128 MB 的 float64）。
_BLOCK_ELEMENTS = 16_000_000


def _as2d(x):
    a = np.asarray(x, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    return a


def _wsum(w):
    """窗口内求和（最后一维）：先排序再用 NumPy 的成对求和。

    同一组数不论在窗口里的先后顺序如何，都得到逐位相同的和：滑动窗口里进出的值相同、
    两个窗口和数学上相等时，不会因为相加顺序不同而差 1 ulp（负责人 2026-11-13 裁决）。
    """
    return np.sort(w, axis=-1).sum(axis=-1)


def clean(x):
    """非有限值（±inf）换成 NaN。"""
    a = np.array(x, dtype=float, copy=True)
    a[~np.isfinite(a)] = np.nan
    return a


def window(n):
    """天数参数取整：向下取整（CONVENTIONS D-15）。"""
    w = math.floor(n)
    if w < 1:
        raise ValueError(f"窗口 {n} 取整后小于 1")
    return w


# ---------------------------------------------------------------- 逐元素运算


def div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return clean(np.asarray(a, dtype=float) / np.asarray(b, dtype=float))


def pw(a, b):
    """实数幂 a^b；负数的非整数次方、0 的负数次方为 NaN。

    任一操作数缺失则结果缺失：``np.power`` 给出的 ``1 ** NaN = 1``、``NaN ** 0 = 1`` 都改成
    NaN（负责人 2026-11-13 裁决）。
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        out = np.power(a, b)
    return clean(np.where(np.isnan(a) | np.isnan(b), np.nan, out))


def log(a):
    a = np.asarray(a, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(np.where(a > 0, a, np.nan))
    return clean(out)


def absv(a):
    return np.abs(np.asarray(a, dtype=float))


def sign(a):
    return np.sign(np.asarray(a, dtype=float))


def emax(a, b):
    """逐元素取大（国君 ``MAX(A, B)`` 两个参数都是序列或 0）；NaN 传播。"""
    return np.maximum(np.asarray(a, dtype=float), np.asarray(b, dtype=float))


def emin(a, b):
    return np.minimum(np.asarray(a, dtype=float), np.asarray(b, dtype=float))


def _nan_either(a, b):
    return np.isnan(np.asarray(a, dtype=float)) | np.isnan(np.asarray(b, dtype=float))


def _compare(a, b, op):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    with np.errstate(invalid="ignore"):
        out = op(a, b).astype(float)
    return np.where(_nan_either(a, b), np.nan, out)


def gt(a, b):
    return _compare(a, b, np.greater)


def ge(a, b):
    return _compare(a, b, np.greater_equal)


def lt(a, b):
    return _compare(a, b, np.less)


def le(a, b):
    return _compare(a, b, np.less_equal)


def eq(a, b):
    return _compare(a, b, np.equal)


def and_(a, b):
    """``&``：有限非零为真；任一侧 NaN 结果为 NaN，不短路。"""
    return _compare(a, b, lambda x, y: (x != 0) & (y != 0))


def or_(a, b):
    """``||`` / ``OR``：同上。"""
    return _compare(a, b, lambda x, y: (x != 0) | (y != 0))


def iff(cond, yes, no):
    """``cond ? yes : no``：条件 NaN 时为 NaN；只取被选中的分支。"""
    cond = np.asarray(cond, dtype=float)
    yes = np.asarray(yes, dtype=float)
    no = np.asarray(no, dtype=float)
    shape = np.broadcast_shapes(cond.shape, yes.shape, no.shape)
    out = np.where(cond != 0, yes, no)
    return np.broadcast_to(np.where(np.isnan(cond), np.nan, out), shape).astype(float)


# ---------------------------------------------------------------- 移位


def delay(x, n):
    """``DELAY(A, n)``：n 行之前的值；最前面 n 行为 NaN。"""
    x = _as2d(x)
    n = window(n)
    out = np.full_like(x, np.nan)
    out[n:] = x[:-n]
    return out


def delta(x, n):
    """``DELTA(A, n) = A − DELAY(A, n)``。"""
    x = _as2d(x)
    return x - delay(x, n)


# ---------------------------------------------------------------- 截面


def rank(x):
    """``RANK``：截面平均秩 / 当天有效个数，取值 (0, 1]；NaN 不参与，输出 NaN。"""
    x = _as2d(x)
    return pd.DataFrame(x).rank(axis=1, method="average", pct=True).to_numpy(float)


# ---------------------------------------------------------------- 滚动窗口


def _rolling(arrays, n, reducer):
    """对若干同形数组的 n 行窗口逐个求 ``reducer(*windows)``。

    ``windows`` 的形状为 (T−n+1, 列数, n)，最后一维从最早一天到今天。窗口里只要有一个
    NaN 结果就是 NaN（窗口必须满）。按列分块以控制内存。
    """
    n = window(n)
    arrays = [_as2d(a) for a in arrays]
    shape = np.broadcast_shapes(*(a.shape for a in arrays))
    arrays = [np.broadcast_to(a, shape) for a in arrays]
    t, k = shape
    out = np.full(shape, np.nan)
    if t < n:
        return out
    step = max(1, _BLOCK_ELEMENTS // max(1, (t - n + 1) * n))
    for start in range(0, k, step):
        cols = slice(start, min(k, start + step))
        wins = [sliding_window_view(a[:, cols], n, axis=0) for a in arrays]
        bad = np.zeros(wins[0].shape[:2], bool)
        for w in wins:
            bad |= np.isnan(w).any(axis=-1)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            value = np.asarray(reducer(*wins), dtype=float)
        value = np.where(bad, np.nan, value)
        out[n - 1 :, cols] = value
    return clean(out)


def ts_sum(x, n):
    return _rolling([x], n, lambda w: _wsum(w))


def ts_mean(x, n):
    """``MEAN(A, n)`` = 窗口和 / n。"""
    m = window(n)
    return _rolling([x], n, lambda w: _wsum(w) / m)


def ts_max(x, n):
    return _rolling([x], n, lambda w: w.max(axis=-1))


def ts_min(x, n):
    return _rolling([x], n, lambda w: w.min(axis=-1))


def ts_prod(x, n):
    return _rolling([x], n, lambda w: w.prod(axis=-1))


def _constant(w):
    return w.max(axis=-1) == w.min(axis=-1)


def ts_std(x, n):
    """样本标准差（分母 n−1）；窗口内数值完全相等时为精确的 0。"""

    def f(w):
        m = w.shape[-1]
        mean = _wsum(w)[..., None] / m
        var = _wsum((w - mean) ** 2) / (m - 1)
        return np.where(_constant(w), 0.0, np.sqrt(var))

    if window(n) < 2:
        raise ValueError("STD 的窗口至少为 2")
    return _rolling([x], n, f)


def ts_cov(x, y, n):
    """样本协方差（分母 n−1）。"""

    def f(a, b):
        m = a.shape[-1]
        da = a - _wsum(a)[..., None] / m
        db = b - _wsum(b)[..., None] / m
        return _wsum(da * db) / (m - 1)

    return _rolling([x, y], n, f)


def ts_corr(x, y, n):
    """Pearson 相关；任一侧窗口内数值完全相等时为 NaN。"""

    def f(a, b):
        m = a.shape[-1]
        da = a - _wsum(a)[..., None] / m
        db = b - _wsum(b)[..., None] / m
        num = _wsum(da * db)
        den = np.sqrt(_wsum(da * da) * _wsum(db * db))
        r = num / den
        return np.where(_constant(a) | _constant(b), np.nan, r)

    return _rolling([x, y], n, f)


def ts_rank(x, n):
    """``TSRANK``：当天值在窗口内的平均秩 / n，取值 (0, 1]。"""

    def f(w):
        last = w[..., -1:]
        less = (w < last).sum(axis=-1)
        equal = (w == last).sum(axis=-1)
        return (less + (equal + 1) / 2) / w.shape[-1]

    return _rolling([x], n, f)


def decay_linear(x, d):
    """``DECAYLINEAR(A, d)``：权重从最早到今天为 1, 2, …, d，除以权重和。"""
    m = window(d)
    weights = np.arange(1, m + 1, dtype=float)
    return _rolling([x], d, lambda w: _wsum(w * weights) / weights.sum())


def wma(x, n):
    """``WMA(A, n)``：权重 0.9^i（i 为距今天数，今天 i=0），除以权重和。"""
    m = window(n)
    weights = 0.9 ** np.arange(m - 1, -1, -1, dtype=float)
    return _rolling([x], n, lambda w: _wsum(w * weights) / weights.sum())


def highday(x, n):
    """``HIGHDAY(A, n)``：窗口最大值距今天数（今天为 0），并列取最近一次。"""

    def f(w):
        reversed_ = w[..., ::-1]  # 第 0 个是今天；argmax 并列取第一个，即最近一次
        return np.argmax(reversed_, axis=-1).astype(float)

    return _rolling([x], n, f)


def lowday(x, n):
    """``LOWDAY(A, n)``：窗口最小值距今天数（今天为 0），并列取最近一次。"""

    def f(w):
        reversed_ = w[..., ::-1]
        return np.argmin(reversed_, axis=-1).astype(float)

    return _rolling([x], n, f)


def count(cond, n):
    """``COUNT(cond, n)``：窗口内条件为真的个数；条件 NaN 的窗口为 NaN。"""
    return _rolling([cond], n, lambda c: (c != 0).sum(axis=-1).astype(float))


def sumif(a, n, cond):
    """``SUMIF(A, n, cond)``：窗口内条件为真的 A 之和；A 和条件都要有效，全假时为 0。"""
    return _rolling([a, cond], n, lambda x, c: _wsum(np.where(c != 0, x, 0.0)))


def regbeta_seq(y, n):
    """``REGBETA(A, SEQUENCE(n), n)``：A 对 1..n 加截距回归的斜率。"""
    m = window(n)
    x = np.arange(1, m + 1, dtype=float)
    dx = x - x.sum() / m

    def f(w):
        dy = w - _wsum(w)[..., None] / m
        return _wsum(dx * dy) / (dx * dx).sum()

    return _rolling([y], n, f)


def regresi_seq(y, n):
    """``REGRESI(A, SEQUENCE(n), n)``：同一回归在当天（x = n）的残差。"""
    m = window(n)
    x = np.arange(1, m + 1, dtype=float)
    xbar = x.sum() / m
    dx = x - xbar

    def f(w):
        ybar = _wsum(w) / m
        dy = w - ybar[..., None]
        beta = _wsum(dx * dy) / (dx * dx).sum()
        alpha = ybar - beta * xbar
        return w[..., -1] - (alpha + beta * x[-1])

    return _rolling([y], n, f)


def regbeta_pairs(y, x, n, min_pairs):
    """``REGBETA(A, B, n)``，窗口内允许缺失：只用两边都有效的配对，配对数不少于
    ``min_pairs`` 才出数（裁决 Q-G-149）；B 在有效配对上完全相等时为 NaN。

    窗口按行数计；开头不满 n 行时窗口只含已有的行。
    """
    n = window(n)
    y = _as2d(y)
    x = _as2d(x)
    shape = np.broadcast_shapes(y.shape, x.shape)
    y = np.broadcast_to(y, shape)
    x = np.broadcast_to(x, shape)
    t, k = shape
    pad = np.full((n - 1, k), np.nan)
    yp = np.vstack([pad, y])
    xp = np.vstack([pad, x])
    out = np.full(shape, np.nan)
    step = max(1, _BLOCK_ELEMENTS // max(1, t * n))
    for start in range(0, k, step):
        cols = slice(start, min(k, start + step))
        wy = sliding_window_view(yp[:, cols], n, axis=0)
        wx = sliding_window_view(xp[:, cols], n, axis=0)
        ok = ~(np.isnan(wy) | np.isnan(wx))
        cnt = ok.sum(axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            yy = np.where(ok, wy, 0.0)
            xx = np.where(ok, wx, 0.0)
            ybar = _wsum(yy)[..., None] / cnt[..., None]
            xbar = _wsum(xx)[..., None] / cnt[..., None]
            dy = np.where(ok, wy - ybar, 0.0)
            dx = np.where(ok, wx - xbar, 0.0)
            beta = _wsum(dx * dy) / _wsum(dx * dx)
            xmax = np.where(ok, wx, -np.inf).max(axis=-1)
            xmin = np.where(ok, wx, np.inf).min(axis=-1)
        beta = np.where((cnt >= min_pairs) & (xmax != xmin), beta, np.nan)
        out[:, cols] = beta
    return clean(out)


# ---------------------------------------------------------------- 递推


def sma(x, n, m):
    """``SMA(A, n, m)``：``Y_t = (m·A_t + (n−m)·Y_{t−1}) / n``，首个有效值作初值。

    当天 A 为 NaN：输出 NaN，状态不变。自开头起累计有效值不少于 n 个才出数。
    """
    x = _as2d(x)
    t, k = x.shape
    out = np.full((t, k), np.nan)
    state = np.full(k, np.nan)
    seen = np.zeros(k)
    for i in range(t):
        value = x[i]
        valid = ~np.isnan(value)
        start = valid & np.isnan(state)
        with np.errstate(invalid="ignore"):
            step = (m * value + (n - m) * state) / n
        new = np.where(start, value, step)
        state = np.where(valid, new, state)
        seen = seen + valid
        out[i] = np.where(valid & (seen >= n), state, np.nan)
    return clean(out)


def filter_(a, cond):
    """``FILTER(A, cond)``：条件为假（或 NaN）的样本视为缺失。"""
    a = np.asarray(a, dtype=float)
    cond = np.asarray(cond, dtype=float)
    shape = np.broadcast_shapes(a.shape, cond.shape)
    keep = (cond != 0) & ~np.isnan(cond)
    return np.broadcast_to(np.where(keep, a, np.nan), shape).astype(float)
