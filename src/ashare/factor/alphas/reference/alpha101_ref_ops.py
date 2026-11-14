"""Alpha101 参考实现的算子：直接用 NumPy / pandas 写，不经过 ``Expr``、``dsl`` 和 ``panelops``。

所有算子都接收、返回 T × N 的 ``numpy.ndarray``（行为交易日，列为证券），口径照
``docs/CONVENTIONS.md`` 与负责人裁决：

- 滚动窗口必须满：第 t 行的 d 日窗口是第 t−d+1..t 行，窗口里任何一格 NaN 结果就是 NaN，
  最前面 d−1 行也是 NaN；天数一律先 ``math.floor``（D-15）。
- ``rank``：截面平均秩 / 当天有效个数，取值 (0, 1]（D-11）；``ts_rank``：当天值在窗口内的
  平均秩 / d（D-12）。
- ``ts_argmax`` / ``ts_argmin``：读法 ``"days_ago"``（距今天数，今天 0，并列取最近）或
  ``"position"``（窗口内位置，最早一天为 1，并列取最早），见 D-13。
- ``correlation``：任一侧窗口逐值相等（方差为 0）时 NaN；标准差、协方差分母 n−1。
- 逐元素运算的非有限结果（除以零、log 非正数、负数的非整数次方）为 NaN；比较、逻辑、
  三目运算遇 NaN 结果为 NaN。

滚动算子用 ``sliding_window_view`` 把窗口摊成最后一维再归约，按列分块控制内存；每个归约
都有朴素循环版本在测试里对拍（``tests/reference_alpha101``）。
"""

import math

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

# 每个分块里窗口数组（T × 列数 × d 个 float64）的大致上限，字节。
_BLOCK_BYTES = 48 * 1024 * 1024

ARGMAX_READINGS = ("days_ago", "position")


def window(d):
    """非整数天数向下取整（原文附录：non-integer number of days d is converted to floor(d)）。"""
    w = math.floor(d)
    if w < 1:
        raise ValueError(f"窗口 {d} 取整后小于 1")
    return w


def _wsum(w):
    """窗口内求和（最后一维）：先排序再用 NumPy 的成对求和。

    同一组数不论在窗口里的先后顺序如何，都得到逐位相同的和：滑动窗口里进出的值相同、
    两个窗口和数学上相等时，不会因为相加顺序不同而差 1 ulp（负责人 2026-11-13 裁决）。
    """
    return np.sort(w, axis=-1).sum(axis=-1)


def clean(x):
    """非有限值（±inf）换成 NaN；返回新的 float 数组。"""
    x = np.array(x, dtype=float)
    x[~np.isfinite(x)] = np.nan
    return x


# ---------------------------------------------------------------- 逐元素


def div(a, b):
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return clean(np.divide(a, b))


def log(x):
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.asarray(x, dtype=float)
        out = np.full(x.shape, np.nan)
        positive = x > 0
        out[positive] = np.log(x[positive])
        return out


def power(x, y):
    """实数幂 x^y；负数的非整数次方、0 的负数次方等非有限结果为 NaN。

    任一操作数缺失则结果缺失：``np.power`` 给出的 ``1 ** NaN = 1``、``NaN ** 0 = 1`` 都改成
    NaN（负责人 2026-11-13 裁决）。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        out = np.power(x, y)
    return clean(np.where(np.isnan(x) | np.isnan(y), np.nan, out))


def signedpower(x, a):
    """sign(x)·|x|^a，a 可以是数组；任一操作数缺失则结果缺失（同 ``power``）。"""
    x = np.asarray(x, dtype=float)
    a = np.asarray(a, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        out = np.sign(x) * np.power(np.abs(x), a)
    return clean(np.where(np.isnan(x) | np.isnan(a), np.nan, out))


def sign(x):
    return np.sign(np.asarray(x, dtype=float))


def abs_(x):
    return np.abs(np.asarray(x, dtype=float))


def _nan_mask(*xs):
    mask = None
    for x in xs:
        m = np.isnan(np.asarray(x, dtype=float))
        mask = m if mask is None else (mask | m)
    return mask


def _compare(result, a, b):
    out = np.asarray(result, dtype=float)
    out = np.broadcast_to(out, np.broadcast_shapes(np.shape(a), np.shape(b))).copy()
    out[np.broadcast_to(_nan_mask(a, b), out.shape)] = np.nan
    return out


def lt(a, b):
    return _compare(np.less(a, b), a, b)


def gt(a, b):
    return _compare(np.greater(a, b), a, b)


def eq(a, b):
    return _compare(np.equal(a, b), a, b)


def or_(a, b):
    """``||``：有限非零为真；任一侧 NaN 结果为 NaN。"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return _compare((a != 0) | (b != 0), a, b)


def where(cond, a, b):
    """``cond ? a : b``：条件为 NaN 时 NaN；只看被选中的分支。"""
    cond = np.asarray(cond, dtype=float)
    shape = np.broadcast_shapes(cond.shape, np.shape(a), np.shape(b))
    out = np.where(cond != 0, a, b).astype(float)
    out = np.broadcast_to(out, shape).copy()
    out[np.broadcast_to(np.isnan(cond), shape)] = np.nan
    return out


def maximum(a, b):
    """两个参数都是序列的 ``max(x, y)``：逐元素取大，NaN 传播（Q-A-01）。"""
    return np.maximum(np.asarray(a, dtype=float), np.asarray(b, dtype=float))


def minimum(a, b):
    return np.minimum(np.asarray(a, dtype=float), np.asarray(b, dtype=float))


# ---------------------------------------------------------------- 时序：移位


def delay(x, d):
    d = window(d)
    x = np.asarray(x, dtype=float)
    out = np.full(x.shape, np.nan)
    if d < x.shape[0]:
        out[d:] = x[:-d]
    return out


def delta(x, d):
    return np.asarray(x, dtype=float) - delay(x, d)


# ---------------------------------------------------------------- 时序：窗口归约


def _rolling(reducer, d, *xs):
    """把 ``reducer(w1, w2, ...)`` 作用在每个满窗口上；w 的形状是 (行, 列, d)。

    窗口里任何输入有 NaN 的格子结果为 NaN；前 d−1 行为 NaN。
    """
    xs = [np.asarray(x, dtype=float) for x in xs]
    t, n = xs[0].shape
    out = np.full((t, n), np.nan)
    if d > t:
        return out
    cols = max(1, _BLOCK_BYTES // (8 * d * max(t - d + 1, 1)))
    for start in range(0, n, cols):
        block = slice(start, min(n, start + cols))
        windows = [sliding_window_view(x[:, block], d, axis=0) for x in xs]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            value = np.asarray(reducer(*windows), dtype=float)
        bad = np.zeros(value.shape, bool)
        for w in windows:
            bad |= np.isnan(w).any(axis=-1)
        value = value.copy()
        value[bad] = np.nan
        value[~np.isfinite(value)] = np.nan
        out[d - 1 :, block] = value
    return out


def ts_sum(x, d):
    return _rolling(lambda w: _wsum(w), window(d), x)


def ts_mean(x, d):
    d = window(d)
    return _rolling(lambda w: _wsum(w) / d, d, x)


def ts_product(x, d):
    return _rolling(lambda w: w.prod(axis=-1), window(d), x)


def ts_min(x, d):
    return _rolling(lambda w: w.min(axis=-1), window(d), x)


def ts_max(x, d):
    return _rolling(lambda w: w.max(axis=-1), window(d), x)


def _centered(w):
    return w - _wsum(w)[..., None] / w.shape[-1]


def stddev(x, d):
    """样本标准差（分母 d−1），两遍算法：先减均值再平方和。"""

    def reducer(w):
        c = _centered(w)
        return np.sqrt(_wsum(c * c) / (w.shape[-1] - 1))

    return _rolling(reducer, window(d), x)


def covariance(x, y, d):
    def reducer(wx, wy):
        return _wsum(_centered(wx) * _centered(wy)) / (wx.shape[-1] - 1)

    return _rolling(reducer, window(d), x, y)


def correlation(x, y, d):
    """Pearson 相关；任一侧窗口逐值相等（方差为 0）时 NaN。"""

    def reducer(wx, wy):
        cx = _centered(wx)
        cy = _centered(wy)
        sxy = _wsum(cx * cy)
        sxx = _wsum(cx * cx)
        syy = _wsum(cy * cy)
        r = sxy / np.sqrt(sxx * syy)
        flat = (np.ptp(wx, axis=-1) == 0) | (np.ptp(wy, axis=-1) == 0)
        r[flat] = np.nan
        return r

    return _rolling(reducer, window(d), x, y)


def decay_linear(x, d):
    """权重从最早到今天为 1, 2, …, d，除以权重和。"""
    d = window(d)
    weights = np.arange(1, d + 1, dtype=float)
    return _rolling(lambda w: _wsum(w * weights) / weights.sum(), d, x)


def ts_rank(x, d):
    """当天值在窗口内的平均秩 / d，取值 (0, 1]。"""
    d = window(d)

    def reducer(w):
        today = w[..., -1:]
        less = (w < today).sum(axis=-1)
        equal = (w == today).sum(axis=-1)
        return (less + (equal + 1) / 2) / d

    return _rolling(reducer, d, x)


def _arg(x, d, how, pick):
    if how not in ARGMAX_READINGS:
        raise ValueError(f"ts_argmax 的读法只能是 {ARGMAX_READINGS}，收到 {how!r}")
    d = window(d)
    if how == "days_ago":
        # 窗口倒过来（今天在前），第一次出现的位置就是距今天数，并列取最近一次。
        return _rolling(lambda w: pick(w[..., ::-1], axis=-1), d, x)
    # 窗口内位置：最早一天为 1，并列取最早。
    return _rolling(lambda w: pick(w, axis=-1) + 1, d, x)


def ts_argmax(x, d, how="days_ago"):
    return _arg(x, d, how, np.argmax)


def ts_argmin(x, d, how="days_ago"):
    return _arg(x, d, how, np.argmin)


# ---------------------------------------------------------------- 截面


def rank(x):
    """截面百分位：平均秩 / 当天有效个数，NaN 不参与（pandas ``rank(pct=True)``）。"""
    frame = pd.DataFrame(clean(x))
    return frame.rank(axis=1, method="average", pct=True, na_option="keep").to_numpy(
        dtype=float
    )


def scale(x, a=1.0):
    """截面上 x 除以有效值绝对值之和再乘 a；绝对值和为 0（含整行缺失）时 NaN。"""
    x = clean(x)
    total = np.nansum(np.abs(x), axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = x / total * a
    return clean(np.where(total == 0, np.nan, out))


def indneutralize(x, groups):
    """当天同组有效值减去组均值；组缺失（NaN）或 x 缺失的格为 NaN。"""
    x = clean(x)
    groups = np.asarray(groups, dtype=float)
    out = np.full(x.shape, np.nan)
    labels = np.unique(groups[np.isfinite(groups)])
    codes = np.full(groups.shape, -1, dtype=np.int64)
    finite = np.isfinite(groups)
    codes[finite] = np.searchsorted(labels, groups[finite])
    k = len(labels)
    for i in range(x.shape[0]):
        valid = np.isfinite(x[i]) & (codes[i] >= 0)
        if not valid.any():
            continue
        idx = codes[i][valid]
        values = x[i][valid]
        sums = np.bincount(idx, weights=values, minlength=k)
        counts = np.bincount(idx, minlength=k)
        out[i, valid] = values - sums[idx] / counts[idx]
    return out


def adv(amount, d):
    """adv{d}：成交额（元）的 d 日均值，窗口必须满（D-05）。"""
    return ts_mean(amount, d)
