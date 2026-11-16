"""阶段数据防火墙：只提供已完成的观测，无源数据句柄与路径接口。"""

import numpy as np
import polars as pl

from ashare.data.crypto import metrics, minute
from ashare.data.crypto.timeutil import HOUR, millis


class View:
    __slots__ = ("_symbol", "_cutoff", "_hourly", "_minute_source", "_metrics_source")

    def __init__(
        self, hourly, *, symbol, cutoff, minute_source=None, metrics_source=None
    ):
        self._symbol = symbol
        self._cutoff = millis(cutoff)
        self._hourly = hourly.filter(pl.col("time") + HOUR < self.cutoff)
        self._minute_source = minute_source
        self._metrics_source = metrics_source

    @property
    def symbol(self):
        return self._symbol

    @property
    def cutoff(self):
        return self._cutoff

    def truncate(self, cutoff):
        cutoff = millis(cutoff)
        if cutoff > self.cutoff:
            raise ValueError("防火墙不能扩大时间范围")
        return View(
            self._hourly,
            symbol=self.symbol,
            cutoff=cutoff,
            minute_source=self._minute_source,
            metrics_source=self._metrics_source,
        )

    def _bounds(self, start, end):
        a = millis(start) if start is not None else None
        b = self.cutoff if end is None else millis(end)
        if b > self.cutoff or (a is not None and (a >= self.cutoff or a > b)):
            raise ValueError("请求超出阶段数据防火墙")
        return a, b

    def hourly(self, start=None, end=None, fields=None):
        a, b = self._bounds(start, end)
        frame = self._hourly.filter(pl.col("time") + HOUR < b).with_columns(
            (pl.col("time") + HOUR).alias("available_at")
        )
        if a is not None:
            frame = frame.filter(pl.col("time") >= a)
        return (
            frame.select(["time", "available_at", *fields])
            if fields is not None
            else frame.clone()
        )

    def minutes(self, start, end=None, *, kind="klines"):
        a, b = self._bounds(start, end)
        if self._minute_source is None:
            raise ValueError("未提供分钟数据版本")
        root, version = self._minute_source
        return (
            minute.bars(self.symbol, "usdm", kind, a, b, root=root, version=version)
            .filter(pl.col("open_time") + 60000 < b)
            .with_columns((pl.col("open_time") + 60000).alias("available_at"))
        )

    def metrics(self, start, end=None):
        a, b = self._bounds(start, end)
        if self._metrics_source is None:
            raise ValueError("未提供 metrics 数据版本")
        root, version = self._metrics_source
        return metrics.read(self.symbol, a, b, root=root, version=version).filter(
            pl.col("available_at") < b
        )


def values(factor, view):
    result = factor(view)
    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError("因子必须返回 (available_at, 一维数值)")
    at, x = np.asarray(result[0]), np.asarray(result[1], dtype=np.float64)
    if (
        at.dtype.kind not in "iu"
        or at.ndim != 1
        or x.shape != at.shape
        or np.any(np.diff(at) <= 0)
        or np.any(at >= view.cutoff)
        or np.any(at % HOUR)
        or np.isinf(x).any()
    ):
        raise ValueError("因子时间/形状/防火墙输出非法")
    return at.astype(np.int64), x


def lookahead_check(factor, view, *, checks=8):
    """定义哈希决定截断日期；对可见前缀逐位比较，同时核对缺失日期。"""
    at, x = values(factor, view)
    if len(at) < 4:
        raise ValueError("前视自检至少需要四个时点")
    candidates = np.unique(at[len(at) // 4 : -1])
    if not len(candidates):
        raise ValueError("前视自检没有可截断时点")
    rng = np.random.default_rng(int(factor.definition_hash[:16], 16))
    cuts = np.unique(
        np.r_[
            candidates[[0, len(candidates) // 2, -1]],
            rng.choice(candidates, size=min(checks, len(candidates)), replace=False),
        ]
    )
    for cut in cuts:
        short_at, short_x = values(factor, view.truncate(int(cut)))
        use = at < cut
        if not np.array_equal(short_at, at[use]):
            raise ValueError("前视自检：截断改变历史日期")
        left, right = short_x.copy(), x[use].copy()
        left[np.isnan(left)] = np.nan
        right[np.isnan(right)] = np.nan
        if left.tobytes() != right.tobytes():
            raise ValueError("前视自检：截断改变历史因子值")
    return (at, x), list(map(int, cuts))
