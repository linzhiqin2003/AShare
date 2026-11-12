"""特质风险的三个指数加权方差改为分块累加（P-006，允许相对误差 1e-12）。

从头计算（`specific_incremental=False`）每天对整段 T×N 残差窗口重算
base（h=84、无滞后）、corr0（h=252、无滞后）和 corrnw（h=252、NW 滞后），
窗口只前进一天，98% 的工作是上一天做过的。

这里把方差拆成每列的加权矩（权重和、移位后的一阶与二阶，以及每个滞后的成对权重、
两侧一阶和交叉积，见 docs/perf_log.d/P-006-specific-incremental.md），
并把窗口切成按绝对日期对齐的 BLOCK 天一块：

    窗口矩 = 衰减·最老的残块（每天现算）+ 衰减·中间整块之和（每块只算一次）
             + 当前未满的块（每天现算）

所有项都以正权重相加，没有「减掉出界项」，因此没有递推漂移；结果只与这一天的窗口和
移位有关，与是全量重建还是追加无关，逐位相同。移位（锚点日窗口的等权均值）每
`specific_reanchor_days` 天按绝对日期重取一次，同时清空全部已算好的块，相当于整体从头重算。

守容差的三道闸（命中的列改用从头算，同一个内核，逐位等于从头算模式）：
- 条件数：每列给出各项绝对值之和 M，`var·TAU < M`（抵消吃掉过多有效位）或 `var ≤ 0`；
- 均值：|均值| > MEAN_RATIO·标准差时从头算本身的舍入就超过 1e-12，跟从头算走；
- 阈值：base、corr0 与 `variance_floor` 相对差在 1e-9 以内，保证分支判断一致。
"""

import numpy as np
import portopt

from . import kernels
from .engine import _short_factor_panel
from .window import ResidualWindow, WindowVariances

BLOCK = 32
TAU = 32.0
MEAN_RATIO = 64.0
THRESHOLD_GUARD = 1e-9


def direct_variances(window, config):
    """从头算的三个方差，与旧实现同一调用（N≤64 走 numpy 参考内核，否则 Rust）。"""
    from .engine import ew_variance

    c = config
    base = ew_variance(window, c.specific_half_life)
    corr0 = ew_variance(window, c.specific_corr_half_life)
    corrnw = ew_variance(
        window, c.specific_corr_half_life, min(c.specific_lags, len(window) - 1)
    )
    return base, corr0, corrnw


def reference_variances(window, config):
    """numpy 参考：与 direct_variances 同义的独立实现，只供测试对照。"""
    c = config
    return (
        kernels.ew_variance(window, c.specific_half_life),
        kernels.ew_variance(window, c.specific_corr_half_life),
        kernels.ew_variance(
            window, c.specific_corr_half_life, min(c.specific_lags, len(window) - 1)
        ),
    )


class SpecificMoments:
    """最近 history 天残差（双份环形缓冲，窗口是连续视图）、移位和已算好的整块矩。"""

    def __init__(self, n, config, ring=None):
        c = config
        self.config = c
        self.n = n
        self.capacity = c.history
        self.ring = ResidualWindow(n, c.history) if ring is None else ring
        if self.ring.capacity != c.history:
            raise ValueError("残差缓冲长度须等于 history")
        self.anchor = None
        self.shift = None
        self.blocks = {}  # (块号, 半衰期) → 块末端的矩
        self.sums = {}  # (首块, 末块, 半衰期) → 以末块末端为参照的整块之和
        self.count = None
        self.count_day = None
        self.fallback_columns = 0  # 诊断：累计改用从头算的列数

    @property
    def latest(self):
        return self.ring.latest

    # 残差窗口 ------------------------------------------------------------

    def push(self, day, residual):
        """推入第 day 天残差（缓冲由调用方共享时，调用方先推、再调 `counted`）。"""
        self.counted(day, residual)
        self.ring.push(day, residual)

    def counted(self, day, residual):
        if self.count_day == day - 1 and self.count is not None:
            # 有效天数是整数，逐日加减精确；出界的一行在覆盖前减掉。
            leaving = self.ring.leaving(day)
            if leaving is not None:
                self.count -= np.isfinite(leaving)
            self.count += np.isfinite(residual)
            self.count_day = day

    def window(self, first, last):
        return self.ring.window(first, last)

    # 矩 -----------------------------------------------------------------

    def _sets(self):
        c = self.config
        # 两组半衰期都按最大滞后建矩，base 和 corr0 只取滞后 0 部分。
        return (c.specific_half_life, c.specific_corr_half_life)

    def _reanchor(self, day):
        # 移位按绝对日期每 period 天重取；period 不超过 history，这样移位所用的
        # 「锚点日之前 history−period+1 天」在这一段里任何一天的窗口内都还在。
        period = min(self.config.specific_reanchor_days, self.capacity)
        anchor = day - day % period
        if anchor == self.anchor:
            return
        window = self.window(max(0, anchor + period - self.capacity), anchor)
        finite = np.isfinite(window)
        count = finite.sum(axis=0)
        total = np.where(finite, window, 0.0).sum(axis=0)
        self.shift = np.divide(total, count, out=np.zeros(self.n), where=count > 0)
        self.anchor = anchor
        self.blocks.clear()
        self.sums.clear()

    def _segment(self, window, first, last, floor, half_life):
        return portopt.ew_moments_segment(
            window,
            first,
            last,
            self.shift,
            half_life,
            self.config.specific_lags,
            floor,
        )

    def _moments(self, day, half_life):
        """第 day 天窗口的矩，按「残块 + 整块之和 + 当前块」拼出。"""
        lags = self.config.specific_lags
        start = max(0, day - self.capacity + 1)
        window = self.window(start, day)
        current = day - day % BLOCK
        # 最老的残块要延伸到覆盖 start+lags，后面整块的滞后对才都落在窗口内。
        edge = (start + lags) - (start + lags) % BLOCK + BLOCK - 1
        if edge + 1 >= current:
            return self._segment(window, 0, day - start, 0, half_life)
        decay = lambda age: np.exp2(-age / half_life)  # noqa: E731
        head = self._segment(window, 0, edge - start, 0, half_life)
        tail = self._segment(window, current - start, day - start, 0, half_life)
        result = decay(day - edge) * head + tail
        first, last = (edge + 1) // BLOCK, current // BLOCK - 1
        if first <= last:
            key = (first, last, half_life)
            if key not in self.sums:
                end = (last + 1) * BLOCK - 1
                total = np.zeros_like(head)
                for k in range(first, last + 1):
                    block = self.blocks.get((k, half_life))
                    if block is None:
                        block = self._segment(
                            window,
                            k * BLOCK - start,
                            (k + 1) * BLOCK - 1 - start,
                            0,
                            half_life,
                        )
                        self.blocks[(k, half_life)] = block
                    total += decay(end - ((k + 1) * BLOCK - 1)) * block
                self.sums = {k: v for k, v in self.sums.items() if k[0] >= first}
                self.sums[key] = total
                for k in [k for k in self.blocks if k[0] < first]:
                    del self.blocks[k]
            result += decay(day - ((last + 1) * BLOCK - 1)) * self.sums[key]
        return result

    def _count(self, day):
        if self.count_day != day:
            start = max(0, day - self.capacity + 1)
            self.count = np.isfinite(self.window(start, day)).sum(axis=0)
            self.count_day = day
        return self.count

    def variances(self, day):
        """第 day 天窗口（最近 history 天）的 base、corr0、corrnw 和有效天数。"""
        c = self.config
        if day != self.latest:
            raise ValueError("只能求最新一天的窗口")
        self._reanchor(day)
        length = day - max(0, day - self.capacity + 1) + 1
        used = min(c.specific_lags, length - 1)
        fast = self._moments(day, c.specific_half_life)
        slow = fast if c.specific_half_life == c.specific_corr_half_life else None
        if slow is None:
            slow = self._moments(day, c.specific_corr_half_life)
        values = []
        for state, half_life, lag_used in (
            (fast, c.specific_half_life, 0),
            (slow, c.specific_corr_half_life, 0),
            (slow, c.specific_corr_half_life, used),
        ):
            variance, magnitude = portopt.ew_moments_variance(
                state, self.n, c.specific_lags, lag_used, half_life
            )
            moments = state.reshape(self.n, -1)
            mean = self.shift + np.divide(
                moments[:, 1],
                moments[:, 0],
                out=np.zeros(self.n),
                where=moments[:, 0] > 0,
            )
            values.append((variance, magnitude, moments[:, 0] > 0, mean))
        bad = np.zeros(self.n, bool)
        for variance, magnitude, observed, mean in values:
            bad |= observed & (
                (variance <= 0)
                | (variance * TAU < magnitude)
                | (mean * mean > MEAN_RATIO**2 * variance)
            )
        base, corr0, corrnw = (np.maximum(v[0], 0.0) for v in values)
        floor = c.variance_floor
        for v in (base, corr0):
            bad |= np.abs(v - floor) <= THRESHOLD_GUARD * floor
        if bad.any():
            columns = np.flatnonzero(bad)
            self.fallback_columns += len(columns)
            window = self.window(max(0, day - self.capacity + 1), day)
            if _short_factor_panel(window):
                # 从头算模式对整块窗口用 numpy 参考内核；同样整块算再取列。
                exact = [v[columns] for v in direct_variances(window, c)]
            else:
                # 从头算模式走 Rust，逐列独立：只算这些列，结果与整块算逐位相同。
                part = np.ascontiguousarray(window[:, columns])
                lags = min(c.specific_lags, len(part) - 1)
                exact = [
                    portopt.ew_variance(part, c.specific_half_life, 0),
                    portopt.ew_variance(part, c.specific_corr_half_life, 0),
                    portopt.ew_variance(part, c.specific_corr_half_life, lags),
                ]
            for target, value in zip((base, corr0, corrnw), exact, strict=True):
                target[columns] = value
        return WindowVariances(base, corr0, corrnw, self._count(day).copy())
