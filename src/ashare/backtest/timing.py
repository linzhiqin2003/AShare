"""信号日到成交日的错位：整个项目只在这里做一次。

约定（DESIGN 4.3）：面板的第 t 行是第 t 个交易日。第 t 日收盘后才知道的信息算出的目标，
最早在第 t + lag 行成交（默认 lag = 1，即下一交易日）。`panelbt` 不做任何移位，
`bt.Targets` 的第 t 行就是第 t 日成交时要达到的目标。

- 行号就是交易日：节假日、周末不在面板里，周五的信号落到下一个交易日那一行。
- 末尾 lag 行的信号在面板里没有成交日。默认丢掉（`tail="drop"`），要求显式报错用
  `tail="error"`。
- 成交日某只股票停牌、涨跌停买卖不了，不在这里处理：引擎按 `can_buy`/`can_sell`
  拦下，`Config(carry_blocked=True)`（默认）会在之后每天重试，直到下一个调仓行。
"""

from __future__ import annotations

import numpy as np
import panelbt

TAILS = ("drop", "error")


def _lag(lag):
    if isinstance(lag, bool) or not isinstance(lag, (int, np.integer)) or lag < 1:
        raise ValueError(
            "lag: 必须是正整数；信号只用开盘前已知信息时请直接构造 panelbt.Targets"
        )
    return int(lag)


def execution_days(signal_days, t, *, lag=1, tail="drop"):
    """信号日行号 → 成交日行号（+lag）；超出面板的按 tail 丢弃或报错。"""
    lag = _lag(lag)
    if tail not in TAILS:
        raise ValueError(f"tail: 只能是 {TAILS}")
    if isinstance(t, bool) or not isinstance(t, (int, np.integer)) or t < 0:
        raise ValueError("t: 需要面板行数（非负整数）")
    days = np.asarray(signal_days)
    if days.ndim != 1 or (days.size and days.dtype.kind not in "iu"):
        raise ValueError("signal_days: 需要一维整数行号")
    days = days.astype(np.int64)
    if days.size and (days.min() < 0 or days.max() >= t):
        raise ValueError(f"signal_days: 行号必须在 [0, {t}) 内")
    if np.any(np.diff(days) <= 0):
        raise ValueError("signal_days: 必须严格递增")
    keep = days + lag < t
    if tail == "error" and not keep.all():
        raise ValueError(
            f"信号日 {days[~keep].tolist()} 之后不足 {lag} 个交易日，面板里没有成交日"
        )
    return days[keep] + lag


def execution_targets(signal, *, t=None, lag=1, tail="drop", kind=None, validate=True):
    """把信号日上的目标移到成交日，返回可直接交给 `panelbt.run` 的 `panelbt.Targets`。

    signal：
      - 稠密 T×N 数组（行=信号日；整行 NaN 表示当日不调仓，其余 NaN 当作 0，与
        `panelbt.Targets.from_dense` 相同）；
      - 或任何带 `days`、`indptr`、`cols`、`weights` 的稀疏目标（`panelops.targets`
        的结果、行号仍是信号日的 `panelbt.Targets`），此时必须给 `t`（面板行数）。
    kind：`"weights"` 或 `"shares"`；缺省时沿用输入的 `kind`，没有则为权重。
    """
    lag = _lag(lag)
    if tail not in TAILS:
        raise ValueError(f"tail: 只能是 {TAILS}")
    if all(hasattr(signal, k) for k in ("days", "indptr", "cols", "weights")):
        if t is None:
            raise ValueError("稀疏信号需要 t=面板行数，用来判断末尾信号有没有成交日")
        kind = kind or getattr(signal, "kind", "weights")
        days = np.asarray(signal.days, dtype=np.int64)
        indptr = np.asarray(signal.indptr, dtype=np.int64)
        if indptr.shape != (len(days) + 1,):
            raise ValueError("indptr: 长度必须是 len(days) + 1")
        shifted = execution_days(days, t, lag=lag, tail=tail)
        rows = len(shifted)
        end = int(indptr[rows])
        return panelbt.Targets(
            shifted,
            indptr[: rows + 1],
            np.asarray(signal.cols)[:end],
            np.asarray(signal.weights)[:end],
            kind,
            validate=validate,
        )
    weights = np.asarray(signal, dtype=np.float64)
    if weights.ndim != 2:
        raise ValueError("signal: 需要 T×N 稠密数组或稀疏目标")
    if t is not None and t != weights.shape[0]:
        raise ValueError("t: 与稠密信号的行数不一致")
    kept = max(weights.shape[0] - lag, 0)
    if tail == "error" and not np.isnan(weights[kept:]).all():
        raise ValueError(f"最后 {lag} 行有信号，面板里没有对应的成交日")
    shifted = np.full_like(weights, np.nan)
    shifted[lag:] = weights[:kept]
    return panelbt.Targets.from_dense(
        shifted, kind=kind or "weights", validate=validate
    )
