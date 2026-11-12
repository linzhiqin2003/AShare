"""信号日到成交日的错位（包内私有，临时）。

``docs/DESIGN.md`` 4.3：``panelbt`` 不做任何移位，错位在 ``ashare`` 里处理。

TODO(合入主干后替换)：主干已有公共函数 ``ashare.backtest.timing.execution_days``
（``execution_days(signal_days, t, *, lag=1, tail="drop")``）。本 worktree 切出时还没有
这个包，所以下面的 :func:`execution_days` 按同样的语义和参数名写了一份私有副本：
信号日行号严格递增、第 t 行收盘后可知的信息最早在第 t + lag 行成交、末尾没有成交日的
信号按 ``tail`` 丢弃或报错。合入后删掉这份副本，改成
``from ashare.backtest.timing import execution_days``，:func:`first_trade_index` 不变。
"""

import numpy as np

TAILS = ("drop", "error")


def execution_days(signal_days, t, *, lag=1, tail="drop"):
    """信号日行号 → 成交日行号（+lag）；超出面板的按 ``tail`` 丢弃或报错。

    与 ``ashare.backtest.timing.execution_days`` 同口径（私有副本，见模块说明）。
    """
    if isinstance(lag, bool) or not isinstance(lag, (int, np.integer)) or lag < 1:
        raise ValueError("lag: 必须是正整数")
    if tail not in TAILS:
        raise ValueError(f"tail: 只能是 {TAILS}")
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
        raise ValueError(f"信号日 {days[~keep].tolist()} 之后不足 {lag} 个交易日")
    return days[keep] + int(lag)


def signal_index(dates, event_dates, pre_open):
    """事件的「信号日」：收盘后可知取公告日当天或之前最后一个交易日；开盘前可知取公告日
    之前（不含当天）最后一个交易日。面板开始之前为 −1。"""
    dates = np.asarray(dates).astype("U8")
    event_dates = np.asarray(event_dates).astype("U8")
    pre_open = np.asarray(pre_open, dtype=bool)
    after = np.searchsorted(dates, event_dates, side="right") - 1
    before = np.searchsorted(dates, event_dates, side="left") - 1
    return np.where(pre_open, before, after).astype(np.int64)


def first_trade_index(dates, event_dates, pre_open):
    """每个事件第一个可以成交的交易日下标 = 信号日经 ``execution_days(lag=1)`` 后的行号。

    收盘后可知 → 严格晚于公告日的第一个交易日；开盘前可知 → 不早于公告日的第一个交易日；
    公告日不是交易日时两者相同。信号日在面板之前时返回 0（由调用方按公告日判断丢弃），
    没有成交日（超出面板）时返回 ``len(dates)``。
    """
    t = len(dates)
    signal = signal_index(dates, event_dates, pre_open)
    out = np.where(signal < 0, 0, t).astype(np.int64)
    inside = signal >= 0
    uniq = np.unique(
        signal[inside]
    )  # 公共函数要求严格递增：先去重排序，再映射回每个事件
    executed = execution_days(uniq, t, lag=1, tail="drop")
    mapping = np.full(len(uniq), t, dtype=np.int64)
    mapping[: len(executed)] = executed  # 丢掉的只可能是末尾
    out[inside] = mapping[np.searchsorted(uniq, signal[inside])]
    return out
