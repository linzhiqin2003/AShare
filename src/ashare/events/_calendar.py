"""公告日到第一个可成交日：信号日经 ``ashare.backtest.timing.execution_days`` 错一天。

``docs/DESIGN.md`` 4.3：``panelbt`` 不做任何移位，信号与成交的错位只在 ``ashare`` 里做，
整个项目共用 ``execution_days``。
"""

import numpy as np

from ashare.backtest.timing import execution_days


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
