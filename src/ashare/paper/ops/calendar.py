"""交易日历与调仓周期。

日历来自数据层原始库的上交所日历（每年一个分区、整年拉取，所以含当年以后的日子）；盘后阶段
每天额外把未来 40 天所在年份同步一次，年底也能知道下一年的开市日。

调仓周期用回测页的 ``period_keys``（`ashare.platform.backtest.engine`），信号日的判断与它的
``signal_days`` 相同：每个周期在交易日历里的最后一个交易日，即「下一个开市日落在另一个周期」。
回测在面板行上判断，这里在交易日历上判断（最新一天在面板里没有下一行）；面板的行就是开市日，
两者一致。
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np

REBALANCE = ("day", "week", "month", "quarter", "year")
CLOSE = time(15, 0)


def period_key(day: str, rebalance: str):
    """单个交易日的周期键：直接用回测页的 ``period_keys``（「每日」按日期本身区分）。"""
    if rebalance not in REBALANCE:
        raise ValueError(f"调仓周期只能是 {REBALANCE}")
    if rebalance == "day":
        return day
    from ashare.platform.backtest.engine import period_keys

    return period_keys(np.array([day]), rebalance)[0].item()


def _date(day: str) -> date:
    return date(int(day[:4]), int(day[4:6]), int(day[6:]))


def _text(value: date) -> str:
    return value.strftime("%Y%m%d")


class TradingCalendar:
    """已知日子（开市或休市）与开市日；日历以外的日子一律当作未知，不猜。"""

    def __init__(self, rows):
        status = {}
        for day, is_open in rows:
            status[str(day)] = bool(is_open)
        self.known = sorted(status)
        self.open = [d for d in self.known if status[d]]
        self._status = status

    @classmethod
    def from_store(cls, data: Path) -> TradingCalendar:
        """原始库里的交易日历（数据层自己的读取，不联网）。"""
        from ashare.data.store import Store

        table = Store(data, readonly=True).read("trade_cal")
        return cls(
            zip(table["cal_date"].to_list(), table["is_open"].to_list(), strict=True)
        )

    def rows(self) -> list[dict]:
        """``ashare.paper.schedule.Schedule`` 要的格式。"""
        return [dict(cal_date=d, is_open=int(self._status[d])) for d in self.known]

    @property
    def last_known(self) -> str | None:
        return self.known[-1] if self.known else None

    def is_open(self, day: str) -> bool | None:
        return self._status.get(day)

    def next_open(self, day: str) -> str | None:
        i = bisect_right(self.open, day)
        return self.open[i] if i < len(self.open) else None

    def previous_open(self, day: str) -> str | None:
        i = bisect_left(self.open, day)
        return self.open[i - 1] if i > 0 else None

    def open_between(self, after: str, before: str) -> list[str]:
        """开区间 (after, before) 内的开市日。"""
        return self.open[
            bisect_right(self.open, after) : bisect_left(self.open, before)
        ]

    def latest_closed(self, now: datetime, close: time = CLOSE) -> str | None:
        """已经收盘的最近一个交易日：今天开市且已过 15:00 是今天，否则是前一个开市日。"""
        today = now.strftime("%Y%m%d")
        if self.is_open(today) and now.time().replace(tzinfo=None) >= close:
            return today
        return self.previous_open(today)

    def is_period_end(self, day: str, rebalance: str) -> bool | None:
        """``day`` 是否是所在周期的最后一个交易日；日历没覆盖到周期末时返回 None（未知）。"""
        if rebalance == "day":
            return True
        nxt = self.next_open(day)
        if nxt is not None:
            return period_key(day, rebalance) != period_key(nxt, rebalance)
        # 后面没有已知开市日：只有已知日历覆盖到周期最后一天（其间全部休市）才算周期末。
        boundary = _date(day) + timedelta(days=1)
        key = period_key(day, rebalance)
        while period_key(_text(boundary), rebalance) == key:
            boundary += timedelta(days=1)
        last_day = _text(boundary - timedelta(days=1))
        return True if self.last_known and self.last_known >= last_day else None

    def next_period_end(self, after: str, rebalance: str) -> str | None:
        """``after`` 之后的第一个信号日；日历不够时为 None。"""
        for day in self.open[bisect_right(self.open, after) :]:
            end = self.is_period_end(day, rebalance)
            if end is None:
                return None
            if end:
                return day
        return None
