"""按交易日历及可配置时段调度；时钟由调用方注入。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time

from .market import TZ


@dataclass(frozen=True)
class SessionTable:
    preopen: time = time(9, 0)
    open_auction: time = time(9, 15)
    auction_result: time = time(9, 26)
    continuous: time = time(9, 30)
    lunch: time = time(11, 30)
    afternoon: time = time(13)
    close_auction: time = time(14, 57)
    close: time = time(15)


class Schedule:
    def __init__(self, calendar: list[dict], table: SessionTable | None = None):
        self.open_days: set[str] = set()
        self.known_days: set[str] = set()
        self.update(calendar)
        self.table = table or SessionTable()

    def update(self, calendar: list[dict]) -> None:
        for r in calendar:
            self.known_days.add(r["cal_date"])
            if int(r["is_open"]) == 1:
                self.open_days.add(r["cal_date"])
            else:
                self.open_days.discard(r["cal_date"])

    def phase(self, now: datetime) -> str:
        now = now.astimezone(TZ)
        if now.strftime("%Y%m%d") not in self.open_days:
            return "closed"
        t, table = now.time().replace(tzinfo=None), self.table
        if t < table.preopen or t >= table.close:
            return "closed"
        if t < table.open_auction:
            return "preopen"
        if t < table.continuous:
            return "open_auction"
        if t < table.lunch:
            return "continuous"
        if t < table.afternoon:
            return "break"
        if t < table.close_auction:
            return "continuous"
        return "close_auction"
