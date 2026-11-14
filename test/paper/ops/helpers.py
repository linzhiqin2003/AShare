"""模拟盘每日运行（2026-11-14 新增）的测试替身：假时钟、假 Tushare 客户端、合成日历。

全部是合成数据：不联网、不读真实行情和真实账户。
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path

from ashare.paper.ops.calendar import TradingCalendar
from ashare.paper.ops.context import Context

from ..test_sources import FakeClock

__all__ = ["FakeClock", "FakeClient", "calendar", "context", "weekday_calendar"]


def context(tmp_path: Path, now: str, data: Path | None = None) -> Context:
    return Context(
        (data or tmp_path / "data").resolve(), tmp_path / "paper", FakeClock(now)
    )


def weekday_calendar(start: str, end: str, holidays=()) -> TradingCalendar:
    """工作日开市、周末与给定节假日休市的合成日历。"""
    day = date(int(start[:4]), int(start[4:6]), int(start[6:]))
    last = date(int(end[:4]), int(end[4:6]), int(end[6:]))
    rows = []
    while day <= last:
        text = day.strftime("%Y%m%d")
        rows.append((text, day.weekday() < 5 and text not in holidays))
        day += timedelta(days=1)
    return TradingCalendar(rows)


def calendar(open_days, known_until: str) -> TradingCalendar:
    """只有 ``open_days`` 开市、其余到 ``known_until`` 为止都已知休市。"""
    first = min(open_days)
    days = weekday_calendar(first, known_until).known
    opened = set(open_days)
    return TradingCalendar([(d, d in opened) for d in days])


class FakeClient:
    """模拟盘运行器要用的几个接口；价格固定、成交量每次轮询递增。"""

    def __init__(self, clock, price=10.0):
        self.clock, self.price = clock, price
        self.calls: list[str] = []
        self.volume = 0
        self.closed = False

    def symbols(self, params):
        code = params.get("ts_code", "")
        return [c for c in code.split(",") if c and "*" not in c]

    def call(self, api_name, params=None, fields=""):
        params = params or {}
        self.calls.append(api_name)
        p = self.price
        if api_name == "rt_k":
            self.volume += 1_000_000
            return [
                dict(
                    ts_code=s,
                    close=p,
                    open=p,
                    high=p,
                    low=p,
                    pre_close=p,
                    vol=self.volume,
                    amount=self.volume * p,
                    trade_time=self.clock.now().isoformat(),
                    ask_price1=p,
                    ask_volume1=1_000_000,
                    bid_price1=p,
                    bid_volume1=1_000_000,
                )
                for s in self.symbols(params)
            ]
        if api_name == "stk_limit":
            return [
                dict(
                    ts_code=s, up_limit=round(p * 1.1, 2), down_limit=round(p * 0.9, 2)
                )
                for s in self.universe
            ]
        if api_name in ("suspend_d", "dividend"):
            return []
        if api_name == "daily":
            return [
                dict(
                    ts_code=s,
                    close=p,
                    open=p,
                    vol=self.volume / 100,
                    amount=self.volume * p / 1000,
                )
                for s in self.universe
            ]
        if api_name == "stk_auction":
            return [
                dict(ts_code=s, price=p, vol=1000, amount=1000 * p)
                for s in self.universe
            ]
        raise AssertionError(f"运行阶段不应调用 {api_name}")

    universe: list[str] = []

    @contextmanager
    def session(self):
        yield self
        self.closed = True


def client_factory(client: FakeClient, seen=None):
    """``run(client_factory=…)`` 要的工厂：记下锁路径，返回上下文管理器。"""

    def factory(lock, clock):
        if seen is not None:
            seen.append(lock)
        client.clock = clock
        return client.session()

    return factory
