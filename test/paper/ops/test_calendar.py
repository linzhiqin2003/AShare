"""交易日历与调仓周期（2026-11-14 新增，模拟盘每日运行）。

信号日的判定与回测页 ``signal_days`` 逐日对照：模拟盘和回测必须在同一批日子调仓。
"""

from datetime import datetime

import numpy as np
import pytest

from ashare.paper.market import TZ
from ashare.paper.ops.calendar import REBALANCE, TradingCalendar, period_key
from ashare.platform.backtest.engine import period_keys, signal_days

from .helpers import calendar, weekday_calendar

# 合成节假日：春节一周、国庆一周、元旦；跨年、跨季、月末落在节假日上的情形都有。
HOLIDAYS = {
    *(f"202502{d:02d}" for d in range(1, 5)),
    *(f"202510{d:02d}" for d in range(1, 9)),
    "20260101",
    "20260102",
    "20250331",  # 季末最后一天休市
}


class Dates:
    def __init__(self, dates):
        self.dates = np.asarray(dates)


@pytest.mark.parametrize("rebalance", REBALANCE)
def test_signal_days_match_backtest_page_parity(rebalance):
    """每个开市日（除最后一行和回测窗口首日外）是否为信号日，与回测页逐日相同。"""
    cal = weekday_calendar("20241201", "20260331", HOLIDAYS)
    opened = cal.open
    expected = set(signal_days(Dates(opened), 0, len(opened), rebalance).tolist())
    for i, day in enumerate(opened[:-1]):
        if i == 0:
            continue  # 回测把窗口首日也当信号日（建仓），模拟盘由账户的首个信号负责。
        assert cal.is_period_end(day, rebalance) == (i in expected), (day, rebalance)


@pytest.mark.parametrize("rebalance", REBALANCE)
def test_period_key_is_backtest_period_keys(rebalance):
    days = weekday_calendar("20241225", "20250110").known
    if rebalance == "day":
        assert [period_key(d, "day") for d in days] == days
    else:
        assert [period_key(d, rebalance) for d in days] == period_keys(
            np.array(days), rebalance
        ).tolist()
    with pytest.raises(ValueError):
        period_key(days[0], "fortnight")


def test_period_end_unknown_until_calendar_covers_the_period():
    # 1 月 15 日（周四）之后没有已知开市日：日历只到 1 月 20 日时月末判断不了，到 1 月底就能。
    short = calendar(["20260114", "20260115"], "20260120")
    assert short.is_period_end("20260115", "month") is None
    assert short.is_period_end("20260115", "week") is True  # 本周（到 18 日）已全部已知
    assert short.is_period_end("20260115", "day") is True
    assert short.next_period_end("20260114", "month") is None
    full = calendar(["20260114", "20260115"], "20260131")
    assert full.is_period_end("20260115", "month") is True
    assert full.is_period_end("20260114", "month") is False
    assert full.next_period_end("20260114", "month") == "20260115"
    assert full.next_period_end("20260115", "month") is None  # 之后没有开市日


def test_latest_closed_open_between_and_unknown_days():
    cal = weekday_calendar("20261101", "20261130")
    at = lambda text: datetime.fromisoformat(text).replace(tzinfo=TZ)  # noqa: E731
    # 周三收盘前是周二，收盘后是周三；周六是周五；周一早上是上周五。
    assert cal.latest_closed(at("2026-11-11T14:59:59")) == "20261110"
    assert cal.latest_closed(at("2026-11-11T15:00:00")) == "20261111"
    assert cal.latest_closed(at("2026-11-14T10:00:00")) == "20261113"
    assert cal.latest_closed(at("2026-11-16T07:30:00")) == "20261113"
    assert cal.open_between("20261111", "20261116") == ["20261112", "20261113"]
    assert cal.is_open("20261114") is False
    assert cal.is_open("20261201") is None  # 日历以外：未知，不猜
    assert cal.next_open("20261127") == "20261130"
    assert cal.next_open("20261130") is None
    assert cal.rows()[0] == dict(cal_date="20261101", is_open=0)


def test_from_store_reads_raw_trade_calendar(tmp_path):
    """原始库里的上交所日历（合成数据目录）。"""
    from test.fixtures.synthetic_datadir import cached_datadir

    info = cached_datadir()
    cal = TradingCalendar.from_store(info.root)
    assert cal.open == sorted(info.dates)
    assert cal.last_known >= info.dates[-1]
