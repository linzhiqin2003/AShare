"""读回报告数据时的形状检查：缺了文本版和页面要用的键时给出直白的说明，而不是在渲染中途报错。

只查结构（键在不在、类型对不对、日度各列长度是否一致），不查数值。补充内容的五块
（`render.EXTRA_KEYS`）是可选的：没有补充内容的旧报告照常读回。
"""

from __future__ import annotations

from .build import ReportError

_DICTS = ("meta", "summary", "daily", "monthly")
_LISTS = ("yearly", "drawdowns", "rebalances", "holdings_latest")
_OPTIONAL = {"rolling": dict, "industry_weights": dict, "attribution": list}
_SUMMARY = ("start", "first_trade_day", "end", "trading_days")
_DAILY = ("date", "nav", "return", "drawdown", "cash_weight", "turnover", "holdings")
_MONTHLY = ("month", "return", "first_partial", "last_partial")


def check_shape(data, path="报告数据") -> None:
    def bad(what):
        raise ReportError(f"{path} 不是完整的报告数据：{what}")

    if not isinstance(data, dict):
        bad("最外层应是一个对象")
    missing = [k for k in (*_DICTS, *_LISTS) if k not in data]
    if missing:
        bad(f"缺 {'、'.join(missing)}")
    for k in _DICTS:
        if not isinstance(data[k], dict):
            bad(f"{k} 应是对象")
    for k in _LISTS:
        if not isinstance(data[k], list):
            bad(f"{k} 应是列表")
    for k, t in _OPTIONAL.items():
        if data.get(k) is not None and not isinstance(data[k], t):
            bad(f"{k} 类型不对")
    for part, keys in (("summary", _SUMMARY), ("daily", _DAILY), ("monthly", _MONTHLY)):
        lack = [k for k in keys if k not in data[part]]
        if lack:
            bad(f"{part} 缺 {'、'.join(lack)}")
    dl = data["daily"]
    cols = {k: v for k, v in dl.items() if v is not None}
    for k, v in cols.items():
        if not isinstance(v, list):
            bad(f"daily.{k} 应是列表")
    n = len(dl["date"])
    if n < 2:
        bad("daily.date 少于两天")
    uneven = [k for k, v in cols.items() if len(v) != n]
    if uneven:
        bad(f"daily 里 {'、'.join(uneven)} 与 date 的长度不同")
    if not all(isinstance(d, str) for d in dl["date"]):
        bad("daily.date 应是日期字符串")
