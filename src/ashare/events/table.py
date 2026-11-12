"""事件表：证券、市场可知时点、事件类型和若干数值字段。

事件表就是一个 polars DataFrame，必需列：

- ``symbol``：证券代码（与面板一致，如 ``600000.SH``）。
- ``date``：公告的日历日期 ``YYYYMMDD``，只到天，没有时刻。
- ``kind``：事件类型（字符串）。
- ``timing``：``"after_close"`` 或 ``"pre_open"``，说明 ``date`` 当天何时可知。

其余列都是数值或说明字段，原样保留。

**可知时点到交易日**（:func:`align`）：

- ``after_close``（默认）：``date`` 当天收盘后才可知，最早在 ``date`` 之后的第一个
  交易日成交。上市公司公告多在交易所网站晚间（收盘后）或次日开盘前披露，Tushare
  只给日期；把开盘前披露的公告也按「收盘后」处理只会晚一天，不会前视。
- ``pre_open``：``date`` 当天开盘前已可知，最早在不早于 ``date`` 的第一个交易日成交。
  只在确知开盘前披露时使用（例如数据层已经错过一天的面板字段）。

``date`` 不是交易日时两种口径都顺延到其后第一个交易日。对齐后的下标 ``avail`` 是
「第一个可以用这条信息成交的交易日」，事件研究的第 0 天、事件转持仓的
``entry_lag`` 都从它算起。

最小例子::

    import numpy as np, polars as pl
    from ashare.events import align

    events = pl.DataFrame({"symbol": ["A", "B"], "date": ["20240105", "20240106"],
                           "kind": ["demo", "demo"], "timing": ["after_close"] * 2})
    dates = np.array(["20240104", "20240105", "20240108"])
    aligned = align(events, dates, np.array(["A", "B"]))
    aligned.avail   # array([2, 2])：周五盘后与周六的公告都在周一成交
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from ._calendar import first_trade_index

REQUIRED = ("symbol", "date", "kind", "timing")
TIMINGS = ("after_close", "pre_open")


def validate(events: pl.DataFrame) -> pl.DataFrame:
    """检查必需列与取值，返回按 (date, symbol, kind) 稳定排序的事件表。"""
    if isinstance(events, pl.LazyFrame):
        events = events.collect()
    missing = [c for c in REQUIRED if c not in events.columns]
    if missing:
        raise ValueError(f"事件表缺少列 {missing}；必需列为 {list(REQUIRED)}")
    for name in ("symbol", "date", "kind", "timing"):
        if events.schema[name] != pl.String:
            raise TypeError(
                f"事件表列 {name} 必须是字符串，实际为 {events.schema[name]}"
            )
        if events[name].null_count():
            raise ValueError(f"事件表列 {name} 不能有空值")
    bad = events.filter(~pl.col("timing").is_in(TIMINGS))
    if bad.height:
        raise ValueError(
            f"timing 只能是 {TIMINGS}，发现 {bad['timing'].unique().to_list()}"
        )
    bad = events.filter(~pl.col("date").str.contains(r"^\d{8}$"))
    if bad.height:
        raise ValueError(f"date 必须是 YYYYMMDD，发现 {bad['date'].head(3).to_list()}")
    return (
        events.with_row_index("_order")
        .sort(["date", "symbol", "kind", "_order"])
        .drop("_order")
    )


def with_timing(events: pl.DataFrame, overrides: dict[str, str]) -> pl.DataFrame:
    """按事件类型覆盖 ``timing``，例如 ``with_timing(ev, {"forecast": "pre_open"})``。"""
    for kind, timing in overrides.items():
        if timing not in TIMINGS:
            raise ValueError(f"timing 只能是 {TIMINGS}，收到 {timing!r}")
        events = events.with_columns(
            pl.when(pl.col("kind") == kind)
            .then(pl.lit(timing))
            .otherwise(pl.col("timing"))
            .alias("timing")
        )
    return events


@dataclass(frozen=True)
class AlignedEvents:
    """对齐到面板日历与证券表后的事件。

    ``frame`` 是保留下来的事件（加了 ``avail``、``avail_date``、``col`` 三列），
    ``avail``/``col`` 是同序的 int64 数组；``dropped`` 记被丢弃的条数与原因。
    """

    frame: pl.DataFrame
    avail: np.ndarray
    col: np.ndarray
    dropped: dict

    def __len__(self) -> int:
        return len(self.avail)

    def column(self, name: str, dtype=np.float64) -> np.ndarray:
        return self.frame[name].to_numpy().astype(dtype)

    def by_year(self) -> pl.DataFrame:
        """按可成交日所在年份的事件数。"""
        return (
            self.frame.group_by(pl.col("avail_date").str.slice(0, 4).alias("year"))
            .agg(pl.len().alias("events"))
            .sort("year")
        )


def align(events: pl.DataFrame, dates, symbols) -> AlignedEvents:
    """把事件对齐到交易日历 ``dates``（YYYYMMDD，升序）和证券列表 ``symbols``。

    丢弃：证券不在面板里的、可成交日超出面板最后一天的。可成交日早于面板第一天的
    （公告在面板开始前）也丢弃，因为信息在面板里不是「新」的。
    """
    events = validate(events)
    dates = np.asarray(dates).astype("U8")
    symbols = np.asarray(symbols).astype(str)
    if len(dates) and np.any(dates[1:] <= dates[:-1]):
        raise ValueError("dates 必须严格升序")
    order = np.argsort(symbols, kind="stable")
    sorted_symbols = symbols[order]
    sym = events["symbol"].to_numpy().astype(str)
    pos = np.searchsorted(sorted_symbols, sym)
    pos_clipped = np.minimum(pos, max(len(symbols) - 1, 0))
    known = (
        (pos < len(symbols)) & (sorted_symbols[pos_clipped] == sym)
        if len(symbols)
        else np.zeros(len(sym), bool)
    )
    col = np.where(known, order[pos_clipped], -1).astype(np.int64)
    avail = first_trade_index(
        dates,
        events["date"].to_numpy().astype("U8"),
        events["timing"].to_numpy() == "pre_open",
    )
    before = (
        events["date"].to_numpy().astype("U8") < dates[0]
        if len(dates)
        else np.ones(len(sym), bool)
    )
    beyond = avail >= len(dates)
    keep = known & ~beyond & ~before
    dropped = {
        "unknown_symbol": int((~known).sum()),
        "after_panel_end": int((known & beyond).sum()),
        "before_panel_start": int((known & ~beyond & before).sum()),
    }
    frame = events.filter(pl.Series(keep)).with_columns(
        pl.Series("avail", avail[keep]),
        pl.Series(
            "avail_date", dates[avail[keep]] if keep.any() else np.array([], "U8")
        ).cast(pl.String),
        pl.Series("col", col[keep]),
    )
    return AlignedEvents(frame, avail[keep].astype(np.int64), col[keep], dropped)


def dedupe(
    events: pl.DataFrame, by: str | None = None, keep: str = "first"
) -> pl.DataFrame:
    """同一证券、同一天、同一类型只留一条。

    ``by`` 给出数值列时按它取最大（``keep="max"``）或最小（``"min"``）；否则留原表
    中最先出现的一条。返回表多一列 ``n_same_day``：合并前同日同类的条数。
    """
    keys = ["symbol", "date", "kind"]
    counts = pl.len().over(keys).alias("n_same_day")
    events = events.with_row_index("_order").with_columns(counts)
    if by is None or keep == "first":
        order = ["_order"]
        desc = False
    else:
        if keep not in ("max", "min"):
            raise ValueError("keep 只能是 first/max/min")
        order = [by, "_order"]
        desc = [keep == "max", False]
    return (
        events.sort(order, descending=desc, nulls_last=True)
        .unique(keys, keep="first", maintain_order=True)
        .sort("_order")
        .drop("_order")
    )
