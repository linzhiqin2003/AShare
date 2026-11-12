"""从数据层取报告的上下文：证券名称、行业、行业换版日期、基准、停牌掩码。

只读；不联网。
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

from .build import Benchmark, ReportError

INDEX_NAMES = {
    "000300.SH": "沪深300",
    "000905.SH": "中证500",
    "000852.SH": "中证1000",
    "000016.SH": "上证50",
    "000001.SH": "上证指数",
    "399001.SZ": "深证成指",
    "399006.SZ": "创业板指",
    "000688.SH": "科创50",
    "000985.CSI": "中证全指",
}

# 数据层的 industry_source 编码：1 = SW2014 当时口径，2 = SW2021 当时口径，3 = 回填/重述
_SOURCE_SW2021 = 2


def _compact(d) -> str:
    return str(d).replace("-", "")[:8]


def industry_switch_date(sources: np.ndarray, dates: Sequence) -> str | None:
    """行业分类换版日：第一次出现 SW2021 当时口径（industry_source == 2）的那一行。

    依据是数据层逐格记录的来源码，不从行业名猜。区间第一行就已是新口径（或从未出现）时返回 None。
    """
    s = np.asarray(sources)
    rows = np.flatnonzero((s == _SOURCE_SW2021).any(axis=1))
    if len(rows) == 0 or rows[0] == 0:
        return None
    return str(dates[rows[0]])


def context_from_data(
    data,
    dates: Sequence,
    symbols: Sequence[str],
    *,
    benchmark: str | None = None,
    benchmark_kind: str | None = None,
    benchmark_name: str | None = None,
) -> dict:
    """返回可直接传给 `build_tearsheet` 的关键字参数。

    benchmark 是 `index_daily` 里的指数代码；给了代码就必须声明 benchmark_kind
    （"price" 或 "total_return"），数据层不记录指数是否含分红。
    """
    days = [_compact(d) for d in dates]
    symbols = [str(s) for s in symbols]
    fields = ["industry", "industry_source", "is_suspended", "has_quote"]
    panel = data.panel(fields, start=days[0], end=days[-1], symbols=symbols)
    if list(panel.dates) != days:
        raise ReportError("数据层的交易日与回测日期对不上；请用同一份面板的日期")
    industry = np.asarray(panel["industry"])
    sources = np.asarray(panel["industry_source"])
    suspended = np.asarray(panel["is_suspended"]) | ~np.asarray(panel["has_quote"])

    sec = data.securities().select("symbol", "name").collect()
    names = dict(zip(sec["symbol"].to_list(), sec["name"].to_list(), strict=True))
    cls = data.table("index_classify").filter(pl.col("level") == "L1").collect()
    industry_names: dict[int, str] = {}
    for code, name, src in zip(
        cls["industry_code"].to_list(),
        cls["industry_name"].to_list(),
        cls["classification"].to_list(),
        strict=True,
    ):
        k = int(str(code).split(".")[0])
        if k not in industry_names or src == "SW2021":
            industry_names[k] = name  # 同一编码两版都有时取 SW2021 名称

    out = {
        "names": {s: names[s] for s in symbols if s in names},
        "industry": industry,
        "industry_names": industry_names,
        "industry_switch": industry_switch_date(sources, days),
        "industry_switch_labels": ("申万2014", "申万2021"),
        "suspended": suspended,
        "calendar": data.calendar()
        .filter(pl.col("is_open"))
        .collect()["date"]
        .to_list(),
    }
    if benchmark is not None:
        if benchmark_kind is None:
            raise ReportError(
                "请声明基准类型 benchmark_kind='price'（价格指数）或 'total_return'（全收益）"
            )
        idx = (
            data.table("index_daily", start=days[0], end=days[-1])
            .filter(pl.col("symbol") == benchmark)
            .select("date", "close")
            .collect()
        )
        if idx.height == 0:
            raise ReportError(f"数据层没有指数 {benchmark} 在区间内的行情")
        m = dict(zip(idx["date"].to_list(), idx["close"].to_list(), strict=True))
        values = np.array([m.get(d, np.nan) for d in days], dtype=float)
        out["benchmark"] = Benchmark(
            values,
            benchmark_name or INDEX_NAMES.get(benchmark, benchmark),
            benchmark_kind,
            benchmark,
        )
    return out
