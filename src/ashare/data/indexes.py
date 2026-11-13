"""指数日线的拼接读取与本地指数名单（研究平台的基准下拉框用）。

原始库里指数日线有两份来源，读取时拼成一张 ``index_daily`` 长表：

- ``index_daily``：股票池用的 5 个指数（``catalog.INDEXES``），逐指数逐月分区，行为与原来相同；
- ``index_daily_all``：全市场指数，按交易日分区；其中那 5 个代码被略去，以 ``index_daily`` 为准，
  拼接后每个（代码，日期）只有一行。
"""

import polars as pl

from .catalog import INDEXES

WIDE = "index_daily_all"
POOL_CODES = tuple(sorted(INDEXES.values()))


def _has(store, name):
    return any(e["status"] == "complete" for e in store.entries(name).values())


def index_daily_frame(store, start=None, end=None):
    """返回原始字段名的 LazyFrame；start、end 为 ``YYYYMMDD``，只用来裁剪按日分区。"""
    frames = []
    missing = None
    try:
        frames.append(store.scan("index_daily"))
    except FileNotFoundError as exc:
        missing = exc
    if _has(store, WIDE):
        frames.append(
            store.scan(WIDE, start, end).filter(~pl.col("ts_code").is_in(POOL_CODES))
        )
    if not frames:
        raise missing
    return frames[0] if len(frames) == 1 else pl.concat(frames, how="diagonal_relaxed")


def local_indexes(store, query=None):
    """本地有日线的指数：代码、简称、市场、类别、本地首末日期，按代码排序。

    has_open 表示本地日线里出现过开盘价：多数中证、国证指数源只给收盘价，
    这类指数作基准时按收盘价算收益，用开盘价的地方会得到缺失值。

    query 不区分大小写，匹配代码、简称或全称的任一部分；名单缺失（未同步 index_basic）时
    简称等列为空，日线照常列出。
    """
    coverage = (
        index_daily_frame(store)
        .group_by("ts_code")
        .agg(
            pl.col("trade_date").min().alias("first_date"),
            pl.col("trade_date").max().alias("last_date"),
            pl.col("open").is_not_null().any().alias("has_open"),
        )
        .collect()
    )
    columns = ("ts_code", "name", "fullname", "market", "category")
    if _has(store, "index_basic"):
        basic = (
            store.scan("index_basic")
            .select(columns)
            .unique("ts_code", keep="first")
            .collect()
        )
    else:
        basic = pl.DataFrame(schema={c: pl.String for c in columns})
    table = coverage.join(basic, on="ts_code", how="left")
    if query:
        needle = str(query).strip().lower()
        hit = pl.lit(False)
        for column in ("ts_code", "name", "fullname"):
            hit = hit | pl.col(column).str.to_lowercase().str.contains(
                needle, literal=True
            ).fill_null(False)
        table = table.filter(hit)
    return table.select(
        pl.col("ts_code").alias("symbol"),
        "name",
        pl.col("fullname").alias("full_name"),
        "market",
        "category",
        "first_date",
        "last_date",
        "has_open",
    ).sort("symbol")


def audit_index_daily(store, audit):
    """verify 用：全市场指数日线与股票池逐月分区的重叠部分逐值相同，名单覆盖与基本取值。

    各项只在相应数据集已采集时加入报告；报告只记计数和代码、日期样例，不记行情数值。
    """
    if not _has(store, WIDE):
        return
    values = ("open", "high", "low", "close", "pre_close", "vol", "amount")
    wide = store.scan(WIDE)
    if _has(store, "index_daily"):
        pool = store.scan("index_daily").select("ts_code", "trade_date", *values)
        both = (
            wide.filter(pl.col("ts_code").is_in(POOL_CODES))
            .select("ts_code", "trade_date", *values)
            .join(pool, on=["ts_code", "trade_date"], suffix="_pool")
        )
        differs = pl.lit(False)
        for name in values:
            a, b = (
                pl.col(name).cast(pl.Float64),
                pl.col(f"{name}_pool").cast(pl.Float64),
            )
            differs = differs | (
                (a.is_null() != b.is_null())
                | ((a - b).abs() > 1e-9 * pl.max_horizontal(a.abs(), b.abs(), 1.0))
            ).fill_null(False)
        checked = both.select(pl.len()).collect().item()
        bad = both.filter(differs).select("ts_code", "trade_date").collect()
        audit.add(
            "index_daily_pool_overlap",
            checked,
            bad.head(20).to_dicts(),
            failed=len(bad),
            note="全市场按日分区里的股票池 5 个指数与逐月分区逐值相同（读取以逐月分区为准）",
        )
    shape = wide.select(
        pl.len().alias("rows"),
        (
            (pl.col("close") <= 0)
            | (pl.col("high") < pl.col("low"))
            | pl.col("close").is_null()
        )
        .fill_null(False)
        .sum()
        .alias("bad"),
    ).collect()
    examples = (
        wide.filter(
            (
                (pl.col("close") <= 0)
                | (pl.col("high") < pl.col("low"))
                | pl.col("close").is_null()
            ).fill_null(False)
        )
        .select("ts_code", "trade_date")
        .head(20)
        .collect()
        .to_dicts()
    )
    audit.add(
        "index_daily_all_values",
        shape["rows"][0],
        examples,
        failed=shape["bad"][0],
        note="全市场指数日线：收盘价存在且为正、最高不低于最低",
    )
    if _has(store, "index_basic"):
        codes = wide.select(pl.col("ts_code").unique()).collect()
        known = store.scan("index_basic").select("ts_code").collect()
        orphan = codes.join(known, on="ts_code", how="anti").sort("ts_code")
        audit.add(
            "index_basic_covers_daily",
            len(codes),
            orphan.head(20).to_dicts(),
            failed=len(orphan),
            note="有日线的指数都在 index_basic 名单里（缺名单时下拉框只显示代码）",
        )
