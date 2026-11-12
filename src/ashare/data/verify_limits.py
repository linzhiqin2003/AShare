"""涨跌幅规则按年度批量连接；保留规则切换窗口和异常样例。"""

import polars as pl

from .rules import limit_expressions


def audit_limits(tables, dates, audit):
    calendar = (
        tables["trade_cal"]
        .filter(pl.col("is_open") == 1)
        .select("cal_date")
        .unique()
        .sort("cal_date")
        .with_row_index("_number")
    )
    sec = tables["stock_basic"].unique(subset="ts_code", keep="last").sort("list_date")
    sec = (
        sec.join_asof(
            calendar, left_on="list_date", right_on="cal_date", strategy="forward"
        )
        .rename({"_number": "_listing_number"})
        .drop("cal_date")
    )
    frame = (
        tables["stk_limit"]
        .filter(pl.col("trade_date").is_in(list(dates)))
        .join(sec, on="ts_code", how="inner")
        .join(calendar, left_on="trade_date", right_on="cal_date", how="left")
        .with_columns((pl.col("_number") - pl.col("_listing_number") + 1).alias("_age"))
    )
    # 日历可能只覆盖样本；早于它上市的证券只需确定不是上市初期。
    frame = frame.with_columns(
        pl.when(pl.col("list_date") < calendar["cal_date"].min())
        .then(9999)
        .otherwise(pl.col("_age"))
        .alias("_age")
    )
    st = (
        tables["stock_st"]
        .select("ts_code", "trade_date")
        .unique()
        .with_columns(pl.lit(True).alias("_st"))
    )
    frame = frame.join(st, on=["ts_code", "trade_date"], how="left").with_columns(
        pl.col("_st").fill_null(False), pl.lit(False).alias("_non_reformed")
    )
    names = tables.get("namechange", pl.DataFrame())
    if len(names):
        names = names.filter(
            pl.col("name").str.starts_with("S") & ~pl.col("name").str.starts_with("ST")
        ).select("ts_code", "start_date", "end_date")
        special = (
            frame.select("ts_code", "trade_date")
            .join(names, on="ts_code", how="inner")
            .filter(
                (pl.col("start_date") <= pl.col("trade_date"))
                & (
                    pl.col("end_date").is_null()
                    | (pl.col("trade_date") <= pl.col("end_date"))
                )
            )
            .select("ts_code", "trade_date")
            .unique()
            .with_columns(pl.lit(True).alias("_special"))
        )
        frame = frame.join(
            special, on=["ts_code", "trade_date"], how="left"
        ).with_columns(pl.col("_special").fill_null(False).alias("_non_reformed"))
    frame = frame.with_columns(*limit_expressions())
    unlimited = frame.filter(pl.col("_unlimited"))
    checked = frame.filter(~pl.col("_unlimited"))
    checked = checked.with_columns(
        (
            ((pl.col("expected_upper") - pl.col("up_limit")).abs() > 0.001)
            | ((pl.col("expected_lower") - pl.col("down_limit")).abs() > 0.001)
        ).alias("_bad")
    )
    audit.frame(
        "price_limits",
        checked,
        pl.col("_bad"),
        columns=[
            "ts_code",
            "trade_date",
            "expected_upper",
            "up_limit",
            "expected_lower",
            "down_limit",
            "_st",
        ],
        note="沪深四舍五入；北交所向内取分；特殊恢复上市/退市整理的未编码例外仍列为公式不一致，不直接认定源数据错误",
    )
    reasons = (
        checked.filter(pl.col("_bad"))
        .with_columns(
            pl.when(pl.col("pre_close") <= 0)
            .then(pl.lit("源昨收非正数"))
            .when(pl.col("up_limit") >= 10000)
            .then(pl.lit("源无限制哨兵/特殊恢复交易"))
            .when(pl.col("_age") <= 0)
            .then(pl.lit("源上市前历史/代码重述"))
            .otherwise(pl.lit("公式或状态不一致"))
            .alias("category")
        )
        .group_by("category")
        .len()
    )
    audit.checks["price_limits"]["failure_categories"] = reasons.to_dicts()
    valid_reference = checked.filter(pl.col("pre_close") > 0)
    audit.checks["price_limits"]["positive_reference_checked"] = len(valid_reference)
    audit.checks["price_limits"]["positive_reference_failed"] = valid_reference[
        "_bad"
    ].sum()
    groups = (
        checked.group_by(
            pl.col("trade_date").str.slice(0, 4).alias("year"), "market", "_st"
        )
        .agg(pl.len().alias("checked"), pl.col("_bad").sum().alias("failed"))
        .sort("year", "market", "_st")
    )
    audit.checks["price_limits"]["by_year_board_st"] = groups.to_dicts()
    audit.add(
        "unlimited_sessions",
        len(unlimited),
        [],
        note="法定新股无涨跌幅日，未套用常规公式",
    )
    audit.checks["unlimited_sessions"]["pass_rate"] = None
    audit.checks["unlimited_sessions"]["examples"] = (
        unlimited.select("ts_code", "trade_date", "up_limit", "down_limit")
        .head(20)
        .to_dicts()
    )
