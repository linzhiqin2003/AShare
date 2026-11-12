"""从原始库提取内置事件：业绩预告、业绩快报、分红、指数成分调整。

每个函数接收 ``ashare.data.open()`` 的句柄，返回事件表（见 :mod:`ashare.events.table`）。
定义、字段口径和已知的坑写在各函数的文档串和 ``docs/EVENTS_NOTES.md``。

**数据入口**：业绩快报、分红走公开的 ``Data.table``。业绩预告的类型、变动幅度、上年同期、
首次公告日，利润表的归母净利润，指数成分权重长表还没有公开，经 :func:`_raw` 读数据层内部的
``Data._store``（已做换码规范化的原始库），源字段名只出现在本模块里；清单见
``docs/EVENTS_NOTES.md``，数据层公开后改成 ``Data.table``。

最小例子::

    import ashare.data as ad
    from ashare.events import extract

    ds = ad.open()
    ev = extract.forecasts(ds)
    ev.filter(ev["forecast_type"] == "预增").head()
"""

from __future__ import annotations

import weakref

import polars as pl

WAN = 10_000.0  # 业绩预告的净利润单位是万元
REVIEW_SWITCH = (
    "20131101"  # 定期调样从 1 月、7 月改到 6 月、12 月（快照实测 2013-12 起）
)
INDEX_CODES = {"csi300": "000300.SH", "csi500": "000905.SH", "csi1000": "000852.SH"}


def _raw(data, name: str) -> pl.LazyFrame:
    """数据层内部原始表（换码已规范化）。只在本模块使用。"""
    return data._store.scan(name)


def _as_int(col: str) -> pl.Expr:
    return pl.col(col).cast(pl.Int64, strict=False)


def _prev_year(col: str) -> pl.Expr:
    return (pl.col(col).str.slice(0, 4).cast(pl.Int64) - 1).cast(pl.String) + pl.col(
        col
    ).str.slice(4, 4)


_INCOME_CACHE: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _income_first(data) -> pl.DataFrame:
    """利润表各版本的公告日与归母净利润（合并报表）。

    原始利润表分区多、整表扫描在负载高时要十几秒，同一个数据句柄只读一次。
    """
    if data not in _INCOME_CACHE:
        _INCOME_CACHE[data] = _income_scan(data)
    return _INCOME_CACHE[data]


def _income_scan(data) -> pl.DataFrame:
    return (
        _raw(data, "income")
        .filter(pl.col("report_type") == "1")
        .select(
            pl.col("ts_code").alias("symbol"),
            pl.col("end_date").alias("period"),
            pl.coalesce("f_ann_date", "ann_date").alias("known"),
            pl.col("update_flag"),
            pl.col("n_income_attr_p").alias("net_income_parent"),
        )
        .drop_nulls(["symbol", "period", "known"])
        .collect()
    )


def forecasts(data) -> pl.DataFrame:
    """业绩预告。

    - ``kind="forecast"``，``date`` = 预告公告日，``timing="after_close"``。
    - ``forecast_type``：预增/略增/扭亏/续盈/预减/略减/首亏/续亏/增亏/减亏/不确定/其他。
      交易所规则下「预增」指同比增长 50% 以上。
    - ``change_low``/``change_high``：归母净利润同比变动幅度下限/上限（%）；只给一端时另一端
      为空，``change_mid`` 是有的那一端或两端平均。
    - ``net_income_low``/``net_income_high``/``prior_net_income``：预计净利润区间与上年同期
      （元）。
    - 首次预告与修正（一套规则）：``first_date`` = min(源首次公告日, 本表该报告期最早的公告日)；
      ``revision`` = 该报告期内这是第几次（本表里排在它前面的条数，若源记载的首次公告日早于
      本表最早一条——真正的首次预告不在表里，多是随定期报告披露——再加 1）；
      ``is_revision`` ≡ ``revision > 0``。
    - 同一证券、报告期、公告日的重复行（换码前后两个代码各报一次）只留一条，优先更新标识 1。
    """
    raw = (
        _raw(data, "forecast")
        .select(
            pl.col("ts_code").alias("symbol"),
            pl.col("ann_date").alias("date"),
            pl.col("end_date").alias("period_end"),
            pl.col("type").str.strip_chars().alias("forecast_type"),
            pl.col("p_change_min").alias("change_low"),
            pl.col("p_change_max").alias("change_high"),
            (pl.col("net_profit_min") * WAN).alias("net_income_low"),
            (pl.col("net_profit_max") * WAN).alias("net_income_high"),
            (pl.col("last_parent_net") * WAN).alias("prior_net_income"),
            pl.col("first_ann_date").alias("first_date"),
            pl.col("update_flag"),
        )
        .drop_nulls(["symbol", "date", "period_end"])
        .collect()
    )
    raw = raw.sort(
        ["symbol", "period_end", "date", "update_flag"],
        descending=[False, False, False, True],
    )
    raw = raw.unique(
        ["symbol", "period_end", "date"], keep="first", maintain_order=True
    )
    return (
        raw.with_columns(
            pl.lit("forecast").alias("kind"),
            pl.lit("after_close").alias("timing"),
            pl.mean_horizontal("change_low", "change_high").alias("change_mid"),
            pl.min_horizontal(
                pl.col("first_date").min().over(["symbol", "period_end"]),
                pl.col("date").min().over(["symbol", "period_end"]),
            ).alias("first_date"),
            pl.int_range(pl.len()).over(["symbol", "period_end"]).alias("_rank"),
            pl.col("date").min().over(["symbol", "period_end"]).alias("_earliest"),
        )
        .with_columns(
            (
                pl.col("_rank")
                + (pl.col("first_date") < pl.col("_earliest")).cast(pl.Int64)
            ).alias("revision")
        )
        .with_columns((pl.col("revision") > 0).alias("is_revision"))
        .drop("update_flag", "_rank", "_earliest")
    )


def expresses(data) -> pl.DataFrame:
    """业绩快报。

    - ``kind="express"``，``date`` = 快报公告日，``timing="after_close"``。
    - ``net_income``：快报净利润（元；与利润表归母净利润口径一致的约六成，快报是未审计
      的初步数）；``revenue``：营业收入。
    - 超预期程度两种：
      ``yoy`` = 净利润 / 上年同期归母净利润 − 1，上年同期取公告日当天及以前已经公告的利润表
      （上年同期 ≤ 0 时为空）；``vs_forecast`` = (净利润 − 最近一次预告区间中点) / |上年同期|，
      预告只给变动幅度时中点 = 预告里的上年同期 × (1 + 幅度中点)；
      预告取同一报告期、公告日严格早于快报的最近一次（没有预告为空）；``forecast_position``
      是净利润在预告区间里的位置（0 下限、1 上限，可越界）。
    - 更正公告：同一报告期多条快报（实测 20 组）只用最早一条；更正后的数字晚于事件日才可知，
      不回写，「之后还有几条更正」这种未来信息也不输出。
    """
    ex = (
        data.table("express")
        .select(
            "symbol",
            pl.col("announcement_date").alias("date"),
            "period_end",
            "revenue",
            "net_income",
        )
        .drop_nulls(["symbol", "date", "period_end"])
        .collect()
        .sort(["symbol", "period_end", "date"])
        .unique(["symbol", "period_end"], keep="first", maintain_order=True)
    )
    inc = _income_first(data).sort(["symbol", "period", "known", "update_flag"])
    inc = inc.unique(["symbol", "period", "known"], keep="last", maintain_order=True)
    ex = ex.with_columns(
        _prev_year("period_end").alias("prior_period"), _as_int("date").alias("_d")
    )
    prior = inc.select(
        "symbol",
        pl.col("period").alias("prior_period"),
        _as_int("known").alias("_k"),
        pl.col("net_income_parent").alias("prior_net_income"),
    ).sort("_k")
    ex = ex.sort("_d").join_asof(
        prior,
        left_on="_d",
        right_on="_k",
        by=["symbol", "prior_period"],
        strategy="backward",
        check_sortedness=False,
    )
    fc = (
        forecasts(data)
        .select(
            "symbol",
            "period_end",
            _as_int("date").alias("_f"),
            pl.coalesce(
                pl.mean_horizontal("net_income_low", "net_income_high"),
                pl.col("prior_net_income") * (1 + pl.col("change_mid") / 100),
            ).alias("forecast_mid"),
            "net_income_low",
            "net_income_high",
        )
        .sort("_f")
    )
    ex = ex.sort("_d").join_asof(
        fc,
        left_on="_d",
        right_on="_f",
        by=["symbol", "period_end"],
        strategy="backward",
        allow_exact_matches=False,
        check_sortedness=False,
    )
    pos = pl.col("prior_net_income") > 0
    width = pl.col("net_income_high") - pl.col("net_income_low")
    return (
        ex.with_columns(
            pl.lit("express").alias("kind"),
            pl.lit("after_close").alias("timing"),
            pl.when(pos)
            .then(pl.col("net_income") / pl.col("prior_net_income") - 1)
            .alias("yoy"),
            (
                (pl.col("net_income") - pl.col("forecast_mid"))
                / pl.col("prior_net_income").abs()
            ).alias("vs_forecast"),
            pl.when(width > 0)
            .then((pl.col("net_income") - pl.col("net_income_low")) / width)
            .alias("forecast_position"),
            pl.col("_f").is_not_null().alias("has_forecast"),
        )
        .drop("_d", "_k", "_f", "net_income_low", "net_income_high")
        .sort(["date", "symbol"])
    )


DIVIDEND_STAGES = ("report", "resolution", "implementation")


def dividends(data, stage: str = "report") -> pl.DataFrame:
    """分红（现金与送转），三个进度日任选其一作事件日。

    本地分红表几乎只有「实施」行（实测 55,842 行里非实施 74 行），每行带三个日期：
    公告日（源 ``ann_date``）实测落在同一报告期财报公告日之后 21–29 天（中位 24 天，峰值
    21–22 天，与年度股东大会「召开 20 日前通知」的最短间隔吻合；22% 落在周六），是股东大会
    决议日，不是预案公告日；实施公告日中位再晚 25 天；然后是除权除息日。

    ``stage``：

    - ``"report"``（默认，近似预案日）：**只对年报期（1231）**用年报首次公告日作事件日——
      年度利润分配预案由审议年报的同一次董事会提出、随年报披露；要求它不晚于股东大会决议日。
      中报、季报期的分红大多在报告之后另行公告（0930 期 37% 的报告日比决议日早 30 天以上），
      用报告日会前视，所以这些报告期**落到决议日**（``date_source="resolution"``）。
      代理的偏差：预案内容取最终实施的数值；从未实施的预案不在表里（选择偏差）。
    - ``"resolution"``：``date`` = 股东大会决议日。
    - ``"implementation"``：``date`` = 实施公告日。

    数值：``cash_per_share``（税前每股派现，元）、``stock_per_share``（每股送转股）。
    同一证券、报告期、除权日有多条实施行时按进度分别去重：决议日取最晚的一条（实测这类重复
    里较早的一条常是年报日，例如 600519.SH FY2024 的 20250403 与 20250520），实施公告日取
    最早的一条。
    """
    if stage not in DIVIDEND_STAGES:
        raise ValueError(f"stage 只能是 {DIVIDEND_STAGES}")
    keys = ["symbol", "period_end", "ex_date"]
    raw = (
        data.table("dividend")
        .filter(pl.col("status").str.strip_chars() == "实施")
        .select(
            "symbol",
            "period_end",
            pl.col("announcement_date").alias("resolution_date"),
            pl.col("implementation_announcement_date").alias("implementation_date"),
            "record_date",
            "ex_date",
            pl.col("cash_dividend")
            .cast(pl.Float64)
            .fill_null(0.0)
            .alias("cash_per_share"),
            pl.col("stock_dividend")
            .cast(pl.Float64)
            .fill_null(0.0)
            .alias("stock_per_share"),
        )
        .drop_nulls(keys)
        .collect()
        .group_by(keys, maintain_order=True)
        .agg(
            pl.col("resolution_date").max(),
            pl.col("implementation_date").min(),
            pl.col("record_date").first(),
            pl.col("cash_per_share").first(),
            pl.col("stock_per_share").first(),
        )
    )
    if stage == "report":
        reports = (
            _income_first(data)
            .group_by("symbol", "period")
            .agg(pl.col("known").min().alias("report_date"))
            .rename({"period": "period_end"})
        )
        raw = raw.join(reports, on=["symbol", "period_end"], how="left")
        annual = pl.col("period_end").str.ends_with("1231")
        raw = raw.filter(
            ~annual
            | (
                pl.col("report_date").is_not_null()
                & (
                    pl.col("resolution_date").is_null()
                    | (pl.col("report_date") <= pl.col("resolution_date"))
                )
            )
        ).with_columns(
            pl.when(annual)
            .then(pl.col("report_date"))
            .otherwise(pl.col("resolution_date"))
            .alias("date"),
            pl.when(annual)
            .then(pl.lit("report"))
            .otherwise(pl.lit("resolution"))
            .alias("date_source"),
            pl.lit("dividend_plan").alias("kind"),
        )
    elif stage == "resolution":
        raw = raw.with_columns(
            pl.col("resolution_date").alias("date"),
            pl.lit("dividend_resolution").alias("kind"),
        )
    else:
        raw = raw.with_columns(
            pl.col("implementation_date").alias("date"),
            pl.lit("dividend_implementation").alias("kind"),
        )
    return (
        raw.drop_nulls("date")
        .with_columns(pl.lit("after_close").alias("timing"))
        .sort(["date", "symbol"])
    )


def index_changes(data, index: str = "csi300") -> pl.DataFrame:
    """指数成分调整（沪深 300、中证 500、中证 1000）。

    原始库只有成分权重的月末快照，没有调样公告：相邻两期快照比较，新出现的是调入
    （``kind="index_add"``），消失的是调出（``"index_remove"``）。``date`` = 新快照的日期，
    ``timing="after_close"``，最早在快照后下一个交易日成交。

    **只能用生效日之后的快照，推不出公告日**：中证指数的定期调样一般在 5 月底、11 月底
    公告，6 月、12 月第二个星期五的下一交易日生效；月末快照最早在生效后约两到三周、公告后
    约五周才反映。所以这里的事件晚于真实公告，**会低估甚至完全错过公告效应**，只能研究
    「生效之后」的走势。``regular_review`` 标记定期调样所在的快照月：2013 年 11 月以前是
    1 月、7 月（实测沪深 300、中证 500 的调入在这两个月成批出现），之后是 6 月、12 月。
    调出里包括退市、吸收合并（``delisted_by`` 标记快照日前已退市或退市日在两期快照之间）。
    ``weight`` 是调入后的权重或调出前的权重（%）。
    """
    if index not in INDEX_CODES:
        raise ValueError(f"index 只能是 {list(INDEX_CODES)}")
    code = INDEX_CODES[index]
    snaps = (
        _raw(data, "index_weight")
        .filter(pl.col("index_code") == code)
        .select(
            pl.col("con_code").alias("symbol"),
            pl.col("trade_date").alias("snapshot"),
            pl.col("weight").cast(pl.Float64),
        )
        .collect()
        .unique(["symbol", "snapshot"], keep="first")
    )
    days = sorted(snaps["snapshot"].unique().to_list())
    prev = dict(zip(days[1:], days[:-1], strict=True))
    by_day = {d: g for (d,), g in snaps.group_by("snapshot")}
    frames = []
    for day in days[1:]:
        before, after = by_day[prev[day]], by_day[day]
        added = after.join(before.select("symbol"), on="symbol", how="anti")
        removed = before.join(after.select("symbol"), on="symbol", how="anti")
        for part, kind in ((added, "index_add"), (removed, "index_remove")):
            if part.height:
                frames.append(
                    part.select(
                        "symbol",
                        "weight",
                        pl.lit(day).alias("date"),
                        pl.lit(prev[day]).alias("previous_snapshot"),
                        pl.lit(kind).alias("kind"),
                    )
                )
    if not frames:
        return pl.DataFrame(
            schema={
                "symbol": pl.String,
                "date": pl.String,
                "kind": pl.String,
                "timing": pl.String,
            }
        )
    out = pl.concat(frames)
    sec = data.securities().select("symbol", "delisting_date").collect()
    out = out.join(sec, on="symbol", how="left")
    return (
        out.with_columns(
            pl.lit("after_close").alias("timing"),
            pl.lit(index).alias("index"),
            pl.when(pl.col("date") < REVIEW_SWITCH)
            .then(pl.col("date").str.slice(4, 2).is_in(["01", "07"]))
            .otherwise(pl.col("date").str.slice(4, 2).is_in(["06", "12"]))
            .alias("regular_review"),
            (
                (pl.col("kind") == "index_remove")
                & pl.col("delisting_date").is_not_null()
                & (pl.col("delisting_date") <= pl.col("date"))
            ).alias("delisted_by"),
        )
        .drop("delisting_date")
        .sort(["date", "symbol"])
    )
