"""交易所涨跌幅规则；公式只用于审计，不代替源涨跌停价。"""

from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP, Decimal

SOURCES = [
    "https://www.szse.cn/aboutus/trends/news/t20200821_580924.html",
    "https://investor.szse.cn/knowledge/stock/deal/t20180801_553961.html",
    "https://www.szse.cn/lawrules/rule/allrules/bussiness/t20101103_565178.html",
    "https://www.bse.cn/jygl_list/200028217.html",
    "https://investor.szse.cn/knowledge/t20170503_538571.html",
    "https://www.sse.com.cn/aboutus/mediacenter/hotandd/c/c_20260424_10816474.shtml",
    "https://www.szse.cn/lawrules/rule/trade/current/t20260424_620190.html",
    "https://www.bse.cn/uploads/6/file/public/202109/20210909101516_y5wn2y3ft9.pdf",
    "https://www.sse.com.cn/lawandrules/sselawsrules2025/fund/trading/c/c_20260424_10817739.shtml",
]


def limit_rates(symbol, day, listing_date, age, is_st, non_reformed=False):
    if symbol.endswith(".BJ"):
        return None if age == 1 else (0.30, 0.30)
    if symbol.startswith(("688", "689")):
        return None if age <= 5 else (0.20, 0.20)
    if symbol.startswith(("30",)) and day >= "20200824":
        return None if listing_date >= "20200824" and age <= 5 else (0.20, 0.20)
    if listing_date >= "20230410" and age <= 5:
        return None
    if age == 1:
        # 核准制新股首日有效申报上限，不等同于常规涨跌停；单列审计。
        return (0.44, 0.36) if listing_date >= "20140101" else None
    rate = (
        0.05
        if (non_reformed and day >= "20070108") or (is_st and day < "20260706")
        else 0.10
    )
    return rate, rate


def limit_price(previous, rate, *, upper=True, inward=False):
    value = Decimal(str(previous)) * (
        Decimal(1) + Decimal(str(rate)) * (1 if upper else -1)
    )
    rounding = (ROUND_FLOOR if upper else ROUND_CEILING) if inward else ROUND_HALF_UP
    rounded = value.quantize(Decimal(".01"), rounding=rounding)
    previous = Decimal(str(previous))
    if abs(rounded - previous) < Decimal(".01"):
        rounded = previous + (Decimal(".01") if upper else -Decimal(".01"))
    return float(rounded)


def limit_expressions():
    """与标量参考实现独立测试的 Polars 表达式，用于年度批量校验。"""
    import polars as pl

    symbol, day, listing, age = (
        pl.col("ts_code"),
        pl.col("trade_date"),
        pl.col("list_date"),
        pl.col("_age"),
    )
    bse = symbol.str.ends_with(".BJ")
    star = symbol.str.starts_with("688") | symbol.str.starts_with("689")
    growth = symbol.str.starts_with("30") & (day >= "20200824")
    unlimited = (
        (bse & (age == 1))
        | (star & (age <= 5))
        | (growth & (listing >= "20200824") & (age <= 5))
        | ((listing >= "20230410") & (age <= 5))
        | ((age == 1) & (listing < "20140101"))
    )
    rate = (
        pl.when(bse)
        .then(0.30)
        .when(star | growth)
        .then(0.20)
        .when(
            (pl.col("_non_reformed") & (day >= "20070108"))
            | (pl.col("_st") & (day < "20260706"))
        )
        .then(0.05)
        .otherwise(0.10)
    )
    up = pl.when((age == 1) & ~bse & ~star & ~growth).then(0.44).otherwise(rate)
    down = pl.when((age == 1) & ~bse & ~star & ~growth).then(0.36).otherwise(rate)
    upper = pl.col("pre_close") * (1 + up) * 100
    lower = pl.col("pre_close") * (1 - down) * 100
    upper = (
        pl.when(bse)
        .then((upper + 1e-8).floor())
        .otherwise((upper + 0.5 + 1e-8).floor())
        / 100
    )
    lower = (
        pl.when(bse).then((lower - 1e-8).ceil()).otherwise((lower + 0.5 + 1e-8).floor())
        / 100
    )
    upper = pl.max_horizontal(upper, pl.col("pre_close") + 0.01)
    lower = pl.min_horizontal(lower, pl.col("pre_close") - 0.01)
    return (
        unlimited.alias("_unlimited"),
        upper.alias("expected_upper"),
        lower.alias("expected_lower"),
    )
