"""Tushare 期货原始库 → 统一字段（见 futures_schema）。Tushare 的代码、字段名和单位
只在这里出现；单位换算规则都由真实数据核对过，证据见 docs/FUTURES_DATA.md。
"""

import re

import polars as pl

from .futures_schema import conform

# Tushare 合约代码后缀 → 交易所
SUFFIX = {
    "CFX": "CFFEX",
    "SHF": "SHFE",
    "DCE": "DCE",
    "ZCE": "CZCE",
    "INE": "INE",
    "GFE": "GFEX",
}
EXCHANGE_SUFFIX = {v: k for k, v in SUFFIX.items()}

# 成交量、持仓量、成交额的统计口径：在这些交易日之前源数据是双边计数（每笔成交买卖各记一次），
# 统一字段除以 2 换成单边。依据：双边计数时成交量和持仓量恒为偶数，切换后奇数约占一半
# （逐交易所逐年统计见 docs/FUTURES_DATA.md「单边与双边」）。中金所和广期所始终单边。
DOUBLE_SIDED_UNTIL = {
    "SHFE": "20200101",
    "INE": "20200101",
    "DCE": "20200101",
    "CZCE": "20200101",
}

# 大商所、广期所的 trading_fee 一列同时用于「元/手」和「万分之几」，源里没有标记。
# 下列品种按成交金额收费（交易所现行标准；历史上是否换过收费方式未核实），其余按手。
# 不在两个集合里的品种视为未知，费用字段留空。
RATE_BASED = {
    "DCE": {"I", "J", "JM", "JD", "BB", "FB", "LH"},
    "GFEX": {"SI", "LC", "PS"},
}
PER_CONTRACT = {
    "DCE": {
        "A",
        "B",
        "C",
        "CS",
        "EB",
        "EG",
        "L",
        "M",
        "P",
        "PG",
        "PP",
        "RR",
        "V",
        "Y",
        "L_F",
        "PP_F",
        "V_F",
    },
    "GFEX": set(),
}
# 中金所：股指期货按金额（万分之几），国债期货按手（元/手）；trading_fee_rate 与 trading_fee 两列填同一个数。
CFFEX_INDEX = {"IF", "IH", "IC", "IM"}


def product_of(code):
    match = re.match(r"^([A-Z]+)", code)
    return match.group(1) if match else None


def unify_symbol(ts_code, product=None):
    """``RB2701.SHF`` → ``RB2701.SHFE``；``L2605F.DCE`` → ``L_F2605.DCE``；连续代码返回 None。"""
    base, suffix = ts_code.split(".")
    exchange = SUFFIX[suffix]
    match = re.match(r"^([A-Z]+)(\d{3,4})(F?)$", base)
    if not match:
        return None
    letters, digits, average = match.groups()
    if len(digits) == 3:
        raise ValueError(f"三位年月的合约代码须由合约表展开：{ts_code}")
    return f"{product or (letters + ('_F' if average else ''))}{digits}.{exchange}"


def _symbol_expr(column="ts_code"):
    code = pl.col(column)
    base = code.str.extract(r"^([A-Z]+\d{4}F?)\.")
    letters = base.str.extract(r"^([A-Z]+)")
    digits = base.str.extract(r"(\d{4})")
    average = base.str.ends_with("F")
    exchange = code.str.extract(r"\.(\w+)$").replace_strict(SUFFIX, default=None)
    return (
        pl.when(base.is_not_null())
        .then(
            pl.concat_str(
                [
                    letters,
                    pl.when(average).then(pl.lit("_F")).otherwise(pl.lit("")),
                    digits,
                    pl.lit("."),
                    exchange,
                ]
            )
        )
        .otherwise(None)
    )


def contracts(store):
    raw = store.read("futures_contracts")
    tick = raw["quote_unit_desc"].str.extract(r"^([0-9.]+)").cast(pl.Float64)
    frame = raw.with_columns(
        symbol=_symbol_expr(),
        product=pl.col("fut_code").str.to_uppercase(),
        declared=pl.coalesce("multiplier", "per_unit").cast(pl.Float64)
        * pl.when(
            pl.col("quote_unit").str.contains("500千克")
            & (pl.col("trade_unit") == "吨")
        )
        .then(2.0)
        .otherwise(1.0),
        tick_size=tick,
        quote_unit=pl.col("quote_unit").str.replace("人民币", ""),
        settlement_type=pl.col("d_mode_desc").replace_strict(
            {"实物交割": "physical", "现金交割": "cash"}, default=None
        ),
        currency=pl.lit("CNY"),
        listing_date=pl.col("list_date"),
        last_trade_date=pl.col("delist_date"),
        delivery_date=pl.col("last_ddate"),
        delivery_month=pl.col("d_month"),
    ).filter(pl.col("symbol").is_not_null())
    implied = implied_multipliers(store, raw)
    frame = frame.join(implied, on="ts_code", how="left").with_columns(
        off=(pl.col("implied") / pl.col("declared") - 1).abs() > SPEC_TOLERANCE
    )
    frame = frame.with_columns(
        multiplier=pl.when(pl.col("off"))
        .then(pl.col("implied").round(0))
        .otherwise(pl.col("declared")),
        multiplier_source=pl.when(pl.col("off"))
        .then(pl.lit("implied_from_amount"))
        .otherwise(pl.lit("contract_table")),
    )
    return (
        conform(
            "contracts",
            frame.select(
                "symbol",
                "exchange",
                "product",
                "name",
                "listing_date",
                "last_trade_date",
                "delivery_date",
                "delivery_month",
                "multiplier",
                "quote_unit",
                "tick_size",
                "settlement_type",
                "currency",
                "multiplier_source",
            ),
        )
        .unique("symbol", keep="first", maintain_order=True)
        .sort("symbol")
    )


# 合约表的交易单位是现值；历史上改过交易单位的品种（RU 5→10 吨、PB 25→5 吨、FU 50→10 吨、
# FB 500 张→10 立方米）旧合约要按成交额反查：每个合约取「成交额 ÷ (成交量 × 结算价)」的中位数，
# 与合约表相差超过 2% 就用反查值（取整），并在 multiplier_source 标明。
SPEC_TOLERANCE = 0.02
MIN_ROWS = 3


def implied_multipliers(store, raw_contracts=None):
    raw = (
        raw_contracts if raw_contracts is not None else store.read("futures_contracts")
    )
    listing = raw.select("ts_code", "list_date")
    rows = (
        store.scan("futures_daily")
        .select("ts_code", "trade_date", "vol", "amount", "settle")
        .join(listing.lazy(), on="ts_code", how="inner")
        .filter(
            (pl.col("trade_date") >= pl.col("list_date"))
            & (pl.col("vol") > 0)
            & (pl.col("amount") > 0)
            & (pl.col("settle") > 0)
        )
        .with_columns(
            ratio=pl.col("amount") * 10000 / (pl.col("vol") * pl.col("settle"))
        )
        .group_by("ts_code")
        .agg(implied=pl.col("ratio").median(), rows=pl.len())
        .filter(pl.col("rows") >= MIN_ROWS)
        .select("ts_code", "implied")
        .collect()
    )
    return rows


def _sided(frame):
    exchange = pl.col("exchange")
    double = pl.lit(False)
    for name, until in DOUBLE_SIDED_UNTIL.items():
        double = double | ((exchange == name) & (pl.col("date") < until))
    return frame.with_columns(_halve=pl.when(double).then(0.5).otherwise(1.0))


def daily(store, start=None, end=None):
    raw = store.read("futures_daily", start, end)
    listing = store.read("futures_contracts").select("ts_code", "list_date")
    # 上期所 2018 年燃料油改合约时沿用了旧代码：新合约挂牌日之前的行属于同名旧合约（交易单位不同），丢弃。
    raw = raw.join(listing, on="ts_code", how="left").filter(
        pl.col("list_date").is_null() | (pl.col("trade_date") >= pl.col("list_date"))
    )
    frame = (
        raw.with_columns(symbol=_symbol_expr(), date=pl.col("trade_date"))
        .filter(pl.col("symbol").is_not_null())
        .with_columns(exchange=pl.col("symbol").str.extract(r"\.(\w+)$"))
    )
    frame = _sided(frame).with_columns(
        volume=pl.col("vol") * pl.col("_halve"),
        amount=pl.col("amount") * 10000 * pl.col("_halve"),
        open_interest=pl.col("oi") * pl.col("_halve"),
        settlement=pl.col("settle"),
        previous_settlement=pl.col("pre_settle"),
    )
    return conform(
        "daily",
        frame.select(
            "date",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "settlement",
            "previous_settlement",
            "volume",
            "amount",
            "open_interest",
        ),
    ).sort("symbol", "date")


def tushare_main_daily(store, start=None, end=None):
    """Tushare 自己的主力连续日线（``RB.SHF`` 这类代码），只用于对账。"""
    raw = store.read("futures_daily", start, end)
    return raw.filter(pl.col("ts_code").str.contains(r"^[A-Z]+\.\w+$")).select(
        continuous=pl.col("ts_code"),
        date=pl.col("trade_date"),
        open="open",
        high="high",
        low="low",
        close="close",
        settlement="settle",
        volume="vol",
        open_interest="oi",
    )


def params(store, start=None, end=None):
    raw = store.read("futures_settle", start, end)
    frame = raw.with_columns(
        symbol=_symbol_expr(),
        date=pl.col("trade_date"),
        exchange=pl.col("ts_code")
        .str.extract(r"\.(\w+)$")
        .replace_strict(SUFFIX, default=None),
        product=pl.col("ts_code").str.extract(r"^([A-Z]+)"),
    ).filter(pl.col("symbol").is_not_null())
    ex, product = pl.col("exchange"), pl.col("product")
    rate_col, unit_col = pl.col("trading_fee_rate"), pl.col("trading_fee")
    today = pl.col("offset_today_fee")
    margin_scale = pl.when(ex == "CFFEX").then(0.01).otherwise(1.0)
    is_index = (ex == "CFFEX") & product.is_in(list(CFFEX_INDEX))
    is_bond = (ex == "CFFEX") & ~product.is_in(list(CFFEX_INDEX))
    shfe = ex.is_in(["SHFE", "INE"])
    rate_based = pl.lit(False)
    per_contract = ex == "CZCE"
    for exchange, names in RATE_BASED.items():
        rate_based = rate_based | ((ex == exchange) & product.is_in(list(names)))
    for exchange, names in PER_CONTRACT.items():
        per_contract = per_contract | ((ex == exchange) & product.is_in(list(names)))
    # 开平仓（平昨）同一费率。
    open_rate = (
        pl.when(shfe)
        .then(rate_col * 1e-3)
        .when(is_index)
        .then(rate_col * 1e-4)
        .when(is_bond)
        .then(0.0)
        .when(rate_based)
        .then(unit_col * 1e-4)
        .when(per_contract)
        .then(0.0)
        .otherwise(None)
    )
    open_unit = (
        pl.when(shfe)
        .then(unit_col)
        .when(is_index)
        .then(0.0)
        .when(is_bond)
        .then(unit_col)
        .when(rate_based)
        .then(0.0)
        .when(per_contract)
        .then(unit_col)
        .otherwise(None)
    )
    # 平今：中金所 offset_today_fee 是相对普通费的百分比（1000 = 10 倍，0 = 免收）；
    # 大商所/郑商所/广期所是与 trading_fee 同口径的数值；上期所/能源中心源里为空。
    cffex_multiple = today / 100
    close_today_rate = (
        pl.when(ex == "CFFEX")
        .then(open_rate * cffex_multiple)
        .when(rate_based)
        .then(today * 1e-4)
        .when(per_contract)
        .then(0.0)
        .otherwise(None)
    )
    close_today_unit = (
        pl.when(ex == "CFFEX")
        .then(open_unit * cffex_multiple)
        .when(rate_based)
        .then(0.0)
        .when(per_contract)
        .then(today)
        .otherwise(None)
    )
    close_today_known = today.is_not_null()
    delivery = pl.col("delivery_fee")
    frame = frame.with_columns(
        margin_long=pl.col("long_margin_rate") * margin_scale,
        margin_short=pl.col("short_margin_rate") * margin_scale,
        hedge_margin_long=pl.col("b_hedging_margin_rate") * margin_scale,
        hedge_margin_short=pl.col("s_hedging_margin_rate") * margin_scale,
        fee_open_rate=open_rate,
        fee_open_per_contract=open_unit,
        fee_close_rate=open_rate,
        fee_close_per_contract=open_unit,
        fee_close_today_rate=pl.when(close_today_known).then(close_today_rate),
        fee_close_today_per_contract=pl.when(close_today_known).then(close_today_unit),
        # 交割费：上期所/能源中心与 trading_fee_rate 同为千分比，按手收取的品种源里同样只给比例列，
        # 中金所、大商所、郑商所、广期所的交割费单位未核实，留空。
        delivery_fee_rate=pl.when(shfe).then(delivery * 1e-3),
        delivery_fee_per_contract=pl.lit(None, pl.Float64),
        price_limit=pl.lit(None, pl.Float64),
    )
    return conform(
        "params",
        frame.select(
            "date",
            "symbol",
            "margin_long",
            "margin_short",
            "hedge_margin_long",
            "hedge_margin_short",
            "fee_open_rate",
            "fee_open_per_contract",
            "fee_close_rate",
            "fee_close_per_contract",
            "fee_close_today_rate",
            "fee_close_today_per_contract",
            "delivery_fee_rate",
            "delivery_fee_per_contract",
            "price_limit",
        ),
    ).sort("symbol", "date")


def calendar(store):
    raw = store.read("futures_calendar")
    return conform(
        "calendar",
        raw.select(
            "exchange",
            date=pl.col("cal_date"),
            is_open=pl.col("is_open").cast(pl.Int64) == 1,
        ),
    ).sort("exchange", "date")


def mapping(store, start=None, end=None):
    """``RB.SHF`` 为主力、``RBL.SHF`` 为连续（近月）。

    不能只看代码末尾的 L（``AL.SHF``、``TL.CFX`` 本身就是品种）：被映射合约的品种等于连续代码
    去掉后缀的部分时为主力，等于再去掉末尾 L 时为连续。
    """
    raw = store.read("futures_mapping", start, end)
    base = pl.col("ts_code").str.extract(r"^([A-Z_]+)\.")
    target = pl.col("mapping_ts_code").str.extract(r"^([A-Z]+)")
    frame = raw.with_columns(
        date=pl.col("trade_date"),
        continuous=pl.col("ts_code"),
        kind=pl.when(target == base)
        .then(pl.lit("main"))
        .when(
            base.str.ends_with("L")
            & (target == base.str.slice(0, base.str.len_chars() - 1))
        )
        .then(pl.lit("nearby"))
        .otherwise(pl.lit("other")),
        symbol=_symbol_expr("mapping_ts_code"),
    ).filter(pl.col("symbol").is_not_null())
    return conform(
        "mapping", frame.select("date", "continuous", "kind", "symbol")
    ).sort("continuous", "date")


class TushareFutures:
    """把原始库包成统一字段的适配器；任何来源的适配器都实现同样五个方法。"""

    name = "Tushare Pro"

    def __init__(self, store):
        self.store = store

    def contracts(self):
        return contracts(self.store)

    def daily(self, start=None, end=None):
        return daily(self.store, start, end)

    def params(self, start=None, end=None):
        return params(self.store, start, end)

    def calendar(self):
        return calendar(self.store)

    def mapping(self, start=None, end=None):
        return mapping(self.store, start, end)

    def source_continuous(self):
        """来源自带的主力连续日线，只用于对账，不进统一字段。"""
        return tushare_main_daily(self.store)


__all__ = ["TushareFutures", "unify_symbol"]
