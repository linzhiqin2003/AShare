"""与来源无关的期货统一字段。任何来源（Tushare、以后的 Databento、Bloomberg）的适配器
都交出下面几张表（multipliers、mapping 可选），字段名、单位和时点以此为准；面板构建与校验只认这些表。

合约标识统一写成「品种代码 + 四位年月 + . + 交易所」，例如 ``SC2612.INE``、
``RB2701.SHFE``、``TA2611.CZCE``、``IF2612.CFFEX``；境外如 ``CL2612.NYMEX``、
``B2612.ICE``。品种代码大写，年月一律四位（郑商所源码的三位年月由适配器展开）。
"""

from dataclasses import dataclass

import polars as pl


@dataclass(frozen=True)
class Column:
    name: str
    dtype: object
    unit: str
    known_at: str
    required: bool = True


def _cols(*items):
    return {c.name: c for c in (Column(*item) for item in items)}


S, F, B = pl.String, pl.Float64, pl.Boolean
DATE = "YYYYMMDD 字符串，交易所交易日口径"

CONTRACTS = _cols(
    ("symbol", S, "统一合约标识", "挂牌前公布"),
    ("exchange", S, "CFFEX/SHFE/DCE/CZCE/INE/GFEX，境外用交易所缩写", "静态"),
    ("product", S, "品种代码，大写", "静态"),
    ("name", S, "名称", "静态", False),
    ("listing_date", S, DATE + "；首个交易日", "挂牌前公布"),
    ("last_trade_date", S, DATE + "；最后交易日", "挂牌前公布；可能因节假日调整"),
    ("delivery_date", S, DATE + "；最后交割日，源缺失时为空", "挂牌前公布", False),
    ("delivery_month", S, "YYYYMM", "静态", False),
    (
        "multiplier",
        F,
        "每手对应的报价单位数量，取最后交易日的值；存续期内改过交易单位的合约按日取值见 multipliers 表与面板 multiplier 字段",
        "挂牌前公布",
    ),
    ("quote_unit", S, "报价单位，如 元/吨、指数点", "静态", False),
    ("tick_size", F, "最小变动价位，报价单位", "挂牌前公布", False),
    ("settlement_type", S, "physical 或 cash", "静态", False),
    ("currency", S, "报价币种，如 CNY", "静态"),
    (
        "multiplier_source",
        S,
        "最后一段乘数的来源，取值同 multipliers.source",
        "静态",
        False,
    ),
)

DAILY = _cols(
    ("date", S, DATE, "交易日日盘收盘后"),
    ("symbol", S, "统一合约标识", "—"),
    (
        "open",
        F,
        "报价单位；境内为夜盘集合竞价开盘价（有夜盘的品种）",
        "交易日开盘（境内为前一晚夜盘）",
    ),
    ("high", F, "报价单位；含前一晚夜盘", "收盘后"),
    ("low", F, "报价单位；含前一晚夜盘", "收盘后"),
    ("close", F, "报价单位", "收盘后"),
    (
        "settlement",
        F,
        "报价单位；逐日盯市所用结算价（境内多为全天成交量加权均价）",
        "结算后",
    ),
    (
        "previous_settlement",
        F,
        "报价单位；上一交易日结算价（新合约首日为挂牌基准价）",
        "盘前",
    ),
    ("volume", F, "手，单边", "收盘后"),
    ("amount", F, "元（报价币种），单边", "收盘后", False),
    ("open_interest", F, "手，单边", "收盘后", False),
)

PARAMS = _cols(
    (
        "date",
        S,
        DATE + "；该交易日结算所用参数",
        "结算后（新标准常在前一交易日结算时已生效）",
    ),
    ("symbol", S, "统一合约标识", "—"),
    ("margin_long", F, "比例；买投机交易保证金率（交易所标准）", "结算后"),
    ("margin_short", F, "比例；卖投机交易保证金率", "结算后"),
    ("hedge_margin_long", F, "比例；买套保保证金率", "结算后", False),
    ("hedge_margin_short", F, "比例；卖套保保证金率", "结算后", False),
    ("fee_open_rate", F, "比例；按成交金额的开仓手续费", "结算后", False),
    ("fee_open_per_contract", F, "元/手；开仓手续费", "结算后", False),
    ("fee_close_rate", F, "比例；平昨仓", "结算后", False),
    ("fee_close_per_contract", F, "元/手；平昨仓", "结算后", False),
    (
        "fee_close_today_rate",
        F,
        "比例；平今仓，源缺失为空（不代表免收）",
        "结算后",
        False,
    ),
    ("fee_close_today_per_contract", F, "元/手；平今仓，源缺失为空", "结算后", False),
    ("delivery_fee_rate", F, "比例；交割手续费", "结算后", False),
    ("delivery_fee_per_contract", F, "元/手；交割手续费", "结算后", False),
    (
        "price_limit",
        F,
        "比例；涨跌停幅度（相对上一结算价）。Tushare 无此字段，恒为空；以后取交易所每日交易参数文件",
        "盘前",
        False,
    ),
)

CALENDAR = _cols(
    ("exchange", S, "交易所", "提前公布"),
    ("date", S, DATE, "提前公布"),
    ("is_open", B, "布尔", "提前公布"),
)

MAPPING = _cols(
    ("date", S, DATE, "见各来源说明"),
    ("continuous", S, "来源的连续代码，如 SC.INE", "—"),
    ("kind", S, "main（主力）或 nearby（连续/近月）", "—"),
    ("symbol", S, "被映射到的统一合约标识", "见各来源说明"),
)

MULTIPLIERS = _cols(
    ("symbol", S, "统一合约标识", "—"),
    (
        "start_date",
        S,
        DATE + "；本段乘数生效的第一个交易日（首段为上市日）",
        "生效前公布",
    ),
    ("multiplier", F, "每手对应的报价单位数量", "生效前公布"),
    (
        "source",
        S,
        "implied_from_amount（成交额反查）、contract_table（合约表，有成交额核对过）、"
        "contract_table_unverified（合约表，无成交额可核对）、manual（人工修正表，见 basis）",
        "—",
    ),
    ("basis", S, "依据；manual 必填，「未核实」开头表示依据只是推断", "—", False),
)

TABLES = {
    "contracts": CONTRACTS,
    "daily": DAILY,
    "params": PARAMS,
    "calendar": CALENDAR,
    "mapping": MAPPING,
    "multipliers": MULTIPLIERS,
}
# 适配器必须交出的表；mapping 可选（没有就只能用自己判定的换月规则），
# multipliers 可选（没有时每个合约整个存续期用 contracts.multiplier）。
REQUIRED_TABLES = ("contracts", "daily", "params", "calendar")

# 面板字段：每个都是「交易日 × 合约」矩阵。
PANEL_FIELDS = {
    **{k: c for k, c in DAILY.items() if k not in ("date", "symbol")},
    **{k: c for k, c in PARAMS.items() if k not in ("date", "symbol")},
    "multiplier": Column(
        "multiplier",
        F,
        "每手对应的报价单位数量，按 multipliers 表逐日取值",
        "生效前公布",
    ),
    "has_quote": Column(
        "has_quote",
        B,
        "布尔；当天日线的开盘价、结算价、成交量至少一个非空。只有持仓量的行（源里极少，"
        "都是零成交）为假；有行情不等于有成交，成交看 volume 与 open",
        "收盘后",
    ),
    "is_listed": Column("is_listed", B, "布尔；上市日到最后交易日（含两端）", "盘前"),
}


def conform(name, frame):
    """把适配器交出的表按统一字段检查、排列和转型；缺必填列或有多余列都报错。"""
    columns = TABLES[name]
    names = (
        frame.collect_schema().names()
        if isinstance(frame, pl.LazyFrame)
        else frame.columns
    )
    missing = [c.name for c in columns.values() if c.required and c.name not in names]
    extra = sorted(set(names) - columns.keys())
    if missing or extra:
        raise ValueError(f"{name}: 缺少必填列 {missing}，多余列 {extra}")
    return frame.select(
        [
            (pl.col(c.name) if c.name in names else pl.lit(None))
            .cast(c.dtype)
            .alias(c.name)
            for c in columns.values()
        ]
    )


def describe():
    """返回统一字段表（表名、字段、类型、单位、时点、是否必填），用于文档与检查。"""
    return pl.DataFrame(
        [
            dict(
                table=table,
                field=c.name,
                dtype=str(c.dtype),
                unit=c.unit,
                known_at=c.known_at,
                required=c.required,
            )
            for table, columns in TABLES.items()
            for c in columns.values()
        ]
    )


__all__ = ["PANEL_FIELDS", "REQUIRED_TABLES", "TABLES", "conform", "describe"]
