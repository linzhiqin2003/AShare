"""唯一的数据集声明、源字段及单位换算目录。上层只使用统一字段名。"""

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Dataset:
    name: str
    api: str
    partition: str
    keys: tuple[str, ...]
    columns: str
    limit: int = 5000
    refresh: int = 5
    date: str = "trade_date"
    known_at: str = "收盘后；历史修订无法还原首次发布版本"
    params: dict = field(default_factory=dict)
    source: str = "tushare"
    units: dict = field(default_factory=dict)
    coverage_start: str | None = None
    empty_allowed: bool = True
    drop_check_min_rows: int = 100
    drop_check_ratio: float = 0.65


def dataset(name, partition, keys, columns, **kwargs):
    return Dataset(
        name,
        kwargs.pop("api", name),
        partition,
        tuple(keys.split(",")),
        columns,
        **kwargs,
    )


INDEXES = {
    "csi300": "000300.SH",
    "csi500": "000905.SH",
    "csi1000": "000852.SH",
    "csi2000": "932000.CSI",
    "csi_all": "000985.CSI",
}
DATASETS = {
    d.name: d
    for d in [
        dataset(
            "trade_cal",
            "year",
            "exchange,cal_date",
            "exchange,cal_date,is_open,pretrade_date",
            date="cal_date",
            params={"exchange": "SSE"},
            known_at="交易所提前公布",
        ),
        dataset(
            "stock_basic",
            "status",
            "ts_code",
            "ts_code,symbol,name,area,industry,market,exchange,list_status,list_date,delist_date",
            known_at="当前证券主表；名称不用于历史信号",
        ),
        dataset(
            "namechange",
            "static",
            "ts_code,start_date,name",
            "ts_code,name,start_date,end_date,ann_date,change_reason",
            date="start_date",
            known_at="公告日起；生效日见 start_date",
        ),
        dataset(
            "daily",
            "day",
            "ts_code,trade_date",
            "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount",
            limit=5000,
            units={"vol": "手", "amount": "千元"},
        ),
        dataset(
            "adj_factor",
            "day",
            "ts_code,trade_date",
            "ts_code,trade_date,adj_factor",
            limit=5000,
        ),
        dataset(
            "daily_basic",
            "day",
            "ts_code,trade_date",
            "ts_code,trade_date,close,turnover_rate,turnover_rate_f,volume_ratio,pe,pe_ttm,pb,ps,ps_ttm,dv_ratio,dv_ttm,total_share,float_share,free_share,total_mv,circ_mv",
            limit=5000,
            units={
                "total_share": "万股",
                "float_share": "万股",
                "free_share": "万股",
                "total_mv": "万元",
                "circ_mv": "万元",
                "turnover_rate": "百分数",
                "dv_ratio": "百分数",
            },
        ),
        dataset(
            "stk_limit",
            "day",
            "ts_code,trade_date",
            "ts_code,trade_date,pre_close,up_limit,down_limit",
            limit=5000,
            known_at="当日盘前；源历史自 2007-01-04",
        ),
        dataset(
            "suspend_d",
            "day",
            "ts_code,trade_date,suspend_type,suspend_timing",
            "ts_code,trade_date,suspend_timing,suspend_type",
            known_at="全天停牌盘前；盘中临停仅事后可知",
        ),
        dataset(
            "stock_st",
            "day",
            "ts_code,trade_date,type",
            "ts_code,name,trade_date,type,type_name",
            limit=1000,
            known_at="当日盘前",
        ),
        dataset(
            "index_daily",
            "index_month_daily",
            "ts_code,trade_date",
            "ts_code,trade_date,open,high,low,close,pre_close,vol,amount",
            limit=5000,
        ),
        dataset(
            "index_weight",
            "index_month",
            "index_code,con_code,trade_date",
            "index_code,con_code,trade_date,weight",
            units={"weight": "百分数"},
            known_at="快照日收盘；下一交易日起使用",
        ),
        dataset(
            "index_classify",
            "classification",
            "index_code,src",
            "index_code,industry_name,level,industry_code,is_pub,parent_code,src",
            known_at="当前分类元信息",
        ),
        dataset(
            "index_member_all",
            "industry",
            "ts_code,l3_code,in_date",
            "l1_code,l1_name,l2_code,l2_name,l3_code,l3_name,ts_code,name,in_date,out_date,is_new",
            limit=2000,
            date="in_date",
            known_at="纳入日含、剔除日含；历史分类可能被修订",
        ),
        dataset(
            "dividend",
            "day",
            "ts_code,end_date,ex_date,div_proc",
            "ts_code,end_date,ann_date,div_proc,stk_div,stk_bo_rate,stk_co_rate,cash_div,cash_div_tax,record_date,ex_date,pay_date,div_listdate,imp_ann_date",
            limit=2000,
            date="ex_date",
            known_at="实施公告日；除权日盘前使用",
            units={"stk_div": "每股送转股", "cash_div_tax": "元/股"},
        ),
        dataset(
            "fut_basic",
            "static",
            "ts_code",
            "ts_code,symbol,exchange,name,fut_code,multiplier,trade_unit,per_unit,quote_unit,quote_unit_desc,d_mode_desc,list_date,delist_date,d_month,last_ddate",
            params={"exchange": "CFFEX", "fut_type": "1"},
            known_at="合约上市前公布",
            units={"multiplier": "元/点"},
        ),
        dataset(
            "fut_daily",
            "day",
            "ts_code,trade_date",
            "ts_code,trade_date,pre_close,pre_settle,open,high,low,close,settle,change1,change2,vol,amount,oi,oi_chg",
            params={"exchange": "CFFEX"},
            units={"vol": "手", "amount": "万元", "settle": "点"},
        ),
        dataset(
            "margin_secs",
            "day",
            "ts_code,trade_date",
            "trade_date,ts_code,name,exchange",
            limit=5000,
            known_at="当日盘前",
        ),
    ]
}

# 财务保留全部版本；本步只公开以下已注明单位的字段，其余原始字段仍落盘。
FINANCIAL = {
    "income": {"revenue": "revenue", "n_income": "net_income"},
    "balancesheet": {"total_assets": "total_assets", "total_liab": "total_liabilities"},
    "cashflow": {"n_cashflow_act": "operating_cashflow"},
    "fina_indicator": {"eps": "earnings_per_share", "roe": "return_on_equity"},
    "forecast": {
        "net_profit_min": "net_income_low",
        "net_profit_max": "net_income_high",
    },
    "express": {"revenue": "revenue", "n_income": "net_income"},
}
for name in FINANCIAL:
    keys = "ts_code,end_date,ann_date,update_flag"
    if name in ("income", "balancesheet", "cashflow"):
        keys += ",f_ann_date,report_type"
    DATASETS[name] = dataset(
        name,
        "report",
        keys,
        "",
        api=name + "_vip",
        date="ann_date",
        known_at="实际公告日下一交易日；未构建时点面板",
        units={"*": "见原始响应字段；未经声明的列不公开"},
    )


@dataclass(frozen=True)
class Field:
    name: str
    dataset: str
    raw: str
    unit: str = "元"
    scale: float = 1.0
    dtype: str = "float64"
    known_at: str = "收盘后"


FIELDS: dict[str, Field] = {}


def add(name, ds, raw, unit="元", scale=1.0, dtype="float64", known_at=None):
    FIELDS[name] = Field(
        name, ds, raw, unit, scale, dtype, known_at or DATASETS[ds].known_at
    )


for name in ("open", "high", "low", "close"):
    add(name, "daily", name, known_at="开盘后" if name == "open" else "收盘后")
add("previous_close", "daily", "pre_close", known_at="盘前除权参考价")
add("volume", "daily", "vol", "股", 100)
add("amount", "daily", "amount", "元", 1000)
add("adj_factor", "adj_factor", "adj_factor", "倍")
add("upper_limit", "stk_limit", "up_limit")
add("lower_limit", "stk_limit", "down_limit")
for name, raw, unit, scale in [
    ("total_market_cap", "total_mv", "元", 10000),
    ("float_market_cap", "circ_mv", "元", 10000),
    ("total_shares", "total_share", "股", 10000),
    ("float_shares", "float_share", "股", 10000),
    ("free_shares", "free_share", "股", 10000),
    ("turnover", "turnover_rate", "比例", 0.01),
    ("pe", "pe", "倍", 1),
    ("pb", "pb", "倍", 1),
    ("ps", "ps", "倍", 1),
    ("dividend_yield", "dv_ratio", "比例", 0.01),
    ("pe_ttm", "pe_ttm", "倍", 1),
    ("ps_ttm", "ps_ttm", "倍", 1),
    ("dividend_yield_ttm", "dv_ttm", "比例", 0.01),
]:
    add(
        name,
        "daily_basic",
        raw,
        unit,
        scale,
        "float32" if unit in ("比例", "倍") else "float64",
    )
for name, ds, unit, dtype, when in [
    ("vwap", "daily", "元/股", "float64", "收盘后"),
    ("is_listed", "stock_basic", "布尔", "bool", "上市日含、退市日不含"),
    ("listing_age", "stock_basic", "交易日", "float64", "盘前；上市当日为 1"),
    ("is_st", "stock_st", "布尔", "bool", "盘前；缺分区时拒绝构建"),
    ("is_suspended", "suspend_d", "布尔", "bool", "全天停牌；盘中临停不视作全天停牌"),
    ("industry", "index_member_all", "行业编码", "float64", "纳入日含、剔除日含"),
    ("split", "dividend", "倍", "float64", "除权日盘前；无事件为 1"),
    ("cash_dividend", "dividend", "元/股", "float64", "除息日盘前；无事件为 0，税前"),
    ("delist", "stock_basic", "布尔", "bool", "退市日或其后首个交易日盘前"),
]:
    add(name, ds, "", unit, dtype=dtype, known_at=when)
for name in INDEXES:
    add(f"weight_{name}", "index_weight", "", "比例", dtype="float64")
    add(f"member_{name}", "index_weight", "", "布尔", dtype="bool")

DAILY_DATASETS = [n for n, d in DATASETS.items() if d.partition != "report"]
TABLE_RENAMES = {
    "ts_code": "symbol",
    "trade_date": "date",
    "ann_date": "announcement_date",
    "f_ann_date": "actual_announcement_date",
    "end_date": "period_end",
    "update_flag": "version_flag",
    "report_type": "report_kind",
    "ex_date": "ex_date",
    "imp_ann_date": "implementation_announcement_date",
    "div_proc": "status",
    "stk_div": "stock_dividend",
    "cash_div_tax": "cash_dividend",
    "pay_date": "payment_date",
    "record_date": "record_date",
    "_fetched_at": "fetched_at",
}

# 2026-11-12 全历史原始库实测：是本库可验证起点，不宣称源再无更早记录。
OBSERVED_STARTS = {
    "daily": "20070104",
    "adj_factor": "20070104",
    "daily_basic": "20070104",
    "stk_limit": "20070104",
    "stock_st": "20070104",
    "suspend_d": "20070104",
    "dividend": "20070104",
    "fut_daily": "20100416",
    "margin_secs": "20100329",
}

for _name, _start in OBSERVED_STARTS.items():
    DATASETS[_name] = replace(
        DATASETS[_name],
        coverage_start=_start,
        empty_allowed=_name in ("suspend_d", "dividend"),
    )
INDEX_STARTS = {
    "csi300": "20061229",
    "csi500": "20061229",
    "csi1000": "20141031",
    "csi2000": "20230831",
    "csi_all": "20110831",
}
SCALES = {
    "index_weight": 0.01,
    "forecast_profit": 10000,
    "roe": 0.01,
    "baostock_volume": 1,
    "tushare_volume": FIELDS["volume"].scale,
}
INDUSTRY_CAVEAT = "纳入剔除两端均含；缺口用SW2021回填，2014-02-21前均为重述；industry_source=1/2才是当时口径，3带后见之明；覆盖统计见字段元数据coverage"
FIELDS["industry"] = replace(FIELDS["industry"], known_at=INDUSTRY_CAVEAT)
add(
    "has_quote",
    "daily",
    "",
    "布尔",
    dtype="bool",
    known_at="收盘后；是否存在源日线，不把缺行情猜成停牌",
)
add(
    "st_known",
    "stock_st",
    "",
    "布尔",
    dtype="bool",
    known_at="盘前；源覆盖且日分区通过完整性校验",
)

DATASETS["index_member"] = dataset(
    "index_member",
    "legacy_industry",
    "index_code,con_code,in_date",
    "index_code,con_code,in_date,out_date,is_new",
    date="in_date",
    limit=2000,
    known_at="旧版申万区间；仅用于 SW2021 切换前",
)
DAILY_DATASETS.append("index_member")

# 公开长表的统一字段；数值换算只在此声明。
PUBLIC_TABLES = {
    "index_daily": {
        "ts_code": ("symbol", 1),
        "trade_date": ("date", 1),
        **{c: (c, 1) for c in ("open", "high", "low", "close")},
        "pre_close": ("previous_close", 1),
        "vol": ("volume", 100),
        "amount": ("amount", 1000),
    },
    "fut_basic": {
        "ts_code": ("symbol", 1),
        "fut_code": ("underlying", 1),
        "name": ("name", 1),
        "exchange": ("exchange", 1),
        "multiplier": ("contract_multiplier", 1),
        "list_date": ("listing_date", 1),
        "delist_date": ("expiry_date", 1),
        "last_ddate": ("last_delivery_date", 1),
    },
    "fut_daily": {
        "ts_code": ("symbol", 1),
        "trade_date": ("date", 1),
        **{c: (c, 1) for c in ("open", "high", "low", "close")},
        "pre_close": ("previous_close", 1),
        "settle": ("settlement", 1),
        "pre_settle": ("previous_settlement", 1),
        "vol": ("volume_contracts", 1),
        "amount": ("amount", 10000),
        "oi": ("open_interest_contracts", 1),
    },
    "margin_secs": {
        "ts_code": ("symbol", 1),
        "trade_date": ("date", 1),
        "name": ("name", 1),
        "exchange": ("exchange", 1),
    },
    "namechange": {
        "ts_code": ("symbol", 1),
        "name": ("name", 1),
        "start_date": ("effective_from", 1),
        "end_date": ("effective_through", 1),
        "ann_date": ("announcement_date", 1),
        "change_reason": ("reason", 1),
    },
    "index_member_all": {
        "ts_code": ("symbol", 1),
        "l1_code": ("industry_code", 1),
        "l1_name": ("industry_name", 1),
        "in_date": ("effective_from", 1),
        "out_date": ("effective_through", 1),
    },
    "index_member": {
        "con_code": ("symbol", 1),
        "index_code": ("industry_code", 1),
        "in_date": ("effective_from", 1),
        "out_date": ("effective_through", 1),
    },
    "index_classify": {
        "index_code": ("industry_code", 1),
        "industry_name": ("industry_name", 1),
        "src": ("classification", 1),
        "level": ("level", 1),
    },
}
TABLE_DATES = {
    "namechange": "effective_from",
    "index_member_all": "effective_from",
    "index_member": "effective_from",
    "fut_basic": "listing_date",
    "index_classify": None,
}
for _name, _mapping in FINANCIAL.items():
    PUBLIC_TABLES[_name] = {
        raw: (
            public,
            SCALES["forecast_profit"]
            if _name == "forecast" and raw.startswith("net_profit_")
            else SCALES["roe"]
            if raw == "roe"
            else 1,
        )
        for raw, public in {**TABLE_RENAMES, **_mapping}.items()
    }
    TABLE_DATES[_name] = "announcement_date"
PUBLIC_TABLES["dividend"] = {raw: (public, 1) for raw, public in TABLE_RENAMES.items()}
TABLE_DATES["dividend"] = "ex_date"

# 2026-11-12 本库实测首个非空分区；静态表为分区名，不推断接口更早不可用。
OBSERVED_FIRST_PARTITIONS = {
    "trade_cal": "1990",
    "stock_basic": "D",
    "namechange": "all",
    "daily": "20070104",
    "adj_factor": "20070104",
    "daily_basic": "20070104",
    "stk_limit": "20070104",
    "suspend_d": "20070104",
    "stock_st": "20070104",
    "index_daily": "000300.SH_200701",
    "index_weight": "000300.SH_200612",
    "index_classify": "SW2014",
    "index_member_all": "801010.SI_N",
    "dividend": "20070104",
    "fut_basic": "all",
    "fut_daily": "20100416",
    "margin_secs": "20100329",
    "income": "20070331",
    "balancesheet": "20070331",
    "cashflow": "20070331",
    "fina_indicator": "20070331",
    "forecast": "20070331",
    "express": "20070331",
    "index_member": "801010.SI_N",
}

# 本库 daily 实测旧码末个成交日与其后新码首个成交日；新码另有源重述历史。
SYMBOL_ALIASES = (
    {
        "old": "000022.SZ",
        "new": "001872.SZ",
        "old_last": "20181220",
        "switch": "20181226",
    },
    {
        "old": "000043.SZ",
        "new": "001914.SZ",
        "old_last": "20191213",
        "switch": "20191216",
    },
    {
        "old": "300114.SZ",
        "new": "302132.SZ",
        "old_last": "20250214",
        "switch": "20250217",
    },
)
ACTION_RULE = "收盘后因子对账；候选顺序为去重全部、仅现金、仅送转、逐条；容差默认0.5%；无匹配时cash=0、split=因子比，含缩股；等效参考价再投资，不代表实际分配股数"
FIELDS["split"] = replace(FIELDS["split"], known_at=ACTION_RULE)
FIELDS["cash_dividend"] = replace(FIELDS["cash_dividend"], known_at=ACTION_RULE)
FIELDS["industry"] = replace(
    FIELDS["industry"],
    dtype="uint32",
    known_at="0缺失；行业代码为整数，须结合industry_source区分当时分类与历史回填",
)
FIELDS["listing_age"] = replace(
    FIELDS["listing_age"], dtype="uint16", known_at="盘前；上市当日为1，非上市为0"
)
FIELDS.pop("st_known")
FIELDS["adj_factor"] = replace(
    FIELDS["adj_factor"],
    known_at="收盘后；保留源微小跳变，公司行为将不超过0.1%的无声明跳变视为噪声",
)
for _limit in ("upper_limit", "lower_limit"):
    FIELDS[_limit] = replace(
        FIELDS[_limit],
        known_at="盘前源价格；少数半分舍入与公式差1分，判断触及涨跌停时应容许1分误差，校验仍逐项报告",
    )
FIELDS["is_suspended"] = replace(
    FIELDS["is_suspended"],
    known_at="盘前源全天停牌记录；全历史曾有52534格上市无行情但无停牌记录、15594格有行情却标全天停牌；能否成交使用tradable",
)
add(
    "tradable",
    "daily",
    "",
    "布尔",
    dtype="bool",
    known_at="收盘后；has_quote且volume>0；是日频可成交代理，不保证盘中任意时刻可成交",
)
add(
    "action_source",
    "dividend",
    "",
    "来源枚举",
    dtype="uint8",
    known_at="收盘后；0无事件，1分红表原样，2经济去重/候选裁决/日期对齐后，3因子合成，4分红表事件被否决；规则见split",
)
add(
    "industry_source",
    "index_member_all",
    "",
    "来源枚举",
    dtype="uint8",
    known_at="0缺失，1 SW2014当时口径，2 SW2021当时口径，3历史回填/重述（带后见之明）；严格研究仅用1/2",
)
for _index in INDEXES:
    for _prefix in ("weight", "member"):
        _key = f"{_prefix}_{_index}"
        FIELDS[_key] = replace(
            FIELDS[_key],
            known_at="月末快照的下一交易日起可用；沿用最近已知快照，实际调样可能早两到三周；日龄见metadata.diagnostics.index_snapshots，超过40天告警",
        )
