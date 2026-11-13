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
    raw_dir: str | None = None


def dataset(name, partition, keys, columns, **kwargs):
    return Dataset(
        name,
        kwargs.pop("api", name),
        partition,
        tuple(keys.split(",")),
        columns,
        **kwargs,
    )


# 股票池用的指数：成分、权重（面板 weight_*/member_* 字段）只覆盖这几个。
INDEXES = {
    "csi300": "000300.SH",
    "csi500": "000905.SH",
    "csi1000": "000852.SH",
    "csi2000": "932000.CSI",
    "csi_all": "000985.CSI",
}
# 研究平台可选的基准：主流指数，写死的名单（用户 2026-11-13 定：基准只用主流指数）。
# 日线与 INDEXES 走同一套逐指数逐月分区；只下日线，不下成分和权重。
# 加一个指数：在这里加一行，运行 sync --datasets index_daily，见 docs/DATA_NOTES.md。
BENCHMARKS = {
    "000001.SH": "上证指数",
    "399001.SZ": "深证成指",
    "399006.SZ": "创业板指",
    "000688.SH": "科创50",
    "899050.BJ": "北证50",
    "000016.SH": "上证50",
    "000300.SH": "沪深300",
    "000510.SH": "中证A500",
    "000905.SH": "中证500",
    "000906.SH": "中证800",
    "000852.SH": "中证1000",
    "932000.CSI": "中证2000",
    "000985.CSI": "中证全指",
    "399303.SZ": "国证2000",
    "399330.SZ": "深证100",
    "399673.SZ": "创业板50",
    "000698.SH": "科创100",
    "000922.CSI": "中证红利",
    "000015.SH": "上证红利",
    # 2016-06 至 2018-05 源里只有收盘价；基准线只用收盘价，需要开盘价的计算要避开这段。
    "H30269.CSI": "红利低波",
    "000919.CSI": "300价值",
    "000918.CSI": "300成长",
}
DEFAULT_BENCHMARK = "000001.SH"
# index_daily 下载的指数：股票池指数在前（与原来同序），再加基准名单里的其余指数。
INDEX_DAILY_CODES = tuple(dict.fromkeys([*INDEXES.values(), *BENCHMARKS]))
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

# 公开长表里取浮点的统一字段，其余为字符串。只用于把「整列为空」（Null 类型）的原始列
# 定成正常类型，例如窗口里只有空分区时；已有类型的列不动。
PUBLIC_FLOAT_COLUMNS = frozenset(
    {
        "open",
        "high",
        "low",
        "close",
        "previous_close",
        "volume",
        "amount",
        "contract_multiplier",
        "settlement",
        "previous_settlement",
        "volume_contracts",
        "open_interest_contracts",
        "stock_dividend",
        "cash_dividend",
        "revenue",
        "net_income",
        "total_assets",
        "total_liabilities",
        "operating_cashflow",
        "earnings_per_share",
        "return_on_equity",
        "net_income_low",
        "net_income_high",
    }
)


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
ACTION_RULE = (
    "收盘后因子对账；候选顺序为去重全部、仅现金、仅送转、逐条、差异化摊薄；容差默认0.5%。"
    "差异化分红（回购专户不参与分配）在公告值按同一比例k∈[0.85,1)摊薄后与因子比一致时记持有人实得的公告值（来源5），"
    "按复权因子算的收益比按事件算的低(1-k)×分配比例。"
    "无匹配时cash=0、split=因子比，含缩股（来源3）：等效参考价再投资，不扣红利税、没有现金流、持股可为非整数，不代表实际分配股数。"
    "派现小于最小报价单位、交易所不除权时按无事件处理，持有人实得的小额现金不计。"
    "按事件记账（除权日开盘调整持仓）不受收盘后裁决影响；当作当日盘前信号使用即前视"
)
FIELDS["split"] = replace(FIELDS["split"], known_at=ACTION_RULE)
FIELDS["cash_dividend"] = replace(FIELDS["cash_dividend"], known_at=ACTION_RULE)
FIELDS["industry"] = replace(
    FIELDS["industry"],
    dtype="uint32",
    known_at="0缺失；行业代码为整数，须结合industry_source区分当时分类与历史回填；回填后同一天可能混有SW2014与SW2021两套编码，计数见metadata.diagnostics.industry_mixing",
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
_OPEN_RULE = (
    "以当日开盘价成交的回测掩码，与panelbt.markets.ashare.tradability逐格相同："
    "有效=开盘价有限且>0、非全天停牌(is_suspended)且有行情(has_quote)；"
    "{side}；容差0.005元；涨跌停价缺失视为该方向无限制。"
    "一字板用当日最高最低价判断（收盘后才知道），只能当回测的成交约束，不能当盘前信号"
)
add(
    "can_buy_open",
    "daily",
    "",
    "布尔",
    dtype="bool",
    known_at=_OPEN_RULE.format(
        side="开盘价≥涨停价−容差不能买；最高最低价相差≤容差且最高价≥涨停价−容差（一字涨停）不能买"
    ),
)
add(
    "can_sell_open",
    "daily",
    "",
    "布尔",
    dtype="bool",
    known_at=_OPEN_RULE.format(
        side="开盘价≤跌停价+容差不能卖；最高最低价相差≤容差且最低价≤跌停价+容差（一字跌停）不能卖"
    ),
)
add(
    "action_source",
    "dividend",
    "",
    "来源枚举",
    dtype="uint8",
    known_at="收盘后；0无事件，1分红表原样，2经济去重/候选裁决/日期对齐后，3因子合成，4分红表事件被否决，5差异化分红（公告值，参考价按摊薄比例k，分布见metadata.diagnostics.actions.diluted）；规则见split",
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


# 境内全部期货交易所（第 1 批期货数据）。与上方中金所对冲腿用的 fut_basic、fut_daily 互不影响：
# 原始文件按数据集名单独成目录，由 ashare.data.futures_sync 采集。它们不并入 DATASETS：
# 股票的同步、状态、校验、版本只看 DATASETS，共用清单里多出期货条目时股票侧的输出逐位不变。
# 只有 Store 这类按名字找声明的地方用 ALL_DATASETS。
FUTURES_EXCHANGES = ("CFFEX", "SHFE", "DCE", "CZCE", "INE", "GFEX")
FUTURES_DATASETS = {
    d.name: d
    for d in [
        dataset(
            "futures_calendar",
            "fut_exchange",
            "exchange,cal_date",
            "exchange,cal_date,is_open,pretrade_date",
            api="trade_cal",
            date="cal_date",
            limit=10000,
            known_at="交易所提前公布",
            raw_dir="futures_calendar",
        ),
        dataset(
            "futures_contracts",
            "fut_exchange",
            "ts_code",
            "ts_code,symbol,exchange,name,fut_code,multiplier,trade_unit,per_unit,quote_unit,quote_unit_desc,d_mode_desc,list_date,delist_date,d_month,last_ddate",
            api="fut_basic",
            date="list_date",
            limit=10000,
            params={"fut_type": "1"},
            known_at="合约挂牌前公布；交易单位与报价单位为当前值，历史修订见文档",
            raw_dir="futures_contracts",
        ),
        dataset(
            "futures_daily",
            "fut_day",
            "ts_code,trade_date",
            "ts_code,trade_date,pre_close,pre_settle,open,high,low,close,settle,change1,change2,vol,amount,oi,oi_chg",
            api="fut_daily",
            limit=2000,
            coverage_start="19950417",
            units={"vol": "手（2020 年前部分交易所为双边）", "amount": "万元"},
            known_at="交易日日盘收盘后；交易日含前一晚夜盘",
            raw_dir="futures_daily",
        ),
        dataset(
            "futures_mapping",
            "fut_day",
            "ts_code,trade_date",
            "ts_code,trade_date,mapping_ts_code",
            api="fut_mapping",
            limit=5000,
            coverage_start="19950417",
            known_at="源未写判定时点；规则由数据反推，见 docs/FUTURES_DATA.md",
            raw_dir="futures_mapping",
        ),
        dataset(
            "futures_settle",
            "fut_day",
            "ts_code,trade_date",
            "ts_code,trade_date,settle,trading_fee_rate,trading_fee,delivery_fee,b_hedging_margin_rate,s_hedging_margin_rate,long_margin_rate,short_margin_rate,offset_today_fee,exchange",
            api="fut_settle",
            limit=1000,
            coverage_start="20110104",
            known_at="交易日结算后公布，作用于下一交易日；单位按交易所不同，见 docs/FUTURES_DATA.md",
            raw_dir="futures_settle",
        ),
    ]
}
ALL_DATASETS = {**DATASETS, **FUTURES_DATASETS}
assert not set(DATASETS) & set(FUTURES_DATASETS)


# 场内基金（ETF、LOF）与可转债（第 2 批）。与期货一样单独成组、不并入 DATASETS：
# 股票的同步、状态、校验、内容版本只看 DATASETS，共用清单里多出这些条目时股票侧逐位不变。
# 采集见 ashare.data.fund_sync，口径见 docs/FUND_CB_DATA.md。原始字段名只出现在 fund_*/cb_* 模块。
FUND_DATASETS = {
    d.name: d
    for d in [
        dataset(
            "fund_calendar",
            "fund_exchange",
            "exchange,cal_date",
            "exchange,cal_date,is_open,pretrade_date",
            api="trade_cal",
            date="cal_date",
            limit=5000,
            known_at="交易所提前公布",
            raw_dir="fund_calendar",
        ),
        dataset(
            "fund_basic",
            "fund_status",
            "ts_code",
            "ts_code,name,management,custodian,fund_type,found_date,due_date,list_date,issue_date,delist_date,issue_amount,m_fee,c_fee,duration_year,p_value,min_amount,exp_return,benchmark,status,invest_type,type,trustee,purc_startdate,redm_startdate,market",
            date="list_date",
            limit=2000,
            params={"market": "E"},
            known_at="当前基金主表（含退市）；名称、类型为现值",
            raw_dir="fund_basic",
        ),
        dataset(
            "fund_daily",
            "day",
            "ts_code,trade_date",
            "ts_code,trade_date,pre_close,open,high,low,close,change,pct_chg,vol,amount",
            limit=2000,
            units={"vol": "手（100 份）", "amount": "千元"},
            coverage_start="19980407",
            empty_allowed=False,
            raw_dir="fund_daily",
        ),
        dataset(
            "fund_adj",
            "day",
            "ts_code,trade_date",
            "ts_code,trade_date,adj_factor",
            limit=2000,
            coverage_start="20050223",
            empty_allowed=False,
            raw_dir="fund_adj",
        ),
        dataset(
            "fund_share",
            "day",
            "ts_code,trade_date",
            "ts_code,trade_date,fd_share,fund_type,market",
            limit=2000,
            units={"fd_share": "万份"},
            coverage_start="20041230",
            known_at="交易所公布的份额，通常为当日收盘后或次日公布；回测按次日可知",
            # 季末那天源还返回场外基金（行数是平日的十几倍），次日行数「骤降」是常态，不做骤降检查。
            raw_dir="fund_share",
        ),
        dataset(
            "fund_div",
            "day",
            "ts_code,ann_date,ex_date,div_proc",
            "ts_code,ann_date,imp_anndate,base_date,div_proc,record_date,ex_date,pay_date,earpay_date,net_ex_date,div_cash,base_unit,ear_distr,ear_amount,account_date,base_year",
            date="ex_date",
            limit=2000,
            units={"div_cash": "元/份（税前）"},
            coverage_start="19990101",
            known_at="除息日盘前已知（实施公告早于除息日）",
            raw_dir="fund_div",
        ),
        dataset(
            "cb_basic",
            "static",
            "ts_code",
            "ts_code,bond_full_name,bond_short_name,cb_type,cb_code,stk_code,stk_short_name,maturity,par,issue_price,issue_size,remain_size,value_date,maturity_date,rate_type,coupon_rate,add_rate,pay_per_year,list_date,delist_date,exchange,conv_start_date,conv_end_date,conv_stop_date,first_conv_price,conv_price,rate_clause",
            date="list_date",
            limit=2000,
            known_at="当前主表；conv_price、remain_size 为现值，历史值见 cb_price_chg、cb_share",
            raw_dir="cb_basic",
        ),
        dataset(
            "cb_issue",
            "static",
            "ts_code",
            "ts_code,ann_date,res_ann_date,plan_issue_size,issue_size,issue_price,issue_type,onl_code,onl_name,onl_date,onl_size,onl_pch_vol,onl_pch_num,onl_pch_excess,shd_ration_code,shd_ration_name,shd_ration_date,shd_ration_record_date,shd_ration_pay_date,shd_ration_price,shd_ration_ratio,shd_ration_size,offl_size",
            date="ann_date",
            limit=2000,
            known_at="发行公告日",
            raw_dir="cb_issue",
        ),
        dataset(
            "cb_call",
            "static",
            "ts_code,call_type,is_call,ann_date",
            "ts_code,call_type,is_call,ann_date,call_date,call_price,call_price_tax,call_vol,call_amount,payment_date,call_reg_date",
            date="ann_date",
            limit=2000,
            known_at="公告日收盘后；按公告日的下一交易日可知",
            raw_dir="cb_call",
        ),
        dataset(
            "cb_rate",
            "static",
            "ts_code,rate_start_date",
            "ts_code,rate_freq,rate_start_date,rate_end_date,coupon_rate",
            date="rate_start_date",
            limit=2000,
            units={"coupon_rate": "%"},
            known_at="发行时约定",
            raw_dir="cb_rate",
        ),
        dataset(
            "cb_price_chg",
            "cb_batch",
            "ts_code,publish_date,change_date",
            "ts_code,bond_short_name,publish_date,change_date,convert_price_initial,convertprice_bef,convertprice_aft",
            date="change_date",
            limit=2000,
            known_at="公告日收盘后；新转股价自 max(生效日, 公告日的下一交易日) 起使用",
            raw_dir="cb_price_chg",
        ),
        dataset(
            "cb_share",
            "cb_code",
            "ts_code,end_date,publish_date",
            "ts_code,bond_short_name,publish_date,end_date,issue_size,convert_price_initial,convert_price,convert_val,convert_vol,convert_ratio,acc_convert_val,acc_convert_vol,acc_convert_ratio,remain_size,total_shares",
            date="end_date",
            limit=2000,
            units={"remain_size": "元（面值）", "convert_val": "元（面值）"},
            known_at="公告日收盘后；按公告日的下一交易日可知",
            raw_dir="cb_share",
        ),
        dataset(
            "cb_daily",
            "day",
            "ts_code,trade_date",
            "ts_code,trade_date,pre_close,open,high,low,close,change,pct_chg,vol,amount,bond_value,bond_over_rate,cb_value,cb_over_rate",
            limit=2000,
            units={
                "vol": "见 docs/FUND_CB_DATA.md（实测）",
                "amount": "见 docs/FUND_CB_DATA.md（实测）",
            },
            coverage_start="19930210",
            empty_allowed=False,
            raw_dir="cb_daily",
        ),
    ]
}
ALL_DATASETS = {**ALL_DATASETS, **FUND_DATASETS}
assert not set(FUND_DATASETS) & (set(DATASETS) | set(FUTURES_DATASETS))


# 港股、美股（2026-11-13 第 2 批）。原始文件写在各自市场的数据根 <root>/hk、<root>/us，
# 与 A 股的 <root>/cn 是不同的清单；登记在 OVERSEAS_DATASETS，不并入 DATASETS，A 股的
# 同步、状态、校验、版本只看 DATASETS，加不加这一段逐位不变。采集见 overseas_sync。
# 分区方式：day 按交易日（沿用 A 股日分区的空表与骤降保护），symbol 按证券取一段历史
# （样本），fx_pair 按货币对，static / status 同 A 股。权限与单页上限见 docs/HKUS_DATA.md。
OVERSEAS_DATASETS = {
    d.name: d
    for d in [
        dataset(
            "hk_calendar",
            "static",
            "cal_date",
            "cal_date,is_open,pretrade_date",
            api="hk_tradecal",
            date="cal_date",
            limit=2000,
            known_at="交易所提前公布；风球、黑雨休市只在事后体现",
            raw_dir="hk_tradecal",
        ),
        dataset(
            "hk_securities",
            "status",
            "ts_code",
            "ts_code,name,fullname,enname,cn_spell,market,list_status,list_date,delist_date,trade_unit,isin,curr_type",
            api="hk_basic",
            date="list_date",
            limit=5000,
            known_at="当前证券主表；每手股数 trade_unit 只有当前值",
            raw_dir="hk_basic",
        ),
        dataset(
            "hk_daily",
            "day",
            "ts_code,trade_date",
            "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount",
            limit=5000,
            coverage_start="19900102",
            empty_allowed=False,
            units={"vol": "股", "amount": "交易币种元（多数为港元）"},
            known_at="收盘后约 18 点；pre_close 为除权除息后的参考价",
            raw_dir="hk_daily",
        ),
        dataset(
            "hk_adjfactor",
            "day",
            "ts_code,trade_date",
            "ts_code,trade_date,cum_adjfactor,close_price",
            limit=6000,
            coverage_start="19900102",
            empty_allowed=False,
            known_at="前复权累计因子，最新日为 1，每次除权后整段历史重写：只在同一快照内可比",
            raw_dir="hk_adjfactor",
        ),
        dataset(
            "hk_daily_hist",
            "symbol",
            "ts_code,trade_date",
            "ts_code,trade_date,open,high,low,close,pre_close,change,pct_chg,vol,amount",
            api="hk_daily",
            limit=5000,
            units={"vol": "股", "amount": "交易币种元（多数为港元）"},
            known_at="同 hk_daily；按证券取 2010 年起的历史（开发库样本）",
            raw_dir="hk_daily_hist",
        ),
        dataset(
            "hk_adjfactor_hist",
            "symbol",
            "ts_code,trade_date",
            "ts_code,trade_date,cum_adjfactor,close_price",
            api="hk_adjfactor",
            limit=6000,
            known_at="同 hk_adjfactor；一个证券一次取完，序列内同一快照",
            raw_dir="hk_adjfactor_hist",
        ),
        dataset(
            "hk_connect_sh",
            "day",
            "ts_code,trade_date",
            "ts_code,trade_date,type,name,type_name",
            api="stock_hsgt",
            limit=2000,
            params={"type": "SH_HK"},
            coverage_start="20250812",
            known_at="当日港股通（沪）标的名单；源自 2025-08-12 起才有",
            raw_dir="hk_connect_sh",
        ),
        dataset(
            "hk_connect_sz",
            "day",
            "ts_code,trade_date",
            "ts_code,trade_date,type,name,type_name",
            api="stock_hsgt",
            limit=2000,
            params={"type": "SZ_HK"},
            coverage_start="20250812",
            known_at="当日港股通（深）标的名单；源自 2025-08-12 起才有",
            raw_dir="hk_connect_sz",
        ),
        dataset(
            "hk_connect_hold",
            "day",
            "ts_code,trade_date",
            "code,trade_date,ts_code,name,vol,ratio,exchange",
            api="hk_hold",
            limit=3000,
            params={"exchange": "HK"},
            coverage_start="20170317",
            units={"vol": "股", "ratio": "百分数"},
            known_at="中央结算系统日终持股（南向）；次一交易日公布",
            raw_dir="hk_connect_hold",
        ),
        dataset(
            "us_calendar",
            "static",
            "cal_date",
            "cal_date,is_open,pretrade_date",
            api="us_tradecal",
            date="cal_date",
            limit=5000,
            known_at="交易所提前公布；临时休市只在事后体现",
            raw_dir="us_tradecal",
        ),
        dataset(
            "us_securities",
            "static",
            "ts_code,list_date,enname",
            "ts_code,name,enname,classify,list_date,delist_date",
            api="us_basic",
            date="list_date",
            limit=6000,
            known_at="当前证券主表；源忽略 list_status，几乎没有退市日；代码会复用",
            raw_dir="us_basic",
        ),
        dataset(
            "us_daily",
            "day",
            "ts_code,trade_date",
            "ts_code,trade_date,close,open,high,low,pre_close,change,pct_change,vol,amount,vwap,turnover_ratio,total_mv,pe,pb",
            limit=6000,
            coverage_start="19900102",
            empty_allowed=False,
            units={"vol": "股", "amount": "美元", "total_mv": "美元"},
            known_at="收盘后；pre_close 为上一交易日原始收盘价（不除权）",
            raw_dir="us_daily",
        ),
        dataset(
            "us_adjfactor",
            "day",
            "ts_code,trade_date,exchange",
            "ts_code,trade_date,exchange,cum_adjfactor,close_price",
            limit=15000,
            coverage_start="20060103",
            empty_allowed=False,
            known_at="前复权累计因子（实测只含拆并股），最新日为 1，每次拆并股后整段历史重写",
            raw_dir="us_adjfactor",
        ),
        dataset(
            "us_daily_hist",
            "symbol",
            "ts_code,trade_date",
            "ts_code,trade_date,close,open,high,low,pre_close,change,pct_change,vol,amount,vwap,turnover_ratio,total_mv,pe,pb",
            api="us_daily",
            limit=6000,
            units={"vol": "股", "amount": "美元", "total_mv": "美元"},
            known_at="同 us_daily；按证券取 2010 年起的历史（开发库样本）",
            raw_dir="us_daily_hist",
        ),
        dataset(
            "us_adjfactor_hist",
            "symbol",
            "ts_code,trade_date,exchange",
            "ts_code,trade_date,exchange,cum_adjfactor,close_price",
            api="us_adjfactor",
            limit=15000,
            known_at="同 us_adjfactor；一个证券一次取完，序列内同一快照",
            raw_dir="us_adjfactor_hist",
        ),
        dataset(
            "overseas_fx_daily",
            "fx_pair",
            "ts_code,trade_date",
            "ts_code,trade_date,bid_open,bid_close,bid_high,bid_low,ask_open,ask_close,ask_high,ask_low,tick_qty",
            api="fx_daily",
            limit=1000,
            known_at="FXCM 报价；trade_date 为格林尼治日期，见 docs/HKUS_DATA.md",
            raw_dir="fx_daily",
        ),
        dataset(
            "overseas_fx_midrate",
            "fx_pair",
            "ts_code,trade_date",
            "trade_date,ts_code,mid_rate",
            api="fx_midrate",
            limit=3000,
            known_at="外汇交易中心当日 9:15 公布的人民币中间价；100 外币或 100 人民币报价",
            raw_dir="fx_midrate",
        ),
    ]
}
ALL_DATASETS.update(OVERSEAS_DATASETS)
assert not set(DATASETS) & set(OVERSEAS_DATASETS)
