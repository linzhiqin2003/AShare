"""港股、美股面板的统一字段（名称、单位、类型、可知时刻）。

同名字段与 A 股面板的含义和单位相同（价格为每股交易币种元，成交量为股，成交额为交易币种元）；
A 股专有的涨跌停、ST、行业、指数权重、估值字段不提供。币种：港股 HKD（少数证券以人民币或美元
交易，见证券表 ``currency``），美股 USD。
"""

from dataclasses import asdict, dataclass

from .overseas_actions import RULE as ACTION_RULE


@dataclass(frozen=True)
class OverseasField:
    name: str
    unit: str
    dtype: str
    known_at: str
    markets: tuple = ("hk", "us")


FIELDS = {
    f.name: f
    for f in [
        OverseasField("open", "交易币种元/股", "float64", "开盘后"),
        OverseasField("high", "交易币种元/股", "float64", "收盘后"),
        OverseasField("low", "交易币种元/股", "float64", "收盘后"),
        OverseasField(
            "close",
            "交易币种元/股",
            "float64",
            "收盘后；港股 2020-07-02 至 2023-12-29 源日线四舍五入到 0.01，改用因子表的精确收盘价（见 price_rounded）",
        ),
        OverseasField(
            "previous_close",
            "交易币种元/股",
            "float64",
            "盘前；港股为除权除息后的参考价，美股为上一交易日原始收盘价（不除权），两地口径不同",
        ),
        OverseasField("volume", "股", "float64", "收盘后；只含常规交易时段（见文档）"),
        OverseasField(
            "amount", "交易币种元", "float64", "收盘后；美股 2016 年前多为空"
        ),
        OverseasField(
            "vwap",
            "交易币种元/股",
            "float64",
            "收盘后；美股取源的 vwap，港股为 amount/volume",
        ),
        OverseasField(
            "adj_factor",
            "倍",
            "float64",
            "收盘后；只累乘执行了的公司行为（拆并股、按股数的调整、港股的等效现金再投资），每列第一个有因子的交易日为 1，"
            "只有比值有意义。美股不含现金分红（价格收益加拆并股），可疑事件不计入",
        ),
        OverseasField("action_ratio", "倍", "float64", ACTION_RULE),
        OverseasField("split", "倍", "float64", ACTION_RULE),
        OverseasField("equivalent_cash", "交易币种元/股", "float64", ACTION_RULE),
        OverseasField(
            "action_source",
            "来源枚举",
            "uint8",
            "0 无事件、1 拆并股、2 等效现金（港股小比值）、3 非整数下调、4 按股数（大比值）、"
            "5 美股不执行（现金类，或小窗口里价格没印证的非整数股数调整）、6 前收或价格未印证不执行、"
            "7 最后成交日之后不执行。" + ACTION_RULE,
        ),
        OverseasField(
            "action_suspect",
            "布尔",
            "bool",
            "事后；因子在这天变了但没有执行（action_source 6、7）。6 要看下一个成交日的价格，7 要知道以后"
            "再没有成交：只用于事后诊断，不能用来选样本或剔除时段（那是前视）。证券表 action_suspect 按证券汇总",
        ),
        OverseasField(
            "action_check",
            "来源枚举",
            "uint8",
            "下一个成交日之后；事件格所在窗口的印证方式：0 无事件、1 交易所前收（港股）、2 开盘价、"
            "3 只靠当日收盘价（日内前视：开盘撮合时还不知道）、4 没有上一个成交日的小比值窗口、"
            "5 没有印证（action_source 6）、6 最后成交日之后（7）。严格的使用者可以把 3 当作当天收盘后才知道",
        ),
        OverseasField(
            "has_quote",
            "布尔",
            "bool",
            "收盘后；源当天有这只证券的日线（港股停牌日源不出行）",
        ),
        OverseasField("tradable", "布尔", "bool", "收盘后；has_quote 且 volume>0"),
        OverseasField(
            "is_listed",
            "布尔",
            "bool",
            "港股按证券表上市日（含）至退市日（不含）；美股按成交推断、只用当时可知的信息：第一次成交起为真，"
            "连续 20 个交易日没有成交后的下一天起为假，再有成交又为真；美股拆段的旧列从新列开始那天起为假"
            "（新旧两列不同时为真）",
        ),
        OverseasField(
            "delist",
            "布尔",
            "bool",
            "港股为证券表退市日或其后首个交易日；美股为最后一次成交后满 20 个交易日的下一天（那时才可知），"
            "且之后再没有成交（是否永久退市是事后判断，源没有退市日）；美股拆段处的旧列在新列开始那天"
            "（复牌日）为真，回收率见 delist_recovery",
        ),
        OverseasField(
            "delist_recovery",
            "倍",
            "float64",
            "复牌日收盘后；只在美股拆段旧列的 delist 格有值，其余为 NaN（按回测配置的退市假设了结）。"
            "清算价 = min(停牌前收盘, 复牌收盘 × 缺口里执行了的倍数)，对多头偏悲观；值为清算价相对引擎记账价"
            "（旧列最后收盘价，按其后执行的拆并股调整）的倍数，与 panelbt 的 delist_recovery 同一口径",
            ("us",),
        ),
        OverseasField(
            "connect_eligible",
            "布尔",
            "bool",
            "盘前；当天在港股通（沪或深）标的名单里。名单源 2025-08-12 起才有，之前恒为 False，须结合 connect_known",
            ("hk",),
        ),
        OverseasField(
            "connect_sell_only",
            "布尔",
            "bool",
            "盘前；不在当天名单里但上一交易日南向（中央结算）有持股：港股通只能卖出。只在 connect_known 为真时有意义",
            ("hk",),
        ),
        OverseasField(
            "connect_known",
            "布尔",
            "bool",
            "当天两份港股通名单分区都已完成（2025-08-12 起）",
            ("hk",),
        ),
        OverseasField(
            "connect_held",
            "布尔",
            "bool",
            "盘前；上一交易日日终南向（港股通）在中央结算的持股大于 0（源次一交易日公布，面板已错位一天）。2017-03-17 起有",
            ("hk",),
        ),
        OverseasField(
            "price_rounded",
            "布尔",
            "bool",
            "源 hk_daily 在 2020-07-02 至 2023-12-29 把价格四舍五入到 0.01：这段日子收盘价低于 0.5 港元（最小价位小于 0.01）的格为真。"
            "收盘价已改用因子表的精确 close_price，开高低价和前收仍是四舍五入后的值",
            ("hk",),
        ),
        OverseasField(
            "total_market_cap",
            "美元",
            "float64",
            "收盘后；源 us_daily.total_mv",
            ("us",),
        ),
    ]
}


def fields_for(market):
    return {n: f for n, f in FIELDS.items() if market in f.markets}


def describe(market):
    return {n: asdict(f) for n, f in fields_for(market).items()}
