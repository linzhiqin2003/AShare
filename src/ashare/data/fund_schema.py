"""基金、可转债面板的字段表：名字、单位、类型、时点。

同名字段的含义和单位与 A 股面板一致（价格「元」、成交量按最小计量单位、成交额「元」、
`is_listed` 上市日含退市日不含、`delist` 退市日或其后首个交易日、`split` 无事件为 1、
`cash_dividend` 无事件为 0）。原始字段名与单位换算只出现在 `fund_panel.py`、`cb_panel.py`。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FieldSpec:
    name: str
    unit: str
    dtype: str
    known_at: str


def _table(rows):
    return {r[0]: FieldSpec(*r) for r in rows}


FUND_FIELDS = _table(
    [
        ("open", "元/份", "float64", "开盘后"),
        ("high", "元/份", "float64", "收盘后"),
        ("low", "元/份", "float64", "收盘后"),
        ("close", "元/份", "float64", "收盘后"),
        (
            "previous_close",
            "元/份",
            "float64",
            "盘前；交易所给的前收盘（除息、拆分后的参考价）",
        ),
        ("volume", "份", "float64", "收盘后"),
        ("amount", "元", "float64", "收盘后"),
        (
            "adj_factor",
            "倍",
            "float64",
            "收盘后；源的复权因子，后复权价 = close × adj_factor",
        ),
        (
            "total_shares",
            "份",
            "float64",
            "交易所公布的基金份额；通常收盘后或次日公布，严格回测请滞后一天使用",
        ),
        (
            "cash_dividend",
            "元/份",
            "float64",
            "盘前；无事件为 0，税前；分红表去重后以复权因子裁决，落在除息日或其后第一个有行情日",
        ),
        (
            "split",
            "倍",
            "float64",
            "盘前；无事件为 1；拆分、折算以复权因子为准，停牌期间的事件落在复牌第一天，见 docs/FUND_CB_DATA.md",
        ),
        (
            "action_source",
            "代码",
            "uint8",
            "split / cash_dividend 的来源：0 无，1 分红与因子一致，2 分红加合成折算，3 因子合成拆分，"
            "4 分红被否决（因子没动），5 因子缺失按前收盘缺口，6 之后没有行情、按公告记",
        ),
        (
            "has_quote",
            "布尔",
            "bool",
            "收盘后；当天有源日线且收盘价有效；开高低收相同而成交均价不在这个价上的沿用价不算",
        ),
        ("is_listed", "布尔", "bool", "上市日含、退市日不含"),
        ("delist", "布尔", "bool", "退市日或其后首个交易日盘前"),
    ]
)

CB_FIELDS = _table(
    [
        ("open", "元/张", "float64", "开盘后；每张面值 100 元"),
        ("high", "元/张", "float64", "收盘后"),
        ("low", "元/张", "float64", "收盘后"),
        ("close", "元/张", "float64", "收盘后"),
        ("previous_close", "元/张", "float64", "盘前"),
        (
            "volume",
            "张",
            "float64",
            "收盘后；单位换算的实测依据见 docs/FUND_CB_DATA.md",
        ),
        (
            "amount",
            "元",
            "float64",
            "收盘后；单位换算的实测依据见 docs/FUND_CB_DATA.md",
        ),
        ("has_quote", "布尔", "bool", "收盘后；当天有源日线且收盘价有效"),
        ("is_listed", "布尔", "bool", "上市日含、退市日（摘牌日）不含"),
        ("delist", "布尔", "bool", "摘牌日或其后首个交易日盘前"),
        (
            "conversion_price",
            "元/股",
            "float64",
            "当时可知的转股价：上市时为初始转股价，每次调整自 max(生效日, 公告日的下一交易日) 起用新值",
        ),
        (
            "stock_close",
            "元/股",
            "float64",
            "正股当日收盘价（不复权），取自股票面板；正股停牌或无数据为 NaN",
        ),
        (
            "conversion_value",
            "元/张",
            "float64",
            "收盘后；100 / conversion_price × stock_close",
        ),
        (
            "conversion_premium",
            "比例",
            "float64",
            "收盘后；close / conversion_value − 1",
        ),
        (
            "double_low",
            "点",
            "float64",
            "收盘后；close + 100 × conversion_premium（常用的「双低」写法）",
        ),
        (
            "remaining_size",
            "元（面值）",
            "float64",
            "转股结果公告的下一交易日起可知；首次公告前取发行规模",
        ),
        ("call_announced", "布尔", "bool", "强赎公告日的下一交易日起为真，直到摘牌"),
        ("days_to_maturity", "自然日", "float64", "盘前；到期日减当天"),
        ("coupon_rate", "比例", "float64", "当天所在计息期的票面利率（年）"),
        (
            "source_conversion_value",
            "元/张",
            "float64",
            "源（Tushare cb_daily.cb_value）给的转股价值，只作核对；源按生效日换转股价，生效早于公告时"
            "用到了当时还没公告的转股价（含前视），不要当信号用，信号用 conversion_value",
        ),
        (
            "source_bond_value",
            "元/张",
            "float64",
            "源（Tushare cb_daily.bond_value）给的纯债价值；估值方法未公开，只作参考",
        ),
        (
            "source_bond_premium",
            "比例",
            "float64",
            "源给的纯债溢价率（bond_over_rate ÷ 100）；方法未公开，只作参考",
        ),
    ]
)

KINDS = {"fund": FUND_FIELDS, "cb": CB_FIELDS}


def describe(kind="fund"):
    """打印字段表（名字、单位、类型、时点）。"""
    rows = KINDS[kind]
    width = max(len(n) for n in rows)
    return "\n".join(
        f"{f.name:<{width}}  {f.unit:<8}  {f.dtype:<7}  {f.known_at}"
        for f in rows.values()
    )
