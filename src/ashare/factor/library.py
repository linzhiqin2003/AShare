"""字段约定、登记表和第一批日频选股因子。"""

import math
from dataclasses import dataclass

from .expression import Expr, call, field

# 由数据层 metadata.fields 的字段名和单位生成；不在因子层重命名或缩放。
FIELDS = {
    "action_source": ("来源枚举", "action_source"),
    "adj_factor": ("倍", "adj_factor"),
    "amount": ("元", "amount"),
    "cash_dividend": ("元/股", "cash_dividend"),
    "close": ("元", "close"),
    "delist": ("布尔", "delist"),
    "dividend_yield": ("比例", "dividend_yield"),
    "dividend_yield_ttm": ("比例", "dividend_yield_ttm"),
    "float_market_cap": ("元", "float_market_cap"),
    "float_shares": ("股", "float_shares"),
    "free_shares": ("股", "free_shares"),
    "has_quote": ("布尔", "has_quote"),
    "high": ("元", "high"),
    "industry": ("行业编码", "industry"),
    "industry_source": ("来源枚举", "industry_source"),
    "is_listed": ("布尔", "is_listed"),
    "is_st": ("布尔", "is_st"),
    "is_suspended": ("布尔", "is_suspended"),
    "listing_age": ("交易日", "listing_age"),
    "low": ("元", "low"),
    "lower_limit": ("元", "lower_limit"),
    "member_csi1000": ("布尔", "member_csi1000"),
    "member_csi2000": ("布尔", "member_csi2000"),
    "member_csi300": ("布尔", "member_csi300"),
    "member_csi500": ("布尔", "member_csi500"),
    "member_csi_all": ("布尔", "member_csi_all"),
    "open": ("元", "open"),
    "pb": ("倍", "pb"),
    "pe": ("倍", "pe"),
    "pe_ttm": ("倍", "pe_ttm"),
    "previous_close": ("元", "previous_close"),
    "ps": ("倍", "ps"),
    "ps_ttm": ("倍", "ps_ttm"),
    "split": ("倍", "split"),
    "total_market_cap": ("元", "total_market_cap"),
    "total_shares": ("股", "total_shares"),
    "tradable": ("布尔", "tradable"),
    "turnover": ("比例", "turnover"),
    "upper_limit": ("元", "upper_limit"),
    "volume": ("股", "volume"),
    "vwap": ("元/股", "vwap"),
    "weight_csi1000": ("比例", "weight_csi1000"),
    "weight_csi2000": ("比例", "weight_csi2000"),
    "weight_csi300": ("比例", "weight_csi300"),
    "weight_csi500": ("比例", "weight_csi500"),
    "weight_csi_all": ("比例", "weight_csi_all"),
}


def adjusted_close():
    """价格类因子统一使用复权收盘价，不改变信号日期。"""
    return field("close") * field("adj_factor")


def returns(periods=1):
    price = adjusted_close()
    return price / call("ts_delay", price, periods=periods) - 1


@dataclass(frozen=True)
class Factor:
    name: str
    expression: Expr
    description: str
    notes: tuple = ()  # 原文问题的标记，元素为 (标记, 疑点编号)，见 docs/factors/CONVENTIONS.md

    @property
    def fields(self):
        return self.expression.fields


class Registry:
    """名称唯一的显式登记表，不允许静默覆盖。"""

    def __init__(self):
        self._factors = {}

    def register(self, name, expression, description="", notes=()):
        if name in self._factors:
            raise ValueError(f"因子重名：{name}")
        if not isinstance(name, str) or not name or not isinstance(expression, Expr):
            raise ValueError("需要非空名称和 Expr 表达式")
        if isinstance(notes, str) or any(isinstance(n, str) for n in notes):
            raise ValueError("notes 的每一项须是 (标记, 编号) 两个非空字符串")
        notes = tuple(tuple(n) for n in notes)
        if any(
            len(n) != 2 or not all(isinstance(v, str) and v for v in n) for n in notes
        ):
            raise ValueError("notes 的每一项须是 (标记, 编号) 两个非空字符串")
        factor = Factor(name, expression, description, notes)
        self._factors[name] = factor
        return factor

    def __getitem__(self, name):
        return self._factors[name]

    def __iter__(self):
        return iter(self._factors.values())

    def names(self):
        """按登记顺序列出名字。"""
        return tuple(self._factors)

    def with_note(self, tag):
        """返回带某个标记（如「两读」）的因子。"""
        return tuple(
            f for f in self._factors.values() if any(n[0] == tag for n in f.notes)
        )

    def using_field(self, name):
        """返回依赖指定字段的因子。"""
        return tuple(f for f in self._factors.values() if name in f.fields)

    def __len__(self):
        return len(self._factors)


def default_registry(*, min_periods=None):
    """滚动窗口默认至少八成有效样本；可按因子名覆盖最小样本数。"""
    overrides = {} if min_periods is None else dict(min_periods)

    def rolling(op, expression, window, name, **params):
        return call(
            op,
            expression,
            window=window,
            min_periods=overrides.get(name, math.ceil(0.8 * window)),
            **params,
        )

    registry = Registry()
    registry.register(
        "log_market_cap", call("log", field("total_market_cap")), "总市值的自然对数"
    )
    for window in (5, 20):
        registry.register(
            f"reversal_{window}",
            -returns(window),
            f"过去 {window} 个交易日收益的相反数",
        )
    price = adjusted_close()
    registry.register(
        "momentum_12_1",
        call("ts_delay", price, periods=21) / call("ts_delay", price, periods=252) - 1,
        "252 个交易日动量，跳过最近 21 日；需 253 行价格",
    )
    daily = returns()
    for window in (20, 250):
        registry.register(
            f"volatility_{window}",
            rolling("ts_std", daily, window, f"volatility_{window}", ddof=1),
            f"{window} 日日收益样本标准差，不年化",
        )
    registry.register(
        "turnover_20",
        rolling("ts_mean", field("turnover"), 20, "turnover_20"),
        "20 日平均换手率，以比例表示",
    )
    registry.register(
        "amihud_20",
        rolling("ts_mean", call("abs", daily) / field("amount"), 20, "amihud_20"),
        "20 日平均绝对收益/元成交额，不乘缩放系数",
    )
    for name, source in [("ep", "pe_ttm"), ("bp", "pb"), ("sp", "ps_ttm")]:
        registry.register(
            name,
            1 / field(source),
            (
                "1/pe_ttm；数据源对亏损公司不给市盈率，亏损股为缺失；保留负值的盈利收益率要等时点财务数据"
                if name == "ep"
                else f"{source} 的倒数；零分母置缺失，负值保留"
            ),
        )
    registry.register(
        "dividend_yield",
        field("dividend_yield_ttm"),
        "数据层滚动股息率，已为比例；无分红可能为缺失，不擅自填零",
    )
    registry.register(
        "max_return_20",
        rolling("ts_max", daily, 20, "max_return_20"),
        "20 日内最大单日收益",
    )
    registry.register(
        "amount_20",
        rolling("ts_mean", field("amount"), 20, "amount_20"),
        "20 日平均成交额，单位元",
    )
    unknown = set(overrides) - {
        "volatility_20",
        "volatility_250",
        "turnover_20",
        "amihud_20",
        "max_return_20",
        "amount_20",
    }
    if unknown:
        raise ValueError(f"无可覆盖滚动窗口的因子：{sorted(unknown)}")
    return registry
