"""内置示例因子集 ``example``：演示一套因子怎么挂进来、怎么被核对。

这两条是为演示编的，不属于任何公开因子库。参考实现在 ``example_reference``（pandas 写法，
不经过 ``Expr``），``python -m ashare.factor.formula.verify example`` 可以直接跑。
"""

from ..library import Registry
from . import dsl as d
from .plugins import FactorSet


def example_spread_mean():
    """mean((vwap - open) / (high + low), 6)"""
    return d.ts_mean((d.vwap - d.open) / (d.high + d.low), 6)


def example_amount_peak():
    """rank(ts_max(amount, 4)) * sign(vwap - open)"""
    return d.rank(d.ts_max(d.amount, 4)) * d.sign(d.vwap - d.open)


def build():
    registry = Registry()
    for function in (example_spread_mean, example_amount_peak):
        registry.register(
            function.__name__, function(), function.__doc__, [("示例", "EX-1")]
        )
    return registry


def register(hub):
    hub.add(
        FactorSet(
            "example",
            build,
            reference="ashare.factor.formula.example_reference",
            description="内置示例（两条自编因子）",
        )
    )
