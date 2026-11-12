"""内置示例因子集 ``example``：演示一套因子怎么挂进来、怎么被核对。

这两条是为演示编的，不属于任何公开因子库。参考实现在 ``example_reference``（pandas 写法，
不经过 ``Expr``），``python -m ashare.factor.formula.verify example`` 可以直接跑。
"""

from ..library import Registry
from . import dsl as d
from .plugins import FactorSet


def example_range_mean():
    """mean((high - low) / (0.5 * open + 0.5 * close), 9)"""
    return d.ts_mean((d.high - d.low) / (0.5 * d.open + 0.5 * d.close), 9)


def example_volume_turn():
    """rank(delta(log(volume), 3)) * sign(delta(close, 3))"""
    return d.rank(d.delta(d.log(d.volume), 3)) * d.sign(d.delta(d.close, 3))


def build():
    registry = Registry()
    for function in (example_range_mean, example_volume_turn):
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
