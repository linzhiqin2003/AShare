"""公式型因子的机制：翻译用语言、Qlib 表达式转换、取数、核对工具、外部因子包的加载。

本仓库只有机制和一个内置示例集；成套的因子（公式、参考实现、约定文件）放在外部因子包里，
经 Python 入口点组 ``ashare.factors`` 被发现（``plugins``，见 ``docs/DESIGN.md`` 6.2.1）。

- ``dsl``：一个原文函数一个构造函数的翻译用语言；``ops``：用现有算子精确组合出的算子；
- ``qlib``：Qlib 表达式字符串到 ``Expr`` 的确定性转换；
- ``inputs``：取数（基准指数、行业组号）和参考实现的输入；
- ``verify``：候选表达式对独立参考实现的逐格核对（``python -m ashare.factor.formula.verify``）；
- ``registry_from_module``：从「一个函数一条因子」的模块构造注册表，挂元数据标记。
"""

from . import ops_extra
from .inputs import PublicSource, load_inputs, reference_frames
from .known_exceptions import KnownException
from .plugins import (
    ENTRY_POINT_GROUP,
    FactorPluginError,
    FactorSet,
    Hub,
    discover,
    factor_set,
    load_registry,
)
from .registry import registry_from_module

__all__ = [
    "ENTRY_POINT_GROUP",
    "FactorPluginError",
    "FactorSet",
    "Hub",
    "KnownException",
    "PublicSource",
    "discover",
    "factor_set",
    "load_inputs",
    "load_registry",
    "ops_extra",
    "reference_frames",
    "registry_from_module",
]
