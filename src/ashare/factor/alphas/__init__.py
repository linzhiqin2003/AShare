"""公开因子库：Qlib Alpha158、WorldQuant Alpha101、国泰君安 191 的公式、参考实现和文档。

机制（翻译用语言、转换、取数、核对）在 ``ashare.factor.formula``；三套作为内置因子集注册
（``plugin.register`` 由 ``formula.plugins`` 直接调用），可以这样用::

    from ashare.factor.formula import load_registry
    registry = load_registry("alpha158")
"""

from .alpha158_formulas import FORMULAS as ALPHA158_FORMULAS
from .registries import alpha101_registry, alpha158_registry, gtja191_registry

__all__ = [
    "ALPHA158_FORMULAS",
    "alpha101_registry",
    "alpha158_registry",
    "gtja191_registry",
]
