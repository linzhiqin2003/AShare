"""因子表达式、取数协议、登记与缓存。因子值不做交易移位。"""

from .engine import Evaluator, PanelSource
from .expression import Expr, call, constant, field, neutralize
from .library import (
    FIELDS,
    Factor,
    Registry,
    adjusted_close,
    default_registry,
    returns,
)

__all__ = [
    "Expr",
    "neutralize",
    "call",
    "constant",
    "field",
    "Evaluator",
    "PanelSource",
    "Factor",
    "Registry",
    "FIELDS",
    "adjusted_close",
    "returns",
    "default_registry",
]

from . import expression as _expression

for _name in _expression.ELEMENTWISE:
    globals()[_name] = getattr(_expression, _name)
    __all__.append(_name)

# 单因子评估与多重检验（因子挖掘线）；搜索器在子包 ashare.factor.mining
from . import stats  # noqa: E402
from .evaluate import FactorEvaluator, FactorReport, evaluate_factor  # noqa: E402

__all__ += ["stats", "FactorEvaluator", "FactorReport", "evaluate_factor"]
