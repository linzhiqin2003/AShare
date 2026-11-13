"""分组回测：全市场因子 → 股票池权重 → 分组与多空组合 → 绩效、风险暴露和图。"""

from .api import FIELDS, GroupBacktest, run, universe_mask
from .portfolio import GroupingError
from .weights import vector_neut

__all__ = [
    "FIELDS",
    "GroupBacktest",
    "GroupingError",
    "run",
    "universe_mask",
    "vector_neut",
]
