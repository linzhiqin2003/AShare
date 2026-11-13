"""标签（实现在 `ashare.labels`，不依赖 LightGBM；这里保留原名，接口与行为不变）。"""

from ..labels import (  # noqa: F401
    EXEC_PRICES,
    MASKS,
    PRESETS,
    RELATIVE,
    RULES,
    SELL_BLOCKED,
    Label,
    LabelValues,
    _has_all,
    compute,
    label,
    tradable,
)
