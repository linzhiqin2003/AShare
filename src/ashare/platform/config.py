"""平台可运行的设置白名单及检验门槛。"""

import itertools
import math
import os
import re
from pathlib import Path

DEFAULTS = dict(
    universe="TOP1800",
    trade_at="open",
    trade_limitation=True,
    n_groups=10,
    group_neutralize=None,
    decay=0,
    by="year",
    weights_portfolio=False,
    sub_universes=True,
)
CHOICES = dict(
    universe=("TOP800", "TOP1800", "ALL"),
    trade_at=("open", "close"),
    trade_limitation=(True, False),
    n_groups=(5, 10),
    group_neutralize=(None, "industry", "cap", "cap+industry"),
    by=("year", "quarter"),
    weights_portfolio=(True, False),
    sub_universes=(True, False),
)
THRESHOLDS = dict(
    sharpe=1.0,
    fitness=0.5,
    alpha=0.05,
    validation_ratio=0.5,
    sub_sharpe=0.5,
    correlation=0.7,
    turnover_min=0.01,
    turnover_max=0.7,
    coverage=0.95,
    exposure=0.15,
    positive_years=0.6,
)


def expand_settings(settings, expression_count=1):
    if not isinstance(settings, dict) or set(settings) - (
        set(DEFAULTS) | {"benchmark"}
    ):
        raise ValueError("设置含不支持的项目")
    benchmark = settings.get("benchmark")
    if benchmark is not None and (
        not isinstance(benchmark, str)
        or not re.fullmatch(r"[A-Za-z0-9._-]{1,32}", benchmark)
    ):
        raise ValueError("基准代码不正确")
    options = {}
    for key, default in DEFAULTS.items():
        value = settings.get(key, default)
        values = value if isinstance(value, list) else [value]
        if not 1 <= len(values) <= 8:
            raise ValueError("设置选项须有 1–8 项")
        for v in values:
            if key == "decay":
                valid = type(v) is int and 0 <= v <= 2520
            else:
                valid = any(type(v) is type(c) and v == c for c in CHOICES[key])
            if not valid:
                raise ValueError(f"设置 {key} 的值不受支持")
        options[key] = list(dict.fromkeys(values))
    if expression_count * math.prod(map(len, options.values())) > 8:
        raise ValueError("一个 tab 最多运行 8 组，请减少参数组合")
    return [
        dict(zip(options, row, strict=True))
        | ({"benchmark": benchmark} if benchmark else {})
        for row in itertools.product(*options.values())
    ]


def platform_dir():
    return Path(
        os.environ.get(
            "ASHARE_PLATFORM_DIR",
            str(Path(os.environ.get("ASHARE_DATA_DIR", "data")) / "platform"),
        )
    )


def max_workers():
    return max(
        1,
        min(
            32,
            int(
                os.environ.get(
                    "ASHARE_PLATFORM_WORKERS",
                    min(4, max(1, (os.cpu_count() or 2) // 2)),
                )
            ),
        ),
    )
