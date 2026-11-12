"""回测之后的分析：收益归因、风险归因、Brinson 行业归因、容量与成本。

```python
import ashare.data
import ashare.analysis as an

panel = an.LazyPanel(ashare.data.open(), start="2015-01-05")
history = an.FactorHistory.build(panel, columns=an.held_columns(weights, symbols=panel.symbols))
report = an.attribute(weights, panel, history, benchmark="000300.SH")
risk = an.risk_series(weights, panel, history, benchmark="000300.SH")
sectors = an.brinson(weights, panel, benchmark="000300.SH")
cap = an.capacity(targets, panel, aum=[1e7, 1e8, 1e9, 1e10], benchmark="000300.SH")
costs = an.cost_breakdown(result, panel.dates)
```

所有结果都是 `Table`（数组 + 行名 + 列名）或由它组成的数据对象，可 `.to_dict()`、
`.to_frame()`（需要 pandas）。时间约定见 `ashare.analysis.inputs`。
"""

from .adapters import ashare_config, market_from_panel, risk_source
from .attribution import Attribution, attribute, decompose
from .brinson import Brinson, brinson, brinson_effects
from .capacity import (
    Capacity,
    capacity,
    default_costs,
    impact_coefficients,
    simulate,
    volatility,
)
from .costs import cost_breakdown
from .history import FactorHistory, factor_groups, held_columns
from .inputs import (
    LazyPanel,
    benchmark_held,
    held_from_weights,
    result_weights,
    total_returns,
)
from .linking import cumulative_active, link
from .risk import (
    RiskDecomposition,
    RiskSeries,
    bias_statistics,
    decompose_risk,
    risk_series,
)
from .table import Table, format_table

__all__ = [
    "Attribution",
    "Brinson",
    "Capacity",
    "FactorHistory",
    "LazyPanel",
    "RiskDecomposition",
    "RiskSeries",
    "Table",
    "ashare_config",
    "attribute",
    "benchmark_held",
    "bias_statistics",
    "brinson",
    "brinson_effects",
    "capacity",
    "cost_breakdown",
    "cumulative_active",
    "decompose",
    "decompose_risk",
    "default_costs",
    "factor_groups",
    "format_table",
    "held_columns",
    "held_from_weights",
    "impact_coefficients",
    "link",
    "market_from_panel",
    "result_weights",
    "risk_series",
    "risk_source",
    "simulate",
    "total_returns",
    "volatility",
]
