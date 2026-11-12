"""回测之后的分析：收益归因、风险归因、Brinson 行业归因、容量与成本。

```python
import panelbt
import ashare.data
import ashare.analysis as an
from ashare.backtest import execution_targets, market_from_panel

data = ashare.data.open()
panel = an.LazyPanel(data, start="2013-01-04")
signal = ...                                   # T×N 信号日目标权重（整行 NaN = 不调仓）
targets = execution_targets(signal)            # 移到下一交易日成交
bm = market_from_panel(data, start="2013-01-04")
result = panelbt.run(bm.market, targets, bm.costs(), bm.config(record_trades=True))

held = result.positions != 0
columns = an.held_columns(held, panel["weight_csi300"], symbols=panel.symbols)  # 组合 + 基准成分
history = an.FactorHistory.build(panel, columns=columns, cache=".cache/analysis-risk")
report = an.attribute(result, panel, history, benchmark="000300.SH")
risk = an.risk_series(result, panel, history, benchmark="000300.SH")
sectors = an.brinson(result, panel, benchmark="000300.SH")
cap = an.capacity(targets, panel, aum=[1e7, 1e8, 1e9, 1e10], benchmark="000300.SH")
costs = an.cost_breakdown(result, panel.dates)
```

`result` 也可以换成收盘后权重 T×N（`weights[t]` 赚 t+1 日收益），此时没有交易和成本项。
所有结果都是 `Table`（数组 + 行名 + 列名）或由它组成的数据对象，可 `.to_dict()`；
各结果对象的 `to_dict()` 是否带诊断见各自文档。时间约定见 `ashare.analysis.inputs`。
行情、账户规则、费用表、信号错位都直接用主干（`ashare.backtest`、`ashare.risk`），
本包只在 `adapters` 里做「只有面板、没有数据句柄」时的转发。
"""

from .adapters import risk_source
from .attribution import Attribution, attribute, decompose
from .brinson import Brinson, brinson, brinson_effects
from .capacity import (
    Capacity,
    capacity,
    impact_coefficients,
    simulate,
    simulate_many,
    volatility,
)
from .costs import cost_breakdown
from .history import FactorHistory, factor_groups, held_columns
from .inputs import (
    LazyPanel,
    benchmark_held,
    cash_returns,
    held_from_weights,
    reference_close,
    reference_prices,
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
    "attribute",
    "benchmark_held",
    "bias_statistics",
    "brinson",
    "brinson_effects",
    "capacity",
    "cash_returns",
    "cost_breakdown",
    "cumulative_active",
    "decompose",
    "decompose_risk",
    "factor_groups",
    "format_table",
    "held_columns",
    "held_from_weights",
    "impact_coefficients",
    "link",
    "reference_close",
    "reference_prices",
    "result_weights",
    "risk_series",
    "risk_source",
    "simulate",
    "simulate_many",
    "total_returns",
    "volatility",
]
