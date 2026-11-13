"""资产配置：几个到几十个大类资产（ETF、指数、期货连续合约、可转债组合）怎么分配权重。

输出是每个调仓日一行权重，行号就是成交行（第 t 个调仓日只用 t−1 日收盘及以前的数据），
可以直接交给 `panelbt`。协方差复用 `ashare.risk`，均值方差和最小方差复用 `ashare.portfolio`。

最短例子（给一张日收益表，出风险平价权重并回测）::

    import numpy as np
    import ashare.allocation as aa

    returns = np.random.default_rng(0).normal(0.0003, 0.01, (1000, 4))
    res = aa.allocate(returns, aa.RiskParity(), rebalance=21)
    ts = aa.tearsheet(res, returns=returns)
    print(ts)

口径、方法和测试见 `docs/ALLOCATION_NOTES.md`。这不是投资建议，只讨论回测。
"""

from .backtest import Backtest, returns_market, run, tearsheet
from .covariance import Covariance
from .engine import Allocation, allocate, schedule
from .methods import (
    EqualWeight,
    FixedWeights,
    InverseVolatility,
    MeanVariance,
    Method,
    MinVariance,
    RiskParity,
    TargetVolatility,
    TrendFilter,
    Window,
)
from .riskparity import risk_contributions, solve_risk_parity

__all__ = [
    "Allocation",
    "Backtest",
    "Covariance",
    "EqualWeight",
    "FixedWeights",
    "InverseVolatility",
    "MeanVariance",
    "Method",
    "MinVariance",
    "RiskParity",
    "TargetVolatility",
    "TrendFilter",
    "Window",
    "allocate",
    "returns_market",
    "risk_contributions",
    "run",
    "schedule",
    "solve_risk_parity",
    "tearsheet",
]
