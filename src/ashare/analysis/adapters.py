"""接主干的入口：风险模型输入、回测行情、账户规则、费用表都直接调用主干，本模块只做转发。

| 本模块 | 主干 |
|---|---|
| `INDEX_FIELDS` | 与 `ashare.data.catalog.INDEXES` 同一份对应（指数代码 → 面板权重字段） |
| `risk_source(panel, config)` | `ashare.risk.risk_inputs`；面板带数据句柄（`LazyPanel`）时行业因子用申万名称 |
| `market_from_panel(panel, exec_price)` | `panelbt.Market(**ashare.backtest.market_inputs(panel, exec))` |
| `ashare_config(symbols, securities=None)` | `ashare.backtest.boards` + `ASHARE.config(board=...)` |
| `ashare_costs(dates)` | `ASHARE.costs(ashare.backtest.iso_dates(dates))`：按日期展开的历史印花税、过户费 |

留在这里而不让调用方直接用主干的原因只有一个：分析函数的输入是「一个面板」，没有数据句柄。
`ashare.backtest.market_from_panel(data, panel=...)` 要用 `data.securities()` 取板块，
`ashare.risk.from_panel(data, ...)` 要用 data 取行业名；面板是 `LazyPanel` 时这里把它的
数据句柄传过去，否则板块按代码前缀推断（推断不出时与主干一样报错，不默认主板），行业因子
以申万代码命名（仍以 `industry_` 开头，`factor_groups` 的分组不受影响）。
"""

from __future__ import annotations

import numpy as np

# 指数代码 → 面板的权重字段。数据层的 `catalog` 是内部模块（层边界检查不许包外导入），
# 所以抄一份；`test_index_fields_match_data_catalog` 守着两者一致。
INDEX_FIELDS = {
    "000300.SH": "weight_csi300",
    "000905.SH": "weight_csi500",
    "000852.SH": "weight_csi1000",
    "932000.CSI": "weight_csi2000",
    "000985.CSI": "weight_csi_all",
}


def _data(panel):
    return getattr(panel, "data", None)


def risk_source(panel, config=None):
    """面板 → `ashare.risk` 的 (source, config)，即 `ashare.risk.risk_inputs`。

    ```python
    source, config = an.risk_source(panel)
    models = ashare.risk.model_series(source, config=config, cache=".cache/risk")
    ```
    """
    from ashare.risk.panel import industry_names, risk_inputs

    data = _data(panel)
    names = None if data is None else (lambda codes: industry_names(data, codes))
    return risk_inputs(panel, names=names, config=config)


def market_from_panel(panel, exec_price="open", validate=True):
    """面板 → `panelbt.Market`，口径即主干 `ashare.backtest.market_inputs`。"""
    import panelbt

    from ashare.backtest import market_inputs

    return panelbt.Market(**market_inputs(panel, exec_price), validate=validate)


def ashare_config(symbols, securities=None, **overrides):
    """A 股整手规则的 `panelbt.Config`；板块用主干 `ashare.backtest.boards`。

    securities：证券表（`data.securities()`）；不给时按代码前缀推断，推断不出报错。"""
    import polars as pl
    from panelbt.markets import ASHARE

    from ashare.backtest import boards

    symbols = [str(s) for s in symbols]
    if securities is None:
        securities = pl.DataFrame(
            {"symbol": symbols, "board": [None] * len(symbols)},
            schema={"symbol": pl.String, "board": pl.String},
        )
    return ASHARE.config(board=boards(securities, symbols), **overrides)


def panel_config(panel, **overrides):
    """面板对应的账户规则：带数据句柄时用证券表，否则按代码前缀。"""
    data = _data(panel)
    securities = None if data is None else data.securities()
    return ashare_config(panel.symbols, securities, **overrides)


def ashare_costs(dates, **overrides):
    """面板日期（YYYYMMDD）→ 主干的 A 股费用表 `ASHARE.costs`（佣金、滑点用其默认值）。"""
    from panelbt.markets import ASHARE

    from ashare.backtest import iso_dates

    return ASHARE.costs(iso_dates(np.asarray(dates)), **overrides)
