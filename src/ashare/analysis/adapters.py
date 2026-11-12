"""与主干的接口对应：本分支自带的三个适配函数，集中在这里，合入后整体替换。

本分支从主干切出时还没有下面这些入口，所以先自带等价版本；口径照主干写，合入后由
负责人把本模块换成对主干的调用：

| 本模块 | 主干 | 说明 |
|---|---|---|
| `RISK_FIELDS` | `ashare.risk.panel.PANEL_FIELDS` | 字段映射完全相同 |
| `risk_source(panel, config)` | `ashare.risk.risk_inputs(panel, config=config)`（或 `ashare.risk.from_panel(data, panel=panel)`） | 行业代码按升序编 0..K-1、缺失 -1，与主干 `industry_codes` 相同；主干另给行业因子起中文名（仍以 `industry_` 开头） |
| `market_from_panel(panel, exec_price)` | `ashare.backtest.market_from_panel(panel, exec=exec_price)` | 缺失送转记 1、派现记 0；开盘成交用面板的可买可卖掩码 |
| `ashare_config(symbols, **kw)` | `ashare.backtest` 的账户规则 | 本模块按代码前缀判断板块（认不出的按主板）；主干用证券表的板块列 |
| `INDEX_FIELDS` | `ashare.data.catalog.INDEXES` 的反向映射 | 指数代码 → 面板权重字段 |
"""

from __future__ import annotations

import numpy as np

# 数据层字段名 → ashare.risk 的字段名（ashare.risk.data.FIELDS）。
RISK_FIELDS = {
    "close": "close",
    "adj_factor": "adj_factor",
    "total_mv": "total_market_cap",
    "circ_mv": "float_market_cap",
    "turnover_rate": "turnover",
    "pb": "pb",
    "pe_ttm": "pe_ttm",
    "is_suspended": "is_suspended",
    "is_listed": "is_listed",
}

# 指数代码 → 面板的权重字段（与 ashare.data.catalog.INDEXES 同一份对应关系）。
INDEX_FIELDS = {
    "000300.SH": "weight_csi300",
    "000905.SH": "weight_csi500",
    "000852.SH": "weight_csi1000",
    "932000.CSI": "weight_csi2000",
    "000985.CSI": "weight_csi_all",
}


def risk_source(panel, config=None):
    """把数据层面板接到 `ashare.risk` 的读取协议，返回 (source, config)。

    ```python
    source, config = an.risk_source(panel)
    models = ashare.risk.model_series(source, config=config, cache=".cache/risk")
    ```
    申万一级行业代码按升序映射到 0..K-1（0 或 NaN 为缺失，记 -1）；无风险收益取 0。
    编码规则与主干 `ashare.risk.panel.industry_codes` 相同（主干另给行业因子起中文名，
    因子名仍以 `industry_` 开头，`factor_groups` 的分类不受影响）。合入后改用
    `ashare.risk.risk_inputs(panel, config=config)`。
    """
    from ashare.risk import FundamentalConfig, MemoryPanel

    fields = {}
    for target, name in RISK_FIELDS.items():
        value = np.asarray(panel[name])
        fields[target] = (
            value.astype(bool)
            if target.startswith("is_")
            else np.asarray(value, dtype=np.float64)
        )
    raw = np.nan_to_num(np.asarray(panel["industry"], dtype=float)).astype(np.int64)
    codes = np.unique(raw[raw > 0])
    lookup = np.full(int(codes.max()) + 1 if len(codes) else 1, -1, dtype=np.int64)
    lookup[codes] = np.arange(len(codes))
    fields["industry"] = np.where(raw > 0, lookup[np.maximum(raw, 0)], -1)
    source = MemoryPanel(
        np.asarray(panel.dates),
        tuple(map(str, panel.symbols)),
        str(getattr(panel, "version", "")),
        fields,
    )
    if config is None:
        config = FundamentalConfig(industry_codes=tuple(range(len(codes))))
    return source, config


def market_from_panel(panel, exec_price="open", validate=True):
    """按 A 股规则由面板构建 `panelbt.Market`（缺失的送转记 1、派现记 0，开盘成交用面板的
    can_buy_open/can_sell_open，没有时按 tradability 现算）。

    主干已有同口径的 `ashare.backtest.market_from_panel`；本分支切出时还没有它，合入后
    这里改为直接调用它。"""
    import panelbt
    from panelbt.markets.ashare import tradability

    p = {}
    for name in ("close", "volume", "split", "cash_dividend", "delist"):
        p[name] = np.asarray(panel[name])
    exec_values = np.asarray(panel[exec_price], dtype=float)
    try:
        can_buy = np.asarray(panel["can_buy_open"]) if exec_price == "open" else None
        can_sell = np.asarray(panel["can_sell_open"]) if exec_price == "open" else None
    except KeyError:
        can_buy = can_sell = None
    if can_buy is None:
        can_buy, can_sell = tradability(
            exec_values,
            np.asarray(panel["high"]),
            np.asarray(panel["low"]),
            np.asarray(panel["upper_limit"]),
            np.asarray(panel["lower_limit"]),
            np.asarray(panel["is_suspended"]).astype(bool)
            | ~np.asarray(panel["has_quote"]).astype(bool),
        )
    return panelbt.Market(
        exec_price=exec_values,
        close=np.asarray(p["close"], dtype=float),
        can_buy=np.asarray(can_buy, bool),
        can_sell=np.asarray(can_sell, bool),
        volume=np.asarray(p["volume"], dtype=float),
        split=np.where(np.isnan(p["split"]), 1.0, p["split"]),
        cash_div=np.nan_to_num(p["cash_dividend"]),
        delist=np.asarray(p["delist"], bool),
        validate=validate,
    )


def ashare_config(symbols, **overrides):
    """按板块给出整手规则的 `panelbt.Config`（代码前缀判断板块，与 bench/e2e 同一口径；
    认不出的按主板。主干的 `ashare.backtest` 改用证券表的板块列，合入后改为调用它）。"""
    import re

    import panelbt
    from panelbt.markets.ashare import Board, lot_rules

    boards = []
    for symbol in symbols:
        code = re.sub(r"\D", "", str(symbol))[:6]
        if code.startswith(("300", "301")):
            boards.append(Board.CHINEXT)
        elif code.startswith(("688", "689")):
            boards.append(Board.STAR)
        elif code.startswith(("43", "83", "87", "88", "92")):
            boards.append(Board.BSE)
        else:
            boards.append(Board.MAIN)
    min_qty, step = lot_rules(np.array(boards))
    return panelbt.Config(min_qty=min_qty, qty_step=step, **overrides)
