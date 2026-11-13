"""把研究信号和数据层面板接到 `panelbt`：信号与成交的错位、A 股行情与账户规则。"""

from .futures import book_from_table, futures_portfolio
from .market import (
    BacktestMarket,
    boards,
    iso_dates,
    market_from_panel,
    market_inputs,
    required_fields,
)
from .timing import execution_days, execution_targets

__all__ = [
    "BacktestMarket",
    "boards",
    "book_from_table",
    "execution_days",
    "execution_targets",
    "futures_portfolio",
    "iso_dates",
    "market_from_panel",
    "market_inputs",
    "required_fields",
]
