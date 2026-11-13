"""从数据层取事件研究与回测要用的数组。

回测行情、费用和账户规则直接用 ``ashare.backtest.market_from_panel``（开盘价成交、面板的
开盘可买可卖掩码、按板块的整手、``panelbt`` 的 A 股历史费率档案，见 ``docs/ASHARE_FEES.md``）；
这里只在同一份面板上多读事件研究要用的字段，并算复权收益和市场指数收益。

最小例子::

    import ashare.data as ad
    from ashare.events.market import load

    m = load(ad.open(), start="2020-01-02")
    m.returns.shape, m.market_return.shape, m.backtest.market.shape
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from ashare.backtest import market_from_panel

STUDY = (
    "open",
    "adj_factor",
    "has_quote",
    "is_listed",
    "total_market_cap",
    "industry",
    "can_buy_open",
    "can_sell_open",
)
MARKET_INDEX = "000985.CSI"  # 中证全指（价格指数，不含分红）


@dataclass
class MarketData:
    """事件研究与回测共用的一份数据。``arrays`` 是面板字段（numpy 数组），
    ``backtest`` 是 ``ashare.backtest.BacktestMarket``。"""

    dates: np.ndarray
    symbols: np.ndarray
    arrays: dict
    returns: np.ndarray
    market_return: np.ndarray
    backtest: object
    version: str

    def __getitem__(self, name):
        return self.arrays[name]


def index_returns(data, code, dates):
    """指数日收益（按收盘价），对齐到 ``dates``；缺失为 NaN。"""
    frame = (
        data.table("index_daily", symbols=[code])
        .filter(pl.col("symbol") == code)
        .select("date", "close")
        .collect()
        .unique("date")
        .sort("date")
    )
    close = dict(zip(frame["date"].to_list(), frame["close"].to_list(), strict=True))
    level = np.array([close.get(str(d), np.nan) for d in dates], dtype=np.float64)
    out = np.full(len(dates), np.nan)
    out[1:] = level[1:] / level[:-1] - 1.0
    return out


def load(data, start=None, end=None, market_index=MARKET_INDEX) -> MarketData:
    """一次读面板：回测行情、复权收益、市场指数收益和分组要用的字段。"""
    from .study import daily_returns

    bt = market_from_panel(data, start, end, fields=STUDY)
    p = bt.panel
    names = ("close", "volume", "split", "cash_dividend", "delist", *STUDY)
    arrays = {name: np.asarray(p[name]) for name in names}
    dates = np.asarray(p.dates).astype("U8")
    returns = daily_returns(
        arrays["close"], arrays["adj_factor"], arrays["has_quote"], arrays["is_listed"]
    )
    market = index_returns(data, market_index, dates)
    return MarketData(
        dates, np.asarray(p.symbols), arrays, returns, market, bt, p.version
    )
