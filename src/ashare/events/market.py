"""从数据层取事件研究与回测要用的数组，并建 ``panelbt`` 行情、费用和账户。

TODO(合入主干后替换)：主干已有 ``ashare.backtest.market_from_panel``。本 worktree 切出时
还没有这个包，下面的 :func:`market_from_panel` 是照它的做法写的最小私有副本（只支持开盘价
成交、用面板的 ``can_buy_open``/``can_sell_open``，板块取证券表、空缺时按代码推断，费用用
``panelbt.markets.ASHARE``）。合入后删掉副本，改成
``from ashare.backtest import market_from_panel``，调用处（``load``、``examples``）不变。

最小例子::

    import ashare.data as ad
    from ashare.events.market import load

    m = load(ad.open(), start="2020-01-02")
    m.returns.shape, m.market_return.shape, m.backtest.market.shape
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import polars as pl

BASE = (
    "open",
    "close",
    "volume",
    "split",
    "cash_dividend",
    "delist",
    "can_buy_open",
    "can_sell_open",
)
STUDY = ("adj_factor", "has_quote", "is_listed", "total_market_cap", "industry")
MARKET_INDEX = "000985.CSI"  # 中证全指
_CODE = re.compile(r"^[A-Z]?(\d{6})\.(SH|SZ|BJ)$")


def iso_dates(dates):
    days = np.asarray(dates).astype(str)
    return np.asarray([f"{d[:4]}-{d[4:6]}-{d[6:8]}" for d in days], "datetime64[D]")


def _boards(data, symbols):
    from panelbt.markets import Board, infer_board

    names = {
        "主板": Board.MAIN,
        "创业板": Board.CHINEXT,
        "科创板": Board.STAR,
        "北交所": Board.BSE,
    }
    sec = data.securities().select("symbol", "board").collect()
    table = dict(zip(sec["symbol"].to_list(), sec["board"].to_list(), strict=True))
    out = np.empty(len(symbols), dtype=np.int64)
    for j, symbol in enumerate(str(s) for s in symbols):
        name = table.get(symbol)
        if name in names:
            out[j] = int(names[name])
            continue
        match = _CODE.match(symbol)
        if match is None:
            raise ValueError(f"认不出 {symbol} 的板块")
        out[j] = int(infer_board([match.group(1)])[0])
    return out


@dataclass(frozen=True)
class BacktestMarket:
    """与 ``ashare.backtest.BacktestMarket`` 同形状：行情、日期、证券、板块、面板。"""

    market: object
    dates: np.ndarray
    symbols: np.ndarray
    board: np.ndarray
    panel: object = field(repr=False)

    def config(self, **overrides):
        from panelbt.markets import ASHARE

        return ASHARE.config(board=self.board, **overrides)

    def costs(self, **overrides):
        """按面板日期展开 A 股费用：印花税按历史税率，过户费默认 1e-5 双边，佣金、滑点用
        ``panelbt.Costs`` 默认值；``overrides`` 覆盖任意一项。

        ``ASHARE`` 档案从 2008-09-19 开始、2022-04-29 前要求显式给过户费，直接调用会在更早的
        日期上报错；2008-09-19 前的卖出印花税在这里补（2008-04-24 前 0.3%，之后 0.1%）。
        """
        import panelbt
        from panelbt.markets import ASHARE

        days = iso_dates(self.dates)
        tax = np.where(days < np.datetime64("2008-04-24"), 0.003, 0.001)
        late = days >= np.datetime64("2008-09-19")
        if late.any():
            tax[late] = ASHARE.fees.expand(days[late])["tax_sell"]
        values = {
            "tax_sell": tax,
            "fee_rate_buy": 1e-5,
            "fee_rate_sell": 1e-5,
        } | overrides
        return panelbt.Costs(**values)


def market_from_panel(data, start=None, end=None, *, fields=(), panel=None):
    """开盘价成交的 A 股回测行情（``ashare.backtest.market_from_panel`` 的私有副本）。"""
    import panelbt

    if panel is None:
        names = list(dict.fromkeys([*BASE, *fields]))
        panel = data.panel(names, start=start, end=end)
    split = np.asarray(panel["split"])
    market = panelbt.Market(
        exec_price=np.asarray(panel["open"]),
        close=np.asarray(panel["close"]),
        can_buy=np.asarray(panel["can_buy_open"]),
        can_sell=np.asarray(panel["can_sell_open"]),
        volume=np.asarray(panel["volume"]),
        split=np.where(np.isnan(split), 1.0, split),
        cash_div=np.nan_to_num(np.asarray(panel["cash_dividend"])),
        delist=np.asarray(panel["delist"]),
    )
    return BacktestMarket(
        market,
        np.asarray(panel.dates),
        np.asarray(panel.symbols),
        _boards(data, panel.symbols),
        panel,
    )


@dataclass
class MarketData:
    """事件研究与回测共用的一份数据。``arrays`` 是面板字段（numpy 数组）。"""

    dates: np.ndarray
    symbols: np.ndarray
    arrays: dict
    returns: np.ndarray
    market_return: np.ndarray
    backtest: BacktestMarket
    version: str

    def __getitem__(self, name):
        return self.arrays[name]


def index_returns(data, code, dates):
    """指数日收益（按收盘价），对齐到 ``dates``；缺失为 NaN。"""
    frame = (
        data.table("index_daily")
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
    arrays = {name: np.asarray(p[name]) for name in (*BASE, *STUDY)}
    dates = np.asarray(p.dates).astype("U8")
    returns = daily_returns(
        arrays["close"], arrays["adj_factor"], arrays["has_quote"], arrays["is_listed"]
    )
    market = index_returns(data, market_index, dates)
    return MarketData(
        dates, np.asarray(p.symbols), arrays, returns, market, bt, p.version
    )
