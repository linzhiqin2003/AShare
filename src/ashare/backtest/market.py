"""从数据层面板建 `panelbt` 行情、账户规则和费用：一步到位，不再手写样板。

面板与引擎的口径差异都在这里换算一次：
- `split`、`cash_dividend` 在面板里「缺失 = 当日无事件」，引擎要 1 和 0；
- 可买可卖：开盘成交直接用面板算好的 `can_buy_open`/`can_sell_open`（旧面板没有时
  现算，口径相同）；收盘或均价成交按同一条规则（`panelbt.markets.ashare.tradability`）
  用对应价格现算；停牌或当天没有源日线都不可买卖；
- 板块与整手：取证券表的板块列，缺失时按代码前缀推断，都认不出就报错列出证券；
- 退市：面板 `delist` 原样交给引擎。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import panelbt
from panelbt.markets import ASHARE, Board, infer_board
from panelbt.markets.ashare import tradability

EXEC_PRICES = {"open": "open", "close": "close", "vwap": "vwap"}
BASE = ("close", "volume", "split", "cash_dividend", "delist")
RULES = ("high", "low", "upper_limit", "lower_limit", "is_suspended", "has_quote")
MASKS = ("can_buy_open", "can_sell_open")
BOARD_NAMES = {
    "主板": Board.MAIN,
    "创业板": Board.CHINEXT,
    "科创板": Board.STAR,
    "北交所": Board.BSE,
}
_CODE = re.compile(r"^[A-Z]?(\d{6})\.(SH|SZ|BJ)$")


def _exec(exec):
    if exec not in EXEC_PRICES:
        raise ValueError(f"exec: 只能是 {tuple(EXEC_PRICES)}")
    return EXEC_PRICES[exec]


def required_fields(exec="open", *, masks=True):
    """建行情需要的面板字段；masks=True 表示开盘成交时直接用面板的可买可卖掩码。"""
    price = _exec(exec)
    if exec == "open" and masks:
        return (price, *BASE, *MASKS)
    return tuple(dict.fromkeys((price, *BASE, *RULES)))


def _has(panel, name):
    try:
        panel[name]
    except KeyError:
        return False
    return True


def market_inputs(panel, exec="open"):
    """面板（`ashare.data.Panel` 或任何按字段名取 T×N 数组的对象）→ `panelbt.Market` 的关键字参数。"""
    price = _exec(exec)
    if exec == "open" and all(_has(panel, m) for m in MASKS):
        can_buy, can_sell = (np.asarray(panel[m]) for m in MASKS)
    else:
        p = {name: np.asarray(panel[name]) for name in RULES}
        can_buy, can_sell = tradability(
            np.asarray(panel[price]),
            p["high"],
            p["low"],
            p["upper_limit"],
            p["lower_limit"],
            p["is_suspended"] | ~p["has_quote"],
        )
    split = np.asarray(panel["split"])
    return {
        "exec_price": np.asarray(panel[price]),
        "close": np.asarray(panel["close"]),
        "can_buy": can_buy,
        "can_sell": can_sell,
        "volume": np.asarray(panel["volume"]),
        "split": np.where(np.isnan(split), 1.0, split),
        "cash_div": np.nan_to_num(np.asarray(panel["cash_dividend"])),
        "delist": np.asarray(panel["delist"]),
    }


def boards(securities, symbols):
    """每只证券的 `Board`。securities：带 symbol、board 两列的表（`data.securities()`）。

    板块列优先；板块列为空时按代码前缀推断（如已退市的 `T600018.SH`）；板块名不认识或
    代码也推断不出时报错并列出证券，不静默归到主板。
    """
    if hasattr(securities, "collect"):
        securities = securities.collect()
    table = dict(
        zip(
            securities["symbol"].to_list(),
            securities["board"].to_list(),
            strict=True,
        )
    )
    symbols = [str(s) for s in symbols]
    missing = [s for s in symbols if s not in table]
    if missing:
        raise KeyError(f"证券表里没有 {missing[:10]}")
    out = np.empty(len(symbols), dtype=np.int64)
    unknown, guess = [], []
    for j, symbol in enumerate(symbols):
        name = table[symbol]
        if name is None or name == "":
            guess.append(j)
        elif name in BOARD_NAMES:
            out[j] = int(BOARD_NAMES[name])
        else:
            unknown.append(f"{symbol}（板块 {name!r}）")
    for j in guess:
        match = _CODE.match(symbols[j])
        try:
            if match is None:
                raise ValueError
            out[j] = int(infer_board([match.group(1)])[0])
        except ValueError:
            unknown.append(f"{symbols[j]}（板块为空，代码前缀也认不出）")
    if unknown:
        raise ValueError(
            "认不出板块，无法确定申报单位："
            + "；".join(unknown[:10])
            + (f" 等 {len(unknown)} 只" if len(unknown) > 10 else "")
            + "。请用 lot_sizes= 或 board= 显式给出"
        )
    return out


def iso_dates(dates):
    """面板的 YYYYMMDD 字符串 → datetime64[D]。"""
    days = np.asarray(dates).astype(str)
    return np.asarray([f"{d[:4]}-{d[4:6]}-{d[6:8]}" for d in days], "datetime64[D]")


@dataclass(frozen=True)
class BacktestMarket:
    """一份面板对应的行情、板块和日期；`config()`、`costs()` 补齐账户和费用。"""

    market: panelbt.Market
    dates: np.ndarray
    symbols: np.ndarray
    board: np.ndarray
    panel: object = field(repr=False)

    @property
    def shape(self):
        return self.market.shape

    def config(self, **overrides):
        """A 股整手规则（按板块）加上 `panelbt.Config` 的其他参数。"""
        return ASHARE.config(board=self.board, **overrides)

    def costs(self, **overrides):
        """按面板日期展开 A 股历史印花税和过户费；佣金、滑点等由调用方给。"""
        return ASHARE.costs(iso_dates(self.dates), **overrides)


def market_from_panel(
    data,
    start=None,
    end=None,
    *,
    exec="open",
    symbols=None,
    fields=(),
    panel=None,
    validate=True,
):
    """一步建好 A 股回测行情。

    data：`ashare.data.open()` 的句柄。start、end、symbols 同 `data.panel`。
    exec：成交价，"open"（默认）、"close" 或 "vwap"。
    fields：顺便一起读出的其他字段（算信号用），之后从 `.panel` 取。
    panel：已经读好的面板（含所需字段）；给了就不再读，start/end/symbols 必须为空。
    validate：交给 `panelbt.Market`，确认数据无误后可传 False 跳过逐格检查。
    """
    _exec(exec)
    if panel is None:
        extra = [fields] if isinstance(fields, str) else list(fields)

        def load(base):
            names = list(dict.fromkeys([*base, *extra]))
            return data.panel(names, start=start, end=end, symbols=symbols)

        try:
            panel = load(required_fields(exec))
        except FileNotFoundError:
            # 旧面板没有开盘可买可卖掩码：读规则字段现算。
            if exec != "open":
                raise
            panel = load(required_fields(exec, masks=False))
    elif start is not None or end is not None or symbols is not None:
        raise ValueError("给了 panel 时 start/end/symbols 不起作用，请在读面板时指定")
    inputs = market_inputs(panel, exec)
    market = panelbt.Market(**inputs, validate=validate)
    return BacktestMarket(
        market,
        np.asarray(panel.dates),
        np.asarray(panel.symbols),
        boards(data.securities(), panel.symbols),
        panel,
    )
