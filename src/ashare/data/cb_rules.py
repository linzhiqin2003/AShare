"""可转债（含可交换债）的回测规则：费用、交易单位、涨跌幅，以及「从可转债面板建 `panelbt.Market`」。

依据与未核实处（详见 docs/FUND_CB_DATA.md「回测规则」）：

- 印花税：债券交易不征。**已核实**。过户费：不收。按 0 处理。
- 佣金与经手费：券商协议，常见万 0.5 到万 2（经手费沪市成交额的 0.001%、深市 0.004‰ 量级，通常并在
  佣金里）；默认给万 1、无最低收费，请按实际给。**量级未逐条核实**。
- 交易单位：沪深两市都是 1 手 = 10 张（面值 1000 元），买入以 10 张为单位。**已核实**（沪市 2022 年
  起由「1 手 = 10 张」报价改为按张报价，最小单位仍为 10 张）。
- 交收：T+0（当日买入当日可卖）。日频回测里只是元数据。
- 涨跌幅（《可转换公司债券交易实施细则》，2022-08-01 起）：上市首日 +57.3% / −43.3%（相对发行价
  100 元），之后每天 ±20%（相对前收盘）。此前没有涨跌幅限制，只有盘中 20% / 30% 临时停牌——
  日线回测里按「无涨跌停」处理。**新规对存量债的生效日未逐只核实**（统一按 2022-08-01）。
- 强赎与到期：最后交易日后未卖出的按强赎价（面值 + 当期应计利息）或到期赎回价兑付；`panelbt` 的退市
  按「上一有效收盘 × delist_recovery」了结，会把强赎价远低于市价（常见 1.3–1.6 倍差距）的情形算错。
  所以持有可转债的策略应在 `call_announced` 为真后、最后交易日前卖出（实测见 docs 的校验结论）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from panelbt.markets import FeeSchedule, MarketProfile

TICK = 0.001
LOT = 10.0
NEW_RULES = "20220801"

CB = MarketProfile(
    "China convertible bonds",
    "CNY",
    252,
    FeeSchedule((("1992-01-01", {}),)),
    lot_rule="direct",
    price_limits=True,
    sell_lock=0,
    regulatory_notes=(
        "no stamp duty or transfer fee on bonds",
        "T+0",
        "price limits from 2022-08-01",
    ),
)


def price_limits(dates, previous_close, first_day):
    """涨跌停价（T×N）。2022-08-01 前全为 NaN（无涨跌停）；之后首日按发行价 100 的 +57.3%/−43.3%，
    其余按前收盘 ±20%。first_day：T×N 布尔，上市首日。"""
    pc = np.asarray(previous_close, dtype=float)
    new = (np.asarray(dates) >= NEW_RULES)[:, None]
    up = np.where(first_day, 157.3, np.round(pc * 1.2 / TICK) * TICK)
    down = np.where(first_day, 56.7, np.round(pc * 0.8 / TICK) * TICK)
    return np.where(new, up, np.nan), np.where(new, down, np.nan)


@dataclass(frozen=True)
class CBMarket:
    market: object
    dates: np.ndarray
    symbols: np.ndarray
    panel: object = field(repr=False)

    @property
    def shape(self):
        return self.market.shape

    def config(self, **overrides):
        """每只 10 张起、按 10 张递增。"""
        return CB.config(lot_sizes=np.full(len(self.symbols), LOT), **overrides)

    def costs(
        self, *, commission_rate=1e-4, commission_min=0.0, slippage=5e-4, **overrides
    ):
        """免印花税、免过户费；佣金（含经手费）默认万 1、无最低收费，滑点默认万 5。"""
        days = np.asarray(
            [f"{d[:4]}-{d[4:6]}-{d[6:8]}" for d in self.dates], "datetime64[D]"
        )
        return CB.costs(
            days,
            commission_rate=commission_rate,
            commission_min=commission_min,
            slippage=slippage,
            **overrides,
        )


FIELDS = (
    "open",
    "high",
    "low",
    "close",
    "previous_close",
    "volume",
    "has_quote",
    "is_listed",
    "delist",
)


def cb_market(funds, symbols=None, start=None, end=None, *, exec="open", validate=True):
    """从可转债面板建回测行情（`funds = ashare.data.open(root).funds()`）。

    可买卖：有成交（`has_quote`）、在上市期内、成交价不在涨跌停（2022-08-01 起）。没有拆分和现金分红
    （票息未计入，见 docs）。退市行用面板的 `delist`；强赎前请自行卖出。
    """
    import panelbt as bt
    from panelbt.markets.ashare import tradability

    if exec not in ("open", "close"):
        raise ValueError("exec: open 或 close")
    p = funds.panel("cb", list(FIELDS), start=start, end=end, symbols=symbols)
    dates = np.asarray(p.dates)
    listed = np.asarray(p["is_listed"])
    first_day = listed & ~np.vstack([np.zeros((1, listed.shape[1]), bool), listed[:-1]])
    # 区间第一行不能凭「前一行不在上市期」判断首日：看完整面板的上市日。
    securities = funds.securities("cb").collect()
    listing = dict(
        zip(
            securities["symbol"].to_list(),
            securities["listing_date"].to_list(),
            strict=True,
        )
    )
    for j, s in enumerate(p.symbols):
        first_day[0, j] = (
            listing.get(str(s)) is not None and dates[0] <= listing[str(s)] <= dates[0]
        )
    up, down = price_limits(dates, p["previous_close"], first_day)
    price = np.asarray(p[exec], dtype=float)
    ok = np.asarray(p["has_quote"]) & listed
    can_buy, can_sell = tradability(price, p["high"], p["low"], up, down, ~ok)
    market = bt.Market(
        np.where(ok, price, np.nan),
        np.where(np.isfinite(p["close"]), p["close"], np.nan),
        can_buy,
        can_sell,
        volume=np.nan_to_num(np.asarray(p["volume"], dtype=float)),
        delist=np.asarray(p["delist"], dtype=bool),
        validate=validate,
    )
    return CBMarket(market, dates, np.asarray(p.symbols), p)


__all__ = ["CB", "CBMarket", "cb_market", "price_limits"]
