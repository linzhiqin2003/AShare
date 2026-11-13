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
- 强赎与到期（`cb_market` 默认就处理，不改 `panelbt`）：
  1. 过了最后交易日仍持有的，按赎回价了结：强赎用公告的赎回价（面值 + 当期应计利息，税前），到期用到期
     赎回价；数据里没有价格时按面值 100，诊断里计数。做法是在最后交易日之后、摘牌之前的行把估值价设为
     赎回价（这几行不能成交），`panelbt` 摘牌时按这个估值价了结；最后交易日与摘牌日相邻时，行情里的
     摘牌行往后挪一行。其他原因摘牌（违约、定向可转债等，没有赎回公告）的不知道兑付多少，仍按最后
     收盘了结（偏乐观），逐只列在诊断里。
  2. 强赎公告可知之后（`call_announced`）不再买入；到期的从清仓日起不再买入。
  3. `CBMarket.targets(weights)` 在最后交易日前 `exit_days` 个交易日（默认 1，0 为最后交易日当天）
     插入一次正常卖出：只对「最后交易日事先可知」的债（已公告实施强赎、到期赎回）做，按正常成交
     （佣金、滑点、交易单位、涨跌停都照常）。插入的那一行把其他持仓调回它们最近一次的目标权重。
  卖出（3）是默认用法，赎回价了结（1）是没卖掉时的兜底；两者的差别是最后几天的价格：转股溢价常常让
  最后收盘远高于赎回价（中位 1.42 倍），持有到摘牌的人只拿到赎回价。
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
    """可转债行情与规则。exit_rows：每列的清仓行（没有为 T）；redemption：每列的赎回价（没有为 NaN）。"""

    market: object
    dates: np.ndarray
    symbols: np.ndarray
    panel: object = field(repr=False)
    exit_rows: np.ndarray = field(default=None, repr=False)
    redemption: np.ndarray = field(default=None, repr=False)
    diagnostics: dict = field(default_factory=dict, repr=False)

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

    def apply_exit(self, weights):
        """稠密目标权重（T×N，全 NaN 行为不调仓）→ 加上清仓后的权重。

        对每个有清仓行的债：清仓行起它的目标一律为 0；若清仓行之前最近一次目标里它的权重大于 0 而清仓行
        本身不是调仓行，就在清仓行插入一行：其余列取最近一次目标权重，这只债为 0。
        """
        w = np.array(weights, dtype=float, copy=True)
        t, n = w.shape
        if (t, n) != self.shape:
            raise ValueError(f"weights 形状应为 {self.shape}")
        exits = self.exit_rows
        for row in sorted({int(r) for r in exits if r < t}):
            cols = np.flatnonzero(exits == row)
            active = np.flatnonzero(~np.all(np.isnan(w[: row + 1]), axis=1))
            if not len(active):
                continue
            last = active[-1]
            if last != row:
                intended = np.nan_to_num(w[last])
                if (intended[cols] > 0).any():
                    w[row] = intended
            later = ~np.all(np.isnan(w[row:]), axis=1)
            block = w[row:, :]
            block[np.ix_(later, cols)] = 0.0
        return w

    def targets(self, weights, *, exit=True, validate=True):
        """`panelbt.Targets`；exit=True（默认）时先 `apply_exit`。"""
        import panelbt as bt

        dense = self.apply_exit(weights) if exit else np.asarray(weights, dtype=float)
        return bt.Targets.from_dense(dense, validate=validate)


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
    "call_announced",
)
CALL_IMPLEMENTED = "公告实施强赎"
MATURITY = "公告到期赎回"


def _quote_end(funds, symbols):
    """完整面板上每只债的最后一个有行情日（YYYYMMDD；从没有行情为 None）。"""
    p = funds.panel("cb", ["has_quote"], symbols=list(symbols))
    has = np.asarray(p["has_quote"])
    dates = np.asarray(p.dates)
    out = []
    for j in range(has.shape[1]):
        q = np.flatnonzero(has[:, j])
        out.append(str(dates[q[-1]]) if len(q) else None)
    return out


def cb_market(
    funds,
    symbols=None,
    start=None,
    end=None,
    *,
    exec="open",
    exit_days=1,
    block_after_call=True,
    validate=True,
):
    """从可转债面板建回测行情（`funds = ashare.data.open(root).funds()`）。

    可买卖：有成交（`has_quote`）、在上市期内、成交价不在涨跌停（2022-08-01 起）；强赎公告可知后不能买
    （block_after_call）。强赎与到期的处理见模块说明：没卖掉的按赎回价了结；`targets()` 默认在最后交易日前
    exit_days 个交易日清仓。票息不单独入账（全价交易，见 docs）。
    """
    import panelbt as bt
    import polars as pl
    from panelbt.markets.ashare import tradability

    if exec not in ("open", "close"):
        raise ValueError("exec: open 或 close")
    if not (isinstance(exit_days, int) and exit_days >= 0):
        raise ValueError("exit_days: 非负整数")
    p = funds.panel("cb", list(FIELDS), start=start, end=end, symbols=symbols)
    dates = np.asarray(p.dates)
    codes = np.asarray(p.symbols)
    t, n = len(dates), len(codes)
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
    for j, s in enumerate(codes):
        first_day[0, j] = (
            listing.get(str(s)) is not None and dates[0] <= listing[str(s)] <= dates[0]
        )
    up, down = price_limits(dates, p["previous_close"], first_day)
    price = np.asarray(p[exec], dtype=float)
    ok = np.asarray(p["has_quote"]) & listed
    can_buy, can_sell = tradability(price, p["high"], p["low"], up, down, ~ok)
    close = np.where(np.isfinite(p["close"]), p["close"], np.nan)
    delist = np.asarray(p["delist"], dtype=bool).copy()

    # 赎回：每只债取第一次「实施强赎」公告，否则取「到期赎回」公告。
    calls = funds.table("cb", "calls").collect()
    chosen = {}
    for status in (MATURITY, CALL_IMPLEMENTED):  # 后写的覆盖先写的：强赎优先
        rows = (
            calls.filter(
                (pl.col("status") == status) & pl.col("symbol").is_in(codes.tolist())
            )
            .sort("announcement_date")
            .unique("symbol", keep="first")
        )
        for r in rows.iter_rows(named=True):
            chosen[r["symbol"]] = (
                status,
                r["announcement_date"],
                r["redemption_price"],
            )
    last_quote = _quote_end(funds, codes)
    redemption = np.full(n, np.nan)
    exit_rows = np.full(n, t, dtype=np.int64)
    diag = dict(
        settled_at_redemption=0,
        redemption_price_missing=[],
        delisted_without_redemption=[],
        delist_row_moved=0,
        exits=0,
    )
    for j, code in enumerate(codes):
        code = str(code)
        stop = np.flatnonzero(delist[:, j])
        last = last_quote[j]
        if code not in chosen:
            if len(stop):
                diag["delisted_without_redemption"].append(code)
            continue
        status, announced, value = chosen[code]
        if value is None or not np.isfinite(value):
            value = 100.0
            diag["redemption_price_missing"].append(code)
        redemption[j] = value
        if last is None:
            continue
        last_row = int(
            np.searchsorted(dates, last)
        )  # 区间内最后交易日所在行（可能为 T）
        if last_row >= t or dates[last_row] != last:
            last_row = t if last > dates[-1] else -1
        # 最后交易日之后、摘牌之前：估值价为赎回价，不能成交。
        if len(stop) and last_row >= 0:
            stop_row = int(stop[0])
            if stop_row <= last_row + 1 and last_row + 2 <= t:
                delist[stop_row:, j] = False
                if last_row + 2 < t:
                    delist[last_row + 2, j] = True
                stop_row = last_row + 2
                diag["delist_row_moved"] += 1
            if last_row + 1 < t:
                close[last_row + 1 : stop_row, j] = value
                diag["settled_at_redemption"] += 1
        # 清仓行：最后交易日前 exit_days 个交易日；强赎的不早于公告可知那天（公告日的下一交易日），
        # 晚于最后交易日就不做（只能按赎回价了结）。
        if 0 <= last_row < t:
            row = last_row - exit_days
            if status == CALL_IMPLEMENTED:
                row = max(row, int(np.searchsorted(dates, announced, side="right")))
            if 0 <= row <= last_row:
                exit_rows[j] = row
                can_buy[row:, j] = False
                diag["exits"] += 1
    if block_after_call:
        can_buy &= ~np.asarray(p["call_announced"], dtype=bool)
    market = bt.Market(
        np.where(ok, price, np.nan),
        close,
        can_buy,
        can_sell,
        volume=np.nan_to_num(np.asarray(p["volume"], dtype=float)),
        delist=delist,
        validate=validate,
    )
    return CBMarket(market, dates, codes, p, exit_rows, redemption, diag)


__all__ = ["CB", "CBMarket", "cb_market", "price_limits"]
