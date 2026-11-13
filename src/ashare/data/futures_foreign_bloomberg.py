"""Bloomberg → 境外期货统一字段。**未用真实终端验证。**

拿到终端后第一次该跑的核对（详见 ``docs/FOREIGN_FUTURES.md``「首次核对清单」）：

1. ``FUT_CHAIN``（含已到期合约）返回的代码写法：已到期合约是 ``CLZ5`` 还是 ``CLZ15``，两种都能解析，但要确认；
2. CLZ5 2025-11-03 至 11-07 的 ``PX_SETTLE`` 与 CME 公布的结算价逐日相同，``PX_LAST`` 是收盘价而不是结算价，
   日期就是 CME 交易日；``OPEN_INT`` 与 CME 日报同一天对上（Bloomberg 的持仓量可能晚一天）；
3. ``LAST_TRADEABLE_DT`` = CME 公布的最后交易日；``FUT_CONT_SIZE`` = 1000；
4. COZ5（ICE 布伦特）同样核对；分钟线 1 天，时间是 UTC、成交量之和与 ``PX_VOLUME`` 接近。

口径：

- 合约链：``FUT_CHAIN``（批量字段，元素名 ``Security Description``）在通用代码 ``CL1 Comdty`` 上取，
  加覆盖 ``INCLUDE_EXPIRED_CONTRACTS = Y`` 才含已到期合约（字段与覆盖名凭经验，**未核实**）。
- 合约表：``FUT_FIRST_TRADE_DT`` → listing_date，``LAST_TRADEABLE_DT`` → last_trade_date，``FUT_DLV_DT_LAST`` →
  delivery_date，``FUT_CONT_SIZE`` → multiplier，``FUT_TICK_SIZE`` → tick_size，``CRNCY`` → currency，``NAME`` → name。
  代码年份按 ``LAST_TRADEABLE_DT`` 解十年回绕。
- 日线：``PX_OPEN``/``PX_HIGH``/``PX_LOW``/``PX_LAST`` → 开高低收，``PX_SETTLE`` → settlement，``PX_VOLUME`` →
  volume，``OPEN_INT`` → open_interest；日期是 Bloomberg 历史请求的日期（按交易所交易日，**未核实**）。
- Bloomberg 的价格对 CL、CO 是美元/桶（有的品种以美分报价，``PX_SCALING_FACTOR`` 一类字段**未核实**，
  品种表只放了美元/桶报价的品种）。
"""

import datetime as dt
import re

import polars as pl

from .futures_foreign import (
    MINUTE,
    MONTH_CODES,
    PRODUCTS,
    ForeignAdapter,
    _as_date,
    bars_on_closed_days,
    calendar_from_open_days,
    check_settlements,
    closed_days_by_exchange,
    conform_minute,
    inside_listing,
    previous_settlement,
    raise_on_closed_trading_days,
    resolve_year,
    trade_dates,
    unified_symbol,
)
from .futures_schema import conform

REFERENCE_FIELDS = (
    "NAME",
    "FUT_CONT_SIZE",
    "FUT_TICK_SIZE",
    "FUT_FIRST_TRADE_DT",
    "LAST_TRADEABLE_DT",
    "FUT_DLV_DT_LAST",
    "CRNCY",
)
# 合约表里缺了就报错的字段（不回退到品种表）；其余字段缺了留空。
REQUIRED_REFERENCE = ("LAST_TRADEABLE_DT", "FUT_CONT_SIZE", "FUT_TICK_SIZE", "CRNCY")
# 日线历史从 start 前 7 个自然日取：内部日历要知道 start 的上一开市日（分钟线从那天傍晚开始），
# start 当天的 previous_settlement 也要有。
HISTORY_LEAD_DAYS = 7
HISTORY_FIELDS = (
    "PX_OPEN",
    "PX_HIGH",
    "PX_LOW",
    "PX_LAST",
    "PX_SETTLE",
    "PX_VOLUME",
    "OPEN_INT",
)
CHAIN_OVERRIDES = {"INCLUDE_EXPIRED_CONTRACTS": "Y"}


def _pattern(root):
    return re.compile(
        rf"^{re.escape(root)} ?(?P<month>[{MONTH_CODES}])(?P<year>\d{{1,2}}) Comdty$"
    )


def _generic(root):
    return re.compile(rf"^{re.escape(root)} ?(?P<rank>\d+) Comdty$")


def product_of_ticker(ticker):
    for p in PRODUCTS.values():
        if p.bloomberg_root and (
            _pattern(p.bloomberg_root).match(ticker)
            or _generic(p.bloomberg_root).match(ticker)
        ):
            return p
    return None


def parse_ticker(ticker, ref):
    """``CLZ6 Comdty`` / ``CLZ16 Comdty`` / ``W N59 Comdty`` 形状 → (统一品种, 四位年份, 月份)；通用代码返回 None。

    ``ref`` 是参照日（最好是最后交易日），一位、两位年份都按它解十年 / 百年回绕。
    """
    for p in PRODUCTS.values():
        if not p.bloomberg_root:
            continue
        m = _pattern(p.bloomberg_root).match(ticker)
        if m:
            month = MONTH_CODES.index(m["month"]) + 1
            return p.code, resolve_year(m["year"], month, ref), month
    return None


def parse_generic(ticker):
    """``CL1 Comdty`` → (统一品种, 序号)；不是通用代码返回 None。"""
    for p in PRODUCTS.values():
        if p.bloomberg_root:
            m = _generic(p.bloomberg_root).match(ticker)
            if m:
                return p.code, int(m["rank"])
    return None


def _text(value):
    return None if value is None else _as_date(value).strftime("%Y%m%d")


def contracts_from_reference(rows, products):
    """``ReferenceDataRequest`` 的行 → (统一合约表, ticker → symbol, 跳过的代码)。

    代码属于所选品种、却缺 ``REQUIRED_REFERENCE`` 里的字段时报错（不回退到品种表的乘数、最小变动价位、
    币种，也不悄悄跳过）；不属于所选品种的代码跳过并列进诊断。"""
    out, tickers, skipped = [], {}, []
    for r in rows:
        if parse_generic(r["security"]) is not None:  # 通用代码（CL1 Comdty）不是合约
            skipped.append(r["security"])
            continue
        selected = product_of_ticker(r["security"])
        if selected is None or selected.code not in products:
            skipped.append(r["security"])
            continue
        missing = [f for f in REQUIRED_REFERENCE if r.get(f) in (None, "")]
        if missing:
            raise ValueError(
                f"{r['security']}: 合约表缺字段 {missing}；不回退到品种表，先核对字段名与终端返回"
            )
        last = r["LAST_TRADEABLE_DT"]
        parsed = parse_ticker(r["security"], last)
        code, year, month = parsed
        p = PRODUCTS[code]
        symbol = unified_symbol(code, year, month)
        tickers[r["security"]] = symbol
        out.append(
            dict(
                symbol=symbol,
                exchange=p.exchange,
                product=code,
                name=r.get("NAME"),
                listing_date=_text(r.get("FUT_FIRST_TRADE_DT")),
                last_trade_date=_text(last),
                delivery_date=_text(r.get("FUT_DLV_DT_LAST")),
                delivery_month=f"{year}{month:02d}",
                multiplier=float(r["FUT_CONT_SIZE"]),
                quote_unit=p.quote_unit,
                tick_size=float(r["FUT_TICK_SIZE"]),
                settlement_type=p.settlement_type,
                currency=str(r["CRNCY"]).strip(),
                multiplier_source="contract_table_unverified",
            )
        )
    table = (
        conform("contracts", pl.DataFrame(out))
        .unique("symbol", keep="last")
        .sort("symbol")
        if out
        else None
    )
    return table, tickers, skipped


def daily_from_history(
    rows, tickers, contracts, calendar, start, end, *, require_settlement=True
):
    """``HistoricalDataRequest`` 的行 → 统一日线。返回 (日线, 诊断)。

    ``rows`` 可以比 [start, end] 多取几天（``calendar`` 是覆盖它们的内部日历）：``previous_settlement``
    在裁到 [start, end] 之前算，所以 start 当天也有。"""
    frame = pl.DataFrame(
        [
            dict(
                symbol=tickers.get(r["security"]),
                date=_text(r["date"]),
                **{
                    f: (None if r.get(f) is None else float(r[f]))
                    for f in HISTORY_FIELDS
                },
            )
            for r in rows
        ],
        schema=dict(
            symbol=pl.String, date=pl.String, **{f: pl.Float64 for f in HISTORY_FIELDS}
        ),
    )
    diag = dict(unknown_ticker_rows=frame["symbol"].null_count())
    frame = frame.filter(pl.col("symbol").is_not_null())
    frame = frame.select(
        "date",
        "symbol",
        open="PX_OPEN",
        high="PX_HIGH",
        low="PX_LOW",
        close="PX_LAST",
        settlement="PX_SETTLE",
        volume="PX_VOLUME",
        amount=pl.lit(None, dtype=pl.Float64),
        open_interest="OPEN_INT",
    )
    frame = previous_settlement(frame, contracts, calendar)
    frame = frame.filter(pl.col("date").is_between(pl.lit(start), pl.lit(end)))
    frame, dropped = inside_listing(frame, contracts)
    diag["outside_listing_rows"] = dropped
    diag["settlement_missing"] = check_settlements(
        frame, contracts, require=require_settlement
    )
    return conform("daily", frame).sort("symbol", "date"), diag


class BloombergFutures(ForeignAdapter):
    """Bloomberg 适配器（桌面 API，本机终端）。**未用真实终端验证。**

    ``source`` 是 ``ashare.data.sources.bloomberg.BloombergSource``。``start``/``end`` 为 YYYYMMDD（含两端）。

    例::

        from ashare.data.sources.bloomberg import BloombergSource
        fut = BloombergFutures(BloombergSource(), ["CL", "B"], "20251103", "20251107")
        fut.contracts(); fut.daily(); fut.minute("20251105", "20251105")
    """

    source_name = "Bloomberg"

    def __init__(
        self,
        source,
        products,
        start,
        end,
        *,
        margin=None,
        generic_ranks=(1, 2),
        require_settlement=True,
        closed_days=None,
    ):
        super().__init__(margin=margin)
        unknown = [
            p
            for p in products
            if p not in PRODUCTS or PRODUCTS[p].bloomberg_root is None
        ]
        if unknown:
            raise ValueError(f"Bloomberg 适配器不认识的品种：{unknown}")
        self.source, self.products = source, list(products)
        self.start, self.end = start, end
        self.generic_ranks = tuple(generic_ranks)
        self.require_settlement = require_settlement
        self._allowed = closed_days_by_exchange(
            closed_days, sorted({PRODUCTS[c].exchange for c in self.products})
        )

    def _lead(self):
        return (_as_date(self.start) - dt.timedelta(days=HISTORY_LEAD_DAYS)).strftime(
            "%Y%m%d"
        )

    def _compute(self):
        generics = [f"{PRODUCTS[c].bloomberg_root}1 Comdty" for c in self.products]
        chain = self.source.reference(
            generics, ["FUT_CHAIN"], overrides=CHAIN_OVERRIDES
        )
        tickers = sorted(
            {
                item["Security Description"]
                for r in chain
                for item in (r.get("FUT_CHAIN") or [])
            }
        )
        reference = (
            self.source.reference(tickers, list(REFERENCE_FIELDS)) if tickers else []
        )
        contracts, symbol_of, skipped = contracts_from_reference(
            reference, self.products
        )
        if contracts is None:
            raise ValueError("FUT_CHAIN 里没有所选品种的合约")
        lead = self._lead()
        live = contracts.filter(
            (pl.col("listing_date").is_null() | (pl.col("listing_date") <= self.end))
            & (pl.col("last_trade_date") >= lead)
        )
        wanted = [t for t, s in symbol_of.items() if s in set(live["symbol"])]
        history = (
            self.source.historical(wanted, list(HISTORY_FIELDS), lead, self.end)
            if wanted
            else []
        )
        if history and all(r.get("PX_SETTLE") is None for r in history):
            raise ValueError(
                "历史请求的 PX_SETTLE 整列为空：结算价字段口径与预期不同，日历和 previous_settlement "
                "都推不出来；先按 docs/FOREIGN_FUTURES.md 的首次核对清单核对"
            )
        exchange_of = dict(zip(live["symbol"], live["exchange"], strict=True))
        open_days = {}
        for r in history:
            symbol = symbol_of.get(r["security"])
            if symbol in exchange_of and r.get("PX_SETTLE") is not None:
                open_days.setdefault(exchange_of[symbol], set()).add(_text(r["date"]))
        # 有行情却没有 PX_SETTLE 的工作日会被当成休市（分钟线并进下一开市日）：不在已知休市日里就报错。
        suspicious, accepted = {}, {}
        for r in history:
            symbol = symbol_of.get(r["security"])
            day = _text(r["date"])
            e = exchange_of.get(symbol)
            known = open_days.get(e)
            if e is None or not known or day in known:
                continue
            if not (min(known) <= day <= max(known)):
                continue
            bucket = accepted if day in self._allowed[e] else suspicious
            bucket.setdefault(e, {}).setdefault(day, 0)
            bucket[e][day] += 1
        raise_on_closed_trading_days(suspicious)
        empty = sorted(set(live["exchange"]) - set(open_days))
        if empty:
            raise ValueError(
                f"{empty}: 历史请求里一个 PX_SETTLE 都没有，推不出开市日；先按首次核对清单核对"
            )
        full = pl.concat(
            [
                calendar_from_open_days(
                    e,
                    open_days.get(e, set()),
                    min([lead, *open_days.get(e, set())]),
                    max([self.end, *open_days.get(e, set())]),
                )
                for e in sorted(set(live["exchange"]))
            ]
        )
        daily, diag = daily_from_history(
            history,
            symbol_of,
            live,
            full,
            self.start,
            self.end,
            require_settlement=self.require_settlement,
        )
        diag["skipped_tickers"] = skipped
        diag["closed_days_with_bars"] = {
            e: dict(sorted(d.items())) for e, d in sorted(accepted.items())
        }
        diag["unknown_listing_contracts"] = sorted(
            set(live.filter(pl.col("listing_date").is_null())["symbol"])
            & set(daily["symbol"])
        )
        calendar = full.filter(
            pl.col("date").is_between(pl.lit(self.start), pl.lit(self.end))
        )
        contracts = live.filter(pl.col("symbol").is_in(list(set(daily["symbol"]))))
        continuous = self._generic_history()
        return dict(
            contracts=contracts,
            daily=daily,
            calendar=calendar,
            diagnostics=diag,
            source_continuous=continuous,
            _tickers=symbol_of,
            _calendar=full,
        )

    def _generic_history(self):
        """通用代码（``CL1 Comdty``、``CL2 Comdty``…）的结算价，只用于对账。Bloomberg 通用代码默认在到期时
        换到下一合约，换月口径可在终端里按代码设置（**未核实**）；历史请求不给当天对应哪个合约，只能按价格对账。"""
        securities = [
            f"{PRODUCTS[c].bloomberg_root}{k} Comdty"
            for c in self.products
            for k in self.generic_ranks
        ]
        rows = self.source.historical(securities, ["PX_SETTLE"], self.start, self.end)
        out = []
        for r in rows:
            product, rank = parse_generic(r["security"])
            out.append(
                dict(
                    date=_text(r["date"]),
                    continuous=r["security"],
                    product=product,
                    rank=rank,
                    settlement=r.get("PX_SETTLE"),
                )
            )
        return pl.DataFrame(
            out,
            schema=dict(
                date=pl.String,
                continuous=pl.String,
                product=pl.String,
                rank=pl.Int64,
                settlement=pl.Float64,
            ),
        )

    def minute(self, start, end):
        """分钟线（``IntradayBarRequest``，1 分钟，``TRADE``）。``start``/``end`` 为 YYYYMMDD 交易日（含两端），
        必须落在适配器的 [start, end] 里，否则不发请求直接报错。

        按合约逐个请求 [start 的上一开市日 UTC 0 点, end 次日 UTC 0 点)（start 那个交易日从上一开市日傍晚开始），
        再按交易时段归到交易日、只留 [start, end]。``time`` 当作 K 线开始时刻（**未核实**，见首次核对清单）。"""
        if not (self.start <= start <= end <= self.end):
            raise ValueError(
                f"minute 的区间 [{start}, {end}] 要落在适配器的区间 [{self.start}, {self.end}] 里"
            )
        tables = self._all()
        contracts, symbol_of, calendar = (
            tables["contracts"],
            tables["_tickers"],
            tables["_calendar"],
        )
        last = dt.datetime.combine(
            _as_date(end) + dt.timedelta(days=1), dt.time(), dt.UTC
        )
        frames = []
        for ticker, symbol in symbol_of.items():
            if symbol not in set(contracts["symbol"]):
                continue
            p = PRODUCTS[contracts.filter(pl.col("symbol") == symbol)["product"][0]]
            open_days = set(
                calendar.filter(pl.col("is_open") & (pl.col("exchange") == p.exchange))[
                    "date"
                ]
            )
            before = [d for d in open_days if d < start]
            if not before:
                raise ValueError(f"{p.exchange}: {start} 之前没有已知的开市日")
            first = dt.datetime.combine(_as_date(max(before)), dt.time(), dt.UTC)
            rows = self.source.intraday_bars(ticker, first, last, interval=1)
            if not rows:
                continue
            raise_on_closed_trading_days(
                {
                    p.exchange: bars_on_closed_days(
                        [
                            t if t.tzinfo else t.replace(tzinfo=dt.UTC)
                            for t in (r["time"] for r in rows)
                        ],
                        p.session,
                        open_days,
                        self._allowed[p.exchange],
                    )
                }
            )
            times = [
                t if t.tzinfo else t.replace(tzinfo=dt.UTC)
                for t in (r["time"] for r in rows)
            ]
            frames.append(
                pl.DataFrame(
                    dict(
                        ts=times,
                        date=trade_dates(times, p.session, open_days),
                        symbol=[symbol] * len(rows),
                        open=[float(r["open"]) for r in rows],
                        high=[float(r["high"]) for r in rows],
                        low=[float(r["low"]) for r in rows],
                        close=[float(r["close"]) for r in rows],
                        volume=[float(r["volume"]) for r in rows],
                    )
                )
            )
        if not frames:
            return conform_minute(
                pl.DataFrame(schema={k: c.dtype for k, c in MINUTE.items()})
            )
        bars = pl.concat(frames).filter(
            pl.col("date").is_between(pl.lit(start), pl.lit(end))
        )
        bars, _ = inside_listing(bars, contracts)
        return conform_minute(bars)


__all__ = [
    "BloombergFutures",
    "contracts_from_reference",
    "daily_from_history",
    "parse_generic",
    "parse_ticker",
]
