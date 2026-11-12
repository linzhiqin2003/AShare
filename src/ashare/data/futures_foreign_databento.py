"""Databento → 境外期货统一字段。**未用真实账号验证。**

拿到账号后第一次该跑的核对（详见 ``docs/FOREIGN_FUTURES.md``「首次核对清单」）：

1. ``adapter.estimate()`` 看一周 CL 的四个请求各多少钱（不计费）；
2. 预算 1 美元取 CL 2026-11-02 至 11-06：CLZ6 的 ``last_trade_date`` = CME 公布的最后交易日；
   每天的 ``settlement`` = CME 公布的结算价；``open_interest`` 与 CME 日报（Daily Bulletin）的持仓量同一天对上；
3. 同一周取 ICE 布伦特（``IFEU.IMPACT``）：结算价 = ICE 公布值，价格是美元/桶；
4. 2020-04-17 至 04-22 的 CLK0：结算价有负值（2020-04-20 为 −37.63），统一字段保留真实负价。

口径（每一项都写在文档的字段映射表里）：

- 合约表来自 ``definition``：同一 ``instrument_id`` 取最后一条（``security_update_action`` 为 D 的删掉）；
  只要 ``instrument_class == "F"``（单月期货，价差合约的代码带 ``-``，类型为 S）。年月优先用定义里的
  ``maturity_year``/``maturity_month``，缺失时按代码解析、以 ``expiration`` 为参照日解十年回绕。
  ``listing_date``/``last_trade_date`` = ``activation``/``expiration`` 按交易时段归属的交易日。
- 日线**不用** ``ohlcv-1d``（官方说明它按 UTC 日切分，与 CME 前一天 17:00 起算的交易日不一致），用
  ``ohlcv-1h`` 按交易日聚合（CME 17:00 芝加哥、ICE 23:00 伦敦都在整点，小时线切得开）：开盘 = 当日第一根
  的开盘，最高 / 最低 = 各根极值，收盘 = 最后一根的收盘。
- 结算价、持仓量、结算量来自 ``statistics``：``stat_type`` 3 / 9 / 6。结算价只取 ``stat_flags`` 第 0 位为 1
  （final）、第 3 位为 0（不是日内结算）的，交易日取 ``ts_ref``（官方说明是 CME 的交易日，只有日期精度，不要
  换时区）。持仓量、结算量一般在下一个 UTC 日发布：有 ``ts_ref`` 用 ``ts_ref``，没有时归发布时刻所属交易日的
  上一开市日（**未核实**）。同一键多条取 ``ts_recv`` 最晚的一条，最后一条是删除（``update_action == 2``）则为空。
- 成交量默认用结算量（``volume="cleared"``，含大宗、EFP，与交易所日报一致）；``volume="electronic"`` 时用
  小时线成交量之和（只含电子盘）。两者不混用，缺了就是空。
- 价格保留来源报价单位（美元/桶），``multiplier`` = ``unit_of_measure_qty``（BBL，1000）。成交额来源没有，为空。
"""

import bisect
import datetime as dt
import re

import polars as pl

from .futures_foreign import (
    MONTH_CODES,
    PRODUCTS,
    ForeignAdapter,
    calendar_from_open_days,
    conform_minute,
    inside_listing,
    previous_settlement,
    resolve_year,
    trade_dates,
    unified_symbol,
)
from .futures_schema import conform

GLBX_RAW = re.compile(
    r"^(?P<root>[A-Z0-9]+?)(?P<month>[FGHJKMNQUVXZ])(?P<year>\d{1,2})$"
)
# ICE iMpact：``BRN FMQ0024!`` = 品种（空格补足 4 位）+ F 期货 + M 月度 + 月份码 + 日（月度为 00）+ 两位年份 + 结束符。
ICE_RAW = re.compile(
    r"^(?P<root>[A-Z0-9]+)\s*F(?P<term>M)(?P<month>[FGHJKMNQUVXZ])(?P<day>\d{2})(?P<year>\d{2})!?$"
)
CONTINUOUS = re.compile(r"^(?P<root>[A-Z0-9]+)\.(?P<rule>[cnv])\.(?P<rank>\d+)$")
VOLUMES = ("cleared", "electronic")
SETTLEMENTS = ("final", "any")
STAT_SETTLEMENT, STAT_CLEARED_VOLUME, STAT_OPEN_INTEREST = 3, 6, 9
INT64_MAX = 2**63 - 1


def parse_raw_symbol(raw, ref):
    """Databento 原始代码 → (root, 四位年份, 月份)；不是单月期货（价差、期权、看不懂）返回 None。

    ``CLZ6``（CME 一位年份）、``CLZ16``、``BRN FMZ0026!``（ICE 两位年份）。``ref`` 为参照日，见 ``resolve_year``。
    """
    text = raw.strip()
    for pattern in (ICE_RAW, GLBX_RAW):
        m = pattern.match(text)
        if m:
            month = MONTH_CODES.index(m["month"]) + 1
            return m["root"], resolve_year(m["year"], month, ref), month
    return None


def parse_continuous(code):
    """``CL.c.0`` → (root, rule, rank)。c = 按到期先后（日历），n = 按前一日持仓量，v = 按前一日成交量。"""
    m = CONTINUOUS.match(code)
    if not m:
        raise ValueError(f"不是 Databento 连续合约记法：{code}")
    return m["root"], m["rule"], int(m["rank"])


def _product_of_root(root):
    for p in PRODUCTS.values():
        if p.databento_root == root:
            return p
    return None


def _date_text(series):
    return [None if v is None else v.strftime("%Y%m%d") for v in series]


def contracts_from_definitions(defs, products):
    """``definition`` 原始表 → (统一合约表, instrument_id → symbol 的表, 诊断)。"""
    diag = dict(
        maturity_conflicts=[],
        no_expiration=[],
        multiplier_mismatch=[],
        display_factor=[],
    )
    defs = defs.filter(pl.col("instrument_class") == "F").sort("ts_recv")
    latest = defs.group_by("instrument_id", maintain_order=True).last()
    latest = latest.filter(pl.col("security_update_action") != "D")
    rows, ids = [], []
    for r in latest.iter_rows(named=True):
        expiration, activation = r["expiration"], r["activation"]
        if expiration is None:
            diag["no_expiration"].append(r["raw_symbol"])
            continue
        parsed = parse_raw_symbol(r["raw_symbol"], expiration)
        if parsed is None:
            continue
        root, year, month = parsed
        product = _product_of_root(root)
        if product is None or product.code not in products:
            continue
        my, mm = r.get("maturity_year"), r.get("maturity_month")
        if my is not None and 1900 < my < 2200 and mm is not None and 1 <= mm <= 12:
            if (my, mm) != (year, month):
                diag["maturity_conflicts"].append(
                    dict(
                        raw=r["raw_symbol"],
                        definition=f"{my}{mm:02d}",
                        parsed=f"{year}{month:02d}",
                    )
                )
            year, month = my, mm
        session = product.session
        listing, last = trade_dates([activation, expiration], session)
        qty = r.get("unit_of_measure_qty")
        multiplier = (
            qty if qty is not None and qty == qty and qty > 0 else product.multiplier
        )
        if multiplier != product.multiplier:
            diag["multiplier_mismatch"].append(
                dict(
                    raw=r["raw_symbol"], definition=multiplier, table=product.multiplier
                )
            )
        if r.get("display_factor") not in (None, 1.0):
            diag["display_factor"].append(
                dict(raw=r["raw_symbol"], display_factor=r["display_factor"])
            )
        symbol = unified_symbol(product.code, year, month)
        ids.append(dict(instrument_id=int(r["instrument_id"]), symbol=symbol))
        rows.append(
            dict(
                symbol=symbol,
                exchange=product.exchange,
                product=product.code,
                name=f"{product.name} {year}-{month:02d}",
                listing_date=listing,
                last_trade_date=last,
                delivery_date=None,
                delivery_month=f"{year}{month:02d}",
                multiplier=float(multiplier),
                quote_unit=product.quote_unit,
                tick_size=r.get("min_price_increment") or product.tick_size,
                settlement_type=product.settlement_type,
                currency=(r.get("currency") or product.currency).strip()
                or product.currency,
                multiplier_source="contract_table_unverified",
            )
        )
    schema = dict(instrument_id=pl.Int64, symbol=pl.String)
    id_map = pl.DataFrame(ids, schema=schema)
    duplicated = id_map.group_by("symbol").len().filter(pl.col("len") > 1)
    if len(duplicated):
        diag["symbols_with_several_ids"] = duplicated["symbol"].to_list()
    contracts = pl.DataFrame(rows) if rows else None
    if contracts is not None:
        contracts = (
            conform("contracts", contracts).unique("symbol", keep="last").sort("symbol")
        )
    return contracts, id_map, diag


def _latest_stat(stats, stat_type, value):
    """同一 (instrument_id, 键) 取 ts_recv 最晚的一条；最后一条是删除则为空。"""
    picked = stats.filter(pl.col("stat_type") == stat_type).sort("ts_recv")
    last = picked.group_by("instrument_id", "key", maintain_order=True).last()
    return last.select(
        "instrument_id",
        date="key",
        value=pl.when(pl.col("update_action") == 2).then(None).otherwise(pl.col(value)),
    )


def settlement_rows(stats, *, settlement="final"):
    """结算价：键 = ts_ref 的 UTC 日期（即交易日）。"""
    if settlement not in SETTLEMENTS:
        raise ValueError(f"settlement 只能是 {SETTLEMENTS}")
    flags = pl.col("stat_flags").cast(pl.Int64)
    keep = (flags & 8) == 0
    if settlement == "final":
        keep = keep & ((flags & 1) == 1)
    picked = stats.filter(
        (pl.col("stat_type") == STAT_SETTLEMENT) & keep & pl.col("ts_ref").is_not_null()
    )
    picked = picked.with_columns(key=pl.col("ts_ref").dt.strftime("%Y%m%d"))
    return _latest_stat(picked, STAT_SETTLEMENT, "price").rename(
        {"value": "settlement"}
    )


def quantity_rows(stats, stat_type, session, open_days, name):
    """持仓量 / 结算量：有 ts_ref 用其 UTC 日期；没有时归「发布时刻所属交易日」的上一开市日（未核实）。"""
    picked = stats.filter(pl.col("stat_type") == stat_type)
    if picked.is_empty():
        return pl.DataFrame(
            schema=dict(instrument_id=pl.UInt32, date=pl.String, **{name: pl.Float64})
        )
    ordered = sorted(open_days)
    published = trade_dates(picked["ts_event"], session, open_days)

    def before(day):
        k = bisect.bisect_left(ordered, day)
        return ordered[k - 1] if k > 0 else None

    fallback = [None if d is None else before(d) for d in published]
    refs = _date_text(picked["ts_ref"].to_list())
    keys = [r if r is not None else f for r, f in zip(refs, fallback, strict=True)]
    picked = picked.with_columns(
        key=pl.Series(keys, dtype=pl.String),
        quantity=pl.when(pl.col("quantity") >= INT64_MAX)
        .then(None)
        .otherwise(pl.col("quantity"))
        .cast(pl.Float64),
    ).filter(pl.col("key").is_not_null())
    return _latest_stat(picked, stat_type, "quantity").rename({"value": name})


def aggregate_bars(bars, session, open_days):
    """K 线（小时或分钟）→ 按交易日的开高低收与电子盘成交量。``bars`` 需有 symbol 列。"""
    bars = bars.sort("ts_event")
    bars = bars.with_columns(
        date=pl.Series(
            trade_dates(bars["ts_event"], session, open_days), dtype=pl.String
        )
    )
    return bars.group_by("symbol", "date", maintain_order=True).agg(
        open=pl.col("open").first(),
        high=pl.col("high").max(),
        low=pl.col("low").min(),
        close=pl.col("close").last(),
        bar_volume=pl.col("volume").cast(pl.Float64).sum(),
    )


def daily_from_raw(
    contracts,
    id_map,
    bars,
    stats,
    calendar,
    start,
    end,
    *,
    volume="cleared",
    settlement="final",
):
    """小时线 + 统计 → 统一日线（只留 [start, end] 且在上市期内的行）。返回 (日线, 诊断)。"""
    if volume not in VOLUMES:
        raise ValueError(f"volume 只能是 {VOLUMES}")
    diag = {}
    ids = id_map.with_columns(pl.col("instrument_id").cast(pl.Int64)).join(
        contracts.select("symbol", "product"), on="symbol"
    )
    keyed = {}
    for name, frame in (("stat", stats), ("bar", bars)):
        joined = frame.with_columns(pl.col("instrument_id").cast(pl.Int64)).join(
            ids, on="instrument_id", how="left"
        )
        diag[f"unknown_instrument_{name}_rows"] = joined["symbol"].null_count()
        keyed[name] = joined.filter(pl.col("symbol").is_not_null())
    pieces = []
    for code in sorted(set(contracts["product"])):
        product = PRODUCTS[code]
        open_days = set(
            calendar.filter(
                pl.col("is_open") & (pl.col("exchange") == product.exchange)
            )["date"]
        )
        own_ids = ids.filter(pl.col("product") == code).select(
            "instrument_id", "symbol"
        )
        own_stats = (
            keyed["stat"].filter(pl.col("product") == code).drop("symbol", "product")
        )
        own_bars = keyed["bar"].filter(pl.col("product") == code).drop("product")
        merged = aggregate_bars(own_bars, product.session, open_days)
        for part in (
            settlement_rows(own_stats, settlement=settlement),
            quantity_rows(
                own_stats,
                STAT_OPEN_INTEREST,
                product.session,
                open_days,
                "open_interest",
            ),
            quantity_rows(
                own_stats,
                STAT_CLEARED_VOLUME,
                product.session,
                open_days,
                "cleared_volume",
            ),
        ):
            part = part.with_columns(pl.col("instrument_id").cast(pl.Int64)).join(
                own_ids, on="instrument_id"
            )
            merged = merged.join(
                part.drop("instrument_id"),
                on=["symbol", "date"],
                how="full",
                coalesce=True,
            )
        pieces.append(merged)
    long = pl.concat(pieces, how="diagonal_relaxed") if pieces else pl.DataFrame()
    for column in (
        "open",
        "high",
        "low",
        "close",
        "settlement",
        "open_interest",
        "cleared_volume",
        "bar_volume",
    ):
        if column not in long.columns:
            long = long.with_columns(pl.lit(None, dtype=pl.Float64).alias(column))
    long = long.filter(pl.col("date").is_between(pl.lit(start), pl.lit(end)))
    long, dropped = inside_listing(long, contracts)
    diag["outside_listing_rows"] = dropped
    long = long.with_columns(
        volume=pl.col("cleared_volume")
        if volume == "cleared"
        else pl.col("bar_volume"),
        amount=pl.lit(None, dtype=pl.Float64),
    )
    long = previous_settlement(long, contracts, calendar)
    diag["nonpositive_price_rows"] = long.filter(
        pl.any_horizontal(
            [pl.col(c) <= 0 for c in ("open", "high", "low", "close", "settlement")]
        )
    ).height
    daily = conform(
        "daily",
        long.select(
            "date",
            "symbol",
            "open",
            "high",
            "low",
            "close",
            "settlement",
            "previous_settlement",
            "volume",
            "amount",
            "open_interest",
        ),
    ).sort("symbol", "date")
    return daily, diag


def _drop_mapped(frame):
    return frame.drop("symbol", strict=False)


def _day(text):
    return dt.date(int(text[:4]), int(text[4:6]), int(text[6:8]))


def _iso(day):
    return day.strftime("%Y-%m-%d")


class DatabentoFutures(ForeignAdapter):
    """Databento 适配器。**未用真实账号验证。**

    ``source`` 是 ``ashare.data.sources.databento.DatabentoSource``（带费用闸门，默认预算 0：只估不取）。
    ``start``/``end`` 为 YYYYMMDD（含两端，交易日口径）。``margin`` 为保证金率（标量或「品种 → 比例」，
    来源没有，不给就留空；回测里用 ``panelbt_inputs(missing=...)`` 给）。

    例::

        from ashare.data.sources.databento import DatabentoSource
        src = DatabentoSource(budget_usd=0)            # 只估价
        fut = DatabentoFutures(src, ["CL"], "20261102", "20261106")
        fut.estimate()                                  # 各请求的估算费用（美元），不计费
        src.budget_usd = 2.0                            # 显式给预算才会取数
        fut.contracts(); fut.daily(); fut.minute("20261104", "20261104")
    """

    source_name = "Databento"

    def __init__(
        self,
        source,
        products,
        start,
        end,
        *,
        margin=None,
        volume="cleared",
        settlement="final",
        bar_schema="ohlcv-1h",
        reference_continuous=False,
    ):
        super().__init__(margin=margin)
        unknown = [
            p
            for p in products
            if p not in PRODUCTS or PRODUCTS[p].databento_dataset is None
        ]
        if unknown:
            raise ValueError(
                f"Databento 适配器不认识的品种：{unknown}；可用 {sorted(PRODUCTS)}"
            )
        if volume not in VOLUMES:
            raise ValueError(f"volume 只能是 {VOLUMES}")
        if settlement not in SETTLEMENTS:
            raise ValueError(f"settlement 只能是 {SETTLEMENTS}")
        if bar_schema not in ("ohlcv-1h", "ohlcv-1m"):
            raise ValueError(
                "bar_schema 只能是 ohlcv-1h 或 ohlcv-1m（ohlcv-1d 按 UTC 日切分，不用）"
            )
        self.source, self.products = source, list(products)
        self.start, self.end = start, end
        self.volume, self.settlement, self.bar_schema = volume, settlement, bar_schema
        self.reference_continuous = reference_continuous

    # ------------------------------------------------------------ 请求计划与估价
    def _groups(self):
        groups = {}
        for code in self.products:
            p = PRODUCTS[code]
            groups.setdefault(p.databento_dataset, []).append(f"{p.databento_root}.FUT")
        return groups

    def plan(self):
        """要发出的请求（dataset, schema, symbols, start, end, stype_in），end 不含。

        日线小时线从 start 前 3 天取（覆盖周日晚和前一晚开盘的时段），统计取到 end 后 5 天（持仓量、结算量
        在下一个 UTC 日或周日发布）。"""
        first, last = _day(self.start), _day(self.end)
        out = []
        for dataset, symbols in self._groups().items():
            out.append(
                (
                    dataset,
                    "definition",
                    symbols,
                    _iso(first),
                    _iso(last + dt.timedelta(days=1)),
                    "parent",
                )
            )
            out.append(
                (
                    dataset,
                    "statistics",
                    symbols,
                    _iso(first),
                    _iso(last + dt.timedelta(days=6)),
                    "parent",
                )
            )
            out.append(
                (
                    dataset,
                    self.bar_schema,
                    symbols,
                    _iso(first - dt.timedelta(days=3)),
                    _iso(last + dt.timedelta(days=1)),
                    "parent",
                )
            )
        return out

    def estimate(self):
        """逐个请求调 ``metadata.get_cost``（不计费），返回 [(请求, 美元)] 与合计。"""
        rows = [
            (
                f"{d} {s} {sym} {a}~{b}",
                self.source.estimate(d, s, sym, a, b, stype_in=st),
            )
            for d, s, sym, a, b, st in self.plan()
        ]
        return dict(requests=rows, total=sum(c for _, c in rows))

    # ------------------------------------------------------------ 取数与换算
    def _compute(self):
        raw = {}
        for dataset, schema, symbols, a, b, stype in self.plan():
            frame = self.source.fetch(dataset, schema, symbols, a, b, stype_in=stype)
            key = "bars" if schema.startswith("ohlcv") else schema
            raw.setdefault(key, []).append(frame)
        # ``to_df()`` 自带的 symbol 列是客户端按请求符号映射的结果（母代码请求下不可靠），一律按 instrument_id 自己对。
        defs, stats, bars = (
            _drop_mapped(pl.concat(raw[k], how="diagonal_relaxed"))
            for k in ("definition", "statistics", "bars")
        )
        contracts, id_map, diag = contracts_from_definitions(defs, self.products)
        if contracts is None:
            raise ValueError("definition 里没有所选品种的单月期货合约")
        calendar = self._calendar(stats, id_map, contracts)
        daily, daily_diag = daily_from_raw(
            contracts,
            id_map,
            bars,
            stats,
            calendar,
            self.start,
            self.end,
            volume=self.volume,
            settlement=self.settlement,
        )
        diag.update(daily_diag)
        listed = set(daily["symbol"])
        contracts = contracts.filter(pl.col("symbol").is_in(list(listed)))
        diag["cost_log"] = list(getattr(self.source, "log", []))
        out = dict(
            contracts=contracts,
            daily=daily,
            calendar=calendar,
            diagnostics=diag,
            _id_map=id_map,
        )
        if self.reference_continuous:
            out.update(self._source_continuous(calendar, id_map, contracts))
        return out

    def _calendar(self, stats, id_map, contracts):
        """开市日 = 该交易所有 final 结算价的交易日（来源没有日历接口；整交易所一天都没有结算价即视为休市）。"""
        settle = settlement_rows(
            stats.with_columns(pl.col("instrument_id").cast(pl.Int64)),
            settlement="final",
        )
        settle = settle.join(
            id_map.with_columns(pl.col("instrument_id").cast(pl.Int64)),
            on="instrument_id",
        )
        settle = settle.join(contracts.select("symbol", "exchange"), on="symbol")
        frames = []
        for exchange in sorted(set(contracts["exchange"])):
            days = settle.filter(pl.col("exchange") == exchange)["date"].to_list()
            frames.append(calendar_from_open_days(exchange, days, self.start, self.end))
        return pl.concat(frames)

    def _source_continuous(self, calendar, id_map, contracts):
        """来源自带的连续合约（``c``/``n``/``v`` 三种规则的 rank 0，经 ``symbology.resolve``）→ 逐交易日
        对应的合约，只用于对账。``n.0``（按前一日持仓量）进 mapping 作 main，``c.0`` 作 nearby。
        resolve 返回的 d0/d1 按「d0 含、d1 不含」的日期理解（**未核实**）。"""
        from .sources.databento import mapping_frame

        rows = []
        for code in self.products:
            p = PRODUCTS[code]
            codes = [f"{p.databento_root}.{rule}.0" for rule in "cnv"]
            resolved = self.source.resolve(
                p.databento_dataset,
                codes,
                "continuous",
                "instrument_id",
                _iso(_day(self.start)),
                _iso(_day(self.end) + dt.timedelta(days=1)),
            )
            open_days = sorted(
                calendar.filter(pl.col("is_open") & (pl.col("exchange") == p.exchange))[
                    "date"
                ]
            )
            for r in mapping_frame(resolved).iter_rows(named=True):
                lo, hi = (
                    r["start_date"].replace("-", ""),
                    r["end_date"].replace("-", ""),
                )
                rows.extend(
                    dict(
                        date=day,
                        continuous=r["continuous"],
                        instrument_id=r["instrument_id"],
                    )
                    for day in open_days
                    if lo <= day < hi
                )
        raw = pl.DataFrame(
            rows,
            schema=dict(date=pl.String, continuous=pl.String, instrument_id=pl.Int64),
        )
        joined = raw.join(
            id_map.with_columns(pl.col("instrument_id").cast(pl.Int64)),
            on="instrument_id",
            how="left",
        )
        source = joined.select("date", "continuous", "symbol").sort(
            "continuous", "date"
        )
        rule = pl.col("continuous").str.extract(r"\.([cnv])\.0$")
        mapping = joined.filter(
            rule.is_in(["c", "n"]) & pl.col("symbol").is_not_null()
        ).with_columns(
            kind=pl.when(rule == "n").then(pl.lit("main")).otherwise(pl.lit("nearby"))
        )
        mapping, _ = inside_listing(
            mapping.select("date", "continuous", "kind", "symbol"), contracts
        )
        return dict(
            source_continuous=source,
            mapping=conform("mapping", mapping).sort("continuous", "date"),
        )

    def minute(self, start, end):
        """分钟线（``ohlcv-1m``，另走一次费用闸门）。``start``/``end`` 为 YYYYMMDD 交易日（含两端）。"""
        tables = self._all()
        contracts, id_map, calendar = (
            tables["contracts"],
            tables["_id_map"],
            tables["calendar"],
        )
        frames = []
        for dataset, symbols in self._groups().items():
            frames.append(
                self.source.fetch(
                    dataset,
                    "ohlcv-1m",
                    symbols,
                    _iso(_day(start) - dt.timedelta(days=3)),
                    _iso(_day(end) + dt.timedelta(days=1)),
                    stype_in="parent",
                )
            )
        bars = _drop_mapped(pl.concat(frames, how="diagonal_relaxed")).with_columns(
            pl.col("instrument_id").cast(pl.Int64)
        )
        bars = bars.join(id_map, on="instrument_id", how="inner")
        out = []
        for code in sorted(set(contracts["product"])):
            product = PRODUCTS[code]
            members = contracts.filter(pl.col("product") == code)["symbol"].to_list()
            own = bars.filter(pl.col("symbol").is_in(members)).sort("ts_event")
            open_days = set(
                calendar.filter(
                    pl.col("is_open") & (pl.col("exchange") == product.exchange)
                )["date"]
            )
            own = own.with_columns(
                date=pl.Series(
                    trade_dates(own["ts_event"], product.session, open_days),
                    dtype=pl.String,
                )
            )
            out.append(own)
        bars = pl.concat(out).filter(
            pl.col("date").is_between(pl.lit(start), pl.lit(end))
        )
        bars, _ = inside_listing(bars, contracts)
        return conform_minute(
            bars.select(
                ts="ts_event",
                date="date",
                symbol="symbol",
                open="open",
                high="high",
                low="low",
                close="close",
                volume=pl.col("volume").cast(pl.Float64),
            )
        )


__all__ = [
    "DatabentoFutures",
    "aggregate_bars",
    "contracts_from_definitions",
    "daily_from_raw",
    "parse_continuous",
    "parse_raw_symbol",
    "quantity_rows",
    "settlement_rows",
]
