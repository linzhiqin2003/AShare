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
# DBN v1 的统计数量是 int32，未定义值是 i32::MAX；v2 起是 int64，未定义值是 i64::MAX。≥ i32::MAX 的数量一律
# 当未定义（任何期货的持仓量、成交量都到不了 21 亿手）。
INT32_MAX = 2**31 - 1
# 小时线从 start 前 5 个自然日取：start 的交易时段从上一开市日傍晚开始，周末加一两个休市日也够。
BAR_LEAD_DAYS = 5
# 统计从 start 前 7 天取到 end 后 7 天：开市日（由结算价推出）要覆盖整个 K 线窗口，start 当天的
# previous_settlement 也要有；持仓量、结算量在下一个 UTC 日或周日发布。
STAT_LEAD_DAYS, STAT_LAG_DAYS = 7, 7


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
    several = (
        defs.group_by("instrument_id")
        .agg(pl.col("raw_symbol").str.strip_chars().unique().sort())
        .filter(pl.col("raw_symbol").list.len() > 1)
    )
    if len(several):
        raise ValueError(
            "同一个 instrument_id 在请求窗口里对应几个合约代码（来源复用了 id，按 id 合并会把两个合约"
            f"混成一个）：{several.rows()[:5]}；缩短区间分段取"
        )
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
        multiplier = _positive(r.get("unit_of_measure_qty"))
        tick = _positive(r.get("min_price_increment"))
        currency = (r.get("currency") or "").strip()
        if multiplier is None or tick is None or not currency:
            raise ValueError(
                f"{r['raw_symbol']}: 定义里缺 unit_of_measure_qty / min_price_increment / currency"
                f"（{r.get('unit_of_measure_qty')!r} / {r.get('min_price_increment')!r} / "
                f"{r.get('currency')!r}）；不回退到品种表，先核对来源"
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
                tick_size=tick,
                settlement_type=product.settlement_type,
                currency=currency,
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


def _positive(value):
    """有限正数原样返回，空、NaN、非正数返回 None。"""
    if value is None or value != value or value <= 0:
        return None
    return float(value)


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
        quantity=pl.when(pl.col("quantity") >= INT32_MAX)
        .then(None)
        .otherwise(pl.col("quantity"))
        .cast(pl.Float64),
    ).filter(pl.col("key").is_not_null())
    return _latest_stat(picked, stat_type, "quantity").rename({"value": name})


# 各数据集的场内发布方（Databento ``publisher_id``）。ICE 的数据集另有一个场外成交发布方（venue XOFF，
# 如 IFEU.IMPACT 的 84：大宗、EFP），同一合约同一时段各出一根 K 线，价格可以远离盘面。K 线只留场内的。
ON_EXCHANGE_PUBLISHERS = {"GLBX.MDP3": 1, "IFEU.IMPACT": 57}


def on_exchange_bars(bars, datasets):
    """K 线只留 ``datasets`` 的场内发布方。返回 (K 线, 丢掉的行数)；没有 ``publisher_id`` 列报错。"""
    if "publisher_id" not in bars.columns:
        raise ValueError(
            "K 线缺少 publisher_id 列：分不出场内和场外成交（ICE 数据集两者都有），不能聚合"
        )
    unknown = sorted(set(datasets) - ON_EXCHANGE_PUBLISHERS.keys())
    if unknown:
        raise ValueError(
            f"数据集 {unknown} 的场内发布方没有登记（ON_EXCHANGE_PUBLISHERS）"
        )
    allowed = [ON_EXCHANGE_PUBLISHERS[d] for d in datasets]
    kept = bars.filter(pl.col("publisher_id").cast(pl.Int64).is_in(allowed))
    return kept, bars.height - kept.height


def dated_bars(bars, session, open_days):
    """K 线加交易日列 ``date``；落在开市日集合覆盖范围之外的为空（见 ``trade_dates``）。"""
    bars = bars.sort("ts_event")
    return bars.with_columns(
        date=pl.Series(
            trade_dates(bars["ts_event"], session, open_days), dtype=pl.String
        )
    )


def aggregate_bars(bars, session, open_days):
    """K 线（小时或分钟）→ 按交易日的开高低收与电子盘成交量。``bars`` 需有 symbol 列。

    交易日归不进开市日集合覆盖范围的 K 线（取数窗口两端多取的那几天）丢掉，不并进任何一天。"""
    return _aggregate(dated_bars(bars, session, open_days))


def _aggregate(bars):
    bars = bars.filter(pl.col("date").is_not_null())
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
    require_settlement=True,
    closed_days=None,
):
    """小时线 + 统计 → 统一日线（只留 [start, end] 且在上市期内的行）。返回 (日线, 诊断)。

    ``closed_days`` 是调用方给的已知休市日（见 ``futures_foreign.closed_days_by_exchange``）：有 K 线成交却没有
    结算价的工作日，只有列在里面才按休市处理（K 线并进下一开市日），否则报错。

    ``calendar`` 是覆盖整个取数窗口的**内部**日历（不止 [start, end]）：K 线、持仓量按它归属交易日，
    归不进去的丢掉并计数；``previous_settlement`` 在裁到 [start, end] 之前算，所以 start 当天也有。"""
    if volume not in VOLUMES:
        raise ValueError(f"volume 只能是 {VOLUMES}")
    diag = {}
    bars, diag["off_exchange_bar_rows"] = on_exchange_bars(
        bars, {PRODUCTS[c].databento_dataset for c in set(contracts["product"])}
    )
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
    pieces, undated = [], 0
    allowed = closed_days_by_exchange(closed_days, sorted(set(contracts["exchange"])))
    suspicious, accepted = {}, {}
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
        own_bars = dated_bars(
            keyed["bar"].filter(pl.col("product") == code).drop("product"),
            product.session,
            open_days,
        )
        undated += own_bars["date"].null_count()
        everything = bars_on_closed_days(
            own_bars["ts_event"], product.session, open_days, ()
        )
        for day, n in everything.items():
            bucket = accepted if day in allowed[product.exchange] else suspicious
            target = bucket.setdefault(product.exchange, {})
            target[day] = target.get(day, 0) + n
        merged = _aggregate(own_bars)
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
    diag["bars_outside_calendar"] = undated
    diag["closed_days_with_bars"] = {
        e: dict(sorted(d.items())) for e, d in sorted(accepted.items())
    }
    raise_on_closed_trading_days(suspicious)
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
    long = long.with_columns(pl.col("date").cast(pl.String))
    long = previous_settlement(long, contracts, calendar)
    long = long.filter(pl.col("date").is_between(pl.lit(start), pl.lit(end)))
    long, dropped = inside_listing(long, contracts)
    diag["outside_listing_rows"] = dropped
    long = long.with_columns(
        volume=pl.col("cleared_volume")
        if volume == "cleared"
        else pl.col("bar_volume"),
        amount=pl.lit(None, dtype=pl.Float64),
    )
    diag["settlement_missing"] = check_settlements(
        long, contracts, require=require_settlement
    )
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
        require_settlement=True,
        closed_days=None,
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
        self.require_settlement = require_settlement
        closed_days_by_exchange(  # 先查格式，免得取完数才报错
            closed_days, sorted({PRODUCTS[c].exchange for c in self.products})
        )
        self.closed_days = closed_days

    # ------------------------------------------------------------ 请求计划与估价
    def _groups(self):
        groups = {}
        for code in self.products:
            p = PRODUCTS[code]
            groups.setdefault(p.databento_dataset, []).append(f"{p.databento_root}.FUT")
        return groups

    def plan(self):
        """要发出的请求（dataset, schema, symbols, start, end, stype_in），end 不含。

        - ``definition``：[start−7, end+1)，官方说明从 UTC 0 点起满 24 小时的请求含期间所有在场合约的定义；
          与统计同一起点，统计、K 线里 start 之前几天的记录（含 start 前到期的合约）都对得上合约；
        - ``statistics``：[start−7, end+7)，由结算价推出的开市日要覆盖整个 K 线窗口（start 之前几天的 K 线
          才能归到它们自己的交易日，而不是被顺延进 start），start 当天的 ``previous_settlement`` 也要有；
          持仓量、结算量在下一个 UTC 日或周日发布；
        - 小时线：[start−5, end+1)，覆盖 start 那个交易日从上一开市日傍晚开始的时段（周末加休市日）。"""
        first, last = _day(self.start), _day(self.end)
        out = []
        for dataset, symbols in self._groups().items():
            out.append(
                (
                    dataset,
                    "definition",
                    tuple(symbols),
                    _iso(first - dt.timedelta(days=STAT_LEAD_DAYS)),
                    _iso(last + dt.timedelta(days=1)),
                    "parent",
                )
            )
            out.append(
                (
                    dataset,
                    "statistics",
                    tuple(symbols),
                    _iso(first - dt.timedelta(days=STAT_LEAD_DAYS)),
                    _iso(last + dt.timedelta(days=STAT_LAG_DAYS)),
                    "parent",
                )
            )
            out.append(
                (
                    dataset,
                    self.bar_schema,
                    tuple(symbols),
                    _iso(first - dt.timedelta(days=BAR_LEAD_DAYS)),
                    _iso(last + dt.timedelta(days=1)),
                    "parent",
                )
            )
        return out

    def estimate(self):
        """逐个请求调 ``metadata.get_cost``（不计费），返回 [(请求, 美元)] 与合计。"""
        rows = [
            (
                f"{d} {s} {list(sym)} {a}~{b}",
                self.source.estimate(d, s, sym, a, b, stype_in=st),
            )
            for d, s, sym, a, b, st in self.plan()
        ]
        return dict(requests=rows, total=sum(c for _, c in rows))

    # ------------------------------------------------------------ 取数与换算
    def _compute(self):
        plan = self.plan()
        # 整份计划一次过闸门：先估全部请求的总价，够了才取第一个；取到的在 source 里缓存，重试不重取。
        frames = self.source.fetch_plan(plan)
        raw = {}
        for (_d, schema, *_rest), frame in zip(plan, frames, strict=True):
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
        unknown_listing = contracts.filter(pl.col("listing_date").is_null())["symbol"]
        full = self._calendar(stats, id_map, contracts)
        self._check_window(full)
        daily, daily_diag = daily_from_raw(
            contracts,
            id_map,
            bars,
            stats,
            full,
            self.start,
            self.end,
            volume=self.volume,
            settlement=self.settlement,
            require_settlement=self.require_settlement,
            closed_days=self.closed_days,
        )
        diag.update(daily_diag)
        calendar = full.filter(
            pl.col("date").is_between(pl.lit(self.start), pl.lit(self.end))
        )
        diag["calendar_unknown_days"] = self._unknown_days(full)
        listed = set(daily["symbol"])
        contracts = contracts.filter(pl.col("symbol").is_in(list(listed)))
        diag["unknown_listing_contracts"] = sorted(
            set(unknown_listing.to_list()) & listed
        )
        diag["cost_log"] = list(getattr(self.source, "log", []))
        out = dict(
            contracts=contracts,
            daily=daily,
            calendar=calendar,
            diagnostics=diag,
            _id_map=id_map,
            _calendar=full,
        )
        if self.reference_continuous:
            out.update(self._source_continuous(calendar, id_map, contracts))
        return out

    def _calendar(self, stats, id_map, contracts):
        """内部日历：开市日 = 该交易所有 final 结算价的交易日（来源没有日历接口；整交易所一天都没有结算价
        即视为休市）。范围是统计请求覆盖到的开市日（start 前约一周到 end 后几天）与 [start, end] 的并集；
        输出给用户的日历再裁到 [start, end]。"""
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
            if not days:
                raise ValueError(
                    f"{exchange}: 取数窗口里一个最终结算价（stat_type 3、final 位）都没有，推不出开市日；"
                    "该数据集的 stat_flags / ts_ref 口径可能与 GLBX 不同，先按 docs/FOREIGN_FUTURES.md "
                    "的首次核对清单核对"
                )
            lo = min([self.start, *days])
            hi = max([self.end, *days])
            frames.append(calendar_from_open_days(exchange, days, lo, hi))
        return pl.concat(frames)

    def _unknown_days(self, full):
        """[start, end] 里落在已知开市日范围之外的自然日（比如 end 当天的结算价还没公布）：日历里记为休市，
        但其实不知道。按交易所列出。"""
        out = {}
        for exchange in sorted(set(full["exchange"])):
            opened = full.filter(pl.col("is_open") & (pl.col("exchange") == exchange))[
                "date"
            ]
            if opened.is_empty():
                out[exchange] = [self.start, self.end]
                continue
            lo, hi = opened.min(), opened.max()
            days = full.filter(
                (pl.col("exchange") == exchange)
                & pl.col("date").is_between(pl.lit(self.start), pl.lit(self.end))
                & ((pl.col("date") < lo) | (pl.col("date") > hi))
            )["date"].to_list()
            if days:
                out[exchange] = days
        return out

    def _check_window(self, full):
        """K 线窗口要从 start 的上一开市日起：更早的休市（连着周末超过 ``BAR_LEAD_DAYS`` 天）会让 start 那个
        交易日开头几个小时取不到，开盘价、最低价悄悄变了，所以报错而不是放过。"""
        lead = _day(self.start) - dt.timedelta(days=BAR_LEAD_DAYS)
        for exchange in sorted(set(full["exchange"])):
            before = full.filter(
                pl.col("is_open")
                & (pl.col("exchange") == exchange)
                & (pl.col("date") < self.start)
            )["date"]
            if before.is_empty():
                raise ValueError(
                    f"{exchange}: start {self.start} 之前 {STAT_LEAD_DAYS} 天里没有开市日，"
                    "无法确定 start 那个交易日从哪天开始；把 start 往后挪几天"
                )
            if _day(before.max()) < lead:
                raise ValueError(
                    f"{exchange}: start {self.start} 的上一开市日是 {before.max()}，早于 K 线窗口起点 "
                    f"{lead}；把 start 往后挪几天"
                )

    def _previous_open(self, exchange, day):
        full = self._all()["_calendar"]
        before = full.filter(
            pl.col("is_open")
            & (pl.col("exchange") == exchange)
            & (pl.col("date") < day)
        )["date"]
        if before.is_empty():
            raise ValueError(f"{exchange}: {day} 之前没有已知的开市日")
        return before.max()

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
        """分钟线（``ohlcv-1m``，另走一次费用闸门）。``start``/``end`` 为 YYYYMMDD 交易日（含两端），必须落在
        适配器的 [start, end] 里（日历只覆盖那一段附近，区间外的分钟线归不了交易日）；不在就报错、不估价也不取数。

        取数窗口从 start 的上一开市日 UTC 0 点到 end 次日 UTC 0 点：start 那个交易日从上一开市日傍晚开始。"""
        if not (self.start <= start <= end <= self.end):
            raise ValueError(
                f"minute 的区间 [{start}, {end}] 要落在适配器的区间 [{self.start}, {self.end}] 里"
            )
        tables = self._all()
        contracts, id_map, calendar = (
            tables["contracts"],
            tables["_id_map"],
            tables["_calendar"],
        )
        groups = self._groups()
        exchanges = {
            d: {
                PRODUCTS[c].exchange
                for c in self.products
                if PRODUCTS[c].databento_dataset == d
            }
            for d in groups
        }
        plan = []
        for dataset, symbols in groups.items():
            first = min(self._previous_open(e, start) for e in exchanges[dataset])
            plan.append(
                (
                    dataset,
                    "ohlcv-1m",
                    tuple(symbols),
                    _iso(_day(first)),
                    _iso(_day(end) + dt.timedelta(days=1)),
                    "parent",
                )
            )
        frames = self.source.fetch_plan(plan)
        bars = _drop_mapped(pl.concat(frames, how="diagonal_relaxed")).with_columns(
            pl.col("instrument_id").cast(pl.Int64)
        )
        bars, _ = on_exchange_bars(bars, groups.keys())
        bars = bars.join(id_map, on="instrument_id", how="inner")
        out = []
        for code in sorted(set(contracts["product"])):
            product = PRODUCTS[code]
            members = contracts.filter(pl.col("product") == code)["symbol"].to_list()
            open_days = set(
                calendar.filter(
                    pl.col("is_open") & (pl.col("exchange") == product.exchange)
                )["date"]
            )
            out.append(
                dated_bars(
                    bars.filter(pl.col("symbol").is_in(members)),
                    product.session,
                    open_days,
                )
            )
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
    "ON_EXCHANGE_PUBLISHERS",
    "aggregate_bars",
    "contracts_from_definitions",
    "daily_from_raw",
    "dated_bars",
    "on_exchange_bars",
    "parse_continuous",
    "parse_raw_symbol",
    "quantity_rows",
    "settlement_rows",
]
