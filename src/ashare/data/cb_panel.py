"""可转债面板的字段计算（行情、当时可知的转股价、派生指标）。

原始单位（实测见 docs/FUND_CB_DATA.md）：成交量「手」= 10 张，成交额「万元」；价格为每张
（面值 100 元）的元。派生字段的公式与时点写在 `fund_schema.CB_FIELDS`。
"""

from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from .fund_panel import listing_masks, next_trading_day, scatter

VOLUME_SCALE = 10.0  # 手 → 张
AMOUNT_SCALE = 10000.0  # 万元 → 元
CALL_IMPLEMENTED = "公告实施强赎"


def cb_inputs(store):
    basic = store.read("cb_basic").unique("ts_code", keep="last").sort("ts_code")
    daily = (
        store.read("cb_daily")
        .unique(["ts_code", "trade_date"], keep="last")
        .select(
            pl.col("ts_code").alias("symbol"),
            pl.col("trade_date").alias("date"),
            *(pl.col(c).cast(pl.Float64) for c in ("open", "high", "low", "close")),
            pl.col("pre_close").cast(pl.Float64).alias("previous_close"),
            (pl.col("vol").cast(pl.Float64) * VOLUME_SCALE).alias("volume"),
            (pl.col("amount").cast(pl.Float64) * AMOUNT_SCALE).alias("amount"),
            pl.col("cb_value").cast(pl.Float64).alias("source_conversion_value"),
            pl.col("bond_value").cast(pl.Float64).alias("source_bond_value"),
            (pl.col("bond_over_rate").cast(pl.Float64) / 100).alias(
                "source_bond_premium"
            ),
        )
    )
    # 源在没有成交的日子（上市前几天、摘牌后数月、停牌日）写占位行：成交量 0，开高低为 0，
    # 收盘为 0 或沿用旧价。开高低一律置空（不能当成交价），收盘为 0 的置空、为正的留作估值价；
    # 有成交才算 has_quote。
    traded = pl.col("volume") > 0
    daily = daily.with_columns(
        *(
            pl.when(traded & (pl.col(c) > 0)).then(pl.col(c)).otherwise(None).alias(c)
            for c in ("open", "high", "low")
        ),
        pl.when(pl.col("close") > 0)
        .then(pl.col("close"))
        .otherwise(None)
        .alias("close"),
        traded.fill_null(False).alias("traded"),
    )
    return basic, daily


def cb_securities(basic, symbols):
    keep = basic.filter(pl.col("ts_code").is_in(list(symbols)))
    return keep.select(
        pl.col("ts_code").alias("symbol"),
        pl.col("bond_short_name").alias("name"),
        pl.col("cb_type").alias("kind"),
        pl.col("stk_code").alias("stock_symbol"),
        pl.col("stk_short_name").alias("stock_name"),
        pl.col("list_date").alias("listing_date"),
        pl.col("delist_date").alias("delisting_date"),
        pl.col("value_date").alias("value_date"),
        pl.col("maturity_date").alias("maturity_date"),
        pl.col("par").cast(pl.Float64).alias("par"),
        pl.col("issue_size").cast(pl.Float64).alias("issue_size"),
        pl.col("first_conv_price").cast(pl.Float64).alias("initial_conversion_price"),
        pl.col("conv_start_date").alias("conversion_start"),
        pl.col("conv_end_date").alias("conversion_end"),
        pl.col("exchange").alias("exchange"),
    ).sort("symbol")


def _step(dates, symbols, events, initial, start_rows):
    """按「可用行」把每个代码的取值向前填：先放初始值，再按行号依次覆盖。

    events：(symbol, row, value) 的 DataFrame，row 是从哪一行起用新值；同一行多条按给出顺序取最后一条。
    """
    t, n = len(dates), len(symbols)
    out = np.full((t, n), np.nan)
    index = {s: j for j, s in enumerate(symbols)}
    per = {}
    for s, row, value in events.iter_rows():
        if s in index and value is not None and np.isfinite(value) and row < t:
            per.setdefault(index[s], []).append((int(row), float(value)))
    every = np.arange(t)
    for j in range(n):
        changes = sorted(
            per.get(j, []), key=lambda x: x[0]
        )  # 稳定排序：同一行保留给出顺序
        rows = np.array([c[0] for c in changes], dtype=np.int64)
        values = np.array([c[1] for c in changes])
        position = np.searchsorted(rows, every, side="right") - 1
        column = np.where(
            position >= 0,
            values[np.maximum(position, 0)] if len(values) else np.nan,
            initial[j],
        )
        column[every < int(start_rows[j])] = np.nan
        out[:, j] = column
    return out


def conversion_price(store, dates, symbols, basic, start_rows):
    """当时可知的转股价。初始值取上市时的初始转股价；每条调整自 max(生效日, 公告日的下一交易日) 起。"""
    raw = store.read("cb_price_chg").unique(
        ["ts_code", "publish_date", "change_date", "convertprice_aft"], keep="last"
    )
    raw = raw.filter(pl.col("convertprice_aft").is_not_null())
    effective = next_trading_day(dates, raw["change_date"].to_list())
    known = next_trading_day(dates, raw["publish_date"].to_list(), strictly_after=True)
    raw = raw.with_columns(
        pl.Series("row", np.maximum(effective, known)),
    ).sort("ts_code", "row", "change_date", "publish_date")
    events = raw.select("ts_code", "row", pl.col("convertprice_aft").cast(pl.Float64))
    info = {r["ts_code"]: r for r in basic.iter_rows(named=True)}
    initial = np.array(
        [
            float(info[s]["first_conv_price"])
            if s in info and info[s]["first_conv_price"] is not None
            else np.nan
            for s in symbols
        ]
    )
    return _step(dates, symbols, events, initial, start_rows), int(len(raw))


def remaining_size(store, dates, symbols, basic, start_rows):
    """剩余规模：转股结果公告的下一交易日起用新值；首次公告前取发行规模。"""
    raw = store.read("cb_share").unique(
        ["ts_code", "end_date", "publish_date"], keep="last"
    )
    raw = raw.filter(
        pl.col("remain_size").is_not_null() & pl.col("publish_date").is_not_null()
    )
    known = next_trading_day(dates, raw["publish_date"].to_list(), strictly_after=True)
    raw = raw.with_columns(pl.Series("row", known)).sort("ts_code", "row", "end_date")
    events = raw.select("ts_code", "row", pl.col("remain_size").cast(pl.Float64))
    info = {r["ts_code"]: r for r in basic.iter_rows(named=True)}
    initial = np.array(
        [
            float(info[s]["issue_size"])
            if s in info and info[s]["issue_size"] is not None
            else np.nan
            for s in symbols
        ]
    )
    return _step(dates, symbols, events, initial, start_rows)


def call_flags(store, dates, symbols, is_listed):
    """强赎实施公告的下一交易日起为真（直到摘牌）。另返回强赎与到期赎回的事件表。"""
    raw = store.read("cb_call").unique(
        ["ts_code", "call_type", "is_call", "ann_date"], keep="last"
    )
    calls = (
        raw.filter(pl.col("is_call") == CALL_IMPLEMENTED)
        .group_by("ts_code")
        .agg(pl.col("ann_date").min())
    )
    flags = np.zeros((len(dates), len(symbols)), dtype=bool)
    rows = next_trading_day(dates, calls["ann_date"].to_list(), strictly_after=True)
    cols = np.searchsorted(symbols, calls["ts_code"].to_numpy().astype(symbols.dtype))
    for row, col, code in zip(rows, cols, calls["ts_code"].to_list(), strict=True):
        if col < len(symbols) and symbols[col] == code and row < len(dates):
            flags[row:, col] = True
    flags &= is_listed
    table = raw.select(
        pl.col("ts_code").alias("symbol"),
        pl.col("call_type").alias("kind"),
        pl.col("is_call").alias("status"),
        pl.col("ann_date").alias("announcement_date"),
        pl.col("call_date").alias("redemption_date"),
        pl.col("call_reg_date").alias("record_date"),
        pl.col("call_price").cast(pl.Float64).alias("redemption_price"),
        pl.col("call_price_tax").cast(pl.Float64).alias("redemption_price_after_tax"),
        pl.col("payment_date").alias("payment_date"),
    ).sort("symbol", "announcement_date")
    return flags, table


def coupon_rates(store, dates, symbols):
    raw = store.read("cb_rate").unique(["ts_code", "rate_start_date"], keep="last")
    out = np.full((len(dates), len(symbols)), np.nan)
    index = {s: j for j, s in enumerate(symbols)}
    for code, start, end, rate in raw.select(
        "ts_code",
        "rate_start_date",
        "rate_end_date",
        pl.col("coupon_rate").cast(pl.Float64),
    ).iter_rows():
        if code not in index or start is None or rate is None:
            continue
        lo = np.searchsorted(dates, start)
        hi = np.searchsorted(dates, end, side="right") if end else len(dates)
        out[lo:hi, index[code]] = rate / 100
    return out


def days_to_maturity(dates, symbols, basic):
    info = {r["ts_code"]: r["maturity_date"] for r in basic.iter_rows(named=True)}
    ordinal = np.array(
        [date(int(d[:4]), int(d[4:6]), int(d[6:])).toordinal() for d in dates],
        dtype=float,
    )
    out = np.full((len(dates), len(symbols)), np.nan)
    for j, s in enumerate(symbols):
        m = info.get(s)
        if m:
            out[:, j] = date(int(m[:4]), int(m[4:6]), int(m[6:])).toordinal() - ordinal
    return out


def stock_closes(stock_root, dates, stock_symbols):
    """正股收盘价（不复权）。读不到股票面板时返回 (全 NaN, 原因)。"""
    out = np.full((len(dates), len(stock_symbols)), np.nan)
    try:
        import ashare.data as ad

        data = ad.open(stock_root)
        panel_dates = np.array(data._meta["dates"], dtype="U8") if data._meta else None
        if panel_dates is None:
            return out, dict(stock_panel=None, reason="股票面板未构建")
        codes = set(data._meta["symbols"])
        wanted = sorted({s for s in stock_symbols if s in codes})
        first = max(str(dates[0]), str(panel_dates[0]))
        last = min(str(dates[-1]), str(panel_dates[-1]))
        if not wanted or first > last:
            return out, dict(stock_panel=data.version, reason="区间或代码不重叠")
        p = data.panel(["close"], start=first, end=last, symbols=wanted)
        close = np.asarray(p["close"])
        rows = np.searchsorted(dates, p.dates)
        ok = (rows < len(dates)) & (dates[np.minimum(rows, len(dates) - 1)] == p.dates)
        col_of = {s: k for k, s in enumerate(wanted)}
        for j, s in enumerate(stock_symbols):
            if s in col_of:
                out[rows[ok], j] = close[ok, col_of[s]]
        return out, dict(
            stock_panel=data.version,
            stock_first=first,
            stock_last=last,
            missing_stocks=sorted({s for s in stock_symbols if s and s not in codes}),
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        return out, dict(stock_panel=None, reason=f"{type(exc).__name__}: {exc}")


def build_cb_arrays(store, dates, *, stock_root=None):
    basic, daily = cb_inputs(store)
    window = daily.filter(
        pl.col("date").is_between(pl.lit(dates[0]), pl.lit(dates[-1]))
    )
    traded = set(window["symbol"].unique().to_list())
    known = set(basic["ts_code"].to_list())
    listed = basic.filter(
        pl.col("list_date").is_not_null()
        & (pl.col("list_date") <= dates[-1])
        & (pl.col("delist_date").is_null() | (pl.col("delist_date") > dates[0]))
    )["ts_code"].to_list()
    symbols = np.array(sorted((traded & known) | set(listed)))
    diagnostics = dict(daily_codes_not_in_basic=sorted(traded - known))
    arrays = {}
    for name in (
        "open",
        "high",
        "low",
        "close",
        "previous_close",
        "volume",
        "amount",
        "source_conversion_value",
        "source_bond_value",
        "source_bond_premium",
    ):
        arrays[name] = scatter(dates, symbols, window, name)
    traded_cells = scatter(dates, symbols, window, "traded", dtype=bool, fill=False)
    arrays["has_quote"] = traded_cells & np.isfinite(arrays["close"])
    securities = cb_securities(basic, symbols)
    info = {r["symbol"]: r for r in securities.iter_rows(named=True)}
    listing = [info[s]["listing_date"] if s in info else None for s in symbols]
    delisting = [info[s]["delisting_date"] if s in info else None for s in symbols]
    arrays["is_listed"], arrays["delist"] = listing_masks(dates, listing, delisting)
    start_rows = next_trading_day(dates, listing)
    arrays["conversion_price"], diagnostics["conversion_price_changes"] = (
        conversion_price(store, dates, symbols, basic, start_rows)
    )
    arrays["conversion_price"][~arrays["is_listed"]] = np.nan
    arrays["remaining_size"] = remaining_size(store, dates, symbols, basic, start_rows)
    arrays["remaining_size"][~arrays["is_listed"]] = np.nan
    arrays["call_announced"], calls = call_flags(
        store, dates, symbols, arrays["is_listed"]
    )
    arrays["coupon_rate"] = coupon_rates(store, dates, symbols)
    arrays["days_to_maturity"] = days_to_maturity(dates, symbols, basic)
    arrays["days_to_maturity"][~arrays["is_listed"]] = np.nan
    stocks = [info[s]["stock_symbol"] if s in info else None for s in symbols]
    unique_stocks = sorted({s for s in stocks if s})
    closes, stock_info = stock_closes(
        stock_root or store.root.parent, dates, unique_stocks
    )
    diagnostics.update(stock_info)
    diagnostics["stock_panel_version"] = stock_info.get("stock_panel")
    col = {s: k for k, s in enumerate(unique_stocks)}
    stock_close = np.full((len(dates), len(symbols)), np.nan)
    for j, s in enumerate(stocks):
        if s in col:
            stock_close[:, j] = closes[:, col[s]]
    stock_close[~arrays["is_listed"]] = np.nan
    arrays["stock_close"] = stock_close
    with np.errstate(invalid="ignore", divide="ignore"):
        value = 100.0 / arrays["conversion_price"] * stock_close
        premium = arrays["close"] / value - 1
    arrays["conversion_value"] = value
    arrays["conversion_premium"] = premium
    arrays["double_low"] = arrays["close"] + 100 * premium
    tables = {"calls": calls.filter(pl.col("symbol").is_in(list(symbols)))}
    diagnostics["calls_announced"] = int(arrays["call_announced"].any(axis=0).sum())
    return symbols, arrays, securities, tables, diagnostics
