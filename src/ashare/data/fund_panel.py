"""场内基金与可转债的「交易日 × 证券」面板。

存储：``<root>/cn/fund_panel/<kind>/<版本>/``（kind 为 ``fund`` 或 ``cb``），``current.json``
指向当前版本，与股票面板同一套代次写法。每个字段一个 T×N 的 ``.npy``（可内存映射），另有
``securities.parquet``（证券表）、``metadata.json``（日期、代码、字段说明、诊断）和几张事件表。

体量：2015 年以来约 2900 个交易日 × 约 3000 只基金，每个浮点字段约 70 MB；可转债约 1200 只。
稠密矩阵足够，不需要期货那样分块。

时点规则全部在这里落地（`docs/FUND_CB_DATA.md`）：
- 基金分红按除息日（实施公告早于除息日）；拆分、折算由交易所前收盘反推，只在复权因子也变了的
  日子认定。
- 可转债的转股价、剩余规模、强赎公告都按「公告日的下一交易日」起可知；转股价另不早于生效日。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import uuid
from pathlib import Path

import numpy as np
import polars as pl

from .fund_schema import CB_FIELDS, FUND_FIELDS
from .store import atomic_json, lock

FORMAT_VERSION = 1
CODE_FILES = ("fund_schema.py", "fund_panel.py", "cb_panel.py")
SPLIT_TOLERANCE = 0.005  # 拆分、折算的最小幅度；小于它的前收盘差异按舍入处理


def code_version():
    here = Path(__file__).parent
    digest = hashlib.sha256()
    for name in CODE_FILES:
        digest.update((here / name).read_bytes())
    return digest.hexdigest()[:16]


def complete_partitions(store, name):
    return sorted(
        e["partition"]
        for e in store.entries(name).values()
        if e["status"] == "complete"
    )


def calendar_days(store, first, last):
    frame = store.read("fund_calendar")
    return np.array(
        sorted(
            set(
                frame.filter(
                    (pl.col("exchange") == "SSE")
                    & (pl.col("is_open").cast(pl.Int64) == 1)
                    & pl.col("cal_date").is_between(pl.lit(first), pl.lit(last))
                )["cal_date"].to_list()
            )
        ),
        dtype="U8",
    )


def next_trading_day(dates, values, *, strictly_after=False):
    """日期 → 主日历行号：当天或之后的第一个交易日（strictly_after 时严格之后）；超出末尾为 T。"""
    v = np.asarray([x if x is not None else "99999999" for x in values], dtype="U8")
    return np.searchsorted(dates, v, side="right" if strictly_after else "left")


def scatter(dates, symbols, frame, column, dtype=np.float64, fill=np.nan):
    """长表 (symbol, date, value) → T×N；日期不在主日历或代码不在证券表的行丢掉。"""
    out = np.full((len(dates), len(symbols)), fill, dtype=dtype)
    if not len(frame):
        return out
    rows = np.searchsorted(dates, frame["date"].to_numpy().astype("U8"))
    cols = np.searchsorted(symbols, frame["symbol"].to_numpy().astype(symbols.dtype))
    ok = (rows < len(dates)) & (cols < len(symbols))
    ok[ok] &= (dates[rows[ok]] == frame["date"].to_numpy().astype("U8")[ok]) & (
        symbols[cols[ok]] == frame["symbol"].to_numpy().astype(symbols.dtype)[ok]
    )
    values = frame[column].to_numpy()
    out[rows[ok], cols[ok]] = values[ok]
    return out


def listing_masks(dates, listing, delisting):
    """is_listed（上市日含、退市日不含）与 delist（退市日或其后首个交易日）。"""
    t, n = len(dates), len(listing)
    start = next_trading_day(dates, listing)
    stop = next_trading_day(dates, delisting)
    rows = np.arange(t)[:, None]
    is_listed = (rows >= start[None, :]) & (rows < stop[None, :])
    delist = np.zeros((t, n), dtype=bool)
    has = np.array([d is not None for d in delisting]) & (stop < t)
    delist[stop[has], np.flatnonzero(has)] = True
    return is_listed, delist


def fund_inputs(store):
    """基金原始表 → 统一长表（单位已换算）；原始字段名只在这里。"""
    basic = store.read("fund_basic").unique("ts_code", keep="last").sort("ts_code")
    daily = (
        store.read("fund_daily")
        .unique(["ts_code", "trade_date"], keep="last")
        .select(
            pl.col("ts_code").alias("symbol"),
            pl.col("trade_date").alias("date"),
            *(pl.col(c).cast(pl.Float64) for c in ("open", "high", "low", "close")),
            pl.col("pre_close").cast(pl.Float64).alias("previous_close"),
            (pl.col("vol").cast(pl.Float64) * 100).alias("volume"),
            (pl.col("amount").cast(pl.Float64) * 1000).alias("amount"),
        )
    )
    adj = (
        store.read("fund_adj")
        .unique(["ts_code", "trade_date"], keep="last")
        .select(
            pl.col("ts_code").alias("symbol"),
            pl.col("trade_date").alias("date"),
            pl.col("adj_factor").cast(pl.Float64),
        )
    )
    share = (
        store.read("fund_share")
        .unique(["ts_code", "trade_date"], keep="last")
        .select(
            pl.col("ts_code").alias("symbol"),
            pl.col("trade_date").alias("date"),
            (pl.col("fd_share").cast(pl.Float64) * 10000).alias("total_shares"),
        )
    )
    dividend = (
        store.read("fund_div")
        .filter((pl.col("div_proc") == "实施") & pl.col("ex_date").is_not_null())
        .unique(["ts_code", "ex_date", "div_cash"], keep="last")
        .group_by("ts_code", "ex_date")
        .agg(pl.col("div_cash").cast(pl.Float64).sum())
        .select(
            pl.col("ts_code").alias("symbol"),
            pl.col("ex_date").alias("date"),
            pl.col("div_cash").alias("cash_dividend"),
        )
    )
    return basic, daily, adj, share, dividend


def fund_securities(basic, symbols):
    keep = basic.filter(pl.col("ts_code").is_in(list(symbols)))
    return keep.select(
        pl.col("ts_code").alias("symbol"),
        "name",
        pl.col("fund_type").alias("fund_type"),
        pl.col("invest_type").alias("invest_type"),
        pl.col("type").alias("category"),
        pl.col("management").alias("manager"),
        pl.col("benchmark").alias("benchmark"),
        pl.col("list_date").alias("listing_date"),
        pl.col("delist_date").alias("delisting_date"),
        pl.col("found_date").alias("found_date"),
        pl.col("m_fee").cast(pl.Float64).truediv(100).alias("management_fee"),
        pl.col("c_fee").cast(pl.Float64).truediv(100).alias("custody_fee"),
        pl.col("status").alias("status"),
    ).sort("symbol")


def fund_actions(close, previous_close, adj, dividend):
    """拆分 / 折算倍数（split）。

    G = 上一有效收盘 / 交易所前收盘，D = 上一有效收盘 / (上一有效收盘 − 分红)，split = G / D。
    只在复权因子当天也变了（与上一有效值相差超过 SPLIT_TOLERANCE）、且 |G − 1| 超过同一容差的
    格子认定；其余为 1。交易所没调整前收盘的分红（部分 REITs）只记现金、不当拆分。
    """
    t, n = close.shape
    split = np.ones((t, n))
    last_close = np.full(n, np.nan)
    last_adj = np.full(n, np.nan)
    for row in range(t):
        g = last_close / previous_close[row]
        r = adj[row] / last_adj
        d = np.where(dividend[row] > 0, last_close / (last_close - dividend[row]), 1.0)
        event = (
            np.isfinite(g)
            & np.isfinite(r)
            & (np.abs(r - 1) > SPLIT_TOLERANCE)
            & (np.abs(g - 1) > SPLIT_TOLERANCE)
            & (d > 0)
        )
        value = g / d
        event &= np.abs(value - 1) > SPLIT_TOLERANCE
        split[row, event] = value[event]
        ok = np.isfinite(close[row])
        last_close[ok] = close[row, ok]
        okadj = np.isfinite(adj[row])
        last_adj[okadj] = adj[row, okadj]
    return split


def build_fund_arrays(store, dates):
    basic, daily, adj, share, dividend = fund_inputs(store)
    traded = set(
        daily.filter(pl.col("date").is_between(pl.lit(dates[0]), pl.lit(dates[-1])))[
            "symbol"
        ]
        .unique()
        .to_list()
    )
    known = set(basic["ts_code"].to_list())
    listed = basic.filter(
        pl.col("list_date").is_not_null()
        & (pl.col("list_date") <= dates[-1])
        & (pl.col("delist_date").is_null() | (pl.col("delist_date") > dates[0]))
    )["ts_code"].to_list()
    symbols = np.array(sorted((traded & known) | set(listed)))
    diagnostics = dict(
        daily_codes_not_in_basic=sorted(traded - known),
        reused_code_markers=sorted(
            {c for c in adj["symbol"].unique().to_list() if "!" in c}
        ),
    )
    arrays = {}
    for name in ("open", "high", "low", "close", "previous_close", "volume", "amount"):
        arrays[name] = scatter(dates, symbols, daily, name)
    arrays["adj_factor"] = scatter(dates, symbols, adj, "adj_factor")
    arrays["total_shares"] = scatter(dates, symbols, share, "total_shares")
    # 分红按除息日；除息日不是交易日时落到其后第一个交易日。
    div = dividend.filter(pl.col("symbol").is_in(list(symbols)))
    rows = next_trading_day(dates, div["date"].to_list())
    cols = np.searchsorted(symbols, div["symbol"].to_numpy().astype(symbols.dtype))
    cash = np.zeros((len(dates), len(symbols)))
    ok = rows < len(dates)
    np.add.at(cash, (rows[ok], cols[ok]), div["cash_dividend"].to_numpy()[ok])
    arrays["cash_dividend"] = cash
    arrays["has_quote"] = np.isfinite(arrays["close"]) & (arrays["close"] > 0)
    securities = fund_securities(basic, symbols)
    info = {r["symbol"]: r for r in securities.iter_rows(named=True)}
    listing = [info[s]["listing_date"] if s in info else None for s in symbols]
    delisting = [info[s]["delisting_date"] if s in info else None for s in symbols]
    # 上市日取「证券表上市日」与「第一个有行情的日子」中较早的一个：分级基金转型、LOF 份额合并后
    # 证券表的上市日被改成转型日（如 2021-01-01），之前几年的行情仍是同一只场内基金。逐只列在诊断里。
    first_quote = np.where(
        arrays["has_quote"].any(axis=0), arrays["has_quote"].argmax(axis=0), len(dates)
    )
    moved = []
    for j, lst in enumerate(listing):
        quoted = dates[first_quote[j]] if first_quote[j] < len(dates) else None
        if quoted is not None and (lst is None or quoted < lst):
            if lst is not None:
                moved.append(
                    dict(
                        symbol=str(symbols[j]),
                        listing_date=lst,
                        first_quote=str(quoted),
                    )
                )
            listing[j] = str(quoted)
    diagnostics["listing_before_table_date"] = moved
    arrays["is_listed"], arrays["delist"] = listing_masks(dates, listing, delisting)
    arrays["split"] = fund_actions(
        arrays["close"], arrays["previous_close"], arrays["adj_factor"], cash
    )
    diagnostics["splits"] = int((arrays["split"] != 1).sum())
    diagnostics["dividends"] = int((cash > 0).sum())
    tables = {"dividends": div.sort("symbol", "date")}
    return symbols, arrays, securities, tables, diagnostics


def build_funds(
    store, *, stock_root=None, kinds=("fund", "cb"), start=None, end=None, keep=2
):
    """构建基金与可转债面板；返回每种面板的版本、形状、耗时。

    stock_root：正股收盘价所在的数据根（算可转债转股价值，读股票面板 `close`），默认与 store 同根；
    读不到股票面板时转股价值整列为 NaN，元数据里记下原因。
    """
    from .cb_panel import build_cb_arrays

    result = {}
    with lock(store.root / ".fund_build.lock"):
        for kind in kinds:
            started = time.perf_counter()
            source = "fund_daily" if kind == "fund" else "cb_daily"
            parts = complete_partitions(store, source)
            if not parts:
                raise FileNotFoundError(
                    f"缺少 {source}；先运行 python -m ashare.data sync-funds"
                )
            first = max(start or parts[0], parts[0])
            last = min(end or parts[-1], parts[-1])
            dates = calendar_days(store, first, last)
            if kind == "fund":
                symbols, arrays, securities, tables, diagnostics = build_fund_arrays(
                    store, dates
                )
                fields = FUND_FIELDS
            else:
                symbols, arrays, securities, tables, diagnostics = build_cb_arrays(
                    store, dates, stock_root=stock_root
                )
                fields = CB_FIELDS
            meta = write_panel(
                store,
                kind,
                dates,
                symbols,
                arrays,
                securities,
                tables,
                diagnostics,
                fields,
                keep=keep,
            )
            meta["elapsed_seconds"] = round(time.perf_counter() - started, 2)
            result[kind] = {k: meta[k] for k in ("version", "shape", "elapsed_seconds")}
    return result


def write_panel(
    store,
    kind,
    dates,
    symbols,
    arrays,
    securities,
    tables,
    diagnostics,
    fields,
    *,
    keep=2,
):
    base = store.root / "fund_panel" / kind
    base.mkdir(parents=True, exist_ok=True)
    inputs = {
        k: v["content_hash"]
        for k, v in sorted(store.manifest().items())
        if v.get("status") == "complete"
        and v.get("dataset", "").startswith(("fund_", "cb_"))
    }
    version = hashlib.sha256(
        json.dumps(
            dict(
                inputs=inputs,
                kind=kind,
                first=str(dates[0]),
                last=str(dates[-1]),
                code=code_version(),
                format=FORMAT_VERSION,
                stock=diagnostics.get("stock_panel_version"),
            ),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    target = base / version
    if not target.exists():
        temp = base / f".building-{uuid.uuid4().hex}"
        temp.mkdir()
        try:
            for name, spec in fields.items():
                np.save(
                    temp / f"{name}.npy",
                    np.ascontiguousarray(arrays[name], dtype=spec.dtype),
                )
            securities.write_parquet(temp / "securities.parquet")
            for name, table in tables.items():
                table.write_parquet(temp / f"table_{name}.parquet")
            atomic_json(
                temp / "metadata.json",
                dict(
                    version=version,
                    kind=kind,
                    dates=[str(d) for d in dates],
                    symbols=[str(s) for s in symbols],
                    fields={
                        n: dict(unit=f.unit, dtype=f.dtype, known_at=f.known_at)
                        for n, f in fields.items()
                    },
                    tables=sorted(tables),
                    diagnostics=diagnostics,
                    raw_version=store.fund_version,
                    code=code_version(),
                    format=FORMAT_VERSION,
                ),
            )
            temp.rename(target)
        finally:
            if temp.exists():
                shutil.rmtree(temp)
    atomic_json(base / "current.json", {"version": version})
    old = sorted(
        (
            p
            for p in base.iterdir()
            if p.is_dir() and not p.name.startswith(".") and p.name != version
        ),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for path in old[max(0, keep - 1) :]:
        shutil.rmtree(path)
    return dict(version=version, shape=[len(dates), len(symbols)])
