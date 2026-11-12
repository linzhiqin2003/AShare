"""场内基金与可转债原始库、面板的独立校验。

原始库部分只读清单与 Parquet，单位、日历、复权关系都在这里另写一遍，不调用同步和面板构建代码；
面板部分读构建好的 ``fund_panel/<kind>/<版本>/``，用原始表重算后逐格比较。

每项结论（与 `futures_verify` 相同）：``pass`` 没有不符；``explained`` 有不符但每一条都在已知例外
清单（``fund_known_exceptions.json``，逐条写代码、日期、原因编号）里；``fail`` 有清单之外的不符，
或不允许例外的检查（面板重算）有不符；``info`` 只给统计、没有对错。清单只放匹配键与原因，不放
行情算出的数值。总结论为 fail 时命令行 ``verify-funds`` 以非零状态退出。

用法::

    from ashare.data.store import Store
    from ashare.data.fund_verify import verify_funds
    report = verify_funds(Store("/path/to/root", readonly=True))
    report["verdict"], report["checks"]["fund_adj_vs_gap"]["verdict"]
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl

from .store import atomic_json

KNOWN_PATH = Path(__file__).with_name("fund_known_exceptions.json")
DAY_SETS = ("fund_daily", "fund_adj", "fund_share", "fund_div", "cb_daily")
KEYS = {
    "fund_basic": ("ts_code",),
    "fund_daily": ("ts_code", "trade_date"),
    "fund_adj": ("ts_code", "trade_date"),
    "fund_share": ("ts_code", "trade_date"),
    "cb_basic": ("ts_code",),
    "cb_daily": ("ts_code", "trade_date"),
    "cb_rate": ("ts_code", "rate_start_date"),
}
EVENT = 0.005  # 拆分、折算、分红的最小幅度（与面板一致，但这里另写）
MATCH = 0.002  # 复权因子变化与前收盘缺口的一致容差
REGISTER_KEYS = {
    "fund_vwap": ("symbol", "date"),
    "cb_vwap": ("symbol", "date"),
    "fund_adj_vs_gap": ("symbol", "date"),
    "cb_traded_outside_life": ("symbol", "date"),
    "fund_code_not_in_table": ("symbol",),
    "cb_conversion_value_source": ("symbol", "date"),
    "fund_price_limits": ("symbol", "date"),
    "partition_missing": ("dataset", "partition"),
}


def load_known(path=KNOWN_PATH):
    if path is None or not Path(path).exists():
        return dict(reasons={}, entries=[], cause_unknown=[])
    return json.loads(Path(path).read_text())


class Register:
    """按检查名与匹配键精确查已知例外；记下用到的条目。"""

    def __init__(self, known):
        self.exact = {}
        for k, e in enumerate(known.get("entries", [])):
            fields = REGISTER_KEYS[e["check"]]
            self.exact[(e["check"], *(e[f] for f in fields))] = k
        self.used = set()

    def hit(self, check, *key):
        k = self.exact.get((check, *key))
        if k is not None:
            self.used.add(k)
        return k

    def stale(self, known):
        entries = known.get("entries", [])
        return [entries[k] for k in range(len(entries)) if k not in self.used]

    def cause_unknown(self, known):
        unknown = set(known.get("cause_unknown", []))
        entries = known.get("entries", [])
        hits = [
            entries[k] for k in sorted(self.used) if entries[k]["reason"] in unknown
        ]
        return dict(count=len(hits), by_reason=dict(Counter(e["reason"] for e in hits)))


def _check(verdict, checked, failed, details=None, note="", explained=0):
    return dict(
        verdict=verdict,
        checked=int(checked),
        failed=int(failed),
        explained=int(explained),
        details=details if details is not None else {},
        note=note,
    )


def _judge(problems, register, check, checked, note="", extra=None):
    fields = REGISTER_KEYS[check]
    unexplained = [
        p for p in problems if register.hit(check, *(p[f] for f in fields)) is None
    ]
    verdict = "fail" if unexplained else ("explained" if problems else "pass")
    details = dict(unexplained=unexplained[:500], unexplained_count=len(unexplained))
    if extra:
        details.update(extra)
    return _check(
        verdict,
        checked,
        len(unexplained),
        details,
        note,
        len(problems) - len(unexplained),
    )


def _strict(problems, checked, note="", extra=None):
    details = dict(examples=problems[:20], count=len(problems))
    if extra:
        details.update(extra)
    return _check("fail" if problems else "pass", checked, len(problems), details, note)


def _entries(store, name):
    return {k: v for k, v in store.manifest().items() if v.get("dataset") == name}


def _read(store, name):
    frames = []
    for v in _entries(store, name).values():
        if v.get("status") == "complete" and v.get("rows"):
            try:
                frames.append(pl.read_parquet(store.root / v["file"]))
            except Exception:  # noqa: BLE001 - 损坏文件由 manifest_files 报告
                continue
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


# ------------------------------------------------------------ 原始库


def check_manifest(store):
    bad, checked = [], 0
    for key, entry in sorted(store.manifest().items()):
        if not (entry.get("dataset", "").startswith(("fund_", "cb_"))):
            continue
        checked += 1
        path = store.root / entry.get("file", "missing")
        ok = (
            entry.get("status") == "complete"
            and path.is_file()
            and hashlib.sha256(path.read_bytes()).hexdigest() == entry.get("file_hash")
        )
        if not ok:
            bad.append(dict(key=key, status=entry.get("status")))
    return _strict(bad, checked, "每个基金、可转债分区为完成状态，文件存在且哈希一致")


def open_days(store):
    cal = _read(store, "fund_calendar")
    return np.array(
        sorted(
            cal.filter(
                (pl.col("exchange") == "SSE") & (pl.col("is_open").cast(pl.Int64) == 1)
            )["cal_date"].to_list()
        ),
        dtype="U8",
    )


def check_partitions(store, days, register):
    out = {}
    for name in DAY_SETS:
        entries = _entries(store, name)
        parts = sorted(e["partition"] for e in entries.values())
        if not parts:
            out[name] = _check("fail", 0, 1, {}, "没有任何分区")
            continue
        window = days[(days >= parts[0]) & (days <= parts[-1])]
        status = {e["partition"]: e for e in entries.values()}
        problems = [
            dict(
                dataset=name,
                partition=str(d),
                status=status.get(str(d), {}).get("status", "missing"),
            )
            for d in window
            if status.get(str(d), {}).get("status") != "complete"
        ]
        out[name] = _judge(
            problems,
            register,
            "partition_missing",
            len(window),
            f"{parts[0]}–{parts[-1]} 间每个上交所开市日都有完成的分区",
        )
    return out


def check_unique(store):
    problems, checked = [], 0
    for name, keys in KEYS.items():
        frame = _read(store, name)
        if not len(frame):
            continue
        checked += len(frame)
        dup = frame.filter(pl.struct(list(keys)).is_duplicated())
        if len(dup):
            problems.append(
                dict(
                    dataset=name,
                    rows=len(dup),
                    example=dup.select(list(keys)).head(3).to_dicts(),
                )
            )
    return _strict(
        problems,
        checked,
        "主键唯一（基金分红、可转债事件表的主键见 catalog，可有同日多条）",
    )


def check_vwap(daily, register, *, check, volume_scale, amount_scale):
    """成交额 ÷ 成交量 折算的均价必须落在当天最高最低价之间（容差：一个价位 + 成交额截断到 1 元）。

    这同时确认了成交量、成交额的单位：单位错一个数量级，几乎每一行都会落到区间外。
    """
    traded = daily.filter(
        (pl.col("vol") > 0) & (pl.col("high") > 0) & (pl.col("low") > 0)
    )
    shares = pl.col("vol") * volume_scale
    frame = traded.with_columns(
        vwap=pl.col("amount") * amount_scale / shares,
        tol=0.0015 + 1.0 / shares,
    )
    bad = frame.filter(
        (pl.col("vwap") < pl.col("low") - pl.col("tol"))
        | (pl.col("vwap") > pl.col("high") + pl.col("tol"))
    )
    problems = [
        dict(symbol=r[0], date=r[1])
        for r in bad.select("ts_code", "trade_date").iter_rows()
    ]
    ratio = frame.select(
        (pl.col("amount") * amount_scale / (shares * pl.col("close"))).median()
    ).item()
    return _judge(
        problems,
        register,
        check,
        len(frame),
        f"成交额折算均价在 [最低, 最高] 内；按成交量 ×{volume_scale:g}、成交额 ×{amount_scale:g} 换算",
        extra=dict(median_amount_over_volume_close=ratio),
    )


def check_zero_rows(daily):
    """可转债的无成交占位行（成交量为 0）计数，以及成交量为正但价格为 0 的行（只统计）。"""
    zero = daily.filter(pl.col("vol") == 0)
    odd = daily.filter(
        (pl.col("vol") > 0) & ((pl.col("open") <= 0) | (pl.col("close") <= 0))
    )
    return _check(
        "info",
        len(daily),
        0,
        dict(
            zero_volume_rows=len(zero),
            zero_volume_rows_with_prices=len(zero.filter(pl.col("open") > 0)),
            positive_volume_zero_price=odd.select("ts_code", "trade_date")
            .head(20)
            .to_dicts(),
            positive_volume_zero_price_count=len(odd),
        ),
        "成交量为 0 的行是源的占位行（上市前、摘牌后、停牌日），面板里开高低置空、不算有行情",
    )


def consecutive(daily, days):
    """每只基金相邻两个交易日都有行情的行：上一日收盘、上一日复权因子。"""
    index = pl.DataFrame({"trade_date": days.tolist(), "k": np.arange(len(days))})
    x = daily.join(index, on="trade_date").sort("ts_code", "trade_date")
    x = x.with_columns(
        last_close=pl.col("close").shift(1).over("ts_code"),
        last_k=pl.col("k").shift(1).over("ts_code"),
        last_adj=pl.col("adj_factor").shift(1).over("ts_code"),
    )
    return x.filter(pl.col("last_k") == pl.col("k") - 1)


def check_adj_vs_gap(daily, adj, dividend, days, register):
    """拆分、折算、分红是否体现在复权因子里。

    G = 上一日收盘 / 交易所前收盘（交易所为除权除息给的参考价缺口），R = 复权因子的变化。
    只看相邻两个交易日都有行情的行；|G−1| 或 |R−1| 超过 0.5% 的行，要求 |R/G − 1| < 0.2%。
    """
    frame = daily.select("ts_code", "trade_date", "pre_close", "close").join(
        adj.select("ts_code", "trade_date", "adj_factor"),
        on=["ts_code", "trade_date"],
        how="left",
    )
    x = consecutive(frame, days).with_columns(
        G=pl.col("last_close") / pl.col("pre_close"),
        R=pl.col("adj_factor") / pl.col("last_adj"),
    )
    events = x.filter(
        ((pl.col("G") - 1).abs() > EVENT) | ((pl.col("R") - 1).abs() > EVENT)
    )
    bad = events.filter(
        ~((pl.col("R") / pl.col("G") - 1).abs() < MATCH).fill_null(False)
    )
    problems = [
        dict(symbol=r[0], date=r[1])
        for r in bad.select("ts_code", "trade_date").iter_rows()
    ]
    div = dividend.select(
        pl.col("ts_code"), pl.col("ex_date").alias("trade_date")
    ).unique()
    matched = events.join(div, on=["ts_code", "trade_date"], how="anti")
    splits = matched.filter(
        ((pl.col("R") / pl.col("G") - 1).abs() < MATCH)
        & ((pl.col("G") - 1).abs() > EVENT)
    )
    by_kind = Counter(
        "structured" if c.startswith(("150", "502")) else "other"
        for c in splits["ts_code"].to_list()
    )
    return _judge(
        problems,
        register,
        "fund_adj_vs_gap",
        len(events),
        "前收盘缺口（除权除息）与复权因子变化一致",
        extra=dict(
            events=len(events),
            split_or_conversion_events=len(splits),
            split_events_by_kind=dict(by_kind),
            split_events_by_year=dict(
                Counter(d[:4] for d in splits["trade_date"].to_list())
            ),
        ),
    )


def check_code_reuse(
    fund_basic, fund_daily, fund_adj, fund_div, cb_basic, cb_daily, register
):
    marked = sorted(
        {
            c
            for f in (fund_daily, fund_adj, fund_div)
            if len(f)
            for c in f["ts_code"].unique().to_list()
            if "!" in c
        }
    )
    known = set(fund_basic["ts_code"].to_list())
    problems = [
        dict(symbol=c)
        for c in sorted(set(fund_daily["ts_code"].unique().to_list()) - known)
    ]
    dup_cb = (
        cb_basic.filter(pl.col("ts_code").is_duplicated())["ts_code"].unique().to_list()
    )
    item = _judge(
        problems,
        register,
        "fund_code_not_in_table",
        len(set(fund_daily["ts_code"].unique().to_list())),
        "日线里的基金代码都在基金表里；带 `!` 的复用代码只统计",
        extra=dict(reused_markers=marked, cb_duplicate_codes=dup_cb),
    )
    if dup_cb:
        item["verdict"] = "fail"
    return item


def check_cb_life(cb_daily, cb_basic, register):
    """有成交的行必须在上市日到摘牌日（不含）之间：同一代码复用或证券表日期错会在这里暴露。"""
    life = cb_basic.select("ts_code", "list_date", "delist_date")
    traded = cb_daily.filter((pl.col("vol") > 0) & (pl.col("close") > 0)).join(
        life, on="ts_code", how="left"
    )
    bad = traded.filter(
        pl.col("list_date").is_null()
        | (pl.col("trade_date") < pl.col("list_date"))
        | (
            pl.col("delist_date").is_not_null()
            & (pl.col("trade_date") >= pl.col("delist_date"))
        )
    )
    problems = [
        dict(symbol=r[0], date=r[1])
        for r in bad.select("ts_code", "trade_date").iter_rows()
    ]
    return _judge(
        problems,
        register,
        "cb_traded_outside_life",
        len(traded),
        "有成交的行在上市期内",
    )


def check_cb_after_end(cb_daily, cb_basic, calls, days):
    """到期或强赎后：源的日线、面板的行各还有多少；最后几天的价格与强赎价的关系。"""
    traded = cb_daily.filter((pl.col("vol") > 0) & (pl.col("close") > 0))
    last = traded.group_by("ts_code").agg(
        pl.col("trade_date").max().alias("last_trade"),
        pl.col("close").sort_by("trade_date").last().alias("last_close"),
    )
    rows_after = cb_daily.join(last, on="ts_code").filter(
        pl.col("trade_date") > pl.col("last_trade")
    )
    implemented = (
        calls.filter(pl.col("is_call") == "公告实施强赎")
        .group_by("ts_code")
        .agg(
            pl.col("call_reg_date").max(),
            pl.col("call_price").max(),
            pl.col("ann_date").min(),
        )
    )
    x = last.join(implemented, on="ts_code").filter(pl.col("call_price").is_not_null())
    ratio = (x["last_close"] / x["call_price"]).to_numpy()
    on_record = x.filter(pl.col("last_trade") == pl.col("call_reg_date")).height
    gaps = Counter(
        int(np.searchsorted(days, b) - np.searchsorted(days, a))
        for a, b in zip(
            x["last_trade"].to_list(), x["call_reg_date"].to_list(), strict=True
        )
        if b is not None
    )
    maturity = (
        calls.filter(pl.col("is_call") == "公告到期赎回")
        .group_by("ts_code")
        .agg(pl.col("call_price").max())
    )
    m = last.join(maturity, on="ts_code")
    m_ratio = (m["last_close"] / m["call_price"]).to_numpy()

    def q(a):
        return (
            {}
            if not len(a)
            else {p: float(np.quantile(a, p / 100)) for p in (5, 25, 50, 75, 95)}
        )

    return _check(
        "info",
        len(last),
        0,
        dict(
            placeholder_rows_after_last_trade=len(rows_after),
            bonds_with_rows_after_last_trade=rows_after["ts_code"].n_unique(),
            called_bonds=len(x),
            last_trade_on_record_date=on_record,
            trading_days_from_last_trade_to_record_date={
                str(k): v for k, v in sorted(gaps.items())
            },
            last_close_over_call_price_quantiles=q(ratio),
            last_close_below_call_price=int((ratio < 1).sum()),
            matured_bonds=len(m),
            last_close_over_maturity_price_quantiles=q(m_ratio),
        ),
        "强赎：最后交易日通常就是赎回登记日；未卖出的按强赎价（面值加当期利息）赎回",
    )


# ------------------------------------------------------------ 面板（不允许例外，或只允许按行登记的少数）


def _panel(store, kind):
    base = store.root / "fund_panel" / kind
    current = base / "current.json"
    if not current.exists():
        return None
    path = base / json.loads(current.read_text())["version"]
    meta = json.loads((path / "metadata.json").read_text())
    return path, meta


def _load(path, name):
    return np.load(path / f"{name}.npy")


def check_panel_listing(path, meta, kind):
    has = _load(path, "has_quote")
    listed = _load(path, "is_listed")
    bad = np.argwhere(has & ~listed)
    problems = [
        dict(symbol=meta["symbols"][j], date=meta["dates"][i]) for i, j in bad[:50]
    ]
    return _strict(
        problems if len(bad) else [],
        int(has.sum()),
        f"{kind}：有行情的格子都在上市期内",
        extra=dict(count=int(len(bad))),
    )


def recompute_fund_split(path, meta, dividend):
    """面板 split 与按原始规则重算的结果逐格相同（另写一遍：先向前填有效收盘与复权因子）。"""
    close = _load(path, "close")
    prev = _load(path, "previous_close")
    adj = _load(path, "adj_factor")
    panel_split = _load(path, "split")
    t, n = close.shape
    dates = np.array(meta["dates"])
    symbols = np.array(meta["symbols"])

    def ffill_shift(a):
        out = np.full_like(a, np.nan)
        last = np.full(a.shape[1], np.nan)
        for i in range(a.shape[0]):
            out[i] = last
            ok = np.isfinite(a[i])
            last = np.where(ok, a[i], last)
        return out

    last_close = ffill_shift(close)
    last_adj = ffill_shift(adj)
    div = np.zeros((t, n))
    d = dividend.filter(pl.col("symbol").is_in(symbols.tolist()))
    for s, day, cash in d.select("symbol", "date", "cash_dividend").iter_rows():
        i = int(np.searchsorted(dates, day))
        if i < t:
            div[i, int(np.searchsorted(symbols, s))] += cash
    with np.errstate(invalid="ignore", divide="ignore"):
        g = last_close / prev
        r = adj / last_adj
        dfac = np.where(div > 0, last_close / (last_close - div), 1.0)
        value = g / dfac
    event = (
        (np.abs(r - 1) > EVENT)
        & (np.abs(g - 1) > EVENT)
        & (dfac > 0)
        & (np.abs(value - 1) > EVENT)
    )
    expected = np.where(event, value, 1.0)
    bad = np.argwhere(~np.isclose(expected, panel_split, rtol=0, atol=1e-12))
    problems = [dict(symbol=str(symbols[j]), date=str(dates[i])) for i, j in bad[:50]]
    return _strict(
        problems if len(bad) else [],
        t * n,
        "面板 split 与原始表重算一致",
        extra=dict(count=int(len(bad)), events=int(event.sum())),
    )


def check_fund_limits(path, meta, register):
    """实际涨跌幅不超过按规则分档的涨跌停（上市首日除外）：反查涨跌停档位是否认对。"""
    from .fund_rules import classify

    close = _load(path, "close")
    high = _load(path, "high")
    low = _load(path, "low")
    prev = _load(path, "previous_close")
    has = _load(path, "has_quote")
    securities = pl.read_parquet(path / "securities.parquet")
    order = {s: i for i, s in enumerate(meta["symbols"])}
    classes = classify(securities).sort(
        pl.col("symbol").replace_strict(order, return_dtype=pl.Int64)
    )
    dates = np.array(meta["dates"])
    limit = np.full(close.shape, 0.1)
    for j, r in enumerate(classes.iter_rows(named=True)):
        if r["limit"] > 0.1:
            limit[dates >= (r["limit_from"] or "0"), j] = r["limit"]
        if r["asset_class"] == "reits":
            limit[:, j] = np.maximum(limit[:, j], 0.1)
    first = np.where(has.any(axis=0), has.argmax(axis=0), -1)
    with np.errstate(invalid="ignore", divide="ignore"):
        move = np.fmax(np.fmax(high, close) / prev - 1, 1 - np.fmin(low, close) / prev)
    over = (move > limit + 0.0005 + 0.0015 / np.where(prev > 0, prev, np.nan)) & has
    over[first[first >= 0], np.flatnonzero(first >= 0)] = False
    problems = [
        dict(symbol=meta["symbols"][j], date=meta["dates"][i])
        for i, j in np.argwhere(over)
    ]
    return _judge(
        problems,
        register,
        "fund_price_limits",
        int(has.sum()),
        "涨跌幅不超过规则档位（上市首日除外）",
    )


def check_cb_limits(path, meta):
    """2022-08-01 起实际最高最低价越过按规则算的涨跌停价（超出 0.005 元）的格子：只统计。"""
    from .cb_rules import price_limits

    dates = np.array(meta["dates"])
    listed = _load(path, "is_listed")
    has = _load(path, "has_quote")
    first = listed & ~np.vstack([np.zeros((1, listed.shape[1]), bool), listed[:-1]])
    up, down = price_limits(dates, _load(path, "previous_close"), first)
    high, low = _load(path, "high"), _load(path, "low")
    over = has & ((high > up + 0.005) | (low < down - 0.005))
    checked = int((has & np.isfinite(up)).sum())
    return _check(
        "info",
        checked,
        0,
        dict(
            over=int(over.sum()),
            examples=[
                dict(symbol=meta["symbols"][j], date=meta["dates"][i])
                for i, j in np.argwhere(over)[:20]
            ],
        ),
        "涨跌幅新规（2022-08-01）之后越过规则涨跌停价的格子；面板起点为区间首行时首日按前收盘计，会多出几格",
    )


def known_conversion_price(dates, symbols, basic, changes, listed):
    """按「max(生效日, 公告日的下一交易日)」另算一遍当时可知的转股价（不调用面板代码）。"""
    t = len(dates)
    first = {r["ts_code"]: r["first_conv_price"] for r in basic.iter_rows(named=True)}
    out = np.full((t, len(symbols)), np.nan)
    groups = {}
    for code, pub, chg, after in changes.select(
        "ts_code", "publish_date", "change_date", "convertprice_aft"
    ).iter_rows():
        if after is None:
            continue
        eff = int(np.searchsorted(dates, chg)) if chg else t
        known = int(np.searchsorted(dates, pub, side="right")) if pub else t
        groups.setdefault(code, []).append(
            (max(eff, known), chg or "", pub or "", float(after))
        )
    for j, code in enumerate(symbols):
        value = first.get(code)
        value = np.nan if value is None else float(value)
        events = sorted(groups.get(code, []))
        k = 0
        for i in range(t):
            while k < len(events) and events[k][0] <= i:
                value = events[k][3]
                k += 1
            if listed[i, j]:
                out[i, j] = value
    return out


def check_cb_conversion(store, path, meta, register):
    dates = np.array(meta["dates"])
    symbols = np.array(meta["symbols"])
    basic = _read(store, "cb_basic").unique("ts_code", keep="last")
    changes = _read(store, "cb_price_chg").unique(
        ["ts_code", "publish_date", "change_date", "convertprice_aft"], keep="last"
    )
    listed = _load(path, "is_listed")
    panel_cp = _load(path, "conversion_price")
    expected = known_conversion_price(dates, symbols, basic, changes, listed)
    diff = ~((expected == panel_cp) | (np.isnan(expected) & np.isnan(panel_cp)))
    strict = _strict(
        [
            dict(symbol=str(symbols[j]), date=str(dates[i]))
            for i, j in np.argwhere(diff)[:50]
        ],
        int(listed.sum()),
        "面板 conversion_price 与原始表重算（当时可知）逐格相同",
        extra=dict(count=int(diff.sum())),
    )
    # 源的转股价值按「生效日」换转股价；另算一遍同口径的转股价值与源比较，确认公式和正股价对得上。
    source_timing = changes.with_columns(publish_date=pl.lit("00000000"))
    cp_source = known_conversion_price(dates, symbols, basic, source_timing, listed)
    stock = _load(path, "stock_close")
    source = _load(path, "source_conversion_value")
    has = _load(path, "has_quote")
    with np.errstate(invalid="ignore", divide="ignore"):
        value = 100.0 / cp_source * stock
        rel = value / source - 1
    compared = has & np.isfinite(rel)
    bad = compared & (np.abs(rel) > 0.01)
    problems = [
        dict(symbol=str(symbols[j]), date=str(dates[i])) for i, j in np.argwhere(bad)
    ]
    with np.errstate(invalid="ignore", divide="ignore"):
        panel_rel = _load(path, "conversion_value") / source - 1
    timing = int((compared & (np.abs(panel_rel) > 0.01) & ~bad).sum())
    loose = _judge(
        problems,
        register,
        "cb_conversion_value_source",
        int(compared.sum()),
        "同口径（按生效日换转股价）算出的转股价值与源相差不超过 1%",
        extra=dict(
            median_abs_relative=float(np.nanmedian(np.abs(rel[compared])))
            if compared.any()
            else None,
            panel_differs_only_by_timing=timing,
        ),
    )
    return strict, loose


def verify_funds(store, *, write=True, known=None):
    """全部校验；返回报告 dict，并写 ``reports/funds_verify.json`` 与 ``.md``。"""
    started = time.perf_counter()
    known = load_known() if known is None else known
    register = Register(known)
    days = open_days(store)
    fund_basic = _read(store, "fund_basic")
    fund_daily = _read(store, "fund_daily")
    fund_adj = _read(store, "fund_adj")
    fund_div = _read(store, "fund_div")
    cb_basic = _read(store, "cb_basic")
    cb_daily = _read(store, "cb_daily")
    calls = _read(store, "cb_call")
    checks = {"manifest_files": check_manifest(store)}
    for name, item in check_partitions(store, days, register).items():
        checks[f"partitions_{name}"] = item
    checks["unique_keys"] = check_unique(store)
    checks["fund_vwap"] = check_vwap(
        fund_daily, register, check="fund_vwap", volume_scale=100, amount_scale=1000
    )
    checks["cb_vwap"] = check_vwap(
        cb_daily, register, check="cb_vwap", volume_scale=10, amount_scale=10000
    )
    checks["cb_zero_rows"] = check_zero_rows(cb_daily)
    implemented = (
        fund_div.filter(pl.col("div_proc") == "实施") if len(fund_div) else fund_div
    )
    checks["fund_adj_vs_gap"] = check_adj_vs_gap(
        fund_daily, fund_adj, implemented, days, register
    )
    checks["code_reuse"] = check_code_reuse(
        fund_basic, fund_daily, fund_adj, fund_div, cb_basic, cb_daily, register
    )
    checks["cb_traded_outside_life"] = check_cb_life(cb_daily, cb_basic, register)
    checks["cb_after_end"] = check_cb_after_end(cb_daily, cb_basic, calls, days)
    versions = {}
    for kind in ("fund", "cb"):
        found = _panel(store, kind)
        if found is None:
            checks[f"{kind}_panel"] = _check(
                "fail", 0, 1, {}, f"{kind} 面板尚未构建：先 build-funds"
            )
            continue
        path, meta = found
        versions[kind] = meta["version"]
        checks[f"{kind}_panel_listing"] = check_panel_listing(path, meta, kind)
        if kind == "fund":
            dividend = pl.read_parquet(path / "table_dividends.parquet")
            checks["fund_panel_split"] = recompute_fund_split(path, meta, dividend)
            checks["fund_price_limits"] = check_fund_limits(path, meta, register)
        else:
            strict, loose = check_cb_conversion(store, path, meta, register)
            checks["cb_panel_conversion_price"] = strict
            checks["cb_conversion_value_source"] = loose
            checks["cb_price_limits"] = check_cb_limits(path, meta)
    verdicts = Counter(item["verdict"] for item in checks.values())
    report = dict(
        verdict="fail" if verdicts["fail"] else "pass",
        verdicts=dict(verdicts),
        generated_at=datetime.now(UTC).isoformat(),
        elapsed_seconds=time.perf_counter() - started,
        panel_versions=versions,
        raw_version=store.fund_version,
        known_exceptions=dict(
            entries=len(known.get("entries", [])),
            used=len(register.used),
            by_check=dict(Counter(e["check"] for e in known.get("entries", []))),
            cause_unknown=register.cause_unknown(known),
        ),
        stale_exceptions=register.stale(known)[:100],
        checks=checks,
    )
    if write:
        folder = store.root / "reports"
        atomic_json(
            folder / "funds_verify.json", json.loads(json.dumps(report, default=str))
        )
        lines = [
            "# 场内基金与可转债校验",
            "",
            f"总结论：**{report['verdict']}**；生成 {report['generated_at']}；已知例外 "
            f"{report['known_exceptions']['used']} / {report['known_exceptions']['entries']} 条用到，"
            f"其中原因未查明 {report['known_exceptions']['cause_unknown']['count']} 条",
            "",
            "| 检查 | 结论 | 检查数 | 未解释 | 已解释 | 说明 |",
            "|---|---|---|---|---|---|",
        ]
        for name, item in checks.items():
            lines.append(
                f"| {name} | {item['verdict']} | {item['checked']} | {item['failed']} | {item['explained']} | {item['note']} |"
            )
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "funds_verify.md").write_text("\n".join(lines) + "\n")
    return report


__all__ = ["Register", "load_known", "verify_funds"]
