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
    report["verdict"], report["checks"]["fund_gap"]["verdict"]
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
EVENT = 0.005  # 面板事件推出的参考价、持有收益与交易所前收盘、复权因子的容差
REGISTER_KEYS = {
    "fund_vwap": ("symbol", "date"),
    "cb_vwap": ("symbol", "date"),
    "fund_gap": ("symbol", "date"),
    "fund_factor": ("symbol", "date"),
    "fund_factor_missing": ("symbol",),
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


def _quote_pairs(has):
    """每列相邻两个有行情日 (p, t)，跨停牌；返回 (列, p, t) 三个数组。"""
    cols, ps, ts = [], [], []
    for j in range(has.shape[1]):
        q = np.flatnonzero(has[:, j])
        if len(q) > 1:
            cols.append(np.full(len(q) - 1, j))
            ps.append(q[:-1])
            ts.append(q[1:])
    if not cols:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty
    return np.concatenate(cols), np.concatenate(ps), np.concatenate(ts)


def check_fund_events(path, meta, register):
    """面板的拆分、分红能不能解释交易所前收盘和复权因子：两条路径，都跨停牌、覆盖全部有行情日。

    对每只基金相邻两个有行情日 p < t，把 (p, t] 里面板的事件按引擎的规则依次作用：
    - 参考价：mark = (mark − 分红) / 拆分，从 p 日收盘开始；要求与 t 日交易所前收盘相差不超过 0.5%
      （``fund_gap``）。漏记拆分、多记分红、事件放错日子都会在这里出现。
    - 持有收益：1 份从 p 日收盘持有到 t 日收盘，拆分改份数，分红按当时的参考价再投资（与复权因子的口径
      相同）；要求与复权因子给出的收益 收盘(t)·因子(t) / (收盘(p)·因子(p)) 相差不超过 0.5%
      （``fund_factor``）。
    - 两天之一没有复权因子的单列计数（``fund_factor_missing``，按代码登记）。
    与面板构建的裁决规则是两条独立的路径：这里不读分红表、不看因子比去找事件。
    """
    has = _load(path, "has_quote")
    close = _load(path, "close")
    prev = _load(path, "previous_close")
    adj = _load(path, "adj_factor")
    split = _load(path, "split")
    cash = _load(path, "cash_dividend")
    dates = np.array(meta["dates"])
    symbols = np.array(meta["symbols"])
    cols, ps, ts = _quote_pairs(has)
    mark = close[ps, cols].copy()
    shares = np.ones(len(cols))
    span = ts - ps
    for step in range(1, int(span.max()) + 1 if len(span) else 1):
        live = span >= step
        rows = ps[live] + step
        c = cash[rows, cols[live]]
        s = split[rows, cols[live]]
        before = mark[live]
        after = np.maximum(before - c, 0)
        with np.errstate(all="ignore"):
            # 分红按除息后的参考价再投资：份数 × before / after；再按拆分改份数。
            shares[live] *= np.where(c > 0, before / after, 1.0) * s
        mark[live] = after / s
    with np.errstate(all="ignore"):
        gap_dev = mark / prev[ts, cols] - 1
        held = shares * close[ts, cols] / close[ps, cols]
        factor = close[ts, cols] * adj[ts, cols] / (close[ps, cols] * adj[ps, cols])
        factor_dev = held / factor - 1
    has_prev = np.isfinite(prev[ts, cols])
    gap_bad = has_prev & ~(np.abs(gap_dev) <= EVENT)
    no_factor = ~np.isfinite(factor)
    factor_bad = ~no_factor & ~(np.abs(factor_dev) <= EVENT)

    def cells(mask):
        return [
            dict(symbol=str(symbols[j]), date=str(dates[t]))
            for j, t in zip(cols[mask], ts[mask], strict=True)
        ]

    events = (split != 1) | (cash > 0)
    gap = _judge(
        cells(gap_bad),
        register,
        "fund_gap",
        int(has_prev.sum()),
        "相邻两个有行情日之间（跨停牌）面板的拆分、分红把上一收盘推成交易所前收盘（容差 0.5%）",
        extra=dict(
            over_two_percent=int((has_prev & (np.abs(gap_dev) > 0.02)).sum()),
            pairs_across_suspension=int((span > 1).sum()),
        ),
    )
    fac = _judge(
        cells(factor_bad),
        register,
        "fund_factor",
        int((~no_factor).sum()),
        "面板事件给出的持有收益与复权因子给出的收益一致（容差 0.5%，跨停牌）",
        extra=dict(
            over_two_percent=int((~no_factor & (np.abs(factor_dev) > 0.02)).sum()),
            event_cells=int(events.sum()),
            split_cells=int((split != 1).sum()),
            dividend_cells=int((cash > 0).sum()),
        ),
    )
    missing_symbols = sorted({str(s) for s in symbols[cols[no_factor]]})
    miss = _judge(
        [dict(symbol=s) for s in missing_symbols],
        register,
        "fund_factor_missing",
        len(cols),
        "有行情的日子都有复权因子（按代码登记例外）",
        extra=dict(cells=int(no_factor.sum())),
    )
    return gap, fac, miss


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
            gap, fac, miss = check_fund_events(path, meta, register)
            checks["fund_gap"] = gap
            checks["fund_factor"] = fac
            checks["fund_factor_missing"] = miss
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
