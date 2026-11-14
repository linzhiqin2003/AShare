"""集成、端到端、冒烟测试共用的合成数据目录：原始库经 `Store.write`，面板经 `panel.build`。

不手拼面板文件，全部走数据层自己的写入接口，磁盘格式与正式库相同；价格全为固定种子生成，
不读真实行情、不联网。默认约 48 只股票、2023-07 至 2026-01（跨 IS / VALIDATION / TEST），
含停牌、涨跌停（含一字板）、ST、送转与派息、上市、退市、换行业、指数成分调整。

    info = build_datadir(tmp_path)          # 写原始库并构建面板，返回 Info
    info = write_raw(tmp_path)              # 只写原始库，留给 build-panel 命令行
    info = cached_datadir()                 # 按代码哈希缓存的只读共享库（测试默认用这个）
    data = ashare.data.open(info.root)

`Info.events` 记下每类边界放在哪只股票、哪一天，测试按它断言面板里的值。
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

REPOSITORY = Path(__file__).resolve().parents[2]
SEED = 20261114
FETCHED_AT = "2026-02-02T00:00:00Z"
START, END = "20230703", "20260130"
N_STOCKS = 48
CALENDAR_STEP = 3
WRITERS = 4
# 固定的非周末休市日（合成，只为让日历不是纯工作日）。
HOLIDAYS = (
    "20231002",
    "20231003",
    "20240101",
    "20240501",
    "20241001",
    "20241002",
    "20250101",
    "20250501",
    "20251001",
    "20251002",
    "20260101",
)
INDUSTRIES = (
    ("801010.SI", "农林牧渔"),
    ("801030.SI", "基础化工"),
    ("801040.SI", "钢铁"),
    ("801080.SI", "电子"),
    ("801150.SI", "医药生物"),
    ("801780.SI", "银行"),
)
# 与 catalog.SYMBOL_ALIASES 最后一条一致；数据层要求覆盖接缝的库两边都有行情。
SYMBOL_ALIAS = dict(
    old="300114.SZ", new="302132.SZ", old_last="20250214", switch="20250217"
)
DAY_TABLES = (
    "daily",
    "adj_factor",
    "daily_basic",
    "stk_limit",
    "suspend_d",
    "stock_st",
    "dividend",
)
BENCHMARK_CODES = ("000001.SH", "000300.SH", "000905.SH", "000852.SH")


@dataclass
class Info:
    root: Path
    dates: list[str]
    symbols: list[str]
    events: dict = field(default_factory=dict)
    seconds: dict = field(default_factory=dict)
    version: str | None = None


def calendar(start=START, end=END, step=CALENDAR_STEP):
    """合成交易日历：工作日去掉固定休市日后每隔 `step` 天取一天，并保留换码接缝两天。

    跨两年半而分区只有两百多个：原始库每个交易日每个日表一个分区，分区数决定生成耗时。
    """
    days = np.arange(
        np.datetime64(f"{start[:4]}-{start[4:6]}-{start[6:]}"),
        np.datetime64(f"{end[:4]}-{end[4:6]}-{end[6:]}") + np.timedelta64(1, "D"),
    )
    days = days[np.is_busday(days)]
    text = np.char.replace(days.astype(str), "-", "")
    days = [d for d in text.tolist() if d not in HOLIDAYS]
    keep = set(days[::step]) | {
        d for d in (SYMBOL_ALIAS["old_last"], SYMBOL_ALIAS["switch"]) if d in days
    }
    return sorted(keep)


def _round(x):
    # A 股价格两位小数；四舍五入加极小偏置避免 .xx5 的二进制误差。
    return np.floor(np.asarray(x) * 100 + 0.5 + 1e-9) / 100


def _plan(dates, n):
    """边界事件放在哪只股票、哪一天；全部是固定位置，测试按这里断言。"""
    t = len(dates)

    def at(day):
        return int(np.searchsorted(dates, day))

    return dict(
        # 连续停牌 6 天（全天停牌，S），之后复牌，复牌当天有 R 记录。
        suspended=dict(symbol=3, rows=list(range(at("20240311"), at("20240311") + 6))),
        # 验证期内单日停牌一次。
        suspended_validation=dict(symbol=16, rows=[at("20250612")]),
        # 普通涨停（收在涨停价）、一字涨停（开高低收均为涨停价）、一字跌停。
        limit_up=dict(symbol=5, rows=[at("20240115"), at("20250303")]),
        limit_up_one_price=dict(symbol=7, rows=[at("20240520"), at("20250715")]),
        limit_down_one_price=dict(symbol=9, rows=[at("20240826"), at("20250910")]),
        # 全程 ST（每天都有 ST 记录，涨跌幅 5%），另一只区间 ST。
        st_always=dict(symbol=n - 1),
        st_window=dict(symbol=14, rows=list(range(at("20241104"), at("20250203")))),
        # 送转（10 送 5）、派息、同日送转加派息。
        split=dict(symbol=2, row=at("20240617"), stk_div=0.5, cash=0.0),
        cash=dict(symbol=6, row=at("20240715"), stk_div=0.0, cash=0.8),
        split_cash=dict(symbol=10, row=at("20250616"), stk_div=0.3, cash=0.5),
        cash_validation=dict(symbol=20, row=at("20250707"), stk_div=0.0, cash=0.3),
        # 样本中途上市、样本中途退市。
        listed=dict(symbol=n - 2, row=at("20240301")),
        delisted=dict(symbol=n - 3, row=at("20250602")),
        # 换行业：从第 0 个行业换到第 5 个。
        industry_change=dict(symbol=1, row=at("20240902"), before=0, after=5),
        # 指数调整：第 11 只从沪深300 调到中证500，第 12 只反向，以 2024-12 快照生效。
        index_swap=dict(out=11, into=12, month="202412"),
        # 换码：真实名单里的 300114.SZ→302132.SZ（2025-02-14 / 02-17 接缝），价格连续。
        alias=dict(symbol=30, **SYMBOL_ALIAS),
        rows=t,
    )


def raw_tables(dates=None, n=N_STOCKS, seed=SEED, *, with_alias=True):
    """返回 (tables, events)：tables 是 Tushare 原始口径的长表，events 是边界位置。"""
    dates = list(dates or calendar())
    rng = np.random.default_rng(seed)
    t = len(dates)
    plan = _plan(dates, n)
    # 代码避开 catalog.SYMBOL_ALIASES 里的换码；plan['alias'] 那只股票专门用来测换码。
    base_codes = [
        f"{600100 + j:06d}.SH" if j % 2 == 0 else f"{2100 + j:06d}.SZ" for j in range(n)
    ]
    alias = plan["alias"]
    names = [f"合成{j:02d}" for j in range(n)]
    # 换码后的新代码沿用原上市日，旧代码不再出现在证券主表（行情经数据层规范化并到新代码）。
    codes = list(base_codes)
    if with_alias:
        codes[alias["symbol"]] = alias["new"]
    code_matrix = np.array([codes] * t, dtype=object)
    if with_alias and alias["switch"] in dates:
        code_matrix[: dates.index(alias["switch"]), alias["symbol"]] = alias["old"]
    symbols = sorted(codes)

    listed = np.ones((t, n), bool)
    first = plan["listed"]
    listed[: first["row"], first["symbol"]] = False
    last = plan["delisted"]
    listed[last["row"] :, last["symbol"]] = False
    suspended = np.zeros((t, n), bool)
    for key in ("suspended", "suspended_validation"):
        suspended[plan[key]["rows"], plan[key]["symbol"]] = True
    st = np.zeros((t, n), bool)
    st[:, plan["st_always"]["symbol"]] = True
    st[plan["st_window"]["rows"], plan["st_window"]["symbol"]] = True
    actions = {}
    for key in ("split", "cash", "split_cash", "cash_validation"):
        e = plan[key]
        actions[(e["row"], e["symbol"])] = (e["stk_div"], e["cash"])
    forced = {}
    for key, kind in (
        ("limit_up", "up"),
        ("limit_up_one_price", "up_one"),
        ("limit_down_one_price", "down_one"),
    ):
        for row in plan[key]["rows"]:
            forced[(row, plan[key]["symbol"])] = kind

    base = np.exp(rng.normal(np.log(15), 0.5, n))
    drift = rng.normal(0.0002, 0.0004, n)
    vol = rng.uniform(0.012, 0.03, n)
    shocks = rng.normal(0, 1, (t, n))
    shares = np.exp(rng.normal(np.log(2e5), 0.8, n))  # 万股
    columns = {
        k: np.full((t, n), np.nan)
        for k in "open high low close pre up down adj".split()
    }
    adj = np.ones(n)
    prev = base.copy()
    for i in range(t):
        for j in range(n):
            if not listed[i, j]:
                continue
            stock, money = actions.get((i, j), (0.0, 0.0))
            if i == first["row"] and j == first["symbol"]:
                pre = _round(prev[j])
                limit = 0.44
            else:
                pre = _round((prev[j] - money) / (1 + stock))
                # 300 开头是创业板，涨跌幅 20%（ST 也是 20%）；主板 10%，主板 ST 5%。
                limit = (
                    0.2
                    if with_alias and j == alias["symbol"]
                    else 0.05
                    if st[i, j]
                    else 0.1
                )
            if stock or money:
                adj[j] *= prev[j] / pre
            up, down = _round(pre * (1 + limit)), _round(pre * (1 - limit))
            columns["pre"][i, j], columns["up"][i, j] = pre, up
            columns["down"][i, j], columns["adj"][i, j] = down, adj[j]
            if suspended[i, j]:
                continue
            kind = forced.get((i, j))
            ret = drift[j] + vol[j] * shocks[i, j]
            close = float(np.clip(_round(pre * np.exp(ret)), down, up))
            opening = float(np.clip(_round(pre * np.exp(0.3 * ret)), down, up))
            if kind == "up":
                close = up
            elif kind == "up_one":
                close = opening = up
            elif kind == "down_one":
                close = opening = down
            if kind in ("up_one", "down_one"):
                high = low = close
            else:
                high = min(up, _round(max(close, opening) * (1 + 0.004 * rng.random())))
                low = max(
                    down, _round(min(close, opening) * (1 - 0.004 * rng.random()))
                )
            columns["open"][i, j], columns["close"][i, j] = opening, close
            columns["high"][i, j], columns["low"][i, j] = high, low
            prev[j] = close

    rows_i, rows_j = np.nonzero(listed & ~suspended)
    day = np.array(dates)
    volume = rng.uniform(1e4, 5e5, len(rows_i))
    close = columns["close"][rows_i, rows_j]
    pre = columns["pre"][rows_i, rows_j]
    daily = pl.DataFrame(
        dict(
            ts_code=code_matrix[rows_i, rows_j].astype(str),
            trade_date=day[rows_i],
            open=columns["open"][rows_i, rows_j],
            high=columns["high"][rows_i, rows_j],
            low=columns["low"][rows_i, rows_j],
            close=close,
            pre_close=pre,
            change=close - pre,
            pct_chg=(close / pre - 1) * 100,
            vol=volume,
            amount=volume * close / 10,  # 手 × 元 × 100 / 1000 = 千元
        )
    )
    total = shares[rows_j]
    daily_basic = pl.DataFrame(
        dict(
            ts_code=code_matrix[rows_i, rows_j].astype(str),
            trade_date=day[rows_i],
            close=close,
            turnover_rate=volume * 100 / (total * 10000) * 100,
            turnover_rate_f=volume * 100 / (total * 6000) * 100,
            volume_ratio=rng.uniform(0.5, 2, len(rows_i)),
            pe=rng.uniform(5, 60, len(rows_i)),
            pe_ttm=rng.uniform(5, 60, len(rows_i)),
            pb=rng.uniform(0.5, 8, len(rows_i)),
            ps=rng.uniform(0.5, 10, len(rows_i)),
            ps_ttm=rng.uniform(0.5, 10, len(rows_i)),
            dv_ratio=rng.uniform(0, 4, len(rows_i)),
            dv_ttm=rng.uniform(0, 4, len(rows_i)),
            total_share=total,
            float_share=total * 0.7,
            free_share=total * 0.5,
            total_mv=close * total,
            circ_mv=close * total * 0.7,
        )
    )
    li, lj = np.nonzero(listed)
    adj_factor = pl.DataFrame(
        dict(
            ts_code=code_matrix[li, lj].astype(str),
            trade_date=day[li],
            adj_factor=columns["adj"][li, lj],
        )
    )
    stk_limit = pl.DataFrame(
        dict(
            ts_code=code_matrix[li, lj].astype(str),
            trade_date=day[li],
            pre_close=columns["pre"][li, lj],
            up_limit=columns["up"][li, lj],
            down_limit=columns["down"][li, lj],
        )
    )
    si, sj = np.nonzero(suspended)
    resume = [
        (i + 1, j)
        for i, j in zip(si, sj, strict=True)
        if i + 1 < t and not suspended[i + 1, j]
    ]
    suspend_d = pl.DataFrame(
        dict(
            ts_code=[code_matrix[i, j] for i, j in zip(si, sj, strict=True)]
            + [code_matrix[i, j] for i, j in resume],
            trade_date=[dates[i] for i in si] + [dates[i] for i, _ in resume],
            suspend_timing=pl.Series([None] * (len(si) + len(resume)), dtype=pl.String),
            suspend_type=["S"] * len(si) + ["R"] * len(resume),
        )
    )
    ti, tj = np.nonzero(st & listed)
    stock_st = pl.DataFrame(
        dict(
            ts_code=code_matrix[ti, tj].astype(str),
            name=[f"ST{names[j]}" for j in tj],
            trade_date=day[ti],
            type="ST",
            type_name="风险警示",
        )
    )
    dividend = pl.DataFrame(
        [
            dict(
                ts_code=code_matrix[i, j],
                end_date=f"{int(dates[i][:4]) - 1}1231",
                ann_date=dates[max(0, i - 40)],
                div_proc="实施",
                stk_div=stock,
                stk_bo_rate=stock,
                stk_co_rate=0.0,
                cash_div=money,
                cash_div_tax=money,
                record_date=dates[i - 1],
                ex_date=dates[i],
                pay_date=dates[i],
                div_listdate=dates[i] if stock else None,
                imp_ann_date=dates[i - 5],
            )
            for (i, j), (stock, money) in sorted(actions.items())
        ],
        schema_overrides=dict(div_listdate=pl.String),
    )
    status = np.where(np.arange(n) == last["symbol"], "D", "L")
    stock_basic = pl.DataFrame(
        dict(
            ts_code=codes,
            symbol=[s[:6] for s in codes],
            name=names,
            area="合成",
            industry=[INDUSTRIES[j % 6][1] for j in range(n)],
            market="主板",
            exchange=["SSE" if s.endswith(".SH") else "SZSE" for s in codes],
            list_status=status,
            list_date=[
                dates[first["row"]] if j == first["symbol"] else "20100104"
                for j in range(n)
            ],
            delist_date=[
                dates[last["row"]] if j == last["symbol"] else None for j in range(n)
            ],
        ),
        schema_overrides=dict(delist_date=pl.String),
    )
    every = np.arange(
        np.datetime64(f"{dates[0][:4]}-01-01"),
        np.datetime64(f"{dates[-1][:4]}-12-31") + np.timedelta64(1, "D"),
    )
    every = np.char.replace(every.astype(str), "-", "").tolist()
    open_days = set(dates)
    previous, pretrade = "20221230", []
    for d in every:
        pretrade.append(previous)
        if d in open_days:
            previous = d
    trade_cal = pl.DataFrame(
        dict(
            exchange="SSE",
            cal_date=every,
            is_open=[int(d in open_days) for d in every],
            pretrade_date=pretrade,
        )
    )
    change = plan["industry_change"]
    member_rows = []
    for j in range(n):
        code_j, name = INDUSTRIES[j % 6]
        in_date = stock_basic["list_date"][j]
        if j == change["symbol"]:
            before, after = INDUSTRIES[change["before"]], INDUSTRIES[change["after"]]
            switch = dates[change["row"]]
            member_rows.append((j, before, in_date, dates[change["row"] - 1], "N"))
            member_rows.append((j, after, switch, None, "Y"))
        else:
            member_rows.append((j, (code_j, name), in_date, None, "Y"))
    index_member_all = pl.DataFrame(
        [
            dict(
                l1_code=industry[0],
                l1_name=industry[1],
                l2_code=industry[0].replace("801", "802"),
                l2_name=industry[1] + "二级",
                l3_code=industry[0].replace("801", "803"),
                l3_name=industry[1] + "三级",
                ts_code=codes[j],
                name=names[j],
                in_date=in_date,
                out_date=out_date,
                is_new=flag,
            )
            for j, industry, in_date, out_date, flag in member_rows
        ],
        schema_overrides=dict(out_date=pl.String),
    )
    index_classify = pl.DataFrame(
        dict(
            index_code=[c for c, _ in INDUSTRIES],
            industry_name=[name for _, name in INDUSTRIES],
            level="L1",
            industry_code=[c[:6] for c, _ in INDUSTRIES],
            is_pub="1",
            parent_code="0",
            src="SW2021",
        )
    )
    tables = dict(
        trade_cal=trade_cal,
        stock_basic=stock_basic,
        daily=daily,
        adj_factor=adj_factor,
        daily_basic=daily_basic,
        stk_limit=stk_limit,
        suspend_d=suspend_d,
        stock_st=stock_st,
        dividend=dividend,
        index_classify=index_classify,
        index_member_all=index_member_all,
        index_weight=_index_weight(dates, codes, plan, rng),
        index_daily=_index_daily(dates, rng),
    )
    events = {
        key: (
            {**value, "symbol": codes[value["symbol"]]}
            if isinstance(value, dict) and "symbol" in value
            else value
        )
        for key, value in plan.items()
    }
    events["index_swap"] = dict(
        out=codes[plan["index_swap"]["out"]],
        into=codes[plan["index_swap"]["into"]],
        month=plan["index_swap"]["month"],
    )
    return tables, events, symbols, dates


def _index_weight(dates, symbols, plan, rng):
    """每月最后一个交易日一张快照；样本首月之前另有一张，首日只继承已公开的成分。"""
    from ashare.data.catalog import INDEXES

    n = len(symbols)
    members = dict(
        csi300=list(range(0, 12)),
        csi500=list(range(12, 24)),
        csi1000=list(range(24, 36)),
        csi2000=list(range(36, 44)),
        csi_all=list(range(n)),
    )
    swap = plan["index_swap"]
    months = sorted({d[:6] for d in dates})
    snapshot_days = ["20230630"] + [max(d for d in dates if d[:6] == m) for m in months]
    rows = []
    for day in snapshot_days:
        for index, code in INDEXES.items():
            chosen = list(members[index])
            if day[:6] >= swap["month"]:
                if index == "csi300":
                    chosen = [swap["into"] if j == swap["out"] else j for j in chosen]
                elif index == "csi500":
                    chosen = [swap["out"] if j == swap["into"] else j for j in chosen]
            weight = rng.uniform(0.5, 1.5, len(chosen))
            weight = weight / weight.sum() * 100
            rows += [
                dict(index_code=code, con_code=symbols[j], trade_date=day, weight=w)
                for j, w in zip(chosen, weight, strict=True)
            ]
    return pl.DataFrame(rows)


def _index_daily(dates, rng):
    rows = []
    for code in BENCHMARK_CODES:
        price = 3000 * np.exp(np.cumsum(rng.normal(0.0001, 0.009, len(dates))))
        opening = price * np.exp(rng.normal(0, 0.002, len(dates)))
        previous = np.concatenate(([price[0]], price[:-1]))
        rows.append(
            pl.DataFrame(
                dict(
                    ts_code=code,
                    trade_date=dates,
                    open=opening,
                    high=np.maximum(price, opening) * 1.002,
                    low=np.minimum(price, opening) * 0.998,
                    close=price,
                    pre_close=previous,
                    vol=rng.uniform(1e8, 3e8, len(dates)),
                    amount=rng.uniform(1e8, 5e8, len(dates)),
                )
            )
        )
    return pl.concat(rows)


def _relaxed(store):
    """合成库只活在临时目录里：关掉清单库每次提交的全盘同步，写入快 7 倍，内容与格式不变。"""
    manifest = getattr(store, "_manifest", None)
    db = getattr(manifest, "db", None)
    if db is not None:
        db.execute("PRAGMA synchronous=OFF")
        db.execute("PRAGMA fullfsync=OFF")


def _close(store):
    db = getattr(getattr(store, "_manifest", None), "db", None)
    if db is not None:
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        db.close()


def write_raw(root, *, n=N_STOCKS, dates=None, seed=SEED, with_alias=True):
    """经 `Store.write` 按正式分区写全部原始表（不构建面板）。"""
    from ashare.data.catalog import DATASETS
    from ashare.data.sources.base import FetchResult
    from ashare.data.store import Store

    started = time.perf_counter()
    root = Path(root)
    tables, events, symbols, dates = raw_tables(dates, n, seed, with_alias=with_alias)
    generated = time.perf_counter()
    store = Store(root)
    _relaxed(store)
    jobs = []

    def put(name, partition, params, table):
        jobs.append((name, partition, params, table))

    for name in DAY_TABLES:
        spec = DATASETS[name]
        table = tables[name]
        groups = table.partition_by(spec.date, as_dict=True) if len(table) else {}
        empty = table.head(0)
        for day in dates:
            put(name, day, {spec.date: day}, groups.get((day,), empty))
    calendar_rows = tables["trade_cal"].with_columns(
        pl.col("cal_date").str.slice(0, 4).alias("_y")
    )
    for (year,), rows in sorted(calendar_rows.partition_by("_y", as_dict=True).items()):
        put(
            "trade_cal",
            year,
            dict(start_date=f"{year}0101", end_date=f"{year}1231"),
            rows.drop("_y"),
        )
    basic = tables["stock_basic"]
    for status in ("L", "D", "P"):
        put(
            "stock_basic",
            status,
            {"list_status": status},
            basic.filter(pl.col("list_status") == status),
        )
    put(
        "index_classify",
        "SW2021",
        {"src": "SW2021", "level": "L1"},
        tables["index_classify"],
    )
    members = tables["index_member_all"]
    for code, _ in INDUSTRIES:
        for flag in ("Y", "N"):
            put(
                "index_member_all",
                f"{code}_{flag}",
                {"l1_code": code, "is_new": flag},
                members.filter(
                    (pl.col("l1_code") == code) & (pl.col("is_new") == flag)
                ),
            )
    for dataset, key in (("index_weight", "index_code"), ("index_daily", "ts_code")):
        frame = tables[dataset].with_columns(
            pl.col("trade_date").str.slice(0, 6).alias("_m")
        )
        for (code, month), rows in sorted(
            frame.partition_by(key, "_m", as_dict=True).items()
        ):
            put(
                dataset,
                f"{code}_{month}",
                {key: code, "start_date": f"{month}01", "end_date": f"{month}31"},
                rows.drop("_m"),
            )
    try:
        # Store.write 支持多写者并发（共享锁 + 清单事务）；Parquet 写入和文件系统调用释放 GIL。
        with ThreadPoolExecutor(WRITERS) as pool:
            for future in [
                pool.submit(
                    store.write, name, part, params, FetchResult(table, FETCHED_AT, 1)
                )
                for name, part, params, table in jobs
            ]:
                future.result()
    finally:
        _close(store)
    info = Info(root=root, dates=dates, symbols=symbols, events=events)
    info.seconds = dict(
        generate=generated - started,
        write=time.perf_counter() - generated,
        partitions=len(jobs),
    )
    return info


def build_datadir(root, **options):
    """写原始库并用 `ashare.data.panel.build` 构建正式面板。"""
    from ashare.data.panel import build
    from ashare.data.store import Store

    info = write_raw(root, **options)
    started = time.perf_counter()
    store = Store(root)
    _relaxed(store)
    try:
        metadata = build(store)
    finally:
        _close(store)
    info.version = metadata["version"]
    info.seconds["build"] = time.perf_counter() - started
    return info


def _info_dict(info):
    return dict(
        dates=info.dates,
        symbols=info.symbols,
        events=info.events,
        seconds=info.seconds,
        version=info.version,
    )


def cache_key():
    """生成器、数据层代码、polars/numpy 版本任一变化都换新目录。"""
    digest = hashlib.sha256()
    sources = [Path(__file__)] + sorted((REPOSITORY / "src/ashare/data").rglob("*.py"))
    for path in sources:
        digest.update(str(path.relative_to(REPOSITORY)).encode())
        digest.update(path.read_bytes())
    digest.update(f"{pl.__version__}|{np.__version__}".encode())
    return digest.hexdigest()[:20]


def _writable(path, writable):
    """文件去掉 / 恢复写权限。目录保持可写，`rm -rf .cache` 照常能删。"""
    for item in [Path(path), *Path(path).rglob("*")]:
        if item.is_symlink() or item.is_dir():
            continue
        mode = item.stat().st_mode
        item.chmod(mode | 0o200 if writable else mode & ~0o222)


def _remove(path):
    path = Path(path)
    if path.exists():
        for item in [path, *path.rglob("*")]:
            if not item.is_symlink():
                item.chmod(item.stat().st_mode | 0o700)
        shutil.rmtree(path)


def _prune(cache, keep, max_age=86400):
    """删掉一天以上没再建过的旧键（生成器或数据层代码改过之后留下的）。"""
    for marker in cache.glob("*.json"):
        key = marker.stem
        if key == keep or time.time() - marker.stat().st_mtime < max_age:
            continue
        with (cache / f"{key}.lock").open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            _remove(cache / key)
            marker.unlink(missing_ok=True)
        (cache / f"{key}.lock").unlink(missing_ok=True)


def cached_datadir(cache=None):
    """建一次、各测试只读共享：同一会话的 xdist 进程和之后的会话都复用，键见 `cache_key`。

    目录默认在仓库 `.cache/test-datadir/`（已被 .gitignore 忽略），环境变量
    `ASHARE_TEST_DATADIR_CACHE` 可改。建好后所有文件去掉写权限：读路径若偷偷改数据文件会直接报错
    （新建文件由 test_datadir 的读前读后对照查）。
    调用方需要可写的库时先 `copy_datadir`。
    """
    cache = Path(
        cache
        or os.environ.get("ASHARE_TEST_DATADIR_CACHE")
        or REPOSITORY / ".cache/test-datadir"
    )
    cache.mkdir(parents=True, exist_ok=True)
    key = cache_key()
    folder = cache / key
    marker = cache / f"{key}.json"
    with (cache / f"{key}.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        if not marker.exists():
            _remove(folder)
            info = build_datadir(folder)
            _writable(folder, False)
            temp = marker.with_name(f".{marker.name}.{os.getpid()}")
            temp.write_text(json.dumps(_info_dict(info), ensure_ascii=False))
            os.replace(temp, marker)
            _prune(cache, key)
    saved = json.loads(marker.read_text())
    return Info(root=folder, **saved)


def copy_datadir(info, target, *, panel=True):
    """把只读共享库复制成可写副本；`panel=False` 时只留原始库（给 build-panel 命令行重建）。"""
    target = Path(target)
    shutil.copytree(
        info.root,
        target,
        ignore=None if panel else shutil.ignore_patterns("panel"),
    )
    _writable(target, True)
    return Info(
        root=target,
        dates=list(info.dates),
        symbols=list(info.symbols),
        events=dict(info.events),
        seconds=dict(info.seconds),
        version=info.version,
    )
