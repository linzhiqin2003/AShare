"""场内基金的代码复用与公司行为（拆分、折算、分红）：以复权因子为准。

两件事都在这里，`fund_panel` 只负责调用：

1. **代码复用**。源对复用过的代码有三种写法：复权因子记在 ``XXXXXX!1.YY``，日线记在六位代码或
   七位代码 ``XXXXXX1.YY``，基金表里六位、七位各一条。规则（`resolve_reused`）：

   - 基金表里六位、七位两条是**同一只基金**（名称和上市日相同）或没有七位那条：全部并到六位代码，
     七位代码不建列；``!1`` 的复权因子补到六位代码没有因子的日子。
   - 是**两只基金**（如 502000.SH：2015–2020 是分级基金 5020001.SH，之后是另一只 LOF）：拆成两列。
     旧基金用七位代码：``!1`` 的复权因子、以及六位代码在这些日子里（且六位代码自己当天没有因子）的
     日线都归七位代码；六位代码只留新基金的部分。
   - 七位代码与六位代码同一天都有日线时，留六位代码那行、丢七位那行（计数进诊断）；七位代码只在六位代码
     缺行的日子有日线时，用来补齐（2018-12 有几只就是这样补上缺的那天）。

2. **公司行为**（`reconcile_actions`）。只在有行情的日子之间比较：p 是上一个有行情的日子，t 是这一天，
   因子比 R = 因子(t) / 因子(p)，前收盘缺口 G = 收盘(p) / 前收盘(t)。停牌期间发生的拆分、折算、
   分红一律落在停牌后的第一个有行情日。

   - 分红先按经济条款去重：同一只基金、同一登记日（缺失时用实施公告日）、同一每份金额只算一次。
     源对同一次分红常给两行，一行的除息日其实是登记日，另一行才是真正的除息日；候选日都保留，
     落在因子对得上的那一天，都对不上时取 ``net_ex_date`` 非空那行的除息日。
   - 每个有行情日按顺序裁决（P 为上一有效收盘，c 为当天的分红合计，θ = P / (P − c)）：

     | 情形 | 结果 | action_source |
     |---|---|---|
     | 有分红，θ 与 R 相差不超过 0.5% | 记分红 | 1 |
     | 有分红，R 与 1 相差不超过 0.1% | 因子没动，否决分红 | 4 |
     | 有分红，其余 | 记分红，另加折算 R / θ | 2 |
     | 无分红，R 与 1 相差超过 0.1% | 拆分 / 折算 = R | 3 |
     | 因子缺失：有分红且 θ 与 G 相差不超过 0.5%，或无分红 | 记分红；无分红而 \\|G − 1\\| > 0.5% 时拆分 = G | 5 |
     | 最后一个有行情日之后、退市之前的分红（无法核对） | 按公告记在除息日 | 6 |

   与 A 股数据层同一原则：对不上的以复权因子为准。
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import polars as pl

TOLERANCE = 0.005  # 分红与因子比、前收盘缺口的匹配容差
NOISE = 0.001  # 因子比与 1 相差不超过它视为没动（因子只有 4 位小数）


# ---------- 代码复用 ----------


def _code(base, exchange, suffix=""):
    return f"{base}{suffix}.{exchange}"


def resolve_reused(basic, daily, adj):
    """返回 (daily, adj, drop, notes)：改写后的日线和复权因子长表、不建列的七位代码、每个复用代码怎么处理。

    daily、adj 为 (symbol, date, ...) 长表。
    """
    info = {r["ts_code"]: r for r in basic.iter_rows(named=True)}
    markers = sorted({c for c in adj["symbol"].unique().to_list() if "!" in c})
    notes, drop = [], set()
    daily_moves = []  # (from, to, dates)
    adj_moves = []  # (from, to, dates)
    stray = []  # (symbol, dates) 丢弃的七位零星日线
    daily_dates = {
        s: set(d) for s, d in daily.group_by("symbol").agg(pl.col("date")).iter_rows()
    }
    adj_dates = {
        s: set(d) for s, d in adj.group_by("symbol").agg(pl.col("date")).iter_rows()
    }
    for marker in markers:
        base, exchange = marker.split("!")[0], marker.split(".")[1]
        six, seven = _code(base, exchange), _code(base, exchange, "1")
        b6, b7 = info.get(six), info.get(seven)
        marked = adj_dates.get(marker, set())
        own6 = daily_dates.get(six, set())
        own7 = daily_dates.get(seven, set())
        distinct = b7 is not None and (
            b6 is None or (b6["name"], b6["list_date"]) != (b7["name"], b7["list_date"])
        )
        if distinct:
            moved = sorted(
                d for d in own6 & marked if d not in adj_dates.get(six, set())
            )
            if moved:
                daily_moves.append((six, seven, set(moved)))
                inside = own7 & set(moved)
                if inside:
                    stray.append((seven, inside))
            adj_moves.append(
                (
                    marker,
                    seven,
                    {d for d in marked if d not in adj_dates.get(seven, set())},
                )
            )
            notes.append(
                dict(
                    code=six,
                    marker=marker,
                    rule="split",
                    old=seven,
                    moved_quotes=len(moved),
                    first=moved[0] if moved else None,
                    last=moved[-1] if moved else None,
                )
            )
        else:
            inside = own7 & own6
            if inside:
                stray.append((seven, inside))
            if own7 - inside:
                daily_moves.append((seven, six, own7 - inside))
            if b7 is not None or own7:
                drop.add(seven)
            adj_moves.append(
                (marker, six, {d for d in marked if d not in adj_dates.get(six, set())})
            )
            notes.append(
                dict(
                    code=six,
                    marker=marker,
                    rule="merge",
                    old=seven,
                    moved_quotes=len(own7 - inside),
                )
            )
    daily = _apply_moves(daily, daily_moves, stray)
    adj = _apply_moves(adj, adj_moves, [])
    adj = adj.filter(~pl.col("symbol").str.contains("!", literal=True))
    for code, dates in stray:
        notes.append(dict(code=code, rule="stray_dropped", quotes=len(dates)))
    return daily, adj, drop, notes


def _apply_moves(frame, moves, stray):
    if not moves and not stray:
        return frame
    rows = []
    for source, target, dates in moves:
        if dates:
            rows.append(
                pl.DataFrame(
                    dict(
                        symbol=[source] * len(dates),
                        date=sorted(dates),
                        _to=[target] * len(dates),
                    )
                )
            )
    for code, dates in stray:
        rows.append(
            pl.DataFrame(
                dict(
                    symbol=[code] * len(dates),
                    date=sorted(dates),
                    _to=[None] * len(dates),
                ),
                schema=dict(symbol=pl.String, date=pl.String, _to=pl.String),
            )
        )
    if not rows:
        return frame
    plan = pl.concat(rows).unique(["symbol", "date"], keep="first")
    out = frame.join(plan, on=["symbol", "date"], how="left")
    dropped = plan.filter(pl.col("_to").is_null()).select("symbol", "date")
    out = out.join(
        dropped.with_columns(pl.lit(True).alias("_drop")),
        on=["symbol", "date"],
        how="left",
    )
    out = out.filter(pl.col("_drop").is_null()).with_columns(
        pl.coalesce("_to", "symbol").alias("symbol")
    )
    return out.drop("_to", "_drop").unique(["symbol", "date"], keep="first")


# ---------- 分红去重 ----------


def distributions(fund_div):
    """原始分红表（实施）→ 每次分配一行：symbol、cash、candidates（候选除息日，升序）、preferred。"""
    frame = fund_div.filter(
        (pl.col("div_proc") == "实施")
        & pl.col("ex_date").is_not_null()
        & (pl.col("div_cash").cast(pl.Float64) > 0)
    ).with_columns(
        pl.col("div_cash").cast(pl.Float64).alias("cash"),
        pl.coalesce("record_date", "imp_anndate", "ann_date", "ex_date").alias("_key"),
    )
    grouped = frame.group_by("ts_code", "_key", "cash").agg(
        pl.col("ex_date").unique().sort().alias("candidates"),
        pl.col("net_ex_date").drop_nulls().unique().sort().alias("_net"),
        pl.col("ex_date")
        .filter(pl.col("net_ex_date").is_not_null())
        .min()
        .alias("_with_net"),
        pl.len().alias("rows"),
    )
    grouped = grouped.with_columns(
        pl.concat_list("candidates", "_net")
        .list.unique()
        .list.sort()
        .alias("candidates"),
        pl.coalesce("_with_net", pl.col("candidates").list.first()).alias("preferred"),
    )
    return grouped.select(
        pl.col("ts_code").alias("symbol"),
        "cash",
        "candidates",
        "preferred",
        "rows",
    ).sort("symbol", "preferred")


# ---------- 裁决 ----------


def reconcile_actions(
    dates, symbols, close, previous_close, adj, has_quote, stop, dist
):
    """返回 split、cash_dividend、action_source（T×N）和诊断。

    stop：每列的退市行（不退市为 T）。dist：`distributions` 的结果。
    """
    t, n = close.shape
    split = np.ones((t, n))
    cash = np.zeros((t, n))
    source = np.zeros((t, n), dtype=np.uint8)
    lookup = {s: j for j, s in enumerate(symbols)}
    per_col = defaultdict(list)
    for row in dist.iter_rows(named=True):
        j = lookup.get(row["symbol"])
        if j is not None:
            per_col[j].append(row)
    counts = dict(
        distributions=0,
        moved_to_matching_date=0,
        rejected=0,
        unverifiable=0,
        dropped_outside=0,
        no_factor_cells=0,
    )
    rejected, moved = [], []
    for j in range(n):
        q = np.flatnonzero(has_quote[:, j])
        if not len(q):
            continue
        prev = np.full(t, -1)
        prev[q[1:]] = q[:-1]
        with np.errstate(all="ignore"):
            ratio = np.full(t, np.nan)
            ratio[q[1:]] = adj[q[1:], j] / adj[q[:-1], j]
            gap = np.full(t, np.nan)
            gap[q[1:]] = close[q[:-1], j] / previous_close[q[1:], j]
        counts["no_factor_cells"] += int((~np.isfinite(adj[q, j])).sum())
        assigned = defaultdict(float)
        for d in per_col.get(j, []):
            counts["distributions"] += 1
            target = _place(d, dates, q, prev, close[:, j], ratio, gap)
            if target is None:
                row = int(np.searchsorted(dates, d["preferred"]))
                if q[0] < row < min(stop[j], t):
                    cash[row, j] += d["cash"]
                    source[row, j] = 6
                    counts["unverifiable"] += 1
                else:
                    counts["dropped_outside"] += 1
                continue
            row, matched = target
            if matched and str(dates[row]) != _first_quote_date(
                d["preferred"], dates, q
            ):
                counts["moved_to_matching_date"] += 1
                if len(moved) < 20:
                    moved.append(
                        dict(
                            symbol=str(symbols[j]),
                            preferred=d["preferred"],
                            date=str(dates[row]),
                        )
                    )
            assigned[row] += d["cash"]
        rows = set(assigned)
        changed = np.flatnonzero(np.isfinite(ratio) & (np.abs(ratio - 1) > NOISE))
        rows.update(int(r) for r in changed)
        missing = np.flatnonzero(
            ~np.isfinite(ratio)
            & (prev >= 0)
            & np.isfinite(gap)
            & (np.abs(gap - 1) > TOLERANCE)
        )
        rows.update(int(r) for r in missing)
        for row in sorted(rows):
            c = assigned.get(row, 0.0)
            s, money, code = _decide(close[prev[row], j], c, ratio[row], gap[row])
            split[row, j], cash[row, j], source[row, j] = s, money, code
            if code == 4:
                counts["rejected"] += 1
                if len(rejected) < 20:
                    rejected.append(
                        dict(symbol=str(symbols[j]), date=str(dates[row]), cash=c)
                    )
    diag = dict(
        counts,
        source_counts={
            str(k): int(v)
            for k, v in zip(*np.unique(source, return_counts=True), strict=True)
        },
        rejected_examples=rejected,
        moved_examples=moved,
    )
    return split, cash, source, diag


def _first_quote_date(day, dates, q):
    row = int(np.searchsorted(dates, day))
    k = int(np.searchsorted(q, row))
    return str(dates[q[k]]) if k < len(q) else None


def _place(d, dates, q, prev, close, ratio, gap):
    """候选除息日 → 有行情日。返回 (行, 是否与因子 / 缺口对上)；之后没有有行情日时为 None。"""
    options = set()
    for day in d["candidates"]:
        k = int(np.searchsorted(q, int(np.searchsorted(dates, day))))
        if 0 < k < len(q):
            options.add(int(q[k]))
    if not options:
        return None
    for row in sorted(options):
        p = close[prev[row]]
        if not (np.isfinite(p) and p > d["cash"]):
            continue
        theta = p / (p - d["cash"])
        reference = ratio[row] if np.isfinite(ratio[row]) else gap[row]
        if np.isfinite(reference) and abs(theta / reference - 1) <= TOLERANCE:
            return row, True
    k = int(np.searchsorted(q, int(np.searchsorted(dates, d["preferred"]))))
    if 0 < k < len(q):
        return int(q[k]), False
    return min(options), False


def _decide(previous, c, ratio, gap):
    theta = previous / (previous - c) if c > 0 and previous > c else np.nan
    if np.isfinite(ratio) and ratio > 0:
        if c > 0:
            if np.isfinite(theta) and abs(theta / ratio - 1) <= TOLERANCE:
                return 1.0, c, 1
            if abs(ratio - 1) <= NOISE:
                return 1.0, 0.0, 4
            if np.isfinite(theta):
                return ratio / theta, c, 2
            return ratio, 0.0, 3
        return ratio, 0.0, 3
    # 因子缺失：退回交易所前收盘缺口。
    if c > 0:
        if np.isfinite(theta) and np.isfinite(gap) and abs(theta / gap - 1) > TOLERANCE:
            return gap / theta, c, 5
        return 1.0, c, 5
    if np.isfinite(gap) and abs(gap - 1) > TOLERANCE:
        return gap, 0.0, 5
    return 1.0, 0.0, 0
