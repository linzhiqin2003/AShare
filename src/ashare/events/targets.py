"""事件转目标持仓：输出 ``panelbt.Targets``（稀疏结构）。

规则（``docs/DESIGN.md`` 6.6）：

- 第 ``avail + entry_lag`` 个交易日（「决策日」）开盘买入，持有 ``holding`` 个交易日，
  在买入日之后第 ``holding`` 个交易日开盘卖出。``panelbt`` 第 t 行是第 t 日成交时要达到
  的目标、内核不移位，所以这里直接把目标写在成交日那一行。
- 买不进（``can_buy`` 为假：停牌、开盘涨停）：``defer=0`` 放弃；``defer=K`` 顺延，
  在决策日起 K 个交易日内第一个可买日买入，都买不进就放弃。持有期从实际买入日算。
- 同一只股票事件重叠（新事件的决策日早于已有持仓的卖出日；正好在卖出日当天的不算重叠，
  按新开一份处理，与原持仓首尾相接、``panelbt`` 里表现为继续持有）：
  ``"extend"`` 把卖出日延到「新决策日 + holding」，数值取最新事件；
  ``"ignore"`` 忽略新事件；``"stack"`` 新开一份，权重叠加。延长不需要买入，不检查 ``can_buy``。
- 同时持有数上限 ``max_positions``：按决策日先后处理；同一决策日按 ``priority`` 列从大到小，
  再相同时按 :func:`tie_key`（证券代码、公告日和 ``seed`` 的固定哈希）排序——只取决于事件
  自身，与代码大小、事件表顺序无关，也不看其他事件。满了就放弃；名额在决策日占用（只看
  决策日已知的情况）。名额释放：不给 ``can_sell`` 时在计划卖出日开盘释放；给了 ``can_sell``
  时在计划卖出日起第一个可卖日释放（跌停、停牌卖不出时 ``panelbt`` 会顺延卖单，名额也跟着
  多占几天）。顺延期满仍没买到的名额在最后一次尝试的次日释放。同一证券的买单还在顺延中时
  来的新事件并入这张买单（``absorbed``），不另占名额。
- 权重：``"equal"`` 当日所有持仓份额等权、满仓；``"value"`` 按事件数值（须为正）
  占当日合计的比例、满仓；``"fixed"`` 每份固定 ``fixed_weight``（不给时取
  ``1 / max_positions``），只在持仓集合变化的日子按这个固定值重写，不按市值漂移再平衡。
  之后单只权重截到 ``max_weight``（多出的留现金），合计超过 1 时同比例缩到 1。
- 只在持仓集合变化的交易日写一行（其余日 ``panelbt`` 不调仓）；那一行对所有持仓重新
  按规则给权重，会产生少量再平衡交易，可用 ``panelbt.Config(trade_band=...)`` 抑制。

最小例子::

    import numpy as np, polars as pl
    from ashare.events import to_targets

    dates = np.array([f"202401{d:02d}" for d in range(2, 12)])
    events = pl.DataFrame({"symbol": ["A"], "date": ["20240103"], "kind": ["demo"],
                           "timing": ["after_close"]})
    out = to_targets(events, dates, np.array(["A", "B"]), holding=3)
    out.targets.days        # array([2, 5])：第 2 天买入、第 5 天清仓
"""

from __future__ import annotations

import heapq
import zlib
from dataclasses import dataclass

import numpy as np
import polars as pl

from .table import AlignedEvents, align

WEIGHTS = ("equal", "value", "fixed")
OVERLAPS = ("extend", "ignore", "stack")


@dataclass(frozen=True)
class EventTargets:
    """``targets``：``panelbt.Targets``；``positions``：每份持仓（证券、买入日、卖出日、
    数值、合并的事件数）；``events``：每个事件的处理结果 ``status``；``counts``：各结果的条数。"""

    targets: object
    positions: pl.DataFrame
    events: pl.DataFrame
    counts: dict


def tie_key(symbols, dates, seed=0):
    """同一决策日、同一 ``priority`` 时的先后键：``crc32(f"{seed}|{symbol}|{date}")``。

    只由事件自己的证券代码和公告日决定，不随代码大小、事件表顺序或其他事件变化；
    换 ``seed`` 可以检查结果对这个次序的敏感程度。
    """
    return np.array(
        [
            zlib.crc32(f"{seed}|{s}|{d}".encode())
            for s, d in zip(symbols, dates, strict=True)
        ],
        dtype=np.int64,
    )


def _sell_day(can_sell, c, day, t):
    """计划卖出日 ``day`` 起第一个可卖日（没有返回 ``t``）；不给 ``can_sell`` 时就是 ``day``。"""
    if can_sell is None or day >= t:
        return day
    ok = can_sell[day:, c]
    return day + int(ok.argmax()) if ok.any() else t


def _buy_days(e0, col, can_buy, defer, t):
    """决策日起 ``defer`` 天内第一个可买日；没有返回 -1。"""
    buy = np.full(len(e0), -1, dtype=np.int64)
    for k in range(defer + 1):
        day = e0 + k
        open_ = (buy < 0) & (day < t)
        if can_buy is not None:
            idx = np.flatnonzero(open_)
            ok = np.zeros(len(e0), dtype=bool)
            ok[idx] = can_buy[day[idx], col[idx]]
            open_ &= ok
        buy[open_] = day[open_]
    return buy


def _select(
    e0,
    buy,
    col,
    value,
    priority,
    tie,
    holding,
    overlap,
    max_positions,
    defer,
    revalue,
    can_sell=None,
    t=None,
):
    """按时间顺序决定每个事件开仓、延长、忽略或放弃；返回持仓、计权分段与状态。

    计权分段：``revalue`` 为真（按数值加权）时，延长事件从它的决策日起改用新数值，
    把持仓切成前后两段；之前那段保持旧数值（否则后来的事件会改写更早的目标，即前视）。
    """
    order = np.lexsort((np.arange(len(e0)), tie, -priority, e0))
    status = np.empty(len(e0), dtype=object)
    entry, exit_, pcol, pval, nev, cur = [], [], [], [], [], []
    seg_col, seg_entry, seg_exit, seg_val = [], [], [], []
    release: list[int] = []  # 每个名额的计划结束日（判断同一证券的事件是否重叠）
    free: list[int] = []  # 每个名额实际让出的日子（有 can_sell 时是实际卖出日）
    held: dict[int, int] = {}  # 证券 -> 最近一个名额
    sid_of: dict[int, int] = {}  # 名额 -> 持仓（买入失败的名额没有）
    heap: list[tuple[int, int]] = []
    active = 0
    for i in order:
        d = int(e0[i])
        while heap and heap[0][0] <= d:
            x, rid = heapq.heappop(heap)
            if free[rid] == x:
                active -= 1
        c = int(col[i])
        rid = held.get(c, -1)
        if overlap != "stack" and rid >= 0 and release[rid] > d:
            sid = sid_of.get(rid, -1)
            if overlap == "ignore":
                status[i] = "ignored"
            elif sid < 0:
                # 同一证券的买单还在顺延中且最终没买到：新事件并入这张买单，不另占名额。
                status[i] = "absorbed"
            else:
                new_exit = max(d + holding, exit_[sid])
                if new_exit > exit_[sid]:
                    exit_[sid] = new_exit
                    release[rid] = new_exit
                    nf = _sell_day(can_sell, c, new_exit, t)
                    if (
                        nf != free[rid]
                    ):  # 卖不出时可能不变：不变就不重复入堆（否则会重复释放）
                        free[rid] = nf
                        heapq.heappush(heap, (nf, rid))
                g = cur[sid]
                if revalue and seg_entry[g] < d:
                    seg_exit[g] = d
                    cur[sid] = len(seg_col)
                    seg_col.append(c)
                    seg_entry.append(d)
                    seg_exit.append(new_exit)
                    seg_val.append(value[i])
                else:
                    seg_exit[g] = new_exit
                    seg_val[g] = value[i]
                pval[sid] = value[i]
                nev[sid] += 1
                status[i] = "extended"
            continue
        # 名额在决策日决定，不看之后几天能否买到（否则会用到决策日之后的行情）。
        if max_positions is not None and active >= max_positions:
            status[i] = "capacity"
            continue
        rid = len(release)
        active += 1
        held[c] = rid
        if buy[i] < 0:
            release.append(d + defer + 1)
            free.append(d + defer + 1)
            heapq.heappush(heap, (d + defer + 1, rid))
            status[i] = "not_buyable"
            continue
        b = int(buy[i])
        release.append(b + holding)
        free.append(_sell_day(can_sell, c, b + holding, t))
        heapq.heappush(heap, (free[rid], rid))
        sid_of[rid] = len(entry)
        entry.append(b)
        exit_.append(b + holding)
        pcol.append(c)
        pval.append(value[i])
        nev.append(1)
        cur.append(len(seg_col))
        seg_col.append(c)
        seg_entry.append(b)
        seg_exit.append(b + holding)
        seg_val.append(value[i])
        status[i] = "opened" if b == d else "deferred"
    as_i = lambda x: np.asarray(x, dtype=np.int64)  # noqa: E731
    positions = (
        as_i(entry),
        as_i(exit_),
        as_i(pcol),
        np.asarray(pval, dtype=np.float64),
        as_i(nev),
    )
    segments = (
        as_i(seg_entry),
        as_i(seg_exit),
        as_i(seg_col),
        np.asarray(seg_val, dtype=np.float64),
    )
    return positions, segments, status


def _weights(entry, exit_, col, value, t, n, weight, fixed_weight, max_weight):
    """把持仓区间展开成调仓日 × 证券的稀疏权重（向量化）。"""
    import panelbt

    exit_c = np.minimum(exit_, t)
    days = np.unique(np.concatenate([entry, exit_c[exit_c < t]]))
    first = np.searchsorted(days, entry)
    last = np.searchsorted(days, exit_c)  # exit_c == t 时为 len(days)
    length = last - first
    total = int(length.sum())
    slot = np.repeat(np.arange(len(entry)), length)
    offset = np.arange(total) - np.repeat(np.cumsum(length) - length, length)
    row = np.repeat(first, length) + offset
    if weight == "fixed":
        raw = np.full(total, float(fixed_weight))
    else:
        raw = np.ones(total) if weight == "equal" else value[slot]
        denom = np.bincount(row, weights=raw, minlength=len(days))
        raw = raw / denom[row]
    key = row * n + col[slot]
    uniq, inverse = np.unique(key, return_inverse=True)
    w = np.bincount(inverse, weights=raw, minlength=len(uniq))
    rows, cols = uniq // n, uniq % n
    if max_weight is not None:
        w = np.minimum(w, max_weight)
    gross = np.bincount(rows, weights=w, minlength=len(days))
    scale = np.where(gross > 1.0, 1.0 / np.where(gross > 0, gross, 1.0), 1.0)
    w = w * scale[rows]
    indptr = np.concatenate([[0], np.cumsum(np.bincount(rows, minlength=len(days)))])
    return panelbt.Targets(days.astype(np.int64), indptr.astype(np.int64), cols, w)


def to_targets(
    events,
    dates,
    symbols,
    can_buy=None,
    *,
    can_sell=None,
    entry_lag: int = 0,
    holding: int = 20,
    weight: str = "equal",
    value: str | None = None,
    fixed_weight: float | None = None,
    max_positions: int | None = None,
    max_weight: float | None = None,
    overlap: str = "extend",
    defer: int = 0,
    priority: str | None = None,
    seed: int = 0,
) -> EventTargets:
    """把事件表变成 ``panelbt.Targets``；规则见模块说明。

    ``events``：事件表（polars DataFrame）或 :func:`align` 的结果；``dates``/``symbols``：
    面板日历与证券；``can_buy``：T×N 布尔（成交日能否买入，通常是面板的 ``can_buy_open``），
    不给表示都能买；``can_sell``：T×N 布尔（通常是 ``can_sell_open``），给了名额按实际卖出日
    释放；``value``：数值列名（``weight="value"`` 时必需）；``priority``：同一天名额不够时的
    先后（列名，从大到小），默认取 ``value``；``seed``：再相同时的哈希种子（见 :func:`tie_key`）。
    """
    dates = np.asarray(dates).astype("U8")
    symbols = np.asarray(symbols)
    t, n = len(dates), len(symbols)
    if holding < 1 or entry_lag < 0 or defer < 0:
        raise ValueError("要求 holding >= 1、entry_lag >= 0、defer >= 0")
    if weight not in WEIGHTS:
        raise ValueError(f"weight 只能是 {WEIGHTS}")
    if overlap not in OVERLAPS:
        raise ValueError(f"overlap 只能是 {OVERLAPS}")
    if weight == "value" and value is None:
        raise ValueError('weight="value" 需要给出 value 列名')
    if weight == "fixed" and not (fixed_weight and fixed_weight > 0):
        if max_positions:
            fixed_weight = 1.0 / max_positions
        else:
            raise ValueError('weight="fixed" 需要 fixed_weight 或 max_positions')
    if max_positions is not None and max_positions < 1:
        raise ValueError("max_positions 必须 >= 1")
    if max_weight is not None and not (0 < max_weight <= 1):
        raise ValueError("max_weight 必须在 (0, 1]")
    if dates.ndim != 1 or symbols.ndim != 1:
        raise ValueError("dates、symbols 必须是一维数组")
    if can_buy is not None:
        can_buy = np.asarray(can_buy, dtype=bool)
        if can_buy.shape != (t, n):
            raise ValueError(f"can_buy 形状应为 {(t, n)}，实际 {can_buy.shape}")
    if can_sell is not None:
        can_sell = np.asarray(can_sell, dtype=bool)
        if can_sell.shape != (t, n):
            raise ValueError(f"can_sell 形状应为 {(t, n)}，实际 {can_sell.shape}")
    aligned = (
        events if isinstance(events, AlignedEvents) else align(events, dates, symbols)
    )
    frame = aligned.frame
    vals = aligned.column(value) if value is not None else np.ones(len(aligned))
    if weight == "value" and not np.all(np.isfinite(vals) & (vals > 0)):
        raise ValueError(f'weight="value" 要求 {value} 全部为有限正数；先过滤或变换')
    prio_name = priority or value
    prio = (
        aligned.column(prio_name) if prio_name is not None else np.zeros(len(aligned))
    )
    prio = np.where(np.isfinite(prio), prio, -np.inf)
    tie = tie_key(frame["symbol"].to_list(), frame["date"].to_list(), seed)
    e0 = aligned.avail + entry_lag
    inside = e0 < t
    buy = _buy_days(np.where(inside, e0, 0), aligned.col, can_buy, defer, t)
    buy[~inside] = -1
    sel = np.flatnonzero(inside)
    (entry, exit_, col, pval, nev), segments, status_sel = _select(
        e0[sel],
        buy[sel],
        aligned.col[sel],
        vals[sel],
        prio[sel],
        tie[sel],
        holding,
        overlap,
        max_positions,
        defer,
        weight == "value",
        can_sell,
        t,
    )
    status = np.full(len(aligned), "after_panel_end", dtype=object)
    status[sel] = status_sel
    targets = _weights(*segments, t, n, weight, fixed_weight, max_weight)
    positions = pl.DataFrame(
        {
            "symbol": symbols[col] if len(col) else np.array([], dtype=str),
            "col": col,
            "entry": entry,
            "exit": exit_,
            "entry_date": dates[entry] if len(entry) else np.array([], dtype="U8"),
            "exit_date": np.where(exit_ < t, dates[np.minimum(exit_, t - 1)], "")
            if len(exit_)
            else np.array([], dtype="U8"),
            "value": pval,
            "events": nev,
        },
        schema_overrides={
            "symbol": pl.String,
            "entry_date": pl.String,
            "exit_date": pl.String,
        },
    )
    frame = frame.with_columns(pl.Series("status", status.astype(str), dtype=pl.String))
    counts = dict(zip(*np.unique(status.astype(str), return_counts=True), strict=True))
    counts = {k: int(v) for k, v in counts.items()}
    counts.update({f"dropped_{k}": v for k, v in aligned.dropped.items()})
    return EventTargets(targets, positions, frame, counts)
