"""直白的参考实现：逐日、逐事件循环，不调用 ashare.events 的任何代码。"""

import zlib

import numpy as np


def trade_day(dates, date, timing):
    """事件第一个可成交交易日的下标（逐个比较）。"""
    for i, d in enumerate(dates):
        if timing == "after_close" and d > date:
            return i
        if timing == "pre_open" and d >= date:
            return i
    return len(dates)


def reference_targets(
    events,
    dates,
    symbols,
    can_buy,
    *,
    can_sell=None,
    seed=0,
    entry_lag,
    holding,
    weight,
    fixed_weight,
    max_positions,
    max_weight,
    overlap,
    defer,
):
    """返回 T×N 稠密目标（不调仓的行全为 NaN）。

    ``events``：字典列表，键 symbol/date/timing/value/priority。同日同 priority 的先后按
    ``crc32(f"{seed}|{symbol}|{date}")``，再按给出的顺序。
    """

    def sell(c, day):
        if can_sell is None:
            return day
        while day < t and not can_sell[day, c]:
            day += 1
        return day

    t, n = len(dates), len(symbols)
    col_of = {s: i for i, s in enumerate(symbols)}
    decided = []
    for k, ev in enumerate(events):
        if ev["symbol"] not in col_of or ev["date"] < dates[0]:
            continue
        a = trade_day(dates, ev["date"], ev["timing"])
        d = a + entry_lag
        if a >= t or d >= t:
            continue
        tie = zlib.crc32(f"{seed}|{ev['symbol']}|{ev['date']}".encode())
        decided.append((d, -ev["priority"], tie, k, col_of[ev["symbol"]], ev["value"]))
    decided.sort()
    revalued = set()
    # 每个名额：[col, entry 或 None（没买到）, 计划结束日, value, 决策日, 实际让出日]
    slots = []
    for d, _, _, _, c, v in decided:
        live = [s for s in slots if s[4] <= d < s[5]]
        mine = [s for s in slots if s[0] == c and s[4] <= d < s[2]]
        if overlap != "stack" and mine:
            s = mine[-1]
            if overlap == "extend" and s[1] is not None:
                if d + holding > s[2]:
                    s[2] = d + holding
                    s[5] = sell(c, s[2])
                if s[1] < d:
                    s[3].append((d, v))  # 已经持有：从今天起用新数值
                    revalued.add(d)
                else:
                    s[3] = [(s[1], v)]  # 还没买入：直接改
            continue
        if max_positions is not None and len(live) >= max_positions:
            continue
        buy = None
        for k in range(defer + 1):
            if d + k < t and (can_buy is None or can_buy[d + k, c]):
                buy = d + k
                break
        if buy is None:
            slots.append([c, None, d + defer + 1, [(d, v)], d, d + defer + 1])
        else:
            end = buy + holding
            slots.append([c, buy, end, [(buy, v)], d, sell(c, end)])
    slots = [s for s in slots if s[1] is not None]
    dense = np.full((t, n), np.nan)
    previous = set()
    for day in range(t):
        active = [i for i, s in enumerate(slots) if s[1] <= day < s[2]]
        if set(active) == previous and not (weight == "value" and day in revalued):
            continue
        previous = set(active)
        row = np.zeros(n)
        total = 0.0
        val = {i: [v for f, v in slots[i][3] if f <= day][-1] for i in active}
        for i in active:
            total += 1.0 if weight == "equal" else val[i]
        for i in active:
            if weight == "fixed":
                w = fixed_weight
            elif weight == "equal":
                w = 1.0 / total
            else:
                w = val[i] / total
            row[slots[i][0]] += w
        if max_weight is not None:
            row = np.minimum(row, max_weight)
        if row.sum() > 1.0:
            row = row / row.sum()
        dense[day] = row
    return dense


def to_dense(targets, t, n):
    out = np.full((t, n), np.nan)
    for r, day in enumerate(targets.days):
        out[day] = 0.0
        lo, hi = targets.indptr[r], targets.indptr[r + 1]
        out[day, targets.cols[lo:hi]] = targets.weights[lo:hi]
    return out


def reference_car(events, dates, symbols, returns, benchmark, a, b):
    """逐事件累计异常收益；窗口出界的事件跳过；缺失记 0。返回 (CAR 列表, 可成交日列表)。"""
    t = len(dates)
    col_of = {s: i for i, s in enumerate(symbols)}
    cars, days = [], []
    for ev in events:
        if ev["symbol"] not in col_of or ev["date"] < dates[0]:
            continue
        day = trade_day(dates, ev["date"], ev["timing"])
        if day >= t or day + a < 0 or day + b >= t:
            continue
        c = col_of[ev["symbol"]]
        total = 0.0
        for k in range(a, b + 1):
            bench = (
                benchmark[day + k] if np.ndim(benchmark) == 1 else benchmark[day + k, c]
            )
            x = returns[day + k, c] - bench
            if np.isfinite(x):
                total += x
        cars.append(total)
        days.append(day)
    return np.array(cars), np.array(days)


def cluster_se(x, groups):
    """均值的聚类标准误，逐组循环（CR1）。"""
    x = np.asarray(x, dtype=float)
    mean = x.mean()
    labels = sorted(set(groups))
    g = len(labels)
    total = 0.0
    for lab in labels:
        s = sum(x[i] - mean for i in range(len(x)) if groups[i] == lab)
        total += s * s
    return np.sqrt(total / len(x) ** 2 * g / (g - 1))
