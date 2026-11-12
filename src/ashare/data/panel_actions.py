"""以复权因子裁决持有人公司行为；合成事件是参考价再投资口径，不是假称真实送股。"""

import math
from collections import Counter, defaultdict

import numpy as np

from .panel_events import dividend_rows

FACTOR_NOISE = 0.001
FACTOR_EPSILON = 1e-10


def prior_prices(close):
    """停牌缺口只用于事件核对找前收，不改价格面板。"""
    rows = np.arange(len(close), dtype=np.int32)[:, None]
    indices = np.maximum.accumulate(np.where(np.isfinite(close), rows, -1), axis=0)
    indices = np.concatenate((np.full((1, close.shape[1]), -1), indices[:-1]))
    return np.where(
        indices >= 0, np.take_along_axis(close, np.maximum(indices, 0), axis=0), np.nan
    )


def implied_ratio(previous, split, cash):
    with np.errstate(all="ignore"):
        # 零派现时前收约去，即使历史首个价格缺失也可核验纯股数事件。
        return np.where(cash == 0, split, previous * split / (previous - cash))


def choose_action(records, ratio, previous, *, tolerance, moved=False):
    """有序候选：全部、仅现金、仅送转、逐条；第一个匹配者胜出。"""
    if not np.isfinite(ratio) or ratio <= 0:
        return 1.0, 0.0, 4 if records else 0
    changed = abs(ratio - 1) > FACTOR_EPSILON
    if records and changed:
        split = 1 + math.fsum(r.get("stk_div") or 0 for r in records)
        cash = math.fsum(r.get("cash_div_tax") or 0 for r in records)
        original = not moved and all(r.get("_records", 1) == 1 for r in records)
        candidates = [
            (split, cash, 1 if original else 2),
            (1.0, cash, 2),
            (split, 0.0, 2),
        ]
        candidates.extend(
            (1 + (r.get("stk_div") or 0), r.get("cash_div_tax") or 0, 2)
            for r in records
        )
        for shares, money, source in candidates:
            if shares <= 0 or (shares == 1 and money == 0):
                continue
            if money and (not np.isfinite(previous) or previous <= money):
                continue
            theta = shares if money == 0 else previous * shares / (previous - money)
            if abs(theta / ratio - 1) <= tolerance:
                return shares, money, source
    if abs(ratio - 1) > FACTOR_NOISE:
        return ratio, 0.0, 3
    return 1.0, 0.0, 4 if records else 0


def reconciliation_check(
    previous, factors, split, cash, listed, *, tolerance=0.005, prior_factor=None
):
    old = np.concatenate(
        (
            np.full((1, factors.shape[1]), np.nan)
            if prior_factor is None
            else prior_factor[None, :],
            factors[:-1],
        )
    )
    with np.errstate(all="ignore"):
        ratio = factors / old
        theta = implied_ratio(previous, split, cash)
        deviation = theta / ratio - 1
    valid = listed & np.isfinite(ratio) & (ratio > 0)
    event = listed & ((split != 1) | (cash != 0))
    changed = valid & (np.abs(ratio - 1) > FACTOR_EPSILON)
    bad = (changed & (~np.isfinite(deviation) | (np.abs(deviation) > tolerance))) | (
        event & ~changed
    )
    return (
        dict(
            checked=int((changed | event).sum()),
            failed=int(bad.sum()),
            over_one_percent=int((valid & (np.abs(deviation) > 0.01)).sum()),
        ),
        bad,
        ratio,
        theta,
    )


def reconcile(
    arrays, dates, symbols, table, *, tolerance=0.005, allow_unreconciled=False
):
    """分列块重核全部事件，保证历史修订增量时诊断与全量完全一致。"""
    if not 0 < tolerance < 1:
        raise ValueError("action tolerance 必须在 0 与 1 之间")
    valid, diagnostics = dividend_rows(table, dates, symbols)
    lookup = {s: j for j, s in enumerate(symbols)}
    groups = defaultdict(list)
    for row in valid.sort("ts_code", "ex_date", "_stock_key", "_cash_key").iter_rows(
        named=True
    ):
        if row["ts_code"] in lookup and (row.get("stk_div") or row.get("cash_div_tax")):
            groups[lookup[row["ts_code"]]].append(row)
    counts, years = Counter(), Counter()
    checks = dict(checked=0, failed=0, over_one_percent=0)
    shifted = 0
    examples, synthetic_examples = [], []
    for left in range(0, len(symbols), 128):
        right = min(left + 128, len(symbols))
        close = np.array(arrays["close"][:, left:right])
        factor = np.array(arrays["adj_factor"][:, left:right])
        listed = np.array(arrays["is_listed"][:, left:right])
        previous = prior_prices(close)
        with np.errstate(all="ignore"):
            ratio = factor / np.concatenate(
                (np.full((1, right - left), np.nan), factor[:-1])
            )
        valid_ratio = listed & np.isfinite(ratio) & (ratio > 0)
        significant = valid_ratio & (np.abs(ratio - 1) > FACTOR_NOISE)
        split = np.where(significant, ratio, 1.0)
        cash = np.zeros_like(split)
        source = np.where(significant, 3, 0).astype(np.uint8)
        for local, j in enumerate(range(left, right)):
            assigned = defaultdict(list)
            moved_targets = set()
            for row in groups.get(j, []):
                origin = int(np.searchsorted(dates, row["ex_date"]))
                if origin >= len(dates):
                    continue
                target = origin
                exact_change = (
                    valid_ratio[origin, local]
                    and abs(ratio[origin, local] - 1) > FACTOR_EPSILON
                )
                if not exact_change and (
                    dates[origin] != row["ex_date"]
                    or not np.isfinite(close[origin, local])
                ):
                    nearby = [
                        i
                        for i in range(max(1, origin - 3), min(len(dates), origin + 4))
                        if significant[i, local] and row["_known"] <= dates[i]
                    ]
                    if nearby:
                        target = min(nearby, key=lambda i: (abs(i - origin), i))
                if row["_known"] > dates[target] or not listed[target, local]:
                    continue
                assigned[target].append(row)
                if target != origin:
                    moved_targets.add(target)
                    shifted += 1
            for i, records in assigned.items():
                split[i, local], cash[i, local], source[i, local] = choose_action(
                    records,
                    ratio[i, local],
                    previous[i, local],
                    tolerance=tolerance,
                    moved=i in moved_targets,
                )
        split[~listed], cash[~listed] = np.nan, np.nan
        source[~listed] = 0
        check, bad, ratios, theta = reconciliation_check(
            previous, factor, split, cash, listed, tolerance=tolerance
        )
        for key, value in check.items():
            checks[key] += value
        for code, count in zip(*np.unique(source, return_counts=True), strict=True):
            counts[int(code)] += int(count)
        for year in sorted({d[:4] for d in dates}):
            years[year] += int((source[np.char.startswith(dates, year)] == 3).sum())
        for i, j in np.argwhere(bad)[: max(0, 20 - len(examples))]:
            examples.append(
                dict(
                    symbol=str(symbols[left + j]),
                    date=str(dates[i]),
                    ratio=float(ratios[i, j]),
                    theta=float(theta[i, j]),
                )
            )
        for i, j in np.argwhere(source == 3)[: max(0, 20 - len(synthetic_examples))]:
            synthetic_examples.append(
                dict(
                    symbol=str(symbols[left + j]),
                    date=str(dates[i]),
                    ratio=float(ratios[i, j]),
                    previous=float(previous[i, j])
                    if np.isfinite(previous[i, j])
                    else None,
                    split=float(split[i, j]),
                    cash=0.0,
                    theta=float(theta[i, j]),
                )
            )
        for name, values in (
            ("split", split),
            ("cash_dividend", cash),
            ("action_source", source),
        ):
            # 未改变的历史块不触发写时复制；普通追加只写发生变化的日期行。
            existing = arrays[name][:, left:right]
            equal = (existing == values) | (np.isnan(existing) & np.isnan(values))
            changed_rows = np.flatnonzero(~equal.all(axis=1))
            arrays[name][changed_rows, left:right] = values[changed_rows]
    result = dict(
        **checks,
        tolerance=tolerance,
        factor_noise=FACTOR_NOISE,
        source_counts={str(k): counts[k] for k in range(5)},
        synthetic_by_year=dict(sorted(years.items())),
        shifted_records=shifted,
        dividends=diagnostics,
        examples=examples,
        synthetic_examples=synthetic_examples,
    )
    if checks["failed"] and not allow_unreconciled:
        raise ValueError(
            f"公司行为与复权因子对账失败 {checks['failed']} 格；{examples[:3]}；核查后仅可显式 --allow-unreconciled 放行"
        )
    return result
