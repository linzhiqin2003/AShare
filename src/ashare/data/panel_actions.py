"""以复权因子裁决持有人公司行为；合成事件是参考价再投资口径，不是假称真实送股。"""

import math
from collections import Counter, defaultdict

import numpy as np

from .panel_events import dividend_rows

FACTOR_NOISE = 0.001
FACTOR_EPSILON = 1e-10
# 差异化分红：回购专户股份不参与分配，交易所按摊薄后的虚拟比例 k 计算参考价。
# 回购专户持股法定上限 10%，k 低于 0.90 需另查原因；0.85 留出放弃分红等余量。
DILUTION_MIN = 0.85


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


def dilution_ratio(previous, ratio, split, cash):
    """公告值按同一比例 k 摊薄后恰好解释因子比时的 k；无解为 NaN。

    参考价 (P - k*c) / (1 + k*s) 与因子比 a 一致：k = P*(a-1) / (a*c + P*s)。
    """
    shares = split - 1
    if cash == 0:
        return (ratio - 1) / shares if shares > 0 else math.nan
    denominator = ratio * cash + previous * shares
    if not np.isfinite(previous) or previous <= 0 or not denominator > 0:
        return math.nan
    return previous * (ratio - 1) / denominator


def choose_action(
    records, ratio, previous, *, tolerance, moved=False, dilution_min=DILUTION_MIN
):
    """有序候选：全部、仅现金、仅送转、逐条、差异化摊薄；第一个匹配者胜出。"""
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
        # 都对不上时才试摊薄：记持有人实得的公告值，参考价差异由 k 解释。
        k = dilution_ratio(previous, ratio, split, cash)
        if (split > 1 or cash > 0) and dilution_min <= k < 1:
            return split, cash, 5
    if abs(ratio - 1) > FACTOR_NOISE:
        return ratio, 0.0, 3
    return 1.0, 0.0, 4 if records else 0


def dilution_ratios(previous, ratio, split, cash):
    """dilution_ratio 的矩阵版本，供构建期对账；校验另有独立实现。"""
    shares = split - 1
    with np.errstate(all="ignore"):
        denominator = ratio * cash + previous * shares
        general = np.where(
            np.isfinite(previous) & (previous > 0) & (denominator > 0),
            previous * (ratio - 1) / denominator,
            np.nan,
        )
        stock_only = np.where(shares > 0, (ratio - 1) / shares, np.nan)
    return np.where(cash == 0, stock_only, general)


def reconciliation_check(
    previous,
    factors,
    split,
    cash,
    listed,
    *,
    tolerance=0.005,
    prior_factor=None,
    source=None,
    dilution_min=DILUTION_MIN,
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
        diluted = (
            np.zeros(split.shape, bool) if source is None else listed & (source == 5)
        )
        k = dilution_ratios(previous, ratio, split, cash)
        # 来源 5 核对摊薄后的参考价比，而不是公告值本身。
        k_used = np.where(diluted, k, 1.0)
        diluted_theta = np.where(
            cash == 0,
            1 + k_used * (split - 1),
            previous * (1 + k_used * (split - 1)) / (previous - k_used * cash),
        )
        theta = np.where(diluted, diluted_theta, theta)
        deviation = theta / ratio - 1
    valid = listed & np.isfinite(ratio) & (ratio > 0)
    event = listed & ((split != 1) | (cash != 0))
    changed = valid & (np.abs(ratio - 1) > FACTOR_EPSILON)
    out_of_range = diluted & ~((k >= dilution_min) & (k < 1))
    bad = (
        (changed & (~np.isfinite(deviation) | (np.abs(deviation) > tolerance)))
        | (event & ~changed)
        | out_of_range
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
    arrays,
    dates,
    symbols,
    table,
    *,
    tolerance=0.005,
    dilution_min=DILUTION_MIN,
    allow_unreconciled=False,
):
    """分列块重核全部事件，保证历史修订增量时诊断与全量完全一致。"""
    if not 0 < tolerance < 1:
        raise ValueError("action tolerance 必须在 0 与 1 之间")
    if not 0 < dilution_min < 1:
        raise ValueError("差异化分红摊薄下限必须在 0 与 1 之间")
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
    examples, synthetic_examples, diluted_examples = [], [], []
    dilution_values = []
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
                    dilution_min=dilution_min,
                )
        split[~listed], cash[~listed] = np.nan, np.nan
        source[~listed] = 0
        check, bad, ratios, theta = reconciliation_check(
            previous,
            factor,
            split,
            cash,
            listed,
            tolerance=tolerance,
            source=source,
            dilution_min=dilution_min,
        )
        for i, j in np.argwhere(source == 5):
            k = dilution_ratio(previous[i, j], ratios[i, j], split[i, j], cash[i, j])
            dilution_values.append(k)
            if len(diluted_examples) < 20:
                diluted_examples.append(
                    dict(
                        symbol=str(symbols[left + j]),
                        date=str(dates[i]),
                        previous=float(previous[i, j]),
                        ratio=float(ratios[i, j]),
                        split=float(split[i, j]),
                        cash=float(cash[i, j]),
                        k=float(k),
                    )
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
    dilution = np.array(dilution_values)
    result = dict(
        **checks,
        tolerance=tolerance,
        factor_noise=FACTOR_NOISE,
        dilution_min=dilution_min,
        source_counts={str(k): counts[k] for k in range(6)},
        diluted=dict(
            cells=len(dilution),
            below_090=int((dilution < 0.9).sum()),
            k_quantiles={
                str(q): float(np.quantile(dilution, q)) for q in (0, 0.1, 0.5, 0.9, 1)
            }
            if len(dilution)
            else {},
            examples=diluted_examples,
        ),
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
