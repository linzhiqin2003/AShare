"""提醒：把看图一眼能看出、但只读数字容易漏掉的问题写成一条条带数字的句子。

只读 `build_tearsheet` 的输出（也就是存下来的报告数据），不重新回测。规则都是保守的阈值判断，
不做主观评价；阈值写在下面的常量里，口径见 `docs/REPORT_NOTES.md`。
"""

from __future__ import annotations

import statistics

from .fmt import pct

UNFILLED_RATIO = 0.05  # 未成交只次 ÷（新进 + 剔除只次）达到这个比例才提醒
CASH_MEDIAN = 0.05  # 首次调仓后现金占比的中位数超过这个值才提醒
CASH_HIGH = 0.10  # 统计「现金占比超过 10% 的天数」用
HOLDINGS_RATIO = 0.8  # 成交后持仓只数不到目标的 80% 记一次「偏少」
HOLDINGS_SHARE = 0.10  # 「偏少」的调仓占有目标调仓的 10% 及以上才提醒
OPEN_DRAWDOWN = -0.05  # 区间末日的回撤不浅于 −5% 不提醒


def _w(code: str, message: str, **values) -> dict:
    return {"code": code, "message": message, "values": values}


def benchmark_price_index(d: dict) -> dict | None:
    b, s = d["meta"].get("benchmark"), d["summary"]
    if not b or b.get("kind") != "price" or s.get("excess_annual_return") is None:
        return None
    att = d.get("attribution") or []
    whole = [a for a, y in zip(att, d["yearly"], strict=False) if not y["partial"]]
    use = whole or att
    if not use or s.get("total_dividend_gross", 0) <= 0:
        mean = None
    else:
        mean = sum(a["dividend_net"] for a in use) / len(use)
    msg = f"基准 {b['name']} 是价格指数（不含分红），策略收益含分红再投资，超额因此偏高"
    if mean is not None:
        msg += f"；策略的税后分红平均每年贡献 {pct(mean, 1, True)}（分年归因，"
        msg += f"{len(use)} 个{'完整' if whole else ''}年份的均值）"
    return _w(
        "benchmark_price_index",
        msg,
        mean_dividend_contribution=mean,
        years=len(use),
    )


def partial_end(d: dict) -> dict | None:
    s = d["summary"]
    end = s["end"]
    month = bool(d["monthly"]["last_partial"])
    y = d["yearly"][-1] if d["yearly"] else None
    # 只有一年时 partial 也可能来自首年；那种情况下只看末月
    year = bool(y and y["partial"]) and (
        len(d["yearly"]) > 1 or s["start"][:4] != s["first_trade_day"][:4]
    )
    if not (month or year):
        return None
    parts = []
    if year:
        parts.append(f"末年 {y['year']} 只有 {y['days']} 个交易日")
    if month:
        parts.append(f"末月 {end[:7]} 不完整")
    return _w(
        "partial_end",
        f"截至 {end}，" + "，".join(parts) + "；分年、分月表里这一格不是整年 / 整月",
        end=end,
        last_month_partial=month,
        last_year_partial=year,
        last_year_days=y["days"] if year else None,
    )


def partial_start(d: dict) -> dict | None:
    s = d["summary"]
    month = d["monthly"]["first_partial"]
    y = d["yearly"][0] if d["yearly"] else None
    # 与 build_tearsheet 判首年不完整的条件相同：基准日和首个交易日同年
    first_year = bool(y) and s["start"][:4] == s["first_trade_day"][:4]
    if not (month or first_year):
        return None
    parts = []
    if first_year:
        parts.append(f"首年 {y['year']} 只有 {y['days']} 个交易日")
    if month:
        parts.append(f"首月 {s['first_trade_day'][:7]} 不完整")
    return _w(
        "partial_start",
        f"自 {s['first_trade_day']} 起，" + "，".join(parts),
        first_trade_day=s["first_trade_day"],
        first_month_partial=bool(month),
        first_year_partial=first_year,
    )


def unfilled(d: dict) -> dict | None:
    rb = d["rebalances"]
    changes = sum(r["new_names"] + r["dropped_names"] for r in rb)
    total = sum(r["unfilled"] for r in rb)
    if not changes or total / changes < UNFILLED_RATIO:
        return None
    by = {
        k: sum(r["unfilled_by_reason"][k] for r in rb)
        for k in ("limit_up", "limit_down", "suspended")
    }
    worst = max(rb, key=lambda r: (r["unfilled"], r["date"]))
    later = sum(r["unfilled_filled_before_next_rebalance"] for r in rb)
    return _w(
        "unfilled",
        f"调仓日开盘未成交 {total} 只次，占新进和剔除合计 {changes} 只次的 {pct(total / changes, 1)}"
        f"（涨停买不进 {by['limit_up']}、跌停卖不出 {by['limit_down']}、停牌 {by['suspended']}）；"
        f"最多一次 {worst['date']} {worst['unfilled']} 只；下次调仓前补成交 {later} 只次",
        unfilled=total,
        changes=changes,
        ratio=total / changes,
        by_reason=by,
        worst_date=worst["date"],
        worst_count=worst["unfilled"],
        filled_later=later,
    )


def cash_high(d: dict) -> dict | None:
    rb = d["rebalances"]
    if not rb:
        return None
    dates = d["daily"]["date"]
    i0 = dates.index(rb[0]["date"])
    cw = [x for x in d["daily"]["cash_weight"][i0:] if x is not None]
    if not cw:
        return None
    med = statistics.median(cw)
    if med <= CASH_MEDIAN:
        return None
    high = sum(1 for x in cw if x > CASH_HIGH) / len(cw)
    k = max(range(len(cw)), key=lambda j: cw[j])
    return _w(
        "cash_high",
        f"现金占比长期偏高：首次调仓（{rb[0]['date']}）后中位数 {pct(med, 1)}，"
        f"超过 {pct(CASH_HIGH, 0)} 的天数占 {pct(high, 1)}，最高 {pct(cw[k], 1)}（{dates[i0 + k]}）",
        median=med,
        share_above_10pct=high,
        max=cw[k],
        max_date=dates[i0 + k],
    )


def holdings_short(d: dict) -> dict | None:
    rb = [r for r in d["rebalances"] if r["targets"] > 0]
    short = [r for r in rb if r["holdings_after"] < HOLDINGS_RATIO * r["targets"]]
    if not short or len(short) < HOLDINGS_SHARE * len(rb):
        return None
    worst = min(short, key=lambda r: (r["holdings_after"] / r["targets"], r["date"]))
    return _w(
        "holdings_short",
        f"持仓只数明显少于目标：{len(short)} 次调仓（占有目标调仓的 {pct(len(short) / len(rb), 1)}）"
        f"成交后持仓不到目标的 {pct(HOLDINGS_RATIO, 0)}；最少一次 {worst['date']} "
        f"目标 {worst['targets']} 只、持仓 {worst['holdings_after']} 只",
        count=len(short),
        rebalances=len(rb),
        worst_date=worst["date"],
        worst_targets=worst["targets"],
        worst_holdings=worst["holdings_after"],
    )


def drawdown_open(d: dict) -> dict | None:
    dd, dates = d["daily"]["drawdown"], d["daily"]["date"]
    cur = dd[-1]
    if cur is None or cur > OPEN_DRAWDOWN:
        return None
    peak = max(i for i, x in enumerate(dd) if x == 0)
    trough = min(range(peak, len(dd)), key=lambda i: (dd[i], i))
    return _w(
        "drawdown_open",
        f"截至 {dates[-1]} 回撤尚未修复：自峰值 {dates[peak]} 已 {len(dd) - 1 - peak} 个交易日，"
        f"当前 {pct(cur, 1)}，期间最深 {pct(dd[trough], 1)}（{dates[trough]}）",
        current=cur,
        peak=dates[peak],
        trough=dates[trough],
        depth=dd[trough],
        days=len(dd) - 1 - peak,
    )


def short_history(d: dict) -> dict | None:
    n, P = d["summary"]["trading_days"], d["meta"].get("periods_per_year") or 252
    if n >= P:
        return None
    return _w(
        "short_history",
        f"历史只有 {n} 个交易日（不足 {P} 个，即一年）：年化收益、年化波动、夏普只是外推，不可靠",
        trading_days=n,
    )


def benchmark_late(d: dict) -> dict | None:
    s = d["summary"]
    if s.get("benchmark_start") in (None, s["start"]):
        return None
    return _w(
        "benchmark_late",
        f"基准自 {s['benchmark_start']} 起才有数据，晚于策略基准日 {s['start']}；"
        "基准和相对指标只在重叠区间计算",
        benchmark_start=s["benchmark_start"],
    )


def benchmark_filled(d: dict) -> dict | None:
    b = d["meta"].get("benchmark")
    if not b or not b.get("filled_days"):
        return None
    return _w(
        "benchmark_filled",
        f"基准序列中间缺 {b['filled_days']} 天，已用前一天的值填充",
        filled_days=b["filled_days"],
    )


def weekday_basis(d: dict) -> dict | None:
    if d["meta"].get("completeness_basis") != "weekday":
        return None
    return _w(
        "weekday_basis",
        "没有传入交易日历：末月、末年是否完整按工作日判断，节假日会被当成交易日",
    )


RULES = (
    short_history,
    benchmark_price_index,
    benchmark_late,
    benchmark_filled,
    partial_start,
    partial_end,
    weekday_basis,
    unfilled,
    holdings_short,
    cash_high,
    drawdown_open,
)


def warnings(data: dict) -> list[dict]:
    """按 `RULES` 的顺序逐条检查，返回 `{code, message, values}` 的列表（没有问题时为空）。"""
    return [w for rule in RULES if (w := rule(data)) is not None]
