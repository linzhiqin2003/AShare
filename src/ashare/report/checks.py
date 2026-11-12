"""提醒：把看图一眼能看出、但只读数字容易漏掉的问题写成一条条带数字的句子。

只读 `build_tearsheet` 的输出（也就是存下来的报告数据），不重新回测。规则都是保守的阈值判断，
不做主观评价；阈值写在下面的常量里，口径见 `docs/REPORT_NOTES.md`。

`warnings(data, a, b)` 按所选区间（`daily.date` 的行号，a 是基准日、b 是末日）检查：与区间有关的
规则（历史长短、首末月年不完整、未成交、持仓不足、现金占比、回撤未修复、成本、基准晚起步、
基准是否收到分红）都只看区间内的数据；只能按全区间给的（基准缺失天数的填充）在自选区间时
标明「全区间」。不给 a、b 时是全区间。
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from .fmt import pct

UNFILLED_RATIO = 0.05  # 未成交只次 ÷（新进 + 剔除只次）达到这个比例才提醒
CASH_MEDIAN = 0.05  # 首次调仓后现金占比的中位数超过这个值才提醒
CASH_HIGH = 0.10  # 统计「现金占比超过 10% 的天数」用
HOLDINGS_RATIO = 0.8  # 成交后持仓只数不到目标的 80% 记一次「偏少」
HOLDINGS_SHARE = 0.10  # 「偏少」的调仓占有目标调仓的 10% 及以上才提醒
OPEN_DRAWDOWN = -0.05  # 区间末日的回撤不浅于 −5% 不提醒
COST_HIGH = 0.03  # 年化成本（费用 + 滑点）÷ 平均权益超过 3% 才提醒


def _w(code: str, message: str, **values) -> dict:
    return {"code": code, "message": message, "values": values}


@dataclass(frozen=True)
class View:
    """报告数据加上所选区间 (a, b]。"""

    data: dict
    a: int
    b: int

    @property
    def dates(self) -> list:
        return self.data["daily"]["date"]

    @property
    def full(self) -> bool:
        return self.a == 0 and self.b == len(self.dates) - 1

    @property
    def lo(self) -> str:  # 区间第一个交易日
        return self.dates[self.a + 1]

    @property
    def hi(self) -> str:
        return self.dates[self.b]

    def rebalances(self) -> list:
        """区间内的调仓（a < 调仓行 ≤ b，与区间费用的口径相同）。"""
        rb = self.data["rebalances"]
        if self.full:
            return rb
        idx = {d: i for i, d in enumerate(self.dates)}
        return [r for r in rb if self.a < idx[r["date"]] <= self.b]

    def days_in_year(self, year: str) -> int:
        return sum(
            1 for i in range(self.a + 1, self.b + 1) if self.dates[i][:4] == year
        )

    def benchmark_first(self) -> int | None:
        """基准第一个有效值的行号；没有基准时为 None。"""
        bnav = self.data["daily"].get("benchmark_nav")
        if not self.data["meta"].get("benchmark") or not bnav:
            return None
        return next((i for i, v in enumerate(bnav) if v is not None), None)


def benchmark_price_index(v: View) -> dict | None:
    d = v.data
    b, s = d["meta"].get("benchmark"), d["summary"]
    if not b or b.get("kind") != "price" or s.get("excess_annual_return") is None:
        return None
    y0, y1 = int(v.lo[:4]), int(v.hi[:4])
    att = d.get("attribution") or []
    pairs = [
        (a, y) for a, y in zip(att, d["yearly"], strict=False) if y0 <= y["year"] <= y1
    ]
    # 区间是否收到过分红：全区间看税前分红合计；自选区间看区间所在年份的分红归因
    if v.full:
        received = s.get("total_dividend_gross", 0) > 0
        approx = False
    else:
        got = [y for a, y in pairs if a["dividend_net"] > 0]
        received = bool(got)
        approx = received and all(
            v.days_in_year(str(y["year"])) < y["days"] for y in got
        )
    msg = f"基准 {b['name']} 是价格指数（不含分红）"
    mean = None
    use = []
    if received:
        msg += "，策略收益含分红再投资，超额因此偏高"
        if approx:
            msg += "（区间按所在年份的分红判断）"
        # 均值只取整个落在区间里的年份；全区间时与以前相同（完整年份，没有就全部年份）
        inside = [
            (a, y) for a, y in pairs if v.days_in_year(str(y["year"])) == y["days"]
        ]
        whole = [a for a, y in inside if not y["partial"]]
        use = whole or [a for a, _ in inside]
        if use:
            mean = sum(a["dividend_net"] for a in use) / len(use)
            msg += f"；策略的税后分红平均每年贡献 {pct(mean, 1, True)}（分年归因，"
            msg += f"{len(use)} 个{'完整' if whole else ''}年份的均值）"
    else:
        msg += (
            "；区间内策略没有收到现金分红" if not v.full else "；策略没有收到现金分红"
        )
    return _w(
        "benchmark_price_index",
        msg,
        dividends_received=received,
        mean_dividend_contribution=mean,
        years=len(use),
    )


def partial_end(v: View) -> dict | None:
    d = v.data
    s = d["summary"]
    end = v.hi
    year_key = end[:4]
    if v.b == len(v.dates) - 1:
        month = bool(d["monthly"]["last_partial"])
        y = d["yearly"][-1] if d["yearly"] else None
        # 只有一年时 partial 也可能来自首年；那种情况下只看末月
        year = bool(y and y["partial"]) and (
            len(d["yearly"]) > 1 or s["start"][:4] != s["first_trade_day"][:4]
        )
    else:  # 区间在报告末日之前结束：下一个交易日还在同一个月 / 年，就是被截断
        nxt = v.dates[v.b + 1]
        month = nxt[:7] == end[:7]
        year = nxt[:4] == year_key
    if not (month or year):
        return None
    days = v.days_in_year(year_key)
    parts = []
    if year:
        parts.append(f"末年 {year_key} 只有 {days} 个交易日")
    if month:
        parts.append(f"末月 {end[:7]} 不完整")
    return _w(
        "partial_end",
        f"截至 {end}，" + "，".join(parts) + "；分年、分月表里这一格不是整年 / 整月",
        end=end,
        last_month_partial=month,
        last_year_partial=year,
        last_year_days=days if year else None,
    )


def partial_start(v: View) -> dict | None:
    d = v.data
    s = d["summary"]
    first = v.lo
    if v.a == 0:
        month = bool(d["monthly"]["first_partial"])
        # 与 build_tearsheet 判首年不完整的条件相同：基准日和首个交易日同年
        first_year = bool(d["yearly"]) and s["start"][:4] == s["first_trade_day"][:4]
    else:  # 区间从报告中间开始：基准日与第一个交易日同月 / 同年，就是被截断
        base = v.dates[v.a]
        month = base[:7] == first[:7]
        first_year = base[:4] == first[:4]
    if not (month or first_year):
        return None
    parts = []
    if first_year:
        parts.append(f"首年 {first[:4]} 只有 {v.days_in_year(first[:4])} 个交易日")
    if month:
        parts.append(f"首月 {first[:7]} 不完整")
    return _w(
        "partial_start",
        f"自 {first} 起，" + "，".join(parts),
        first_trade_day=first,
        first_month_partial=month,
        first_year_partial=first_year,
    )


def unfilled(v: View) -> dict | None:
    rb = v.rebalances()
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


def cash_high(v: View) -> dict | None:
    rb = v.data["rebalances"]
    if not rb:
        return None
    dates = v.dates
    first = dates.index(rb[0]["date"])
    # 首次调仓之前的空仓不算；自选区间从区间第一个交易日（或更晚的首次调仓）算起
    i0 = first if v.full else max(first, v.a + 1)
    if i0 > v.b:
        return None
    cw = [x for x in v.data["daily"]["cash_weight"][i0 : v.b + 1] if x is not None]
    if not cw:
        return None
    med = statistics.median(cw)
    if med <= CASH_MEDIAN:
        return None
    high = sum(1 for x in cw if x > CASH_HIGH) / len(cw)
    k = max(range(len(cw)), key=lambda j: cw[j])
    since = (
        f"首次调仓（{rb[0]['date']}）后" if i0 == first else f"区间内（{dates[i0]} 起）"
    )
    return _w(
        "cash_high",
        f"现金占比长期偏高：{since}中位数 {pct(med, 1)}，"
        f"超过 {pct(CASH_HIGH, 0)} 的天数占 {pct(high, 1)}，最高 {pct(cw[k], 1)}（{dates[i0 + k]}）",
        median=med,
        share_above_10pct=high,
        max=cw[k],
        max_date=dates[i0 + k],
    )


def holdings_short(v: View) -> dict | None:
    rb = [r for r in v.rebalances() if r["targets"] > 0]
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


def _window_drawdown(v: View) -> list:
    """区间内的回撤：以区间基准日为起点重新取峰值（全区间时就是报告的 daily.drawdown）。"""
    if v.full:
        return v.data["daily"]["drawdown"]
    nav = v.data["daily"]["nav"][v.a : v.b + 1]
    out, peak = [], nav[0]
    for x in nav:
        peak = max(peak, x)
        out.append(x / peak - 1)
    return out


def drawdown_open(v: View) -> dict | None:
    dd = _window_drawdown(v)
    dates = v.dates[v.a : v.b + 1]
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


def costs_high(v: View) -> dict | None:
    """年化成本 = 区间内调仓的费用 + 滑点合计 ÷ 区间平均权益 ÷ 年数。"""
    cap = v.data["meta"].get("initial_capital")
    P = v.data["meta"].get("periods_per_year") or 252
    if not cap:
        return None
    rb = v.rebalances()
    fees = sum(r["fees"] for r in rb)
    slip = sum(r["slippage"] for r in rb)
    nav = v.data["daily"]["nav"][v.a + 1 : v.b + 1]
    avg = sum(nav) / len(nav) * cap
    years = (v.b - v.a) / P
    if not avg > 0:
        return None
    ratio = (fees + slip) / avg / years
    if ratio <= COST_HIGH:
        return None
    return _w(
        "costs_high",
        f"换手或成本很高：年化成本（费用 + 滑点）占平均权益的 {pct(ratio, 2)}，"
        f"超过 {pct(COST_HIGH, 0)}；其中费用 {pct(fees / avg / years, 2)}、滑点 {pct(slip / avg / years, 2)}",
        ratio=ratio,
        fees=fees,
        slippage=slip,
        average_equity=avg,
        years=years,
    )


def short_history(v: View) -> dict | None:
    n = v.data["summary"]["trading_days"] if v.full else v.b - v.a
    P = v.data["meta"].get("periods_per_year") or 252
    if n >= P:
        return None
    what = "历史" if v.full else "所选区间"
    return _w(
        "short_history",
        f"{what}只有 {n} 个交易日（不足 {P} 个，即一年）：年化收益、年化波动、夏普只是外推，不可靠",
        trading_days=n,
    )


def benchmark_late(v: View) -> dict | None:
    s = v.data["summary"]
    if v.full:
        if s.get("benchmark_start") in (None, s["start"]):
            return None
        start = s["benchmark_start"]
    else:
        i = v.benchmark_first()
        if i is None or i <= v.a:
            return None
        start = v.dates[i] if i <= v.b else None
    if start is None:
        return _w(
            "benchmark_late",
            f"基准在所选区间内没有数据（自 {v.dates[v.benchmark_first()]} 起才有）",
            benchmark_start=v.dates[v.benchmark_first()],
        )
    base = s["start"] if v.full else v.dates[v.a]
    return _w(
        "benchmark_late",
        f"基准自 {start} 起才有数据，晚于{'策略' if v.full else '区间'}基准日 {base}；"
        "基准和相对指标只在重叠区间计算",
        benchmark_start=start,
    )


def benchmark_filled(v: View) -> dict | None:
    b = v.data["meta"].get("benchmark")
    if not b or not b.get("filled_days"):
        return None
    tag = "" if v.full else "（全区间）"
    return _w(
        "benchmark_filled",
        f"基准序列中间缺 {b['filled_days']} 天，已用前一天的值填充{tag}",
        filled_days=b["filled_days"],
        full_range_only=not v.full,
    )


def weekday_basis(v: View) -> dict | None:
    if v.data["meta"].get("completeness_basis") != "weekday":
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
    costs_high,
    drawdown_open,
)


def warnings(data: dict, a: int | None = None, b: int | None = None) -> list[dict]:
    """按 `RULES` 的顺序逐条检查，返回 `{code, message, values}` 的列表（没有问题时为空）。

    a、b：区间的基准日和末日（`daily.date` 的行号）；不给时为全区间。
    """
    T = len(data["daily"]["date"])
    v = View(data, 0 if a is None else a, T - 1 if b is None else b)
    return [w for rule in RULES if (w := rule(v)) is not None]
