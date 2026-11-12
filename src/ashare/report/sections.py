"""文本版的各个小节：每节给出排版块（文本行和表格）以及 JSON 用的原始数值。

数字只从报告数据里取，或由 `window.window_summary`（页面的区间公式）重算；格式化统一走 `fmt`。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from . import checks
from .fmt import MISSING, num, pct, width, wrap
from .window import resolve, window_summary

SUMMARY_MAX_LINES = 40
TITLE_LINES = 2  # 默认摘要里标题最多占两行，更长的截断加「…」（其余小节不截）


def _clip(text: str, lines: int) -> str:
    """截短到折行后不超过 lines 行，末尾加「…」。"""
    if len(wrap(text)) <= lines:
        return text
    lo, hi = 0, len(text)  # 二分：保留前 k 个字符
    while lo < hi:
        k = (lo + hi + 1) // 2
        if len(wrap(text[:k].rstrip() + "…")) <= lines:
            lo = k
        else:
            hi = k - 1
    return text[:lo].rstrip() + "…"


@dataclass
class Table:
    cols: list[tuple[str, str]]  # (表头, 对齐 "l" / "r")
    rows: list[list[str]] = field(default_factory=list)


@dataclass
class Context:
    data: dict
    lang: str
    a: int
    b: int
    full: bool
    ws: dict  # 区间指标（summary 的键名）

    @classmethod
    def make(cls, data, *, lang="zh", start=None, end=None):
        a, b = resolve(data, start, end)
        full = a == 0 and b == len(data["daily"]["date"]) - 1
        return cls(data, lang, a, b, full, window_summary(data, a, b))

    # ------------------------------------------------------------ 工具
    def L(self, zh: str, en: str) -> str:
        return en if self.lang == "en" else zh

    @property
    def lo(self) -> str:  # 区间第一个交易日
        return self.data["daily"]["date"][self.a + 1]

    @property
    def hi(self) -> str:
        return self.data["daily"]["date"][self.b]

    @property
    def bench(self) -> dict | None:
        b = self.data["meta"].get("benchmark")
        return b if b and self.data["daily"].get("benchmark_nav") else None

    def bname(self) -> str:
        return self.bench["name"] if self.bench else ""

    def days(self, n) -> str:
        return MISSING if n is None else f"{n} " + self.L("天", "d")

    def years(self):
        """区间内的分年行。全区间原样返回报告的 yearly；自选区间时，被区间截断的年份按
        `daily.nav` 重算区间内收益（与 build 的「期末净值 ÷ 期初净值 − 1」同一公式），
        其余列为 None，并记下 `clipped = (起, 止)`。"""
        y0, y1 = int(self.lo[:4]), int(self.hi[:4])
        rows = [y for y in self.data["yearly"] if y0 <= y["year"] <= y1]
        if self.full:
            return rows
        dl = self.data["daily"]
        dates, nav, bnav = dl["date"], dl["nav"], dl.get("benchmark_nav")
        out = []
        for y in rows:
            idx = [
                i
                for i in range(self.a + 1, self.b + 1)
                if dates[i][:4] == str(y["year"])
            ]
            whole = [i for i in range(1, len(dates)) if dates[i][:4] == str(y["year"])]
            if idx == whole:
                out.append(y)
                continue
            prev, last = idx[0] - 1, idx[-1]
            br = None
            if bnav is not None and bnav[prev] is not None:
                br = bnav[last] / bnav[prev] - 1
            out.append(
                {
                    "year": y["year"],
                    "days": len(idx),
                    "return": nav[last] / nav[prev] - 1,
                    "volatility": None,
                    "max_drawdown": None,
                    "sharpe": None,
                    "benchmark_return": br,
                    "benchmark_volatility": None,
                    "benchmark_max_drawdown": None,
                    "benchmark_sharpe": None,
                    "partial": True,
                    "clipped": [dates[idx[0]], dates[last]],
                }
            )
        return out

    def in_range(self, d: str) -> bool:
        return self.lo <= d <= self.hi

    def range_info(self) -> dict:
        dl = self.data["daily"]["date"]
        return {
            "start": dl[self.a],
            "first_trade_day": self.lo,
            "end": self.hi,
            "full_range": self.full,
            "report_start": dl[0],
            "report_end": dl[-1],
        }


@dataclass(frozen=True)
class Section:
    title: str
    blocks: Callable[[Context], list]
    json: Callable[[Context], object]


def kv(ctx: Context, pairs) -> str:
    """一行若干「标签 值」；英文时写成 key=value。"""
    if ctx.lang == "en":
        return "  ".join(f"{en}={v}" for _, en, v in pairs)
    return "，".join(f"{zh} {v}" for zh, _, v in pairs)


# ====================================================================== summary
def _yearly_note(ctx: Context, rows) -> str:
    """不完整年份的说明：末年写「至 月-日」，首年写「自 月-日」。"""
    s, Y = ctx.data["summary"], ctx.data["yearly"]
    start_partial = s["start"][:4] == s["first_trade_day"][:4]
    notes = []
    for y in rows:
        if not y["partial"]:
            continue
        if y.get("clipped"):
            a, b = y["clipped"]
            notes.append(
                f"{y['year']} "
                + ctx.L(f"只算 {a[5:]} 至 {b[5:]}", f"{a[5:]}..{b[5:]} only")
            )
        elif y is Y[0] and start_partial:
            notes.append(
                f"{y['year']} " + ctx.L("自 ", "from ") + s["first_trade_day"][5:]
            )
        else:
            notes.append(f"{y['year']} " + ctx.L("至 ", "to ") + s["end"][5:])
    return ("；" if ctx.lang == "zh" else "; ").join(notes)


def _yearly_short(ctx: Context, rows) -> Table:
    hasb = ctx.bench is not None
    cols = [(ctx.L("年份", "year"), "l"), (ctx.L("策略", "return"), "r")]
    if hasb:
        cols += [
            (ctx.L(ctx.bname(), "benchmark_return"), "r"),
            (ctx.L("超额", "excess_return"), "r"),
        ]
    t = Table(cols)
    for y in rows:
        row = [f"{y['year']}{'*' if y['partial'] else ''}", pct(y["return"], 1, True)]
        if hasb:
            br = y["benchmark_return"]
            row += [
                pct(br, 1, True),
                pct(None if br is None else y["return"] - br, 1, True),
            ]
        t.rows.append(row)
    return t


def _drawdown_table(ctx: Context, eps) -> Table:
    t = Table(
        [
            (ctx.L("深度", "depth"), "r"),
            (ctx.L("峰值", "peak"), "l"),
            (ctx.L("谷底", "trough"), "l"),
            (ctx.L("修复", "recovery"), "l"),
            (ctx.L("见底", "days_to_trough"), "r"),
            (ctx.L("持续", "duration_days"), "r"),
        ]
    )
    for e in eps:
        t.rows.append(
            [
                pct(e["depth"], 1),
                e["peak"],
                e["trough"],
                e["recovery"] or ctx.L("未修复", "not_recovered"),
                ctx.days(e["days_to_trough"]),
                ctx.days(e["duration_days"]),
            ]
        )
    return t


def _drawdowns_in(ctx: Context):
    out = []
    for e in ctx.data["drawdowns"]:
        stop = e["recovery"] or ctx.data["daily"]["date"][-1]
        if e["peak"] <= ctx.hi and stop >= ctx.lo:
            out.append(e)
    return out


def _header_lines(ctx: Context) -> list[str]:
    s, m = ctx.data["summary"], ctx.data["meta"]
    w = ctx.ws
    n = w["trading_days"]
    P = m.get("periods_per_year") or 252
    yrs = n / P
    if ctx.lang == "en":
        line = (
            f"period first_trade_day={ctx.lo} end={ctx.hi} start={w['start']} "
            f"trading_days={n} years={yrs:.1f}"
        )
        if not ctx.full:
            line += " custom_range=true"
    else:
        line = f"区间 {ctx.lo} 至 {ctx.hi}（基准日 {w['start']}），{n} 个交易日（{yrs:.1f} 年）"
        if not ctx.full:
            line += "，自选区间"
    b = ctx.bench
    if b is None:
        bl = ctx.L("无基准", "benchmark none")
    elif ctx.lang == "en":
        bl = f"benchmark name={b['name']} symbol={b.get('symbol') or MISSING} kind={b['kind']}"
        if s.get("benchmark_start") and s["benchmark_start"] != s["start"]:
            bl += f" benchmark_start={s['benchmark_start']}"
    else:
        sym = f"（{b['symbol']}）" if b.get("symbol") else ""
        bl = f"基准 {b['name']}{sym}，{b['kind_label']}"
        if s.get("benchmark_start") and s["benchmark_start"] != s["start"]:
            bl += f"，自 {s['benchmark_start']} 起"
    out = [m.get("title") or "回测报告", line, bl]
    if not ctx.full:
        out.append(
            ctx.L(
                f"报告全区间 {s['first_trade_day']} 至 {s['end']}；指标按页面「切换区间」的公式重算",
                f"report_first_trade_day={s['first_trade_day']}  report_end={s['end']}",
            )
        )
    return out


def _metrics_table(ctx: Context) -> Table:
    w = ctx.ws
    hasb = ctx.bench is not None
    cols = [(ctx.L("指标", "metric"), "l"), (ctx.L("策略", "strategy"), "r")]
    if hasb:
        cols.append((ctx.L(ctx.bname(), "benchmark"), "r"))
    t = Table(cols)
    rows = [
        (
            "累计收益",
            "cumulative_return",
            lambda x: pct(x, 1, True),
            "benchmark_cumulative_return",
        ),
    ]
    if w["annualized"]:
        rows.append(
            (
                "年化收益",
                "annual_return",
                lambda x: pct(x, 1, True),
                "benchmark_annual_return",
            )
        )
    rows += [
        (
            "年化波动",
            "annual_volatility",
            lambda x: pct(x, 1),
            "benchmark_annual_volatility",
        ),
        ("夏普", "sharpe", num, "benchmark_sharpe"),
        ("最大回撤", "max_drawdown", lambda x: pct(x, 1), "benchmark_max_drawdown"),
    ]
    for zh, key, f, bkey in rows:
        row = [ctx.L(zh, key), f(w[key])]
        if hasb:
            row.append(f(w[bkey]))
        t.rows.append(row)
    return t


def excess_sign_note(ctx: Context, ex, ir, w) -> str | None:
    """超额与信息比率符号相反时说明两者的口径（波动大时几何差与日均超额可以反号）。"""
    if ex is None or ir is None or not ex * ir < 0:
        return None
    exar = w.get("excess_daily_mean_annualized")
    if ctx.lang == "en":
        return (
            "note: excess return and information_ratio have opposite signs; excess is the "
            "difference of compounded returns, information_ratio uses the mean daily excess "
            f"(excess_daily_mean_annualized={pct(exar, 1, True)})"
        )
    what = (
        "年化超额是策略与基准几何年化收益之差"
        if w["annualized"]
        else "区间超额是两者区间收益之差"
    )
    return (
        f"注：超额与信息比率符号相反——{what}，信息比率的分子是日超额的均值"
        f"（日超额算术年化 {pct(exar, 1, True)}）；两边波动差得多时两者可以反号"
    )


def summary_blocks(ctx: Context) -> list:
    s, w = ctx.data["summary"], ctx.ws
    head = _header_lines(ctx)
    head[0] = _clip(head[0], TITLE_LINES)
    blocks: list = [*head, "", _metrics_table(ctx)]
    if ctx.full and s.get("max_drawdown_peak"):
        rec = s["max_drawdown_recovery"]
        if ctx.lang == "en":
            blocks.append(
                f"max_drawdown_peak={s['max_drawdown_peak']}  "
                f"max_drawdown_trough={s['max_drawdown_trough']}  "
                f"max_drawdown_recovery={rec or MISSING}"
            )
        else:
            blocks.append(
                f"最大回撤 {s['max_drawdown_peak']} 见顶，{s['max_drawdown_trough']} 见底，"
                + (f"{rec} 修复" if rec else "尚未修复")
            )
    if ctx.bench is not None and (w["excess_annual_return"] is not None):
        ann = w["annualized"]
        ex = w["excess_annual_return"] if ann else w["excess_cumulative_return"]
        blocks.append(
            kv(
                ctx,
                [
                    (
                        "年化超额" if ann else "区间超额",
                        "excess_annual_return" if ann else "excess_cumulative_return",
                        pct(ex, 1, True),
                    ),
                    ("信息比率", "information_ratio", num(w["information_ratio"])),
                    ("跟踪误差", "tracking_error", pct(w["tracking_error"], 1)),
                    ("Beta", "beta", num(w["beta"])),
                ],
            )
        )
        note = excess_sign_note(ctx, ex, w["information_ratio"], w)
        if note:
            blocks.append(note)
    blocks.append(
        kv(
            ctx,
            [
                ("年化换手（双边）", "annual_turnover", pct(w["annual_turnover"], 0)),
                (
                    "费用 / 期初权益",
                    "total_fees_to_initial",
                    pct(w["total_fees_to_initial"], 2),
                ),
                (
                    "滑点 / 期初权益",
                    "total_slippage_to_initial",
                    pct(w["total_slippage_to_initial"], 2),
                ),
            ],
        )
    )
    # 分年、回撤、提醒；分年行数按剩余行数截取最近若干年
    years = ctx.years()
    eps = _drawdowns_in(ctx)[:3]
    warns = checks.warnings(ctx.data, ctx.a, ctx.b)
    tail: list = []
    if eps:
        tail += [
            "",
            ctx.L("最深三次回撤" if len(eps) == 3 else "最深回撤", "drawdowns"),
            _drawdown_table(ctx, eps),
        ]
    codes = [x["code"] for x in warns]
    tail += [
        "",
        (
            ctx.L("提醒：", "warnings: ")
            + (("、" if ctx.lang == "zh" else ", ").join(codes) or ctx.L("无", "none"))
        )
        + (
            ctx.L("（--section warnings 看详情）", " (see --section warnings)")
            if codes
            else ""
        ),
    ]

    def count(bl):
        """按文本版的实际排版数行：表格按 `_table_text` 的输出，文本行按折行后的行数。"""
        from .text import _table_text

        return sum(
            len(_table_text(x)) if isinstance(x, Table) else len(wrap(x)) for x in bl
        )

    def yearly_part(shown):
        notes = []
        if len(shown) < len(years):
            notes.append(
                ctx.L(
                    f"更早的 {len(years) - len(shown)} 年见 --section yearly",
                    f"{len(years) - len(shown)} earlier years: --section yearly",
                )
            )
        note = _yearly_note(ctx, shown)
        if note:
            notes.append(f"* {note}")
        title = ctx.L("分年收益", "yearly")
        if notes:
            title += ctx.L(f"（{'；'.join(notes)}）", f" ({'; '.join(notes)})")
        return ["", title, _yearly_short(ctx, shown)]

    room = SUMMARY_MAX_LINES - count(blocks) - count(tail)
    # 能放下几年就放几年（从最近的一年往前）；标题里的说明随所放年数变化，所以逐个试
    for k in range(min(len(years), room), 0, -1):
        part = yearly_part(years[-k:])
        if count(part) <= room:
            blocks += part
            break
    return blocks + tail


def summary_json(ctx: Context):
    s = ctx.data["summary"]
    m = ctx.data["meta"]
    out = {
        "title": m.get("title"),
        "benchmark": m.get("benchmark"),
        "summary": dict(s) if ctx.full else ctx.ws,
        "yearly": [
            {
                "year": y["year"],
                "return": y["return"],
                "benchmark_return": y["benchmark_return"],
                "excess_return": None
                if y["benchmark_return"] is None
                else y["return"] - y["benchmark_return"],
                "partial": y["partial"],
            }
            for y in ctx.years()
        ],
        "drawdowns": _drawdowns_in(ctx)[:3],
        "warnings": [w["code"] for w in checks.warnings(ctx.data, ctx.a, ctx.b)],
    }
    if ctx.full:
        out["summary"]["annualized"] = ctx.ws["annualized"]
        # 报告数据里没有这一项（页面也不显示）：按区间公式在嵌入的日度序列上算
        out["summary"]["excess_daily_mean_annualized"] = ctx.ws[
            "excess_daily_mean_annualized"
        ]
    return out


# ====================================================================== yearly
def _attr(ctx):
    return {a["year"]: a for a in ctx.data["attribution"]}


def yearly_blocks(ctx: Context) -> list:
    hasb = ctx.bench is not None
    A = _attr(ctx)
    cols = [
        (ctx.L("年份", "year"), "l"),
        (ctx.L("交易日", "days"), "r"),
        (ctx.L("策略", "return"), "r"),
    ]
    if hasb:
        cols += [
            (ctx.L(ctx.bname(), "benchmark_return"), "r"),
            (ctx.L("超额", "excess_return"), "r"),
        ]
    cols += [
        (ctx.L("波动", "volatility"), "r"),
        (ctx.L("最大回撤", "max_drawdown"), "r"),
        (ctx.L("夏普", "sharpe"), "r"),
        (ctx.L("分红", "dividend_net"), "r"),
        (ctx.L("成本", "fees_and_slippage"), "r"),
    ]
    t = Table(cols)
    rows = ctx.years()
    for y in rows:
        a = None if y.get("clipped") else A.get(y["year"])
        r = [
            f"{y['year']}{'*' if y['partial'] else ''}",
            ctx.days(y["days"]),
            pct(y["return"], 1, True),
        ]
        if hasb:
            br = y["benchmark_return"]
            r += [
                pct(br, 1, True),
                pct(None if br is None else y["return"] - br, 1, True),
            ]
        r += [
            pct(y["volatility"], 1),
            pct(y["max_drawdown"], 1),
            num(y["sharpe"]),
            pct(a["dividend_net"], 1, True) if a else MISSING,
            pct(a["fees"] + a["slippage"], 2) if a else MISSING,
        ]
        t.rows.append(r)
    out: list = []
    note = _yearly_note(ctx, rows)
    if note:
        out.append(ctx.L(f"* 不完整：{note}", f"* partial: {note}"))
    out.append(t)
    out.append(
        ctx.L(
            "分红为税后分红 ÷ 上年末权益；成本为费用与滑点 ÷ 上年末权益（负数表示拖累）",
            "dividend_net and fees_and_slippage are divided by the previous year-end equity",
        )
    )
    return out


def yearly_json(ctx: Context):
    A = _attr(ctx)
    return [
        {**y, "attribution": None if y.get("clipped") else A.get(y["year"])}
        for y in ctx.years()
    ]


# ====================================================================== monthly
def monthly_blocks(ctx: Context) -> list:
    mo = ctx.data["monthly"]
    lo, hi = ctx.lo[:7], ctx.hi[:7]
    idx = [i for i, mk in enumerate(mo["month"]) if lo <= mk <= hi]
    years = sorted({int(mo["month"][i][:4]) for i in idx})
    Y = {y["year"]: y for y in ctx.data["yearly"]}
    hasb = ctx.bench is not None and mo.get("benchmark_return") is not None

    def grid(values, annual, sign_pct=True):
        for keep in (True, False):
            cols = [(ctx.L("年份", "year"), "l")]
            cols += [(ctx.L(f"{k}月", f"m{k:02d}"), "r") for k in range(1, 13)]
            cols.append((ctx.L("全年", "year_total"), "r"))
            t = Table(cols)
            for y in years:
                row = [str(y)]
                for k in range(1, 13):
                    mk = f"{y}-{k:02d}"
                    v = values.get(mk, "absent")
                    if v == "absent":
                        row.append("")
                    else:
                        s = pct(v, 1, True)
                        row.append(s if keep or s == MISSING else s[:-1])
                v = annual(y)
                s = pct(v, 1, True)
                row.append(s if keep or s == MISSING else s[:-1])
                t.rows.append(row)
            from .text import _table_text

            if keep and any(width(x) > 100 for x in _table_text(t)):
                continue
            return t, keep

    vals = {mo["month"][i]: mo["return"][i] for i in idx}
    out: list = []
    t, keep = grid(vals, lambda y: Y[y]["return"] if y in Y else None)
    out.append(t)
    if not keep:
        out.append(ctx.L("（表中数字单位为 %）", "(values in %)"))
    if hasb:
        ex = {}
        for i in idx:
            b = mo["benchmark_return"][i]
            ex[mo["month"][i]] = None if b is None else mo["return"][i] - b

        def yex(y):
            r = Y.get(y)
            return (
                None
                if not r or r["benchmark_return"] is None
                else r["return"] - r["benchmark_return"]
            )

        t2, keep2 = grid(ex, yex)
        out += [
            "",
            ctx.L(f"月度超额（策略 − {ctx.bname()}）", "monthly excess_return"),
            t2,
        ]
        if not keep2:
            out.append(ctx.L("（表中数字单位为 %）", "(values in %)"))
    notes = []
    if mo["first_partial"] and mo["month"][0] >= lo:
        notes.append(
            ctx.L(
                f"{mo['month'][0]} 自 {ctx.data['summary']['first_trade_day'][5:]}",
                f"{mo['month'][0]} partial",
            )
        )
    if mo["last_partial"] and mo["month"][-1] <= hi:
        notes.append(
            ctx.L(
                f"{mo['month'][-1]} 至 {ctx.data['summary']['end'][5:]}",
                f"{mo['month'][-1]} partial",
            )
        )
    if notes:
        out.append(
            ctx.L("不完整的月份：", "partial months: ")
            + ("，" if ctx.lang == "zh" else ", ").join(notes)
        )
    s = ctx.data["summary"]
    if ctx.full:
        pairs = [
            ("月胜率", "monthly_win_rate", pct(s["monthly_win_rate"], 1)),
            ("月数", "months", f"{s['months']}" + ctx.L(" 个", "")),
        ]
        if hasb:
            pairs.append(
                (
                    "跑赢基准的月份占比",
                    "monthly_win_rate_vs_benchmark",
                    pct(s["monthly_win_rate_vs_benchmark"], 1),
                )
            )
        out.append(kv(ctx, pairs))
    return out


def monthly_json(ctx: Context):
    mo = ctx.data["monthly"]
    lo, hi = ctx.lo[:7], ctx.hi[:7]
    idx = [i for i, mk in enumerate(mo["month"]) if lo <= mk <= hi]
    return {
        "month": [mo["month"][i] for i in idx],
        "return": [mo["return"][i] for i in idx],
        "benchmark_return": None
        if mo.get("benchmark_return") is None
        else [mo["benchmark_return"][i] for i in idx],
        "first_partial": mo["first_partial"],
        "last_partial": mo["last_partial"],
    }


# ====================================================================== drawdowns
def drawdowns_blocks(ctx: Context) -> list:
    eps = _drawdowns_in(ctx)
    if not eps:
        return [ctx.L("没有回撤", "none")]
    return [
        ctx.L(
            f"按深度排列的前 {len(eps)} 次不重叠回撤；见底、持续为交易日数",
            "top non-overlapping drawdowns by depth; days are trading days",
        ),
        _drawdown_table(ctx, eps),
    ]


# ====================================================================== rolling
def _rolling_rows(ctx: Context):
    ro = ctx.data["rolling"]
    rows = []
    dates = ro["date"]
    for i, d in enumerate(dates):
        if not ctx.in_range(d):
            continue
        q = (d[:4], (int(d[5:7]) - 1) // 3)
        nxt = dates[i + 1] if i + 1 < len(dates) else None
        nq = None if nxt is None else (nxt[:4], (int(nxt[5:7]) - 1) // 3)
        if nq != q or not ctx.in_range(nxt):
            rows.append(i)
    return [
        {
            "date": dates[i],
            "annual_return": ro["annual_return"][i],
            "volatility": ro["volatility"][i],
            "sharpe": ro["sharpe"][i],
            "beta": None if ro.get("beta") is None else ro["beta"][i],
        }
        for i in rows
    ]


def rolling_blocks(ctx: Context) -> list:
    ro = ctx.data["rolling"]
    rows = _rolling_rows(ctx)
    if not rows:
        return [
            ctx.L(
                f"历史不足 {ro['window']} 个交易日，没有滚动指标",
                "history shorter than the window",
            )
        ]
    hasb = ro.get("beta") is not None
    cols = [
        (ctx.L("日期", "date"), "l"),
        (ctx.L("滚动收益", "annual_return"), "r"),
        (ctx.L("波动", "volatility"), "r"),
        (ctx.L("夏普", "sharpe"), "r"),
    ]
    if hasb:
        cols.append((ctx.L("Beta", "beta"), "r"))
    t = Table(cols)
    for r in rows:
        row = [
            r["date"],
            pct(r["annual_return"], 1, True),
            pct(r["volatility"], 1),
            num(r["sharpe"]),
        ]
        if hasb:
            row.append(num(r["beta"]))
        t.rows.append(row)
    return [
        ctx.L(
            f"窗口 {ro['window']} 个交易日，每季最后一个交易日一行；收益为窗口累计收益",
            f"window={ro['window']} trading days, one row per quarter end",
        ),
        t,
    ]


# ====================================================================== rebalances
RECENT_REBALANCES = 10


def _rebalances_in(ctx):
    return [r for r in ctx.data["rebalances"] if ctx.in_range(r["date"])]


def _rebalance_stats(rb):
    if not rb:
        return None
    by = {
        k: sum(r["unfilled_by_reason"][k] for r in rb)
        for k in ("limit_up", "limit_down", "suspended")
    }
    return {
        "count": len(rb),
        "mean_targets": sum(r["targets"] for r in rb) / len(rb),
        "mean_turnover": sum(r["turnover"] for r in rb) / len(rb),
        "fees": sum(r["fees"] for r in rb),
        "slippage": sum(r["slippage"] for r in rb),
        "unfilled": sum(r["unfilled"] for r in rb),
        "unfilled_by_reason": by,
        "unfilled_filled_before_next_rebalance": sum(
            r["unfilled_filled_before_next_rebalance"] for r in rb
        ),
        "new_names": sum(r["new_names"] for r in rb),
        "dropped_names": sum(r["dropped_names"] for r in rb),
    }


def rebalances_blocks(ctx: Context) -> list:
    rb = _rebalances_in(ctx)
    st = _rebalance_stats(rb)
    if st is None:
        return [ctx.L("区间内没有调仓", "none")]
    by = st["unfilled_by_reason"]
    out = [
        kv(
            ctx,
            [
                ("调仓", "count", f"{st['count']}" + ctx.L(" 次", "")),
                (
                    "平均目标",
                    "mean_targets",
                    f"{st['mean_targets']:.1f}" + ctx.L(" 只", ""),
                ),
                ("平均单次换手（双边）", "mean_turnover", pct(st["mean_turnover"], 1)),
                ("新进", "new_names", f"{st['new_names']}" + ctx.L(" 只次", "")),
                (
                    "剔除",
                    "dropped_names",
                    f"{st['dropped_names']}" + ctx.L(" 只次", ""),
                ),
            ],
        ),
        kv(
            ctx,
            [
                (
                    "调仓日开盘未成交",
                    "unfilled",
                    f"{st['unfilled']}" + ctx.L(" 只次", ""),
                ),
                ("涨停买不进", "limit_up", str(by["limit_up"])),
                ("跌停卖不出", "limit_down", str(by["limit_down"])),
                ("停牌", "suspended", str(by["suspended"])),
                (
                    "下次调仓前补成交",
                    "unfilled_filled_before_next_rebalance",
                    str(st["unfilled_filled_before_next_rebalance"]),
                ),
            ],
        ),
        ctx.L(
            f"最近 {min(RECENT_REBALANCES, len(rb))} 次（费用、滑点单位为元，含到下次调仓前的重试成交）：",
            f"recent {min(RECENT_REBALANCES, len(rb))} (fees, slippage in yuan):",
        ),
    ]
    t = Table(
        [
            (ctx.L("日期", "date"), "l"),
            (ctx.L("目标", "targets"), "r"),
            (ctx.L("买入", "buys"), "r"),
            (ctx.L("卖出", "sells"), "r"),
            (ctx.L("新进", "new_names"), "r"),
            (ctx.L("剔除", "dropped_names"), "r"),
            (ctx.L("换手", "turnover"), "r"),
            (ctx.L("费用", "fees"), "r"),
            (ctx.L("滑点", "slippage"), "r"),
            (ctx.L("未成交", "unfilled"), "r"),
            (ctx.L("持仓", "holdings_after"), "r"),
            (ctx.L("现金", "cash_weight_after"), "r"),
        ]
    )
    for r in rb[-RECENT_REBALANCES:]:
        t.rows.append(
            [
                r["date"],
                str(r["targets"]),
                str(r["buys"]),
                str(r["sells"]),
                str(r["new_names"]),
                str(r["dropped_names"]),
                pct(r["turnover"], 1),
                f"{r['fees']:,.2f}",
                f"{r['slippage']:,.2f}",
                str(r["unfilled"]),
                str(r["holdings_after"]),
                pct(r["cash_weight_after"], 2),
            ]
        )
    out.append(t)
    return out


def rebalances_json(ctx: Context):
    rb = _rebalances_in(ctx)
    return {"summary": _rebalance_stats(rb), "recent": rb[-RECENT_REBALANCES:]}


# ====================================================================== holdings
HOLDINGS_LIMIT = 50


def holdings_blocks(ctx: Context) -> list:
    H = ctx.data["holdings_latest"]
    dl = ctx.data["daily"]
    out = []
    if not ctx.full and ctx.hi != dl["date"][-1]:
        out.append(
            ctx.L(
                f"只有报告末日 {dl['date'][-1]} 的持仓；所选区间不影响本节",
                "holdings are for the report end date only",
            )
        )
    out.append(
        kv(
            ctx,
            [
                ("日期", "date", dl["date"][-1]),
                ("持仓", "holdings", f"{len(H)}" + ctx.L(" 只", "")),
                ("现金", "cash_weight", pct(dl["cash_weight"][-1], 2)),
            ],
        )
    )
    if not H:
        return out
    hasi = any(h.get("industry") for h in H)
    cols = [(ctx.L("代码", "symbol"), "l"), (ctx.L("名称", "name"), "l")]
    if hasi:
        cols.append((ctx.L("行业", "industry"), "l"))
    cols += [(ctx.L("权重", "weight"), "r"), (ctx.L("股数", "shares"), "r")]
    t = Table(cols)
    for h in H[:HOLDINGS_LIMIT]:
        row = [h["symbol"], h.get("name") or MISSING]
        if hasi:
            row.append(h.get("industry") or MISSING)
        row += [pct(h["weight"], 2), f"{h['shares']:,.0f}" + ctx.L(" 股", "")]
        t.rows.append(row)
    out.append(t)
    if len(H) > HOLDINGS_LIMIT:
        rest = H[HOLDINGS_LIMIT:]
        out.append(
            ctx.L(
                f"其余 {len(rest)} 只合计 {pct(sum(h['weight'] for h in rest), 2)}",
                f"other {len(rest)} names: {pct(sum(h['weight'] for h in rest), 2)}",
            )
        )
    return out


def holdings_json(ctx: Context):
    dl = ctx.data["daily"]
    return {
        "date": dl["date"][-1],
        "cash_weight": dl["cash_weight"][-1],
        "holdings": ctx.data["holdings_latest"],
    }


# ====================================================================== industry
def _industry_rows(ctx):
    iw = ctx.data.get("industry_weights")
    if not iw or not iw["industries"]:
        return None, None, []
    js = [j for j, d in enumerate(iw["dates"]) if ctx.in_range(d)]
    if not js:
        return iw, [], []
    rows = []
    for i, name in enumerate(iw["industries"]):
        w = [iw["weights"][i][j] for j in js]
        rows.append(
            {"industry": name, "latest": w[-1], "mean": sum(w) / len(w), "max": max(w)}
        )
    cash = [iw["cash"][j] for j in js]
    rows.sort(key=lambda r: (-r["latest"], -r["mean"], r["industry"]))
    rows.append(
        {
            "industry": "cash",
            "latest": cash[-1],
            "mean": sum(cash) / len(cash),
            "max": max(cash),
        }
    )
    return iw, rows, js


def industry_blocks(ctx: Context) -> list:
    iw, rows, js = _industry_rows(ctx)
    if iw is None:
        return [ctx.L("报告里没有行业数据", "no industry data")]
    if not rows:
        return [ctx.L("区间内没有取样点", "no samples in range")]
    samp = (
        ctx.L("调仓日", "rebalance")
        if iw["sampling"] == "rebalance"
        else ctx.L("月末", "month_end")
    )
    out = [
        kv(
            ctx,
            [
                ("取样", "sampling", samp),
                ("取样点", "samples", f"{len(js)}" + ctx.L(" 个", "")),
                ("最新", "date", iw["dates"][js[-1]]),
            ],
        )
    ]
    t = Table(
        [
            (ctx.L("行业", "industry"), "l"),
            (ctx.L("最新", "latest"), "r"),
            (ctx.L("平均", "mean"), "r"),
            (ctx.L("最高", "max"), "r"),
        ]
    )
    for r in rows:
        name = (
            ctx.L("现金", "cash")
            if r["industry"] == "cash" and r is rows[-1]
            else r["industry"]
        )
        if r is not rows[-1] and r["latest"] == 0 and r["max"] == 0:
            continue
        t.rows.append([name, pct(r["latest"], 1), pct(r["mean"], 1), pct(r["max"], 1)])
    out.append(t)
    sw = ctx.data["meta"].get("industry_switch")
    if sw and ctx.in_range(sw):
        lab = ctx.data["meta"].get("industry_switch_labels") or ["", ""]
        out.append(
            ctx.L(f"行业分类自 {sw} 由 {lab[0]} 换成 {lab[1]}", f"industry_switch={sw}")
        )
    return out


def industry_json(ctx: Context):
    iw, rows, js = _industry_rows(ctx)
    if iw is None:
        return None
    return {
        "sampling": iw["sampling"],
        "dates": [iw["dates"][j] for j in js],
        "rows": rows,
        "industry_switch": ctx.data["meta"].get("industry_switch"),
    }


# ====================================================================== costs
def _rate(x):
    if x is None:
        return MISSING
    if isinstance(x, list):
        return f"{pct(x[0], 3)}–{pct(x[1], 3)}"
    return pct(x, 3)


def costs_blocks(ctx: Context) -> list:
    s, m, w = ctx.data["summary"], ctx.data["meta"], ctx.ws
    out = []
    c = m.get("costs")
    if c:
        pairs = []
        if "commission_rate" in c:
            pairs += [
                ("佣金率", "commission_rate", _rate(c["commission_rate"])),
                (
                    "最低佣金",
                    "commission_min",
                    MISSING
                    if c["commission_min"] is None
                    else f"{c['commission_min']:.2f}" + ctx.L(" 元", ""),
                ),
                ("卖出印花税", "stamp_tax_sell", _rate(c["stamp_tax_sell"])),
                ("滑点", "slippage", _rate(c["slippage"])),
            ]
        if "dividend_tax" in c:
            pairs.append(("红利税", "dividend_tax", pct(c["dividend_tax"], 1)))
        out.append(kv(ctx, pairs))
    if ctx.full:
        out.append(
            kv(
                ctx,
                [
                    (
                        "初始资金",
                        "initial_capital",
                        f"{m['initial_capital']:,.2f}" + ctx.L(" 元", ""),
                    ),
                    (
                        "费用合计",
                        "total_fees",
                        f"{s['total_fees']:,.2f}" + ctx.L(" 元", ""),
                    ),
                    (
                        "佣金",
                        "total_commission",
                        f"{s['total_commission']:,.2f}" + ctx.L(" 元", ""),
                    ),
                    (
                        "印花税",
                        "total_stamp_tax",
                        f"{s['total_stamp_tax']:,.2f}" + ctx.L(" 元", ""),
                    ),
                ],
            )
        )
        out.append(
            kv(
                ctx,
                [
                    (
                        "滑点与冲击",
                        "total_slippage",
                        f"{s['total_slippage']:,.2f}" + ctx.L(" 元", ""),
                    ),
                    (
                        "税前分红",
                        "total_dividend_gross",
                        f"{s['total_dividend_gross']:,.2f}" + ctx.L(" 元", ""),
                    ),
                    (
                        "红利税",
                        "total_dividend_tax",
                        f"{s['total_dividend_tax']:,.2f}" + ctx.L(" 元", ""),
                    ),
                ],
            )
        )
    out.append(
        kv(
            ctx,
            [
                (
                    "费用 / 期初权益",
                    "total_fees_to_initial",
                    pct(w["total_fees_to_initial"], 2),
                ),
                (
                    "滑点 / 期初权益",
                    "total_slippage_to_initial",
                    pct(w["total_slippage_to_initial"], 2),
                ),
                ("年化换手（双边）", "annual_turnover", pct(w["annual_turnover"], 0)),
            ]
            + (
                [
                    (
                        "年化换手（单边）",
                        "annual_turnover_one_sided",
                        pct(s["annual_turnover_one_sided"], 0),
                    )
                ]
                if ctx.full
                else []
            ),
        )
    )
    A = _attr(ctx)
    t = Table(
        [
            (ctx.L("年份", "year"), "l"),
            (ctx.L("价格", "price"), "r"),
            (ctx.L("税后分红", "dividend_net"), "r"),
            (ctx.L("费用", "fees"), "r"),
            (ctx.L("滑点", "slippage"), "r"),
            (ctx.L("合计", "total"), "r"),
        ]
    )
    for y in ctx.years():
        a = None if y.get("clipped") else A.get(y["year"])
        if a:
            t.rows.append(
                [
                    f"{y['year']}{'*' if y['partial'] else ''}",
                    pct(a["price"], 2, True),
                    pct(a["dividend_net"], 2, True),
                    pct(a["fees"], 2, True),
                    pct(a["slippage"], 2, True),
                    pct(a["total"], 2, True),
                ]
            )
    out += [
        ctx.L(
            "分年归因（÷ 上年末权益）",
            "attribution (divided by previous year-end equity)",
        ),
        t,
    ]
    return out


def costs_json(ctx: Context):
    s = ctx.data["summary"]
    keys = [
        "total_fees",
        "total_fees_to_initial",
        "total_slippage",
        "total_slippage_to_initial",
        "total_commission",
        "total_stamp_tax",
        "total_dividend_gross",
        "total_dividend_tax",
        "annual_turnover",
        "annual_turnover_one_sided",
    ]
    tot = (
        {k: s[k] for k in keys}
        if ctx.full
        else {k: ctx.ws.get(k) for k in keys if k in ctx.ws}
    )
    A = _attr(ctx)
    return {
        "settings": ctx.data["meta"].get("costs"),
        "initial_capital": ctx.data["meta"].get("initial_capital"),
        "totals": tot,
        "attribution": [
            A[y["year"]] for y in ctx.years() if y["year"] in A and not y.get("clipped")
        ],
    }


# ====================================================================== warnings
def warnings_blocks(ctx: Context) -> list:
    ws = checks.warnings(ctx.data, ctx.a, ctx.b)
    if not ws:
        return [ctx.L("没有需要提醒的问题", "none")]
    return [f"- [{w['code']}] {w['message']}" for w in ws]


def warnings_json(ctx: Context):
    return checks.warnings(ctx.data, ctx.a, ctx.b)


SECTIONS: dict[str, Section] = {
    "summary": Section("摘要", summary_blocks, summary_json),
    "yearly": Section("分年", yearly_blocks, yearly_json),
    "monthly": Section("月度收益", monthly_blocks, monthly_json),
    "drawdowns": Section("回撤", drawdowns_blocks, _drawdowns_in),
    "rolling": Section("滚动指标", rolling_blocks, _rolling_rows),
    "rebalances": Section("调仓", rebalances_blocks, rebalances_json),
    "holdings": Section("最新持仓", holdings_blocks, holdings_json),
    "industry": Section("行业权重", industry_blocks, industry_json),
    "costs": Section("费用与归因", costs_blocks, costs_json),
    "warnings": Section("提醒", warnings_blocks, warnings_json),
}


def resolve_sections(sections) -> list[str]:
    if sections is None:
        return ["summary"]
    if isinstance(sections, str):
        sections = [x for x in sections.split(",") if x]
    out = []
    for s in sections:
        s = s.strip()
        if s == "all":
            return list(SECTIONS)
        if s not in SECTIONS:
            raise ValueError(f"没有小节 {s!r}；可选：{', '.join(SECTIONS)}, all")
        if s not in out:
            out.append(s)
    return out
