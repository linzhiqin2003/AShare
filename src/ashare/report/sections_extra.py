"""补充内容的文本小节：benchmarks、winrate、contributions、tails、concentration。

数字取自报告数据里对应的键（`extras.py` 算出）；报告是在加入这些内容之前存的、或用
`extras=False` 生成的，小节只写一行说明。
"""

from __future__ import annotations

from .fmt import MISSING, num, pct
from .sections import SECTIONS, Context, Section, Table, kv

KIND_SHORT = {
    "price": ("价格指数", "price"),
    "total_return": ("全收益", "total_return"),
}


def _absent(ctx: Context, key: str):
    if key not in ctx.data:
        return [
            ctx.L(
                "报告数据里没有这一节（用旧版或 extras=False 生成）",
                f"no '{key}' in report data",
            )
        ]
    return None


# ====================================================================== benchmarks
def _bench_rows(ctx: Context) -> list[dict]:
    s, b = ctx.data["summary"], ctx.data["meta"].get("benchmark")
    rows = []
    if b and s.get("benchmark_annual_return") is not None:
        rows.append(
            {
                "name": b["name"],
                "symbol": b.get("symbol"),
                "kind": b["kind"],
                "main": True,
                "start": s["benchmark_start"],
                "cumulative_return": s["benchmark_cumulative_return"],
                "annual_return": s["benchmark_annual_return"],
                "annual_volatility": s["benchmark_annual_volatility"],
                "sharpe": s["benchmark_sharpe"],
                "max_drawdown": s["benchmark_max_drawdown"],
                "excess_annual_return": s["excess_annual_return"],
                "information_ratio": s["information_ratio"],
                "tracking_error": s["tracking_error"],
                "beta": s["beta"],
            }
        )
    for x in ctx.data.get("benchmarks") or []:
        rows.append({**{k: v for k, v in x.items() if k != "yearly"}, "main": False})
    return rows


def benchmarks_blocks(ctx: Context) -> list:
    if (miss := _absent(ctx, "benchmarks")) is not None:
        return miss
    rows = _bench_rows(ctx)
    if not rows:
        return [ctx.L("没有基准，也没有对比指数", "none")]
    out = []
    if not ctx.full:
        out.append(ctx.L("本节为全区间数字", "full-range numbers"))
    t = Table(
        [
            (ctx.L("指数", "name"), "l"),
            (ctx.L("口径", "kind"), "l"),
            (ctx.L("累计", "cumulative_return"), "r"),
            (ctx.L("年化", "annual_return"), "r"),
            (ctx.L("波动", "annual_volatility"), "r"),
            (ctx.L("最大回撤", "max_drawdown"), "r"),
            (ctx.L("年化超额", "excess_annual_return"), "r"),
            (ctx.L("信息比率", "information_ratio"), "r"),
            (ctx.L("跟踪误差", "tracking_error"), "r"),
        ]
    )
    for r in rows:
        zh, en = KIND_SHORT[r["kind"]]
        t.rows.append(
            [
                r["name"] + (ctx.L("（主）", "*") if r["main"] else ""),
                ctx.L(zh, en),
                pct(r["cumulative_return"], 1, True),
                pct(r["annual_return"], 1, True),
                pct(r["annual_volatility"], 1),
                pct(r["max_drawdown"], 1),
                pct(r["excess_annual_return"], 1, True),
                num(r["information_ratio"]),
                pct(r["tracking_error"], 1),
            ]
        )
    out.append(t)
    late = [r for r in rows if r["start"] != ctx.data["summary"]["start"]]
    for r in late:
        out.append(
            ctx.L(
                f"{r['name']} 自 {r['start']} 起有数据，相对指标只算重叠区间",
                f"{r['name']} starts {r['start']}",
            )
        )
    extra = ctx.data.get("benchmarks") or []
    if extra:
        cols = [(ctx.L("年份", "year"), "l"), (ctx.L("策略", "return"), "r")]
        main = ctx.bench
        if main:
            cols.append((main["name"], "r"))
        cols += [(x["name"], "r") for x in extra]
        yt = Table(cols)
        for k, y in enumerate(ctx.data["yearly"]):
            if not (int(ctx.lo[:4]) <= y["year"] <= int(ctx.hi[:4])):
                continue
            row = [
                f"{y['year']}{'*' if y['partial'] else ''}",
                pct(y["return"], 1, True),
            ]
            if main:
                row.append(pct(y["benchmark_return"], 1, True))
            row += [pct(x["yearly"][k]["return"], 1, True) for x in extra]
            yt.rows.append(row)
        out += [
            "",
            ctx.L("分年收益（* 不完整的年份）", "yearly return (* partial)"),
            yt,
        ]
    return out


def benchmarks_json(ctx: Context):
    if "benchmarks" not in ctx.data:
        return None
    return {"rows": _bench_rows(ctx), "extra": ctx.data.get("benchmarks") or []}


# ====================================================================== winrate
def winrate_blocks(ctx: Context) -> list:
    if (miss := _absent(ctx, "win_rates")) is not None:
        return miss
    W = ctx.data.get("win_rates")
    if not W:
        return [ctx.L("没有基准，无法算胜率", "no benchmark")]
    out = []
    if not ctx.full:
        out.append(ctx.L("本节为全区间数字", "full-range numbers"))
    t = Table(
        [
            (ctx.L("指数", "name"), "l"),
            (ctx.L("持有期", "period"), "l"),
            (ctx.L("样本", "windows"), "r"),
            (ctx.L("跑赢", "won"), "r"),
            (ctx.L("胜率", "ratio"), "r"),
            (ctx.L("最差一段", "worst"), "l"),
            (ctx.L("策略", "strategy"), "r"),
            (ctx.L("指数", "benchmark"), "r"),
            (ctx.L("超额", "excess"), "r"),
        ]
    )
    for w in W:
        t.rows.append(
            [
                w["name"],
                ctx.L("自然年", "calendar_year"),
                f"{w['years_counted']}" + ctx.L(" 年", ""),
                f"{w['years_won']}" + ctx.L(" 年", ""),
                pct(w["years_ratio"], 1),
                "",
                "",
                "",
                "",
            ]
        )
        worst_year = min(
            (y for y in w["yearly"] if not y["partial"] and y["excess"] is not None),
            key=lambda y: (y["excess"], y["year"]),
            default=None,
        )
        if worst_year:
            t.rows[-1][5:] = [
                str(worst_year["year"]),
                pct(worst_year["strategy"], 1, True),
                pct(worst_year["benchmark"], 1, True),
                pct(worst_year["excess"], 1, True),
            ]
        for k, r in w["rolling"].items():
            label = ctx.L(f"任意 {k[:-1]} 年", f"rolling_{k}")
            if r is None:
                t.rows.append(["", label, MISSING, MISSING, MISSING, "", "", "", ""])
                continue
            wst = r["worst"]
            t.rows.append(
                [
                    "",
                    label,
                    f"{r['windows']}" + ctx.L(" 段", ""),
                    f"{r['won']}" + ctx.L(" 段", ""),
                    pct(r["ratio"], 1),
                    f"{wst['start']}~{wst['end']}",
                    pct(wst["strategy"], 1, True),
                    pct(wst["benchmark"], 1, True),
                    pct(wst["excess"], 1, True),
                ]
            )
    out.append(t)
    parts = sorted({y for w in W for y in w["partial_years"]})
    note = ctx.L(
        "滚动持有期按月末滑动；跑赢指策略收益高于指数收益",
        "rolling windows slide by month end; won = strategy return > index return",
    )
    if parts:
        note += ctx.L(
            "；不完整的年份不计入：" + "、".join(map(str, parts)),
            "; partial years excluded: " + ", ".join(map(str, parts)),
        )
    out.append(note)
    return out


def winrate_json(ctx: Context):
    return ctx.data.get("win_rates") if "win_rates" in ctx.data else None


# ====================================================================== contributions
YEAR_NAMES = 3


def _who(h) -> str:
    return h["name"] or h["symbol"]


def contributions_blocks(ctx: Context) -> list:
    if (miss := _absent(ctx, "contributions")) is not None:
        return miss
    C = ctx.data["contributions"]
    tot = C["total"]
    out = []
    if not ctx.full:
        out.append(
            ctx.L(
                "全区间一节不随所选区间变化；分年只列区间内的年份", "full-range table"
            )
        )
    out.append(
        kv(
            ctx,
            [
                (
                    "全区间涉及证券",
                    "securities",
                    f"{tot['securities']}" + ctx.L(" 只", ""),
                ),
                (
                    "盈亏合计",
                    "total_pnl",
                    f"{tot['total_pnl']:,.2f}" + ctx.L(" 元", ""),
                ),
                (
                    "现金利息",
                    "cash_interest",
                    f"{tot['cash_interest']:,.2f}" + ctx.L(" 元", ""),
                ),
                (
                    "权益变化",
                    "equity_change",
                    f"{tot['equity_change']:,.2f}" + ctx.L(" 元", ""),
                ),
            ],
        )
    )
    for key, zh, en in (("top", "贡献前十", "top"), ("bottom", "贡献后十", "bottom")):
        t = Table(
            [
                (ctx.L("代码", "symbol"), "l"),
                (ctx.L("名称", "name"), "l"),
                (ctx.L("贡献", "contribution"), "r"),
                (ctx.L("盈亏（元）", "pnl"), "r"),
                (ctx.L("其中分红", "dividend"), "r"),
                (ctx.L("持有", "holding_days"), "r"),
            ]
        )
        for h in tot[key]:
            t.rows.append(
                [
                    h["symbol"],
                    h["name"] or MISSING,
                    pct(h["contribution"], 2, True),
                    f"{h['pnl']:,.2f}",
                    f"{h['dividend']:,.2f}",
                    ctx.days(h["holding_days"]),
                ]
            )
        out += ["", ctx.L(f"{zh}（占期初权益）", en), t]
    rows = [
        y for y in C["yearly"] if int(ctx.lo[:4]) <= int(y["period"]) <= int(ctx.hi[:4])
    ]
    if rows:
        out += [
            "",
            ctx.L(
                f"分年（每年前 {YEAR_NAMES} 与后 {YEAR_NAMES}，占当年期初权益；* 不完整）",
                f"yearly top/bottom {YEAR_NAMES}",
            ),
        ]
        for y in rows:
            top = "、".join(
                f"{_who(h)} {pct(h['contribution'], 2, True)}"
                for h in y["top"][:YEAR_NAMES]
            )
            bot = "、".join(
                f"{_who(h)} {pct(h['contribution'], 2, True)}"
                for h in y["bottom"][:YEAR_NAMES]
            )
            star = "*" if y["partial"] else ""
            out.append(
                ctx.L(
                    f"{y['period']}{star} 前：{top or MISSING}；后：{bot or MISSING}",
                    f"{y['period']}{star} top: {top or MISSING}; bottom: {bot or MISSING}",
                )
            )
    out.append(
        ctx.L(
            "贡献 = 价格盈亏（含滑点）+ 税后分红 − 费用；各股合计 + 现金利息 = 权益变化",
            "contribution = price (incl. slippage) + dividend - fees",
        )
    )
    return out


def contributions_json(ctx: Context):
    if "contributions" not in ctx.data:
        return None
    C = ctx.data["contributions"]
    rows = [
        y for y in C["yearly"] if int(ctx.lo[:4]) <= int(y["period"]) <= int(ctx.hi[:4])
    ]
    return {"total": C["total"], "yearly": rows}


# ====================================================================== tails
def tails_blocks(ctx: Context) -> list:
    if (miss := _absent(ctx, "tails")) is not None:
        return miss
    t = ctx.data["tails"]
    out = []
    if not ctx.full:
        out.append(ctx.L("本节为全区间数字", "full-range numbers"))
    out.append(
        kv(
            ctx,
            [
                ("日收益偏度", "daily_skew", num(t["daily_skew"])),
                ("超额峰度", "daily_kurtosis", num(t["daily_kurtosis"])),
                ("月收益偏度", "monthly_skew", num(t["monthly_skew"])),
                ("月超额峰度", "monthly_kurtosis", num(t["monthly_kurtosis"])),
            ],
        )
    )
    out.append(
        kv(
            ctx,
            [
                ("日 VaR 95%", "var_95", pct(t["var_95"], 2)),
                ("CVaR 95%", "cvar_95", pct(t["cvar_95"], 2)),
                ("VaR 99%", "var_99", pct(t["var_99"], 2)),
                ("CVaR 99%", "cvar_99", pct(t["cvar_99"], 2)),
            ],
        )
    )
    n = t["longest_losing_months"]
    out.append(
        ctx.L(
            f"最长连续亏损 {n} 个月"
            + (
                f"（{t['longest_losing_start']} 至 {t['longest_losing_end']}）"
                if n
                else ""
            ),
            f"longest_losing_months={n} start={t['longest_losing_start'] or MISSING} "
            f"end={t['longest_losing_end'] or MISSING}",
        )
    )
    hasb = ctx.bench is not None
    bn = ctx.L(ctx.bname(), "benchmark_return")
    dt = Table(
        [(ctx.L("日期", "date"), "l"), (ctx.L("策略", "return"), "r")]
        + ([(bn, "r")] if hasb else [])
    )
    for d in t["worst_days"]:
        dt.rows.append(
            [d["date"], pct(d["return"], 2, True)]
            + ([pct(d["benchmark_return"], 2, True)] if hasb else [])
        )
    mt = Table(
        [(ctx.L("月份", "month"), "l"), (ctx.L("策略", "return"), "r")]
        + ([(bn, "r")] if hasb else [])
    )
    for d in t["worst_months"]:
        mt.rows.append(
            [d["month"], pct(d["return"], 1, True)]
            + ([pct(d["benchmark_return"], 1, True)] if hasb else [])
        )
    out += [
        "",
        ctx.L("最差的 5 天", "worst_days"),
        dt,
        "",
        ctx.L("最差的 5 个月", "worst_months"),
        mt,
    ]
    return out


def tails_json(ctx: Context):
    return ctx.data.get("tails")


# ====================================================================== concentration
def concentration_blocks(ctx: Context) -> list:
    if (miss := _absent(ctx, "concentration")) is not None:
        return miss
    c = ctx.data["concentration"]
    out = []
    if not ctx.full:
        out.append(ctx.L("本节为全区间数字", "full-range numbers"))
    out.append(
        ctx.L(
            f"按有持仓的 {c['days']} 个交易日统计；权重为占权益比例",
            f"days_with_holdings={c['days']}",
        )
    )
    t = Table(
        [
            (ctx.L("指标", "metric"), "l"),
            (ctx.L("平均", "mean"), "r"),
            (ctx.L("最高", "max"), "r"),
            (ctx.L("最高日", "max_date"), "l"),
            (ctx.L("最新", "latest"), "r"),
        ]
    )
    for key, zh, f in (
        ("top10_weight", "前十大权重", lambda x: pct(x, 1)),
        (
            "effective_holdings",
            "有效持仓数",
            lambda x: MISSING if x is None else f"{x:.1f}" + ctx.L(" 只", ""),
        ),
        ("top_industry_weight", "第一大行业", lambda x: pct(x, 1)),
    ):
        st = c.get(key)
        if st is None:
            continue
        t.rows.append(
            [
                ctx.L(zh, key),
                f(st["mean"]),
                f(st["max"]),
                st["max_date"] or MISSING,
                f(st["latest"]),
            ]
        )
    out.append(t)
    out.append(
        ctx.L(
            "有效持仓数 = 1 ÷ Σ(个股权重 ÷ 股票总权重)²",
            "effective_holdings = 1 / sum((w_i / sum w)^2)",
        )
    )
    return out


def concentration_json(ctx: Context):
    c = ctx.data.get("concentration")
    if c is None:
        return None
    return {k: v for k, v in c.items() if k != "daily"}


EXTRA_SECTIONS = {
    "benchmarks": Section("对比指数", benchmarks_blocks, benchmarks_json),
    "winrate": Section("胜率", winrate_blocks, winrate_json),
    "contributions": Section("个股贡献", contributions_blocks, contributions_json),
    "tails": Section("收益分布与尾部", tails_blocks, tails_json),
    "concentration": Section("集中度", concentration_blocks, concentration_json),
}


def _register():
    """接到 `sections.SECTIONS` 里，放在 warnings 之前。"""
    items = [(k, v) for k, v in SECTIONS.items() if k not in EXTRA_SECTIONS]
    warn = [kv_ for kv_ in items if kv_[0] == "warnings"]
    items = [kv_ for kv_ in items if kv_[0] != "warnings"]
    SECTIONS.clear()
    SECTIONS.update(items + list(EXTRA_SECTIONS.items()) + warn)


_register()
