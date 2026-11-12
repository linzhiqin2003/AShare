"""任选区间上的指标重算，与页面「切换区间」用同一套公式。

页面在浏览器里用嵌入的数据（`render.compact` 取整后的日度序列）按 `metrics(a, b)` 重算；
这里逐行照搬那段 JS：同样的输入（取整后的序列）、同样的求和顺序、同样的公式。
全区间（a = 0、b = 末行）时和页面一样直接取 `summary` 里的数。

a 是区间的基准日（区间收益从 a+1 行算起），b 是区间末日；都是 `daily.date` 的行号。
"""

from __future__ import annotations

import math

from .render import compact


def _stats(v: list[float]) -> tuple[float, float]:
    n = len(v)
    s = 0.0
    for x in v:
        s += x
    m = s / n if n else math.nan
    q = 0.0
    for x in v:
        q += (x - m) * (x - m)
    return m, (math.sqrt(q / (n - 1)) if n > 1 else math.nan)


def _mdd(arr) -> float:
    pk, m = -math.inf, 0.0
    for x in arr:
        if x is None:
            continue
        if x > pk:
            pk = x
        m = min(m, x / pk - 1)
    return m


def _nan_none(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else x


def resolve(data: dict, start=None, end=None) -> tuple[int, int]:
    """把 `--from/--to` 的日期换成行号 (a, b)。

    start：区间第一个收益日不早于它，基准日 a 取它前一个交易日（没有更早的行时取第 0 行）；
    end：b 取不晚于它的最后一个交易日。都不给时是全区间 (0, T−1)。
    """
    from .build import ReportError, _iso

    dates = data["daily"]["date"]
    T = len(dates)
    a, b = 0, T - 1
    if start is not None:
        s = _iso(start)
        first = next((i for i, d in enumerate(dates) if d >= s), None)
        if first is None:
            raise ReportError(f"起始日 {s} 晚于报告最后一天 {dates[-1]}")
        a = max(0, first - 1)
    if end is not None:
        e = _iso(end)
        last = next((i for i in range(T - 1, -1, -1) if dates[i] <= e), None)
        if last is None:
            raise ReportError(f"截止日 {e} 早于报告第一天 {dates[0]}")
        b = last
    if b - a < 1:
        raise ReportError("所选区间内没有交易日")
    return a, b


def metrics(
    data: dict,
    a: int,
    b: int,
    embedded: dict | None = None,
    *,
    use_summary: bool = True,
) -> dict:
    """页面 `metrics(a, b)` 的 Python 版。返回的键与页面一致：

    n、cum、ann、vol、sharpe、mdd、turnover、fees、slip，基准一组 bn、bcum、bann、bvol、bsharpe、
    bmdd、excess、excum、te、ir、beta（无基准或重叠不足两天时为 None）。
    use_summary=False 时全区间也按公式重算（只供测试核对公式用；页面不这样做）。
    """
    D = embedded if embedded is not None else compact(data)
    dl, S, M = D["daily"], D["summary"], D["meta"]
    N = len(dl["date"])
    PY = M.get("periods_per_year") or 252
    full = a == 0 and b == N - 1 and use_summary
    n = b - a
    nav = dl["nav"]
    r = dl["return"][a + 1 : b + 1]
    cum = nav[b] / nav[a] - 1
    ann = math.pow(1 + cum, PY / n) - 1
    sm, ssd = _stats(r)
    vol = ssd * math.sqrt(PY)
    sharpe = sm / ssd * math.sqrt(PY) if ssd > 0 else None
    to = 0.0
    for i in range(a + 1, b + 1):
        to += dl["turnover"][i]
    idx = {d: i for i, d in enumerate(dl["date"])}
    fees = slip = 0.0
    for rb in D["rebalances"]:
        k = idx[rb["date"]]
        if a < k <= b:
            fees += rb["fees"]
            slip += rb["slippage"]
    eq0 = nav[a] * (M.get("initial_capital") or 1)
    yrs = n / PY
    m = {
        "n": n,
        "cum": cum,
        "ann": ann,
        "vol": vol,
        "sharpe": sharpe,
        "mdd": _mdd(nav[a : b + 1]),
        "turnover": to / yrs,
        "fees": fees / eq0,
        "slip": slip / eq0,
        "bcum": None,
        "bann": None,
        "bvol": None,
        "bsharpe": None,
        "bmdd": None,
        "excess": None,
        "excum": None,
        "te": None,
        "ir": None,
        "beta": None,
        "bn": 0,
    }
    bnav = dl.get("benchmark_nav")
    hasb = bool(M.get("benchmark") and bnav)
    BS = next((i for i, v in enumerate(bnav) if v is not None), -1) if hasb else -1
    o = max(a, BS) if hasb else -1
    if hasb and b - o >= 2:
        bn = b - o
        q = dl["benchmark_return"][o + 1 : b + 1]
        rr = dl["return"][o + 1 : b + 1]
        bcum = bnav[b] / bnav[o] - 1
        scum = nav[b] / nav[o] - 1
        qm, qsd = _stats(q)
        ex = [x - q[i] for i, x in enumerate(rr)]
        em, esd = _stats(ex)
        te = esd * math.sqrt(PY)
        rm, _ = _stats(rr)
        cv = 0.0
        for i in range(len(rr)):
            cv += (rr[i] - rm) * (q[i] - qm)
        cv /= len(rr) - 1
        bann = math.pow(1 + bcum, PY / bn) - 1
        m.update(
            bn=bn,
            bcum=bcum,
            bann=bann,
            bvol=qsd * math.sqrt(PY),
            bsharpe=qm / qsd * math.sqrt(PY) if qsd > 0 else None,
            bmdd=_mdd(bnav[o : b + 1]),
            te=te,
            ir=em * PY / te if te > 0 else None,
            beta=cv / (qsd * qsd) if qsd > 0 else None,
        )
        m["excess"] = (math.pow(1 + scum, PY / bn) - 1) - m["bann"]
        m["excum"] = scum - bcum
    if full:
        m.update(
            cum=S["cumulative_return"],
            ann=S["annual_return"],
            vol=S["annual_volatility"],
            sharpe=S["sharpe"],
            mdd=S["max_drawdown"],
            turnover=S["annual_turnover"],
            fees=S["total_fees_to_initial"],
            slip=S["total_slippage_to_initial"],
        )
        if hasb:
            m.update(
                bcum=S["benchmark_cumulative_return"],
                bann=S["benchmark_annual_return"],
                bvol=S["benchmark_annual_volatility"],
                bsharpe=S["benchmark_sharpe"],
                bmdd=S["benchmark_max_drawdown"],
                excess=S["excess_annual_return"],
                te=S["tracking_error"],
                ir=S["information_ratio"],
                beta=S["beta"],
            )
    return {k: _nan_none(v) for k, v in m.items()}


# 页面键 → summary 键（文本版和 JSON 输出统一用 summary 的键名）
SUMMARY_KEYS = {
    "cum": "cumulative_return",
    "ann": "annual_return",
    "vol": "annual_volatility",
    "sharpe": "sharpe",
    "mdd": "max_drawdown",
    "turnover": "annual_turnover",
    "fees": "total_fees_to_initial",
    "slip": "total_slippage_to_initial",
    "bcum": "benchmark_cumulative_return",
    "bann": "benchmark_annual_return",
    "bvol": "benchmark_annual_volatility",
    "bsharpe": "benchmark_sharpe",
    "bmdd": "benchmark_max_drawdown",
    "excess": "excess_annual_return",
    "excum": "excess_cumulative_return",
    "te": "tracking_error",
    "ir": "information_ratio",
    "beta": "beta",
    "n": "trading_days",
}

# 页面在区间不足这么多个交易日时只显示区间收益，不显示年化
MIN_ANNUAL_DAYS = 240


def window_summary(data: dict, a: int, b: int) -> dict:
    """`metrics` 换成 summary 的键名，并带上区间起止。"""
    m = metrics(data, a, b)
    dates = data["daily"]["date"]
    out = {SUMMARY_KEYS[k]: v for k, v in m.items() if k in SUMMARY_KEYS}
    out.update(
        start=dates[a],
        first_trade_day=dates[a + 1],
        end=dates[b],
        annualized=m["n"] >= MIN_ANNUAL_DAYS,
        full_range=(a == 0 and b == len(dates) - 1),
    )
    return out
