"""报告的补充内容：对比指数、胜率、个股贡献、收益分布与尾部、集中度。

都由 `build_tearsheet` 调用、写进报告数据的新键（`benchmarks`、`win_rates`、`contributions`、
`tails`、`concentration`），不改动原有的键；页面暂不显示（`render.compact` 会去掉它们）。
口径见 `docs/REPORT_NOTES.md`「补充内容」。
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from . import metrics as m
from .metrics import rnd

TOP_N = 10
ROLLING_YEARS = (1, 3, 5)
VAR_LEVELS = (0.95, 0.99)
WORST_N = 5


# ====================================================================== 对比指数
def bench_stats(r, br, bnav, bstart, dates, P) -> dict:
    """策略相对一条指数的全区间指标（只在指数有数据的重叠区间上算，口径同 summary）。"""
    o = bstart
    ro, bo = r[o + 1 :], br[o + 1 :]
    out = {
        "start": dates[o],
        "cumulative_return": rnd(bnav[-1] - 1),
        "annual_return": None,
        "annual_volatility": None,
        "sharpe": None,
        "max_drawdown": None,
        "strategy_annual_return": None,
        "excess_annual_return": None,
        "tracking_error": None,
        "information_ratio": None,
        "beta": None,
    }
    if len(bo) < 2:
        return out
    sa, ba = m.annual_return(ro, P), m.annual_return(bo, P)
    active = ro - bo
    te = m.volatility(active, P)
    out.update(
        annual_return=rnd(ba),
        annual_volatility=rnd(m.volatility(bo, P)),
        sharpe=rnd(m.sharpe(bo, P), 4),
        max_drawdown=rnd(m.max_drawdown_of_returns(bo)),
        strategy_annual_return=rnd(sa),
        excess_annual_return=rnd(sa - ba),
        tracking_error=rnd(te),
        information_ratio=rnd(active.mean() * P / te, 4) if te else None,
        beta=rnd(m.beta(ro, bo), 4),
    )
    return out


# ====================================================================== 胜率
def year_rows(nav, bnav, bstart, dates, year_list, partial_flags):
    """每年策略与指数的收益（期末 ÷ 期初 − 1），指数缺期初时为 None。"""
    ykeys = np.array([d[:4] for d in dates])
    out = []
    for y, partial in zip(year_list, partial_flags, strict=True):
        idxs = np.flatnonzero(ykeys == y)
        idxs = idxs[idxs >= 1]
        prev, last = int(idxs[0]) - 1, int(idxs[-1])
        s = nav[last] / nav[prev] - 1
        b = bnav[last] / bnav[prev] - 1 if prev >= bstart else None
        out.append((int(y), s, b, bool(partial)))
    return out


def win_rate(nav, bnav, bstart, dates, year_list, partial_flags, points) -> dict:
    """按年胜率和按月滚动持有期胜率。

    points：月末行号（首月不完整时不含基准日、末月不完整时不含末日），相邻两个相差一个月。
    """
    rows = []
    won = counted = 0
    partial_years = []
    for y, s, b, partial in year_rows(
        nav, bnav, bstart, dates, year_list, partial_flags
    ):
        w = None if b is None else bool(s > b)
        if partial:
            partial_years.append(y)
        elif w is not None:
            counted += 1
            won += w
        rows.append(
            {
                "year": y,
                "strategy": rnd(s),
                "benchmark": rnd(b),
                "excess": rnd(None if b is None else s - b),
                "won": w,
                "partial": partial,
            }
        )
    roll = {}
    pts = [p for p in points if p >= bstart]
    for k in ROLLING_YEARS:
        span = 12 * k
        res = []
        for i in range(len(pts) - span):
            a, b = pts[i], pts[i + span]
            sr, brr = nav[b] / nav[a] - 1, bnav[b] / bnav[a] - 1
            res.append((sr - brr, a, b, sr, brr))
        if not res:
            roll[f"{k}y"] = None
            continue
        worst = min(res, key=lambda x: (x[0], x[1]))
        roll[f"{k}y"] = {
            "windows": len(res),
            "won": int(sum(x[0] > 0 for x in res)),
            "ratio": rnd(sum(x[0] > 0 for x in res) / len(res), 4),
            "worst": {
                "start": dates[worst[1]],
                "end": dates[worst[2]],
                "strategy": rnd(worst[3]),
                "benchmark": rnd(worst[4]),
                "excess": rnd(worst[0]),
            },
        }
    return {
        "years_won": won,
        "years_counted": counted,
        "years_ratio": rnd(won / counted, 4) if counted else None,
        "partial_years": partial_years,
        "yearly": rows,
        "rolling": roll,
    }


def month_points(dates, month_last: dict, first_partial: bool, last_partial: bool):
    pts = [] if first_partial else [0]
    pts += [month_last[mk] for mk in month_last]
    if last_partial:
        pts = pts[:-1]
    return pts


# ====================================================================== 个股贡献
def marks(market, T: int, N: int) -> np.ndarray:
    """引擎的估值价：收盘价；没有收盘价的日子沿用上一个估值价，并按当天的现金分红和送转调整
    （与 `panelbt` 内核的 `last` / `past_close` 相同）。"""
    close = np.asarray(market.close, dtype=float)
    div = getattr(market, "cash_div", None)
    split = getattr(market, "split", None)
    div = None if div is None else np.asarray(div, dtype=float)
    split = None if split is None else np.asarray(split, dtype=float)
    out = np.empty((T, N))
    last = np.full(N, np.nan)
    for t in range(T):
        if t > 0:
            if div is not None:
                last = np.where(last > 0, np.maximum(last - div[t], 0.0), last)
            if split is not None:
                last = last / split[t]
        c = close[t]
        last = np.where(np.isfinite(c), c, last)
        out[t] = last
    return out


def security_flows(result, market, T: int, N: int):
    """逐笔现金流（每个元素对应一个 (日, 证券)）。

    返回 dict：day、col（int 数组）、trade（买入为 −成交额、卖出为 +成交额，含滑点）、
    fees（佣金 + 税 + 其他费用，正数）、dividend（税后分红，可为负）、delist（退市清算所得）。
    """
    days, cols, trade, fees, div, delist = [], [], [], [], [], []
    for tr in result.trades or []:
        days.append(tr.day)
        cols.append(tr.col)
        amt = abs(tr.qty) * tr.price
        trade.append(-amt if tr.qty > 0 else amt)
        fees.append(tr.commission + tr.tax + tr.other_fee)
        div.append(0.0)
        delist.append(0.0)
    for ev in result.delist_events or []:
        days.append(ev.day)
        cols.append(ev.col)
        trade.append(0.0)
        fees.append(0.0)
        div.append(0.0)
        delist.append(ev.proceeds)
    cd = getattr(market, "cash_div", None)
    if cd is not None:
        cd = np.asarray(cd, dtype=float)
        pos = np.asarray(result.positions, dtype=float)
        prev = np.vstack([np.zeros((1, N)), pos[:-1]])
        gross = prev * cd
        tt, cc = np.nonzero(gross)
        g = gross[tt, cc]
        dtax = np.asarray(
            result.dividend_tax if result.dividend_tax is not None else np.zeros(T),
            dtype=float,
        )
        pos_sum = np.zeros(T)
        np.add.at(pos_sum, tt, np.maximum(g, 0.0))
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = np.where(pos_sum > 0, dtax / pos_sum, 0.0)
        net = g - np.where(g > 0, g * rate[tt], 0.0)
        days += tt.tolist()
        cols += cc.tolist()
        trade += [0.0] * len(g)
        fees += [0.0] * len(g)
        div += net.tolist()
        delist += [0.0] * len(g)
    return {
        "day": np.asarray(days, dtype=np.int64),
        "col": np.asarray(cols, dtype=np.int64),
        "trade": np.asarray(trade, dtype=float),
        "fees": np.asarray(fees, dtype=float),
        "dividend": np.asarray(div, dtype=float),
        "delist": np.asarray(delist, dtype=float),
    }


def window_pnl(flows, value_a, value_b, a: int, b: int, N: int) -> dict:
    """区间 (a, b] 上每只证券的盈亏分解（金额）。

    price = 期末市值 − 期初市值 + 买卖成交额 + 退市所得；contribution = price + dividend − fees。
    """
    sel = (flows["day"] > a) & (flows["day"] <= b)
    c = flows["col"][sel]

    def agg(k):
        return np.bincount(c, weights=flows[k][sel], minlength=N)

    price = value_b - value_a + agg("trade") + agg("delist")
    dividend, fees = agg("dividend"), agg("fees")
    return {
        "price": price,
        "dividend": dividend,
        "fees": fees,
        "contribution": price + dividend - fees,
    }


def contributions(
    result, market, pos, equity, dates, symbols, names, windows, interest
) -> dict:
    """windows：[(键, a, b, partial)]，a 是期初行（基准日）、b 是期末行。"""
    T, N = pos.shape
    mk = marks(market, T, N)
    value = np.where(pos != 0, pos * np.nan_to_num(mk), 0.0)
    flows = security_flows(result, market, T, N)
    held = pos > 0
    cum_held = np.vstack([np.zeros((1, N), dtype=np.int64), np.cumsum(held, axis=0)])
    out = []
    for key, a, b, partial in windows:
        p = window_pnl(flows, value[a], value[b], a, b, N)
        days = cum_held[b + 1] - cum_held[a + 1]
        e0 = equity[a]
        touched = np.flatnonzero((days > 0) | (p["contribution"] != 0))
        order = sorted(touched, key=lambda i: (-p["contribution"][i], symbols[i]))

        def item(i, p=p, days=days, e0=e0):
            return {
                "symbol": symbols[i],
                "name": names.get(symbols[i]),
                "contribution": rnd(p["contribution"][i] / e0),
                "pnl": rnd(p["contribution"][i], 2),
                "price": rnd(p["price"][i], 2),
                "dividend": rnd(p["dividend"][i], 2),
                "fees": rnd(p["fees"][i], 2),
                "holding_days": int(days[i]),
            }

        top = order[:TOP_N]
        bottom = [i for i in reversed(order) if i not in set(top)][:TOP_N]
        tot = float(p["contribution"].sum())
        out.append(
            {
                "period": key,
                "start": dates[a],
                "end": dates[b],
                "partial": partial,
                "securities": len(touched),
                "top": [item(i) for i in top],
                "bottom": [item(i) for i in bottom],
                "total_pnl": rnd(tot, 2),
                "total_price": rnd(p["price"].sum(), 2),
                "total_dividend": rnd(p["dividend"].sum(), 2),
                "total_fees": rnd(p["fees"].sum(), 2),
                "cash_interest": rnd(float(interest[a + 1 : b + 1].sum()), 2),
                "equity_change": rnd(equity[b] - e0, 2),
            }
        )
    return {"total": out[0], "yearly": out[1:]}


# ====================================================================== 尾部
def moments(x: np.ndarray):
    """总体偏度 m3 / m2^1.5 与超额峰度 m4 / m2² − 3；样本少于 3 个或方差为 0 时为 None。"""
    if len(x) < 3:
        return None, None
    d = x - x.mean()
    m2 = np.mean(d**2)
    if not m2 > 0:
        return None, None
    return float(np.mean(d**3) / m2**1.5), float(np.mean(d**4) / m2**2 - 3)


def var_cvar(x: np.ndarray, level: float):
    """历史法：VaR 为日收益的 (1 − level) 分位数（取不小于该概率的最小样本值，即经验分布的
    左逆），CVaR 为不高于它的样本均值；都是收益（亏损为负）。"""
    if len(x) == 0:
        return None, None
    s = np.sort(x)
    # 用十进制精确值算 ⌈n·(1 − level)⌉：浮点下 100 × (1 − 0.95) = 5.0000000000000044
    k = math.ceil(len(s) * (1 - Fraction(str(level)))) - 1
    q = s[max(k, 0)]
    return float(q), float(s[s <= q].mean())


def tails(r, br, dates, months, month_r, bmonth) -> dict:
    rr = r[1:]
    sk, ku = moments(rr)
    msk, mku = moments(np.asarray(month_r, dtype=float))
    order = sorted(range(1, len(r)), key=lambda i: (r[i], dates[i]))[:WORST_N]
    worst_days = [
        {
            "date": dates[i],
            "return": rnd(r[i], 6),
            "benchmark_return": None if br is None else rnd(br[i], 6),
        }
        for i in order
    ]
    morder = sorted(range(len(months)), key=lambda i: (month_r[i], months[i]))[:WORST_N]
    worst_months = [
        {
            "month": months[i],
            "return": rnd(month_r[i]),
            "benchmark_return": None if bmonth is None else rnd(bmonth[i]),
        }
        for i in morder
    ]
    var = {}
    for lv in VAR_LEVELS:
        v, c = var_cvar(rr, lv)
        tag = f"{round(lv * 100)}"
        var[f"var_{tag}"] = rnd(v)
        var[f"cvar_{tag}"] = rnd(c)
    best = (0, None, None)
    run, run_start = 0, None
    for i, x in enumerate(month_r):
        if x < 0:
            run_start = i if run == 0 else run_start
            run += 1
            if run > best[0]:
                best = (run, run_start, i)
        else:
            run = 0
    return {
        "daily_skew": rnd(sk, 4),
        "daily_kurtosis": rnd(ku, 4),
        "monthly_skew": rnd(msk, 4),
        "monthly_kurtosis": rnd(mku, 4),
        **var,
        "worst_days": worst_days,
        "worst_months": worst_months,
        "longest_losing_months": best[0],
        "longest_losing_start": None if best[1] is None else months[best[1]],
        "longest_losing_end": None if best[2] is None else months[best[2]],
    }


# ====================================================================== 集中度
def industry_ids(industry, label):
    """把行业矩阵的每格映射成整数编号（标签相同的编号相同）；返回 (编号矩阵, 标签列表)。"""
    ind = np.asarray(industry)
    flat = ind.ravel()
    cache: dict = {}
    labels: list[str] = []
    lab_id: dict[str, int] = {}
    try:
        uniq, inv = np.unique(flat, return_inverse=True)
        keys = list(uniq)
    except TypeError:  # 混合类型的对象数组
        keys, inv_l = [], []
        for x in flat:
            k = (type(x), x) if not isinstance(x, float) or x == x else ("nan",)
            if k not in cache:
                cache[k] = len(keys)
                keys.append(x)
            inv_l.append(cache[k])
        inv = np.asarray(inv_l)
    kid = []
    for k in keys:
        lab = label(k)
        if lab not in lab_id:
            lab_id[lab] = len(labels)
            labels.append(lab)
        kid.append(lab_id[lab])
    return np.asarray(kid, dtype=np.int64)[np.asarray(inv).ravel()].reshape(
        ind.shape
    ), labels


def concentration(pos, mark, long_value, equity, dates, ind_ids, n_labels) -> dict:
    """逐日：前十大持仓权重之和、有效持仓数 1 / Σ(wᵢ / Σw)²、第一大行业权重。

    权重口径与 `holdings_latest`、`industry_weights` 相同：持股数 × 估值价，按引擎的 long_value
    缩放后除以权益。
    """
    T, _ = pos.shape
    top10 = np.full(T, np.nan)
    eff = np.full(T, np.nan)
    topind = np.full(T, np.nan)
    for d in range(T):
        held = np.flatnonzero(pos[d] > 0)
        if len(held) == 0:
            continue
        val = pos[d, held] * mark[d, held]
        val = np.where(np.isfinite(val), val, 0.0)
        total = val.sum()
        lv = long_value[d] if long_value is not None else total
        scale = lv / total if total > 0 else 0.0
        w = val * scale / equity[d]
        s = w.sum()
        if not s > 0:
            continue
        k = min(TOP_N, len(w))
        top10[d] = np.sort(w)[::-1][:k].sum()
        eff[d] = 1.0 / np.sum((w / s) ** 2)
        if ind_ids is not None:
            topind[d] = np.bincount(
                ind_ids[d, held], weights=w, minlength=n_labels
            ).max()

    def stats(x):
        ok = np.flatnonzero(np.isfinite(x))
        if len(ok) == 0:
            return {"mean": None, "max": None, "max_date": None, "latest": None}
        j = ok[np.argmax(x[ok])]
        return {
            "mean": rnd(x[ok].mean()),
            "max": rnd(x[j]),
            "max_date": dates[j],
            "latest": rnd(x[-1]),
        }

    return {
        "days": int(np.isfinite(top10).sum()),
        "top10_weight": stats(top10),
        "effective_holdings": {
            k: (rnd(v, 2) if isinstance(v, float) else v) for k, v in stats(eff).items()
        },
        "top_industry_weight": stats(topind) if ind_ids is not None else None,
        "daily": {
            "top10_weight": [rnd(x, 5) for x in top10],
            "effective_holdings": [rnd(x, 3) for x in eff],
            "top_industry_weight": None
            if ind_ids is None
            else [rnd(x, 5) for x in topind],
        },
    }
