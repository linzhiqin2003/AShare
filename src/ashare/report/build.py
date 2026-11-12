"""从一次 `panelbt` 回测结果算出报告需要的全部序列和指标。

入口是 `build_tearsheet`，输出一个只含 JSON 基本类型的 dict。字段和公式见
`docs/REPORT_NOTES.md`。
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from . import extras as ex
from . import metrics as m
from .metrics import rnd

log = logging.getLogger(__name__)

UNCLASSIFIED = "未分类"
MAX_INDUSTRY_COLUMNS = 260  # 超过就改按月末取样，防止日度调仓把报告撑大


class ReportError(ValueError):
    """输入不满足报告的前提；消息是给使用者看的中文说明。"""


@dataclass(frozen=True)
class Benchmark:
    """基准序列。

    values: 与回测日期逐行对齐的收盘点位或净值（长度 T）；开头可以是 NaN（基准晚于策略起步），
        中间的 NaN 用前值填充并计数，末尾不能缺。
    name: 页面上显示的名称，如「沪深300」。
    kind: "price"（价格指数，不含分红再投资）或 "total_return"（全收益）。必须由调用方声明。
    symbol: 代码，仅作记录。
    """

    values: Sequence[float] | np.ndarray
    name: str
    kind: str
    symbol: str | None = None


KIND_LABEL = {
    "price": "价格指数（不含分红）",
    "total_return": "全收益指数（含分红再投资）",
}


def _iso(d) -> str:
    s = str(d)
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:]}"
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:10]
    raise ReportError(f"日期 {d!r} 无法识别；请用 YYYYMMDD 或 YYYY-MM-DD")


def _arr(result, name):
    v = getattr(result, name, None)
    return None if v is None else np.asarray(v, dtype=float)


def _check_result(result, market, T, N):
    nav = np.asarray(result.nav, dtype=float)
    if nav.ndim != 1:
        raise ReportError("只支持单个策略的回测结果；批量结果请先取出其中一个策略")
    if len(nav) != T:
        raise ReportError(f"日期有 {T} 个，回测结果有 {len(nav)} 行，两者必须逐行对齐")
    if result.positions is None:
        raise ReportError(
            "回测结果里没有持仓矩阵；请用 panelbt.Config(record_trades=True) 重跑"
        )
    pos = np.asarray(result.positions, dtype=float)
    if pos.shape != (T, N):
        raise ReportError(f"持仓矩阵形状 {pos.shape} 与日期 × 证券 ({T}, {N}) 不一致")
    if getattr(market, "futures", None) is not None:
        raise ReportError(
            "回测含期货腿；报告暂不支持期货的保证金和名义敞口口径，无法给出正确的换手、"
            "行业权重和持仓指标"
        )
    if (pos < 0).any() or (
        result.short_value is not None and np.any(np.asarray(result.short_value) != 0)
    ):
        raise ReportError(
            "回测含空头持仓；报告的持仓、行业权重和换手口径只适用于纯多头组合"
        )
    cf = _arr(result, "cash_flow")
    if cf is not None and np.any(cf != 0):
        raise ReportError(
            "回测有资金进出；报告按「权益 ÷ 初始资金」作净值，资金进出会让它失真，暂不支持"
        )
    if not np.all(np.isfinite(nav)) or np.any(nav <= 0):
        raise ReportError("净值出现非正数或非有限值（账户爆仓或输入有误），无法出报告")
    if abs(nav[0] - 1) > 1e-12:
        raise ReportError(
            f"第一行净值是 {nav[0]:.6f}，说明第一行已有成交或费用；报告把第一行当作"
            "基准日（净值 1、尚未持仓），请把回测区间往前多留一个交易日"
        )
    return nav, pos


def _benchmark(bench: Benchmark | None, T: int):
    if bench is None:
        return None, None, 0
    if not isinstance(bench, Benchmark):
        raise ReportError("benchmark 需要是 ashare.report.Benchmark")
    if bench.kind not in KIND_LABEL:
        raise ReportError(
            "Benchmark.kind 只能是 'price'（价格指数）或 'total_return'（全收益）"
        )
    v = np.asarray(bench.values, dtype=float).copy()
    if v.shape != (T,):
        raise ReportError(f"基准序列长度 {v.shape} 与回测日期 {T} 不一致")
    ok = np.isfinite(v) & (v > 0)
    if not ok.any():
        raise ReportError("基准序列全部缺失")
    first = int(np.argmax(ok))
    last_ok = T - 1 - int(np.argmax(ok[::-1]))
    if last_ok != T - 1:
        raise ReportError("基准序列在区间末尾缺失；请补齐或截短回测区间")
    filled = 0
    for i in range(first + 1, T):
        if not ok[i]:
            v[i] = v[i - 1]
            filled += 1
    if filled:
        log.warning("基准序列中间缺 %d 天，已用前值填充", filled)
    v[:first] = np.nan
    return v / v[first], first, filled


def _open_day_between(end: str, until: str, cal: list[str] | None) -> bool:
    """(end, until) 之间是否还有交易日。日历覆盖的部分查日历，覆盖不到的部分按工作日算。"""
    lo = np.datetime64(end, "D") + np.timedelta64(1, "D")
    hi = np.datetime64(until, "D")
    if cal:
        cmax = np.datetime64(cal[-1], "D")
        if cmax >= lo:
            if any(end < c < until for c in cal):
                return True
            lo = cmax + np.timedelta64(1, "D")
    if lo >= hi:
        return False
    return int(np.busday_count(lo, hi)) > 0


def _label_fn(industry, industry_names):
    names = dict(industry_names or {})

    def label(x):
        if x is None:
            return UNCLASSIFIED
        if isinstance(x, (float, np.floating)):
            if not np.isfinite(x) or x == 0:
                return UNCLASSIFIED
            if float(x).is_integer():
                x = int(x)
        if isinstance(x, (np.integer,)):
            x = int(x)
        if x == 0 or x == "":
            return UNCLASSIFIED
        return str(names.get(x, x))

    return label


def _costs_meta(costs, config):
    if costs is None and config is None:
        return None
    out = {}
    if costs is not None:

        def scalar(name):
            a = np.asarray(getattr(costs, name), dtype=float)
            if a.size == 0:
                return None
            lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
            return lo if lo == hi else [lo, hi]

        out["commission_rate"] = float(costs.commission_rate)
        out["commission_min"] = (
            None if costs.commission_min is None else float(costs.commission_min)
        )
        out["stamp_tax_sell"] = scalar("tax_sell")
        out["slippage"] = scalar("slippage")
        fb, fs = scalar("fee_rate_buy"), scalar("fee_rate_sell")
        out["fee_rate_buy"], out["fee_rate_sell"] = fb, fs
    if config is not None:
        a = np.asarray(config.dividend_tax_rate, dtype=float)
        out["dividend_tax"] = float(a.max()) if a.size else 0.0
    return out


def build_tearsheet(
    result,
    market,
    targets,
    dates: Sequence,
    symbols: Sequence[str],
    *,
    names: Mapping[str, str] | None = None,
    industry: np.ndarray | None = None,
    industry_names: Mapping | None = None,
    industry_switch: str | None = None,
    industry_switch_labels: tuple[str, str] | None = None,
    benchmark: Benchmark | None = None,
    suspended: np.ndarray | None = None,
    title: str = "回测报告",
    description: str | None = None,
    rules: str | None = None,
    costs=None,
    config=None,
    initial_capital: float | None = None,
    periods_per_year: int = 252,
    rolling_window: int | None = None,
    top_drawdowns: int = 5,
    calendar: Sequence | None = None,
    benchmarks: Sequence[Benchmark] | None = None,
    extras: bool = True,
) -> dict:
    """把一次回测整理成报告数据。

    result: `panelbt.run` 的结果，必须用 `Config(record_trades=True)` 跑（需要持仓矩阵）。
    market / targets: 回测用的 `panelbt.Market` 与 `panelbt.Targets`（取收盘价、可买卖掩码和调仓日）。
    dates / symbols: 行、列标签，长度 T / N。第一行是基准日：净值 1、尚未持仓。
    names: 证券代码到名称。industry: T×N 行业码或行业名（0、空串、NaN 视为未分类），
        industry_names 把码映射成名称。industry_switch: 行业分类换版的日期（显式传入；不传就不画），
        industry_switch_labels: 换版前后两套分类的名称，如 ("申万2014", "申万2021")。
    benchmark: `Benchmark`，可省略。suspended: T×N 停牌掩码，用来区分未成交原因；
        不给时以执行价缺失判为停牌。
    costs / config: 回测用的 `panelbt.Costs` / `Config`，只用来在页面上写明费用设定。
    calendar: 交易日历（全部开市日，可含区间末日之后的未来日期），用来判断最后一个月、最后一年
        是否完整；不给或覆盖不到时按「之后同月 / 同年是否还有工作日」判断。
    benchmarks: 主基准之外的对比指数（`Benchmark` 列表），输出每条的指标和策略相对它的超额、胜率。
    extras: 是否计算补充内容（对比指数、胜率、个股贡献、尾部、集中度，见 `extras.py`）；
        False 时输出与加入这些内容之前完全相同。
    """
    P = int(periods_per_year)
    window = int(rolling_window or P)
    ddates = [_iso(d) for d in dates]
    T, N = len(ddates), len(symbols)
    if T < 2:
        raise ReportError("至少需要基准日加一个交易日")
    if any(a >= b for a, b in zip(ddates, ddates[1:], strict=False)):
        raise ReportError("日期必须严格递增")
    nav_raw, pos = _check_result(result, market, T, N)
    if tuple(market.shape) != (T, N):
        raise ReportError(f"行情形状 {market.shape} 与日期 × 证券 ({T}, {N}) 不一致")
    if getattr(targets, "kind", "weights") not in ("weights", "shares"):
        raise ReportError("targets 类型无法识别")
    tw = np.asarray(targets.weights, dtype=float)
    if (tw < 0).any():
        raise ReportError("目标里有负权重（做空）；报告只适用于纯多头组合")
    symbols = [str(s) for s in symbols]
    names = dict(names or {})

    nav = nav_raw / nav_raw[0]
    equity = _arr(result, "equity")
    if initial_capital is None:
        initial_capital = float(equity[0] / nav_raw[0]) if equity is not None else None
    if equity is None:
        equity = nav_raw * initial_capital
    r = _arr(result, "returns")
    if r is None:
        r = np.r_[0.0, nav[1:] / nav[:-1] - 1]
    r = r.copy()
    r[0] = 0.0
    bnav, bstart, bfilled = _benchmark(benchmark, T)
    if bnav is not None:
        br = np.full(T, np.nan)
        br[bstart] = 0.0
        br[bstart + 1 :] = bnav[bstart + 1 :] / bnav[bstart:-1] - 1
        if bstart == 0:
            br[0] = 0.0
    else:
        br = None

    draw = m.drawdown(nav)
    n_hold = (pos > 0).sum(axis=1)
    cash = _arr(result, "cash")
    cash_w = cash / equity
    turnover = 2 * np.asarray(result.turnover, dtype=float)  # 引擎是单边，这里用双边

    # ------------------------------------------------------------ 日度
    daily = {
        "date": ddates,
        "nav": [rnd(x, 6) for x in nav],
        "benchmark_nav": None if bnav is None else [rnd(x, 6) for x in bnav],
        "return": [rnd(x, 7) for x in r],
        "benchmark_return": None if br is None else [rnd(x, 7) for x in br],
        "drawdown": [rnd(x, 6) for x in draw],
        "benchmark_drawdown": None,
        "holdings": [int(x) for x in n_hold],
        "cash_weight": [rnd(x, 6) for x in cash_w],
        "turnover": [rnd(x, 6) for x in turnover],
    }
    if bnav is not None:
        bd = np.full(T, np.nan)
        bd[bstart:] = m.drawdown(bnav[bstart:])
        daily["benchmark_drawdown"] = [rnd(x, 6) for x in bd]

    # ------------------------------------------------------------ 汇总
    rr = r[1:]
    years_span = len(rr) / P
    eps = m.episodes(nav, ddates)
    eps_sorted = sorted(eps, key=lambda e: (e["depth"], e["peak"]))[:top_drawdowns]
    mdd = eps_sorted[0] if eps_sorted else None

    mkeys = np.array([d[:7] for d in ddates])
    month_list = list(dict.fromkeys(mkeys[1:].tolist()))
    month_last = {mk: int(np.flatnonzero(mkeys == mk)[-1]) for mk in month_list}
    month_first = {mk: int(np.flatnonzero(mkeys == mk)[0]) for mk in month_list}
    monthly = {
        "month": [],
        "return": [],
        "benchmark_return": None if bnav is None else [],
    }
    month_r, bmonth_r = [], []
    for mk in month_list:
        last, prev = month_last[mk], month_first[mk] - 1
        if mk == month_list[0]:
            prev = 0
        mr = nav[last] / nav[prev] - 1
        monthly["month"].append(mk)
        monthly["return"].append(rnd(mr, 6))
        month_r.append(mr)
        if bnav is not None:
            if prev >= bstart:
                bmr = bnav[last] / bnav[prev] - 1
                monthly["benchmark_return"].append(rnd(bmr, 6))
                bmonth_r.append((mr, bmr))
            else:
                monthly["benchmark_return"].append(None)
    month_r = np.array(month_r)
    cal = sorted({_iso(c) for c in calendar}) if calendar is not None else None
    end = ddates[-1]
    nm_y, nm_m = (
        (int(end[:4]) + 1, 1) if end[5:7] == "12" else (int(end[:4]), int(end[5:7]) + 1)
    )
    month_open = _open_day_between(end, f"{nm_y:04d}-{nm_m:02d}-01", cal)
    year_open = _open_day_between(end, f"{int(end[:4]) + 1:04d}-01-01", cal)
    # 第一行是基准日（前一个交易日）：它和首个交易日同月 / 同年，说明首月 / 首年不是从头开始
    monthly["first_partial"] = ddates[0][:7] == ddates[1][:7]
    monthly["last_partial"] = month_open

    ykeys = np.array([d[:4] for d in ddates])
    yearly = []
    year_list = list(dict.fromkeys(ykeys[1:].tolist()))
    for y in year_list:
        idxs = np.flatnonzero(ykeys == y)
        idxs = idxs[idxs >= 1]
        prev = idxs[0] - 1
        sr = r[idxs]
        row = {
            "year": int(y),
            "days": int(len(idxs)),
            "return": rnd(nav[idxs[-1]] / nav[prev] - 1),
            "volatility": rnd(m.volatility(sr, P)),
            "max_drawdown": rnd(m.max_drawdown_of_returns(sr)),
            "sharpe": rnd(m.sharpe(sr, P), 4),
            "benchmark_return": None,
            "benchmark_volatility": None,
            "benchmark_max_drawdown": None,
            "benchmark_sharpe": None,
            "partial": bool(
                (y == year_list[0] and ddates[0][:4] == y)
                or (y == year_list[-1] and year_open)
            ),
        }
        if bnav is not None and prev >= bstart:
            sb = br[idxs]
            row["benchmark_return"] = rnd(bnav[idxs[-1]] / bnav[prev] - 1)
            row["benchmark_volatility"] = rnd(m.volatility(sb, P))
            row["benchmark_max_drawdown"] = rnd(m.max_drawdown_of_returns(sb))
            row["benchmark_sharpe"] = rnd(m.sharpe(sb, P), 4)
        yearly.append(row)

    def arr0(name):
        a = _arr(result, name)
        return np.zeros(T) if a is None else a

    commission, tax, other = arr0("commission"), arr0("tax"), arr0("other_fee")
    slip = arr0("slippage_cost") + arr0("impact_cost")
    div_g, div_t = arr0("dividend_gross"), arr0("dividend_tax")
    fees_day = commission + tax + other
    fees_total = float(fees_day.sum())
    ann = m.annual_return(rr, P)
    summary = {
        "start": ddates[0],
        "first_trade_day": ddates[1],
        "end": ddates[-1],
        "trading_days": int(len(rr)),
        "years": rnd(years_span, 4),
        "final_nav": rnd(nav[-1]),
        "cumulative_return": rnd(nav[-1] - 1),
        "annual_return": rnd(ann),
        "annual_volatility": rnd(m.volatility(rr, P)),
        "sharpe": rnd(m.sharpe(rr, P), 4),
        "sharpe_rf2": rnd(m.sharpe(rr, P, 0.02), 4),
        "sortino": rnd(m.sortino(rr, P), 4),
        "sortino_rf2": rnd(m.sortino(rr, P, 0.02), 4),
        "max_drawdown": mdd["depth"] if mdd else 0.0,
        "max_drawdown_peak": mdd["peak"] if mdd else None,
        "max_drawdown_trough": mdd["trough"] if mdd else None,
        "max_drawdown_recovery": mdd["recovery"] if mdd else None,
        "calmar": rnd(ann / -mdd["depth"], 4) if mdd and mdd["depth"] < 0 else None,
        "monthly_win_rate": rnd((month_r > 0).mean(), 4) if len(month_r) else None,
        "months": int(len(month_r)),
        "annual_turnover": rnd(turnover[1:].sum() / years_span, 4),
        "annual_turnover_one_sided": rnd(turnover[1:].sum() / 2 / years_span, 4),
        "total_fees": rnd(fees_total, 2),
        "total_fees_to_initial": rnd(fees_total / initial_capital),
        "total_slippage": rnd(slip.sum(), 2),
        "total_slippage_to_initial": rnd(slip.sum() / initial_capital),
        "total_commission": rnd(commission.sum(), 2),
        "total_stamp_tax": rnd(tax.sum(), 2),
        "total_dividend_gross": rnd(div_g.sum(), 2),
        "total_dividend_tax": rnd(div_t.sum(), 2),
        "rebalances": int(len(np.asarray(targets.days))),
        "benchmark_start": None,
    }
    bkeys = (
        "benchmark_cumulative_return",
        "benchmark_annual_return",
        "benchmark_annual_volatility",
        "benchmark_sharpe",
        "benchmark_max_drawdown",
        "excess_annual_return",
        "excess_annual_return_geometric",
        "tracking_error",
        "information_ratio",
        "beta",
        "monthly_win_rate_vs_benchmark",
    )
    for k in bkeys:
        summary[k] = None
    if bnav is not None:
        o = max(bstart, 0)
        ro, bo = r[o + 1 :], br[o + 1 :]
        if len(bo) >= 2:
            sa, ba = m.annual_return(ro, P), m.annual_return(bo, P)
            active = ro - bo
            te = m.volatility(active, P)
            summary.update(
                benchmark_start=ddates[o],
                benchmark_cumulative_return=rnd(bnav[-1] - 1),
                benchmark_annual_return=rnd(ba),
                benchmark_annual_volatility=rnd(m.volatility(bo, P)),
                benchmark_sharpe=rnd(m.sharpe(bo, P), 4),
                benchmark_max_drawdown=rnd(m.max_drawdown_of_returns(bo)),
                excess_annual_return=rnd(sa - ba),
                excess_annual_return_geometric=rnd((1 + sa) / (1 + ba) - 1),
                tracking_error=rnd(te),
                information_ratio=rnd(active.mean() * P / te, 4) if te else None,
                beta=rnd(m.beta(ro, bo), 4),
                monthly_win_rate_vs_benchmark=rnd(
                    np.mean([a > b for a, b in bmonth_r]), 4
                )
                if bmonth_r
                else None,
            )

    # ------------------------------------------------------------ 滚动
    rb = None if br is None else np.where(np.isfinite(br), br, np.nan)
    ro_ = m.rolling(r, rb, window, P)
    rolling = {
        "window": window,
        "date": [ddates[i] for i in ro_.pop("date_index")],
        **ro_,
    }
    if bnav is None:
        rolling["beta"] = None

    # ------------------------------------------------------------ 调仓
    label = _label_fn(industry, industry_names) if industry is not None else None
    if industry is not None:
        industry = np.asarray(industry)
        if industry.shape != (T, N):
            raise ReportError(f"行业矩阵形状 {industry.shape} 与 ({T}, {N}) 不一致")
    can_buy = np.asarray(market.can_buy)
    can_sell = np.asarray(market.can_sell)
    exec_px = np.asarray(market.exec_price)
    close = np.asarray(market.close, dtype=float)
    if suspended is not None:
        suspended = np.asarray(suspended, dtype=bool)
        if suspended.shape != (T, N):
            raise ReportError("停牌掩码形状与行情不一致")
    mark = close.copy()
    for i in range(1, T):  # 停牌、无行情日沿用上一个收盘价估值（仅用于行业权重分摊）
        bad = ~np.isfinite(mark[i])
        if bad.any():
            mark[i, bad] = mark[i - 1, bad]
    long_value = _arr(result, "long_value")
    blocked_buy = np.asarray(result.n_blocked_buy, dtype=np.int64)
    blocked_sell = np.asarray(result.n_blocked_sell, dtype=np.int64)
    trade_cols_by_day: dict[int, set[int]] = {}
    buy_cols: dict[int, set[int]] = {}
    sell_cols: dict[int, set[int]] = {}
    for tr in result.trades or []:
        trade_cols_by_day.setdefault(tr.day, set()).add(tr.col)
        (buy_cols if tr.qty > 0 else sell_cols).setdefault(tr.day, set()).add(tr.col)

    def reason(day, col, side):
        if (suspended is not None and suspended[day, col]) or not np.isfinite(
            exec_px[day, col]
        ):
            return "suspended"
        return "limit_up" if side == "buy" else "limit_down"

    tdays = np.asarray(targets.days, dtype=np.int64)
    tptr = np.asarray(targets.indptr, dtype=np.int64)
    tcols = np.asarray(targets.cols, dtype=np.int64)
    rebalances = []
    for k, d in enumerate(tdays):
        d = int(d)
        if d <= 0 or d >= T:
            continue
        nxt = int(tdays[k + 1]) if k + 1 < len(tdays) else T
        cols = tcols[tptr[k] : tptr[k + 1]]
        wts = tw[tptr[k] : tptr[k + 1]]
        tgt = set(cols[wts > 0].tolist())
        held_before = set(np.flatnonzero(pos[d - 1] > 0).tolist())
        new, dropped = tgt - held_before, held_before - tgt
        unfilled = []
        for c in sorted(new):
            if not can_buy[d, c]:
                unfilled.append(
                    {"symbol": symbols[c], "side": "buy", "reason": reason(d, c, "buy")}
                )
        for c in sorted(dropped):
            if not can_sell[d, c]:
                unfilled.append(
                    {
                        "symbol": symbols[c],
                        "side": "sell",
                        "reason": reason(d, c, "sell"),
                    }
                )
        later = set()
        for dd in range(d + 1, nxt):
            later |= trade_cols_by_day.get(dd, set())
        idx = {s: i for i, s in enumerate(symbols)} if unfilled else {}
        counts = {"limit_up": 0, "limit_down": 0, "suspended": 0}
        for u in unfilled:
            counts[u["reason"]] += 1
        rebalances.append(
            {
                "date": ddates[d],
                "targets": len(tgt),
                "buys": len(buy_cols.get(d, ())),
                "sells": len(sell_cols.get(d, ())),
                "new_names": len(new),
                "dropped_names": len(dropped),
                "turnover": rnd(turnover[d], 6),
                "window_turnover": rnd(turnover[d:nxt].sum(), 6),
                "fees": rnd(fees_day[d:nxt].sum(), 2),
                "slippage": rnd(slip[d:nxt].sum(), 2),
                "unfilled": len(unfilled),
                "unfilled_by_reason": counts,
                "unfilled_detail": unfilled,
                "unfilled_filled_before_next_rebalance": sum(
                    1 for u in unfilled if idx[u["symbol"]] in later
                ),
                "blocked_buy_days": int(blocked_buy[d:nxt].sum()),
                "blocked_sell_days": int(blocked_sell[d:nxt].sum()),
                "holdings_after": int(n_hold[d]),
                "cash_weight_after": rnd(cash_w[d], 6),
            }
        )

    # ------------------------------------------------------------ 行业权重
    def weights_at(d):
        held = np.flatnonzero(pos[d] > 0)
        if len(held) == 0:
            return held, np.zeros(0)
        val = pos[d, held] * mark[d, held]
        val = np.where(np.isfinite(val), val, 0.0)
        total = val.sum()
        lv = long_value[d] if long_value is not None else total
        scale = lv / total if total > 0 else 0.0
        return held, val * scale / equity[d]

    industry_weights = None
    if label is not None:
        sample = [int(x) for x in tdays if 0 < x < T]
        sampling = "rebalance"
        if len(sample) > MAX_INDUSTRY_COLUMNS:
            sample = [month_last[mk] for mk in month_list]
            sampling = "month_end"
        rows_ = []
        for d in sample:
            held, w = weights_at(d)
            agg: dict[str, float] = {}
            for c, wi in zip(held, w, strict=True):
                lab = label(industry[d, c])
                agg[lab] = agg.get(lab, 0.0) + float(wi)
            rows_.append((d, agg, 1.0 - float(w.sum())))
        inds = sorted({k for _, a, _ in rows_ for k in a})
        industry_weights = {
            "sampling": sampling,
            "industries": inds,
            "dates": [ddates[d] for d, _, _ in rows_],
            "weights": [[rnd(a.get(n, 0.0), 6) for _, a, _ in rows_] for n in inds],
            "cash": [rnd(c, 6) for _, _, c in rows_],
        }

    # ------------------------------------------------------------ 最新持仓
    last = T - 1
    held, w = weights_at(last)
    holdings_latest = sorted(
        (
            {
                "symbol": symbols[c],
                "name": names.get(symbols[c]),
                "weight": rnd(wi, 6),
                "shares": float(pos[last, c]),
                "industry": label(industry[last, c]) if label is not None else None,
            }
            for c, wi in zip(held, w, strict=True)
        ),
        key=lambda x: (-x["weight"], x["symbol"]),
    )

    # ------------------------------------------------------------ 年度归因
    div_net = div_g - div_t
    dE = np.r_[0.0, np.diff(equity)]
    price_pnl = dE - div_net + fees_day + slip
    attribution = []
    for y in year_list:
        idxs = np.flatnonzero(ykeys == y)
        idxs = idxs[idxs >= 1]
        e0 = equity[idxs[0] - 1]
        attribution.append(
            {
                "year": int(y),
                "total": rnd((equity[idxs[-1]] - e0) / e0),
                "price": rnd(price_pnl[idxs].sum() / e0),
                "dividend_net": rnd(div_net[idxs].sum() / e0),
                "dividend_tax": rnd(-div_t[idxs].sum() / e0),
                "fees": rnd(-fees_day[idxs].sum() / e0),
                "slippage": rnd(-slip[idxs].sum() / e0),
            }
        )

    meta = {
        "title": title,
        "description": description,
        "rules": rules,
        "start": ddates[0],
        "end": ddates[-1],
        "initial_capital": rnd(initial_capital, 2),
        "periods_per_year": P,
        "costs": _costs_meta(costs, config),
        "benchmark": None
        if benchmark is None
        else {
            "symbol": benchmark.symbol,
            "name": benchmark.name,
            "kind": benchmark.kind,
            "kind_label": KIND_LABEL[benchmark.kind],
            "filled_days": int(bfilled),
        },
        "industry_switch": _iso(industry_switch) if industry_switch else None,
        "industry_switch_labels": list(industry_switch_labels)
        if industry_switch and industry_switch_labels
        else None,
        "has_names": bool(names),
        # 判断末月 / 末年是否完整的依据：交易日历覆盖到年底 / 日历加工作日 / 只按工作日
        "completeness_basis": "calendar"
        if cal and cal[-1] >= end[:4] + "-12-31"
        else ("calendar+weekday" if cal else "weekday"),
    }
    out = {
        "meta": meta,
        "summary": summary,
        "daily": daily,
        "monthly": monthly,
        "yearly": yearly,
        "drawdowns": eps_sorted,
        "rolling": rolling,
        "rebalances": rebalances,
        "holdings_latest": holdings_latest,
        "industry_weights": industry_weights,
        "attribution": attribution,
    }
    if not extras:
        if benchmarks:
            raise ReportError("对比指数 benchmarks 需要 extras=True")
        return out

    # ------------------------------------------------------------ 补充内容
    extra_b = []
    for k, eb in enumerate(benchmarks or ()):
        if not isinstance(eb, Benchmark):
            raise ReportError("benchmarks 需要是 ashare.report.Benchmark 的列表")
        v, st, fd = _benchmark(eb, T)
        if st >= T - 1:
            raise ReportError(f"对比指数 {eb.name} 只有最后一天有数据")
        r_ = np.full(T, np.nan)
        r_[st] = 0.0
        r_[st + 1 :] = v[st + 1 :] / v[st:-1] - 1
        extra_b.append((k, eb, v, st, fd, r_))
    year_partial = [row["partial"] for row in yearly]
    pts = ex.month_points(
        ddates, month_last, monthly["first_partial"], monthly["last_partial"]
    )
    out["benchmarks"] = [
        {
            "name": eb.name,
            "symbol": eb.symbol,
            "kind": eb.kind,
            "kind_label": KIND_LABEL[eb.kind],
            "filled_days": int(fd),
            **ex.bench_stats(r, r_, v, st, ddates, P),
            "yearly": [
                {"year": y, "return": rnd(b)}
                for y, _, b, _ in ex.year_rows(
                    nav, v, st, ddates, year_list, year_partial
                )
            ],
        }
        for _, eb, v, st, fd, r_ in extra_b
    ]
    wins = []
    if bnav is not None:
        wins.append(
            {
                "name": benchmark.name,
                "symbol": benchmark.symbol,
                "main": True,
                **ex.win_rate(nav, bnav, bstart, ddates, year_list, year_partial, pts),
            }
        )
    for _, eb, v, st, _, _ in extra_b:
        wins.append(
            {
                "name": eb.name,
                "symbol": eb.symbol,
                "main": False,
                **ex.win_rate(nav, v, st, ddates, year_list, year_partial, pts),
            }
        )
    out["win_rates"] = wins or None

    windows = [("all", 0, T - 1, False)]
    for y, part in zip(year_list, year_partial, strict=True):
        idxs = np.flatnonzero(ykeys == y)
        idxs = idxs[idxs >= 1]
        windows.append((y, int(idxs[0]) - 1, int(idxs[-1]), part))
    out["contributions"] = ex.contributions(
        result,
        market,
        pos,
        equity,
        ddates,
        symbols,
        names,
        windows,
        arr0("cash_interest"),
    )
    bm_all = None
    if bnav is not None:
        bm_all = []
        for mk in month_list:
            last_, prev_ = month_last[mk], month_first[mk] - 1
            if mk == month_list[0]:
                prev_ = 0
            bm_all.append(bnav[last_] / bnav[prev_] - 1 if prev_ >= bstart else None)
    out["tails"] = ex.tails(r, br, ddates, month_list, month_r, bm_all)
    ids, labs = (None, [])
    if label is not None:
        ids, labs = ex.industry_ids(industry, label)
    out["concentration"] = ex.concentration(
        pos, mark, long_value, equity, ddates, ids, len(labs)
    )
    return out
