"""两个真实数据上的完整例子：事件研究 + ``panelbt`` 带完整 A 股规则回测。

运行（只读数据，不联网）::

    python -m ashare.events.examples --root /path/to/data --out report.json

参数在下面的 ``PARAMS`` 里事先定好，结果写进 ``docs/EVENTS_NOTES.md``；不按结果调参。
"""

from __future__ import annotations

import argparse
import json
import time
import warnings

import numpy as np
import polars as pl

from . import extract
from .market import load
from .study import event_study, group_benchmark, size_industry_groups
from .table import align
from .targets import to_targets

PARAMS = {
    "start": "2010-01-04",
    "window": (-20, 60),
    "tests": [(-20, -1), (0, 0), (1, 5), (1, 20), (1, 60), (0, 20)],
    "size_buckets": 5,
    "bootstrap_reps": 1000,
    "backtest": {
        "entry_lag": 0,
        "holding": 20,
        "overlap": "extend",
        "defer": 0,
        "weight": "equal",
        "max_positions": 50,
        "max_weight": 0.05,
    },
    # 名额不够时的先后：例一按预告同比下限，例二按快报同比（从大到小）；再相同按哈希
    "priority": {"forecast_strong": "change_low", "express_growth": "yoy"},
    # 佣金、滑点由调用方给；印花税、过户费用 panelbt 的 A 股历史费率档案
    "costs": {"commission_rate": 2.5e-4, "commission_min": 5.0, "slippage": 5e-4},
}

CURVE_AT = (-20, -10, -5, -1, 0, 1, 2, 5, 10, 20, 40, 60)


def strong_forecasts(ds):
    """例一：首次业绩预告、类型「预增」、净利润同比变动下限 ≥ 100%。"""
    ev = extract.forecasts(ds)
    return ev.filter(
        (pl.col("forecast_type") == "预增")
        & ~pl.col("is_revision")
        & (pl.col("change_low") >= 100)
    )


def strong_expresses(ds):
    """例二：业绩快报、上年同期归母净利润为正、净利润同比 ≥ 50%。"""
    ev = extract.expresses(ds)
    return ev.filter((pl.col("prior_net_income") > 0) & (pl.col("yoy") >= 0.5))


def _tables(study, mask=None):
    """每个区间一行：按事件平均的 CAR 与朴素/按月聚类/自助法，加日历时间组合的 NW t
    （按日等权与按事件数加权两种）。"""
    rows = []
    for a, b in PARAMS["tests"]:
        row = study.test(a, b, mask)
        ct = study.calendar_time(a, b, mask).select(
            "t_nw", "p_nw", "implied_car", pl.col("days").alias("ct_days")
        )
        ce = study.calendar_time(a, b, mask, weight="events").select(
            pl.col("t_nw").alias("t_nw_events"),
            pl.col("implied_car").alias("implied_car_events"),
        )
        rows.append(pl.concat([row, ct, ce], how="horizontal"))
    return pl.concat(rows, how="diagonal_relaxed")


def _size_pct(m):
    """每日按上一交易日总市值的横截面分位（0 最小、1 最大），T×N。"""
    cap = np.full(m["total_market_cap"].shape, np.nan)
    cap[1:] = m["total_market_cap"][:-1]
    cap[~(np.isfinite(cap) & (cap > 0))] = np.nan
    order = np.argsort(np.where(np.isnan(cap), np.inf, cap), axis=1)
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(cap.shape[1])[None, :], axis=1)
    count = np.isfinite(cap).sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        pct = rank / np.maximum(count - 1, 1)
    return np.where(np.isfinite(cap), pct, np.nan)


def _size_profile(study, pct, plan, m):
    """事件和持仓的市值分位，以及按市值五分位分组的 CAR[1,20]（同行业同市值基准）。"""
    ev = study.events
    p = pct[study.avail, ev["col"].to_numpy()]
    q = np.floor(np.clip(p, 0, 0.999999) * 5)
    by_q = []
    for k in range(5):
        mask = q == k
        if mask.sum() >= 2:
            r = study.test(1, 20, mask).row(0, named=True)
            ct = study.calendar_time(1, 20, mask).row(0, named=True)
            by_q.append(
                {
                    "quintile": k + 1,
                    "events": int(mask.sum()),
                    "car_1_20": r["mean"],
                    "t_month": r.get("t_cluster"),
                    "t_nw": ct.get("t_nw"),
                }
            )
    tg = plan.targets
    rows = np.repeat(np.arange(len(tg.days)), np.diff(tg.indptr))
    span = (
        np.append(np.diff(tg.days), len(m.dates) - tg.days[-1]) if len(tg.days) else []
    )
    hold_pct = pct[tg.days[rows], tg.cols]
    w = tg.weights * np.asarray(span)[rows]
    ok = np.isfinite(hold_pct) & (w > 0)
    return {
        "event_size_pct_mean": float(np.nanmean(p)),
        "holding_size_pct_weighted_mean": float(
            np.average(hold_pct[ok], weights=w[ok])
        ),
        "car_1_20_by_size_quintile": by_q,
    }


def _open_rate_by_board(plan):
    prefix = plan.events["symbol"].str.slice(0, 3)
    return (
        plan.events.with_columns(prefix.alias("prefix"))
        .group_by("prefix")
        .agg(
            pl.len().alias("events"),
            (pl.col("status").is_in(["opened", "deferred"]))
            .mean()
            .alias("opened_share"),
            (pl.col("status") == "capacity").mean().alias("capacity_share"),
        )
        .filter(pl.col("events") >= 50)
        .sort("prefix")
        .to_dicts()
    )


def _curve(study, mask=None):
    c = study.caar(mask)
    return c.filter(pl.col("offset").is_in(list(CURVE_AT)))


def _open_entry_car(m, study, bench_full, horizon=20):
    """开盘买入的可实现超额：第 0 天只算开盘到收盘，之后逐日收盘到收盘。"""
    ev = study.events
    avail = ev["avail"].to_numpy()
    col = ev["col"].to_numpy()
    o = m["open"][avail, col]
    c = m["close"][avail, col]
    intraday = np.where((o > 0) & np.isfinite(c), c / o - 1, np.nan)
    b0 = bench_full[avail, col] if bench_full.ndim == 2 else bench_full[avail]
    # 基准第 0 天按全天收益扣，偏保守（开盘前的指数跳空也被扣掉一部分）
    day0 = np.nan_to_num(intraday - b0)
    return day0 + study.car(1, horizon)


def run_example(key, name, ev, m, groups_bench, pct):
    out = {"name": name, "events_raw": ev.height}
    aligned = align(ev, m.dates, m.symbols)
    out["dropped"] = aligned.dropped
    out["by_year"] = aligned.by_year().rows()
    p = PARAMS
    studies = {}
    for label, bench in (("market", m.market_return), ("industry_size", groups_bench)):
        t0 = time.perf_counter()
        st = event_study(
            aligned,
            m.returns,
            bench,
            m.dates,
            m.symbols,
            window=p["window"],
            can_buy=m["can_buy_open"],
            reps=p["bootstrap_reps"],
        )
        res = {
            "seconds": time.perf_counter() - t0,
            "events": len(st),
            "dropped": st.dropped,
        }
        res["tests"] = _tables(st).to_dicts()
        res["curve"] = _curve(st).to_dicts()
        buy = st.events["entry_buyable"].to_numpy()
        res["buyable_share"] = float(buy.mean())
        res["split_1_20"] = st.split(buy, 1, 20, ("buyable", "not_buyable")).to_dicts()
        res["split_0_0"] = st.split(buy, 0, 0, ("buyable", "not_buyable")).to_dicts()
        res["split_m20_m1"] = st.split(
            buy, -20, -1, ("buyable", "not_buyable")
        ).to_dicts()
        res["curve_buyable"] = _curve(st, buy).to_dicts()
        near = st.caar().filter(pl.col("offset").is_between(-3, 1))
        res["aar_near_zero"] = near.select("offset", "aar").to_dicts()
        oc = _open_entry_car(m, st, bench if np.ndim(bench) == 2 else np.asarray(bench))
        res["open_entry_car_0_20"] = {
            "all": float(np.mean(oc)),
            "buyable": float(np.mean(oc[buy])),
            "close_to_close_all": float(st.car(0, 20).mean()),
        }
        res["date_cluster_1_20"] = st.test(1, 20, cluster="date").to_dicts()
        # 敏感性（事后补做，不改主口径）：NW 滞后加大；缺失记 0 的宽松口径
        res["lag_sensitivity"] = [
            {"window": f"[{a},{b}]", "lags": lag, "weight": w,
             "t_nw": st.calendar_time(a, b, lags=lag, weight=w)["t_nw"][0]}
            for a, b in ((1, 20), (1, 60))
            for lag in (None, 20, 60)
            for w in ("days", "events")
        ]  # fmt: skip
        loose = event_study(
            aligned,
            m.returns,
            bench,
            m.dates,
            m.symbols,
            window=p["window"],
            reps=p["bootstrap_reps"],
            complete=False,
        )
        res["complete_false"] = {
            "events": len(loose),
            "tests": _tables(loose).to_dicts(),
        }
        out[label] = res
        studies[label] = st
    plan = to_targets(
        aligned,
        m.dates,
        m.symbols,
        m["can_buy_open"],
        can_sell=m["can_sell_open"],
        priority=PARAMS["priority"][key],
        **PARAMS["backtest"],
    )
    out["backtest"] = backtest(plan, m)
    out["size"] = _size_profile(studies["industry_size"], pct, plan, m)
    out["open_rate_by_board"] = _open_rate_by_board(plan)
    return out


def backtest(plan, m):
    import panelbt
    from panelbt.metrics import summarize

    bt = m.backtest
    t0 = time.perf_counter()
    result = panelbt.run(
        bt.market, plan.targets, bt.costs(**PARAMS["costs"]), bt.config()
    )
    run = time.perf_counter() - t0
    free = panelbt.run(
        bt.market,
        plan.targets,
        bt.costs(
            commission_rate=0.0,
            commission_min=0.0,
            slippage=0.0,
            tax_buy=0.0,
            tax_sell=0.0,
            fee_rate_buy=0.0,
            fee_rate_sell=0.0,
        ),
        bt.config(),
    )
    bench = np.nan_to_num(m.market_return)
    first = int(plan.targets.days[0]) if len(plan.targets.days) else 0
    s = summarize(result, benchmark=bench)
    s0 = summarize(free, benchmark=bench)
    gross_exposure = 1 - result.cash / result.equity
    years = len(m.dates[first:]) / 252
    nav = result.nav
    bench_nav = np.cumprod(1 + bench)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ew = np.nan_to_num(
            np.nanmean(np.where(m["has_quote"], m.returns, np.nan), axis=1)
        )
    ew_nav = np.cumprod(1 + ew)
    s_ew = summarize(result, benchmark=ew)
    by_year = {}
    yrs = np.array([d[:4] for d in m.dates])
    for y in sorted(set(yrs[first:])):
        idx = np.flatnonzero(yrs == y)
        lo, hi = max(idx[0] - 1, 0), idx[-1]
        by_year[y] = {
            "strategy": float(nav[hi] / nav[lo] - 1),
            "market": float(bench_nav[hi] / bench_nav[lo] - 1),
            "mean_exposure": float(gross_exposure[idx].mean()),
        }
    keep = (
        "annual_return",
        "annual_volatility",
        "sharpe",
        "max_drawdown",
        "excess_annual_return",
        "information_ratio",
        "annual_turnover",
        "tracking_error",
    )
    return {
        "counts": plan.counts,
        "positions": plan.positions.height,
        "rebalance_days": len(plan.targets.days),
        "seconds_run": run,
        "first_day": str(m.dates[first]),
        "years": years,
        "net": {k: s.get(k) for k in keep if k in s},
        "gross": {k: s0.get(k) for k in keep if k in s0},
        "summary_keys": sorted(s),
        "total_return_net": float(nav[-1] / nav[first] - 1),
        "total_return_gross": float(free.nav[-1] / free.nav[first] - 1),
        "market_total_return": float(bench_nav[-1] / bench_nav[first] - 1),
        "equal_weight_market_total_return": float(ew_nav[-1] / ew_nav[first] - 1),
        "net_vs_equal_weight": {
            k: s_ew.get(k)
            for k in ("excess_annual_return", "information_ratio", "tracking_error")
        },
        "mean_exposure": float(gross_exposure[first:].mean()),
        "costs_total_over_initial": float(
            (
                result.commission + result.tax + result.other_fee + result.slippage_cost
            ).sum()
            / result.equity[0]
        ),
        "by_year": by_year,
    }


def main(argv=None):
    import ashare.data as ad

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    ds = ad.open(root=args.root)
    t0 = time.perf_counter()
    m = load(ds, start=PARAMS["start"])
    timings = {"load": time.perf_counter() - t0}
    t0 = time.perf_counter()
    groups = size_industry_groups(
        m["industry"], m["total_market_cap"], PARAMS["size_buckets"], m["is_listed"]
    )
    gb = group_benchmark(m.returns, groups, member=m["has_quote"])
    pct = _size_pct(m)
    timings["group_benchmark"] = time.perf_counter() - t0
    report = {"params": PARAMS, "data_version": m.version, "timings": timings}
    report["forecast_strong"] = run_example(
        "forecast_strong", "业绩预告大幅预增", strong_forecasts(ds), m, gb, pct
    )
    report["express_growth"] = run_example(
        "express_growth", "业绩快报高增长", strong_expresses(ds), m, gb, pct
    )
    with open(args.out, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=str)
    print(json.dumps(timings))


if __name__ == "__main__":
    main()
