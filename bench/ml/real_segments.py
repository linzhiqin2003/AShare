"""真实例子的分段、小盘基准、容量和持仓市值分布。

不重新训练：直接读 `real_example.py` 那一轮各模型存下的预测（运行目录 `runs/<run_id>/pred.npy`，
run_id 取自 `results/real_<模型>.json`），按特征库的样本索引摊回 T×N，先核对 Rank IC 均值
与当时记录的逐位一致，再做下面这些。所以不碰账本，也不产生新的试验。

- 分段：同一条净值曲线截取 全期 / 2012–2016 / 2017 起 / 2020 起，年化与夏普；
- 基准：股票池等权（每 5 日）、股票池里总市值最小的 50 只等权（每 5 日），同一引擎同一成本；
- 容量：单日成交量占比上限 5%，1000 万加平方根冲击；1 亿分别「不算冲击」和「加平方根冲击」。
  不报「不限成交量、加冲击」：不限量时会有单笔成交超过当日全天成交量的订单，按平方根律
  算出的冲击没有意义
  （`ashare.analysis.impact_coefficients`，k = 此前 20 日日波动率，与 `an.capacity` 同口径），
  另给 `an.capacity` 的解析表（1000 万 / 1 亿 / 10 亿下成交量占比分布和年化冲击成本）；
- 持仓市值：每个调仓日 top 50 在股票池总市值截面中的分位，落在最小 10% / 20% 的比例与分位均值；
- 预测上下半截面的 Rank IC，以及 top 50 的 5 日平均超额（相对股票池均值）。

用法：MallocLargeCache=0 .venv/bin/python bench/ml/real_segments.py
结果写进 bench/ml/results/real_segments.json。
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import replace

import numpy as np
import panelbt
import panelops
from mlbench import DATA, ML_ROOT, RESULTS, log, save

import ashare.analysis as an
import ashare.data as ad
import ashare.ml as ml
from ashare.backtest import market_from_panel

MODELS = ("naive", "ridge", "lgbm", "lgbm_rank", "ensemble")
START = "20120101"
SEGMENTS = {
    "full": ("20120101", None),
    "2012_2016": ("20120101", "20161231"),
    "from_2017": ("20170101", None),
    "from_2020": ("20200101", None),
}
TOP = 50
EVERY = 5
STORE = "942d84a2b85db69456d4517dc17a2849a657d49de57e4db7035e2cfc250981d2"
FIELDS = ("volume", "amount", "total_market_cap", "close", "adj_factor")


def seg_stats(nav, dates, first, lo, hi):
    """净值 nav（行 = 日）在 [lo, hi] 日期内的年化、夏普；起点不早于 first。"""
    d = dates.astype(str)
    a = max(first, int(np.searchsorted(d, lo)))
    b = len(d) if hi is None else int(np.searchsorted(d, hi, side="right"))
    # 分段收益从 a-1 的净值算起，保证各段首尾相接
    v = np.asarray(nav, dtype=float)[max(a - 1, first) : b]
    r = v[1:] / v[:-1] - 1
    years = len(r) / 243
    return {
        "annual_return": float((v[-1] / v[0]) ** (1 / years) - 1),
        "sharpe": float(r.mean() / r.std(ddof=1) * math.sqrt(243)),
        "max_drawdown": float((v / np.maximum.accumulate(v) - 1).min()),
    }


def all_segments(nav, dates, first):
    return {k: seg_stats(nav, dates, first, lo, hi) for k, (lo, hi) in SEGMENTS.items()}


def yearly(nav, dates, first):
    d = dates.astype(str)
    years = sorted({x[:4] for x in d[first:]})
    out = {}
    for y in years:  # 该自然年的收益（不年化）
        a = max(first, int(np.searchsorted(d, y + "0101")))
        b = int(np.searchsorted(d, y + "1231", side="right"))
        v = np.asarray(nav)[max(a - 1, first) : b]
        out[y] = float(v[-1] / v[0] - 1)
    return out


def size_profile(score, cap, universe, first):
    """每个信号日 top 50 在股票池市值截面中的分位（0 = 最小）。"""
    rows = np.arange(first, score.shape[0], EVERY)
    pct, b10, b20 = [], [], []
    for t in rows:
        u = universe[t] & np.isfinite(cap[t])
        s = np.where(u & np.isfinite(score[t]), score[t], np.nan)
        if np.isfinite(s).sum() < TOP:
            continue
        pick = np.argsort(-np.nan_to_num(s, nan=-np.inf))[:TOP]
        c = cap[t][u]
        q = np.searchsorted(np.sort(c), cap[t][pick], side="right") / len(c)
        pct.append(q.mean())
        b10.append((q <= 0.1).mean())
        b20.append((q <= 0.2).mean())
    return {
        "mean_cap_pct": float(np.mean(pct)),
        "bottom10_share": float(np.mean(b10)),
        "bottom20_share": float(np.mean(b20)),
        "days": len(pct),
    }


def half_ic(pred, raw, rows):
    """预测上半截面、下半截面各自的逐日 Rank IC 均值。"""
    p = pred[rows]
    y = raw[rows]
    ok = np.isfinite(p) & np.isfinite(y)
    p = np.where(ok, p, np.nan)
    med = np.nanmedian(p, axis=1, keepdims=True)
    upper = np.where(p >= med, p, np.nan)
    lower = np.where(p < med, p, np.nan)
    return (
        float(np.nanmean(panelops.rank_ic(upper, y))),
        float(np.nanmean(panelops.rank_ic(lower, y))),
    )


def top_excess(score, raw, universe, first):
    """信号日 top 50 的 5 日平均收益减去股票池平均（基点）。"""
    rows = np.arange(first, score.shape[0], EVERY)
    out = []
    for t in rows:
        u = universe[t] & np.isfinite(raw[t])
        s = np.where(u & np.isfinite(score[t]), score[t], np.nan)
        if np.isfinite(s).sum() < TOP:
            continue
        pick = np.argsort(-np.nan_to_num(s, nan=-np.inf))[:TOP]
        out.append(np.nanmean(raw[t][pick]) - np.nanmean(raw[t][u]))
    return float(np.mean(out) * 1e4)


def main():
    t0 = time.perf_counter()
    data = ad.open(root=DATA)
    feats = ml.basic_features()
    fields = tuple(dict.fromkeys((*ml.required_fields(feats, "ret5"), *FIELDS)))
    bm = market_from_panel(data, fields=fields)
    panel = bm.panel
    dates = np.asarray(panel.dates).astype(str)
    universe = ml.default_universe(panel)
    if bm.market.volume is None:
        raise SystemExit("行情里没有成交量，成交量上限无法生效")
    costs = bm.costs()
    first = int(np.searchsorted(dates, START))
    cap = np.asarray(panel["total_market_cap"], dtype=float)
    raw = ml.compute_label(panel, "ret5", universe).raw

    idx = np.load(ML_ROOT / "stores" / STORE / "index.npz")
    rows_s, cols_s = idx["rows"], idx["cols"]

    def scatter(v):
        out = np.full(raw.shape, np.nan)
        out[rows_s, cols_s] = v
        return out

    k = an.impact_coefficients(panel, 1.0)
    impact_costs = replace(costs, impact_k=np.ascontiguousarray(k), impact_exponent=0.5)

    def backtests(targets):
        out = {}
        for name, cash, part, c in (
            ("1e7_uncapped", 1e7, 0.0, costs),
            ("1e7_cap5_impact", 1e7, 0.05, impact_costs),
            ("1e8_cap5", 1e8, 0.05, costs),
            ("1e8_cap5_impact", 1e8, 0.05, impact_costs),
        ):
            res = panelbt.run(
                bm.market,
                targets,
                c,
                bm.config(initial_cash=cash, max_participation=part),
            )
            # 第一个信号日成交在 first+1，净值从 first+1 起算（与 real_example 一致）
            out[name] = all_segments(res.nav, dates, first + 1)
            if name == "1e7_uncapped":
                out["yearly"] = yearly(res.nav, dates, first + 1)
        return out

    def capacity_table(targets):
        c = an.capacity(
            targets, panel, aum=[1e7, 1e8, 1e9], participation_cap=0.05, costs=costs
        )
        tab = c.table.to_dict()
        return {
            "table": tab,
            "by_participation": c.capacity["by_participation"],
            "by_participation_status": c.capacity["by_participation_status"],
        }

    report = {"segments_from": SEGMENTS, "top": TOP, "every": EVERY}
    test_rows = None
    for m in MODELS:
        meta = json.loads((RESULTS / f"real_{m}.json").read_text())
        run_dir = ML_ROOT / "runs" / meta["run_id"]
        pred = scatter(np.load(run_dir / "pred.npy"))
        rows = np.flatnonzero(np.isfinite(pred).any(axis=1))
        test_rows = rows if test_rows is None else test_rows
        ic = float(np.nanmean(panelops.rank_ic(pred[rows], raw[rows])))
        if abs(ic - meta["ic_mean"]) > 1e-12:
            raise SystemExit(f"{m}: Rank IC {ic} 与记录 {meta['ic_mean']} 不一致")
        targets = ml.to_targets(pred, top=TOP, every=EVERY, lag=1, start=first)
        report[m] = {
            "run_id": meta["run_id"],
            "ic_mean": ic,
            "half_ic": half_ic(pred, raw, rows),
            "top_excess_bp": top_excess(pred, raw, universe, first),
            "size": size_profile(pred, cap, universe, first),
            "backtest": backtests(targets),
            "capacity": capacity_table(targets),
        }
        log(m, report[m]["backtest"]["1e7_uncapped"], report[m]["size"])

    ones = np.where(universe, 1.0, np.nan)
    uni_targets = ml.to_targets(
        ones, top=int(universe.sum(axis=1).max()), every=EVERY, lag=1, start=first
    )
    report["universe_ew"] = {"backtest": backtests(uni_targets)}
    small = np.where(universe & np.isfinite(cap), -cap, np.nan)
    small_targets = ml.to_targets(small, top=TOP, every=EVERY, lag=1, start=first)
    report["smallcap50"] = {
        "definition": "股票池里总市值（total_market_cap）最小的 50 只，每 5 日等权",
        "size": size_profile(small, cap, universe, first),
        "top_excess_bp": top_excess(small, raw, universe, first),
        "backtest": backtests(small_targets),
        "capacity": capacity_table(small_targets),
    }
    log("smallcap50", report["smallcap50"]["backtest"]["1e7_uncapped"])
    report["seconds"] = time.perf_counter() - t0
    save("real_segments", report)


if __name__ == "__main__":
    main()
