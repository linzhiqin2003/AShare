"""分组回测的指标：时序移位、风险因子与暴露、分段业绩表和净值曲线。

面板均为 (T, N)：行是交易日，列是股票，缺失用 NaN。所有函数不修改传入数组。
"""

from __future__ import annotations

import numpy as np
import polars as pl


def ts_delay(x, d):
    """把面板沿时间轴后移 d 行，前 d 行为 NaN；d = 0 返回拷贝。"""
    d = int(d)
    if d < 0:
        raise ValueError("d 必须是非负整数")
    a = np.asarray(x, dtype=np.float64)
    if d == 0:
        return a.copy()
    out = np.full_like(a, np.nan)
    if d < a.shape[0]:
        out[d:] = a[:-d]
    return out


def risk_factors(close_adj, total_mv):
    """按收盘价与总市值构造风险因子；非正的价格或市值先当 NaN。"""
    close = np.array(close_adj, dtype=np.float64, copy=True)
    mv = np.array(total_mv, dtype=np.float64, copy=True)
    close[~(close > 0)] = np.nan
    mv[~(mv > 0)] = np.nan

    delay20 = ts_delay(close, 20)
    delay60 = ts_delay(close, 60)
    delay240 = ts_delay(close, 240)
    return {
        "size": np.log(mv),
        "reversal_20": 1.0 - close / delay20,
        "reversal_60": 1.0 - close / delay60,
        "momentum": delay20 / delay240 - 1.0,
    }


def risk_exposure(weights, risk, universe):
    """每行截面去均值后的风险因子按权重加总；整行无有效权重的暴露为 NaN。"""
    # 只分时间块，完整保留每行的列序及 NumPy 归约路径。
    if len(weights) > 32:
        out = {k: np.empty(len(weights)) for k in risk}
        for first in range(0, len(weights), 32):
            s = slice(first, first + 32)
            part = risk_exposure(
                weights[s], {k: v[s] for k, v in risk.items()}, universe[s]
            )
            for k in risk:
                out[k][s] = part[k]
        return out
    w = np.array(weights, dtype=np.float64, copy=True)
    w = np.where(np.isnan(w), 0.0, w)
    active = np.any(w != 0.0, axis=1)
    univ = np.asarray(universe, dtype=bool)

    out = {}
    for name, factor in risk.items():
        f = np.asarray(factor, dtype=np.float64)
        valid = univ & np.isfinite(f)
        count = valid.sum(axis=1)
        total = np.where(valid, f, 0.0).sum(axis=1)
        mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
        demeaned = np.where(valid, f - mean[:, None], 0.0)
        exposure = (w * demeaned).sum(axis=1)
        exposure = np.where(active, exposure, np.nan)
        out[name] = exposure
    return out


def risk_exposure_from_prices(weights, close, total_mv, universe, first):
    """分块算风险因子并立即归约；只保留最终日序列，不保存四张风险面板。

    close 是截至止日的完整前填复权价；first 是持仓面板首行的全历史行号。
    成交行 t 对应信号行 t-1，四个风险因子的运算及每行求和顺序均不变。
    """
    names = ("size", "reversal_20", "reversal_60", "momentum")
    out = {name: np.full(len(weights), np.nan) for name in names}
    for begin in range(1, len(weights), 32):
        end = min(begin + 32, len(weights))
        lo, hi = first + begin - 1, first + end - 1

        def delayed(d, lo=lo, hi=hi):
            values = np.full((hi - lo, close.shape[1]), np.nan)
            start = max(lo, d)
            if start < hi:
                values[start - lo :] = close[start - d : hi - d]
            return values

        current = close[lo:hi]
        cap = np.asarray(total_mv[lo:hi], dtype=np.float64)
        delay20 = delayed(20)
        risk = {
            "size": np.log(np.where(cap > 0, cap, np.nan)),
            "reversal_20": 1.0 - current / delay20,
            "reversal_60": 1.0 - current / delayed(60),
            "momentum": delay20 / delayed(240) - 1.0,
        }
        part = risk_exposure(weights[begin:end], risk, universe[begin - 1 : end - 1])
        for name in names:
            out[name][begin:end] = part[name]
    return out


def period_labels(dates, by):
    """把 YYYYMMDD 日期映射到年份（"2021"）或季度（"2021Q3"）标签。"""
    if by not in ("year", "quarter"):
        raise ValueError("by 只能是 'year' 或 'quarter'")
    labels = []
    for date in dates:
        s = str(date)
        if by == "year":
            labels.append(s[:4])
        else:
            month = int(s[4:6])
            labels.append(f"{s[:4]}Q{(month - 1) // 3 + 1}")
    return np.asarray(labels, dtype=np.str_)


def performance(pnl, turnover, periods=240):
    """一段日序列的收益、夏普、换手、最大回撤和每万元换手收益。"""
    p = np.asarray(pnl, dtype=np.float64)
    t = np.asarray(turnover, dtype=np.float64)
    t = np.where(np.isnan(t), 0.0, t)
    keep = ~np.isnan(p)
    p = p[keep]
    t = t[keep]

    keys = ("return", "sharpe", "turnover", "maxdrawdown", "margin")
    result = dict.fromkeys(keys)
    n = p.size
    if n == 0:
        return result

    result["return"] = float(np.mean(p) * periods)
    sd = float(np.std(p, ddof=1)) if n >= 2 else 0.0
    if n >= 2 and sd != 0.0:
        result["sharpe"] = float(np.mean(p) / sd * np.sqrt(periods))
    result["turnover"] = float(np.mean(t))

    cumulative = np.cumsum(p)
    peak = np.maximum(np.maximum.accumulate(cumulative), 0.0)
    result["maxdrawdown"] = float(np.max(peak - cumulative))

    total_turnover = float(np.sum(t))
    if total_turnover != 0.0:
        result["margin"] = float(np.sum(p) / total_turnover * 10000.0)
    return result


def summary_table(pnl, turnover, dates, by="year", periods=240):
    """按年份或季度分段汇总，最后一行 ALL 用全部日子；整段没有盈亏的时间段不列。"""
    p = np.asarray(pnl, dtype=np.float64)
    t = np.asarray(turnover, dtype=np.float64)
    labels = period_labels(dates, by)

    rows = []
    for label in sorted(set(labels.tolist())):
        mask = labels == label
        if np.isnan(p[mask]).all():
            continue  # 这一段还没开始回测
        rows.append((label, performance(p[mask], t[mask], periods)))
    rows.append(("ALL", performance(p, t, periods)))

    columns = {
        "period": [label for label, _ in rows],
        "return": [m["return"] for _, m in rows],
        "sharpe": [m["sharpe"] for _, m in rows],
        "turnover": [m["turnover"] for _, m in rows],
        "maxdrawdown": [m["maxdrawdown"] for _, m in rows],
        "margin": [m["margin"] for _, m in rows],
    }
    schema = {
        "period": pl.String,
        "return": pl.Float64,
        "sharpe": pl.Float64,
        "turnover": pl.Float64,
        "maxdrawdown": pl.Float64,
        "margin": pl.Float64,
    }
    return pl.DataFrame(columns, schema=schema)


def curves(pnl):
    """累计盈亏与累计净值曲线，pnl 的 NaN 当 0。"""
    p = np.array(pnl, dtype=np.float64, copy=True)
    p = np.where(np.isnan(p), 0.0, p)
    return {"cumsum": np.cumsum(p), "cumprod": np.cumprod(1.0 + p) - 1.0}
