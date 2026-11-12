"""报告用到的绩效公式。

每个函数只接收一维日收益或净值数组，口径见 `docs/REPORT_NOTES.md`。
无法定义的量（样本不足、标准差为 0）返回 None，而不是 NaN 或 0。
"""

from __future__ import annotations

import math

import numpy as np


def finite(x) -> float | None:
    """把 numpy 标量转成 float；非有限值返回 None。"""
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


def rnd(x, k: int = 6) -> float | None:
    x = finite(x)
    return None if x is None else round(x, k)


def annual_return(r: np.ndarray, periods: int) -> float | None:
    """几何年化：`∏(1+r)^(periods/n) − 1`。"""
    if len(r) == 0:
        return None
    growth = float(np.prod(1 + r))
    if growth <= 0:
        return -1.0
    return growth ** (periods / len(r)) - 1


def volatility(r: np.ndarray, periods: int) -> float | None:
    if len(r) < 2:
        return None
    return float(np.std(r, ddof=1) * np.sqrt(periods))


def sharpe(r: np.ndarray, periods: int, rf: float = 0.0) -> float | None:
    if len(r) < 2:
        return None
    ex = r - rf / periods
    v = np.std(r, ddof=1)
    if not v > 0:
        return None
    return float(ex.mean() / v * np.sqrt(periods))


def sortino(r: np.ndarray, periods: int, rf: float = 0.0) -> float | None:
    if len(r) == 0:
        return None
    ex = r - rf / periods
    dn = np.sqrt(np.mean(np.minimum(ex, 0) ** 2))
    if not dn > 0:
        return None
    return float(ex.mean() * np.sqrt(periods) / dn)


def max_drawdown_of_returns(r: np.ndarray) -> float:
    """以区间起点为 1 的最大回撤（≤ 0）。"""
    w = np.r_[1.0, np.cumprod(1 + r)]
    return float((w / np.maximum.accumulate(w) - 1).min())


def drawdown(nav: np.ndarray) -> np.ndarray:
    """`nav_t / max_{s≤t} nav_s − 1`。"""
    return nav / np.maximum.accumulate(nav) - 1


def beta(r: np.ndarray, b: np.ndarray) -> float | None:
    if len(r) < 2:
        return None
    vb = np.var(b, ddof=1)
    if not vb > 0:
        return None
    return float(np.cov(r, b)[0, 1] / vb)


def episodes(nav: np.ndarray, dates: list[str]) -> list[dict]:
    """非重叠回撤段：从净值峰值开始，到净值重新回到（≥）峰值的那天结束。"""
    out = []
    peak_i, i, n = 0, 1, len(nav)
    while i < n:
        if nav[i] >= nav[peak_i]:
            peak_i = i
            i += 1
            continue
        j, trough = i, i
        while j < n and nav[j] < nav[peak_i]:
            if nav[j] < nav[trough]:
                trough = j
            j += 1
        rec = j if j < n else None
        out.append(
            {
                "peak": dates[peak_i],
                "trough": dates[trough],
                "recovery": dates[rec] if rec is not None else None,
                "depth": rnd(nav[trough] / nav[peak_i] - 1, 6),
                "days_to_trough": int(trough - peak_i),
                "duration_days": int((rec if rec is not None else n - 1) - peak_i),
                "recovered": rec is not None,
            }
        )
        if j >= n:
            break
        peak_i = j
        i = j + 1
    return out


def rolling(
    r: np.ndarray, b: np.ndarray | None, window: int, periods: int
) -> dict[str, list]:
    """截至每一天的最近 `window` 个日收益；`r[0]` 是基准日（不计入）。

    第一条落在第 `window` 行（窗口恰好是 r[1..window]）。`b` 中含 NaN 的窗口 beta 为 None。
    """
    out = {
        "date_index": [],
        "annual_return": [],
        "volatility": [],
        "sharpe": [],
        "beta": [],
    }
    n = len(r)
    if n <= window:
        return out
    from numpy.lib.stride_tricks import sliding_window_view

    w = sliding_window_view(r[1:], window)  # 第 k 个窗口截至第 k+window 行
    growth = np.prod(1 + w, axis=1) - 1
    sd = np.std(w, axis=1, ddof=1)
    mean = w.mean(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sh = np.where(sd > 0, mean / sd * np.sqrt(periods), np.nan)
    vol = sd * np.sqrt(periods)
    if b is not None:
        wb = sliding_window_view(b[1:], window)
        mb = wb.mean(axis=1)
        cov = ((w - mean[:, None]) * (wb - mb[:, None])).sum(axis=1) / (window - 1)
        vb = ((wb - mb[:, None]) ** 2).sum(axis=1) / (window - 1)
        with np.errstate(invalid="ignore", divide="ignore"):
            bt = np.where(vb > 0, cov / vb, np.nan)
    else:
        bt = np.full(len(w), np.nan)
    out["date_index"] = list(range(window, n))
    out["annual_return"] = [rnd(x, 5) for x in growth]
    out["volatility"] = [rnd(x, 5) for x in vol]
    out["sharpe"] = [rnd(x, 4) for x in sh]
    out["beta"] = [rnd(x, 4) for x in bt]
    return out
