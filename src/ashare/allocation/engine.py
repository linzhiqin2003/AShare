"""按调仓日逐个求权重，结果直接交给 `panelbt`。

时点（与 `ashare.backtest.timing` 一致）：调仓日 t 是**成交日**（面板第 t 行），对应的信号日是
t−1。第 t 个调仓日的权重只用 ``returns[:t]``，即 t−1 日收盘及以前的收益；`returns` 第 r 行是
第 r−1 日收盘到第 r 日收盘的收益。所以 `Allocation.targets()` 给出的行号就是成交行，不再移位。
"""

from __future__ import annotations

import json
import time
import warnings
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .covariance import Covariance
from .methods import FixedWeights, Method, RiskParity, Window, walk


def _as_arrays(returns, dates, symbols):
    if hasattr(returns, "to_numpy") and hasattr(returns, "columns"):
        frame = returns
        values = frame.to_numpy(dtype=float)
        if symbols is None:
            symbols = [str(c) for c in frame.columns]
        if dates is None:
            index = getattr(frame, "index", None)
            if index is not None and not isinstance(index, range):
                dates = [str(d)[:10] for d in index]
    else:
        values = np.asarray(returns, dtype=float)
    if values.ndim != 2 or values.shape[0] < 2 or values.shape[1] < 1:
        raise ValueError("returns: 需要 T×N 的日收益（T ≥ 2，N ≥ 1）")
    if np.isinf(values).any():
        raise ValueError("returns: 不能有无穷大，缺失用 NaN")
    if ((values <= -1) & np.isfinite(values)).any():
        raise ValueError("returns: 日收益必须大于 -1")
    t, n = values.shape
    symbols = (
        [f"asset{j}" for j in range(n)]
        if symbols is None
        else [str(s) for s in symbols]
    )
    if len(symbols) != n or len(set(symbols)) != n:
        raise ValueError("symbols: 长度必须等于列数且不重复")
    if dates is None:
        # 没给日期时按工作日从 2000-01-03 起编日期，只为按月调仓和出报告；不代表真实日历。
        days = np.busday_offset("2000-01-03", np.arange(t), roll="forward")
        dates = [str(d) for d in days]
    dates = [str(d) for d in dates]
    if len(dates) != t:
        raise ValueError("dates: 长度必须等于行数")
    return values, dates, symbols


def _month_key(day):
    digits = day.replace("-", "")
    return digits[:6] if len(digits) >= 6 and digits[:6].isdigit() else None


def schedule(dates, rebalance="monthly"):
    """调仓日（成交行号）。

    "monthly" / "quarterly"：每月 / 每季的第一个交易日（信号日是上一月最后一个交易日）；
    "weekly"：日期换了 ISO 周的第一行；"daily"：每一行；正整数 k：每 k 行一次；
    或者直接给成交行号序列。第 0 行没有可用数据，永远不是调仓日。
    """
    t = len(dates)
    if isinstance(rebalance, str):
        if rebalance == "daily":
            rows = np.arange(1, t)
        elif rebalance in ("monthly", "quarterly"):
            keys = [_month_key(d) for d in dates]
            if any(k is None for k in keys):
                raise ValueError("按月、按季调仓需要 YYYYMMDD 或 YYYY-MM-DD 日期")
            if rebalance == "quarterly":
                keys = [k[:4] + str((int(k[4:]) - 1) // 3) for k in keys]
            rows = np.array(
                [i for i in range(1, t) if keys[i] != keys[i - 1]], dtype=np.int64
            )
        elif rebalance == "weekly":
            from datetime import date

            def week(d):
                s = d.replace("-", "")
                return date(int(s[:4]), int(s[4:6]), int(s[6:8])).isocalendar()[:2]

            weeks = [week(d) for d in dates]
            rows = np.array(
                [i for i in range(1, t) if weeks[i] != weeks[i - 1]], dtype=np.int64
            )
        else:
            raise ValueError(
                "rebalance: monthly / quarterly / weekly / daily、正整数或行号序列"
            )
    elif isinstance(rebalance, (int, np.integer)) and not isinstance(rebalance, bool):
        if rebalance < 1:
            raise ValueError("rebalance: 正整数")
        rows = np.arange(int(rebalance), t, int(rebalance))
    else:
        rows = np.asarray(rebalance, dtype=np.int64)
        if (
            rows.ndim != 1
            or (rows.size and (rows.min() < 1 or rows.max() >= t))
            or np.any(np.diff(rows) <= 0)
        ):
            raise ValueError("rebalance: 行号必须严格递增且在 [1, T) 内")
    return rows.astype(np.int64)


def _plain(value):
    """诊断转成只含基本类型的结构。"""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        v = float(value)
        return v if np.isfinite(v) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


@dataclass
class Allocation:
    """`allocate` 的结果。

    rows：调仓日（成交行号）；dates：对应日期；weights：R×N 目标权重（和可以小于 1，余下为现金）；
    traded：该调仓日是否真的下单（再平衡带内的调仓日为 False，引擎不动持仓）；
    diagnostics：每个调仓日一条说明（可用资产数、剔除原因、风险平价残差、目标波动缩放等）。
    """

    rows: np.ndarray
    dates: list
    symbols: list
    weights: np.ndarray
    traded: np.ndarray
    diagnostics: list
    method: dict
    settings: dict
    n_rows: int
    all_dates: list = field(repr=False, default_factory=list)
    elapsed_seconds: float = 0.0

    def dense(self):
        """T×N：调仓行是目标权重，其余行（含带内不调仓的日子）整行 NaN，`panelbt` 视为不调仓。"""
        out = np.full((self.n_rows, len(self.symbols)), np.nan)
        out[self.rows[self.traded]] = self.weights[self.traded]
        return out

    def targets(self, *, validate=True):
        """交给 `panelbt.run` 的目标（行号就是成交行）。"""
        import panelbt as bt

        return bt.Targets.from_dense(self.dense(), validate=validate)

    def to_dict(self):
        """只含基本类型（str、int、float、bool、None、list、dict），可以直接 json.dumps。"""
        return _plain(
            dict(
                rows=self.rows,
                dates=list(self.dates),
                symbols=list(self.symbols),
                weights=self.weights,
                traded=self.traded,
                diagnostics=self.diagnostics,
                method=self.method,
                settings=self.settings,
                n_rows=int(self.n_rows),
            )
        )

    def to_json(self):
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def frame(self):
        """pandas 表（行为调仓日期，列为资产）；需要安装 pandas。"""
        import pandas as pd

        return pd.DataFrame(self.weights, index=self.dates, columns=self.symbols)


def allocate(
    returns,
    method: Method,
    *,
    dates=None,
    symbols=None,
    rebalance="monthly",
    lookback=252,
    min_history=60,
    covariance: Covariance | None = None,
    band: float | None = None,
):
    """逐个调仓日求权重。

    returns：T×N 日收益（numpy 数组或 pandas DataFrame，缺失为 NaN；第 r 行是 r−1 日收盘到
        r 日收盘的收益，第 0 行通常是 NaN）。
    method：配置方法（`RiskParity()`、`TargetVolatility(RiskParity(), 0.08)` ……）。
    rebalance：见 `schedule`；lookback：回看窗口（行数）；min_history：资产在窗口内至少要有
        这么多个有效收益、且 t−1 日有收益，才算可用。
    covariance：协方差估计方法，默认 `Covariance()`（半衰期 63 日、常相关目标、Ledoit–Wolf 强度）。
    band：再平衡带。给了时，若上次调仓后按收益漂移到 t−1 日收盘的权重与本次目标的最大偏离
        不超过 band，这一天不下单（`traded=False`）。

    最短例子::

        import numpy as np
        import ashare.allocation as aa

        r = np.random.default_rng(0).normal(0, 0.01, (600, 4))
        res = aa.allocate(r, aa.RiskParity(), rebalance=21)
        res.weights[-1], res.diagnostics[-1]["risk_parity"]["residual"]
    """
    started = time.perf_counter()
    if not isinstance(method, Method):
        raise TypeError("method: 需要 ashare.allocation 里的配置方法对象")
    values, dates, symbols = _as_arrays(returns, dates, symbols)
    t, n = values.shape
    if not isinstance(lookback, int) or lookback < 2:
        raise ValueError("lookback: 至少为 2 的整数")
    if not isinstance(min_history, int) or not 2 <= min_history <= lookback:
        raise ValueError("min_history: 需要 2 ≤ min_history ≤ lookback")
    if band is not None and not (np.isfinite(band) and band >= 0):
        raise ValueError("band: 非负数或 None")
    covariance = covariance or Covariance()
    rows = schedule(dates, rebalance)
    fixed, budget = None, None
    for m in walk(method):
        if isinstance(m, FixedWeights):
            fixed = m.resolve(symbols)
        if isinstance(m, RiskParity) and m.budget is not None:
            budget = (
                FixedWeights(m.budget).resolve(symbols)
                if isinstance(m.budget, dict)
                else np.asarray(m.budget, dtype=float)
            )
            if (
                budget.shape != (n,)
                or (budget < 0).any()
                or not np.isfinite(budget).all()
            ):
                raise ValueError(
                    "RiskParity.budget: 需要长度 N 的非负数组或 {代码: 份额}"
                )
    out_rows, weights, traded, diagnostics = [], [], [], []
    previous = np.zeros(n)
    last_trade = None
    for row in rows:
        history = values[:row]  # 只含 t−1 日收盘及以前
        window = history[max(0, row - lookback) :]
        counts = np.isfinite(window).sum(axis=0)
        available = (counts >= min_history) & np.isfinite(history[-1])
        if not available.any():
            continue
        columns = np.flatnonzero(available)
        notes = {}
        if fixed is not None:
            notes["fixed_target"] = fixed
        if budget is not None:
            notes["budget_full"] = budget
        win = Window(
            window[:, columns],
            history,
            available,
            columns,
            covariance,
            previous.copy(),
            notes,
        )
        w = np.asarray(method.weights(win), dtype=float)
        if w.shape != (n,) or not np.isfinite(w).all():
            raise RuntimeError(f"{method.name}: 第 {row} 行给出了非法权重")
        w[np.abs(w) < 1e-15] = 0.0
        if (w < 0).any():
            raise RuntimeError(f"{method.name}: 第 {row} 行出现负权重")
        trade = True
        drift = None
        if band is not None and last_trade is not None:
            growth = np.nan_to_num(values[last_trade + 1 : row]) + 1.0
            held = previous * np.prod(growth, axis=0)
            total = held.sum() + max(0.0, 1.0 - previous.sum())
            drifted = held / total if total > 0 else held
            drift = float(np.max(np.abs(drifted - w)))
            trade = drift > band
        notes.pop("fixed_target", None)
        notes.pop("budget_full", None)
        info = dict(
            row=int(row),
            date=dates[row],
            available=int(available.sum()),
            invested=float(w.sum()),
            traded=bool(trade),
            **({"drift": drift} if drift is not None else {}),
            **notes,
        )
        out_rows.append(int(row))
        weights.append(w)
        traded.append(trade)
        diagnostics.append(_plain(info))
        if trade:
            previous = w
            last_trade = int(row)
    weights = np.array(weights) if weights else np.zeros((0, n))
    solver = Counter(
        d["risk_parity"]["status"] for d in diagnostics if "risk_parity" in d
    )
    issues = {k: v for k, v in solver.items() if k != "converged"}
    if issues:
        warnings.warn(
            f"风险平价有 {sum(issues.values())} 个调仓日没有在原协方差上解到（{issues}）；"
            "见各调仓日 diagnostics 的 risk_parity",
            RuntimeWarning,
            stacklevel=2,
        )
    return Allocation(
        rows=np.array(out_rows, dtype=np.int64),
        dates=[dates[r] for r in out_rows],
        symbols=list(symbols),
        weights=weights,
        traded=np.array(traded, dtype=bool),
        diagnostics=diagnostics,
        method=_plain(method.describe()),
        settings=_plain(
            dict(
                rebalance=rebalance if isinstance(rebalance, (str, int)) else "rows",
                lookback=lookback,
                min_history=min_history,
                band=band,
                covariance=dict(
                    half_life=covariance.half_life,
                    shrinkage=covariance.shrinkage,
                    target=covariance.target,
                ),
                risk_parity_status=dict(solver),
            )
        ),
        n_rows=t,
        all_dates=list(dates),
        elapsed_seconds=time.perf_counter() - started,
    )
