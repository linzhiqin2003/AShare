"""ASHare 单因子检查：表达式 -> 因子矩阵 -> 组合/IC -> plotly 图。

和 `自营投研/machinelib.py` 的 `run_expression_backtest_from_store` 用法对齐：
你只需要给表达式和 setting，函数返回 `{"summary": ..., "figures": ...}`，
并在 notebook 里直接用 plotly 画图（同一个包、同一套图名/口径）。

口径：
- 因子第 s 行（面板日期轴）信号，s+1 成交（open 或 close），持有到下一天；
- 分组为池内等权纯多头，longshort = 第 1 组 − 最后一组，pure_long = 第 1 组；
- 累计收益用 **加法 cumsum**（和参考代码一致）；
- 不扣成本（参考代码同样不扣；要成本回测用仓库里的 panelbt 路径）。
"""

from __future__ import annotations

import ast
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

ASHARE_DIR = Path(__file__).resolve().parents[2]  # docs/notebooks -> 仓库根目录
if not (ASHARE_DIR / "src" / "ashare").is_dir():
    ASHARE_DIR = Path(os.environ.get("ASHARE_REPO", Path.cwd()))
DATA_DIR = Path(os.environ.get("ASHARE_DATA_DIR", ASHARE_DIR.parent / "ashare-data"))
if str(ASHARE_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ASHARE_DIR / "src"))
os.environ.setdefault("ASHARE_DATA_DIR", str(DATA_DIR))

ANN_FACTOR = 244

_LAYOUT = {
    "legend": dict(
        x=0.02,
        y=0.98,
        xanchor="left",
        yanchor="top",
        bgcolor="rgba(255,255,255,0.5)",
    ),
    "showlegend": True,
    "margin": dict(l=40, r=40, t=40, b=40),
    "height": 320,
}


def _series_fig(series_dict, title, ylabel="Value"):
    fig = go.Figure()
    for name, series in series_dict.items():
        if series is None:
            continue
        clean = pd.Series(series).dropna()
        fig.add_trace(go.Scatter(x=clean.index, y=clean.values, name=name, mode="lines"))
    fig.update_layout(**_LAYOUT, title=title, xaxis_title="Date", yaxis_title=ylabel)
    return fig


def _cum(series):
    series = pd.Series(series).astype(float).replace([np.inf, -np.inf], np.nan)
    return series.cumsum().where(series.notna())


def _pearson(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    x0 = x - x.mean()
    y0 = y - y.mean()
    den = np.sqrt((x0 * x0).sum() * (y0 * y0).sum())
    return float((x0 * y0).sum() / den) if den > 0 else np.nan


def _rank(x):
    order = np.argsort(np.asarray(x, float), kind="stable")
    ranks = np.empty(len(order), float)
    ranks[order] = np.arange(len(order), dtype=float)
    return ranks


def _spearman(x, y):
    return _pearson(_rank(x), _rank(y))


class _PanelSlice:
    """面板的 [start, stop) 只读视图，给 Evaluator 用。"""

    def __init__(self, panel, start=0, stop=None):
        self._panel, self._start = panel, start
        self._stop = len(panel.dates) if stop is None else stop
        self.dates = np.asarray(panel.dates)[start:self._stop]
        self.symbols = np.asarray(panel.symbols)
        self.version = f"{getattr(panel, 'version', 'panel')}:{start}-{self._stop}"

    def __getitem__(self, name):
        return np.asarray(self._panel[name])[self._start:self._stop]

    @property
    def shape(self):
        return len(self.dates), len(self.symbols)


# ---- 表达式：平台 DSL 优先，BRAIN 风格兜底 ---------------------------------


def _expression_names():
    from ashare.factor import FIELDS as FACTOR_FIELDS, field
    from ashare.factor.formula import dsl

    names = {name: field(name) for name in FACTOR_FIELDS}
    names.update(
        returns=dsl.returns,
        cap=dsl.cap,
        close=dsl.close,
        open=dsl.open,
        high=dsl.high,
        low=dsl.low,
        vwap=dsl.vwap,
        volume=dsl.volume,
        amount=dsl.amount,
    )
    return names


def _brain_build(node, names):
    from ashare.factor import call, constant
    from ashare.factor.formula import ops_extra

    if isinstance(node, ast.Constant):
        return constant(node.value)
    if isinstance(node, ast.Name):
        if node.id not in names:
            raise ValueError(f"未知字段/变量：{node.id}")
        return names[node.id]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return call("multiply", constant(-1.0), _brain_build(node.operand, names))
    binary = {
        ast.Add: "add",
        ast.Sub: "subtract",
        ast.Mult: "multiply",
        ast.Div: "divide",
        ast.Pow: "power",
    }
    if isinstance(node, ast.BinOp) and type(node.op) in binary:
        return call(
            binary[type(node.op)],
            _brain_build(node.left, names),
            _brain_build(node.right, names),
        )
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        op = node.func.id
        args = [
            arg.value if isinstance(arg, ast.Constant) else _brain_build(arg, names)
            for arg in node.args
        ]
        kwargs = {
            kw.arg: kw.value if isinstance(kw.value, ast.Constant) else _brain_build(kw.value, names)
            for kw in node.keywords
        }
        if op == "winsorize":
            warnings.warn("winsorize 在 ASHare 里没有实现，已忽略（建议直接用 zscore/rank）", stacklevel=2)
            return args[0]
        if op in ops_extra.FORMULAS and op in ops_extra.BUILDERS:
            return ops_extra.build(op, *args, **kwargs)
        from ashare.platform import language

        if op in language.OPERATORS:
            return language.OPERATORS[op](*args, **kwargs)
        raise ValueError(f"不支持的算子：{op}")
    raise ValueError(f"不支持的写法：{type(node).__name__}")


def _parse_expression(text):
    from ashare.platform import language

    try:
        variants = language.parse(text)
        if variants:
            return variants[0].expression
    except language.ExpressionError:
        pass
    names = _expression_names()
    return _brain_build(ast.parse(text, mode="eval").body, names)


# ---- 组合 -------------------------------------------------------------------


def _portfolio_weights(factor_row, mask_row, n_groups, top_n_list):
    base = np.asarray(mask_row, bool) & np.isfinite(factor_row)
    idx = np.flatnonzero(base)
    out = {}
    if idx.size < 2:
        return out
    order = np.argsort(-np.asarray(factor_row)[idx], kind="stable")
    sorted_idx = idx[order]
    count = len(sorted_idx)

    members = [[] for _ in range(n_groups)]
    for rank, col in enumerate(sorted_idx):
        members[rank * n_groups // count].append(int(col))
    for g, group in enumerate(members):
        if group:
            out[f"group_{g + 1}"] = {col: 1.0 / len(group) for col in group}

    top, bottom = members[0], members[-1]
    weights = {}
    for col in top:
        weights[col] = weights.get(col, 0.0) + 0.5 / len(top)
    for col in bottom:
        weights[col] = weights.get(col, 0.0) - 0.5 / len(bottom)
    out["longshort"] = {col: w for col, w in weights.items() if w != 0.0}
    out["pure_long"] = out.get("group_1", {})

    for n in top_n_list:
        head = sorted_idx[: min(n, count)]
        tail = sorted_idx[-min(n, count):]
        out[f"top{n}_head"] = {int(col): 1.0 / len(head) for col in head}
        out[f"top{n}_tail"] = {int(col): 1.0 / len(tail) for col in tail}
    return out


def _turnover(previous, current):
    if previous is None:
        return 1.0 if current else 0.0
    if not current:
        return 1.0
    universe = set(previous) | set(current)
    total = sum(abs(current.get(col, 0.0) - previous.get(col, 0.0)) for col in universe)
    return 0.5 * total


def _stats(pnl, turnover):
    clean = pd.Series(pnl).dropna()
    if clean.empty:
        return dict(return_=np.nan, sharpe=np.nan, max_drawdown=np.nan, turnover=np.nan)
    annual = float(clean.mean() * ANN_FACTOR)
    vol = float(clean.std() * np.sqrt(ANN_FACTOR))
    sharpe = annual / vol if vol > 0 else np.nan
    cum = clean.cumsum()
    drawdown = float((cum - cum.cummax()).min())
    return dict(
        return_=annual,
        sharpe=sharpe,
        max_drawdown=drawdown,
        turnover=float(pd.Series(turnover).dropna().mean()),
    )


def run_single_factor(
    expression,
    *,
    universe="TOP1800",
    start="2016-01-01",
    end="2025-01-01",
    trade_at="open",
    n_groups=10,
    top_n_list=(5, 10, 20, 50),
    rebalance=1,
    show_figures=True,
):
    """单因子检查：返回 {"summary": dict, "figures": {name: plotly Figure}}。"""
    import ashare.data
    from ashare.factor import FIELDS as FACTOR_FIELDS, Evaluator, groupbt
    from ashare.factor.groupbt.api import _adjusted, _forward_return

    data = ashare.data.open()
    fields = sorted(
        set(FACTOR_FIELDS)
        | set(groupbt.FIELDS)
        | {"volume", "split", "cash_dividend", "delist", "high", "low"}
    )
    panel = data.panel(fields)
    dates = np.asarray(panel.dates)
    stop = int(np.searchsorted(dates, str(end).replace("-", "")))
    start_i = int(np.searchsorted(dates, str(start).replace("-", "")))
    if stop - start_i < 3:
        raise ValueError("窗口内交易日不足")

    expr = _parse_expression(expression)
    history = np.asarray(
        Evaluator().evaluate(expr, _PanelSlice(panel, 0, stop)), dtype=float
    )
    alpha = history[start_i:stop]
    mask = np.asarray(groupbt.universe_mask(panel, universe))[start_i:stop]

    price = _adjusted(_PanelSlice(panel, 0, stop), trade_at)
    forward = _forward_return(price)
    ret = forward[start_i:stop]
    dates_w = pd.to_datetime([str(d) for d in dates[start_i:stop]])

    names = (
        [f"group_{i + 1}" for i in range(n_groups)]
        + ["longshort", "pure_long"]
        + [f"top{n}_head" for n in top_n_list]
        + [f"top{n}_tail" for n in top_n_list]
    )
    pnl = {name: np.full(len(dates_w), np.nan) for name in names}
    turnover = {name: np.zeros(len(dates_w)) for name in names}
    previous = {name: None for name in names}

    signal_rows = set(range(0, len(dates_w) - 1, max(1, int(rebalance))))
    for d in range(1, len(dates_w)):
        s = d - 1
        if s in signal_rows:
            weights = _portfolio_weights(alpha[s], mask[s], n_groups, top_n_list)
            for name in names:
                current = weights.get(name)
                if current is not None:
                    turnover[name][d] = _turnover(previous[name], current)
                    previous[name] = current
        row = ret[d]
        row_ok = np.isfinite(row)
        for name in names:
            weights = previous[name]
            if not weights:
                continue
            cols = np.fromiter(weights.keys(), dtype=np.int64)
            w = np.fromiter(weights.values(), dtype=float)
            values = row[cols]
            keep = np.isfinite(values)
            if keep.any():
                pnl[name][d] = float(np.nansum(values[keep] * w[keep]))

    ic = np.full(len(dates_w), np.nan)
    ric = np.full(len(dates_w), np.nan)
    for s in signal_rows:
        d = s + 1
        base = mask[s] & np.isfinite(alpha[s]) & np.isfinite(ret[d])
        x = alpha[s][base]
        y = ret[d][base]
        if x.size >= 10:
            ic[d] = _pearson(x, y)
            ric[d] = _spearman(x, y)

    coverage = (mask & np.isfinite(alpha)).sum(1) / np.maximum(mask.sum(1), 1)
    unique = np.array(
        [
            np.unique(alpha[row][np.isfinite(alpha[row])]).size
            / max(np.isfinite(alpha[row]).sum(), 1)
            for row in range(len(dates_w))
        ]
    )

    pnl_series = {name: pd.Series(values, index=dates_w) for name, values in pnl.items()}
    turnover_series = {
        name: pd.Series(values, index=dates_w) for name, values in turnover.items()
    }
    ic_series = pd.Series(ic, index=dates_w).dropna()
    ric_series = pd.Series(ric, index=dates_w).dropna()
    coverage_series = pd.Series(coverage, index=dates_w)
    unique_series = pd.Series(unique, index=dates_w)

    figures = {}
    figures["group_and_longshort_cum_pnl"] = _series_fig(
        {
            **{f"group_{i + 1}": _cum(pnl_series[f"group_{i + 1}"]) for i in range(n_groups)},
            "longshort": _cum(pnl_series["longshort"]),
            "pure_long": _cum(pnl_series["pure_long"]),
        },
        title="Grouped Cumulative Returns and Long-Short / Pure-Long Cumulative Return",
        ylabel="Cumulative Return Rate",
    )
    figures["group_and_longshort_pnl"] = _series_fig(
        {
            **{f"group_{i + 1}": pnl_series[f"group_{i + 1}"] for i in range(n_groups)},
            "longshort": pnl_series["longshort"],
            "pure_long": pnl_series["pure_long"],
        },
        title="Grouped Daily Returns and Long-Short / Pure-Long Return",
        ylabel="Return Rate",
    )
    figures["top_head_cum_pnl"] = _series_fig(
        {f"Top {n}": _cum(pnl_series[f"top{n}_head"]) for n in top_n_list},
        title="Head Top-N Cumulative Returns",
        ylabel="Cumulative Return Rate",
    )
    figures["top_tail_cum_pnl"] = _series_fig(
        {f"Tail {n}": _cum(pnl_series[f"top{n}_tail"]) for n in top_n_list},
        title="Tail Top-N Cumulative Returns",
        ylabel="Cumulative Return Rate",
    )
    figures["cum_ic_rank_ic"] = _series_fig(
        {"cum_ic": ic_series.cumsum(), "cum_rank_ic": ric_series.cumsum()},
        title="Cumulative IC and RankIC",
        ylabel="Cumulative Value",
    )
    figures["daily_rank_ic"] = _series_fig(
        {"RIC": ric_series, "RIC_MA20": ric_series.rolling(20).mean()},
        title="Daily RankIC",
        ylabel="RankIC",
    )
    figures["coverage"] = _series_fig(
        {"coverage": coverage_series, "coverage_MA20": coverage_series.rolling(20).mean()},
        title="Coverage",
        ylabel="Coverage",
    )
    figures["unique_ratio"] = _series_fig(
        {"unique_ratio": unique_series, "unique_MA20": unique_series.rolling(20).mean()},
        title="Unique Value Ratio",
        ylabel="Unique Ratio",
    )

    longshort_stats = _stats(pnl_series["longshort"], turnover_series["longshort"])
    pure_long_stats = _stats(pnl_series["pure_long"], turnover_series["pure_long"])
    head_stats = {
        f"top{n}": _stats(pnl_series[f"top{n}_head"], turnover_series[f"top{n}_head"])
        for n in top_n_list
    }
    summary = {
        "coverage_mean": float(coverage_series.mean()),
        "unique_ratio_mean": float(unique_series.mean()),
        "ic_mean": float(ic_series.mean()) if not ic_series.empty else np.nan,
        "rank_ic_mean": float(ric_series.mean()) if not ric_series.empty else np.nan,
        "ir": float(ic_series.mean() / ic_series.std()) if ic_series.std() else np.nan,
        "rank_ir": float(ric_series.mean() / ric_series.std()) if ric_series.std() else np.nan,
        "cum_ic_mean": float(ic_series.cumsum().mean()),
        "cum_rank_ic_mean": float(ric_series.cumsum().mean()),
        "longshort_sharpe": longshort_stats["sharpe"],
        "longshort_turnover": longshort_stats["turnover"],
        "longshort_return": longshort_stats["return_"],
        "longshort_max_drawdown": longshort_stats["max_drawdown"],
        "pure_long_sharpe": pure_long_stats["sharpe"],
        "pure_long_turnover": pure_long_stats["turnover"],
        "pure_long_return": pure_long_stats["return_"],
        "pure_long_max_drawdown": pure_long_stats["max_drawdown"],
        "period_start": str(dates_w[0].date()),
        "period_end": str(dates_w[-1].date()),
        "universe": universe,
        "trade_at": trade_at,
        "n_groups": n_groups,
        "expression": expression,
    }
    for n, stats in head_stats.items():
        summary[f"{n}_sharpe"] = stats["sharpe"]
        summary[f"{n}_return"] = stats["return_"]
        summary[f"{n}_turnover"] = stats["turnover"]

    if show_figures:
        for figure in figures.values():
            figure.show()

    return {"summary": summary, "figures": figures, "pnl": pnl_series, "ic": ic_series, "rank_ic": ric_series}


def show_figures(result):
    for figure in result["figures"].values():
        figure.show()
