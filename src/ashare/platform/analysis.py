"""检验表、原始序列和绘图传输；回测数值全部来自 groupbt。"""

import base64
import math
import threading
import time
from statistics import NormalDist

import numpy as np
import panelops

from ashare.factor import Evaluator, groupbt
from ashare.factor.groupbt.api import _adjusted, _forward_return

from .config import THRESHOLDS as GATES
from .holdings import mean_days


class CutPanel:
    """在表达式求值之前截断，TEST 行不进入任何计算。"""

    def __init__(self, panel, stop):
        self.panel = panel
        self.stop = int(np.searchsorted(panel.dates, stop))
        self.dates = np.asarray(panel.dates)[: self.stop]
        self.symbols = panel.symbols
        self.version = getattr(panel, "version", "synthetic") + f":{stop}"

    @property
    def shape(self):
        return len(self.dates), len(self.symbols)

    def __getitem__(self, name):
        return self.panel[name][: self.stop]


def finite(value):
    return float(value) if value is not None and math.isfinite(value) else None


def stats(result):
    values = result.table().to_dicts()[-1]
    p = result.pnl["long_short"]
    valid = np.isfinite(p)
    n = int(valid.sum())
    sh, ret, turn = values["sharpe"], values["return"], values["turnover"]
    values.update(
        n=n,
        fitness=finite(sh * math.sqrt(abs(ret) / max(turn, 0.125)))
        if sh is not None
        else None,
        t=finite(sh * math.sqrt(n / 240)) if sh is not None else None,
    )
    return values


def evaluate_signal(expression, panel):
    """只在验证期末之前求值一次；所有开放算子均为因果运算。"""
    cut = CutPanel(panel, groupbt.VALIDATION[1])
    alpha = Evaluator().evaluate(expression, cut)
    alpha.flags.writeable = False
    return alpha


_IC_LOCK = threading.Lock()


def _ic_inputs(panel, cut, universe, start, trade_at):
    """覆盖率和 IC 里与信号无关的部分：股票池、前移一天的股票池、未来收益。

    平台的面板缓存（`PanelCache.diagnostics`，每个任务开始时清空）里按截断和设置只算一次，
    同一任务的其他信号直接复用；只读，防止被下游改写。没有缓存的面板照常现算。
    """
    cache = getattr(panel, "diagnostics", None)
    if cache is None:
        return _compute_ic_inputs(cut, universe, start, trade_at)
    key = ("ic-inputs", cut.stop, universe, start, trade_at)
    # 同一任务的变体并发启动：第一个算，其余等它算完直接用，不各算一遍。
    with _IC_LOCK:
        entry = cache.get(key)
        if entry is None:
            entry = cache[key] = [threading.Lock(), None]
    with entry[0]:
        if entry[1] is None:
            entry[1] = _compute_ic_inputs(cut, universe, start, trade_at)
    return entry[1]


def _compute_ic_inputs(cut, universe, start, trade_at):
    pool = groupbt.universe_mask(cut, universe, start)
    shifted_pool = np.zeros_like(pool)
    shifted_pool[1:] = pool[:-1]
    forward = _forward_return(_adjusted(cut, trade_at))
    for array in (pool, shifted_pool, forward):
        array.flags.writeable = False
    return pool, shifted_pool, forward


def run_variant(
    expression,
    panel,
    settings,
    progress=lambda phase: None,
    *,
    signal=None,
    timings=None,
    diagnostics=None,
):
    """唯一计算入口，窗口为代码常量，调用者没有窗口参数。"""
    arrays, summaries = {}, {}
    if signal is None:
        progress("evaluate")
        signal = evaluate_signal(expression, panel)
    for part, window in (("is", groupbt.IS), ("validation", groupbt.VALIDATION)):
        progress(part)
        cut = CutPanel(panel, window[1])
        alpha = signal[: cut.stop]
        options = {key: value for key, value in settings.items() if key != "benchmark"}
        if part == "validation":
            options["sub_universes"] = False
        started = time.perf_counter()
        result = groupbt.run(alpha, cut, **options, window=window, record_holdings=True)
        if timings is not None:
            timings[f"{part}.backtest"] = time.perf_counter() - started
        started = time.perf_counter()
        summary = stats(result)
        summary["holding_days"], summary["short_holding_days"] = mean_days(
            result.holdings, len(result.dates)
        )
        for key, value in result.holdings.items():
            arrays[f"{part}.holdings.{key}"] = value
        # 验证的预热历史保留在计算里，界面只接收窗口内日期。
        keep = np.asarray(result.dates) >= window[0]
        dates = np.asarray(result.dates)[keep]
        if not np.isfinite(result.pnl["long_short"]).any():
            raise ValueError("所选研究窗口没有有效收益")
        if diagnostics is not None and part in diagnostics:
            coverage, ic = diagnostics[part]
        else:
            pool, shifted_pool, forward = _ic_inputs(
                panel, cut, settings["universe"], window[0], settings["trade_at"]
            )
            count = pool.sum(axis=1)
            covered = (pool & np.isfinite(alpha)).sum(axis=1)
            coverage = np.divide(
                covered, count, out=np.full(len(count), np.nan), where=count > 0
            )
            shifted = np.full_like(alpha, np.nan)
            shifted[1:] = alpha[:-1]
            ic = panelops.rank_ic(shifted, forward, mask=shifted_pool)
            if diagnostics is not None:
                diagnostics[part] = coverage, ic
        summary["ic"] = finite(np.nanmean(ic[keep]))
        summary["coverage"] = finite(coverage[-1])
        summary["exposure"] = {k: finite(v) for k, v in result.exposure_mean().items()}
        summary["periods"] = result.table().to_dicts()[:-1]
        summary["years"] = result.table(by="year").to_dicts()[:-1]
        summary["subs"] = {k: stats(v) for k, v in result.subs.items()}
        summaries[part] = summary
        arrays[f"{part}.dates"] = dates
        arrays[f"{part}.groups"] = result.group_pnl[:, keep]
        arrays[f"{part}.ic"] = ic[keep]
        arrays[f"{part}.coverage"] = coverage[keep]
        for name, series in result.pnl.items():
            arrays[f"{part}.pnl.{name}"] = series[keep]
        for name, series in result.turnover.items():
            arrays[f"{part}.turnover.{name}"] = series[keep]
        for name, series in result.exposure.items():
            arrays[f"{part}.exposure.{name}"] = series[keep]
        for name, sub in result.subs.items():
            arrays[f"{part}.sub.{name}"] = sub.pnl["long_short"][keep]
        if timings is not None:
            timings[f"{part}.statistics"] = time.perf_counter() - started
        del alpha
    return summaries, arrays


def checks(summary, trials, correlations=()):
    a, b = summary["is"], summary["validation"]
    gate = NormalDist().inv_cdf(1 - GATES["alpha"] / (2 * max(1, trials)))
    ratio = (
        b["sharpe"] / a["sharpe"] if a["sharpe"] and b["sharpe"] is not None else None
    )
    sub = a["subs"].get("TOP800", {}).get("sharpe")
    corr = correlations[0] if correlations else None
    years = [y for y in a["years"] if y["return"] is not None]
    positive = sum(y["return"] > 0 for y in years)
    exposures = {k: v for k, v in a["exposure"].items() if v is not None}
    largest = max(exposures, key=lambda k: abs(exposures[k]), default=None)
    rows = []

    def add(key, label, value, threshold, passed, *, hint=False):
        rows.append(
            dict(
                key=key,
                label=label,
                value=finite(value),
                threshold=threshold,
                status="hint"
                if hint or value is None
                else "pass"
                if passed
                else "fail",
            )
        )

    def minimum(key, label, value, threshold):
        add(
            key,
            label,
            value,
            f"≥ {threshold:.2f}",
            value is not None and value >= threshold,
        )

    minimum("sharpe", "Sharpe（样本内）", a["sharpe"], GATES["sharpe"])
    minimum("fitness", "Fitness（样本内）", a["fitness"], GATES["fitness"])
    minimum("t", f"t 值（已试 {trials} 个，Bonferroni）", a["t"], gate)
    minimum(
        "validation_ratio", "验证 Sharpe ÷ 样本内", ratio, GATES["validation_ratio"]
    )
    minimum("sub_sharpe", "子池 TOP800 Sharpe", sub, GATES["sub_sharpe"])
    add(
        "correlation",
        "库内最大相关（" + (corr["name"] if corr else "无独立因子") + "）",
        abs(corr["value"]) if corr else None,
        f"≤ {GATES['correlation']:.2f}",
        bool(corr and abs(corr["value"]) <= GATES["correlation"]),
    )
    turn = a["turnover"]
    add(
        "turnover",
        "日均换手",
        turn,
        f"{GATES['turnover_min']:.0%}–{GATES['turnover_max']:.0%}",
        turn is not None and GATES["turnover_min"] <= turn <= GATES["turnover_max"],
    )
    coverage = a["coverage"]
    add(
        "coverage",
        "覆盖率",
        coverage,
        f"≥ {GATES['coverage']:.0%}",
        coverage is not None and coverage >= GATES["coverage"],
    )
    add(
        "exposure",
        f"风格暴露绝对值最大（{largest or '—'}）",
        abs(exposures[largest]) if largest else None,
        f"≤ {GATES['exposure']:.2f}",
        False,
        hint=True,
    )
    add(
        "positive_years",
        f"收益为正的年份（{positive}/{len(years)}）",
        positive / len(years) if years else None,
        f"≥ {GATES['positive_years']:.0%}",
        bool(years and positive / len(years) >= GATES["positive_years"]),
    )
    return rows


def packed(array):
    a = np.asarray(array, dtype="<f4")
    return {"shape": list(a.shape), "data": base64.b64encode(a.tobytes()).decode()}


def plot_payload(arrays, view="returns"):
    """每次仅发送当前视图需要的序列；不发送证券面板。"""
    parts = []
    for part in ("is", "validation"):
        dates = arrays[f"{part}.dates"].tolist()
        names = {
            "returns": ["pnl.long_short"]
            + (["pnl.weights"] if f"{part}.pnl.weights" in arrays else []),
            "groups": ["groups"],
            "long": ["pnl.long"],
            "ic": ["ic"],
            "subs": [
                k[len(part) + 1 :] for k in arrays if k.startswith(f"{part}.sub.")
            ],
        }[view]
        parts.append(
            {
                "part": part,
                "dates": dates,
                "series": {
                    name: packed(
                        arrays[f"{part}.groups"][0]
                        if name == "pnl.long"
                        else arrays[f"{part}.{name}"]
                    )
                    for name in names
                },
            }
        )
    return parts
