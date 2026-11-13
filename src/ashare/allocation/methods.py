"""配置方法：每个方法把「一个调仓日能看到的数据」变成权重。

方法只看 `Window`：它只含第 t 个调仓日（成交日）之前、即 t−1 日收盘及以前的收益，
由 `ashare.allocation.allocate` 截好再交进来，方法本身拿不到更晚的数据。

基础配置：`FixedWeights`、`EqualWeight`、`InverseVolatility`、`RiskParity`、`MinVariance`、
`MeanVariance`。叠加层：`TargetVolatility`（加杠杆或降仓到目标波动）、`TrendFilter`（价格在
均线下方的资产权重归零转现金），叠加层可以套任意配置，也可以互相套。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np

from .covariance import Covariance, sample_volatility
from .riskparity import risk_contributions, solve_risk_parity

ANNUAL = 252
ZERO_VOL = 1e-12


@dataclass
class Window:
    """一个调仓日的输入。

    returns：L×n，所有可用资产在回看窗口内的日收益（列已按 `available` 筛过）。
    history：t×N 截至 t−1 日的全部日收益（趋势过滤用来算价格和均线）。
    available：长度 N 的布尔数组；columns：可用资产在全部资产中的下标。
    covariance：协方差估计方法；previous：上一次目标权重（长度 N，首次为 0）。
    """

    returns: np.ndarray
    history: np.ndarray
    available: np.ndarray
    columns: np.ndarray
    covariance: Covariance
    previous: np.ndarray
    notes: dict = field(default_factory=dict)

    @cached_property
    def sigma(self):
        sigma, meta = self.covariance.estimate(self.returns)
        self.notes.update(covariance=meta)
        return sigma

    @cached_property
    def volatility(self):
        return sample_volatility(self.returns)

    def full(self, sub):
        """可用资产上的权重 → 长度 N（不可用为 0）。"""
        out = np.zeros(len(self.available))
        out[self.columns] = sub
        return out

    def zero_vol(self):
        vol = self.volatility
        return ~(np.isfinite(vol) & (vol > ZERO_VOL))


class Method:
    """配置方法的基类：`weights(window)` 返回长度 N 的权重。"""

    name = "method"

    def weights(self, window: Window) -> np.ndarray:  # pragma: no cover - 接口
        raise NotImplementedError

    def describe(self) -> dict:
        return {"method": self.name}


def _exclude_zero_vol(window, solver):
    """对非零波动的资产求解，零波动资产权重为 0 并记进诊断；全部为零波动时等权。"""
    zero = window.zero_vol()
    if zero.any():
        window.notes["zero_volatility"] = window.columns[zero].tolist()
    keep = ~zero
    sub = np.zeros(len(window.columns))
    if not keep.any():
        window.notes["fallback"] = "全部可用资产窗口内波动为 0，退回等权"
        sub[:] = 1.0 / len(sub)
        return sub
    sub[keep] = solver(keep)
    return sub


@dataclass(frozen=True)
class FixedWeights(Method):
    """固定比例。weights：{代码: 比例} 或长度 N 的数组（按资产顺序），非负、和不超过 1。

    某资产当天不可用时，它的比例转为现金（`redistribute=True` 时按其余资产的比例摊给它们）。
    再平衡带由 `allocate(band=...)` 控制，对任何方法都适用。
    """

    weights_spec: object
    redistribute: bool = False
    name = "fixed"

    def resolve(self, symbols):
        spec = self.weights_spec
        if isinstance(spec, dict):
            unknown = set(spec) - set(symbols)
            if unknown:
                raise ValueError(f"FixedWeights: 未知资产 {sorted(unknown)}")
            out = np.array([float(spec.get(s, 0.0)) for s in symbols])
        else:
            out = np.asarray(spec, dtype=float)
            if out.shape != (len(symbols),):
                raise ValueError("FixedWeights: 数组长度必须等于资产数")
        if not np.isfinite(out).all() or (out < 0).any() or out.sum() > 1 + 1e-12:
            raise ValueError("FixedWeights: 比例需非负、有限、和不超过 1")
        return out

    def weights(self, window):
        target = window.notes["fixed_target"]
        out = np.where(window.available, target, 0.0)
        if self.redistribute and out.sum() > 0:
            out = out * target.sum() / out.sum()
        return out

    def describe(self):
        spec = self.weights_spec
        return {
            "method": self.name,
            "weights": dict(spec) if isinstance(spec, dict) else list(map(float, spec)),
            "redistribute": self.redistribute,
        }


@dataclass(frozen=True)
class EqualWeight(Method):
    """可用资产等权。"""

    name = "equal"

    def weights(self, window):
        n = len(window.columns)
        return window.full(np.full(n, 1.0 / n))


@dataclass(frozen=True)
class InverseVolatility(Method):
    """权重与窗口内波动率（协方差对角线）成反比；零波动资产剔除。"""

    name = "inverse_volatility"

    def weights(self, window):
        def solve(keep):
            vol = np.sqrt(np.diag(window.sigma))[keep]
            w = 1.0 / vol
            return w / w.sum()

        return window.full(_exclude_zero_vol(window, solve))


@dataclass(frozen=True)
class RiskParity(Method):
    """等风险贡献；budget 为风险预算（{代码: 份额} 或长度 N 数组，按可用资产重新归一）。

    诊断里的 ``residual`` 是风险贡献占比与预算的最大绝对差。
    """

    budget: object = None
    tol: float = 1e-11
    max_iter: int = 100
    name = "risk_parity"

    def weights(self, window):
        budget = window.notes.get("budget_full")

        def solve(keep):
            sigma = window.sigma[np.ix_(keep, keep)]
            b = None if budget is None else budget[window.columns][keep]
            if b is not None and b.sum() <= 0:
                raise ValueError("RiskParity: 可用资产的风险预算全为 0")
            w, info = solve_risk_parity(sigma, b, tol=self.tol, max_iter=self.max_iter)
            window.notes.update(
                risk_parity=info, risk_shares=risk_contributions(w, sigma).tolist()
            )
            return w

        return window.full(_exclude_zero_vol(window, solve))

    def describe(self):
        b = self.budget
        if isinstance(b, dict):
            b = {k: float(v) for k, v in b.items()}
        elif b is not None:
            b = list(map(float, b))
        return {"method": self.name, "budget": b}


def _optimize(window, alpha, risk_aversion, max_weight):
    """长仓、满仓的均值方差，交给 `ashare.portfolio.optimize`（默认后端）。"""
    from ashare.portfolio import Constraints, PortfolioProblem, optimize
    from ashare.risk import RiskModel

    sigma = window.sigma
    values, vectors = np.linalg.eigh(sigma)
    values = np.maximum(values, 0.0)
    b = vectors * np.sqrt(values)
    n = len(sigma)
    model = RiskModel(
        tuple(str(c) for c in window.columns),
        b,
        tuple(f"pc{i}" for i in range(n)),
        np.eye(n),
        np.zeros(n),
    )
    cap = 1.0 if max_weight is None else max(float(max_weight), 1.0 / n)
    problem = PortfolioProblem(
        alpha=alpha,
        risk=model,
        benchmark=np.zeros(n),
        current=np.zeros(n),
        risk_aversion=risk_aversion,
        constraints=Constraints(invested=1.0, long_only=True, max_weight=cap),
    )
    result = optimize(problem)
    window.notes["optimizer"] = dict(status=result.status, solver=result.solver)
    if not result.success:
        raise RuntimeError(f"组合优化没有求到最优解：{result.status}")
    w = np.clip(result.weights, 0.0, None)
    return w / w.sum()


@dataclass(frozen=True)
class MinVariance(Method):
    """长仓满仓的最小方差（复用 `ashare.portfolio` 的优化器）；max_weight 为单资产上限。"""

    max_weight: float | None = None
    name = "min_variance"

    def weights(self, window):
        n = len(window.columns)
        return window.full(_optimize(window, np.zeros(n), 1.0, self.max_weight))

    def describe(self):
        return {"method": self.name, "max_weight": self.max_weight}


@dataclass(frozen=True)
class MeanVariance(Method):
    """长仓满仓的均值方差：max μᵀw − λ wᵀΣw（日频）。

    μ 默认取窗口内日收益的均值（`expected="mean"`），也可以给一个函数
    ``f(returns L×n) -> 长度 n``。历史均值作为预期收益噪声很大，仅作对照。
    """

    risk_aversion: float = 5.0
    expected: object = "mean"
    max_weight: float | None = None
    name = "mean_variance"

    def __post_init__(self):
        if not (np.isfinite(self.risk_aversion) and self.risk_aversion > 0):
            raise ValueError("risk_aversion: 必须为正")

    def weights(self, window):
        if self.expected == "mean":
            mu = np.nanmean(window.returns, axis=0)
        elif callable(self.expected):
            mu = np.asarray(self.expected(window.returns), dtype=float)
        else:
            raise ValueError("expected: 'mean' 或函数")
        mu = np.nan_to_num(mu)
        return window.full(_optimize(window, mu, self.risk_aversion, self.max_weight))

    def describe(self):
        return {
            "method": self.name,
            "risk_aversion": self.risk_aversion,
            "expected": self.expected if isinstance(self.expected, str) else "custom",
            "max_weight": self.max_weight,
        }


@dataclass(frozen=True)
class TargetVolatility(Method):
    """把基础配置整体放大或缩小到目标年化波动：scale = min(max_leverage, target / σ̂)。

    σ̂ 用同一窗口的协方差估计（年化按 252 日）。scale < 1 的部分是现金；max_leverage > 1
    时权重和可以超过 1，交给 `panelbt` 的现金账户会被拒绝，需要保证金账户。
    """

    base: Method
    target: float = 0.10
    max_leverage: float = 1.0
    name = "target_volatility"

    def __post_init__(self):
        if not (np.isfinite(self.target) and self.target > 0):
            raise ValueError("target: 必须为正的年化波动")
        if not (np.isfinite(self.max_leverage) and self.max_leverage > 0):
            raise ValueError("max_leverage: 必须为正")

    def weights(self, window):
        w = self.base.weights(window)
        sub = w[window.columns]
        vol = float(np.sqrt(max(sub @ window.sigma @ sub, 0.0) * ANNUAL))
        if vol > 0:
            scale = min(self.max_leverage, self.target / vol)
            window.notes["target_volatility"] = dict(ex_ante=vol, scale=scale)
        else:
            # 事前波动为 0（全部资产窗口内不动、或基础配置全是现金）：没法按波动缩放，
            # 不加杠杆，原样用基础配置（不超过 1）。
            scale = min(1.0, self.max_leverage)
            window.notes["target_volatility"] = dict(
                ex_ante=vol, scale=scale, fallback="事前波动为 0，用基础配置"
            )
        return w * scale

    def describe(self):
        return {
            "method": self.name,
            "target": self.target,
            "max_leverage": self.max_leverage,
            "base": self.base.describe(),
        }


@dataclass(frozen=True)
class TrendFilter(Method):
    """趋势过滤：t−1 日收盘价低于其 window 日均线的资产，权重归零转为现金。

    价格由日收益连乘得到（缺失收益当作 0，即价格不变）。历史不足 window 日的资产不过滤。
    `redistribute=True` 时把被过滤的权重按比例摊给其余资产（不转现金）。
    """

    base: Method
    window: int = 200
    redistribute: bool = False
    name = "trend_filter"

    def __post_init__(self):
        if not isinstance(self.window, int) or self.window < 2:
            raise ValueError("window: 至少为 2 的整数")

    def weights(self, window):
        w = self.base.weights(window)
        history = window.history
        off = []
        if len(history) >= self.window:
            price = np.cumprod(1.0 + np.nan_to_num(history), axis=0)
            last = price[-1]
            average = price[-self.window :].mean(axis=0)
            counts = np.isfinite(history[-self.window :]).sum(axis=0)
            below = (last < average) & (counts >= self.window // 2)
            off = [int(j) for j in np.flatnonzero(below & (w != 0))]
        total = w.sum()
        if off:
            w = w.copy()
            w[off] = 0.0
            if self.redistribute and w.sum() > 0:
                w *= total / w.sum()
        window.notes["trend_off"] = off
        return w

    def describe(self):
        return {
            "method": self.name,
            "window": self.window,
            "redistribute": self.redistribute,
            "base": self.base.describe(),
        }


def walk(method):
    """方法及其嵌套的基础方法（由外到内）。"""
    while method is not None:
        yield method
        method = getattr(method, "base", None)
