"""按公开方法实现的基本面因子模型；不是 MSCI 模型的数值复刻。"""

from dataclasses import dataclass

import numpy as np

from .engine import ew_covariance, ew_variance, factor_covariance
from .kernels import (
    exp_weights,
    psd,
)
from .window import WindowVariances

STYLE_NAMES = (
    "size",
    "beta",
    "momentum",
    "residual_volatility",
    "nonlinear_size",
    "liquidity",
    "book_to_price",
    "earnings_yield",
)


@dataclass(frozen=True)
class FundamentalConfig:
    industry_codes: tuple[int, ...] = tuple(range(31))
    min_listing_days: int = 60
    min_descriptor_days: int = 20
    winsor_mad: float = 5.0
    beta_window: int = 252
    beta_half_life: float = 63.0
    momentum_window: int = 504
    momentum_skip: int = 21
    momentum_half_life: float = 126.0
    vol_half_life: float = 84.0
    corr_half_life: float = 504.0
    vol_lags: int = 5
    corr_lags: int = 2
    specific_half_life: float = 84.0
    specific_corr_half_life: float = 252.0
    specific_lags: int = 5
    history: int = 1008
    structural_days: int = 252
    shrinkage: float = 0.1
    regime_half_life: float = 42.0
    regime: bool = True
    eigen_simulations: int = 0
    eigen_seed: int = 871
    eigen_scale: float = 1.0
    variance_floor: float = 1e-10
    cold_specific_variance: float = 0.02**2
    cold_factor_variance: float = 0.005**2
    # 特质风险三个指数加权方差：逐日递推（相对误差 ≤1e-12）或每天从头算。
    specific_incremental: bool = True
    specific_reanchor_days: int = 252

    def __post_init__(self):
        if (
            len(set(self.industry_codes)) != len(self.industry_codes)
            or -1 in self.industry_codes
        ):
            raise ValueError("industry_codes: 需要唯一编码，-1 保留给未知行业")
        if (
            self.min_listing_days < 1
            or self.min_descriptor_days < 2
            or self.history < 2
            or self.beta_window < 2
            or self.momentum_window < 1
            or self.momentum_skip < 0
            or self.structural_days < 1
            or not isinstance(self.specific_reanchor_days, int)
            or self.specific_reanchor_days < 1
        ):
            raise ValueError("样本窗口过短")
        if not isinstance(self.specific_incremental, bool) or isinstance(
            self.specific_reanchor_days, bool
        ):
            raise ValueError(
                "specific_incremental 须为 bool，specific_reanchor_days 须为整数"
            )
        for name in (
            "beta_half_life",
            "momentum_half_life",
            "vol_half_life",
            "corr_half_life",
            "specific_half_life",
            "specific_corr_half_life",
            "regime_half_life",
            "winsor_mad",
            "variance_floor",
            "cold_specific_variance",
            "cold_factor_variance",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name}: 必须为正有限值")
        if (
            min(
                self.vol_lags,
                self.corr_lags,
                self.specific_lags,
                self.eigen_simulations,
                self.shrinkage,
            )
            < 0
        ):
            raise ValueError("滞后、模拟次数、收缩参数不能为负")
        if (
            not np.isfinite(self.shrinkage)
            or not np.isfinite(self.eigen_scale)
            or self.eigen_scale < 0
        ):
            raise ValueError("收缩和特征调整参数必须为非负有限数")


def standardize(values, cap, estimation, industries, size=None, limit=5.0):
    """MAD 截断后，以行业和市值拟合缺失值；最后作估计池标准化。"""
    x = np.asarray(values, dtype=float).copy()
    est = estimation & np.isfinite(cap) & (cap > 0)
    observed = est & np.isfinite(x)
    if not observed.any():
        return np.zeros(len(x))
    med = np.median(x[observed])
    mad = 1.4826 * np.median(np.abs(x[observed] - med))
    if mad > 0:
        x = np.clip(x, med - limit * mad, med + limit * mad)
    missing = ~np.isfinite(x)
    if missing.any():
        design = industries if size is None else np.column_stack((industries, size))
        if observed.sum() >= 2:
            coef = np.linalg.lstsq(design[observed], x[observed], rcond=1e-12)[0]
            x[missing] = design[missing] @ coef
            # 未见过的行业不能把零系数当作零原始暴露；回退到全池的 Size 回归。
            known_groups = industries[observed].any(axis=0)
            unsupported = missing & ~industries[:, known_groups].any(axis=1)
            if unsupported.any():
                fallback = (
                    np.ones((len(x), 1))
                    if size is None
                    else np.column_stack((np.ones(len(x)), size))
                )
                beta = np.linalg.lstsq(fallback[observed], x[observed], rcond=1e-12)[0]
                x[unsupported] = fallback[unsupported] @ beta
        else:
            x[missing] = med
    mean = np.average(x[est], weights=cap[est])
    sd = np.std(x[est])
    return (x - mean) / sd if sd > 1e-12 else np.zeros_like(x)


def orthogonalize(values, controls, cap, estimation):
    design = np.column_stack((np.ones(len(values)), controls))
    weights = np.sqrt(cap[estimation])
    weights /= max(weights.mean(), 1e-30)
    coef = np.linalg.lstsq(
        design[estimation] * np.sqrt(weights[:, None]),
        values[estimation] * np.sqrt(weights),
        rcond=1e-12,
    )[0]
    return values - design @ coef


def _log_excess(source, returns, start, end):
    """简单超额收益还原为 log(1+r)−log(1+rf)，不把两种定义混用。"""
    excess = returns[start:end]
    try:
        rf = source["risk_free"][start:end]
    except KeyError:
        rf = 0.0
    return np.log1p(np.maximum(excess + rf, -0.999999999)) - np.log1p(rf)


def exposures_at(
    source,
    returns,
    day,
    config,
    external=None,
    external_names=(),
    *,
    listing_ages=None,
    rolling=None,
):
    """day 收盘信息生成 B_day；外部暴露调用方须保证同样时点。"""
    c = config
    cap = np.asarray(source["total_mv"][day], dtype=float)
    raw_size = np.log(np.where(np.isfinite(cap) & (cap > 0), cap, np.nan))
    cap = np.where(np.isfinite(cap) & (cap > 0), cap, 1.0)
    listed = np.asarray(source["is_listed"][: day + 1], dtype=bool)
    ages = (
        np.cumsum(listed, axis=0, dtype=np.int32)
        if listing_ages is None
        else listing_ages[: day + 1]
    )
    est = (
        listed[-1]
        & (ages[-1] >= c.min_listing_days)
        & ~source["is_suspended"][day].astype(bool)
    )
    est &= np.isfinite(source["total_mv"][day]) & (source["total_mv"][day] > 0)
    codes = np.asarray(source["industry"][day])
    inds = np.column_stack(
        [codes == code for code in c.industry_codes]
        + [~np.isin(codes, c.industry_codes)]
    )
    # 冷启动时只用于暴露标度，正式收益回归仍使用 est。
    scale_est = (
        est
        if est.any()
        else (listed[-1] if listed[-1].any() else np.ones(len(cap), bool))
    )

    def norm(v, size=None):
        return standardize(v, cap, scale_est, inds, size, c.winsor_mad)

    size = norm(raw_size)
    start = max(1, day - c.beta_window + 1)
    hist = returns[start : day + 1]
    beta = np.full(len(cap), np.nan)
    hsigma = beta.copy()
    dastd = beta.copy()
    cmra = beta.copy()
    momentum = beta.copy()
    if rolling is None and len(hist) >= 2:
        prev_cap = np.asarray(source["total_mv"][start - 1 : day], dtype=float)
        mvalid = np.isfinite(hist) & np.isfinite(prev_cap) & (prev_cap > 0)
        prev_listed = source["is_listed"][start - 1 : day].astype(bool)
        prev_suspended = source["is_suspended"][start - 1 : day].astype(bool)
        mvalid &= (
            prev_listed
            & ~prev_suspended
            & (ages[start - 1 : day] >= c.min_listing_days)
        )
        mw = np.where(mvalid, prev_cap, 0)
        denom = mw.sum(axis=1)
        market = np.divide(
            (mw * np.nan_to_num(hist)).sum(axis=1),
            denom,
            out=np.zeros(len(hist)),
            where=denom > 0,
        )
        valid = np.isfinite(hist) & (denom[:, None] > 0)
        weights = exp_weights(len(hist), c.beta_half_life)[:, None] * valid
        weights /= np.maximum(weights.sum(axis=0), 1e-30)
        y = np.nan_to_num(hist)
        mx = np.sum(weights * market[:, None], axis=0)
        my = np.sum(weights * y, axis=0)
        dx, dy = market[:, None] - mx, y - my
        beta = np.sum(weights * dx * dy, axis=0) / np.maximum(
            np.sum(weights * dx * dx, axis=0), 1e-30
        )
        hsigma = np.sqrt(np.sum(weights * (dy - dx * beta) ** 2, axis=0))
        enough = valid.sum(axis=0) >= c.min_descriptor_days
        for v in (beta, hsigma):
            v[~enough] = np.nan
    raw = None if rolling is None else rolling.at(day)
    if raw is not None:
        beta, hsigma, dastd, momentum, cmra = (
            raw[name] for name in ("beta", "hsigma", "dastd", "momentum", "cmra")
        )
    vol_start = max(1, day - 251)
    vol_hist = returns[vol_start : day + 1]
    if len(vol_hist) >= 2:
        if raw is None:
            dastd = np.sqrt(ew_variance(vol_hist, 42))
            dastd[np.isfinite(vol_hist).sum(axis=0) < c.min_descriptor_days] = np.nan
        if raw is None and len(vol_hist) >= 252:
            logs = _log_excess(source, returns, vol_start, day + 1)
            cumulative = np.cumsum(logs[::-1], axis=0)[20::21]
            # 累积 log 收益范围在大跌时也有定义；公开 A4 的二次 log 另见笔记。
            cmra = np.max(cumulative, axis=0) - np.min(cumulative, axis=0)
    end = day + 1 - c.momentum_skip
    if raw is None and end > 1:
        first = max(1, end - c.momentum_window)
        mh = returns[first:end]
        mw = exp_weights(len(mh), c.momentum_half_life)[:, None]
        valid = np.isfinite(mh)
        momentum = np.sum(
            np.where(valid, _log_excess(source, returns, first, end), 0) * mw, axis=0
        )
        momentum /= np.maximum(np.sum(mw * valid, axis=0), 1e-30)
        momentum[valid.sum(axis=0) < c.min_descriptor_days] = np.nan
    beta, momentum = norm(beta, size), norm(momentum, size)
    residual = (
        0.74 * norm(dastd, size) + 0.16 * norm(cmra, size) + 0.10 * norm(hsigma, size)
    )
    residual = norm(
        orthogonalize(residual, np.column_stack((size, beta)), cap, scale_est), size
    )
    nonlinear = norm(orthogonalize(size**3, size, cap, scale_est), size)
    if raw is None:
        turnover_start = max(0, day - 251)
        turnover = np.asarray(
            source["turnover_rate"][turnover_start : day + 1], dtype=float
        )
        turnover = np.where(
            listed[turnover_start : day + 1] & np.isfinite(turnover) & (turnover >= 0),
            turnover,
            np.nan,
        )
    liquidity = np.zeros(len(cap))
    for j, (window, weight) in enumerate(((21, 0.35), (63, 0.35), (252, 0.30))):
        if raw is None:
            h = turnover[-window:]
            count = np.isfinite(h).sum(axis=0)
            total_sum = np.nansum(h, axis=0)
        else:
            total_sum, count = raw["turnover_sum"][j], raw["turnover_count"][j]
        # 完整历史等于公开公式的月/季/年分母；短史按可观测交易日折算。
        total = total_sum * 21 / np.maximum(count, 1)
        v = np.log(np.maximum(total, 1e-30))
        v[(count < min(c.min_descriptor_days, window)) | (total <= 0)] = np.nan
        liquidity += weight * norm(v, size)
    liquidity = norm(orthogonalize(liquidity, size, cap, scale_est), size)

    def inverse(field):
        v = np.asarray(source[field][day], dtype=float)
        return norm(
            np.divide(
                1, v, out=np.full_like(v, np.nan), where=np.isfinite(v) & (v != 0)
            ),
            size,
        )

    style = np.column_stack(
        (
            size,
            beta,
            momentum,
            residual,
            nonlinear,
            liquidity,
            inverse("pb"),
            inverse("pe_ttm"),
        )
    )
    names = (
        ("country",)
        + tuple(f"industry_{code}" for code in c.industry_codes)
        + ("industry_unknown",)
        + STYLE_NAMES
    )
    b = np.column_stack((np.ones(len(cap)), inds, style))
    if external is not None:
        ex = np.asarray(external[day], dtype=float)
        if ex.shape != (len(cap), len(external_names)) or len(
            set(external_names)
        ) != len(external_names):
            raise ValueError("external: 形状或名称错误")
        b = np.column_stack((b, *[norm(ex[:, j], size) for j in range(ex.shape[1])]))
        names += tuple(external_names)
    return b, names, est, cap


def eigen_adjustment(
    covariance, observations, half_life, simulations=100, seed=871, scale=1.0
):
    """USE4 附录 B 的高斯 Monte Carlo 调整，固定种子保证可重算。"""
    eig, vec = np.linalg.eigh(covariance)
    root = vec * np.sqrt(np.maximum(eig, 0))
    rng = np.random.default_rng(seed)
    ratios = np.zeros(len(eig))
    for _ in range(simulations):
        sample = rng.normal(size=(observations, len(eig))) @ root.T
        estimate = ew_covariance(sample, half_life)
        values, vectors = np.linalg.eigh(estimate)
        truth = np.sum(vectors * (covariance @ vectors), axis=0)
        ratios += np.divide(
            truth, values, out=np.ones_like(values), where=values > 1e-15
        )
    multiplier = 1 + scale * (np.sqrt(ratios / simulations) - 1)
    return psd((vec * (eig * multiplier**2)) @ vec.T)


def specific_variance(residuals, b, cap, config, estimation=None):
    """residuals 是 T×N 残差窗口；也可以是已经算好的 `WindowVariances`（逐日分块累加）。"""
    c = config
    if isinstance(residuals, WindowVariances):
        base, corr0, corrnw, count = residuals
    else:
        base = ew_variance(residuals, c.specific_half_life)
        corr0 = ew_variance(residuals, c.specific_corr_half_life)
        corrnw = ew_variance(
            residuals,
            c.specific_corr_half_life,
            min(c.specific_lags, len(residuals) - 1),
        )
        count = np.isfinite(residuals).sum(axis=0)
    adjustment = np.divide(
        corrnw, corr0, out=np.ones_like(corr0), where=corr0 > c.variance_floor
    )
    vol = np.sqrt(np.maximum(base * adjustment, c.variance_floor))
    estimation = (
        (count > 0) if estimation is None else np.asarray(estimation, dtype=bool)
    )
    estimation = estimation & np.isfinite(cap) & (cap > 0)
    reliable = estimation & (count >= c.structural_days) & (base > c.variance_floor)
    structural = np.full(len(cap), np.sqrt(c.cold_specific_variance))
    if reliable.sum() >= 2:
        # 对 log 特质波动做截面回归，以指数残差的均值校正反变换偏差。
        rootw = np.sqrt(np.sqrt(cap[reliable]) / np.mean(np.sqrt(cap[reliable])))
        coef = np.linalg.lstsq(
            b[reliable] * rootw[:, None], np.log(vol[reliable]) * rootw, rcond=1e-12
        )[0]
        fit = b @ coef
        correction = np.mean(np.exp(np.log(vol[reliable]) - fit[reliable]))
        structural = correction * np.exp(np.clip(fit, -20, 5))
    gamma = np.minimum(count / c.structural_days, 1.0)
    gamma[base <= c.variance_floor] = 0.0
    vol = gamma * vol + (1 - gamma) * structural
    # USE4 5.6–5.9：市值分位组内收缩的是波动，不是方差。
    eligible = np.flatnonzero(estimation)
    ordered = eligible[np.argsort(cap[eligible], kind="stable")]
    groups = np.array_split(ordered, min(10, len(ordered))) if len(ordered) else []
    shrunk = vol.copy()
    for ix in groups:
        mean = np.average(vol[ix], weights=cap[ix])
        spread = np.sqrt(np.mean((vol[ix] - mean) ** 2))
        distance = c.shrinkage * np.abs(vol[ix] - mean)
        weight = distance / np.maximum(spread + distance, 1e-30)
        shrunk[ix] = weight * mean + (1 - weight) * vol[ix]
    return np.maximum(shrunk**2, c.variance_floor), base


def estimate_covariances(factor_returns, residuals, b, cap, config, estimation=None):
    c = config
    f = factor_covariance(
        factor_returns,
        c.vol_half_life,
        c.corr_half_life,
        min(c.vol_lags, len(factor_returns) - 1),
        min(c.corr_lags, len(factor_returns) - 1),
    )
    if c.eigen_simulations:
        f = eigen_adjustment(
            f,
            len(factor_returns),
            c.vol_half_life,
            c.eigen_simulations,
            c.eigen_seed,
            c.eigen_scale,
        )
    d, base_d = specific_variance(residuals, b, cap, c, estimation)
    base_f = ew_variance(factor_returns, c.vol_half_life)
    return f, d, base_f, base_d


def cross_sectional_bias(realized, predicted_variance, weights=None):
    w = np.ones(len(realized)) if weights is None else np.asarray(weights)
    valid = (
        np.isfinite(realized)
        & np.isfinite(predicted_variance)
        & (predicted_variance > 1e-12)
        & np.isfinite(w)
        & (w > 0)
    )
    if not valid.any():
        return np.nan
    w = np.where(valid, w, 0)
    return float(
        np.sum(
            w
            * np.divide(
                realized**2,
                predicted_variance,
                out=np.zeros_like(realized),
                where=valid,
            )
        )
        / w.sum()
    )
