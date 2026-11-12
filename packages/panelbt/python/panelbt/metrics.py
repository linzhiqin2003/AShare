"""NumPy-only period-return analytics with explicit annualization conventions."""

from __future__ import annotations

import numpy as np


def _drawdown(returns):
    wealth = np.r_[1.0, np.cumprod(1 + returns)]
    peaks = np.maximum.accumulate(wealth)
    draw = np.divide(wealth, peaks, out=np.ones_like(wealth), where=peaks != 0) - 1
    duration = longest = 0
    for x in draw[1:]:
        duration = duration + 1 if x < 0 else 0
        longest = max(longest, duration)
    return float(-draw.min()), longest


def _ratio(a, b):
    return float(a / b) if b else float("nan")


def calculate(
    returns,
    *,
    periods_per_year=252,
    risk_free=0.0,
    benchmark=None,
    turnover=None,
    cost_ratio=None,
    allow_bankruptcy=False,
):
    """Summarize simple returns (sample volatility, per-period risk-free rate).

    Sortino uses sqrt(mean(min(r-rf, 0)**2)). Alpha is annualized arithmetic
    CAPM intercept. Excess annual return and excess drawdown use relative wealth.
    Zero-denominator ratios are NaN. Drawdown duration is measured in periods,
    including an unrecovered final drawdown. The initial wealth is a peak.
    """
    r = np.asarray(returns, dtype=float)
    if (
        r.ndim != 1
        or not len(r)
        or not np.isfinite(r).all()
        or (not allow_bankruptcy and (r < -1).any())
    ):
        raise ValueError("returns: expected a nonempty finite vector >= -1")
    if (
        not np.isfinite(periods_per_year)
        or periods_per_year <= 0
        or not np.isfinite(risk_free)
    ):
        raise ValueError("invalid annualization or risk_free")
    p = float(periods_per_year)
    growth = float(np.prod(1 + r))
    annual = float("nan") if (r < -1).any() else growth ** (p / len(r)) - 1
    volatility = float(np.std(r, ddof=1) * np.sqrt(p)) if len(r) > 1 else 0.0
    excess = r - risk_free
    dd, duration = _drawdown(r)
    out = dict(
        cumulative_return=growth - 1,
        annual_return=annual,
        annual_volatility=volatility,
        sharpe=_ratio(float(excess.mean()) * p, volatility),
        sortino=_ratio(
            float(excess.mean()) * np.sqrt(p),
            float(np.sqrt(np.mean(np.minimum(excess, 0) ** 2))),
        ),
        max_drawdown=dd,
        drawdown_duration=duration,
        calmar=_ratio(annual, dd),
    )
    for key, values in (
        ("annual_turnover", turnover),
        ("annual_cost_ratio", cost_ratio),
    ):
        if values is None:
            out[key] = float("nan")
        else:
            a = np.asarray(values, dtype=float)
            if a.shape != r.shape or not np.isfinite(a).all():
                raise ValueError(f"{key}: expected finite vector matching returns")
            out[key] = float(a.mean() * p)
    if benchmark is not None:
        b = np.asarray(benchmark, dtype=float)
        if b.shape != r.shape or not np.isfinite(b).all() or (b <= -1).any():
            raise ValueError("benchmark: expected finite matching returns > -1")
        active = r - b
        tracking = float(np.std(active, ddof=1) * np.sqrt(p)) if len(r) > 1 else 0.0
        relative = (1 + r) / (1 + b) - 1
        beta = _ratio(
            float(np.sum((r - r.mean()) * (b - b.mean()))),
            float(np.sum((b - b.mean()) ** 2)),
        )
        out.update(
            excess_annual_return=float("nan")
            if (r < -1).any()
            else float(np.prod(1 + relative) ** (p / len(r)) - 1),
            tracking_error=tracking,
            information_ratio=_ratio(float(active.mean()) * p, tracking),
            beta=beta,
            alpha=float(((r.mean() - risk_free) - beta * (b.mean() - risk_free)) * p),
            excess_max_drawdown=_drawdown(relative)[0],
        )
    return out


def summarize(result, *, periods_per_year=252, risk_free=0.0, benchmark=None):
    """Summarize a single Result, using same-day equity for cost ratios."""
    costs = (
        result.commission
        + result.tax
        + result.other_fee
        + result.slippage_cost
        + result.impact_cost
        + result.dividend_tax
        + result.borrow_cost
        + result.financing_interest
    )
    ratio = np.divide(
        costs, result.equity, out=np.zeros_like(costs), where=result.equity > 0
    )
    out = calculate(
        result.returns,
        periods_per_year=periods_per_year,
        risk_free=risk_free,
        benchmark=benchmark,
        turnover=result.turnover,
        cost_ratio=ratio,
        allow_bankruptcy=True,
    )
    out["bankrupt"] = bool(np.any(result.equity <= 0) and result.bankruptcy_events)
    out["geometric_metrics_valid"] = bool(np.all(result.returns >= -1))
    return out
