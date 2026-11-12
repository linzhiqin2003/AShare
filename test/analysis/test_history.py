"""FactorHistory.collect：补算的因子收益与风险模型内部的回归逐位一致，时点不前视。"""

import numpy as np
import pytest
from analysis_fixtures import make_panel

import ashare.analysis as an
import ashare.risk.series as series
from ashare.risk import model_series
from ashare.risk.kernels import cross_section as numpy_cross_section


def _capture(monkeypatch):
    seen = {}
    original = series.cross_section_batch

    def wrapped(returns, designs, caps, estimates, industry):
        f, residual = original(returns, designs, caps, estimates, industry)
        # 记录每个回归的输入收益（用于定位日期）与系数。
        for j in range(len(returns)):
            seen[returns[j].tobytes()] = f[j]
        return f, residual

    monkeypatch.setattr(series, "cross_section_batch", wrapped)
    return seen


def test_collect_reproduces_model_factor_returns(monkeypatch):
    panel = make_panel(t=150, n=16, seed=21)
    source, cfg = an.risk_source(panel)
    seen = _capture(monkeypatch)
    hist = an.FactorHistory.collect(
        model_series(source, config=cfg, threads=1), source, config=cfg
    )
    from ashare.risk.data import adjusted_returns

    r = adjusted_returns(source, len(source.dates) - 1)
    r[np.asarray(source["is_suspended"], bool)] = np.nan
    matched = 0
    for day in range(1, len(hist.dates)):
        key = r[day].tobytes()
        if key in seen:
            np.testing.assert_array_equal(hist.factor_returns[day], seen[key])
            matched += 1
        else:
            assert np.isnan(hist.factor_returns[day]).all()
    # 上市满 60 日后每天都有回归，且全部逐位对上。
    assert matched == np.isfinite(hist.factor_returns).all(axis=1).sum() >= 85
    assert hist.groups.count("market") == 1
    assert hist.groups.count("industry") == 5  # 4 个行业 + 未知
    assert hist.covariance.shape == (150, len(hist.factors), len(hist.factors))


def test_collect_factor_returns_match_independent_numpy_wls():
    panel = make_panel(t=130, n=16, seed=22)
    source, cfg = an.risk_source(panel)
    models = list(model_series(source, config=cfg, threads=1))
    hist = an.FactorHistory.collect(iter(models), source, config=cfg)
    from ashare.risk.data import adjusted_returns

    r = adjusted_returns(source, len(source.dates) - 1)
    cap = np.asarray(source["total_mv"])
    ind = [i for i, g in enumerate(hist.groups) if g == "industry"]
    checked = 0
    for day in range(cfg.min_listing_days + 1, len(models)):
        est = np.isfinite(r[day]) & ~np.asarray(source["is_suspended"][day - 1], bool)
        f, _ = numpy_cross_section(r[day], models[day - 1].B, cap[day - 1], est, ind)
        np.testing.assert_allclose(hist.factor_returns[day], f, rtol=1e-9, atol=1e-12)
        checked += 1
    assert checked > 50


def test_collect_does_not_read_future_returns():
    """改动第 t+1 日之后的收益，不影响 ≤ t 日的因子收益（前缀不变）。"""
    panel = make_panel(t=110, n=14, seed=23)
    source, cfg = an.risk_source(panel)
    a = an.FactorHistory.collect(
        model_series(source, config=cfg, threads=1), source, config=cfg
    )
    panel.fields["close"][90:] *= np.linspace(1, 1.5, 14)
    source2, _ = an.risk_source(panel)
    b = an.FactorHistory.collect(
        model_series(source2, config=cfg, threads=1), source2, config=cfg
    )
    np.testing.assert_array_equal(a.factor_returns[:90], b.factor_returns[:90])
    assert not np.array_equal(
        a.factor_returns[90:], b.factor_returns[90:], equal_nan=True
    )


def test_collect_columns_subset_and_errors():
    panel = make_panel(t=90, n=10, seed=24)
    source, cfg = an.risk_source(panel)
    keep = panel.symbols[:3]
    hist = an.FactorHistory.collect(
        model_series(source, config=cfg, threads=1), source, keep, config=cfg
    )
    assert hist.symbols == keep and hist.exposures.shape[1] == 3
    with pytest.raises(KeyError):
        an.FactorHistory.collect(iter([]), source, ["nope"], config=cfg)
    with pytest.raises(ValueError, match="为空"):
        an.FactorHistory.collect(iter([]), source, config=cfg)
    models = list(model_series(source, config=cfg, threads=1))
    with pytest.raises(ValueError, match="连续"):
        an.FactorHistory.collect(iter([models[0], models[2]]), source, config=cfg)
    w = np.zeros((90, 10))
    w[5:, 1] = 0.5
    assert an.held_columns(w, None, symbols=panel.symbols) == (panel.symbols[1],)
