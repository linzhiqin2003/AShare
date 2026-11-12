from dataclasses import replace
from pathlib import Path

import numpy as np
import portopt
import pytest

import ashare
from ashare.risk.data import MemoryPanel
from ashare.risk.fundamental import FundamentalConfig, exposures_at, standardize
from ashare.risk.series import model_series

ROOT = Path(__file__).resolve().parents[2]
assert Path(ashare.__file__).resolve().is_relative_to(ROOT)
assert Path(portopt.__file__).resolve().is_relative_to(ROOT)


def source(t=90, n=75):
    rng = np.random.default_rng(71)
    close = 20 * np.exp(np.cumsum(rng.normal(0, 0.02, (t, n)), axis=0))
    fields = dict(
        close=close,
        adj_factor=np.ones((t, n)),
        total_mv=np.exp(rng.normal(20, 1, (t, n))),
        circ_mv=np.full((t, n), 1e9),
        turnover_rate=rng.uniform(0.001, 0.05, (t, n)),
        pb=rng.uniform(0.5, 5, (t, n)),
        pe_ttm=rng.uniform(5, 40, (t, n)),
        industry=np.broadcast_to(np.arange(n) % 5, (t, n)).copy(),
        is_suspended=np.zeros((t, n), bool),
        is_listed=np.ones((t, n), bool),
    )
    fields["is_listed"][:70, 0] = False
    fields["is_suspended"][50:, 1] = True
    fields["pb"][30:45, 2] = np.nan
    dates = np.datetime64("2020-01-01") + np.arange(t)
    return MemoryPanel(dates, tuple(map(str, range(n))), "synthetic-v1", fields)


def test_standardization_and_missing():
    rng = np.random.default_rng(4)
    cap = np.exp(rng.normal(size=50))
    est = np.arange(50) > 5
    industries = np.eye(4)[np.arange(50) % 4]
    raw = rng.normal(size=50)
    raw[:8] = np.nan
    z = standardize(raw, cap, est, industries)
    assert np.isfinite(z).all()
    assert abs(np.average(z[est], weights=cap[est])) < 1e-14
    assert np.std(z[est]) == pytest.approx(1, abs=1e-14)


def test_no_lookahead_and_incremental(tmp_path, monkeypatch):
    p = source()
    config = FundamentalConfig(
        industry_codes=tuple(range(5)),
        min_listing_days=15,
        min_descriptor_days=10,
        structural_days=30,
        history=60,
    )
    full = list(model_series(p, config=config))
    split = 69
    fields = {k: v.copy() for k, v in p.fields.items()}
    for name, values in fields.items():
        if values.dtype == bool:
            values[split + 1 :] = ~values[split + 1 :]
        elif name == "industry":
            values[split + 1 :] = 4
        else:
            values[split + 1 :] *= 3
    changed = MemoryPanel(p.dates, p.symbols, p.version, fields)
    other = list(model_series(changed, config=config, end=split + 1))
    for a, b in zip(full[: split + 1], other, strict=True):
        for name in ("B", "F", "D"):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    prefix = MemoryPanel(
        p.dates[:-1], p.symbols, p.version, {k: v[:-1] for k, v in p.fields.items()}
    )
    cached = list(model_series(prefix, config=config, cache=tmp_path))
    import ashare.risk.series as module

    original = module.exposures_at
    calls = []

    def spy(*args, **kwargs):
        calls.append(args[2])
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "exposures_at", spy)
    incremental = list(
        model_series(p, config=config, cache=tmp_path, start=len(p.dates) - 1)
    )
    assert calls == [89]
    for a, b in zip(cached + incremental, full, strict=True):
        for name in ("B", "F", "D"):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    assert full[60].D[0] > 0 and full[60].D[1] > 0
    assert full[-1].metadata["estimation_count"] == 74


def test_exposure_shift_external_and_momentum():
    p = source(t=100)
    from ashare.risk.data import adjusted_returns

    c = FundamentalConfig(
        industry_codes=tuple(range(5)), min_listing_days=10, min_descriptor_days=10
    )
    r = adjusted_returns(p, 99)
    b, names, est, cap = exposures_at(p, r, 80, c)
    ages = np.cumsum(p["is_listed"], axis=0, dtype=np.int32)
    cached_b, _, cached_est, _ = exposures_at(p, r, 80, c, listing_ages=ages)
    np.testing.assert_array_equal(b, cached_b)
    np.testing.assert_array_equal(est, cached_est)
    r2 = r.copy()
    r2[61:81] *= 100
    b2, _, _, _ = exposures_at(p, r2, 80, c)
    np.testing.assert_array_equal(
        b[:, names.index("momentum")], b2[:, names.index("momentum")]
    )
    ext = np.random.default_rng(7).normal(size=(100, 75, 2))
    e, enames, _, _ = exposures_at(p, r, 80, c, ext, ("growth", "leverage"))
    assert enames[-2:] == ("growth", "leverage") and e.shape[1] == b.shape[1] + 2
    assert abs(np.average(e[est, -1], weights=cap[est])) < 1e-14
    # 显式用 B_t 生成下一日收益，错用 B_{t+1} 会无法精确还原。
    from ashare.risk.kernels import cross_section

    f = np.random.default_rng(5).normal(size=b.shape[1])
    f[1:7] = 0
    fitted, u = cross_section(b @ f, b, cap, est, np.arange(1, 7))
    np.testing.assert_allclose(u, 0, atol=1e-10)
    assert np.isfinite(fitted).all()
    assert replace(c, regime=False).regime is False


@pytest.mark.parametrize("kind", ["fundamental", "statistical"])
def test_statistical_series_and_external_prefix(tmp_path, kind):
    p = source(t=40, n=60)
    c = FundamentalConfig(
        industry_codes=tuple(range(5)),
        min_listing_days=10,
        min_descriptor_days=10,
        history=25,
    )
    ext = np.random.default_rng(123).normal(size=(40, 60, 1))
    options = dict(
        config=c,
        kind=kind,
        statistical_options={"components": 5},
        external=ext,
        external_names=("growth",),
    )
    whole = list(model_series(p, **options))
    prefix = list(model_series(p, end=30, cache=tmp_path, **options))
    changed_external = ext.copy()
    changed_external[30:] *= -1000
    future_changed = list(
        model_series(p, end=30, **{**options, "external": changed_external})
    )
    for a, b in zip(whole[:30], prefix, strict=True):
        np.testing.assert_array_equal(a.B, b.B)
        np.testing.assert_array_equal(a.F, b.F)
        np.testing.assert_array_equal(a.D, b.D)
    for a, b in zip(prefix, future_changed, strict=True):
        for name in ("B", "F", "D"):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
    assert whole[-1].metadata["kind"] == kind


def test_regression_uses_previous_exposure(tmp_path):
    import json

    from ashare.risk.data import adjusted_returns
    from ashare.risk.kernels import cross_section

    p = source(t=35, n=75)
    c = FundamentalConfig(
        industry_codes=tuple(range(5)), min_listing_days=10, min_descriptor_days=10
    )
    models = list(model_series(p, config=c, cache=tmp_path))
    r = adjusted_returns(p, 34)
    _, _, est, cap = exposures_at(p, r, 29, c)
    expected, _ = cross_section(r[30], models[29].B, cap, est, np.arange(1, 7))
    found = False
    for path in tmp_path.glob("*.npz"):
        with np.load(path, allow_pickle=False) as data:
            if json.loads(str(data["meta"]))["date"] == p.dates[30]:
                np.testing.assert_allclose(data["f"], expected, atol=1e-10, rtol=1e-10)
                found = True
    assert found


def test_nonzero_risk_free_log_momentum():
    from ashare.risk.data import adjusted_returns
    from ashare.risk.fundamental import _log_excess
    from ashare.risk.kernels import exp_weights

    p = source(t=100)
    rf = np.broadcast_to(np.linspace(0.0001, 0.001, 100)[:, None], (100, 75)).copy()
    p.fields["risk_free"] = rf
    raw = adjusted_returns(p, 99)
    excess = raw - rf
    np.testing.assert_allclose(
        _log_excess(p, excess, 1, 80),
        np.log1p(raw[1:80]) - np.log1p(rf[1:80]),
        atol=1e-16,
        rtol=1e-14,
    )
    c = FundamentalConfig(
        industry_codes=tuple(range(5)), min_listing_days=10, min_descriptor_days=10
    )
    b, names, est, cap = exposures_at(p, excess, 80, c)
    logs = np.log1p(raw[1:60]) - np.log1p(rf[1:60])
    valid = np.isfinite(logs)
    weights = exp_weights(59, c.momentum_half_life)[:, None] * valid
    momentum = (np.nan_to_num(logs) * weights).sum(axis=0) / np.maximum(
        weights.sum(axis=0), 1e-30
    )
    momentum[valid.sum(axis=0) < 10] = np.nan
    industries = b[:, 1:7]
    expected = standardize(
        momentum, cap, est, industries, b[:, names.index("size")], c.winsor_mad
    )
    np.testing.assert_allclose(
        b[:, names.index("momentum")], expected, atol=1e-13, rtol=1e-13
    )


def test_panel_validation_reports_missing_and_coverage(tmp_path):
    from ashare.risk.validate import validate_panel

    p = source(t=70, n=60)
    result = validate_panel(p, tmp_path, start=25, window=10)
    for report in result["models"].values():
        assert all(row["windows"] > 0 for row in report["bias"].values())
    assert (tmp_path / "validation.md").exists()
    assert (tmp_path / "validation.json").exists()


def test_liquidity_ignores_prelisting_values_and_scales_short_history():
    from ashare.risk.data import adjusted_returns

    p = source(t=100)
    fields = {key: value.copy() for key, value in p.fields.items()}
    fields["turnover_rate"][:] = 0.01
    fields["turnover_rate"][:70, 0] = np.nan
    first = MemoryPanel(p.dates, p.symbols, p.version, fields)
    fields["turnover_rate"][:70, 0] = 1000.0
    second = MemoryPanel(p.dates, p.symbols, p.version, fields)
    c = FundamentalConfig(industry_codes=tuple(range(5)), min_listing_days=15)
    returns = adjusted_returns(first, 99)
    a, names, _, _ = exposures_at(first, returns, 99, c)
    b, _, _, _ = exposures_at(second, returns, 99, c)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(a[:, names.index("liquidity")], np.zeros(75))


def test_unobserved_industry_uses_pool_imputation():
    raw = np.r_[np.arange(1, 31, dtype=float), np.full(10, np.nan)]
    industries = np.eye(4)[np.repeat(np.arange(4), 10)]
    z = standardize(raw, np.ones(40), np.ones(40, bool), industries)
    np.testing.assert_allclose(z[-10:], 0, atol=1e-14, rtol=0)
    assert np.std(z) == pytest.approx(1, abs=1e-14)


def test_first_listing_day_has_no_previous_tradable_price():
    from ashare.risk.data import adjusted_returns

    p = source(t=100)
    r = adjusted_returns(p, 99)
    assert np.isnan(r[:71, 0]).all()
    assert np.isfinite(r[71:, 0]).all()
