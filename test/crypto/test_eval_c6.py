"""来源：C6 推断口径与开发中实测问题；独立逐项展开核对正向 1B。"""

import numpy as np
import pytest

from ashare.data.crypto.timeutil import DAY, HOUR, millis
from ashare.factor.crypto_eval import (
    BOUNDARY,
    CONFIRM_FAMILY,
    PHI_GRID,
    NullTable,
    SimulationBank,
    _batch_components,
    _component_t,
    calibrate,
    conservative_test,
    crypto_labels,
    empirical_p,
    forward_sum,
    hodrick_1b,
    holm_confirm,
    quantile_description,
    sample_mask,
)
from test.crypto.test_factors_c6 import Synthetic


def independent_1b(x, g, h):
    y = []
    z = []
    domain = set()
    for i, value in enumerate(x):
        if (
            np.isfinite(value)
            and i + h <= len(g)
            and all(np.isfinite(g[i + j]) for j in range(h))
        ):
            domain.add(i)
            z.append([1, value])
            y.append(sum(g[i + j] for j in range(h)))
    z = np.array(z)
    y = np.array(y)
    xx = np.zeros((2, 2))
    xy = np.zeros(2)
    for row, yi in zip(z, y, strict=True):
        for j in range(2):
            xy[j] += row[j] * yi
            for k in range(2):
                xx[j, k] += row[j] * row[k]
    inv = np.linalg.inv(xx)
    beta = inv @ xy
    mean = sum(gi for gi in g if np.isfinite(gi)) / sum(np.isfinite(g))
    meat = np.zeros((2, 2))
    for i in range(h - 1, len(g)):
        if not np.isfinite(g[i]) or not all(i - j in domain for j in range(h)):
            continue
        q = [h, sum(x[i - j] for j in range(h))]
        for j in range(2):
            for k in range(2):
                meat[j, k] += (g[i] - mean) ** 2 * q[j] * q[k]
    v = inv @ meat @ inv
    return beta[1], np.sqrt(v[1, 1])


def test_1b_four_point_hand():
    # h=1,x=[0,1,2,3],g=[1,0,3,2]；斜率 .6，S 的斜率方差=2.25/25。
    r = hodrick_1b([0, 1, 2, 3], [1, 0, 3, 2], 1)
    assert r["beta"] == pytest.approx(0.6)
    # centered q=[-1.5,-.5,.5,1.5],e=[-.5,-1.5,1.5,.5]，和为2.25。
    assert r["se"] == pytest.approx(0.3)
    assert r["t"] == pytest.approx(2)


@pytest.mark.parametrize("h", [1, 2, 3, 7, 30])
def test_1b_independent_expansion_and_missing_past(h):
    rng = np.random.default_rng(2026)
    x = rng.normal(size=160)
    g = rng.normal(size=160)
    x[[20, 50, 51]] = np.nan
    g[40] = np.nan
    beta, se = independent_1b(x, g, h)
    observed = hodrick_1b(x, g, h)
    assert observed["beta"] == pytest.approx(beta, abs=1e-12)
    assert observed["se"] == pytest.approx(se, abs=1e-12)
    a = np.nan_to_num(x)[None, :]
    b = np.zeros_like(a)
    parts = _batch_components(
        a, b, np.nan_to_num(g)[None, :], np.isfinite(x), np.isfinite(g), h
    )
    assert _component_t(parts, 1)[0] == pytest.approx(observed["t"], abs=1e-12)


def test_labels_additive_cash_boundaries_and_untradable():
    ds = Synthetic(40 * 24 + 1)
    at = ds.time + HOUR
    r = crypto_labels(ds, ["BTCUSDT"], (at, np.zeros((len(at), 1))))
    assert r["entry"][0] == millis("2023-10-02")
    i = 0
    entry = 24
    exit = 48
    paid = sum(
        ds.values["funding_rate"][k, 1] * 100
        for k in range(entry, exit)
        if np.isfinite(ds.values["funding_rate"][k, 1])
    )
    expected = (
        ds.values["open"][exit, 1] - ds.values["open"][entry, 1] - paid
    ) / ds.values["open"][entry, 1]
    assert r["descriptive"][1]["usdm"][i, 0] == pytest.approx(expected)
    for kind in ("spot", "usdm"):
        for h in (1, 7, 30):
            assert r["inference"][h][kind][0, 0] == pytest.approx(
                sum(r["daily"][kind][:h, 0])
            )
    ds.values["tradable"][48, 1] = False
    r = crypto_labels(ds, ["BTCUSDT"], (at, np.zeros((len(at), 1))))
    assert np.isnan(r["descriptive"][1]["usdm"][0, 0])
    assert np.isnan(r["inference"][7]["usdm"][0, 0])


def test_split_excludes_cross_boundary():
    entries = BOUNDARY + np.arange(-10, 4) * DAY
    ins = sample_mask(entries, 7, "in")
    out = sample_mask(entries, 7, "out")
    assert list(entries[ins]) == list(BOUNDARY + np.arange(-10, -7) * DAY)
    assert np.all(entries[out] >= BOUNDARY)
    assert not np.any(ins & out)


def test_calibration_uses_only_in_sample():
    rng = np.random.default_rng(7)
    n = 600
    entries = BOUNDARY + np.arange(-400, 200) * DAY
    x = rng.normal(size=n)
    g = rng.normal(size=n)
    a = calibrate(x, g, entries)
    x[400:] = 1e9
    g[400:] = -1e12
    b = calibrate(x, g, entries)
    for key in a:
        np.testing.assert_equal(a[key], b[key])


def test_count_formula_and_fixed_holm():
    assert empirical_p([1, 2, 3], 2) == 0.75
    table = NullTable([(0, 0)], np.array([[1.0, 2.0, 3.0]]))
    assert table.pvalue(4) == pytest.approx(0.251)
    assert table.pvalue(3) == pytest.approx(0.501)
    out = holm_confirm({CONFIRM_FAMILY[0]: 0.001})
    assert len(out) == 8
    assert out[CONFIRM_FAMILY[0]] == 0.008
    assert sum(np.isnan(v) for v in out.values()) == 7


def test_grid_regression_floating_arange_endpoint():
    # 开发首跑发现 np.arange(.991,1,.001) 含接近 1 的第十项，四舍五入为 1。
    assert len(PHI_GRID) == 37
    assert PHI_GRID[-1] == 0.999


def test_null_zero_predictive_slope_and_same_day_correlation():
    bank = SimulationBank(
        np.linspace(-0.1, 0.1, 1001), 700, repetitions=500, phi_grid=[0, 0.8]
    )
    a, b = bank.basis(0.8)
    x = 0.5 * a + np.sqrt(0.75) * b
    g = bank.returns[:, 1:]
    numerator = ((x - x.mean(axis=1)[:, None]) * (g - g.mean(axis=1)[:, None])).sum(
        axis=1
    )
    slopes = numerator / ((x - x.mean(axis=1)[:, None]) ** 2).sum(axis=1)
    assert abs(slopes.mean()) < 0.0005
    innovations = x[:, 1:] - 0.8 * x[:, :-1]
    assert np.corrcoef(innovations.ravel(), g[:, :-1].ravel())[0, 1] == pytest.approx(
        0.5, abs=0.01
    )


def test_conservative_table_and_critical_equivalence():
    bank = SimulationBank(
        np.linspace(-1, 1, 100), 40, repetitions=400, phi_grid=[0, 0.5, 0.9]
    )
    mask = np.ones(40, bool)
    table, details = conservative_test(bank, mask, mask, 3, [-0.3, 0, 0.3], 1.5)
    assert details["p"] >= 0.001 + 1 / 401
    threshold = table.critical(0.05)
    values = np.r_[np.linspace(0, 10, 1000), threshold]
    np.testing.assert_array_equal(table.pvalue(values) <= 0.05, values > threshold)


def test_quantile_past_only_ties_lower_group():
    entries = np.arange(400) * DAY
    x = np.ones(400)
    y = np.arange(400, dtype=float)
    rows = quantile_description(x, y, entries, 7)
    assert rows[0]["n"] == 220
    assert all(r["n"] == 0 for r in rows[1:])
    assert rows[0]["overlapping_pairs"] == 219


def test_forward_sum_missing_and_three_point():
    np.testing.assert_equal(forward_sum([1, 2, 3, 4, 5], 3), [6, 9, 12, np.nan, np.nan])
    np.testing.assert_equal(
        forward_sum([1, np.nan, 3, 4, 5], 3), [np.nan, np.nan, 12, np.nan, np.nan]
    )


def test_stationary_common_random_numbers_refined_point():
    # 来源：复核发现新增内点按不同 burn 长度抽样会改变每次重复的随机数。
    eps = np.linspace(-1, 1, 21)
    a = SimulationBank(eps, 10, repetitions=10, phi_grid=[0, 0.9])
    b = SimulationBank(eps, 10, repetitions=10, phi_grid=[0, 0.7, 0.9])
    np.testing.assert_allclose(a.basis(0.7)[0], b.basis(0.7)[0], rtol=0, atol=1e-14)


def test_power_polynomial_equals_daily_alternative():
    from ashare.factor.crypto_eval import _power_components

    rng = np.random.default_rng(4)
    a, b, g = rng.normal(size=(3, 13, 50))
    xm = np.ones(50, bool)
    gm = xm.copy()
    xm[10] = False
    gm[30] = False
    parts = _power_components(a, b, g, xm, gm, 7)
    for d in [-0.8, 0, 0.4]:
        s = np.sqrt(1 - d * d)
        co = [
            sum(v * d ** (len(c) - 1 - i) * s**i for i, v in enumerate(c))
            for c in parts
        ]
        for slope in [-0.03, 0, 0.2]:
            expected = _component_t(
                _batch_components(
                    d * a + s * b,
                    np.zeros_like(a),
                    g + slope * (d * a + s * b),
                    xm,
                    gm,
                    7,
                ),
                1,
            )
            n0, n1, v0, v1, v2 = co
            observed = (n0 + slope * n1) / np.sqrt(v0 + slope * v1 + slope * slope * v2)
            np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)


def test_explicit_sample_mask_matches_simulation():
    # 来源：复核发现观察 OLS 会读 sample 外的未来 g，而模拟将其屏蔽。
    x = np.arange(20, dtype=float) ** 0.7
    g = np.sin(np.arange(20))
    sample = np.arange(20) < 15
    r = hodrick_1b(x, g, 3, sample=sample)
    parts = _batch_components(
        x[None, :], np.zeros((1, 20)), g[None, :], sample, sample, 3
    )
    assert r["t"] == pytest.approx(_component_t(parts, 1)[0])


def test_public_client_retries_remote_disconnect():
    # 来源：C6 实际季度下载遭遇 RemoteProtocolError，原重试只覆盖 NetworkError。
    import httpx

    from ashare.data.crypto.sources import PublicClient

    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.RemoteProtocolError("synthetic disconnect")
        return httpx.Response(200, content=b"ok")

    with PublicClient(
        transport=httpx.MockTransport(handler), sleep=lambda _: None
    ) as c:
        assert c.get("https://example.test/archive") == b"ok"
    assert len(calls) == 2


def test_missing_observed_statistic_never_minimum_p():
    # 来源：开发复核，NaN 比较全为 False 会误变成最小 p。
    assert np.isnan(empirical_p([1.0, 2.0, 3.0], np.nan))
    table = NullTable([(0, 0)], np.array([[1.0, 2.0, 3.0]]))
    assert np.isnan(table.pvalue(np.nan))


def test_1b_five_points_regression_domain_hand():
    # 来源：4.18“样本索引为 x 与 R 都有效”的复核；Z 只含四个有效两日标签。
    # x=0..4，R=[1,3,5,6]，β=1.7，xbar=1.5，Z'Z 的中心平方和=5。
    # 完整过去两项窗口只在 t=1,2,3；e=[-1,-2,1,0,2]，q中心=[-2,0,2]。
    # S 的斜率中心和=16，所以 se=sqrt(16)/5=.8；不能把不在 Z 的 x[4] 算进 S。
    r = hodrick_1b(np.arange(5), [1, 0, 3, 2, 4], 2)
    assert r["beta"] == pytest.approx(1.7)
    assert r["se"] == pytest.approx(0.8)
    assert r["meat_terms"] == 3
