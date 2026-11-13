"""风险归因与 Brinson：参考实现、恒等式、教科书例子、已知答案。"""

import numpy as np
import pytest
from analysis_fixtures import make_panel, random_history, random_weights
from analysis_reference import ref_brinson, ref_factor_risk, ref_risk

import ashare.analysis as an
from ashare.risk import RiskModel

EPS = np.finfo(float).eps


def _model(n=9, k=4, seed=0):
    rng = np.random.default_rng(seed)
    b = rng.normal(0, 1, (n, k))
    b[:, 0] = 1
    a = rng.normal(0, 0.01, (k, k))
    f = a @ a.T + 1e-5 * np.eye(k)
    d = rng.uniform(1e-4, 5e-4, n)
    names = ("country", "industry_a", "style_x", "style_y")
    return RiskModel(tuple(f"s{i}" for i in range(n)), b, names, f, d, "20200102")


def test_decompose_risk_matches_dense_reference():
    m = _model()
    rng = np.random.default_rng(1)
    w, wb = rng.uniform(0, 1, 9), rng.uniform(0, 1, 9)
    w, wb = w / w.sum(), wb / wb.sum()
    rd = an.decompose_risk(w, m, benchmark=wb)
    vol, stock = ref_risk((w - wb).tolist(), m.B.tolist(), m.F.tolist(), m.D.tolist())
    fac, spec = ref_factor_risk(
        (w - wb).tolist(), m.B.tolist(), m.F.tolist(), m.D.tolist()
    )
    assert abs(rd.volatility - vol) < 1e-15
    assert abs(rd.volatility - m.volatility(w, wb)) < 1e-15
    np.testing.assert_allclose(rd.table["variance"][:-2], fac, rtol=1e-12, atol=1e-20)
    assert abs(rd.table["variance"][-2] - spec) < 1e-20
    np.testing.assert_allclose(rd.stocks["contribution"], stock, rtol=1e-11, atol=1e-18)
    np.testing.assert_allclose(
        rd.stocks["contribution"], m.contributions(w, wb), rtol=1e-12
    )
    # 恒等式：因子 + 特质的贡献之和 = σ；占比之和 = 1；类别与明细一致。
    contrib = rd.table["contribution"]
    assert abs(contrib[:-1].sum() - rd.volatility) < 4 * EPS * rd.volatility
    assert abs(contrib[-1] - rd.volatility) < 4 * EPS * rd.volatility
    assert abs(rd.table["percent"][:-1].sum() - 1) < 8 * EPS
    assert (
        abs(rd.groups["contribution"].sum() - rd.volatility) < 4 * EPS * rd.volatility
    )
    assert (
        abs(rd.stocks["contribution"].sum() - rd.volatility) < 16 * EPS * rd.volatility
    )
    # 因子边际 × 暴露 = 因子贡献。
    np.testing.assert_allclose(
        rd.table["marginal"][:-2] * rd.table["exposure"][:-2], contrib[:-2], rtol=1e-12
    )
    assert set(rd.to_dict()) >= {"factors", "groups", "stocks"}


def test_single_factor_risk_is_all_in_that_factor():
    m = _model(seed=3)
    b = m.B
    others = np.delete(b, 2, axis=1)
    q, _ = np.linalg.qr(others)
    a = b[:, 2] - q @ (q.T @ b[:, 2])
    zero_d = RiskModel(m.symbols, b, m.factors, m.F, np.zeros(9))
    rd = an.decompose_risk(a, zero_d)
    pct = dict(zip(rd.table.index, rd.table["percent"], strict=True))
    assert abs(pct["style_x"] - 1) < 1e-12
    assert all(
        abs(v) < 1e-12 for name, v in pct.items() if name not in ("style_x", "total")
    )


def test_risk_series_matches_daily_decomposition():
    panel = make_panel(seed=31)
    hist = random_history(panel, seed=4)
    w = random_weights(panel)
    rs = an.risk_series(w, panel, hist, benchmark="weight_bench", window=20)
    r = an.total_returns(panel)
    hb = an.benchmark_held(panel, "weight_bench", r)
    day = 37
    bench_end = hb[day + 1]
    model = (
        hist.exposures[day],
        hist.covariance[day],
        hist.specific_variance[day],
        hist.factors,
    )
    rd = an.decompose_risk(w[day], model, benchmark=bench_end)
    assert abs(rs.predicted["total"][day] - rd.volatility) < 1e-15
    np.testing.assert_allclose(
        rs.contributions.values[day],
        rd.table["contribution"][:-1],
        rtol=1e-12,
        atol=1e-20,
    )
    # 每天的贡献之和 = 预测波动率；四类占比之和 = 1。
    np.testing.assert_allclose(
        rs.contributions.values.sum(1), rs.predicted["total"], rtol=1e-12
    )
    pct = rs.predicted.values[:, 3:].sum(1)
    np.testing.assert_allclose(pct, 1, atol=1e-13)
    # 实现值是 t 日收盘后的主动权重乘 t+1 日的收益。
    realized = ((w[day] - bench_end) * np.nan_to_num(r[day + 1])).sum()
    assert abs(rs.realized["next_day_active"][day] - realized) < 1e-17
    assert np.isnan(rs.realized["next_day_active"][-1])
    assert rs.bias["count"] == len(panel.dates) - 1


def test_bias_statistic_recovers_known_scale():
    rng = np.random.default_rng(5)
    z = rng.normal(0, 1.3, 5000)
    b = an.bias_statistics(z, window=250)
    assert abs(b["statistic"] - 1.3) < 0.05
    assert b["interval"][0] < 1 < b["interval"][1]
    assert b["rolling_outside"] > 0.9
    ok = an.bias_statistics(rng.normal(0, 1, 5000), window=250)
    assert ok["rolling_outside"] < 0.2


def test_risk_series_bias_near_one_when_model_is_true():
    """收益按风险模型真实生成时，偏差统计量应接近 1。"""
    from analysis_fixtures import FakePanel

    base = make_panel(t=600, n=30, sectors=3, seed=32, suspend=False)
    hist = random_history(base, seed=6)
    rng = np.random.default_rng(7)
    t, n, k = hist.exposures.shape
    r = np.zeros((t, n))
    for day in range(1, t):
        f = rng.multivariate_normal(np.zeros(k), hist.covariance[day - 1])
        u = rng.normal(0, np.sqrt(hist.specific_variance[day - 1]))
        r[day] = hist.exposures[day - 1] @ f + u
    fields = dict(base.fields)
    fields["close"] = 10 * np.cumprod(1 + r, axis=0)
    fields["adj_factor"] = np.ones((t, n))
    panel = FakePanel(base.dates, base.symbols, fields)
    w = random_weights(panel, every=1)
    rs = an.risk_series(w, panel, hist, benchmark="weight_bench", window=100)
    lo, hi = rs.bias["interval"]
    assert lo < rs.bias["statistic"] < hi


# ---------------- Brinson ----------------


def test_brinson_textbook_example():
    """Bacon (2008)《Practical Portfolio Performance Measurement and Attribution》第 2 版
    第 5 章的三国例子：英、日、美组合权重 40/30/30%，基准 40/20/40%；组合收益 20/−5/6%，
    基准收益 10/−4/8%；组合 8.3%，基准 6.4%。数值由下面逐项手算核对。"""
    wp, wb = np.array([0.4, 0.3, 0.3]), np.array([0.4, 0.2, 0.4])
    rp, rb = np.array([0.20, -0.05, 0.06]), np.array([0.10, -0.04, 0.08])
    a, s, i = an.brinson_effects(wp, wb, rp, rb, "fachler")
    np.testing.assert_allclose(a, [0.0, -0.0104, -0.0016], atol=1e-15)
    np.testing.assert_allclose(s, [0.04, -0.002, -0.008], atol=1e-15)
    np.testing.assert_allclose(i, [0.0, -0.001, 0.002], atol=1e-15)
    assert abs(a.sum() + s.sum() + i.sum() - (0.083 - 0.064)) < 1e-15
    a2, s2, i2 = an.brinson_effects(wp, wb, rp, rb, "bhb")
    np.testing.assert_allclose(a2, [0.0, -0.004, -0.008], atol=1e-15)
    assert abs(a2.sum() - a.sum()) < 1e-15  # 总配置相同，分到行业不同
    np.testing.assert_array_equal(s2, s)


def test_brinson_effects_match_reference_with_missing_sectors():
    rng = np.random.default_rng(8)
    for _ in range(50):
        s = 6
        wp, wb = rng.uniform(0, 1, s), rng.uniform(0, 1, s)
        wp[rng.integers(s)] = 0
        wb[rng.integers(s)] = 0
        wp, wb = wp / wp.sum(), wb / wb.sum()
        rp, rb = rng.normal(0, 0.05, s), rng.normal(0, 0.05, s)
        for bhb in (False, True):
            got = an.brinson_effects(
                wp,
                wb,
                np.where(wp > 0, rp, np.nan),
                np.where(wb > 0, rb, np.nan),
                "bhb" if bhb else "fachler",
            )
            ref = ref_brinson(
                {j: wp[j] for j in range(s) if wp[j]},
                {j: wb[j] for j in range(s) if wb[j]},
                {j: rp[j] for j in range(s)},
                {j: rb[j] for j in range(s)},
                bhb,
            )
            for j in range(s):
                np.testing.assert_allclose(
                    [g[j] for g in got], ref.get(j, (0, 0, 0)), atol=1e-15
                )
            total = sum(x.sum() for x in got)
            assert abs(total - ((wp * rp).sum() - (wb * rb).sum())) < 1e-15


def test_brinson_identity_and_reference_on_panel():
    panel = make_panel(seed=41)
    w = random_weights(panel)
    b = an.brinson(w, panel, "weight_bench")
    ident = b.summary["identity"]
    assert ident["daily_max_abs"] < 1e-16
    assert ident["cumulative_max_eps"] < 64
    assert ident["sector_vs_total_max_abs"] < 1e-15
    # 抽一天与逐股循环的参考实现比。
    r = np.nan_to_num(an.total_returns(panel))
    held = an.held_from_weights(w)
    hb = an.benchmark_held(panel, "weight_bench", an.total_returns(panel))
    codes, names = an.inputs.industry_codes(panel)
    day = 65
    wp, wb, cp, cb = {}, {}, {}, {}
    for i in range(len(panel.symbols)):
        s = codes[day - 1, i]
        wp[s] = wp.get(s, 0) + held[day, i]
        wb[s] = wb.get(s, 0) + hb[day, i]
        cp[s] = cp.get(s, 0) + held[day, i] * r[day, i]
        cb[s] = cb.get(s, 0) + hb[day, i] * r[day, i]
    wp["cash"], wb["cash"] = 1 - sum(wp.values()), 1 - sum(wb.values())
    cp["cash"] = cb["cash"] = 0.0
    ref = ref_brinson(
        {s: v for s, v in wp.items() if v},
        {s: v for s, v in wb.items() if v},
        {s: cp[s] / wp[s] for s in wp if wp[s] and s != "cash"} | {"cash": 0.0},
        {s: cb[s] / wb[s] for s in wb if wb[s] and s != "cash"} | {"cash": 0.0},
        known=("cash",),
    )
    row = list(b.daily.index).index(panel.dates[day])
    for j in range(3):
        assert abs(b.daily.values[row, j] - sum(v[j] for v in ref.values())) < 1e-16
    assert "cash" in b.sectors.index


def test_brinson_pure_allocation_has_zero_selection():
    panel = make_panel(seed=42, sectors=4)
    r = an.total_returns(panel)
    hb = an.benchmark_held(panel, "weight_bench", r)
    codes, _ = an.inputs.industry_codes(panel)
    t, n = hb.shape
    w = np.zeros((t, n))
    tilt = np.array([1.6, 0.6, 1.0, 0.8, 1.0])
    for day in range(t - 1):
        # t 日收盘后的组合 = 基准（t+1 日持有）在行业内同比例、行业间重新配比。
        x = hb[day + 1] * tilt[codes[day]]
        w[day] = x / x.sum()
    b = an.brinson(w, panel, "weight_bench")
    assert np.abs(b.daily["selection"]).max() < 1e-16
    assert np.abs(b.daily["interaction"]).max() < 1e-16
    assert np.abs(b.daily["allocation"]).max() > 1e-4


def test_brinson_with_panelbt_result():
    from test_attribution import _result_case

    panel, result, _ = _result_case(seed=43)
    b = an.brinson(result, panel, "weight_bench", method="bhb", link_method="carino")
    assert b.summary["identity"]["daily_max_abs"] < 1e-16
    assert b.summary["identity"]["cumulative_max_eps"] < 64
    assert b.cumulative.last()["cost"] < 0
    with pytest.raises(ValueError, match="基准"):
        an.brinson(result, panel, None)


def test_results_convert_to_plain_json():
    import json

    panel = make_panel(seed=44)
    hist = random_history(panel)
    w = random_weights(panel)
    rep = an.attribute(w, panel, hist, benchmark="weight_bench")
    rs = an.risk_series(w, panel, hist, benchmark="weight_bench", window=20)
    b = an.brinson(w, panel, "weight_bench")
    for obj in (rep, rs, b):
        d = obj.to_dict()
        text = json.dumps(d, allow_nan=False)
        assert "NaN" not in text
    d = rep.to_dict()
    assert d["groups_cumulative"]["date"][0] == rep.groups.index[0]
    assert set(d["groups"]) == {
        "date",
        "market",
        "industry",
        "style",
        "specific",
        "trading",
        "cost",
    }
    assert b.to_dict()["sectors"]["sector"][-1] == "cash"


def test_brinson_cash_drag_is_allocation():
    """组合留现金、行业内与基准同比例：现金拖累全部在现金一行的配置项，选择与交互为 0。"""
    panel = make_panel(seed=45)
    r = an.total_returns(panel)
    hb = an.benchmark_held(panel, "weight_bench", r)
    w = np.zeros_like(hb)
    w[:-1] = 0.9 * hb[1:]
    b = an.brinson(w, panel, "weight_bench")
    cash = list(b.sectors.index).index("cash")
    assert abs(b.sectors["selection"][cash]) < 1e-16
    assert abs(b.sectors["interaction"][cash]) < 1e-16
    assert np.abs(b.daily["selection"]).max() < 1e-16
    assert np.abs(b.daily["interaction"]).max() < 1e-16
    # 每天的配置项 = −10% × 基准收益。
    rb = (hb * np.nan_to_num(r)).sum(axis=1)[1:]
    np.testing.assert_allclose(b.daily["allocation"], -0.1 * rb, atol=1e-16)
    bhb = an.brinson(w, panel, "weight_bench", method="bhb")
    np.testing.assert_allclose(bhb.daily["allocation"], -0.1 * rb, atol=1e-16)
