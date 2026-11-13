"""ashare.factor.groupbt.metrics：手算例子、直白参考实现和随机对拍。"""

import importlib.util
import math
from pathlib import Path

import numpy as np
import polars as pl
import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "gbt_metrics", _ROOT / "src/ashare/factor/groupbt/metrics.py"
)
m = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(m)


# ---- 最笨的逐行逐列参考实现 ----------------------------------------------------


def ref_delay(x, d):
    t_len, n_len = x.shape
    out = np.full((t_len, n_len), np.nan)
    for t in range(t_len):
        for j in range(n_len):
            if t >= d:
                out[t, j] = x[t - d, j]
    return out


def ref_risk_factors(close_adj, total_mv):
    t_len, n_len = close_adj.shape
    close = np.array(close_adj, dtype=np.float64)
    mv = np.array(total_mv, dtype=np.float64)
    for t in range(t_len):
        for j in range(n_len):
            if not close[t, j] > 0:
                close[t, j] = np.nan
            if not mv[t, j] > 0:
                mv[t, j] = np.nan
    size = np.log(mv)
    rev20 = np.full_like(close, np.nan)
    rev60 = np.full_like(close, np.nan)
    mom = np.full_like(close, np.nan)
    for t in range(t_len):
        for j in range(n_len):
            if t >= 20:
                rev20[t, j] = 1.0 - close[t, j] / close[t - 20, j]
            if t >= 60:
                rev60[t, j] = 1.0 - close[t, j] / close[t - 60, j]
            if t >= 240:
                mom[t, j] = close[t - 20, j] / close[t - 240, j] - 1.0
    return {
        "size": size,
        "reversal_20": rev20,
        "reversal_60": rev60,
        "momentum": mom,
    }


def ref_exposure(weights, risk, universe):
    t_len, n_len = weights.shape
    out = {}
    for name, factor in risk.items():
        exposure = np.full(t_len, np.nan)
        for t in range(t_len):
            active = False
            for j in range(n_len):
                w = weights[t, j]
                if not math.isnan(w) and w != 0.0:
                    active = True
            if not active:
                continue
            values = []
            for j in range(n_len):
                if universe[t, j] and not math.isnan(factor[t, j]):
                    values.append(factor[t, j])
            mean = sum(values) / len(values) if values else 0.0
            total = 0.0
            for j in range(n_len):
                w = 0.0 if math.isnan(weights[t, j]) else weights[t, j]
                if universe[t, j] and not math.isnan(factor[t, j]):
                    total += w * (factor[t, j] - mean)
            exposure[t] = total
        out[name] = exposure
    return out


def ref_labels(dates, by):
    out = []
    for date in dates:
        s = str(date)
        if by == "year":
            out.append(s[:4])
        else:
            out.append(f"{s[:4]}Q{(int(s[4:6]) - 1) // 3 + 1}")
    return out


def ref_performance(pnl, turnover, periods):
    p = []
    t = []
    for i in range(len(pnl)):
        if not math.isnan(pnl[i]):
            p.append(pnl[i])
            t.append(0.0 if math.isnan(turnover[i]) else turnover[i])
    keys = ("return", "sharpe", "turnover", "maxdrawdown", "margin")
    out = dict.fromkeys(keys)
    n = len(p)
    if n == 0:
        return out
    mean = sum(p) / n
    out["return"] = mean * periods
    variance = sum((v - mean) ** 2 for v in p) / (n - 1) if n >= 2 else 0.0
    sd = math.sqrt(variance)
    if n >= 2 and sd != 0.0:
        out["sharpe"] = mean / sd * math.sqrt(periods)
    out["turnover"] = sum(t) / n
    cumulative = 0.0
    peak = 0.0
    max_dd = 0.0
    for v in p:
        cumulative += v
        if cumulative > peak:
            peak = cumulative
        if peak - cumulative > max_dd:
            max_dd = peak - cumulative
    out["maxdrawdown"] = max_dd
    total_turnover = sum(t)
    if total_turnover != 0.0:
        out["margin"] = sum(p) / total_turnover * 10000.0
    return out


def ref_curves(pnl):
    p = [0.0 if math.isnan(v) else v for v in pnl]
    cumsum = []
    cumprod = []
    run_sum = 0.0
    run_prod = 1.0
    for v in p:
        run_sum += v
        run_prod *= 1.0 + v
        cumsum.append(run_sum)
        cumprod.append(run_prod - 1.0)
    return np.asarray(cumsum), np.asarray(cumprod)


# ---- ts_delay ------------------------------------------------------------------


def test_ts_delay_hand():
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    expected = np.array([[np.nan, np.nan], [1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_allclose(m.ts_delay(x, 1), expected, equal_nan=True)
    np.testing.assert_allclose(m.ts_delay(x, 2)[2], [1.0, 2.0], equal_nan=True)
    assert np.isnan(m.ts_delay(x, 3)).all()
    # d = 0 返回拷贝，且不改原数组
    zero = m.ts_delay(x, 0)
    np.testing.assert_array_equal(zero, x)
    zero[0, 0] = -1.0
    assert x[0, 0] == 1.0
    with pytest.raises(ValueError):
        m.ts_delay(x, -1)


def test_ts_delay_random_matches_reference():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(7, 4))
    x[2, 1] = np.nan
    original = x.copy()
    for d in (0, 1, 3, 7, 9):
        np.testing.assert_allclose(m.ts_delay(x, d), ref_delay(x, d), equal_nan=True)
    np.testing.assert_array_equal(x, original)


# ---- risk_factors --------------------------------------------------------------


def _risk_panel():
    t_len = 250
    t = np.arange(t_len, dtype=np.float64)
    close = np.column_stack([np.full(t_len, 100.0), 100.0 + t, 100.0 - t])
    mv = np.column_stack(
        [np.full(t_len, 1000.0), np.full(t_len, 10000.0), np.full(t_len, 5000.0)]
    )
    mv[10, 0] = -1.0
    return close, mv


def test_risk_factors_hand_values():
    close, mv = _risk_panel()
    risk = m.risk_factors(close, mv)

    assert risk["size"][0, 0] == pytest.approx(math.log(1000.0))
    assert risk["size"][0, 1] == pytest.approx(math.log(10000.0))
    assert risk["size"][0, 2] == pytest.approx(math.log(5000.0))
    assert np.isnan(risk["size"][10, 0])

    # 第 0 列价格不变，反转与动量都是 0
    assert risk["reversal_20"][60, 0] == pytest.approx(0.0)
    assert risk["momentum"][240, 0] == pytest.approx(0.0)

    # 第 1 列价格 100 + t
    assert risk["reversal_20"][60, 1] == pytest.approx(-1.0 / 7.0)
    assert risk["reversal_60"][60, 1] == pytest.approx(-0.6)
    assert risk["momentum"][240, 1] == pytest.approx(2.2)
    assert risk["momentum"][249, 1] == pytest.approx(329.0 / 109.0 - 1.0)
    assert risk["reversal_20"][249, 1] == pytest.approx(1.0 - 349.0 / 329.0)

    # 第 2 列价格 100 − t：t = 0..99 为正，之后被当 NaN
    assert risk["reversal_20"][99, 2] == pytest.approx(1.0 - 1.0 / 21.0)
    assert np.isnan(risk["reversal_20"][100, 2])
    assert np.isnan(risk["reversal_20"][120, 2])

    # 前若干行必须是 NaN
    assert np.isnan(risk["reversal_20"][:20]).all()
    assert np.isnan(risk["reversal_60"][:60]).all()
    assert np.isnan(risk["momentum"][:240]).all()
    assert not np.isnan(risk["reversal_20"][20:, :2].copy()).any()


def test_risk_factors_random_matches_reference():
    rng = np.random.default_rng(0)
    steps = rng.normal(0.0, 0.02, size=(260, 4))
    close = 50.0 * np.exp(np.cumsum(steps, axis=0))
    close[3, 0] = 0.0
    close[4, 1] = -2.0
    close[5, 2] = np.nan
    mv = rng.lognormal(mean=10.0, sigma=0.5, size=(260, 4))
    mv[7, 3] = 0.0
    mv[8, 0] = np.nan
    close_before = close.copy()
    mv_before = mv.copy()

    got = m.risk_factors(close, mv)
    want = ref_risk_factors(close, mv)
    for name in ("size", "reversal_20", "reversal_60", "momentum"):
        np.testing.assert_allclose(got[name], want[name], equal_nan=True)
    np.testing.assert_array_equal(close, close_before)
    np.testing.assert_array_equal(mv, mv_before)


# ---- risk_exposure -------------------------------------------------------------


def test_risk_exposure_hand():
    weights = np.array(
        [[0.5, -0.5, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, -1.0, 0.0]]
    )
    factor = np.array(
        [[1.0, 3.0, 5.0], [2.0, 4.0, np.nan], [10.0, 20.0, 30.0], [5.0, 5.0, 9.0]]
    )
    universe = np.array(
        [
            [True, True, True],
            [True, False, True],
            [True, True, True],
            [True, True, True],
        ]
    )
    got = m.risk_exposure(weights, {"f": factor}, universe)["f"]
    np.testing.assert_allclose(got, [-1.0, 0.0, np.nan, 0.0], equal_nan=True)


def test_risk_exposure_constant_shift_invariant():
    weights = np.array([[0.5, -0.5, 0.0], [0.3, 0.2, -0.5], [0.0, 0.0, 0.0]])
    factor = np.array([[1.0, 3.0, 5.0], [2.0, np.nan, 8.0], [10.0, 20.0, 30.0]])
    universe = np.ones((3, 3), dtype=bool)
    base = m.risk_exposure(weights, {"f": factor}, universe)["f"]
    shifted = m.risk_exposure(weights, {"f": factor + 7.0}, universe)["f"]
    np.testing.assert_allclose(base, shifted, equal_nan=True)


def test_risk_exposure_boundaries():
    # 整行权重 NaN / 全 0 → NaN
    weights = np.array([[np.nan, np.nan, np.nan], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    factor = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [5.0, 6.0, 7.0]])
    universe = np.ones((3, 3), dtype=bool)
    got = m.risk_exposure(weights, {"f": factor}, universe)["f"]
    assert np.isnan(got[0]) and np.isnan(got[1])
    # 三只都有效，均值为 6，第 0 只去均值后是 −1
    assert got[2] == pytest.approx(-1.0)
    # 只有一只有效股票 → 去均值后为 0
    one = m.risk_exposure(
        np.array([[1.0, 0.0, 0.0]]),
        {"f": np.array([[5.0, 6.0, 7.0]])},
        np.array([[True, False, False]]),
    )["f"]
    assert one[0] == pytest.approx(0.0)
    # 因子整行 NaN → 均值按 0、暴露为 0
    blank = m.risk_exposure(
        np.array([[1.0, 1.0, 0.0]]),
        {"f": np.array([[np.nan, np.nan, np.nan]])},
        np.ones((1, 3), dtype=bool),
    )["f"]
    assert blank[0] == pytest.approx(0.0)


def test_risk_exposure_random_matches_reference():
    rng = np.random.default_rng(0)
    t_len, n_len = 8, 5
    weights = rng.normal(size=(t_len, n_len))
    weights[0, :] = np.nan
    weights[1, :] = 0.0
    weights[2, 3] = np.nan
    f1 = rng.normal(size=(t_len, n_len))
    f2 = rng.normal(size=(t_len, n_len))
    f2[4, :] = np.nan
    f1[5, 1] = np.nan
    universe = rng.random((t_len, n_len)) > 0.3
    universe[6, :] = False
    risk = {"f1": f1, "f2": f2}
    got = m.risk_exposure(weights, risk, universe)
    want = ref_exposure(weights, risk, universe)
    for name in risk:
        np.testing.assert_allclose(got[name], want[name], equal_nan=True)


# ---- period_labels -------------------------------------------------------------


def test_period_labels_hand():
    dates = ["20211231", "20210104", "20210401", "20211001"]
    assert list(m.period_labels(dates, "year")) == ["2021"] * 4
    assert list(m.period_labels(dates, "quarter")) == [
        "2021Q4",
        "2021Q1",
        "2021Q2",
        "2021Q4",
    ]
    with pytest.raises(ValueError):
        m.period_labels(dates, "month")


def test_period_labels_random_matches_reference():
    rng = np.random.default_rng(0)
    dates = []
    for _ in range(30):
        year = int(rng.integers(2019, 2024))
        month = int(rng.integers(1, 13))
        day = int(rng.integers(1, 29))
        dates.append(f"{year:04d}{month:02d}{day:02d}")
    for by in ("year", "quarter"):
        assert list(m.period_labels(dates, by)) == ref_labels(dates, by)


# ---- performance ---------------------------------------------------------------


def test_performance_hand():
    pnl = [-0.02, 0.01, 0.03, -0.01, 0.04]
    turnover = [0.1, 0.2, 0.1, 0.3, 0.2]
    got = m.performance(pnl, turnover)
    # 第一天就亏，回撤从 0 起算：累计 −0.02 → 回撤 0.02
    assert got["return"] == pytest.approx(2.4)
    assert got["sharpe"] == pytest.approx(6.076436202502)
    assert got["turnover"] == pytest.approx(0.18)
    assert got["maxdrawdown"] == pytest.approx(0.02)
    assert got["margin"] == pytest.approx(555.5555555555555)


def test_performance_boundaries():
    empty = m.performance([np.nan, np.nan], [0.1, 0.2])
    assert all(v is None for v in empty.values())

    single = m.performance([0.01], [0.1])
    assert single["return"] == pytest.approx(2.4)
    assert single["sharpe"] is None
    assert single["turnover"] == pytest.approx(0.1)
    assert single["maxdrawdown"] == pytest.approx(0.0)
    assert single["margin"] == pytest.approx(1000.0)

    flat = m.performance([0.01, 0.01], [0.0, 0.0])
    assert flat["sharpe"] is None
    assert flat["margin"] is None
    assert flat["maxdrawdown"] == pytest.approx(0.0)

    # pnl 为 NaN 的日子连同 turnover 一起丢掉，turnover 的 NaN 当 0
    mixed = m.performance([-0.1, np.nan, 0.2], [0.05, 0.99, np.nan])
    assert mixed["turnover"] == pytest.approx(0.025)
    assert mixed["margin"] == pytest.approx(0.1 / 0.05 * 10000.0)
    # 先 −0.1 再 +0.2：累计 −0.1、0.1，回撤 0.1
    assert mixed["maxdrawdown"] == pytest.approx(0.1)


def test_performance_random_matches_reference():
    rng = np.random.default_rng(0)
    pnl = rng.normal(0.0005, 0.01, size=30)
    turnover = rng.uniform(0.0, 0.5, size=30)
    pnl[4] = np.nan
    pnl[17] = np.nan
    turnover[9] = np.nan
    got = m.performance(pnl, turnover, periods=252)
    want = ref_performance(pnl, turnover, 252)
    for key in want:
        if want[key] is None:
            assert got[key] is None
        else:
            assert got[key] == pytest.approx(want[key])


# ---- summary_table -------------------------------------------------------------


_SUMMARY_DATES = [
    "20201231",
    "20210104",
    "20210331",
    "20210401",
    "20210630",
    "20211008",
]
_SUMMARY_PNL = [0.01, -0.02, 0.03, 0.01, 0.02, -0.01]
_SUMMARY_TURNOVER = [0.1, 0.2, 0.15, 0.05, 0.1, 0.2]


def test_summary_table_by_year_hand():
    df = m.summary_table(_SUMMARY_PNL, _SUMMARY_TURNOVER, _SUMMARY_DATES, by="year")
    assert df.columns == [
        "period",
        "return",
        "sharpe",
        "turnover",
        "maxdrawdown",
        "margin",
    ]
    assert df.height == 3
    assert df["period"].to_list() == ["2020", "2021", "ALL"]
    for name in ("return", "sharpe", "turnover", "maxdrawdown", "margin"):
        assert df.schema[name] == pl.Float64

    rows = df.iter_rows(named=True)
    first = next(rows)
    assert first["return"] == pytest.approx(2.4)
    assert first["sharpe"] is None
    assert first["turnover"] == pytest.approx(0.1)
    assert first["maxdrawdown"] == pytest.approx(0.0)
    assert first["margin"] == pytest.approx(1000.0)

    second = next(rows)
    assert second["return"] == pytest.approx(1.44)
    assert second["sharpe"] == pytest.approx(4.48252420582577)
    assert second["turnover"] == pytest.approx(0.14)
    assert second["maxdrawdown"] == pytest.approx(0.02)
    assert second["margin"] == pytest.approx(428.5714285714285)

    last = next(rows)
    overall = m.performance(_SUMMARY_PNL, _SUMMARY_TURNOVER)
    assert last["return"] == pytest.approx(overall["return"])
    assert last["sharpe"] == pytest.approx(overall["sharpe"])
    assert last["turnover"] == pytest.approx(overall["turnover"])
    assert last["maxdrawdown"] == pytest.approx(overall["maxdrawdown"])
    assert last["margin"] == pytest.approx(overall["margin"])


def test_summary_table_by_quarter_hand():
    df = m.summary_table(_SUMMARY_PNL, _SUMMARY_TURNOVER, _SUMMARY_DATES, by="quarter")
    assert df.height == 5
    assert df["period"].to_list() == ["2020Q4", "2021Q1", "2021Q2", "2021Q4", "ALL"]

    rows = list(df.iter_rows(named=True))
    assert rows[1]["return"] == pytest.approx(1.2)
    assert rows[1]["sharpe"] == pytest.approx(2.190890230020664)
    assert rows[1]["turnover"] == pytest.approx(0.175)
    assert rows[1]["maxdrawdown"] == pytest.approx(0.02)
    assert rows[1]["margin"] == pytest.approx(285.7142857142857)

    assert rows[2]["return"] == pytest.approx(3.6)
    assert rows[2]["sharpe"] == pytest.approx(32.863353450309965)
    assert rows[2]["turnover"] == pytest.approx(0.075)
    assert rows[2]["maxdrawdown"] == pytest.approx(0.0)
    assert rows[2]["margin"] == pytest.approx(2000.0)

    assert rows[3]["return"] == pytest.approx(-2.4)
    assert rows[3]["sharpe"] is None
    assert rows[3]["maxdrawdown"] == pytest.approx(0.01)
    assert rows[3]["margin"] == pytest.approx(-500.0)

    overall = m.performance(_SUMMARY_PNL, _SUMMARY_TURNOVER)
    for name in ("return", "sharpe", "turnover", "maxdrawdown", "margin"):
        assert rows[-1][name] == pytest.approx(overall[name])


def test_summary_table_random_matches_reference():
    rng = np.random.default_rng(0)
    t_len = 40
    pnl = rng.normal(0.001, 0.02, size=t_len)
    turnover = rng.uniform(0.0, 0.4, size=t_len)
    pnl[3] = np.nan
    turnover[11] = np.nan
    dates = []
    for _ in range(t_len):
        year = int(rng.integers(2021, 2023))
        month = int(rng.integers(1, 13))
        day = int(rng.integers(1, 29))
        dates.append(f"{year:04d}{month:02d}{day:02d}")

    labels = ref_labels(dates, "quarter")
    df = m.summary_table(pnl, turnover, dates, by="quarter")
    assert df["period"].to_list() == sorted(set(labels)) + ["ALL"]
    for row, label in zip(
        df.iter_rows(named=True), sorted(set(labels)) + [None], strict=True
    ):
        if label is None:
            mask = [True] * t_len
        else:
            mask = [v == label for v in labels]
        want = ref_performance(
            [pnl[i] for i in range(t_len) if mask[i]],
            [turnover[i] for i in range(t_len) if mask[i]],
            240,
        )
        for name in ("return", "sharpe", "turnover", "maxdrawdown", "margin"):
            if want[name] is None:
                assert row[name] is None
            else:
                assert row[name] == pytest.approx(want[name])


# ---- curves --------------------------------------------------------------------


def test_curves_hand():
    pnl = [0.1, np.nan, -0.05, 0.02]
    got = m.curves(pnl)
    np.testing.assert_allclose(got["cumsum"], [0.1, 0.1, 0.05, 0.07])
    np.testing.assert_allclose(got["cumprod"], [0.1, 0.1, 0.045, 0.0659])
    # 不改原数组
    assert math.isnan(pnl[1])


def test_curves_random_matches_reference():
    rng = np.random.default_rng(0)
    pnl = rng.normal(0.0, 0.02, size=20)
    pnl[2] = np.nan
    pnl[13] = np.nan
    original = pnl.copy()
    got = m.curves(pnl)
    want_sum, want_prod = ref_curves(pnl)
    np.testing.assert_allclose(got["cumsum"], want_sum)
    np.testing.assert_allclose(got["cumprod"], want_prod)
    np.testing.assert_array_equal(pnl, original)
