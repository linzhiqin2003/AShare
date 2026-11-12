import numpy as np
import polars as pl
import pytest

from ashare.events import (
    daily_returns,
    event_study,
    group_benchmark,
    size_industry_groups,
)
from ashare.events.study import _cluster_se
from test.events.reference import cluster_se, reference_car

DATES = np.array([str(20240101 + i) for i in range(12)])
SYMS = np.array(["A", "B", "C"])


def frame(symbols, dates, timing="pre_open"):
    return pl.DataFrame(
        {
            "symbol": symbols,
            "date": dates,
            "kind": ["k"] * len(symbols),
            "timing": [timing] * len(symbols),
        }
    )


def hand_returns():
    r = np.zeros((12, 3))
    r[:, 0] = 0.01 * np.arange(12)  # A：第 i 天 i%
    r[:, 1] = -0.02
    r[:, 2] = 0.005
    r[7, 2] = np.nan  # C 第 7 天停牌缺失
    return r


def test_hand_example():
    market = np.full(12, 0.001)
    ev = frame(
        ["A", "B", "C", "A", "C"],
        ["20240103", "20240105", "20240106", "20240109", "20240111"],
    )
    st = event_study(ev, hand_returns(), market, DATES, SYMS, window=(-1, 2), reps=50)
    # 第 5 个事件可成交日 10，窗口到 12 出界，丢弃
    assert len(st) == 4 and st.dropped["window_outside_panel"] == 1
    car = st.car(-1, 2)
    # A@2: (0.01+0.02+0.03+0.04) − 4×0.001；B@4: −0.08 − 0.004；C@5: 第 7 天缺失记 0
    np.testing.assert_allclose(
        car, [0.096, -0.084, 0.005 * 3 - 0.003, (0.07 + 0.08 + 0.09 + 0.10) - 0.004]
    )
    assert st.coverage.sum() == 15
    np.testing.assert_allclose(st.caar()["caar"].to_numpy()[-1], np.mean(car))
    np.testing.assert_allclose(st.car(0, 0), [0.019, -0.021, 0.004, 0.079])


def test_hand_group_benchmark():
    r = np.array([[0.01, 0.03, 0.05, 0.10], [0.02, np.nan, 0.04, 0.0]])
    g = np.array([[0, 0, 0, 1], [0, 0, 0, -1]])
    out = group_benchmark(r, g, min_size=2)
    np.testing.assert_allclose(out[0], [0.04, 0.03, 0.02, np.nan])
    np.testing.assert_allclose(
        out[1], [0.04, 0.03, 0.02, np.nan]
    )  # 自己缺收益时基准照常给出


@pytest.mark.parametrize("seed", range(10))
def test_matches_reference(seed):
    rng = np.random.default_rng(seed)
    t, n, m = 80, 7, 60
    dates = np.array([str(20200101 + 2 * i) for i in range(t)])
    syms = np.array([f"S{i}" for i in range(n)])
    ret = rng.normal(0, 0.02, (t, n))
    ret[rng.random((t, n)) < 0.1] = np.nan
    bench = rng.normal(0, 0.01, (t, n)) if seed % 2 else rng.normal(0, 0.01, t)
    ev = pl.DataFrame(
        {
            "symbol": rng.choice(syms, m).tolist(),
            "date": [str(20200101 + int(x)) for x in rng.integers(-5, 2 * t + 5, m)],
            "kind": ["k"] * m,
            "timing": rng.choice(["after_close", "pre_open"], m).tolist(),
        }
    )
    st = event_study(ev, ret, bench, dates, syms, window=(-5, 8), reps=50)
    rows = list(
        ev.with_row_index("_o")
        .sort(["date", "symbol", "kind", "_o"])
        .iter_rows(named=True)
    )
    cars, days = reference_car(rows, dates, syms, ret, bench, -5, 8)
    np.testing.assert_allclose(st.car(-5, 8), cars, atol=1e-13)
    res = st.test(-5, 8).row(0, named=True)
    np.testing.assert_allclose(
        res["se_cluster"], cluster_se(cars, list(days)), rtol=1e-12
    )
    np.testing.assert_allclose(
        res["se_naive"], np.std(cars, ddof=1) / np.sqrt(len(cars)), rtol=1e-12
    )
    assert res["clusters"] < res["events"]  # 用例里确实有同日扎堆的事件


def test_daily_returns_suspension():
    close = np.array([[10.0, np.nan], [11.0, 5.0], [np.nan, 5.5], [12.1, 6.05]])
    adj = np.ones_like(close)
    adj[3, 0] = 2.0  # 第 3 天复权因子翻倍（10 送 10），收益按复权价算
    quote = np.isfinite(close)
    r = daily_returns(close, adj, quote)
    np.testing.assert_allclose(r[:, 0], [np.nan, 0.1, 0.0, 12.1 * 2 / 11 - 1])
    np.testing.assert_allclose(r[:, 1], [np.nan, np.nan, 0.1, 0.1])
    listed = np.ones_like(quote)
    listed[3, 1] = False
    assert np.isnan(daily_returns(close, adj, quote, listed)[3, 1])


def test_size_industry_groups():
    ind = np.array([[1, 1, 1, 1, 2, 0]] * 3)
    cap = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]] * 3)
    g = size_industry_groups(ind, cap, buckets=2)
    assert (g[0] == -1).all()  # 第一天没有前一天市值
    assert g[1, 5] == -1  # 行业缺失
    assert g[1, 0] == g[1, 1] == g[1, 2] and g[1, 3] != g[1, 0]  # 分位点 3.5
    assert g[1, 4] not in g[1, :4]


def test_cluster_se_single_cluster_is_nan():
    se, g = _cluster_se(np.array([1.0, 2.0]), np.array([0, 0]))
    assert np.isnan(se) and g == 1


def test_tradable_split_and_years():
    rng = np.random.default_rng(1)
    ret = rng.normal(0, 0.01, (12, 3))
    can_buy = np.ones((12, 3), dtype=bool)
    can_buy[3, 1] = False
    ev = frame(["A", "B", "C"], ["20240103", "20240104", "20240105"])
    st = event_study(
        ev, ret, np.zeros(12), DATES, SYMS, window=(-1, 3), can_buy=can_buy, reps=20
    )
    assert st.events["entry_buyable"].to_list() == [True, False, True]
    out = st.split(
        st.events["entry_buyable"].to_numpy(), 0, 3, labels=("buyable", "limit")
    )
    assert out["events"].to_list() == [2, 1]
    assert st.by_year().rows() == [("2024", 3)]
    with pytest.raises(ValueError):
        st.car(-2, 0)


# ---------- 合成数据的已知答案 ----------


def synthetic(rng, *, drift=0.0, shock=0.0, t=320, n=400, dates_used=40, per_date=10):
    """事件日彼此相隔 7 天以上；同一天的事件在窗口内共享冲击 shock。"""
    dates = np.array([str(20000101 + i) for i in range(t)])
    syms = np.array([f"S{i}" for i in range(n)])
    ret = rng.normal(0, 0.02, (t, n))
    market = ret.mean(axis=1)
    days = 20 + 7 * np.arange(dates_used)
    rows = []
    for d in days:
        cols = rng.choice(n, per_date, replace=False)
        common = rng.normal(0, shock, 6) if shock else np.zeros(6)
        for c in cols:
            ret[d : d + 6, c] += common + drift
            rows.append((syms[c], dates[d]))
    ev = pl.DataFrame(
        {
            "symbol": [r[0] for r in rows],
            "date": [r[1] for r in rows],
            "kind": ["k"] * len(rows),
            "timing": ["pre_open"] * len(rows),
        }
    )
    return ev, ret, market, dates, syms


def test_drift_is_recovered():
    rng = np.random.default_rng(7)
    ev, ret, market, dates, syms = synthetic(rng, drift=0.004)
    st = event_study(ev, ret, market, dates, syms, window=(-5, 10), reps=200)
    res = st.test(0, 5).row(0, named=True)
    # 注入 6 天 × 0.4% = 2.4%（市场收益在注入前算好）；单个事件 CAR 的标准差约 4.9%，均值误差约 0.25%
    assert abs(res["mean"] - 0.024) < 0.006
    assert res["p_cluster"] < 1e-6 and res["p_naive"] < 1e-6
    pre = st.test(-5, -1).row(0, named=True)
    assert abs(pre["mean"]) < 0.005


def rejection_rates(shock, sims=300):
    rng = np.random.default_rng(11 if shock else 12)
    hits = np.zeros(3)
    ratio = []
    for _ in range(sims):
        ev, ret, market, dates, syms = synthetic(rng, shock=shock)
        st = event_study(
            ev,
            ret,
            market,
            dates,
            syms,
            window=(0, 5),
            reps=200,
            seed=int(rng.integers(1 << 30)),
        )
        r = st.test(0, 5).row(0, named=True)
        hits += [r["p_naive"] < 0.05, r["p_cluster"] < 0.05, r["p_bootstrap"] < 0.05]
        ratio.append(r["se_ratio"])
    return hits / sims, float(np.median(ratio))


def test_null_rejection_near_nominal():
    rates, ratio = rejection_rates(shock=0.0)
    assert 0.01 <= rates[0] <= 0.10
    assert 0.01 <= rates[1] <= 0.10
    assert 0.01 <= rates[2] <= 0.12
    assert 0.8 < ratio < 1.2


def test_clustered_shock_naive_over_rejects():
    rates, ratio = rejection_rates(shock=0.02)
    assert rates[0] > 0.30  # 朴素 t 明显过度拒绝
    assert 0.01 <= rates[1] <= 0.10  # 按事件日聚类回到名义水平附近
    assert 0.01 <= rates[2] <= 0.12
    assert ratio > 2.0
