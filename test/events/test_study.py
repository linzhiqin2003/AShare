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
    st = event_study(
        ev, hand_returns(), market, DATES, SYMS, window=(-1, 2), reps=50, complete=False
    )
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
    np.testing.assert_array_equal(
        st.caar()["n"].to_numpy(), [4, 4, 4, 3]
    )  # C 第 +2 天缺失
    # 默认剔除窗口不完整的事件：C 第 7 天缺失
    full = event_study(ev, hand_returns(), market, DATES, SYMS, window=(-1, 2), reps=50)
    assert len(full) == 3 and full.dropped["incomplete_window"] == 1
    np.testing.assert_allclose(full.car(-1, 2), car[[0, 1, 3]])


def test_hand_group_benchmark():
    r = np.array([[0.01, 0.03, 0.05, 0.10], [0.02, np.nan, 0.04, 0.0]])
    g = np.array([[0, 0, 0, 1], [0, 0, 0, -1]])
    out = group_benchmark(r, g, min_size=2)
    np.testing.assert_allclose(out[0], [0.04, 0.03, 0.02, np.nan])
    np.testing.assert_allclose(
        out[1], [0.04, 0.03, 0.02, np.nan]
    )  # 自己缺收益时基准照常给出


@pytest.mark.filterwarnings("ignore:只有")
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
    st = event_study(
        ev,
        ret,
        bench,
        dates,
        syms,
        window=(-5, 8),
        reps=50,
        cluster="date",
        complete=False,
        dedupe=False,
    )
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


def test_groups_carry_cap_through_suspension_and_skip_unlisted():
    ind = np.array([[1, 1, 1, 1]] * 4)
    cap = np.array(
        [
            [1.0, 2.0, 3.0, 4.0],
            [1.0, np.nan, 3.0, 4.0],  # B 第 1 天停牌，没有市值
            [1.0, np.nan, 3.0, 4.0],
            [1.0, 2.0, 3.0, 9.0],
        ]
    )
    listed = np.ones((4, 4), dtype=bool)
    listed[3, 3] = False  # D 第 3 天已退市
    g = size_industry_groups(ind, cap, buckets=2, listed=listed)
    # B 停牌期间沿用停牌前的市值 2（只看过去），仍和 A 同组
    assert g[2, 1] == g[2, 0] == g[3, 1] and g[2, 1] >= 0
    assert g[3, 3] == -1  # 退市后不参与
    # 分位点只用在市的三只：3 天的中位数是 2，所以 C 单独在大的一组
    assert g[3, 2] != g[3, 1]


def test_group_benchmark_members_only():
    r = np.array([[0.0, 0.02, 0.04, 0.06]])
    g = np.zeros((1, 4), dtype=np.int64)
    member = np.array([[False, True, True, True]])  # 第 0 只停牌：收益 0 不计入同组
    out = group_benchmark(r, g, min_size=2, member=member)
    np.testing.assert_allclose(out[0], [0.04, 0.05, 0.04, 0.03])


def test_cluster_se_single_cluster_is_nan():
    se, g = _cluster_se(np.array([1.0, 2.0]), np.array([0, 0]))
    assert np.isnan(se) and g == 1


@pytest.mark.filterwarnings("ignore:只有")
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
    st = event_study(
        ev, ret, market, dates, syms, window=(-5, 10), reps=200, cluster="date"
    )
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
            cluster="date",
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


# ---------- 缺失、重复、形状 ----------


def test_complete_window_rules():
    t = 12
    r = np.full((t, 4), 0.01)
    r[:3, 1] = np.nan  # B 第 3 天才有第一个有效收益（上市）
    r[8:, 2] = np.nan  # C 第 7 天之后退市
    bench = np.zeros((t, 4))
    bench[6, 3] = np.nan  # D 第 6 天基准组不足：基准缺失
    syms = np.array(["A", "B", "C", "D"])
    ev = frame(["A", "B", "C", "D"], [DATES[4], DATES[4], DATES[5], DATES[4]])
    st = event_study(ev, r, bench, DATES, syms, window=(-2, 3), reps=20)
    # B：窗口起点 2 早于上市 3 → listed_too_short；D：基准缺失 → incomplete_window；
    # C：退市后的格子记 0、保留（不然就是只挑活下来的股票）
    assert st.dropped["listed_too_short"] == 1
    assert st.dropped["incomplete_window"] == 1
    assert st.events["symbol"].to_list() == ["A", "C"]
    np.testing.assert_allclose(st.car(-2, 3), [0.06, 0.05])  # C 第 8 天退市，记 0
    assert st.coverage[1].sum() == 5
    loose = event_study(
        ev, r, bench, DATES, syms, window=(-2, 3), reps=20, complete=False
    )
    assert len(loose) == 4 and loose.dropped["incomplete_window"] == 0
    # 基准缺失不当 0：宽松口径下该格记缺失（AR 0），而不是记成个股收益本身
    d = loose.events["symbol"].to_list().index("D")
    np.testing.assert_allclose(loose.car(-2, 3)[d], 0.05)


def test_same_day_same_stock_events_merge():
    r = np.zeros((12, 3))
    ev = pl.DataFrame(
        {
            "symbol": ["A", "A", "A", "B"],
            "date": [DATES[4]] * 4,
            "kind": ["k", "k", "other", "k"],
            "timing": ["pre_open"] * 4,
        }
    )
    st = event_study(ev, r, np.zeros(12), DATES, SYMS, window=(-1, 1), reps=10)
    assert len(st) == 3 and st.dropped["same_day_duplicates"] == 1
    st2 = event_study(
        ev, r, np.zeros(12), DATES, SYMS, window=(-1, 1), reps=10, dedupe=False
    )
    assert len(st2) == 4


def test_shape_errors_are_readable():
    r = np.zeros((12, 3))
    ev = frame(["A"], [DATES[4]])
    with pytest.raises(ValueError, match="returns 形状"):
        event_study(ev, r[:, :2], np.zeros(12), DATES, SYMS)
    with pytest.raises(ValueError, match="benchmark 形状"):
        event_study(ev, r, np.zeros(11), DATES, SYMS)
    with pytest.raises(ValueError, match="can_buy 形状"):
        event_study(ev, r, np.zeros(12), DATES, SYMS, can_buy=np.ones((12, 2), bool))
    with pytest.raises(ValueError, match="cluster"):
        event_study(ev, r, np.zeros(12), DATES, SYMS, cluster="week")
    st = event_study(ev, r, np.zeros(12), DATES, SYMS, window=(-1, 1), reps=10)
    with pytest.raises(ValueError, match="mask"):
        st.test(0, 1, mask=np.ones(3, bool))
    with pytest.raises(ValueError, match="cluster 标签"):
        st.test(0, 1, cluster=np.zeros(3))


def test_few_clusters_warn():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.01, (12, 3))
    ev = frame(["A", "B", "C"], [DATES[3], DATES[4], DATES[5]])
    st = event_study(ev, r, np.zeros(12), DATES, SYMS, window=(-1, 2), reps=10)
    with pytest.warns(RuntimeWarning, match="聚类"):
        st.test(0, 2)


# ---------- 日历时间组合检验 ----------


def newey_west_loop(r, lags):
    n = len(r)
    m = sum(r) / n
    var = sum((x - m) ** 2 for x in r) / n
    for k in range(1, lags + 1):
        cov = sum((r[i] - m) * (r[i - k] - m) for i in range(k, n)) / n
        var += 2 * (1 - k / (lags + 1)) * cov
    return (var / n) ** 0.5


@pytest.mark.parametrize("seed", range(5))
def test_calendar_time_matches_loop(seed):
    rng = np.random.default_rng(seed)
    t, n, m = 90, 8, 40
    dates = np.array([str(20200101 + i) for i in range(t)])
    syms = np.array([f"S{i}" for i in range(n)])
    ret = rng.normal(0, 0.02, (t, n))
    bench = rng.normal(0, 0.01, t)
    ev = pl.DataFrame(
        {
            "symbol": rng.choice(syms, m).tolist(),
            "date": dates[rng.integers(10, t - 15, m)].tolist(),
            "kind": ["k"] * m,
            "timing": ["pre_open"] * m,
        }
    )
    st = event_study(ev, ret, bench, dates, syms, window=(-3, 6), reps=10)
    a, b = 1, 5
    # 逐日：找出当天处在 [a, b] 的事件，等权平均它们当天的异常收益
    daily = []
    for day in range(t):
        xs = [
            ret[day, c] - bench[day]
            for c, av in zip(
                [list(syms).index(s) for s in st.events["symbol"].to_list()],
                st.avail,
                strict=True,
            )
            if a <= day - av <= b
        ]
        if xs:
            daily.append(sum(xs) / len(xs))
    res = st.calendar_time(a, b, lags=3).row(0, named=True)
    assert res["days"] == len(daily)
    np.testing.assert_allclose(res["mean_daily"], np.mean(daily), rtol=1e-12)
    np.testing.assert_allclose(res["se_nw"], newey_west_loop(daily, 3), rtol=1e-10)
    np.testing.assert_allclose(res["implied_car"], np.mean(daily) * 5, rtol=1e-12)


# ---------- 审查给的两个扎堆场景：拒绝率回归 ----------


def review_scenario(name, rep):
    """T=500、N=300、e~N(0, 0.02)。

    c：10 个行业 × 30 只，行业因子 AR(1) φ=0.3、扰动 sd 1%；25 个「行业-周」各 8 只事件，
       同一组的事件日落在相邻 5 天内；基准 = 全市场等权；看 CAR[1,20]。
    d：200 个高 β（1.5）事件挤在连续 40 天里，基准 = 市场 m~N(0, 0.012)；看 CAR[1,60]。
    """
    big_t = 500
    days = np.busday_offset(
        np.datetime64("2020-01-01"), np.arange(big_t), roll="forward"
    )
    dates = np.array([str(d).replace("-", "") for d in days])
    rng = np.random.default_rng({"c": 3, "d": 4}[name] * 100000 + rep)
    n = 300
    e = rng.normal(0, 0.02, (big_t, n))
    if name == "c":
        ind = np.repeat(np.arange(10), 30)
        f = np.zeros((big_t, 10))
        u = rng.normal(0, 0.01, (big_t, 10))
        for i in range(1, big_t):
            f[i] = 0.3 * f[i - 1] + u[i]
        ret = e + f[:, ind]
        bench = ret.mean(1)
        cols, di = [], []
        for _ in range(25):
            i = rng.integers(10)
            d0 = rng.integers(30, big_t - 70)
            cols += list(rng.choice(np.flatnonzero(ind == i), 8, replace=False))
            di += list(d0 + rng.integers(0, 5, 8))
    else:
        m = rng.normal(0, 0.012, big_t)
        beta = np.full(n, 1.0)
        beta[:200] = 1.5
        ret = e + beta * m[:, None]
        bench = m
        cols = rng.choice(200, 200, replace=False)
        di = 30 + np.sort(rng.integers(0, 40, 200))
    ev = pl.DataFrame(
        {
            "symbol": [str(c) for c in cols],
            "date": list(dates[np.asarray(di)]),
            "kind": ["x"] * len(cols),
            "timing": ["pre_open"] * len(cols),
        }
    )
    syms = np.array([str(i) for i in range(n)])
    return event_study(ev, ret, bench, dates, syms, window=(-5, 60), reps=50, seed=rep)


@pytest.mark.filterwarnings("ignore:只有")
@pytest.mark.parametrize("name", ["c", "d"])
def test_review_scenarios_rejection_rates(name):
    """名义水平 5%。容差：200 次模拟时拒绝率的抽样误差约 ±3 个百分点（95%），
    日历时间检验要求不超过 10%；按日聚类在两个场景都明显超标（审查实测 18%、59%），
    这里只要求它超过 12%，用来证明场景确实有扎堆；按月聚类在场景 d 只有 2–3 个聚类、
    仍然超标（审查实测 30%），所以只在场景 c 要求它不超过 11%。
    """
    a, b = (1, 20) if name == "c" else (1, 60)
    sims = 200
    hits = {"calendar": 0, "month": 0, "date": 0}
    for rep in range(sims):
        st = review_scenario(name, rep)
        hits["calendar"] += st.calendar_time(a, b).row(0, named=True)["p_nw"] < 0.05
        hits["month"] += st.test(a, b).row(0, named=True)["p_cluster"] < 0.05
        hits["date"] += (
            st.test(a, b, cluster="date").row(0, named=True)["p_cluster"] < 0.05
        )
    rate = {k: v / sims for k, v in hits.items()}
    assert rate["calendar"] <= 0.10, rate
    assert rate["date"] > 0.12, rate
    if name == "c":
        assert rate["month"] <= 0.11, rate
    else:
        assert rate["month"] > 0.15, rate


def test_group_benchmark_own_return_missing_uses_own_group():
    """自己当天没有收益（或不计入）时，基准仍取自己当天所在组，不能落到别的日子、别的组。"""
    r = np.array([[0.10, 0.10, 0.10, 0.50], [np.nan, 0.01, 0.03, 0.05]])
    g = np.array([[0, 0, 0, 1], [1, 1, 1, 0]])
    out = group_benchmark(r, g, min_size=2)
    np.testing.assert_allclose(out[1, 0], 0.02)
