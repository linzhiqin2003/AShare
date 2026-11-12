"""补充内容：对比指数、胜率、个股贡献、尾部、集中度、多策略对比。

每块都与直白写法的参考实现对拍；个股贡献另与引擎自己累计的 `result.contribution` 核对。
"""

from __future__ import annotations

import json
import math

import numpy as np
import panelbt
import pytest
from scipy import stats as sps

import ashare.report as rp
from ashare.report import Benchmark, build_tearsheet
from ashare.report import extras as ex
from ashare.report.render import EXTRA_KEYS, compact, to_html
from test.report.helpers import build, business_days, make_case

ORIG_KEYS = (
    "meta",
    "summary",
    "daily",
    "monthly",
    "yearly",
    "drawdowns",
    "rolling",
    "rebalances",
    "holdings_latest",
    "industry_weights",
    "attribution",
)


@pytest.fixture(scope="module")
def case():
    return make_case(T=900, N=10, seed=11, every=21)


@pytest.fixture(scope="module")
def ts(case):
    return build(case)


def action_case(T=420, N=6, seed=21):
    """有现金分红、送转和停牌（收盘价缺失）的小回测。"""
    rng = np.random.default_rng(seed)
    close = 10 * np.cumprod(1 + rng.normal(0.0005, 0.015, (T, N)), axis=0)
    div = np.zeros((T, N))
    split = np.ones((T, N))
    for t in range(30, T, 60):
        div[t, rng.integers(0, N)] = 0.3
        split[t + 5, rng.integers(0, N)] = 1.5
    close_obs = close.copy()
    # 停牌：收盘价缺失，含一个停牌期间的除权日
    close_obs[100:110, 2] = np.nan
    div[105, 2] = 0.2
    split[107, 2] = 1.2
    can = np.isfinite(close_obs)
    market = panelbt.Market(
        close_obs.copy(),
        close_obs.copy(),
        can.copy(),
        can.copy(),
        split=split,
        cash_div=div,
    )
    days = list(range(1, T - 1, 15))
    ptr, cols, w = [0], [], []
    for _ in days:
        ch = sorted(rng.choice(N, size=4, replace=False).tolist())
        cols += ch
        w += [0.24] * 4
        ptr.append(len(cols))
    targets = panelbt.Targets(
        np.array(days), np.array(ptr), np.array(cols), np.array(w)
    )
    costs = panelbt.Costs(
        commission_rate=3e-4, commission_min=5.0, tax_sell=1e-3, slippage=1e-3
    )
    cfg = panelbt.Config(
        initial_cash=1e6,
        min_qty=100,
        qty_step=100,
        record_trades=True,
        dividend_tax_rate=0.2,
        cash_rate=np.r_[0.0, np.full(T - 1, 1e-4)],  # 第一行是基准日，不计利息
    )
    res = panelbt.run(market, targets, costs, cfg)
    dates = business_days("2021-01-01", T)
    symbols = [f"{600100 + i:06d}.SH" for i in range(N)]
    return res, market, targets, dates, symbols


def test_original_keys_unchanged_and_html_identical(case, ts):
    old = build(case, extras=False)
    assert set(old) == set(ORIG_KEYS)
    for k in ORIG_KEYS:
        assert ts[k] == old[k], k
    assert set(ts) == set(ORIG_KEYS) | set(EXTRA_KEYS)
    assert to_html(ts) == to_html(old)
    with pytest.raises(rp.ReportError):
        build(case, extras=False, benchmarks=[Benchmark(case["bench"], "x", "price")])


# ---------------------------------------------------------------------- 对比指数
def test_extra_benchmark_equal_to_main_matches_summary(case, ts):
    d = build(case, benchmarks=[Benchmark(case["bench"], "同一条", "price", "X")])
    b, s = d["benchmarks"][0], d["summary"]
    assert b["cumulative_return"] == s["benchmark_cumulative_return"]
    assert b["annual_return"] == s["benchmark_annual_return"]
    assert b["annual_volatility"] == s["benchmark_annual_volatility"]
    assert b["max_drawdown"] == s["benchmark_max_drawdown"]
    assert b["excess_annual_return"] == s["excess_annual_return"]
    assert b["information_ratio"] == s["information_ratio"]
    assert b["tracking_error"] == s["tracking_error"]
    assert b["beta"] == s["beta"]
    assert [y["return"] for y in b["yearly"]] == [
        y["benchmark_return"] for y in d["yearly"]
    ]
    w0, w1 = d["win_rates"]
    assert {k: v for k, v in w0.items() if k not in ("name", "symbol", "main")} == {
        k: v for k, v in w1.items() if k not in ("name", "symbol", "main")
    }


def test_extra_benchmark_reference(case):
    rng = np.random.default_rng(99)
    v = 50 * np.cumprod(1 + rng.normal(0.0001, 0.01, len(case["dates"])))
    v[:200] = np.nan
    d = build(case, benchmarks=[Benchmark(v, "晚指数", "total_return")])
    b = d["benchmarks"][0]
    res = case["result"]
    nav = res.nav / res.nav[0]
    r = nav[1:] / nav[:-1] - 1
    s0 = 200
    br = v[s0 + 1 :] / v[s0:-1] - 1
    rr = r[s0:]
    n = len(br)
    assert b["start"] == case["dates"][s0]
    ann = (v[-1] / v[s0]) ** (252 / n) - 1
    sann = (nav[-1] / nav[s0]) ** (252 / n) - 1
    assert b["annual_return"] == pytest.approx(ann, abs=1e-6)
    assert b["excess_annual_return"] == pytest.approx(sann - ann, abs=1e-6)
    te = np.std(rr - br, ddof=1) * math.sqrt(252)
    assert b["tracking_error"] == pytest.approx(te, abs=1e-6)
    assert b["information_ratio"] == pytest.approx(
        np.mean(rr - br) * 252 / te, abs=1e-4
    )
    w = np.r_[1.0, np.cumprod(1 + br)]
    assert b["max_drawdown"] == pytest.approx(
        (w / np.maximum.accumulate(w) - 1).min(), abs=1e-6
    )
    # 起始年指数缺期初：该年为 None
    assert b["yearly"][0]["return"] is None


# ---------------------------------------------------------------------- 胜率
def ref_month_ends(dates):
    """直白写法：每个月最后一个交易日的行号。"""
    out = {}
    for i, d in enumerate(dates):
        if i == 0:
            continue
        out[d[:7]] = i
    return list(out.values())


def test_win_rate_reference(case, ts):
    res = case["result"]
    nav = res.nav / res.nav[0]
    b = case["bench"] / case["bench"][0]
    dates = case["dates"]
    W = ts["win_rates"][0]
    assert W["main"] and W["name"] == "基准指数"
    # 按年
    won = counted = 0
    for y in ts["yearly"]:
        idx = [i for i in range(1, len(dates)) if dates[i][:4] == str(y["year"])]
        s_ = nav[idx[-1]] / nav[idx[0] - 1] - 1
        b_ = b[idx[-1]] / b[idx[0] - 1] - 1
        row = next(r for r in W["yearly"] if r["year"] == y["year"])
        assert row["won"] == (s_ > b_)
        assert row["excess"] == pytest.approx(s_ - b_, abs=2e-6)
        if not y["partial"]:
            counted += 1
            won += s_ > b_
    assert (W["years_won"], W["years_counted"]) == (won, counted)
    assert W["partial_years"] == [y["year"] for y in ts["yearly"] if y["partial"]]
    # 滚动：月末点，首月 / 末月不完整时去掉相应端点
    pts = ref_month_ends(dates)
    if not ts["monthly"]["first_partial"]:
        pts = [0] + pts
    if ts["monthly"]["last_partial"]:
        pts = pts[:-1]
    for k in (1, 3):
        span = 12 * k
        ex_ = [
            (nav[pts[i + span]] / nav[pts[i]] - b[pts[i + span]] / b[pts[i]], i)
            for i in range(len(pts) - span)
        ]
        R = W["rolling"][f"{k}y"]
        if not ex_:
            assert R is None
            continue
        assert R["windows"] == len(ex_)
        assert R["won"] == sum(e > 0 for e, _ in ex_)
        e, i = min(ex_)
        assert R["worst"]["excess"] == pytest.approx(e, abs=2e-6)
        assert R["worst"]["start"] == dates[pts[i]]
        assert R["worst"]["end"] == dates[pts[i + span]]
    assert W["rolling"]["5y"] is None  # 不足 5 年


def test_win_rate_none_without_benchmark(case):
    d = build(case, benchmark=False)
    assert d["win_rates"] is None and d["benchmarks"] == []
    assert "无法算胜率" in rp.Tearsheet(d).text("winrate")


# ---------------------------------------------------------------------- 个股贡献
def ref_contrib(res, market, a, b):
    """直白写法：逐日逐只模拟持仓市值与现金流（不复用 extras 的任何函数）。"""
    pos = np.asarray(res.positions)
    T, N = pos.shape
    close = np.asarray(market.close)
    div = np.zeros((T, N)) if market.cash_div is None else np.asarray(market.cash_div)
    split = np.ones((T, N)) if market.split is None else np.asarray(market.split)
    mark = np.full(N, np.nan)
    price = np.zeros(N)
    dividend = np.zeros(N)
    fees = np.zeros(N)
    prev_value = np.zeros(N)
    trades_by_day = {}
    for tr in res.trades:
        trades_by_day.setdefault(tr.day, []).append(tr)
    tax_rate = 0.2
    for t in range(T):
        for i in range(N):
            if t > 0:
                if mark[i] > 0:
                    mark[i] = max(mark[i] - div[t, i], 0.0)
                mark[i] = mark[i] / split[t, i]
            if np.isfinite(close[t, i]):
                mark[i] = close[t, i]
        value = np.where(pos[t] != 0, pos[t] * np.nan_to_num(mark), 0.0)
        if a < t <= b:
            price += value - prev_value
            for i in range(N):
                g = (pos[t - 1, i] if t > 0 else 0.0) * div[t, i]
                dividend[i] += g - (g * tax_rate if g > 0 else 0.0)
            for tr in trades_by_day.get(t, []):
                amt = abs(tr.qty) * tr.price
                price[tr.col] += -amt if tr.qty > 0 else amt
                fees[tr.col] += tr.commission + tr.tax + tr.other_fee
        prev_value = value
    return price, dividend, fees


def test_contribution_matches_engine_and_reference():
    res, market, targets, dates, symbols = action_case()
    assert np.nansum(res.dividend_gross) > 0  # 用例确实走到分红分支
    assert any(np.asarray(market.split)[np.asarray(res.positions)[:-1].nonzero()] != 1)
    T, N = np.asarray(res.positions).shape
    mk = ex.marks(market, T, N)
    pos = np.asarray(res.positions)
    value = np.where(pos != 0, pos * np.nan_to_num(mk), 0.0)
    flows = ex.security_flows(res, market, T, N)
    p = ex.window_pnl(flows, value[0], value[T - 1], 0, T - 1, N)
    # 引擎自己的逐只累计：现金流 + 期末市值
    np.testing.assert_allclose(p["contribution"], res.contribution, rtol=0, atol=1e-6)
    price, dividend, fees = ref_contrib(res, market, 0, T - 1)
    np.testing.assert_allclose(p["price"], price, atol=1e-6)
    np.testing.assert_allclose(p["dividend"], dividend, atol=1e-6)
    np.testing.assert_allclose(p["fees"], fees, atol=1e-9)
    # 恒等式：各股贡献之和 + 现金利息 = 权益变化；各股之和 = 价格 + 分红 − 费用
    eq = np.asarray(res.equity)
    interest = np.asarray(res.cash_interest)
    assert interest.sum() > 0
    assert p["contribution"].sum() + interest[1:].sum() == pytest.approx(
        eq[-1] - eq[0], abs=1e-5
    )
    # 任意子区间
    for a, b in ((0, 150), (99, 111), (150, T - 1), (104, 108)):
        q = ex.window_pnl(flows, value[a], value[b], a, b, N)
        lhs = (
            q["price"].sum()
            + q["dividend"].sum()
            - q["fees"].sum()
            + interest[a + 1 : b + 1].sum()
        )
        assert lhs == pytest.approx(eq[b] - eq[a], abs=1e-5), (a, b)
        pr, dv, fe = ref_contrib(res, market, a, b)
        np.testing.assert_allclose(q["price"], pr, atol=1e-6)
        np.testing.assert_allclose(q["dividend"], dv, atol=1e-6)


def test_contributions_block():
    res, market, targets, dates, symbols = action_case()
    d = build_tearsheet(res, market, targets, dates, symbols, names={symbols[0]: "甲"})
    C = d["contributions"]
    eq = np.asarray(res.equity)
    for w in [C["total"], *C["yearly"]]:
        assert w["total_pnl"] + w["cash_interest"] == pytest.approx(
            w["equity_change"], abs=0.02
        )
        assert w["total_price"] + w["total_dividend"] - w[
            "total_fees"
        ] == pytest.approx(w["total_pnl"], abs=0.02)
        e0 = eq[dates.index(w["start"])]
        for h in w["top"] + w["bottom"]:
            assert h["contribution"] == pytest.approx(h["pnl"] / e0, abs=1e-6)
        tops = [h["contribution"] for h in w["top"]]
        assert tops == sorted(tops, reverse=True)
        assert not {h["symbol"] for h in w["top"]} & {h["symbol"] for h in w["bottom"]}
    assert C["total"]["equity_change"] == pytest.approx(eq[-1] - eq[0], abs=0.01)
    # 持有天数：直白计数
    pos = np.asarray(res.positions)
    for h in C["total"]["top"]:
        i = symbols.index(h["symbol"])
        assert h["holding_days"] == int((pos[1:, i] > 0).sum())
    y = C["yearly"][0]
    idx = [k for k in range(1, len(dates)) if dates[k][:4] == y["period"]]
    for h in y["top"]:
        i = symbols.index(h["symbol"])
        assert h["holding_days"] == int((pos[idx, i] > 0).sum())
    assert [w["period"] for w in C["yearly"]] == [str(x["year"]) for x in d["yearly"]]
    assert [w["partial"] for w in C["yearly"]] == [x["partial"] for x in d["yearly"]]


# ---------------------------------------------------------------------- 尾部
def test_tails_reference(case, ts):
    res = case["result"]
    r = np.asarray(res.returns)[1:]
    T = ts["tails"]
    assert T["daily_skew"] == pytest.approx(sps.skew(r, bias=True), abs=1e-4)
    assert T["daily_kurtosis"] == pytest.approx(sps.kurtosis(r, bias=True), abs=1e-4)
    mr = np.array(ts["monthly"]["return"])
    assert T["monthly_skew"] == pytest.approx(sps.skew(mr, bias=True), abs=2e-4)
    s = sorted(r.tolist())
    for lv in (95, 99):
        k = -(-len(s) * (100 - lv) // 100) - 1  # 整数算 ⌈n·(1 − lv%)⌉
        q = s[k]
        assert T[f"var_{lv}"] == pytest.approx(q, abs=1e-6)
        tail = [x for x in s if x <= q]
        assert T[f"cvar_{lv}"] == pytest.approx(sum(tail) / len(tail), abs=1e-6)
    worst = sorted(range(len(r)), key=lambda i: r[i])[:5]
    assert [w["date"] for w in T["worst_days"]] == [case["dates"][i + 1] for i in worst]
    bret = ts["daily"]["benchmark_return"]
    for w in T["worst_days"]:
        assert w["benchmark_return"] == pytest.approx(
            bret[case["dates"].index(w["date"])], abs=1e-6
        )
    wm = sorted(range(len(mr)), key=lambda i: mr[i])[:5]
    assert [w["month"] for w in T["worst_months"]] == [
        ts["monthly"]["month"][i] for i in wm
    ]
    # 最长连续亏损月：直白计数
    best = cur = 0
    for x in ts["monthly"]["return"]:
        cur = cur + 1 if x < 0 else 0
        best = max(best, cur)
    assert T["longest_losing_months"] == best


def test_moments_and_var_edges():
    assert ex.moments(np.array([0.1, 0.2])) == (None, None)
    assert ex.moments(np.zeros(10)) == (None, None)
    assert ex.var_cvar(np.array([]), 0.95) == (None, None)
    x = np.arange(1, 101) / 100.0 - 0.5  # -0.49 .. 0.50
    v, c = ex.var_cvar(x, 0.95)
    assert v == pytest.approx(-0.45) and c == pytest.approx(np.mean(x[:5]))


# ---------------------------------------------------------------------- 集中度
def test_concentration_reference(case, ts):
    res = case["result"]
    pos = np.asarray(res.positions)
    close = np.asarray(case["market"].close)
    eq = np.asarray(res.equity)
    lv = np.asarray(res.long_value)
    ind = case["industry"]
    names = {1: "银行", 2: "电子", 3: "医药生物", 4: "汽车"}
    top10, eff, topi = [], [], []
    for d in range(len(pos)):
        held = [i for i in range(pos.shape[1]) if pos[d, i] > 0]
        if not held:
            continue
        val = [pos[d, i] * close[d, i] for i in held]
        scale = lv[d] / sum(val)
        w = [x * scale / eq[d] for x in val]
        top10.append(sum(sorted(w, reverse=True)[:10]))
        s = sum(w)
        eff.append(1 / sum((x / s) ** 2 for x in w))
        agg = {}
        for i, x in zip(held, w, strict=True):
            lab = names.get(int(ind[d, i]), "未分类")
            agg[lab] = agg.get(lab, 0) + x
        topi.append(max(agg.values()))
    c = ts["concentration"]
    assert c["days"] == len(top10)
    assert c["top10_weight"]["mean"] == pytest.approx(np.mean(top10), abs=1e-6)
    assert c["top10_weight"]["max"] == pytest.approx(max(top10), abs=1e-6)
    assert c["effective_holdings"]["mean"] == pytest.approx(np.mean(eff), abs=0.006)
    assert c["top_industry_weight"]["mean"] == pytest.approx(np.mean(topi), abs=1e-6)
    # 最新值与 holdings_latest 一致
    H = ts["holdings_latest"]
    assert c["top10_weight"]["latest"] == pytest.approx(
        sum(h["weight"] for h in H[:10]), abs=1e-5
    )
    agg = {}
    for h in H:
        agg[h["industry"]] = agg.get(h["industry"], 0) + h["weight"]
    assert c["top_industry_weight"]["latest"] == pytest.approx(
        max(agg.values()), abs=1e-5
    )


def test_concentration_without_industry_and_empty_book(case):
    d = build(case, industry=False)
    assert d["concentration"]["top_industry_weight"] is None
    c = make_case(T=200, empty_rows=tuple(range(1, 200)))
    e = build(c)
    assert e["concentration"]["days"] == 0
    assert e["concentration"]["top10_weight"]["mean"] is None
    assert e["contributions"]["total"]["top"] == []
    for s in ("concentration", "contributions", "tails", "winrate", "benchmarks"):
        rp.Tearsheet(e).text(s)
        json.loads(rp.Tearsheet(e).text(s, fmt="json"))


def test_short_history_extras_are_null():
    d = build(make_case(T=40, every=10))  # 两个月：月度偏度无定义
    assert all(v is None for v in d["win_rates"][0]["rolling"].values())
    assert d["tails"]["monthly_skew"] is None
    out = rp.Tearsheet(d).text("winrate")
    assert "任意 1 年" in out


# ---------------------------------------------------------------------- 文本小节 = dict
def test_extra_sections_numbers_match_dict(case):
    v = case["bench"] * 1.0001 ** np.arange(len(case["bench"]))
    d = build(case, benchmarks=[Benchmark(v, "甲指数", "total_return")])
    t = rp.Tearsheet(d)
    out = t.text("benchmarks")
    line = next(ln for ln in out.splitlines() if ln.startswith("甲指数"))
    b = d["benchmarks"][0]
    assert rp.fmt.pct(b["annual_return"], 1, True) in line.split()
    assert rp.fmt.pct(b["excess_annual_return"], 1, True) in line.split()
    out = t.text("concentration")
    c = d["concentration"]
    assert rp.fmt.pct(c["top10_weight"]["mean"], 1) in out
    out = t.text("tails")
    assert rp.fmt.pct(d["tails"]["var_95"], 2) in out
    out = t.text("contributions")
    for h in d["contributions"]["total"]["top"]:
        assert rp.fmt.pct(h["contribution"], 2, True) in out
    js = json.loads(t.text(["winrate", "tails"], fmt="json"))
    assert js["winrate"] == d["win_rates"] and js["tails"] == d["tails"]


def test_old_report_without_extras_still_renders(case):
    d = build(case, extras=False)
    for s in ("benchmarks", "winrate", "contributions", "tails", "concentration"):
        assert "没有这一节" in rp.Tearsheet(d).text(s)
    rp.Tearsheet(d).text("all")


# ---------------------------------------------------------------------- 多策略对比
def test_compare_reference():
    ca = make_case(T=600, seed=3)
    cb = make_case(T=500, seed=4, start="2020-03-02")
    a, b = build(ca), build(cb, benchmark=False)
    c = rp.compare([a, rp.Tearsheet(b)], names=["甲", "乙"])
    D = c.data
    assert D["start"] == "2020-03-02"
    assert D["end"] == min(ca["dates"][-1], cb["dates"][-1])
    assert not D["aligned"]
    # 乙 的交集就是它的全区间：直接取 summary
    mb = D["metrics"][1]
    assert mb["full_range"] and mb["annual_return"] == b["summary"]["annual_return"]
    # 甲 截断：按页面公式
    i0 = ca["dates"].index("2020-03-02")
    i1 = ca["dates"].index(D["end"])
    nav = a["daily"]["nav"]
    assert D["metrics"][0]["cumulative_return"] == pytest.approx(
        nav[i1] / nav[i0] - 1, rel=1e-12
    )
    # 相关系数：numpy 参考
    da, db = a["daily"], b["daily"]
    common = sorted(set(da["date"][1:]) & set(db["date"][1:]))
    common = [x for x in common if "2020-03-02" < x <= D["end"]]
    ra = [compact(a)["daily"]["return"][da["date"].index(x)] for x in common]
    rb = [compact(b)["daily"]["return"][db["date"].index(x)] for x in common]
    assert D["common_days"] == len(common)
    assert D["correlation"][0][1] == pytest.approx(np.corrcoef(ra, rb)[0, 1], abs=1e-12)
    assert D["correlation"][1][0] == D["correlation"][0][1]
    # 分年：整年取报告值，截断年重算
    y21 = next(y for y in D["yearly"] if y["year"] == 2021)
    assert y21["甲"]["return"] == next(
        y["return"] for y in a["yearly"] if y["year"] == 2021
    )
    assert not y21["甲"]["partial"]
    y20 = next(y for y in D["yearly"] if y["year"] == 2020)
    assert y20["甲"]["partial"]
    last20 = max(i for i, x in enumerate(da["date"]) if x[:4] == "2020")
    assert y20["甲"]["return"] == pytest.approx(nav[last20] / nav[i0] - 1, rel=1e-12)
    out = str(c)
    assert "甲 的报告区间是" in out and "日收益相关系数" in out
    assert max(rp.fmt.width(x) for x in out.splitlines()) <= 100
    json.loads(c.text("json"))
    assert "|---" in c.text("markdown")


def test_compare_identical_and_errors(case, ts):
    c = rp.compare([ts, ts], names=["a", "b"])
    assert c.data["aligned"]
    assert c.data["correlation"][0][1] == pytest.approx(1.0)
    assert c.data["metrics"][0]["sharpe"] == ts["summary"]["sharpe"]
    with pytest.raises(rp.ReportError):
        rp.compare([ts])
    with pytest.raises(rp.ReportError):
        rp.compare([ts, ts], names=["a", "a"])
    late = build(make_case(T=100, start="2030-01-01"))
    with pytest.raises(rp.ReportError):
        rp.compare([ts, late])
