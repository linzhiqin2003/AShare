"""build_tearsheet：守恒性质、与独立重算一致、各种边界情形。"""

import dataclasses
import math
from types import SimpleNamespace

import numpy as np
import panelbt
import pytest

import ashare.report as rp
from ashare.report import Benchmark, ReportError, build_tearsheet
from test.report.helpers import build, make_case


@pytest.fixture(scope="module")
def case():
    return make_case()


@pytest.fixture(scope="module")
def ts(case):
    return build(case)


def test_conservation(case, ts):
    d = ts["daily"]
    assert d["nav"][0] == 1.0
    assert d["benchmark_nav"][0] == 1.0
    assert max(d["drawdown"]) <= 0 and max(d["benchmark_drawdown"]) <= 0
    prod = np.prod([1 + x for x in ts["monthly"]["return"]])
    assert prod == pytest.approx(case["result"].nav[-1], rel=1e-5)
    iw = ts["industry_weights"]
    for j in range(len(iw["dates"])):
        assert sum(w[j] for w in iw["weights"]) + iw["cash"][j] == pytest.approx(
            1, abs=1e-5
        )
    for a in ts["attribution"]:
        assert a["price"] + a["dividend_net"] + a["fees"] + a[
            "slippage"
        ] == pytest.approx(a["total"], abs=5e-6)
    yprod = np.prod([1 + y["return"] for y in ts["yearly"]])
    assert yprod == pytest.approx(case["result"].nav[-1], rel=1e-5)


def test_summary_independent_recompute(case, ts):
    res = case["result"]
    nav = res.nav / res.nav[0]
    r = nav[1:] / nav[:-1] - 1
    b = case["bench"]
    br = b[1:] / b[:-1] - 1
    n = len(r)
    s = ts["summary"]
    assert s["trading_days"] == n
    assert s["cumulative_return"] == pytest.approx(nav[-1] - 1, abs=1e-6)
    ann = nav[-1] ** (252 / n) - 1
    bann = (b[-1] / b[0]) ** (252 / n) - 1
    assert s["annual_return"] == pytest.approx(ann, abs=1e-6)
    assert s["annual_volatility"] == pytest.approx(
        r.std(ddof=1) * math.sqrt(252), abs=1e-6
    )
    assert s["sharpe"] == pytest.approx(
        r.mean() / r.std(ddof=1) * math.sqrt(252), abs=1e-4
    )
    assert s["benchmark_annual_return"] == pytest.approx(bann, abs=1e-6)
    assert s["excess_annual_return"] == pytest.approx(ann - bann, abs=1e-6)
    te = (r - br).std(ddof=1) * math.sqrt(252)
    assert s["tracking_error"] == pytest.approx(te, abs=1e-6)
    assert s["information_ratio"] == pytest.approx((r - br).mean() * 252 / te, abs=1e-4)
    assert s["beta"] == pytest.approx(np.cov(r, br)[0, 1] / br.var(ddof=1), abs=1e-4)
    dd = nav / np.maximum.accumulate(nav) - 1
    assert s["max_drawdown"] == pytest.approx(dd.min(), abs=1e-6)
    turnover2 = 2 * res.turnover[1:].sum() / (n / 252)
    assert s["annual_turnover"] == pytest.approx(turnover2, abs=1e-4)
    fees = res.commission.sum() + res.tax.sum() + res.other_fee.sum()
    assert s["total_fees"] == pytest.approx(fees, abs=0.01)
    assert s["total_fees_to_initial"] == pytest.approx(fees / 1e6, abs=1e-6)
    assert s["rebalances"] == len(case["targets"].days)


def test_rebalance_unfilled_counts_names(case, ts):
    """未成交按只数计：新进股开盘不可买、剔除股开盘不可卖。"""
    res, tg, mk = case["result"], case["targets"], case["market"]
    pos = res.positions
    for k, rb in enumerate(ts["rebalances"]):
        d = int(tg.days[k])
        tgt = set(tg.cols[tg.indptr[k] : tg.indptr[k + 1]].tolist())
        held = set(np.flatnonzero(pos[d - 1] > 0).tolist())
        want = sum(1 for c in tgt - held if not mk.can_buy[d, c]) + sum(
            1 for c in held - tgt if not mk.can_sell[d, c]
        )
        assert rb["unfilled"] == want
        nxt = int(tg.days[k + 1]) if k + 1 < len(tg.days) else len(pos)
        assert rb["blocked_buy_days"] == int(res.n_blocked_buy[d:nxt].sum())
        assert rb["new_names"] == len(tgt - held) and rb["dropped_names"] == len(
            held - tgt
        )
    assert sum(rb["unfilled"] for rb in ts["rebalances"]) > 0  # 用例确实覆盖到了未成交


def test_holdings_latest(case, ts):
    pos = case["result"].positions[-1]
    assert len(ts["holdings_latest"]) == int((pos > 0).sum())
    w = [h["weight"] for h in ts["holdings_latest"]]
    assert w == sorted(w, reverse=True)
    assert sum(w) + ts["daily"]["cash_weight"][-1] == pytest.approx(1, abs=1e-4)
    assert all(h["name"].startswith("股票") for h in ts["holdings_latest"])


def test_industry_labels_and_unclassified(ts):
    inds = set(ts["industry_weights"]["industries"])
    assert inds <= {"银行", "电子", "医药生物", "汽车", "未分类"}


def test_no_benchmark(case):
    t = build(case, benchmark=False)
    s = t["summary"]
    for k in (
        "benchmark_annual_return",
        "excess_annual_return",
        "beta",
        "information_ratio",
    ):
        assert s[k] is None
    assert t["daily"]["benchmark_nav"] is None and t["rolling"]["beta"] is None
    assert t["monthly"]["benchmark_return"] is None
    assert all(y["benchmark_return"] is None for y in t["yearly"])
    assert t["meta"]["benchmark"] is None


def test_benchmark_starts_later(case):
    b = case["bench"].copy()
    b[:50] = np.nan
    t = build(case, benchmark=Benchmark(b, "后起基准", "total_return"))
    d = t["daily"]
    assert d["benchmark_nav"][:50] == [None] * 50 and d["benchmark_nav"][50] == 1.0
    s = t["summary"]
    assert s["benchmark_start"] == case["dates"][50]
    nav = case["result"].nav
    r = nav[51:] / nav[50:-1] - 1
    br = b[51:] / b[50:-1] - 1
    n = len(r)
    assert s["benchmark_annual_return"] == pytest.approx(
        (b[-1] / b[50]) ** (252 / n) - 1, abs=1e-6
    )
    assert s["excess_annual_return"] == pytest.approx(
        (nav[-1] / nav[50]) ** (252 / n) - 1 - ((b[-1] / b[50]) ** (252 / n) - 1),
        abs=1e-6,
    )
    assert s["beta"] == pytest.approx(np.cov(r, br)[0, 1] / br.var(ddof=1), abs=1e-4)
    assert t["monthly"]["benchmark_return"][0] is None
    assert t["meta"]["benchmark"]["kind"] == "total_return"


def test_benchmark_interior_gap_filled_and_trailing_gap_rejected(case):
    b = case["bench"].copy()
    b[100] = np.nan
    t = build(case, benchmark=Benchmark(b, "x", "price"))
    assert t["meta"]["benchmark"]["filled_days"] == 1
    assert t["daily"]["benchmark_nav"][100] == t["daily"]["benchmark_nav"][99]
    b[-1] = np.nan
    with pytest.raises(ReportError, match="末尾"):
        build(case, benchmark=Benchmark(b, "x", "price"))


def test_benchmark_kind_must_be_declared(case):
    with pytest.raises(ReportError, match="kind"):
        build(case, benchmark=Benchmark(case["bench"], "x", "net"))


def test_short_history_has_empty_rolling():
    c = make_case(T=120, every=10)
    t = build(c)
    assert t["rolling"]["date"] == [] and t["rolling"]["sharpe"] == []
    assert t["summary"]["annual_return"] is not None
    assert t["yearly"][0]["partial"]


def test_daily_rebalance_samples_industry_monthly():
    c = make_case(T=320, every=1, N=8)
    t = build(c)
    assert len(t["rebalances"]) == 319
    iw = t["industry_weights"]
    assert iw["sampling"] == "month_end" and len(iw["dates"]) == len(
        t["monthly"]["month"]
    )


def test_varying_holdings_and_empty_period():
    sizes = [2, 8, 1, 6, 0, 0, 3, 5, 8, 4, 2, 7]
    c = make_case(
        T=250, every=21, pick_by_row=lambda k: sizes[k % len(sizes)], blocked=False
    )
    t = build(c)
    h = t["daily"]["holdings"]
    assert max(h) == 8 and min(h[1:]) == 0
    empty = [i for i, x in enumerate(h) if i > 0 and x == 0]
    assert all(t["daily"]["cash_weight"][i] == pytest.approx(1) for i in empty)
    for j in range(len(t["industry_weights"]["dates"])):
        tot = (
            sum(w[j] for w in t["industry_weights"]["weights"])
            + t["industry_weights"]["cash"][j]
        )
        assert tot == pytest.approx(1, abs=1e-5)


def test_no_industry(case):
    t = build(case, industry=False)
    assert t["industry_weights"] is None
    assert all(h["industry"] is None for h in t["holdings_latest"])


def test_industry_switch_is_explicit(case):
    assert build(case)["meta"]["industry_switch"] is None
    t = build(case, industry_switch="20200601", industry_switch_labels=("旧", "新"))
    assert t["meta"]["industry_switch"] == "2020-06-01"
    assert t["meta"]["industry_switch_labels"] == ["旧", "新"]


def test_requires_positions():
    c = make_case(record=False)
    with pytest.raises(ReportError, match="record_trades"):
        build(c)


def test_rejects_short_targets(case):
    tg = case["targets"]
    neg = SimpleNamespace(
        days=tg.days,
        indptr=tg.indptr,
        cols=tg.cols,
        weights=-tg.weights,
        kind="weights",
    )
    with pytest.raises(ReportError, match="做空"):
        build_tearsheet(
            case["result"], case["market"], neg, case["dates"], case["symbols"]
        )


def test_rejects_short_positions(case):
    res = dataclasses.replace(case["result"], positions=-case["result"].positions)
    with pytest.raises(ReportError, match="空头"):
        build_tearsheet(
            res, case["market"], case["targets"], case["dates"], case["symbols"]
        )


def test_rejects_futures(case):
    mk = case["market"]
    fake = SimpleNamespace(
        **{k: getattr(mk, k) for k in ("exec_price", "close", "can_buy", "can_sell")},
        futures=object(),
        shape=mk.shape,
    )
    with pytest.raises(ReportError, match="期货"):
        build_tearsheet(
            case["result"], fake, case["targets"], case["dates"], case["symbols"]
        )


def test_rejects_cash_flows(case):
    cf = np.zeros(len(case["dates"]))
    cf[5] = 1000.0
    res = dataclasses.replace(case["result"], cash_flow=cf)
    with pytest.raises(ReportError, match="资金进出"):
        build_tearsheet(
            res, case["market"], case["targets"], case["dates"], case["symbols"]
        )


def test_rejects_batch_and_misaligned(case):
    res = dataclasses.replace(case["result"], nav=np.vstack([case["result"].nav] * 2))
    with pytest.raises(ReportError, match="单个策略"):
        build_tearsheet(
            res, case["market"], case["targets"], case["dates"], case["symbols"]
        )
    with pytest.raises(ReportError, match="逐行对齐"):
        build_tearsheet(
            case["result"],
            case["market"],
            case["targets"],
            case["dates"][:-1],
            case["symbols"],
        )


def test_rejects_trading_on_first_row():
    c = make_case(T=60)
    tg = c["targets"]
    days = tg.days.copy()
    days[0] = 0
    targets = panelbt.Targets(days, tg.indptr, tg.cols, tg.weights)
    res = panelbt.run(c["market"], targets, c["costs"], c["config"])
    with pytest.raises(ReportError, match="基准日"):
        build_tearsheet(res, c["market"], targets, c["dates"], c["symbols"])


def test_convenience_requires_data_for_index_code(case):
    with pytest.raises(ReportError, match="data"):
        rp.tearsheet(
            case["result"],
            case["market"],
            case["targets"],
            case["dates"],
            case["symbols"],
            benchmark="000300.SH",
            benchmark_kind="price",
        )
    t = rp.tearsheet(
        case["result"],
        case["market"],
        case["targets"],
        case["dates"],
        case["symbols"],
        benchmark=Benchmark(case["bench"], "x", "price"),
    )
    assert t.summary["benchmark_annual_return"] is not None


def test_industry_switch_date_from_sources():
    src = np.array([[1, 3, 0], [1, 1, 0], [2, 2, 0], [2, 2, 2]])
    assert rp.industry_switch_date(src, ["a", "b", "c", "d"]) == "c"
    assert rp.industry_switch_date(np.full((3, 2), 2), ["a", "b", "c"]) is None
    assert rp.industry_switch_date(np.ones((3, 2)), ["a", "b", "c"]) is None


def test_open_day_between_calendar_and_weekday_fallback():
    from ashare.report.build import _open_day_between as f

    # 只按工作日：2020-01-31 是周五，之后到 2 月 1 日没有工作日
    assert not f("2020-01-31", "2020-02-01", None)
    assert f("2020-01-29", "2020-02-01", None)
    # 日历说 01-29 之后本月不再开市（例如节假日），即使还有工作日也算完整
    cal = ["2020-01-28", "2020-01-29", "2020-02-03"]
    assert not f("2020-01-29", "2020-02-01", cal)
    assert f("2020-01-28", "2020-02-01", cal)
    # 日历只覆盖到区间末日：之后按工作日判断
    assert f("2020-01-29", "2020-02-01", ["2020-01-28", "2020-01-29"])


def test_partial_flags_follow_trading_days():
    c = make_case(T=60, start="2020-01-01")
    end = c["dates"][-1]
    t = build(c)  # 无日历：按工作日
    assert t["monthly"]["first_partial"]  # 基准日 2020-01-01 与首个交易日同月
    assert t["yearly"][0]["partial"]
    assert t["meta"]["completeness_basis"] == "weekday"
    # 日历显示末日之后本月、本年都不再开市 → 末月、末年都完整
    cal = c["dates"]
    t2 = build(c, calendar=cal + [f"{int(end[:4]) + 1}-01-05"])
    assert not t2["monthly"]["last_partial"]
    # 日历里末日之后同月还有开市日 → 末月不完整
    nxt = str(np.datetime64(end, "D") + np.timedelta64(1, "D"))
    if nxt[:7] == end[:7]:
        t3 = build(c, calendar=cal + [nxt])
        assert t3["monthly"]["last_partial"]
        assert t3["yearly"][-1]["partial"]


def test_first_year_complete_when_base_day_in_previous_year():
    c = make_case(T=300, start="2019-12-31")
    t = build(c)
    assert not t["monthly"]["first_partial"]
    assert not t["yearly"][0]["partial"]
