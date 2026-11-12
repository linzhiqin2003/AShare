"""文本版：数字与报告数据逐项一致、区间重算与页面 JS 一致、排版约束、命令行。"""

from __future__ import annotations

import copy
import io
import json
import math
import re
import shutil
import subprocess
from contextlib import redirect_stdout
from importlib import resources

import numpy as np
import pytest

import ashare.report as rp
from ashare.report import Benchmark, Tearsheet
from ashare.report import window as win
from ashare.report.__main__ import main as cli
from ashare.report.fmt import num, pct, to_fixed, width
from ashare.report.render import compact
from ashare.report.sections import SECTIONS
from test.report.helpers import build, make_case

NODE = shutil.which("node")


@pytest.fixture(scope="module")
def case():
    return make_case(T=600, seed=3)


@pytest.fixture(scope="module")
def ts(case):
    return build(case)


@pytest.fixture(scope="module")
def ts_nob(case):
    return build(case, benchmark=False, industry=False)


@pytest.fixture(scope="module")
def ts_late(case):
    b = case["bench"].copy()
    b[:150] = np.nan
    return build(case, benchmark=Benchmark(b, "晚基准", "total_return"))


@pytest.fixture(scope="module")
def ts_long():
    return build(make_case(T=252 * 26, N=10, seed=5, every=63))


@pytest.fixture(scope="module")
def ts_short():
    return build(make_case(T=120, seed=7, every=10))


# ---------------------------------------------------------------------- 独立解析
def parse_pct(s: str) -> float | None:
    """独立于 fmt 的解析：'+8.6%' → 0.086。"""
    if s == "-":
        return None
    assert re.fullmatch(r"[+-]?\d+(\.\d+)?%", s), s
    return float(s[:-1]) / 100


def table_rows(text: str, first_cell: str) -> list[list[str]]:
    """从表头第一格为 first_cell 的表里取数据行（按两个以上空格切列）。"""
    lines = text.splitlines()
    i = next(
        k for k, ln in enumerate(lines) if ln.split() and ln.split()[0] == first_cell
    )
    out = []
    for ln in lines[i + 1 :]:
        if not ln.strip():
            break
        out.append(re.split(r"\s{2,}", ln.strip()))
    return out


# ---------------------------------------------------------------------- 格式
def test_to_fixed_rounds_like_js():
    # 二进制精确的一半时 JS 取绝对值大的一侧；不是精确一半时按真实值
    assert to_fixed(0.125, 2) == "0.13"
    assert to_fixed(-0.125, 2) == "-0.13"
    assert to_fixed(2.5, 0) == "3"
    assert to_fixed(1.005, 2) == "1.00"  # 1.005 实为 1.00499999…
    assert to_fixed(-0.0001, 2) == "-0.00"
    assert to_fixed(-0.0, 1) == "0.0"
    assert pct(-0.0001, 1, True) == "0.0%"
    assert pct(0.0863, 1, True) == "+8.6%"
    assert pct(None) == "-" and pct(float("nan")) == "-"
    assert num(-0.004) == "0.00" and num(None) == "-"


def _js_src():
    return (
        resources.files("ashare.report")
        .joinpath("templates/tearsheet.html")
        .read_text(encoding="utf-8")
    )


def _node(script: str, payload) -> object:
    p = subprocess.run(
        [NODE, "-e", script],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    return json.loads(p.stdout)


@pytest.mark.skipif(NODE is None, reason="需要 node 运行页面里的 JS")
def test_formatters_match_page_js():
    src = _js_src()
    fns = "\n".join(
        ln
        for ln in src.splitlines()
        if ln.startswith(("function pct(", "function num("))
    )
    rng = np.random.default_rng(0)
    xs = [
        0.125,
        -0.125,
        0.0005,
        -0.0005,
        0.00049999,
        1.005,
        0.0,
        -0.0,
        12.3456,
        0.99995,
    ]
    xs += (rng.normal(0, 0.2, 300)).tolist() + (
        rng.integers(-2000, 2000, 200) / 8000
    ).tolist()
    script = (
        "const MINUS='−';"
        + fns
        + "let s='';process.stdin.on('data',c=>s+=c).on('end',()=>{const xs=JSON.parse(s);"
        "console.log(JSON.stringify(xs.map(x=>[pct(x,1,true),pct(x,0),pct(x,2),num(x),num(x,4)])))})"
    )
    js = _node(script, xs)
    for x, row in zip(xs, js, strict=True):
        py = [pct(x, 1, True), pct(x, 0), pct(x, 2), num(x), num(x, 4)]
        assert [s.replace("−", "-") for s in row] == py, x


# ---------------------------------------------------------------------- 区间重算 = 页面 JS
def _js_metrics(data, pairs):
    src = _js_src()
    body = src[
        src.index("function stats(") : src.index("/* ---------- state ---------- */")
    ]
    script = (
        "let s='';process.stdin.on('data',c=>s+=c).on('end',()=>{"
        "const P=JSON.parse(s); const D=P.D; const dl=D.daily, S=D.summary, M=D.meta;"
        "const N=dl.date.length; const HASB=!!(M.benchmark && dl.benchmark_nav);"
        "const BS=HASB? dl.benchmark_nav.findIndex(v=>v!=null) : -1;"
        "const idxOf={}; dl.date.forEach((d,i)=>idxOf[d]=i);"
        "const RB=D.rebalances, rebIdx=RB.map(r=>idxOf[r.date]);"
        "const PY=M.periods_per_year||252; const yrs=n=>n/(M.periods_per_year||252);"
        + body
        + "console.log(JSON.stringify(P.pairs.map(([a,b])=>{const m=metrics(a,b);"
        "for(const k in m){ if(typeof m[k]==='number' && !isFinite(m[k])) m[k]=null; } return m;})));});"
    )
    return _node(script, {"D": compact(data), "pairs": pairs})


@pytest.mark.skipif(NODE is None, reason="需要 node 运行页面里的 JS")
@pytest.mark.parametrize("which", ["ts", "ts_nob", "ts_late"])
def test_window_metrics_match_page_js(which, request):
    data = request.getfixturevalue(which)
    T = len(data["daily"]["date"])
    pairs = [
        (0, T - 1),
        (0, 300),
        (37, T - 1),
        (100, 400),
        (5, 8),
        (140, 152),
        (T - 3, T - 1),
    ]
    js = _js_metrics(data, pairs)
    for (a, b), j in zip(pairs, js, strict=True):
        py = win.metrics(data, a, b)
        for k, v in j.items():
            assert k in py, k
            if v is None or py[k] is None:
                assert v is None and py[k] is None, (a, b, k, v, py[k])
            else:
                # 同样的输入、同样的求和顺序：只可能差在 pow 的最后一位
                assert py[k] == pytest.approx(v, rel=1e-13, abs=1e-15), (a, b, k)


def test_full_range_window_is_summary(ts, ts_nob, ts_late):
    for d in (ts, ts_nob, ts_late):
        T = len(d["daily"]["date"])
        w = win.window_summary(d, 0, T - 1)
        s = d["summary"]
        for k in (
            "cumulative_return",
            "annual_return",
            "annual_volatility",
            "sharpe",
            "max_drawdown",
            "annual_turnover",
            "total_fees_to_initial",
            "total_slippage_to_initial",
            "benchmark_cumulative_return",
            "benchmark_annual_return",
            "benchmark_annual_volatility",
            "benchmark_sharpe",
            "benchmark_max_drawdown",
            "excess_annual_return",
            "tracking_error",
            "information_ratio",
            "beta",
        ):
            assert w[k] == s[k], k  # 逐位相同（与页面一样直接取 summary）


@pytest.mark.parametrize("which", ["ts", "ts_late"])
def test_full_range_formula_agrees_with_summary_at_display_precision(which, request):
    """公式路径（不取 summary）在全区间上与 summary 的显示值一致；数值差只来自嵌入时的取整。"""
    d = request.getfixturevalue(which)
    T = len(d["daily"]["date"])
    m = win.metrics(d, 0, T - 1, use_summary=False)
    s = d["summary"]
    pairs = {
        "cum": ("cumulative_return", lambda x: pct(x, 1, True), 2e-6),
        "ann": ("annual_return", lambda x: pct(x, 1, True), 2e-6),
        "vol": ("annual_volatility", pct, 2e-6),
        "sharpe": ("sharpe", num, 2e-4),
        "mdd": ("max_drawdown", pct, 2e-6),
        "turnover": ("annual_turnover", lambda x: pct(x, 0), 2e-4),
        "fees": ("total_fees_to_initial", lambda x: pct(x, 2), 2e-6),
        "slip": ("total_slippage_to_initial", lambda x: pct(x, 2), 2e-6),
        "bcum": ("benchmark_cumulative_return", lambda x: pct(x, 1, True), 2e-6),
        "bann": ("benchmark_annual_return", lambda x: pct(x, 1, True), 2e-6),
        "bvol": ("benchmark_annual_volatility", pct, 2e-6),
        "te": ("tracking_error", pct, 2e-6),
        "ir": ("information_ratio", num, 2e-4),
        "beta": ("beta", num, 2e-4),
        "excess": ("excess_annual_return", lambda x: pct(x, 1, True), 2e-6),
    }
    for pk, (sk, f, tol) in pairs.items():
        assert m[pk] == pytest.approx(s[sk], abs=tol), pk
        assert f(m[pk]) == f(s[sk]), pk


def test_resolve_dates(ts):
    dates = ts["daily"]["date"]
    assert win.resolve(ts) == (0, len(dates) - 1)
    a, b = win.resolve(ts, dates[10], dates[50])
    assert (a, b) == (9, 50)
    # 非交易日：起点取之后第一个交易日，终点取之前最后一个
    sat = "2020-01-04"
    a, _ = win.resolve(ts, sat)
    assert dates[a + 1] >= sat and dates[a] < sat
    a, b = win.resolve(ts, "19990101", "2099-12-31")
    assert (a, b) == (0, len(dates) - 1)
    with pytest.raises(rp.ReportError):
        win.resolve(ts, "2099-01-01")
    with pytest.raises(rp.ReportError):
        win.resolve(ts, dates[20], dates[20].replace("-", "")[:0] + dates[19])


# ---------------------------------------------------------------------- 默认摘要：数字 = dict
def test_default_summary_numbers_match_dict(ts):
    out = str(Tearsheet(ts))
    s = ts["summary"]
    rows = {r[0]: r[1:] for r in table_rows(out, "指标")}
    keys = {
        "累计收益": ("cumulative_return", "benchmark_cumulative_return"),
        "年化收益": ("annual_return", "benchmark_annual_return"),
        "年化波动": ("annual_volatility", "benchmark_annual_volatility"),
        "最大回撤": ("max_drawdown", "benchmark_max_drawdown"),
    }
    for lab, (k, bk) in keys.items():
        sv, bv = rows[lab]
        assert parse_pct(sv) == pytest.approx(s[k], abs=0.0005 + 1e-12), lab
        assert parse_pct(bv) == pytest.approx(s[bk], abs=0.0005 + 1e-12), lab
        assert sv == pct(s[k], 1, k != "annual_volatility" and k != "max_drawdown")
    assert float(rows["夏普"][0]) == pytest.approx(s["sharpe"], abs=0.005 + 1e-12)
    assert float(rows["夏普"][1]) == pytest.approx(
        s["benchmark_sharpe"], abs=0.005 + 1e-12
    )
    m = re.search(r"年化超额 (\S+)，信息比率 (\S+)，跟踪误差 (\S+)，Beta (\S+)", out)
    assert parse_pct(m[1]) == pytest.approx(
        s["excess_annual_return"], abs=0.0005 + 1e-12
    )
    assert float(m[2]) == pytest.approx(s["information_ratio"], abs=0.005 + 1e-12)
    assert parse_pct(m[3]) == pytest.approx(s["tracking_error"], abs=0.0005 + 1e-12)
    assert float(m[4]) == pytest.approx(s["beta"], abs=0.005 + 1e-12)
    m = re.search(
        r"年化换手（双边） (\S+)，费用 / 期初权益 (\S+)，滑点 / 期初权益 (\S+)", out
    )
    assert parse_pct(m[1]) == pytest.approx(s["annual_turnover"], abs=0.005 + 1e-12)
    assert parse_pct(m[2]) == pytest.approx(
        s["total_fees_to_initial"], abs=0.00005 + 1e-12
    )
    assert parse_pct(m[3]) == pytest.approx(
        s["total_slippage_to_initial"], abs=0.00005 + 1e-12
    )
    assert f"{s['max_drawdown_peak']} 见顶，{s['max_drawdown_trough']} 见底" in out
    # 分年
    yr = table_rows(out, "年份")
    assert [r[0].rstrip("*") for r in yr] == [str(y["year"]) for y in ts["yearly"]]
    for r, y in zip(yr, ts["yearly"], strict=True):
        assert r[0].endswith("*") == y["partial"]
        assert parse_pct(r[1]) == pytest.approx(y["return"], abs=0.0005 + 1e-12)
        assert parse_pct(r[2]) == pytest.approx(
            y["benchmark_return"], abs=0.0005 + 1e-12
        )
        assert parse_pct(r[3]) == pytest.approx(
            y["return"] - y["benchmark_return"], abs=0.0005 + 1e-12
        )
    # 回撤
    dd = table_rows(out, "深度")
    for r, e in zip(dd, ts["drawdowns"][:3], strict=False):
        assert parse_pct(r[0]) == pytest.approx(e["depth"], abs=0.0005 + 1e-12)
        assert r[1:3] == [e["peak"], e["trough"]]
        assert r[3] == (e["recovery"] or "未修复")
        assert (
            r[4] == f"{e['days_to_trough']} 天" and r[5] == f"{e['duration_days']} 天"
        )


def test_json_sections_carry_dict_values(ts):
    js = json.loads(
        rp.Tearsheet(ts).text(["summary", "yearly", "drawdowns"], fmt="json")
    )
    assert set(js) == {"range", "summary", "yearly", "drawdowns"}
    s = js["summary"]["summary"]
    for k, v in ts["summary"].items():
        assert s[k] == v
    assert [y["return"] for y in js["yearly"]] == [y["return"] for y in ts["yearly"]]
    assert js["drawdowns"] == ts["drawdowns"]
    only = json.loads(rp.Tearsheet(ts).text("warnings", fmt="json"))
    assert set(only) == {"range", "warnings"}


def test_window_text_uses_page_formula(ts):
    d = ts["daily"]["date"]
    out = rp.Tearsheet(ts).text(start=d[100], end=d[400])
    w = win.window_summary(ts, 99, 400)
    rows = {r[0]: r[1:] for r in table_rows(out, "指标")}
    assert rows["累计收益"][0] == pct(w["cumulative_return"], 1, True)
    assert rows["年化收益"][0] == pct(w["annual_return"], 1, True)
    assert rows["夏普"][0] == num(w["sharpe"])
    assert rows["最大回撤"][1] == pct(w["benchmark_max_drawdown"], 1)
    assert "自选区间" in out and f"区间 {d[100]} 至 {d[400]}" in out
    # 不足 240 个交易日时与页面一样只给区间收益
    short = rp.Tearsheet(ts).text(start=d[100], end=d[200])
    assert "年化收益" not in short and "区间超额" in short


def test_window_clips_years(ts):
    d = ts["daily"]["date"]
    js = json.loads(
        rp.Tearsheet(ts).text("yearly", fmt="json", start=d[100], end=d[400])
    )
    rows = js["yearly"]
    nav = ts["daily"]["nav"]
    first = rows[0]
    assert first["clipped"][0] == d[100]
    last_i = max(i for i in range(100, 401) if d[i][:4] == str(first["year"]))
    assert first["return"] == pytest.approx(nav[last_i] / nav[99] - 1, abs=1e-12)


# ---------------------------------------------------------------------- 排版约束
@pytest.mark.parametrize("which", ["ts", "ts_nob", "ts_late", "ts_long", "ts_short"])
def test_default_summary_fits_one_screen(which, request):
    d = request.getfixturevalue(which)
    for lang in ("zh", "en"):
        out = rp.Tearsheet(d).text(lang=lang)
        lines = out.rstrip("\n").splitlines()
        assert len(lines) <= 40, (which, lang, len(lines))
        assert max(width(x) for x in lines) <= 100
        assert "\x1b" not in out
        assert not set(out) & set("─│┌┐└┘├┤┬┴┼═║")


def test_long_history_keeps_recent_years(ts_long):
    out = str(rp.Tearsheet(ts_long))
    yr = table_rows(out, "年份")
    years = [y["year"] for y in ts_long["yearly"]]
    assert len(yr) < len(years)
    assert yr[-1][0].rstrip("*") == str(years[-1])
    assert f"更早的 {len(years) - len(yr)} 年见 --section yearly" in out


@pytest.mark.parametrize("which", ["ts", "ts_nob", "ts_late", "ts_long", "ts_short"])
@pytest.mark.parametrize("fmt", ["text", "markdown"])
def test_all_sections_render_within_width(which, fmt, request):
    d = request.getfixturevalue(which)
    for lang in ("zh", "en"):
        out = rp.Tearsheet(d).text("all", fmt=fmt, lang=lang)
        if fmt == "text":
            assert max(width(x) for x in out.splitlines()) <= 100
        else:
            assert any(ln.startswith("|---") for ln in out.splitlines())
        json.loads(rp.Tearsheet(d).text("all", fmt="json", lang=lang))


def test_missing_is_dash_and_signs_explicit(ts_nob, ts_short):
    out = rp.Tearsheet(ts_nob).text()
    assert "无基准" in out and "沪深" not in out
    assert " 0%" not in out.replace("0.0%", "")
    js = json.loads(rp.Tearsheet(ts_short).text("rolling", fmt="json"))
    assert js["rolling"] == []
    assert "没有滚动指标" in rp.Tearsheet(ts_short).text("rolling")


def test_en_headers_are_json_keys(ts):
    out = rp.Tearsheet(ts).text(["yearly", "drawdowns", "rebalances"], lang="en")
    yhead = next(ln for ln in out.splitlines() if ln.startswith("year "))
    ykeys = set(ts["yearly"][0]) | {
        "excess_return",
        "dividend_net",
        "fees_and_slippage",
    }
    assert set(yhead.split()) <= ykeys
    dhead = next(ln for ln in out.splitlines() if ln.strip().startswith("depth"))
    assert set(dhead.split()) <= set(ts["drawdowns"][0])
    rhead = next(ln for ln in out.splitlines() if ln.startswith("date "))
    assert set(rhead.split()) <= set(ts["rebalances"][0])
    summ = rp.Tearsheet(ts).text(lang="en")
    for k in ("cumulative_return", "annual_return", "sharpe", "max_drawdown"):
        assert re.search(rf"^{k}\s", summ, re.M)
        assert k in ts["summary"]


def test_section_names_and_errors(ts):
    t = rp.Tearsheet(ts)
    assert set(SECTIONS) >= {
        "summary",
        "yearly",
        "monthly",
        "drawdowns",
        "rolling",
        "rebalances",
        "holdings",
        "industry",
        "costs",
        "warnings",
    }
    with pytest.raises(ValueError):
        t.text("nope")
    with pytest.raises(ValueError):
        t.text(fmt="html")
    with pytest.raises(ValueError):
        t.text(lang="fr")
    assert t.text("yearly,warnings").count("\n\n") >= 1


def test_rolling_one_row_per_quarter(ts_long):
    rows = json.loads(rp.Tearsheet(ts_long).text("rolling", fmt="json"))["rolling"]
    ro = ts_long["rolling"]
    qs = [(r["date"][:4], (int(r["date"][5:7]) - 1) // 3) for r in rows]
    assert len(qs) == len(set(qs))
    assert rows[-1]["date"] == ro["date"][-1]
    for r in rows:
        i = ro["date"].index(r["date"])
        assert r["sharpe"] == ro["sharpe"][i]
        nxt = ro["date"][i + 1] if i + 1 < len(ro["date"]) else None
        assert nxt is None or (nxt[:4], (int(nxt[5:7]) - 1) // 3) != (
            r["date"][:4],
            (int(r["date"][5:7]) - 1) // 3,
        )


def test_holdings_and_industry_sections_match_dict(ts):
    out = rp.Tearsheet(ts).text(["holdings", "industry"])
    H = ts["holdings_latest"]
    rows = table_rows(out, "代码")
    assert [r[0] for r in rows] == [h["symbol"] for h in H]
    for r, h in zip(rows, H, strict=True):
        assert parse_pct(r[3]) == pytest.approx(h["weight"], abs=0.00005 + 1e-12)
    js = json.loads(rp.Tearsheet(ts).text("industry", fmt="json"))["industry"]
    iw = ts["industry_weights"]
    for row in js["rows"][:-1]:
        i = iw["industries"].index(row["industry"])
        assert row["latest"] == iw["weights"][i][-1]
        assert row["max"] == max(iw["weights"][i])
    assert js["rows"][-1]["latest"] == iw["cash"][-1]


# ---------------------------------------------------------------------- 提醒：每条规则正反例
def W(d):
    return {w["code"]: w for w in rp.checks.warnings(d)}


def test_warning_benchmark_price_index(ts, ts_late):
    d = copy.deepcopy(ts)
    assert ts["meta"]["benchmark"]["kind"] == "price"
    assert W(d)["benchmark_price_index"]["values"]["mean_dividend_contribution"] is None
    d["summary"]["total_dividend_gross"] = 1000.0
    for k, a in enumerate(d["attribution"]):
        a["dividend_net"] = 0.01 * (k + 1)
    w = W(d)["benchmark_price_index"]
    whole = [
        a
        for a, y in zip(d["attribution"], d["yearly"], strict=True)
        if not y["partial"]
    ]
    assert 0 < len(whole) < len(d["yearly"])
    assert w["values"]["mean_dividend_contribution"] == pytest.approx(
        sum(a["dividend_net"] for a in whole) / len(whole)
    )
    assert w["values"]["years"] == len(whole)
    assert "benchmark_price_index" not in W(ts_late)  # 全收益基准
    d["meta"]["benchmark"]["kind"] = "total_return"
    assert "benchmark_price_index" not in W(d)


def test_warning_partial_end(ts):
    d = copy.deepcopy(ts)
    d["monthly"]["last_partial"] = True
    d["yearly"][-1]["partial"] = True
    w = W(d)["partial_end"]
    assert w["values"]["last_year_partial"] and w["values"]["last_month_partial"]
    assert (
        f"末年 {d['yearly'][-1]['year']} 只有 {d['yearly'][-1]['days']} 个交易日"
        in w["message"]
    )
    d["monthly"]["last_partial"] = False
    d["yearly"][-1]["partial"] = False
    assert "partial_end" not in W(d)


def test_warning_partial_start(ts):
    d = copy.deepcopy(ts)
    d["monthly"]["first_partial"] = False
    d["summary"]["start"] = "2019-12-31"
    d["summary"]["first_trade_day"] = "2020-01-01"
    assert "partial_start" not in W(d)
    d["monthly"]["first_partial"] = True
    d["summary"]["start"] = "2020-01-01"
    d["summary"]["first_trade_day"] = "2020-01-02"
    w = W(d)["partial_start"]
    assert w["values"]["first_year_partial"] and w["values"]["first_month_partial"]


def test_warning_unfilled(ts):
    d = copy.deepcopy(ts)
    for r in d["rebalances"]:
        r["unfilled"] = 0
        r["unfilled_by_reason"] = {"limit_up": 0, "limit_down": 0, "suspended": 0}
        r["new_names"], r["dropped_names"] = 2, 2
    assert "unfilled" not in W(d)
    n = len(d["rebalances"])
    changes = 4 * n
    need = math.ceil(0.05 * changes)
    for r in d["rebalances"][:need]:
        r["unfilled"] = 1
        r["unfilled_by_reason"]["suspended"] = 1
    w = W(d)["unfilled"]
    assert w["values"]["unfilled"] == need and w["values"]["changes"] == changes
    assert w["values"]["by_reason"]["suspended"] == need
    d["rebalances"][0]["unfilled"] = 0  # 低于 5%
    if (need - 1) / changes < 0.05:
        assert "unfilled" not in W(d)


def test_warning_cash_high(ts):
    d = copy.deepcopy(ts)
    i0 = d["daily"]["date"].index(d["rebalances"][0]["date"])
    n = len(d["daily"]["cash_weight"]) - i0
    d["daily"]["cash_weight"][i0:] = [0.02] * n
    assert "cash_high" not in W(d)
    d["daily"]["cash_weight"][i0:] = [0.06] * (n // 2 + 1) + [0.0] * (n - n // 2 - 1)
    d["daily"]["cash_weight"][-1] = 0.3
    w = W(d)["cash_high"]
    assert w["values"]["median"] == pytest.approx(0.06)
    assert (
        w["values"]["max"] == 0.3 and w["values"]["max_date"] == d["daily"]["date"][-1]
    )
    # 首次调仓之前的空仓期不算
    d["daily"]["cash_weight"][i0:] = [0.0] * n
    d["daily"]["cash_weight"][:i0] = [1.0] * i0
    assert "cash_high" not in W(d)


def test_warning_holdings_short(ts):
    d = copy.deepcopy(ts)
    for r in d["rebalances"]:
        r["targets"], r["holdings_after"] = 10, 9
    assert "holdings_short" not in W(d)
    k = math.ceil(0.10 * len(d["rebalances"]))
    for r in d["rebalances"][:k]:
        r["holdings_after"] = 7
    d["rebalances"][1]["holdings_after"] = 3
    w = W(d)["holdings_short"]
    assert w["values"]["count"] == k and w["values"]["worst_holdings"] == 3
    for r in d["rebalances"][: k - 1]:
        r["holdings_after"] = 9
    if k - 1 < 0.10 * len(d["rebalances"]):
        assert "holdings_short" not in W(d)


def test_warning_drawdown_open(ts):
    d = copy.deepcopy(ts)
    dd = d["daily"]["drawdown"]
    T = len(dd)
    dd[:] = [0.0] * T
    assert "drawdown_open" not in W(d)
    dd[T - 30 :] = [-0.02] * 10 + [-0.12] * 5 + [-0.06] * 15
    w = W(d)["drawdown_open"]
    assert w["values"]["peak"] == d["daily"]["date"][T - 31]
    assert w["values"]["depth"] == -0.12 and w["values"]["current"] == -0.06
    assert w["values"]["trough"] == d["daily"]["date"][T - 20]
    dd[-1] = -0.049
    assert "drawdown_open" not in W(d)


def test_warning_short_history(ts, ts_short):
    assert "short_history" in W(ts_short)
    assert W(ts_short)["short_history"]["values"]["trading_days"] == 119
    assert "short_history" not in W(ts)


def test_warning_benchmark_late_and_filled(ts, ts_late):
    assert "benchmark_late" in W(ts_late) and "benchmark_late" not in W(ts)
    assert (
        W(ts_late)["benchmark_late"]["values"]["benchmark_start"]
        == ts_late["summary"]["benchmark_start"]
    )
    d = copy.deepcopy(ts)
    assert "benchmark_filled" not in W(d)
    d["meta"]["benchmark"]["filled_days"] = 3
    assert W(d)["benchmark_filled"]["values"]["filled_days"] == 3


def test_warning_weekday_basis(ts):
    d = copy.deepcopy(ts)
    d["meta"]["completeness_basis"] = "calendar"
    assert "weekday_basis" not in W(d)
    d["meta"]["completeness_basis"] = "weekday"
    assert "weekday_basis" in W(d)


def test_warnings_listed_in_default_summary(ts_short):
    out = str(rp.Tearsheet(ts_short)).replace("\n  ", "")  # 去掉折行
    codes = [w["code"] for w in rp.checks.warnings(ts_short)]
    assert codes and ("提醒：" + "、".join(codes)) in out
    full = rp.Tearsheet(ts_short).text("warnings")
    assert all(f"[{c}]" in full for c in codes)


# ---------------------------------------------------------------------- 存取与命令行
def run_cli(args) -> tuple[int, str]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = cli(args)
    return code, buf.getvalue()


def test_save_load_and_cli(ts, tmp_path):
    p = rp.Tearsheet(ts).save_json(tmp_path / "r.json")
    back = rp.Tearsheet.load(p)
    assert back.data == json.loads(json.dumps(ts))
    code, out = run_cli(["show", str(p)])
    assert code == 0 and out == str(rp.Tearsheet(ts))
    code, out = run_cli(
        [
            "show",
            str(p),
            "-s",
            "yearly",
            "--section",
            "warnings,costs",
            "-f",
            "markdown",
        ]
    )
    assert code == 0 and out == rp.Tearsheet(ts).text(
        ["yearly", "warnings", "costs"], "markdown"
    )
    d = ts["daily"]["date"]
    code, out = run_cli(
        ["show", str(p), "--from", d[50], "--to", d[300], "-f", "json", "--lang", "en"]
    )
    assert code == 0 and json.loads(out)["range"]["first_trade_day"] == d[50]
    code, out = run_cli(["html", str(p), "-o", str(tmp_path / "r.html")])
    assert code == 0
    assert (tmp_path / "r.html").read_text(encoding="utf-8") == rp.to_html(back.data)


def test_cli_errors(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    code, _ = run_cli(["show", str(bad)])
    assert code == 2
    code, _ = run_cli(["show", str(tmp_path / "missing.json")])
    assert code == 2


def test_cli_module_entry(ts, tmp_path):
    import sys

    p = rp.Tearsheet(ts).save_json(tmp_path / "r.json")
    r = subprocess.run(
        [sys.executable, "-m", "ashare.report", "show", str(p), "-s", "warnings"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0 and r.stdout.startswith("提醒")


def test_repr_and_str(ts):
    t = rp.Tearsheet(ts)
    assert "Tearsheet" in repr(t)
    assert str(t).startswith(ts["meta"]["title"])
