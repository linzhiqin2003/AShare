"""边界与口径：阈值恰好相等 / 差一个最小量、年化门槛、宽字符对齐、持平不算跑赢、
期初持仓不算持有天数、多于十只时的前十大；以及自选区间的提醒、命令行的错误输入、折行。"""

from __future__ import annotations

import copy
import io
import json
import math
import re
from contextlib import redirect_stderr, redirect_stdout

import numpy as np
import pytest

import ashare.report as rp
from ashare.report import Benchmark, build_tearsheet
from ashare.report import checks as ck
from ashare.report import extras as ex
from ashare.report import window as win
from ashare.report.__main__ import main as cli
from ashare.report.fmt import pad, width, wrap
from ashare.report.render import EXTRA_KEYS, compact
from ashare.report.sections import SUMMARY_MAX_LINES, Table
from ashare.report.text import _table_text
from test.report.helpers import build, business_days, make_case

TINY = 1e-12


@pytest.fixture(scope="module")
def case():
    return make_case(T=600, seed=3, start="2019-12-31")  # 基准日在上一年末：首年完整


@pytest.fixture(scope="module")
def ts(case):
    return build(case)


def W(d, a=None, b=None):
    return {w["code"]: w for w in ck.warnings(d, a, b)}


def blank(ts):
    """一份所有提醒都不触发的报告数据（在它上面逐条把某个量推到阈值）。"""
    d = copy.deepcopy(ts)
    T = len(d["daily"]["date"])
    d["meta"]["benchmark"]["kind"] = "total_return"
    d["meta"]["benchmark"]["filled_days"] = 0
    d["meta"]["completeness_basis"] = "calendar"
    d["summary"]["benchmark_start"] = d["summary"]["start"]
    d["monthly"]["first_partial"] = d["monthly"]["last_partial"] = False
    for y in d["yearly"]:
        y["partial"] = False
    d["daily"]["drawdown"] = [0.0] * T
    d["daily"]["cash_weight"] = [0.0] * T
    for r in d["rebalances"]:
        r.update(
            unfilled=0,
            unfilled_by_reason={"limit_up": 0, "limit_down": 0, "suspended": 0},
            new_names=1,
            dropped_names=1,
            targets=10,
            holdings_after=10,
            fees=0.0,
            slippage=0.0,
        )
    assert W(d) == {}
    return d


# ====================================================================== 提醒的阈值
def test_blank_triggers_nothing(ts):
    blank(ts)


def test_short_history_threshold(ts):
    d = blank(ts)
    P = d["meta"]["periods_per_year"]
    d["summary"]["trading_days"] = P  # 恰好一年：不提醒
    assert "short_history" not in W(d)
    d["summary"]["trading_days"] = P - 1
    assert W(d)["short_history"]["values"]["trading_days"] == P - 1


def test_benchmark_price_index_needs_dividends(ts):
    d = blank(ts)
    d["meta"]["benchmark"]["kind"] = "price"
    d["summary"]["total_dividend_gross"] = 0.0
    w = W(d)["benchmark_price_index"]
    assert "偏高" not in w["message"] and "没有收到现金分红" in w["message"]
    assert w["values"]["dividends_received"] is False
    d["summary"]["total_dividend_gross"] = TINY
    w = W(d)["benchmark_price_index"]
    assert "超额因此偏高" in w["message"] and w["values"]["dividends_received"]


def test_benchmark_late_threshold(ts):
    d = blank(ts)
    dates = d["daily"]["date"]
    d["summary"]["benchmark_start"] = d["summary"]["start"]
    assert "benchmark_late" not in W(d)
    d["summary"]["benchmark_start"] = dates[1]
    assert W(d)["benchmark_late"]["values"]["benchmark_start"] == dates[1]


def test_benchmark_filled_threshold(ts):
    d = blank(ts)
    assert "benchmark_filled" not in W(d)
    d["meta"]["benchmark"]["filled_days"] = 1
    assert W(d)["benchmark_filled"]["values"]["filled_days"] == 1


def test_partial_flags_threshold(ts):
    d = blank(ts)
    d["monthly"]["first_partial"] = True
    assert W(d)["partial_start"]["values"]["first_month_partial"]
    d = blank(ts)
    d["summary"]["start"] = d["summary"]["first_trade_day"][:4] + "-01-01"  # 同年
    assert W(d)["partial_start"]["values"]["first_year_partial"]
    d = blank(ts)
    d["monthly"]["last_partial"] = True
    assert W(d)["partial_end"]["values"]["last_month_partial"]
    d = blank(ts)
    d["yearly"][-1]["partial"] = True
    assert W(d)["partial_end"]["values"]["last_year_partial"]


def test_weekday_basis_threshold(ts):
    d = blank(ts)
    d["meta"]["completeness_basis"] = "weekday"
    assert "weekday_basis" in W(d)


def test_unfilled_threshold(ts):
    d = blank(ts)
    rb = d["rebalances"]
    for r in rb:
        r["new_names"], r["dropped_names"] = 0, 0
    rb[0]["new_names"] = 100  # 合计 100 只次
    rb[0]["unfilled"] = 5  # 恰好 5%：提醒
    rb[0]["unfilled_by_reason"]["suspended"] = 5
    assert W(d)["unfilled"]["values"]["ratio"] == 0.05
    rb[0]["new_names"] = 101  # 5 / 101 < 5%
    assert "unfilled" not in W(d)


def test_holdings_short_threshold(ts):
    d = blank(ts)
    rb = d["rebalances"]
    assert len(rb) >= 20
    rb[:] = rb[:20]
    rb[0]["holdings_after"] = 8  # 恰好 80%：不算偏少
    assert "holdings_short" not in W(d)
    rb[0]["holdings_after"] = 7
    rb[1]["holdings_after"] = 7  # 2 / 20 = 10%：提醒
    assert W(d)["holdings_short"]["values"]["count"] == 2
    rb[1]["holdings_after"] = 8  # 1 / 20 < 10%
    assert "holdings_short" not in W(d)


def test_cash_high_threshold(ts):
    d = blank(ts)
    T = len(d["daily"]["date"])
    d["daily"]["cash_weight"] = [ck.CASH_MEDIAN] * T  # 中位数恰好 5%：不提醒
    assert "cash_high" not in W(d)
    d["daily"]["cash_weight"] = [ck.CASH_MEDIAN + TINY] * T
    assert W(d)["cash_high"]["values"]["median"] == ck.CASH_MEDIAN + TINY


def test_drawdown_open_threshold(ts):
    d = blank(ts)
    dd = d["daily"]["drawdown"]
    dd[-1] = ck.OPEN_DRAWDOWN  # 恰好 −5%：提醒
    assert W(d)["drawdown_open"]["values"]["current"] == ck.OPEN_DRAWDOWN
    dd[-1] = ck.OPEN_DRAWDOWN + TINY
    assert "drawdown_open" not in W(d)


def test_costs_high_threshold(ts):
    d = blank(ts)
    T = len(d["daily"]["date"])
    d["meta"]["periods_per_year"] = T - 1  # 区间正好一年
    d["meta"]["initial_capital"] = 1e6
    d["daily"]["nav"] = [1.0] * T  # 平均权益 = 初始资金
    d["rebalances"][0]["fees"] = 20000.0
    d["rebalances"][1]["slippage"] = 10000.0  # 合计 3%：恰好等于阈值，不提醒
    assert "costs_high" not in W(d)
    d["rebalances"][1]["slippage"] = 10001.0
    w = W(d)["costs_high"]
    assert w["values"]["ratio"] == pytest.approx(0.030001)
    assert "费用 2.00%、滑点 1.00%" in w["message"]


def test_costs_high_on_a_real_high_turnover_backtest():
    c = make_case(T=300, seed=4, every=1)  # 每天调仓
    d = build(c)
    w = W(d)["costs_high"]
    rb = d["rebalances"]
    cap = d["meta"]["initial_capital"]
    nav = d["daily"]["nav"]
    years = (len(nav) - 1) / d["meta"]["periods_per_year"]
    ref = sum(r["fees"] + r["slippage"] for r in rb) / (np.mean(nav[1:]) * cap) / years
    assert w["values"]["ratio"] == pytest.approx(ref, rel=1e-12)
    assert ref > ck.COST_HIGH


# ====================================================================== 自选区间的提醒
def test_window_warnings_are_recomputed(ts):
    d = blank(ts)
    dates = d["daily"]["date"]
    T = len(dates)
    nav = list(np.linspace(1.0, 2.0, T))
    a, b = 100, 300
    low = nav[200] * 0.9
    nav[200:301] = [low] * 101  # 区间末日在回撤里（约 −10%），之后修复
    d["daily"]["nav"] = nav
    w = W(d, a, b)
    # 全区间看不到的问题，区间里看得到
    assert "drawdown_open" not in W(d)
    dw = w["drawdown_open"]["values"]
    assert dw["current"] == pytest.approx(low / nav[199] - 1, rel=1e-12)
    assert dw["peak"] == dates[199]
    assert w["short_history"]["values"]["trading_days"] == b - a
    assert "所选区间只有" in w["short_history"]["message"]
    # 区间起止把月截断（前后一天同月）
    if dates[a][:7] == dates[a + 1][:7]:
        assert w["partial_start"]["values"]["first_month_partial"]
    if dates[b + 1][:7] == dates[b][:7]:
        assert w["partial_end"]["values"]["last_month_partial"]
    # 区间内的调仓：未成交只看区间里的
    idx = {x: i for i, x in enumerate(dates)}
    inside = [r for r in d["rebalances"] if a < idx[r["date"]] <= b]
    outside = [r for r in d["rebalances"] if not a < idx[r["date"]] <= b]
    for r in outside:
        r["unfilled"] = r["new_names"] + r["dropped_names"]
        r["unfilled_by_reason"]["suspended"] = r["unfilled"]
    assert "unfilled" in W(d) and "unfilled" not in W(d, a, b)
    inside[0]["unfilled"] = 2
    inside[0]["unfilled_by_reason"]["suspended"] = 2
    assert W(d, a, b)["unfilled"]["values"]["unfilled"] == 2
    # 现金占比只看区间
    d["daily"]["cash_weight"] = [0.0] * T
    d["daily"]["cash_weight"][a + 1 : b + 1] = [0.2] * (b - a)
    assert W(d, a, b)["cash_high"]["values"]["median"] == 0.2
    assert "cash_high" not in W(d, 0, a)
    d["daily"]["cash_weight"] = [0.5] * 301 + [0.0] * (T - 301)  # 区间之前的高现金不算
    assert "cash_high" in W(d) and "cash_high" not in W(d, 300, 400)
    # 只能按全区间给的提醒标明「全区间」
    d["meta"]["benchmark"]["filled_days"] = 2
    assert "（全区间）" in W(d, a, b)["benchmark_filled"]["message"]
    assert "（全区间）" not in W(d)["benchmark_filled"]["message"]


def test_window_benchmark_late_and_dividends(ts):
    d = blank(ts)
    dates = d["daily"]["date"]
    bnav = d["daily"]["benchmark_nav"]
    d["daily"]["benchmark_nav"] = [None] * 150 + bnav[150:]
    w = W(d, 100, 400)["benchmark_late"]["values"]
    assert w["benchmark_start"] == dates[150]
    assert "benchmark_late" not in W(d, 200, 400)
    # 价格指数：区间所在年份有没有分红
    d["meta"]["benchmark"]["kind"] = "price"
    d["summary"]["total_dividend_gross"] = 1.0
    for a in d["attribution"]:
        a["dividend_net"] = 0.0
    y_last = d["yearly"][-1]["year"]
    d["attribution"][-1]["dividend_net"] = 0.01
    first_year_end = max(i for i, x in enumerate(dates) if x[:4] == dates[1][:4])
    assert not W(d, 0, first_year_end)["benchmark_price_index"]["values"][
        "dividends_received"
    ]
    last_start = min(i for i, x in enumerate(dates) if x[:4] == str(y_last))
    assert W(d, last_start - 1, len(dates) - 1)["benchmark_price_index"]["values"][
        "dividends_received"
    ]


# ====================================================================== 年化门槛
def test_annualized_threshold_239_240_241(ts):
    assert win.MIN_ANNUAL_DAYS == 240
    for n, want in ((239, False), (240, True), (241, True)):
        assert win.window_summary(ts, 10, 10 + n)["annualized"] is want
        out = rp.Tearsheet(ts).text(
            "summary", start=ts["daily"]["date"][11], end=ts["daily"]["date"][10 + n]
        )
        assert ("年化收益" in out) is want
        assert ("区间超额" in out) is (not want)


# ====================================================================== 宽字符对齐
def cell_spans(line: str, cells: list[str]) -> list[tuple[int, int]]:
    """每格在这一行里的起止显示列（逐格向后找）。"""
    out, pos = [], 0
    for c in cells:
        k = line.index(c, pos)
        out.append((width(line[:k]), width(line[: k + len(c)])))
        pos = k + len(c)
    return out


def assert_aligned(t: Table, lines: list[str]):
    spans = [
        cell_spans(ln, row)
        for ln, row in zip(lines, [[h for h, _ in t.cols], *t.rows], strict=True)
    ]
    for j, (_, al) in enumerate(t.cols):
        edge = 0 if al == "l" else 1
        assert len({s[j][edge] for s in spans}) == 1, (j, lines)


def test_fullwidth_brackets_count_two_columns():
    assert width("（主）") == 6 and width("沪深300（主）") == 13
    assert pad("（主）", 8) == "（主）  "
    t = Table(
        [("指数", "l"), ("年化", "r"), ("口径", "l"), ("超额", "r")],
        [
            ["沪深300（主）", "+8.6%", "价格", "+1.0%"],
            ["CSI500", "-12.3%", "全收益（含分红）", "-0.4%"],
            ["中证1000", "+0.0%", "price", "+10.2%"],
        ],
    )
    assert_aligned(t, _table_text(t))


def test_real_benchmarks_table_aligned(case):
    rng = np.random.default_rng(9)
    T = len(case["dates"])
    other = 1000 * np.cumprod(1 + rng.normal(0, 0.01, T))
    d = build(case, benchmarks=[Benchmark(other, "中证500", "price", "000905.SH")])
    out = rp.Tearsheet(d).text("benchmarks")
    lines = out.splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith("指数"))
    rows = lines[i : i + 3]
    assert "（主）" in rows[1]
    ends = {width(r) for r in rows}  # 最后一列右对齐：每行显示宽度相同
    assert len(ends) == 1, rows
    # 英文：列名用指数代码，全是 ASCII
    en = rp.Tearsheet(d).text("benchmarks", lang="en").splitlines()
    head = next(ln for ln in en if ln.startswith("year "))
    assert head.isascii() and head.split() == [
        "year",
        "return",
        "000000.SH",
        "000905.SH",
    ]


# ====================================================================== 胜率：持平不算跑赢
def test_year_tie_is_not_a_win():
    dates = business_days("2020-01-01", 800)
    years = sorted({x[:4] for x in dates[1:]})
    yk = np.array([x[:4] for x in dates])
    rng = np.random.default_rng(1)
    nav = np.cumprod(1 + rng.normal(0, 0.01, len(dates)))
    bnav = nav.copy()  # 逐位相同：每年都持平
    last_2020 = int(np.flatnonzero(yk == "2020")[-1])
    bnav[last_2020] *= 0.99  # 只动 2020 年末：2020 跑赢、2021 跑输，其余年份仍持平
    r = ex.win_rate(nav, bnav, 0, dates, years, [False] * len(years), [])
    rows = {y["year"]: y for y in r["yearly"]}
    assert rows[2020]["won"] is True and rows[2021]["won"] is False
    for y in (2022, 2023):
        assert rows[y]["excess"] == 0.0
        assert rows[y]["won"] is False  # 持平不算跑赢
    assert r["years_won"] == 1 and r["years_counted"] == len(years)


# ====================================================================== 个股贡献：持有天数
def test_holding_days_exclude_the_base_day():
    from test.report.test_extras import action_case

    res, market, targets, dates, symbols = action_case()
    d = build_tearsheet(res, market, targets, dates, symbols)
    pos = np.asarray(res.positions)
    seen_base_holding = False
    for w in d["contributions"]["yearly"]:
        a = dates.index(w["start"])
        b = dates.index(w["end"])
        for h in w["top"] + w["bottom"]:
            i = symbols.index(h["symbol"])
            assert h["holding_days"] == int((pos[a + 1 : b + 1, i] > 0).sum())
            seen_base_holding |= bool(pos[a, i] > 0)
    assert seen_base_holding  # 用例里确有期初就持仓的证券


# ====================================================================== 集中度：多于十只
def test_top10_with_more_than_ten_holdings():
    c = make_case(T=200, N=16, seed=8, every=20, pick=13)
    d = build(c)
    pos = np.asarray(c["result"].positions)
    close = np.asarray(c["market"].close)
    eq = np.asarray(c["result"].equity)
    lv = np.asarray(c["result"].long_value)
    top10, nmax = [], 0
    for t in range(len(pos)):
        held = [i for i in range(pos.shape[1]) if pos[t, i] > 0]
        if not held:
            continue
        nmax = max(nmax, len(held))
        val = [pos[t, i] * close[t, i] for i in held]
        w = [x * lv[t] / sum(val) / eq[t] for x in val]
        top10.append(sum(sorted(w, reverse=True)[:10]))
    assert nmax > 10
    got = d["concentration"]["top10_weight"]
    assert got["mean"] == pytest.approx(np.mean(top10), abs=1e-6)
    assert got["max"] == pytest.approx(max(top10), abs=1e-6)


# ====================================================================== 默认摘要：行数与折行
def test_long_title_still_fits_forty_lines(ts):
    d = copy.deepcopy(ts)
    d["meta"]["title"] = "很长的标题，" * 60
    lines = rp.Tearsheet(d).text().splitlines()
    assert len(lines) <= SUMMARY_MAX_LINES
    assert all(width(x) <= 100 for x in lines)
    assert lines[1].endswith("…")  # 标题截到两行
    long = build(make_case(T=252 * 26, N=10, seed=5, every=63))
    long["meta"]["title"] = "(x)(x)长 abc def；" * 30
    lines = rp.Tearsheet(long).text().splitlines()
    assert len(lines) <= SUMMARY_MAX_LINES
    # 分年表标题里的「更早的 N 年」与实际显示的年数一致
    m = re.search(r"更早的 (\d+) 年", "\n".join(lines))
    shown = [x for x in lines if re.match(r"^\d{4}\*?\s", x)]
    if m:
        assert int(m.group(1)) + len(shown) == len(long["yearly"])


def test_wrap_keeps_command_fragments_whole():
    for k in range(60, 110):
        line = "提醒：" + "x" * (k - 6) + "（--section warnings 看详情）" + "、abc" * 20
        out = wrap(line)
        assert all(width(x) <= 100 for x in out)
        joined = "\n".join(out)
        assert "--section warnings" in joined, (k, out)
    for k in range(80, 100):
        out = wrap("a" * k + " --lang en --format json --from 2020-01-01")
        assert "--lang en" in "\n".join(out) and "--format json" in "\n".join(out)


# ====================================================================== 命令行的错误输入
def run(args):
    o, e = io.StringIO(), io.StringIO()
    with redirect_stdout(o), redirect_stderr(e):
        code = cli(args)
    return code, o.getvalue(), e.getvalue()


@pytest.mark.parametrize("bad", ["2020-13-45", "2020-02-30", "20201345", "2020/01/01"])
def test_invalid_dates_exit_2(ts, tmp_path, bad):
    p = rp.Tearsheet(ts).save_json(tmp_path / "r.json")
    for flag in ("--from", "--to"):
        code, out, err = run(["show", str(p), flag, bad])
        assert code == 2 and out == ""
        assert err.startswith("错误：") and "Traceback" not in err


@pytest.mark.parametrize(
    "content, hint",
    [
        ("not json", "不是有效的 JSON"),
        ("[1, 2]", "最外层应是一个对象"),
        ('{"meta":{},"summary":{},"daily":{}}', "缺 monthly"),
        (b"\xff\xfe\x00d", "不是文本文件"),
    ],
)
def test_bad_report_files_exit_2(tmp_path, content, hint):
    p = tmp_path / "bad.json"
    if isinstance(content, bytes):
        p.write_bytes(content)
    else:
        p.write_text(content, encoding="utf-8")
    code, out, err = run(["show", str(p)])
    assert code == 2 and out == "" and hint in err and "Traceback" not in err


def test_shape_errors_name_the_missing_part(ts, tmp_path):
    for drop in ("yearly", "rebalances"):
        d = copy.deepcopy(ts)
        del d[drop]
        p = tmp_path / f"{drop}.json"
        p.write_text(json.dumps(d), encoding="utf-8")
        code, _, err = run(["show", str(p)])
        assert code == 2 and f"缺 {drop}" in err
    d = copy.deepcopy(ts)
    d["daily"]["nav"] = d["daily"]["nav"][:-1]
    p = tmp_path / "short.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    code, _, err = run(["show", str(p)])
    assert code == 2 and "nav" in err
    # 形状检查之外的缺字段：仍是退出码 2、没有 Python 栈
    d = copy.deepcopy(ts)
    del d["yearly"][0]["return"]
    p = tmp_path / "deep.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    code, _, err = run(["show", str(p), "-s", "yearly"])
    assert code == 2 and "'return'" in err and "Traceback" not in err
    # 没有补充内容的旧报告照常读回
    d = copy.deepcopy(ts)
    for k in EXTRA_KEYS:
        d.pop(k, None)
    p = tmp_path / "old.json"
    p.write_text(json.dumps(d), encoding="utf-8")
    assert run(["show", str(p), "-s", "all"])[0] == 0


# ====================================================================== 超额与信息比率反号
def test_excess_and_ir_sign_note(ts):
    out = rp.Tearsheet(ts).text("summary", fmt="json")
    s = json.loads(out)["summary"]["summary"]
    dl = ts["daily"]
    o = 0
    r = np.array(compact(ts)["daily"]["return"][o + 1 :])
    q = np.array(compact(ts)["daily"]["benchmark_return"][o + 1 :])
    assert s["excess_daily_mean_annualized"] == pytest.approx(
        (r - q).mean() * ts["meta"]["periods_per_year"], rel=1e-12
    )
    assert len(dl["date"]) == len(r) + 1
    same = math.copysign(1, s["excess_annual_return"]) == math.copysign(
        1, s["information_ratio"]
    )
    text = rp.Tearsheet(ts).text()
    assert ("符号相反" in text) is (not same)
    d = copy.deepcopy(ts)
    d["summary"]["excess_annual_return"] = -abs(s["information_ratio"]) * 0.01
    d["summary"]["information_ratio"] = abs(s["information_ratio"])
    text = rp.Tearsheet(d).text()
    assert "注：超额与信息比率符号相反" in text and "日超额算术年化" in text
    en = rp.Tearsheet(d).text(lang="en")
    assert "excess_daily_mean_annualized=" in en


# ====================================================================== 对比表的星号
def test_compare_asterisk_does_not_shift_columns(case):
    a = rp.Tearsheet(build(case))
    c2 = make_case(T=500, seed=6, start="2020-03-02")
    b = rp.Tearsheet(build(c2))
    out = rp.compare([a, b], names=["甲", "乙"]).text()
    lines = out.splitlines()
    i = next(k for k, ln in enumerate(lines) if ln.startswith("年份"))
    rows = []
    for ln in lines[i + 1 :]:
        if not ln.strip():
            break
        rows.append(ln)
    assert any("*" in x for x in rows) and any(
        "* " not in x and "*" not in x for x in rows
    )
    # 每一列的「%」落在同一显示列上
    cols = [[width(ln[: m.start()]) for m in re.finditer("%", ln)] for ln in rows]
    assert len({tuple(c) for c in cols}) == 1, rows


def test_cli_traceback_flag_reraises_data_errors(monkeypatch, capsys):
    """默认把渲染阶段的缺字段压成一行提示；加 --traceback 时原样抛出，便于排查渲染代码。"""
    import pytest

    from ashare.report import Tearsheet
    from ashare.report import __main__ as cli

    class Broken:
        def text(self, *args, **kwargs):
            raise KeyError("yearly")

    monkeypatch.setattr(Tearsheet, "load", staticmethod(lambda path: Broken()))
    assert cli.main(["show", "x.json"]) == 2
    err = capsys.readouterr().err
    assert "yearly" in err and "--traceback" in err and "Traceback" not in err
    with pytest.raises(KeyError):
        cli.main(["--traceback", "show", "x.json"])
