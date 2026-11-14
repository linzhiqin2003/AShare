"""回测页（2026-11-14 新增）：设置白名单、选股与调仓日、与直接调用 panelbt/报告逐位一致、
HTTP 合同（不重跑、分页、窗口纪律）。全部使用合成数据。"""

import time

import numpy as np
import panelbt
import pytest
from panelbt.markets import ASHARE

import ashare.data
from ashare.backtest import boards, execution_targets, iso_dates, market_from_panel
from ashare.data.reader import Panel
from ashare.factor.windows import IS, VALIDATION
from ashare.platform.backtest import settings as config
from ashare.platform.backtest.engine import (
    WINDOWS,
    build_targets,
    run_backtest,
    signal_days,
)
from ashare.platform.language import parse
from ashare.platform.runner import PanelCache
from ashare.report import build_tearsheet
from bench.platform.synthetic import panel as synthetic
from bench.platform.synthetic import write_indexes, write_panel

ORIGIN = {"Origin": "http://127.0.0.1"}
SOURCE = "-ts_std(returns, 20)"


def backtest_panel(full=False):
    """合成面板补上回测要的公司行为与退市字段（缺失 = 当日无事件）。"""
    p = synthetic(full=full)
    t, n = len(p.dates), len(p.symbols)
    arrays = dict(p._arrays)
    split = np.full((t, n), np.nan)
    split[t // 2, 3] = 2.0  # 一次送转，检验股数与价格换算走引擎
    dividend = np.full((t, n), np.nan)
    dividend[t // 3, :10] = 0.3
    arrays.update(split=split, cash_dividend=dividend, delist=np.zeros((t, n), bool))
    # 几只股票在若干天涨停、停牌，成交约束要有真实计数。
    buy = np.array(arrays["can_buy_open"])
    buy[:, 0] = False
    arrays["can_buy_open"] = buy
    for value in arrays.values():
        value.flags.writeable = False
    return Panel(p.dates, p.symbols, arrays, "synthetic-v1")


@pytest.fixture(scope="module")
def data_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("backtest") / "data"
    source = backtest_panel(full=True)
    write_panel(root, source)
    write_indexes(root, source)
    return root


# ---------------------------------------------------------------- 设置


def test_settings_whitelist_rejects_windows_and_bad_types():
    assert config.normalize({}) == config.DEFAULTS
    for key in ("window", "start", "end", "TEST", "validation_window"):
        with pytest.raises(ValueError):
            config.normalize({key: "20260101"})
    for bad in (
        {"top_n": True},
        {"top_n": 2.5},
        {"top_n": 0},
        {"capital": "1000"},
        {"universe": "TEST"},
        {"carry_blocked": 1},
        {"dividend_tax": 0.15},
        {"benchmark": "../x"},
    ):
        with pytest.raises(ValueError):
            config.normalize(bad)
    # 0 与 0.0 是同一个税率，身份相同。
    assert config.normalize({"dividend_tax": 0})["dividend_tax"] == 0.0
    a = config.identity("d", config.normalize({"dividend_tax": 0}), "v")
    assert a == config.identity("d", config.normalize({"dividend_tax": 0.0}), "v")
    assert a != config.identity("d", config.normalize({"top_n": 31}), "v")
    assert a != config.identity("d", config.normalize({"dividend_tax": 0}), "v2")


# ---------------------------------------------------------------- 调仓日与选股


class Dates:
    def __init__(self, dates):
        self.dates = np.asarray(dates)


def test_signal_days_period_ends_and_first_day():
    days = np.arange("2024-01-01", "2024-03-10", dtype="datetime64[D]")
    days = np.char.replace(days[np.is_busday(days)].astype(str), "-", "")
    panel = Dates(days)
    month = signal_days(panel, 0, len(days), "month")
    assert [days[i] for i in month] == ["20240101", "20240131", "20240229", "20240308"]
    week = signal_days(panel, 0, len(days), "week")
    assert all(
        np.datetime64(f"{d[:4]}-{d[4:6]}-{d[6:]}").astype(object).weekday() == 4
        for d in days[week[1:-1]]
    )
    assert len(signal_days(panel, 0, len(days), "day")) == len(days)
    # 窗口中间起步：窗口首日也是信号日，月末判断看整张面板的下一行。
    start = int(np.searchsorted(days, "20240115"))
    inner = signal_days(panel, start, len(days) - 10, "month") + start
    assert days[inner[0]] == "20240115" and "20240131" in days[inner]


def test_build_targets_top_n_inside_pool_and_shift_one_day():
    alpha = np.array([[5.0, 4, 3, 2, 1], [1, 2, 3, 4, 5], [1, 1, 1, 1, 1]])
    pool = np.array([[False, True, True, True, True]] * 3)
    targets = build_targets(alpha, pool, np.array([0, 1]), 2, 3)
    assert targets.days.tolist() == [1, 2]
    dense = targets.to_dense(3, 5)
    assert dense[1].tolist() == [0, 0.5, 0.5, 0, 0]
    assert dense[2].tolist() == [0, 0, 0, 0.5, 0.5]


# ---------------------------------------------------------------- 与直接调用逐位一致


def direct(data_root, settings, window):
    """不经过平台：数据层读面板、手写选股、panelbt.run、build_tearsheet。"""
    data = ashare.data.open(str(data_root))
    full = data.panel(["close"])
    dates = np.asarray(full.dates)
    allowed = dates[dates < VALIDATION[1]]
    rows = np.flatnonzero((allowed >= window[0]) & (allowed < window[1]))
    first, last = allowed[rows[0]], allowed[rows[-1]]
    bt = market_from_panel(
        data,
        first,
        last,
        fields=[
            "member_csi300",
            "member_csi500",
            "is_listed",
            "industry",
            "is_suspended",
            "has_quote",
        ],
    )
    panel = bt.panel
    expression = parse(SOURCE)[0].expression
    from ashare.factor import Evaluator
    from ashare.platform.analysis import CutPanel

    cache = PanelCache(str(data_root))
    cache.ensure(set(expression.fields))
    alpha = Evaluator().evaluate(expression, CutPanel(cache, VALIDATION[1]))
    alpha = alpha[rows[0] : rows[-1] + 1]
    pool = (
        np.asarray(panel["member_csi300"]) | np.asarray(panel["member_csi500"])
    ) & np.asarray(panel["is_listed"])
    keys = np.array([d[:6] for d in allowed])
    signal = np.full(alpha.shape, np.nan)
    for i, r in enumerate(rows):
        if i == 0 or r + 1 >= len(allowed) or keys[r] != keys[r + 1]:
            x = np.where(pool[i] & np.isfinite(alpha[i]), alpha[i], -np.inf)
            top = np.argsort(-x, kind="stable")[: settings["top_n"]]
            top = top[np.isfinite(x[top])]
            signal[i] = 0.0
            signal[i, top] = 1 / len(top)
    targets = execution_targets(signal)
    costs = ASHARE.costs(
        iso_dates(panel.dates),
        commission_rate=settings["commission"],
        commission_min=settings["commission_min"],
        slippage=settings["slippage_bps"] / 1e4,
    )
    cfg = ASHARE.config(
        board=boards(data.securities(), panel.symbols),
        initial_cash=settings["capital"] * 1e4,
        record_trades=True,
        carry_blocked=settings["carry_blocked"],
        dividend_tax_rate=settings["dividend_tax"],
    )
    result = panelbt.run(bt.market, targets, costs, cfg)
    report = build_tearsheet(
        result,
        bt.market,
        targets,
        panel.dates,
        panel.symbols,
        initial_capital=settings["capital"] * 1e4,
        extras=False,
        top_drawdowns=10,
    )
    return result, report


def test_platform_backtest_matches_direct_panelbt_and_report_bitwise(data_root):
    settings = config.normalize({"universe": "TOP800", "top_n": 7})
    cache = PanelCache(str(data_root))
    outputs = run_backtest(cache, parse(SOURCE)[0].expression, settings)
    for part, window in WINDOWS.items():
        result, report = direct(data_root, settings, window)
        doc = outputs[part]["document"]
        assert doc["series"]["nav"] == report["daily"]["nav"]
        assert doc["series"]["drawdown"] == report["daily"]["drawdown"]
        for key in (
            "annual_return",
            "sharpe",
            "max_drawdown",
            "final_nav",
            "annual_turnover_one_sided",
            "total_fees_to_initial",
            "total_slippage_to_initial",
            "calmar",
            "monthly_win_rate",
        ):
            assert doc["summary"][key] == report["summary"][key], key
        assert [y["strategy"] for y in doc["yearly"]] == [
            y["return"] for y in report["yearly"]
        ]
        assert [y["price"] for y in doc["yearly"]] == [
            a["price"] for a in report["attribution"]
        ]
        trades = outputs[part]["trades"]
        assert len(trades["day"]) == len(result.trades)
        assert np.array_equal(trades["qty"], [t.qty for t in result.trades])
        assert np.array_equal(trades["price"], [t.price for t in result.trades])
        assert doc["constraints"]["rebalances"] == report["summary"]["rebalances"]
        assert doc["constraints"]["unfilled"] == sum(
            r["unfilled"] for r in report["rebalances"]
        )
        for reason in ("limit_up", "limit_down", "suspended"):
            assert doc["constraints"][reason] == sum(
                r["unfilled_by_reason"][reason] for r in report["rebalances"]
            )
    assert outputs["is"]["document"]["start"] >= "2016-01-01"
    assert outputs["validation"]["document"]["end"] < "2026-01-01"


def test_single_index_pool_holds_only_members(data_root):
    settings = config.normalize({"universe": "csi300", "top_n": 5})
    cache = PanelCache(str(data_root))
    outputs = run_backtest(cache, parse(SOURCE)[0].expression, settings)
    n = len(cache.symbols)
    members = {f"S{i:04d}" for i in range(n // 3)}
    for part in WINDOWS:
        held = {h["symbol"] for h in outputs[part]["document"]["holdings"]}
        assert held and held <= members
        assert set(outputs[part]["trades"]["symbol"].tolist()) <= members


# ---------------------------------------------------------------- HTTP


@pytest.fixture
def client(tmp_path, data_root):
    from fastapi.testclient import TestClient

    from ashare.platform.app import create_app

    app = create_app(tmp_path / "platform", data_root=str(data_root))
    with TestClient(app, base_url="http://127.0.0.1") as client:
        yield client


def wait(client, key, deadline=120):
    stop = time.monotonic() + deadline
    while time.monotonic() < stop:
        state = client.get(f"/api/backtests/{key}").json()
        if state["status"] not in ("queued", "running"):
            return state
        time.sleep(0.05)
    pytest.fail("回测未在期限内完成")


def test_http_run_reuse_pages_and_window_discipline(client):
    options = client.get("/api/backtests/options").json()
    assert options["windows"]["is"][0] == f"{IS[0][:4]}-{IS[0][4:6]}-{IS[0][6:]}"
    assert "test" not in options["windows"]
    body = {"source": SOURCE, "name": "低波", "settings": {"top_n": 5}}
    assert client.post("/api/backtests", json=body).status_code == 403  # 写请求要同源
    response = client.post("/api/backtests", json=body, headers=ORIGIN)
    assert response.status_code == 202, response.text
    key = response.json()["id"]
    assert wait(client, key)["status"] == "complete"
    # 注释、空格不同的同一表达式与设置：直接复用，不再排队。
    again = client.post(
        "/api/backtests",
        json=body | {"source": "# 低波\n-ts_std( returns, 20 )"},
        headers=ORIGIN,
    )
    assert again.json()["id"] == key and again.json()["status"] == "complete"
    for part in ("is", "validation"):
        doc = client.get(f"/api/backtests/{key}/result", params={"part": part}).json()
        assert len(doc["series"]["dates"]) == len(doc["series"]["nav"])
        trades = client.get(
            f"/api/backtests/{key}/trades", params={"part": part, "limit": 3}
        ).json()
        assert len(trades["items"]) == min(3, trades["total"])
        assert trades["total"] == doc["trade_count"]
        assert trades["items"][0]["date"] >= trades["items"][-1]["date"]
        rebalances = client.get(
            f"/api/backtests/{key}/rebalances", params={"part": part, "limit": 2}
        ).json()
        assert rebalances["total"] == doc["rebalance_count"]
    # TEST 不可达：没有这个窗口，也不收窗口参数。
    assert (
        client.get(f"/api/backtests/{key}/result", params={"part": "test"}).status_code
        == 422
    )
    for key_name in ("window", "start", "end"):
        bad = client.post(
            "/api/backtests",
            json=body | {"settings": {key_name: "20260101"}},
            headers=ORIGIN,
        )
        assert bad.status_code == 422 and "20260101" not in bad.text
    assert (
        client.post(
            "/api/backtests",
            json=body | {"window": ["20260101", "99991231"]},
            headers=ORIGIN,
        ).status_code
        == 422
    )
    listed = client.post(
        "/api/backtests",
        json=body | {"source": "-ts_std(returns, [10, 20])"},
        headers=ORIGIN,
    )
    assert listed.status_code == 422 and "一个表达式" in listed.json()["message"]
    assert client.get("/api/backtests/zzzz").status_code == 422
    assert client.get("/api/backtests/" + "0" * 24).status_code == 404
    assert (
        str(client.app.state.store.root) not in client.get(f"/api/backtests/{key}").text
    )


def test_http_cancel_queued_and_rerun_after_cancel(client):
    first = client.post(
        "/api/backtests",
        json={"source": SOURCE, "settings": {"top_n": 3}},
        headers=ORIGIN,
    ).json()["id"]
    second = client.post(
        "/api/backtests",
        json={"source": SOURCE, "settings": {"top_n": 4}},
        headers=ORIGIN,
    ).json()["id"]
    assert client.get("/api/backtests/status").json()["running"] >= 1
    # 顶栏「N 个任务运行中」把回测一起算上（第三轮，2026-11-14）。
    assert client.get("/api/meta").json()["running"] >= 2
    assert client.post(f"/api/backtests/{second}/cancel", headers=ORIGIN).json()[
        "cancelled"
    ]
    assert client.get(f"/api/backtests/{second}").json()["status"] == "cancelled"
    assert client.get(f"/api/backtests/{second}/result").status_code == 409
    assert wait(client, first)["status"] == "complete"
    rerun = client.post(
        "/api/backtests",
        json={"source": SOURCE, "settings": {"top_n": 4}},
        headers=ORIGIN,
    ).json()
    assert rerun["id"] == second and rerun["status"] in ("queued", "running")
    assert wait(client, second)["status"] == "complete"


def test_http_cancel_running_kills_worker(client):
    key = client.post(
        "/api/backtests",
        json={"source": SOURCE, "settings": {"top_n": 9}},
        headers=ORIGIN,
    ).json()["id"]
    stop = time.monotonic() + 30
    while client.get(f"/api/backtests/{key}").json()["status"] != "running":
        assert time.monotonic() < stop
        time.sleep(0.01)
    assert client.post(f"/api/backtests/{key}/cancel", headers=ORIGIN).json()[
        "cancelled"
    ]
    state = client.get(f"/api/backtests/{key}").json()
    assert state["status"] == "cancelled" and state["progress"] is None
    assert client.get("/api/backtests/status").json()["running"] == 0


# ---------------------------------------------------------------- 第二轮（2026-11-14）：年度调仓


def test_yearly_rebalance_last_trading_day_and_window_first_day():
    days = np.arange("2023-11-01", "2025-02-10", dtype="datetime64[D]")
    days = days[np.is_busday(days)]
    days = np.char.replace(days.astype(str), "-", "")
    panel = Dates(days)
    year = [days[i] for i in signal_days(panel, 0, len(days), "year")]
    # 窗口首日 + 每年最后一个交易日 + 面板末日（其信号在窗口内没有成交日）。
    assert year == ["20231101", "20231229", "20241231", "20250207"]
    # 窗口从某年第一个交易日起：首日本身是信号日，下一次是当年最后一个交易日。
    start = int(np.searchsorted(days, "20240101"))
    stop = int(np.searchsorted(days, "20250101"))
    inner = [days[i + start] for i in signal_days(panel, start, stop, "year")]
    assert inner == ["20240101", "20241231"]
    # 名单在年末定、下一交易日（次年首个交易日）成交。
    alpha = np.ones((len(days), 2))
    pool = np.ones((len(days), 2), bool)
    rel = signal_days(panel, 0, len(days), "year")
    targets = build_targets(alpha, pool, rel, 1, len(days))
    assert [days[d] for d in targets.days] == ["20231102", "20240101", "20250101"]
    assert config.normalize({"rebalance": "year"})["rebalance"] == "year"


# ---------------------------------------------------------------- 第二轮（2026-11-14）：结果清理


def fake_results(jobs, rows, size=100):
    """在结果目录里放几份假结果（每份 4 个文件共 size×4 字节）并登记状态。"""
    from ashare.platform.backtest.jobs import PARTS, SUFFIXES

    with jobs.connect() as db:
        for key, status, finished, opened in rows:
            db.execute(
                "INSERT INTO backtests (id,name,source,settings,status,created,finished,opened) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (key, key, "close", "{}", status, 1.0, finished, opened),
            )
            for part in PARTS:
                for suffix in SUFFIXES:
                    (jobs.folder / f"{key}.{part}.{suffix}").write_bytes(b"x" * size)


def test_prune_oldest_opened_first_and_keeps_busy_and_fresh(tmp_path):
    from ashare.platform.backtest.jobs import Backtests

    jobs = Backtests(tmp_path, max_bytes=10**9, keep_seconds=600)
    try:
        now = time.time()
        fake_results(
            jobs,
            [
                ("a" * 24, "complete", now - 9000, now - 5000),  # 最久没打开
                ("b" * 24, "complete", now - 8000, now - 100),
                ("c" * 24, "complete", now - 7000, None),  # 没打开过：按完成时间
                ("d" * 24, "running", None, None),  # 正在跑
                ("e" * 24, "complete", now - 10, None),  # 刚完成
            ],
        )
        (jobs.folder / ("f" * 24 + ".is.json")).write_bytes(b"x" * 400)  # 孤儿文件
        assert sum(jobs.usage().values()) == 2400
        jobs.max_bytes = 1300
        removed = jobs.prune()
        # 孤儿最先，然后 c（完成于 7000 秒前、从未打开）早于 a（5000 秒前打开过）。
        assert removed == ["f" * 24, "c" * 24, "a" * 24]
        assert sum(jobs.usage().values()) <= 1300
        assert jobs.status("a" * 24)["status"] == "evicted"
        assert jobs.row("a" * 24)["status"] == "evicted"  # 数据库状态同步改掉
        assert jobs.status("b" * 24)["status"] == "complete"
        assert not any(jobs.folder.glob("a" * 24 + ".*"))
        # 上限压到 0：正在跑和刚完成的仍然保留。
        jobs.max_bytes = 0
        jobs.prune()
        assert set(jobs.usage()) == {"d" * 24, "e" * 24}
        # 打开会刷新最近打开时间。
        jobs.touch("e" * 24)
        assert jobs.row("e" * 24)["opened"] >= now
    finally:
        jobs.close()


def test_prune_respects_environment_limit(tmp_path, monkeypatch):
    from ashare.platform.backtest.jobs import MAX_BYTES, Backtests

    assert MAX_BYTES == 2 << 30
    monkeypatch.setenv("ASHARE_PLATFORM_BACKTEST_MAX_BYTES", "12345")
    monkeypatch.setenv("ASHARE_PLATFORM_BACKTEST_KEEP_SECONDS", "0")
    jobs = Backtests(tmp_path)
    try:
        assert (jobs.max_bytes, jobs.keep_seconds) == (12345, 0.0)
    finally:
        jobs.close()


def test_http_evicted_result_asks_to_rerun(tmp_path, data_root, monkeypatch):
    from fastapi.testclient import TestClient

    from ashare.platform.app import create_app

    # 上限 1 字节、不保留刚完成的：每份结果完成即被清理。
    monkeypatch.setenv("ASHARE_PLATFORM_BACKTEST_MAX_BYTES", "1")
    monkeypatch.setenv("ASHARE_PLATFORM_BACKTEST_KEEP_SECONDS", "0")
    app = create_app(tmp_path / "platform", data_root=str(data_root))
    with TestClient(app, base_url="http://127.0.0.1") as client:
        body = {"source": SOURCE, "settings": {"top_n": 4}}
        key = client.post("/api/backtests", json=body, headers=ORIGIN).json()["id"]
        assert wait(client, key)["status"] == "evicted"
        response = client.get(f"/api/backtests/{key}/result")
        assert response.status_code == 410 and "重新运行" in response.json()["message"]
        assert client.get(f"/api/backtests/{key}/trades").status_code == 410
        again = client.post("/api/backtests", json=body, headers=ORIGIN).json()
        assert again["id"] == key and again["status"] in ("queued", "running")
        assert wait(client, key)["status"] == "evicted"


def test_options_follow_settings_page_default_benchmark(client):
    """设置页保存的默认基准同样是回测页的默认基准（第三轮，2026-11-14）。"""
    assert (
        client.get("/api/backtests/options").json()["defaults"]["benchmark"]
        == "000001.SH"
    )
    saved = client.put(
        "/api/preferences", json={"benchmark": "000300.SH"}, headers=ORIGIN
    )
    assert saved.status_code == 200, saved.text
    assert (
        client.get("/api/backtests/options").json()["defaults"]["benchmark"]
        == "000300.SH"
    )
