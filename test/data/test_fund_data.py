"""场内基金：采集、面板（与直白的参考实现逐格对拍）、规则、校验、读取接口。"""

import asyncio
import json
from pathlib import Path

import numpy as np
import panelbt as bt
import polars as pl
import pytest
from futures_fixtures import DAYS
from test_fund_fixtures import (
    DIV_DAY,
    DIVIDEND,
    LIMIT_DAY,
    SPLIT_DAY,
    STOCK,
    FakeFundTushare,
    stock_price,
)

import ashare
import ashare.data as ad
from ashare.data import cb_panel
from ashare.data.catalog import FUND_DATASETS
from ashare.data.fund_panel import build_funds
from ashare.data.fund_rules import classify, fund_market, price_limits
from ashare.data.fund_sync import sync_funds
from ashare.data.fund_verify import KNOWN_PATH, REGISTER_KEYS, verify_funds
from ashare.data.store import Store

assert (
    Path(ashare.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[2])
)


def fake_stock_closes(stock_root, dates, symbols):
    out = np.full((len(dates), len(symbols)), np.nan)
    for j, code in enumerate(symbols):
        if code in STOCK:
            out[:, j] = [stock_price(code, DAYS.index(d)) for d in dates]
    return out, dict(stock_panel="fake")


@pytest.fixture
def built(tmp_path, monkeypatch):
    monkeypatch.setattr(cb_panel, "stock_closes", fake_stock_closes)
    store = Store(tmp_path / "root")
    source = FakeFundTushare()
    asyncio.run(sync_funds(source, store, DAYS[0], DAYS[-1]))
    meta = build_funds(store)
    return store, source, meta


# ---------- 采集 ----------


def test_sync_completes_and_skips(tmp_path):
    store = Store(tmp_path / "root")
    source = FakeFundTushare(fail={("fund_daily", DAYS[3])})
    with pytest.raises(RuntimeError):
        asyncio.run(sync_funds(source, store, DAYS[0], DAYS[-1]))
    assert store.manifest()[f"fund_daily/{DAYS[3]}"]["status"] == "failed"
    calls = len(source.calls)
    asyncio.run(sync_funds(source, store, DAYS[0], DAYS[-1], refresh=False))
    retried = source.calls[calls:]
    assert ("fund_daily", {"trade_date": DAYS[3]}) in [
        (a, {k: v for k, v in p.items() if k == "trade_date"}) for a, p in retried
    ]
    # 不刷新时只补失败的分区和必须先取的日历、证券表。
    assert {a for a, _ in retried} <= {"trade_cal", "fund_daily", "cb_basic"}
    entries = store.entries()
    assert all(e["status"] == "complete" for e in entries.values())
    assert {e["dataset"] for e in entries.values()} == set(FUND_DATASETS)
    # cb_share 按债券、cb_price_chg 按「前四位 + 交易所」成组。
    assert {e["partition"] for e in store.entries("cb_share").values()} == {
        "113001.SH",
        "123001.SZ",
    }
    assert {e["partition"] for e in store.entries("cb_price_chg").values()} == {
        "1130_SH",
        "1230_SZ",
    }


def test_refresh_only_live_bonds(tmp_path):
    store = Store(tmp_path / "root")
    source = FakeFundTushare()
    asyncio.run(sync_funds(source, store, DAYS[0], DAYS[-1]))
    calls = len(source.calls)
    asyncio.run(sync_funds(source, store, DAYS[0], DAYS[-1]))
    again = source.calls[calls:]
    # 两只债都在「未退市或退市不足 60 天」内，都刷新；日分区只刷新最近 5 天。
    assert sorted(p["ts_code"] for a, p in again if a == "cb_share") == [
        "113001.SH",
        "123001.SZ",
    ]
    assert sorted(p["trade_date"] for a, p in again if a == "fund_daily") == DAYS[-5:]


def test_stock_sync_rejects_fund_datasets(tmp_path):
    from ashare.data.sync import sync

    with pytest.raises(ValueError, match="未知数据集"):
        asyncio.run(
            sync(
                FakeFundTushare(),
                Store(tmp_path / "r"),
                DAYS[0],
                DAYS[-1],
                ["fund_daily"],
            )
        )


# ---------- 面板：直白的参考实现 ----------


def reference_fund(source):
    """逐行循环、不做向量化地把原始表变成期望的面板值（与 fund_panel 不共用代码）。"""
    t = source.tables
    expected = {}
    for r in t["fund_daily"].iter_rows(named=True):
        key = (r["ts_code"], r["trade_date"])
        expected.setdefault(key, {}).update(
            open=r["open"],
            high=r["high"],
            low=r["low"],
            close=r["close"],
            previous_close=r["pre_close"],
            volume=r["vol"] * 100,
            amount=r["amount"] * 1000,
        )
    for r in t["fund_adj"].iter_rows(named=True):
        expected.setdefault((r["ts_code"], r["trade_date"]), {})["adj_factor"] = r[
            "adj_factor"
        ]
    for r in t["fund_share"].iter_rows(named=True):
        expected.setdefault((r["ts_code"], r["trade_date"]), {})["total_shares"] = (
            r["fd_share"] * 10000
        )
    return expected


def test_fund_panel_matches_reference(built):
    store, source, meta = built
    f = ad.open(store.root.parent).funds()
    names = [
        "open",
        "high",
        "low",
        "close",
        "previous_close",
        "volume",
        "amount",
        "adj_factor",
        "total_shares",
    ]
    p = f.panel(
        "fund", names + ["split", "cash_dividend", "has_quote", "is_listed", "delist"]
    )
    assert list(p.symbols) == [
        "159001.SZ",
        "510001.SH",
        "511001.SH",
    ]  # 161022A.SZ 不在基金表
    assert f.diagnostics("fund")["daily_codes_not_in_basic"] == ["161022A.SZ"]
    expected = reference_fund(source)
    for j, code in enumerate(p.symbols):
        for i, day in enumerate(p.dates):
            row = expected.get((code, day), {})
            for name in names:
                want = row.get(name, np.nan)
                got = p[name][i, j]
                assert (np.isnan(want) and np.isnan(got)) or want == got, (
                    code,
                    day,
                    name,
                )
            assert bool(p["has_quote"][i, j]) == ("close" in row)
    j = list(p.symbols).index("510001.SH")
    split = np.asarray(p["split"])[:, j]
    assert split[SPLIT_DAY] == pytest.approx(2.0, abs=1e-12)
    assert (np.delete(split, SPLIT_DAY) == 1).all()
    cash = np.asarray(p["cash_dividend"])[:, j]
    assert (
        cash[DIV_DAY] == DIVIDEND and cash.sum() == DIVIDEND
    )  # 「预案」与场外代码的行不进面板
    assert (np.asarray(p["split"])[:, [0, 2]] == 1).all()


def test_fund_listing_and_delist(built):
    store, _, _ = built
    f = ad.open(store.root.parent).funds()
    p = f.panel("fund", ["is_listed", "delist", "has_quote"], symbols=["159001.SZ"])
    listed = np.asarray(p["is_listed"])[:, 0]
    # 基金表上市日（第 2 天）晚于第一笔行情（第 0 天）：上市日取较早的那个，并记进诊断。
    assert listed[:19].all() and not listed[19:].any()
    assert np.flatnonzero(np.asarray(p["delist"])[:, 0]).tolist() == [19]
    moved = f.diagnostics("fund")["listing_before_table_date"]
    assert moved == [
        dict(symbol="159001.SZ", listing_date=DAYS[2], first_quote=DAYS[0])
    ]
    assert f.field("fund", "volume")["unit"] == "份"
    with pytest.raises(KeyError):
        f.panel("fund", ["close"], symbols=["999999.SH"])
    with pytest.raises(ValueError, match="超出"):
        f.panel("fund", ["close"], start="2025-01-02")


# ---------- 规则与回测行情 ----------


def test_classify_and_limits():
    securities = pl.DataFrame(
        dict(
            symbol=[
                "510300.SH",
                "159915.SZ",
                "513100.SH",
                "518880.SH",
                "511010.SH",
                "508000.SH",
                "511990.SH",
            ],
            name=[
                "沪深300ETF",
                "创业板ETF",
                "纳斯达克100ETF",
                "黄金ETF",
                "国债ETF",
                "张江REIT",
                "华宝添益货币",
            ],
            fund_type=[
                "股票型",
                "股票型",
                "股票型",
                "其他",
                "债券型",
                "REITs",
                "货币型",
            ],
        )
    )
    order = {s: i for i, s in enumerate(securities["symbol"].to_list())}
    c = classify(securities).sort(
        pl.col("symbol").replace_strict(order, return_dtype=pl.Int64)
    )
    assert c["asset_class"].to_list() == [
        "stock",
        "stock",
        "cross_border",
        "gold_commodity",
        "bond",
        "reits",
        "money",
    ]
    assert c["t0"].to_list() == [False, False, True, True, True, False, True]
    assert c.filter(pl.col("symbol") == "159915.SZ")["limit"].item() == 0.2
    dates = np.array(["20200821", "20200824"])
    up, down = price_limits(
        dates, np.full((2, 7), 1.0), c, listing_rows=[9, 9, 9, 9, 9, 0, 9]
    )
    assert up[:, 1].tolist() == [1.1, 1.2]  # 创业板 ETF 20% 自 2020-08-24
    assert up[0, 5] == pytest.approx(1.3) and up[1, 5] == pytest.approx(
        1.1
    )  # REITs 首日 30%


def test_fund_market_runs(built):
    store, _, _ = built
    f = ad.open(store.root.parent).funds()
    fm = fund_market(f, ["510001.SH", "511001.SH"])
    can_buy = np.asarray(fm.market.can_buy)
    assert not can_buy[LIMIT_DAY, 0] and can_buy[LIMIT_DAY - 1, 0]  # 一字涨停买不进
    weights = np.full(fm.shape, np.nan)
    weights[1] = [0.5, 0.5]
    result = bt.run(
        fm.market,
        bt.Targets.from_dense(weights),
        fm.costs(slippage=0.0),
        fm.config(initial_cash=1e6, record_trades=True),
    )
    shares = result.positions[:, 0]
    assert shares[SPLIT_DAY] == 2 * shares[SPLIT_DAY - 1]  # 拆分后份额翻倍
    assert result.dividend_gross[DIV_DAY] == pytest.approx(shares[DIV_DAY] * DIVIDEND)
    assert (shares % 100 == 0).all()
    costs = fm.costs()
    assert np.all(np.asarray(costs.tax_sell) == 0) and np.all(
        np.asarray(costs.fee_rate_buy) == 0
    )


# ---------- 校验 ----------


def test_verify_passes_and_fails(built):
    store, source, _ = built
    report = verify_funds(store, write=False, known={})
    failing = {
        k: v["verdict"] for k, v in report["checks"].items() if v["verdict"] == "fail"
    }
    # 合成数据里唯一的不符：161022A.SZ 不在基金表。
    assert failing == {"code_reuse": "fail"}
    known = dict(
        reasons={"x": "说明"},
        entries=[dict(check="fund_code_not_in_table", symbol="161022A.SZ", reason="x")],
    )
    report = verify_funds(store, write=True, known=known)
    assert (
        report["verdict"] == "pass"
        and report["checks"]["code_reuse"]["verdict"] == "explained"
    )
    for name in ("fund_gap", "fund_factor", "fund_factor_missing"):
        assert report["checks"][name]["verdict"] == "pass", name
    details = report["checks"]["fund_factor"]["details"]
    assert details["split_cells"] == 1 and details["dividend_cells"] == 1
    assert (store.root / "reports/funds_verify.md").exists()


def test_verify_catches_broken_adj(tmp_path, monkeypatch):
    monkeypatch.setattr(cb_panel, "stock_closes", fake_stock_closes)
    store = Store(tmp_path / "root")
    source = FakeFundTushare()
    adj = source.tables["fund_adj"]
    # 拆分日的复权因子没变：前收盘缺口与复权因子对不上，校验必须失败。
    source.tables["fund_adj"] = adj.with_columns(
        pl.when(
            (pl.col("ts_code") == "510001.SH")
            & (pl.col("trade_date") >= DAYS[SPLIT_DAY])
        )
        .then(pl.col("adj_factor") / 2)
        .otherwise(pl.col("adj_factor"))
        .alias("adj_factor")
    )
    asyncio.run(sync_funds(source, store, DAYS[0], DAYS[-1]))
    build_funds(store)
    report = verify_funds(store, write=False, known={})
    item = report["checks"]["fund_gap"]
    assert item["verdict"] == "fail"
    assert item["details"]["unexplained"] == [
        dict(symbol="510001.SH", date=DAYS[SPLIT_DAY])
    ]
    # 面板以复权因子为准：因子没变，这一天不当拆分（与因子一致，所以只有前收盘那条路径报错）。
    assert report["checks"]["fund_factor"]["verdict"] == "pass"
    p = (
        ad.open(store.root.parent)
        .funds()
        .panel("fund", ["split"], symbols=["510001.SH"])
    )
    assert (np.asarray(p["split"]) == 1).all()


def test_known_exceptions_only_keys():
    """清单只放匹配键与原因（不放行情算出的数值），原因都有说明。"""
    doc = json.loads(KNOWN_PATH.read_text())
    for entry in doc["entries"]:
        allowed = {"check", "reason", *REGISTER_KEYS[entry["check"]]}
        assert set(entry) == allowed, entry
        assert entry["reason"] in doc["reasons"]
    assert set(doc["cause_unknown"]) <= set(doc["reasons"])


# ---------- 停牌期间的拆分、重复分红、代码复用、沿用价 ----------


def build_from(tmp_path, monkeypatch, source):
    monkeypatch.setattr(cb_panel, "stock_closes", fake_stock_closes)
    store = Store(tmp_path / "root")
    asyncio.run(sync_funds(source, store, DAYS[0], DAYS[-1]))
    build_funds(store)
    return store, ad.open(store.root.parent).funds()


def panel_path(store):
    base = store.root / "fund_panel" / "fund"
    return base / json.loads((base / "current.json").read_text())["version"]


def tamper(store, name, row, symbol, value):
    """改写已构建面板的一个格子（模拟构建代码的错误），再看校验是否抓到。"""
    folder = panel_path(store)
    col = json.loads((folder / "metadata.json").read_text())["symbols"].index(symbol)
    path = folder / f"{name}.npy"
    a = np.load(path)
    a[row, col] = value
    np.save(path, a)


def suspend_over_split(source):
    """510001.SH 在拆分日及前一天停牌：源里没有这两天的日线，复权因子照常在拆分日翻倍；
    复牌第一天的交易所前收盘 = 停牌前收盘 / 2。"""
    t = source.tables
    gone = [DAYS[SPLIT_DAY - 1], DAYS[SPLIT_DAY]]
    daily = t["fund_daily"]
    last = daily.filter(
        (pl.col("ts_code") == "510001.SH")
        & (pl.col("trade_date") == DAYS[SPLIT_DAY - 2])
    )["close"].item()
    t["fund_daily"] = daily.filter(
        ~((pl.col("ts_code") == "510001.SH") & pl.col("trade_date").is_in(gone))
    ).with_columns(
        pl.when(
            (pl.col("ts_code") == "510001.SH")
            & (pl.col("trade_date") == DAYS[SPLIT_DAY + 1])
        )
        .then(pl.lit(last / 2))
        .otherwise(pl.col("pre_close"))
        .alias("pre_close")
    )
    return last


def test_split_during_suspension_lands_on_resumption(tmp_path, monkeypatch):
    source = FakeFundTushare()
    suspend_over_split(source)
    store, f = build_from(tmp_path, monkeypatch, source)
    p = f.panel("fund", ["split", "has_quote", "action_source"], symbols=["510001.SH"])
    split = np.asarray(p["split"])[:, 0]
    assert not np.asarray(p["has_quote"])[[SPLIT_DAY - 1, SPLIT_DAY], 0].any()
    assert split[SPLIT_DAY + 1] == pytest.approx(2.0, abs=1e-12)
    assert (np.delete(split, SPLIT_DAY + 1) == 1).all()
    assert np.asarray(p["action_source"])[SPLIT_DAY + 1, 0] == 3
    report = verify_funds(store, write=False, known={})
    assert report["checks"]["fund_gap"]["verdict"] == "pass"
    assert report["checks"]["fund_factor"]["verdict"] == "pass"
    # 面板若漏掉这次拆分（修复前的写法），两条校验路径都要报出复牌那一天。
    tamper(store, "split", SPLIT_DAY + 1, "510001.SH", 1.0)
    report = verify_funds(store, write=False, known={})
    for name in ("fund_gap", "fund_factor"):
        assert report["checks"][name]["details"]["unexplained"] == [
            dict(symbol="510001.SH", date=DAYS[SPLIT_DAY + 1])
        ], name


def test_dividend_during_suspension_moves_to_resumption(tmp_path, monkeypatch):
    source = FakeFundTushare()
    t = source.tables
    day = DIV_DAY
    daily = t["fund_daily"]
    t["fund_daily"] = daily.filter(
        ~((pl.col("ts_code") == "510001.SH") & (pl.col("trade_date") == DAYS[day]))
    )
    last = daily.filter(
        (pl.col("ts_code") == "510001.SH") & (pl.col("trade_date") == DAYS[day - 1])
    )["close"].item()
    t["fund_daily"] = t["fund_daily"].with_columns(
        pl.when(
            (pl.col("ts_code") == "510001.SH") & (pl.col("trade_date") == DAYS[day + 1])
        )
        .then(pl.lit(last - DIVIDEND))
        .otherwise(pl.col("pre_close"))
        .alias("pre_close")
    )
    store, f = build_from(tmp_path, monkeypatch, source)
    p = f.panel("fund", ["cash_dividend"], symbols=["510001.SH"])
    cash = np.asarray(p["cash_dividend"])[:, 0]
    assert cash[day + 1] == DIVIDEND and cash.sum() == DIVIDEND
    report = verify_funds(store, write=False, known={})
    assert report["checks"]["fund_gap"]["verdict"] == "pass"
    assert report["checks"]["fund_factor"]["verdict"] == "pass"


def duplicate_dividend(source):
    """同一次分红两行：一行除息日写成登记日（前一天）、net_ex_date 为空；另一行是真正的除息日。"""
    t = source.tables
    div = t["fund_div"]
    real = div.filter(
        (pl.col("ts_code") == "510001.SH") & (pl.col("div_proc") == "实施")
    ).with_columns(
        pl.lit(DAYS[DIV_DAY - 1]).alias("record_date"),
        pl.lit(DAYS[DIV_DAY]).alias("net_ex_date"),
    )
    fake = real.with_columns(
        pl.lit(DAYS[DIV_DAY - 1]).alias("ex_date"),
        pl.lit(None, dtype=pl.String).alias("net_ex_date"),
    )
    others = div.filter(
        ~((pl.col("ts_code") == "510001.SH") & (pl.col("div_proc") == "实施"))
    )
    t["fund_div"] = pl.concat([others, fake, real])


def test_duplicate_dividend_counted_once(tmp_path, monkeypatch):
    source = FakeFundTushare()
    duplicate_dividend(source)
    store, f = build_from(tmp_path, monkeypatch, source)
    p = f.panel("fund", ["cash_dividend", "action_source"], symbols=["510001.SH"])
    cash = np.asarray(p["cash_dividend"])[:, 0]
    assert cash[DIV_DAY] == DIVIDEND and cash.sum() == DIVIDEND
    assert f.diagnostics("fund")["actions"]["distributions"] == 1
    report = verify_funds(store, write=False, known={})
    assert report["checks"]["fund_gap"]["verdict"] == "pass"
    # 修复前的写法会在登记日再记一次：两条校验路径都要报出那一天。
    tamper(store, "cash_dividend", DIV_DAY - 1, "510001.SH", DIVIDEND)
    report = verify_funds(store, write=False, known={})
    for name in ("fund_gap", "fund_factor"):
        assert report["checks"][name]["details"]["unexplained"] == [
            dict(symbol="510001.SH", date=DAYS[DIV_DAY - 1])
        ], name


def test_dividend_rejected_when_factor_does_not_move():
    from ashare.data.fund_actions import _decide

    assert _decide(4.0, 0.4, 1.0, 1.0) == (1.0, 0.0, 4)  # 因子没动：否决
    assert _decide(4.0, 0.4, 4.0 / 3.6, 4.0 / 3.6) == (1.0, 0.4, 1)
    s, c, code = _decide(4.0, 0.4, 2 * 4.0 / 3.6, 1.0)  # 分红加 1 拆 2
    assert code == 2 and c == 0.4 and s == pytest.approx(2.0)
    assert _decide(4.0, 0.0, 1.0004, 1.0) == (
        1.0004,
        0.0,
        3,
    )  # 噪声以内仍记（因子为准）
    assert _decide(4.0, 0.0, 1.0, 2.0)[2] == 3  # 因子没动：前收盘缺口不算拆分
    assert _decide(4.0, 0.0, np.nan, 2.0) == (2.0, 0.0, 5)  # 没有因子才看缺口


def test_reused_codes_resolved():
    from ashare.data.fund_actions import resolve_reused

    basic = pl.DataFrame(
        dict(
            ts_code=["502000.SH", "5020001.SH", "160123.SZ", "1601231.SZ"],
            name=["新LOF", "旧分级", "同一只", "同一只"],
            list_date=["20200219", "20150427", "20110725", "20110725"],
        )
    )
    daily = pl.DataFrame(
        dict(
            symbol=["502000.SH"] * 4 + ["5020001.SH"] * 2 + ["1601231.SZ"] * 2,
            date=["d1", "d2", "d3", "d4", "d2", "d2x", "d1", "d2"],
            close=[1.0, 2.0, 3.0, 4.0, 9.0, 8.0, 5.0, 6.0],
        )
    )
    adj = pl.DataFrame(
        dict(
            symbol=["502000!1.SH", "502000!1.SH", "502000.SH", "502000.SH"]
            + ["160123!1.SZ"] * 2,
            date=["d1", "d2", "d3", "d4", "d1", "d2"],
            adj_factor=[0.5, 0.5, 1.0, 1.0, 1.2, 1.2],
        )
    )
    d, a, drop, notes = resolve_reused(basic, daily, adj)
    rows = {(r["symbol"], r["date"]): r["close"] for r in d.iter_rows(named=True)}
    # 两只基金：旧段（六位代码当天没有自己的因子）归七位代码；七位代码同日的零星行丢掉、别的日子保留。
    assert rows[("5020001.SH", "d1")] == 1.0 and rows[("5020001.SH", "d2")] == 2.0
    assert rows[("5020001.SH", "d2x")] == 8.0
    assert ("502000.SH", "d1") not in rows and rows[("502000.SH", "d3")] == 3.0
    # 同一只基金：并到六位代码，七位代码不建列。
    assert rows[("160123.SZ", "d1")] == 5.0 and "1601231.SZ" in drop
    factors = {
        (r["symbol"], r["date"]): r["adj_factor"] for r in a.iter_rows(named=True)
    }
    assert factors[("5020001.SH", "d1")] == 0.5 and factors[("160123.SZ", "d2")] == 1.2
    assert not any("!" in s for s, _ in factors)
    assert {n["rule"] for n in notes} == {"split", "merge", "stray_dropped"}


def test_stale_bar_is_not_a_quote(tmp_path, monkeypatch):
    source = FakeFundTushare()
    t = source.tables
    k = 20
    t["fund_daily"] = (
        t["fund_daily"]
        .with_columns(
            [
                pl.when(
                    (pl.col("ts_code") == "511001.SH")
                    & (pl.col("trade_date") == DAYS[k])
                )
                .then(pl.col("pre_close"))
                .otherwise(pl.col(c))
                .alias(c)
                for c in ("open", "high", "low", "close")
            ]
        )
        .with_columns(
            pl.when(
                (pl.col("ts_code") == "511001.SH") & (pl.col("trade_date") == DAYS[k])
            )
            .then(pl.col("amount") * 0.9)
            .otherwise(pl.col("amount"))
            .alias("amount")
        )
    )
    store, f = build_from(tmp_path, monkeypatch, source)
    p = f.panel("fund", ["has_quote"], symbols=["511001.SH"])
    has = np.asarray(p["has_quote"])[:, 0]
    assert not has[k] and has[k - 1] and has[k + 1]
    assert f.diagnostics("fund")["stale_bars"] == 1
    fm = fund_market(f, ["511001.SH"])
    assert not np.asarray(fm.market.can_buy)[k, 0]


def test_classify_known_misnames():
    """名称里的「有色」「MSCI中国A」「黄金…股票」都是 A 股股票 ETF（T+1），不能判成商品或跨境（T+0）。"""
    names = {
        "512400.SH": ("南方中证申万有色金属ETF", "股票型", "stock"),
        "560000.SH": ("华夏MSCI中国A50互联互通ETF", "股票型", "stock"),
        "517520.SH": ("永赢中证沪深港黄金产业股票ETF", "股票型", "stock"),
        "510170.SH": ("国联安上证大宗商品股票ETF", "股票型", "stock"),
        "159985.SZ": ("华夏饲料豆粕期货ETF", "其他", "gold_commodity"),
        "518850.SH": ("华夏黄金ETF", "其他", "gold_commodity"),
        "513000.SH": ("汇添富MSCI美国50ETF(QDII)", "股票型", "cross_border"),
    }
    securities = pl.DataFrame(
        dict(
            symbol=list(names),
            name=[v[0] for v in names.values()],
            fund_type=[v[1] for v in names.values()],
        )
    )
    got = dict(classify(securities).select("symbol", "asset_class").iter_rows())
    assert got == {k: v[2] for k, v in names.items()}
