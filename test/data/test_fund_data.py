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
    assert (
        report["checks"]["fund_adj_vs_gap"]["details"]["split_or_conversion_events"]
        == 1
    )
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
    item = report["checks"]["fund_adj_vs_gap"]
    assert item["verdict"] == "fail"
    assert item["details"]["unexplained"] == [
        dict(symbol="510001.SH", date=DAYS[SPLIT_DAY])
    ]
    # 面板只在「复权因子也变了」时认拆分：这一天不当拆分。
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
