"""模拟盘页接口（2026-11-14 新增）：合成账户，与直接调用 paperbroker 的结果逐项对照。

账户全部在临时目录里用 paperbroker + ashare.paper 的存储建立，不读真实数据。
"""

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from paperbroker import Broker, money

from ashare.paper.market import timestamp
from ashare.paper.profiles import configuration
from ashare.paper.storage import ParquetStore
from ashare.platform import paper_api
from ashare.platform.paper_api import navs, period_return

SYMBOLS = ["600000.SH", "000001.SZ", "601398.SH"]


def quote(i, t, price, volume, previous):
    p = money(price)
    return [
        i,
        t,
        p,
        p,
        p,
        p,
        money(previous),
        volume,
        volume * p,
        p,
        1_000_000,
        p - 100,
        1_000_000,
        money(previous * 1.1),
        money(previous * 0.9),
        0,
        0,
        0,
    ]


def build_account(directory, days, prices, plan, *, live_last=False, transfers=None):
    """逐日：开盘建量基线 → 下单 → 放量成交 → 收盘估值 → 结算。live_last 时最后一天停在盘中。"""
    store = ParquetStore(directory / "quotes")
    broker = Broker(
        SYMBOLS,
        1_000_000,
        configuration(SYMBOLS),
        journal=directory,
        quote_store=store,
    )
    previous = {s: prices[s][0] for s in SYMBOLS}
    for k, day in enumerate(days):
        iso = f"{day[:4]}-{day[4:6]}-{day[6:]}"

        def at(clock, iso=iso):
            return timestamp(f"{iso} {clock}")

        broker.session(at("09:00:00"), int(day), "preopen")
        if transfers and k in transfers:
            broker.transfer(at("09:01:00"), transfers[k])
        broker.session(at("09:30:00"), int(day), "continuous")
        open_rows = [
            quote(i, at("09:30:01"), previous[s], 1000, previous[s])
            for i, s in enumerate(SYMBOLS)
        ]
        broker.feed(np.array(open_rows, dtype=np.int64), received_at=at("09:30:01"))
        for symbol, side, qty in plan.get(k, []):
            limit = round(previous[symbol] * (1.05 if side == "buy" else 0.95), 2)
            broker.order(at("09:31:00"), symbol, side, qty, limit)
        # 下单后先观察一次（量不变），之后的放量区间才可用于成交。
        for clock, volume in (("09:31:30", 1000), ("09:32:00", 10_000_000)):
            rows = [
                quote(i, at(clock), previous[s], volume, previous[s])
                for i, s in enumerate(SYMBOLS)
            ]
            broker.feed(np.array(rows, dtype=np.int64), received_at=at(clock))
        close = [
            quote(i, at("14:59:00"), prices[s][k], 10_000_100, previous[s])
            for i, s in enumerate(SYMBOLS)
        ]
        broker.feed(np.array(close, dtype=np.int64), received_at=at("14:59:00"))
        if live_last and k == len(days) - 1:
            break
        broker.session(at("15:00:00"), int(day), "closed")
        broker.settle(at("15:02:00"), int(day))
        previous = {s: prices[s][k] for s in SYMBOLS}
    broker.events()
    broker.checkpoint()
    return broker


DAYS = ["20261030", "20261102", "20261103", "20261104"]
PRICES = {
    "600000.SH": [10.0, 10.2, 10.1, 10.5],
    "000001.SZ": [12.0, 11.8, 12.3, 12.6],
    "601398.SH": [6.0, 6.05, 6.1, 6.0],
}
PLAN = {
    0: [("600000.SH", "buy", 20_000), ("000001.SZ", "buy", 10_000)],
    2: [("600000.SH", "sell", 5_000), ("601398.SH", "buy", 30_000)],
    3: [("000001.SZ", "sell", 2_000), ("601398.SH", "buy", 100)],
}


class FakeReference:
    def lookup(self, symbol):
        return {
            "600000.SH": ("浦发银行", "银行"),
            "000001.SZ": ("平安银行", "银行"),
            "601398.SH": ("工商银行", "银行"),
        }.get(symbol, ("", None))

    def benchmark(self, start, end):
        closes = {"20261030": 4000.0, "20261102": 4040.0, "20261103": 3960.0}
        return dict(symbol="000300.SH", name="沪深300"), closes


@pytest.fixture
def root(tmp_path):
    return tmp_path / "paper"


def client_for(root, reference=None):
    app = FastAPI()
    app.include_router(
        paper_api.router(root=root, reference=reference or FakeReference())
    )
    return TestClient(app, base_url="http://127.0.0.1")


def test_navs_adjust_for_transfers_and_period_bases():
    # 入金不算收益：100 → 110（其中入金 10）净值不变。
    assert navs([(100, 100), (110, 110), (121, 110)], 100) == [1.0, 1.0, 1.1]
    assert navs([(0, 100), (10, 100)], 100) == [0.0, None]
    dates = ["20261030", "20261102"]
    # 本月以上月最后一个点为基；今年没有上一年的点，以开户（1）为基。
    assert period_return(dates, [1.1, 1.2], "20261104", 1.32, 6) == pytest.approx(0.2)
    assert period_return(dates, [1.1, 1.2], "20261104", 1.32, 4) == pytest.approx(0.32)


def test_empty_root_and_unknown_account(root):
    with client_for(root) as client:
        assert client.get("/api/paper/accounts").json() == {"items": []}
        root.mkdir()
        (root / "notes").mkdir()
        assert client.get("/api/paper/accounts").json() == {"items": []}
        # 账户根的上一级放一个诱饵账户，路径穿越必须找不到它。
        build_account(root.parent / "decoy", DAYS[:1], PRICES, {}).close()
        (root.parent / "manifest.json").write_bytes(
            (root.parent / "decoy" / "manifest.json").read_bytes()
        )
        for name in ("missing", "..", "%2E%2E", "a%2Fb", ".hidden", "..%2Fdecoy"):
            response = client.get(f"/api/paper/accounts/{name}")
            assert response.status_code == 404
            assert str(root) not in response.text


def test_summary_matches_broker_and_is_read_only(root):
    directory = root / "demo"
    broker = build_account(directory, DAYS, PRICES, PLAN, live_last=True)
    account = broker.account()
    state = json.loads(broker.dump())
    expected_positions = {
        broker.symbols[i]: p for i, p in enumerate(account["positions"]) if p["total"]
    }
    archived = sorted((directory / "days").glob("*.json"))
    assert [p.stem for p in archived] == DAYS[:-1]
    previous = json.loads(archived[-1].read_bytes())["account"]
    broker.close()
    before = {p: p.stat().st_mtime_ns for p in directory.rglob("*") if p.is_file()}

    with client_for(root) as client:
        listing = client.get("/api/paper/accounts").json()["items"]
        assert listing == [dict(id="demo", since="2026-10-30", running=False)]
        response = client.get("/api/paper/accounts/demo")
        assert response.status_code == 200
        body = response.json()

    assert body["day"] == "2026-11-04" and body["phase"] == "连续竞价"
    assert body["settled"] is False and body["running"] is False
    assert body["quote_time"] == "2026-11-04 14:59:00"
    assert body["equity"] == round(account["equity"] / 10_000, 2)
    assert body["cash"] == round(account["cash"] / 10_000, 2)
    assert body["frozen"] == round(account["frozen"] / 10_000, 2)
    assert body["day_pnl"] == round(
        (account["equity"] - previous["equity"]) / 10_000, 2
    )
    assert body["day_return"] == pytest.approx(
        account["equity"] / previous["equity"] - 1
    )
    assert body["nav"] == pytest.approx(account["equity"] / money(1_000_000))
    # 月初基为 10-30 的净值；今年以开户为基。
    october = json.loads(archived[0].read_bytes())["account"]["equity"]
    assert body["month_return"] == pytest.approx(account["equity"] / october - 1)
    assert body["year_return"] == pytest.approx(account["equity"] / money(1e6) - 1)
    by_symbol = {p["symbol"]: p for p in body["positions"]}
    assert set(by_symbol) == set(expected_positions)
    for symbol, p in expected_positions.items():
        mark = state["marks"][broker.symbol_ids[symbol]]
        row = by_symbol[symbol]
        assert row["total"] == p["total"] and row["sellable"] == p["sellable"]
        assert row["mark"] == mark / 10_000
        assert row["value"] == round(p["total"] * mark / 10_000, 2)
        assert row["pnl"] == round((p["total"] * mark - p["cost"]) / 10_000, 2)
        assert row["weight"] == pytest.approx(
            p["total"] * mark / account["equity"], abs=1e-6
        )
    # 当天买的 100 股工行 T+1 不可卖。
    assert by_symbol["601398.SH"]["total"] - by_symbol["601398.SH"]["sellable"] == 100
    values = [p["value"] for p in body["positions"]]
    assert values == sorted(values, reverse=True)
    assert body["industries"][0][0] == "银行"
    assert body["industries"][0][1] == pytest.approx(
        account["market_value"] / account["equity"], abs=1e-6
    )
    assert body["concentration"]["cash_weight"] == pytest.approx(
        (account["cash"] + account["frozen"]) / account["equity"]
    )
    trading = body["trading"]
    assert trading["buys"] == 1 and trading["sells"] == 1
    assert trading["fees"] == round(sum(account["day_fees"].values()) / 10_000, 2)
    assert {o["symbol"] for o in body["orders"]} == {"000001.SZ", "601398.SH"}
    assert all(
        o["filled"] == o["qty"] and o["status"] == "filled" for o in body["orders"]
    )
    assert body["fills_day"] == "2026-11-04"
    assert {(f["symbol"], f["side"], f["qty"]) for f in body["fills"]} == {
        ("000001.SZ", "sell", 2000),
        ("601398.SH", "buy", 100),
    }
    assert body["fills"][0]["name"] in ("平安银行", "工商银行")
    after = {p: p.stat().st_mtime_ns for p in directory.rglob("*") if p.is_file()}
    assert after == before


def test_curve_aligns_benchmark_and_fill_history_pages(root):
    build_account(root / "demo", DAYS, PRICES, PLAN, live_last=True).close()
    with client_for(root) as client:
        curve = client.get("/api/paper/accounts/demo/curve").json()
        history = client.get("/api/paper/accounts/demo/fills?limit=3").json()
        rest = client.get("/api/paper/accounts/demo/fills?offset=3&limit=10").json()
    assert curve["dates"] == ["2026-10-30", "2026-11-02", "2026-11-03"]
    assert len(curve["nav"]) == 3 and curve["nav"][0] < 1  # 首日只付了费用
    assert curve["benchmark"]["symbol"] == "000300.SH"
    assert curve["closes"] == [4000.0, 4040.0, 3960.0]
    assert curve["today_close"] is None
    # 历史成交：最新在前，跨日分页不重不漏。
    assert history["total"] == 6
    items = history["items"] + rest["items"]
    assert len(items) == 6
    assert [i["day"] for i in items] == sorted((i["day"] for i in items), reverse=True)
    assert {(i["day"], i["symbol"], i["side"]) for i in items} == {
        ("2026-11-04", "000001.SZ", "sell"),
        ("2026-11-04", "601398.SH", "buy"),
        ("2026-11-03", "600000.SH", "sell"),
        ("2026-11-03", "601398.SH", "buy"),
        ("2026-10-30", "600000.SH", "buy"),
        ("2026-10-30", "000001.SZ", "buy"),
    }


def test_settled_day_falls_back_to_latest_fills_and_transfer_is_not_profit(root):
    plan = {0: [("600000.SH", "buy", 20_000)]}
    broker = build_account(
        root / "flat", DAYS[:3], PRICES, plan, transfers={1: 50_000, 2: 20_000}
    )
    account = broker.account()
    days = sorted((root / "flat" / "days").glob("*.json"))
    previous = json.loads(days[-2].read_bytes())["account"]
    broker.close()
    with client_for(root) as client:
        body = client.get("/api/paper/accounts/flat").json()
    assert body["settled"] is True and body["phase"] == "已收盘"
    assert body["invested"] == 1_070_000
    # 当天入金 2 万不算盈亏。
    assert body["day_pnl"] == round(
        (account["equity"] - previous["equity"] - money(20_000)) / 10_000, 2
    )
    # 只列有持仓的证券（另外两只从未买入）。
    assert [p["symbol"] for p in body["positions"]] == ["600000.SH"]
    # 最近一天没有成交时，成交面板退回最近一个有成交的交易日。
    assert body["fills"] and body["fills_day"] == "2026-10-30"
    # 交易统计与成交面板是同一天（10-30：一笔买入，无卖出）。
    stats = body["trading"]
    assert (stats["buys"], stats["sells"], stats["unfilled"]) == (1, 0, 0)
    first = json.loads(days[0].read_bytes())
    bought = sum(f["qty"] * f["price"] for f in first["fills"])
    assert stats["buy_amount"] == round(bought / 10_000, 2)
    assert stats["turnover"] == pytest.approx(
        bought / 2 / first["account"]["equity"], abs=1e-6
    )
    assert stats["fees"] == round(sum(first["account"]["day_fees"].values()) / 1e4, 2)
    # 入金那天净值只反映价格变动：(E1 − 50000) / E0。
    e = [json.loads(p.read_bytes())["account"]["equity"] for p in days]
    assert body["nav"] == pytest.approx(
        e[0]
        / money(1e6)
        * (e[1] - money(50_000))
        / e[0]
        * (e[2] - money(20_000))
        / e[1]
    )


def test_etag_304_and_refresh_after_new_journal_entries(root):
    directory = root / "demo"
    build_account(directory, DAYS[:2], PRICES, PLAN, live_last=True).close()
    with client_for(root) as client:
        first = client.get("/api/paper/accounts/demo")
        tag = first.headers["etag"]
        assert first.headers["cache-control"] == "private, no-cache"
        again = client.get("/api/paper/accounts/demo", headers={"If-None-Match": tag})
        assert again.status_code == 304 and again.content == b""
        # 运行器继续写日志后，指纹变化，接口重放出新状态。
        manifest = json.loads((directory / "manifest.json").read_bytes())
        from ashare.paper.profiles import from_native

        cash, config = from_native(manifest["config"])
        with Broker(
            manifest["symbols"],
            cash,
            config,
            journal=directory,
            quote_store=ParquetStore(directory / "quotes"),
        ) as broker:
            broker.session(timestamp("2026-11-02 15:00:00"), 20261102, "closed")
            broker.settle(timestamp("2026-11-02 15:02:00"), 20261102)
        changed = client.get("/api/paper/accounts/demo", headers={"If-None-Match": tag})
        assert changed.status_code == 200 and changed.headers["etag"] != tag
        assert changed.json()["settled"] is True
        assert changed.json()["phase"] == "已收盘"


def test_running_flag_follows_status_file_freshness(root):
    directory = root / "demo"
    build_account(directory, DAYS[:1], PRICES, {}, live_last=True).close()
    status = directory / "status.json"
    status.write_text("{}")
    with client_for(root) as client:
        assert client.get("/api/paper/accounts").json()["items"][0]["running"] is True
        assert client.get("/api/paper/accounts/demo").json()["running"] is True
        stale = time.time() - paper_api.RUNNING_WITHIN - 5
        os.utime(status, (stale, stale))
        assert client.get("/api/paper/accounts").json()["items"][0]["running"] is False
        assert client.get("/api/paper/accounts/demo").json()["running"] is False


def test_corrupt_account_gives_friendly_error(root):
    directory = root / "broken"
    build_account(directory, DAYS[:1], PRICES, {}, live_last=True).close()
    for path in directory.glob("state-*.json"):
        path.write_text("not json")
    (directory / "events.jsonl").write_text('{"bad":\n')
    with client_for(root) as client:
        response = client.get("/api/paper/accounts/broken")
    assert response.status_code == 503
    assert response.json() == {"message": "账户暂时读取不到，请稍后再试"}


def test_registered_in_create_app(tmp_path, monkeypatch):
    from ashare.platform.app import create_app

    monkeypatch.setenv("ASHARE_PAPER_DIR", str(tmp_path / "paper"))
    app = create_app(tmp_path / "platform", data_root=str(tmp_path / "nodata"))
    with TestClient(app, base_url="http://127.0.0.1") as client:
        response = client.get("/api/paper/accounts")
    assert response.status_code == 200 and response.json() == {"items": []}


def test_default_root_under_data_dir(monkeypatch, tmp_path):
    monkeypatch.delenv("ASHARE_PAPER_DIR", raising=False)
    monkeypatch.setenv("ASHARE_DATA_DIR", str(tmp_path))
    assert paper_api.paper_root() == Path(tmp_path) / "paper"


def test_open_orders_and_rejections_are_reported(root):
    # 风控状态：挂单未成交与拒单原因按当天事件统计。
    plan = {0: [("600000.SH", "buy", 1000), ("601398.SH", "sell", 100)]}
    broker = build_account(root / "risk", DAYS[:1], PRICES, plan, live_last=True)
    at = timestamp("2026-10-30 14:59:30")
    broker.order(at, "000001.SZ", "buy", 100, 11.0)  # 低于卖一，挂着
    broker.events()
    broker.close()
    with client_for(root) as client:
        body = client.get("/api/paper/accounts/risk").json()
    stats = body["trading"]
    assert stats["open_orders"] == 1 and stats["unfilled"] == 1
    assert body["frozen"] > 0
    assert body["concentration"]["cash_weight"] == pytest.approx(
        (body["cash"] + body["frozen"]) / body["equity"], abs=1e-6
    )
    assert stats["rejected"] == [["可卖不足", 1]]
    assert body["risk"] == dict(open_orders=1, rejected=[["可卖不足", 1]])
    pending = [o for o in body["orders"] if o["status"] == "open"]
    assert [(o["symbol"], o["filled"], o["qty"]) for o in pending] == [
        ("000001.SZ", 0, 100)
    ]
    assert pending[0]["time"] == "14:59:30"
