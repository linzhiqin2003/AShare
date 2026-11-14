"""模拟盘页新建账户（2026-11-14 新增，模拟盘每日运行）：参数白名单、同源校验、回测记录、列表里的策略与计划。

账户全部建在临时目录；回测结果库用回测页同一建表语句在临时目录里合成。
"""

import json
import sqlite3

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ashare.paper.ops.context import write_json
from ashare.platform import paper_api
from ashare.platform.backtest.jobs import SCHEMA

KEY = "0123456789abcdef01234567"
ORIGIN = {"Origin": "http://127.0.0.1"}


class FakeBacktests:
    def __init__(self, records=()):
        self.records = {r["id"]: r for r in records}

    def get(self, key):
        return self.records.get(key)

    def completed(self):
        return [r for r in self.records.values() if r["status"] == "complete"]


def record(**extra):
    return (
        dict(
            id=KEY,
            name="低波动 30",
            source="rank(-ts_std(close, 20))",
            settings={
                "universe": "csi300",
                "rebalance": "month",
                "top_n": 30,
                "capital": 2000,
            },
            status="complete",
            data_version="v1",
        )
        | extra
    )


def client_for(root, backtests=None):
    app = FastAPI()
    app.include_router(
        paper_api.router(
            root=root, reference=object(), backtests=backtests or FakeBacktests()
        )
    )
    return TestClient(app, base_url="http://127.0.0.1")


def test_create_from_backtest_and_list_shows_strategy(tmp_path):
    root = tmp_path / "paper"
    with client_for(
        root, FakeBacktests([record(), record(id="f" * 24, status="failed")])
    ) as c:
        listed = c.get("/api/paper/backtests").json()["items"]
        assert listed == [
            dict(
                id=KEY,
                name="低波动 30",
                universe="csi300",
                rebalance="month",
                top_n=30,
                capital=2000,
            )
        ]
        response = c.post(
            "/api/paper/accounts", json=dict(name="低波动30", backtest=KEY, capital=500)
        )
        assert response.status_code == 201, response.text
        item = response.json()
        assert (
            item["id"] == "低波动30"
            and item["started"] is False
            and item["running"] is False
        )
        assert item["strategy"] == dict(
            name="低波动 30",
            universe="csi300",
            rebalance="month",
            top_n=30,
            trade_at="open",
            initial_cash=5_000_000.0,
            enabled=True,
            origin="backtest",
        )
        assert item["plan"] == dict(
            last_signal_day=None,
            last_execution_day=None,
            next_signal_day=None,
            pending_execution_day=None,
        )
        raw = json.loads((root / "低波动30" / "strategy.json").read_text())
        assert raw["source"] == record()["source"] and raw["origin"]["id"] == KEY
        # 计划里的日子按页面格式给出。
        write_json(
            root / "低波动30" / "ops" / "plan.json",
            dict(last_execution_day="20261102", next_signal_day="20261130"),
        )
        listed = c.get("/api/paper/accounts").json()["items"]
        assert [i["id"] for i in listed] == ["低波动30"]
        assert listed[0]["plan"]["last_execution_day"] == "2026-11-02"
        assert listed[0]["plan"]["next_signal_day"] == "2026-11-30"
        # 还没开始运行的账户没有账本：摘要接口照旧 404，不重放。
        assert c.get("/api/paper/accounts/低波动30").status_code == 404


def test_create_parameter_whitelist(tmp_path):
    root = tmp_path / "paper"
    with client_for(
        root, FakeBacktests([record(), record(id="e" * 24, status="failed")])
    ) as c:
        bad = [
            (dict(name="a", backtest=KEY, extra=1), 422),  # 不认识的参数
            (dict(name="a"), 422),  # 两样都没给
            (dict(name="a", backtest=KEY, source="rank(close)"), 422),  # 两样都给
            (dict(name="a", backtest="XYZ"), 422),
            (dict(name="a", backtest="f" * 24), 404),
            (dict(name="a", backtest="e" * 24), 422),  # 没完成的回测
            (dict(name="a", backtest=KEY, settings={"top_n": 5}), 422),
            (dict(name="../a", backtest=KEY), 422),
            (dict(name="a", backtest=KEY, capital=5), 422),
            (dict(name="a", source="rank(close)", settings={"window": "test"}), 422),
            (dict(name="a", source="x" * 16001), 422),
            (dict(name="a", source="rank(close"), 422),
        ]
        for body, status in bad:
            response = c.post("/api/paper/accounts", json=body)
            assert response.status_code == status, (body, response.text)
            if status != 422 or "detail" not in response.json():
                assert set(response.json()) == {"message"}
                assert str(tmp_path) not in response.text
        assert not root.exists() or list(root.iterdir()) == []
        ok = c.post(
            "/api/paper/accounts",
            json=dict(
                name="expr", source="rank(close)", settings={"top_n": 5}, title="表达式"
            ),
        )
        assert ok.status_code == 201 and ok.json()["strategy"]["name"] == "表达式"
        again = c.post(
            "/api/paper/accounts", json=dict(name="expr", source="rank(close)")
        )
        assert again.status_code == 409 and again.json() == {"message": "已有同名账户"}


def test_create_app_requires_same_origin_and_reads_platform_backtests(
    tmp_path, monkeypatch
):
    from ashare.platform.app import create_app

    monkeypatch.setenv("ASHARE_PAPER_DIR", str(tmp_path / "paper"))
    platform = tmp_path / "platform"
    (platform / "backtests").mkdir(parents=True)
    with sqlite3.connect(platform / "backtests" / "index.sqlite3") as db:
        db.execute(SCHEMA)
        db.execute(
            "INSERT INTO backtests (id, name, source, settings, status, created, finished)"
            " VALUES (?, ?, ?, ?, 'complete', 1, 1)",
            (KEY, "回测", "rank(close)", json.dumps({"top_n": 8})),
        )
    app = create_app(platform, data_root=str(tmp_path / "nodata"))
    body = dict(name="bt", backtest=KEY)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        assert client.post("/api/paper/accounts", json=body).status_code == 403
        assert (
            client.post(
                "/api/paper/accounts",
                json=body,
                headers={"Origin": "http://evil.example"},
            ).status_code
            == 403
        )
        assert not (tmp_path / "paper" / "bt").exists()
        assert client.get("/api/paper/backtests").json()["items"][0]["id"] == KEY
        response = client.post("/api/paper/accounts", json=body, headers=ORIGIN)
        assert response.status_code == 201, response.text
        assert response.json()["strategy"]["top_n"] == 8
        bad = client.post("/api/paper/accounts", json=body | {"x": 1}, headers=ORIGIN)
        assert bad.status_code == 422 and bad.json() == {"message": "请求参数不正确"}


@pytest.mark.parametrize("payload", [b"{", b"[]"])
def test_malformed_body(tmp_path, payload):
    with client_for(tmp_path / "paper") as c:
        response = c.post(
            "/api/paper/accounts",
            content=payload,
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422
