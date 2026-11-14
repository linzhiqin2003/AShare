"""账户与策略绑定（2026-11-14 新增，模拟盘每日运行）：建账户、回测记录约定、启用停用。

账户全部建在临时目录里。
"""

import json
import sqlite3

import pytest

from ashare.paper.ops import accounts
from ashare.paper.ops.strategy import load
from ashare.platform.backtest.jobs import SCHEMA

from .helpers import context

KEY = "0123456789abcdef01234567"
NOW = "2026-11-14T10:00:00+08:00"


def record(**extra):
    return (
        dict(
            id=KEY,
            name="低波动 30",
            source="rank(-ts_std(close, 20))",
            settings={"universe": "csi300", "rebalance": "month", "top_n": 30},
            status="complete",
            data_version="v1",
        )
        | extra
    )


def test_create_writes_strategy_and_rejects_bad_input(tmp_path):
    ctx = context(tmp_path, NOW)
    strategy = accounts.create(
        ctx, "演示-1", source="rank(close)", settings={"top_n": 5}, capital=200
    )
    raw = json.loads((ctx.paper / "演示-1" / "strategy.json").read_text())
    assert raw["version"] == 1 and raw["enabled"] is True
    assert raw["initial_cash"] == 2_000_000.0 and raw["settings"]["capital"] == 200.0
    assert raw["origin"] == {"kind": "expression"} and raw["created_at"] == NOW
    assert load(ctx.paper / "演示-1") == strategy
    assert (ctx.paper / "演示-1" / "ops").is_dir()
    # 不会出现账本：manifest 由运行器在第一个运行日建立。
    assert not (ctx.paper / "演示-1" / "manifest.json").exists()

    cases = [
        (dict(name="../x"), 422),
        (dict(name=".ops"), 422),
        (dict(name="a" * 65), 422),
        (dict(name="演示-1"), 409),
        (dict(source="rank(close"), 422),
        (dict(source="rank(ts_mean(close, [5, 10]))"), 422),
        (dict(settings={"window": "test"}), 422),
        (dict(settings={"top_n": 0}), 422),
        (dict(capital=1), 422),
    ]
    for change, status in cases:
        options = dict(name="新账户", source="rank(close)", settings={}) | change
        with pytest.raises(accounts.AccountError) as caught:
            accounts.create(
                ctx,
                options.pop("name"),
                source=options["source"],
                settings=options["settings"],
                capital=options.get("capital"),
            )
        assert caught.value.status == status, change
    # 被拒绝的请求一个目录都不留下（.ops 是建账户用的暂存区，里面也是空的）。
    assert sorted(p.name for p in ctx.paper.iterdir()) == [".ops", "演示-1"]
    assert list((ctx.ops / "tmp").iterdir()) == []


def test_from_backtest_record_contract(tmp_path):
    ctx = context(tmp_path, NOW)
    strategy = accounts.from_backtest(ctx, "lowvol", record(), capital=500)
    assert strategy.name == "低波动 30"
    assert strategy.origin == dict(kind="backtest", id=KEY, data_version="v1")
    assert strategy.settings["top_n"] == 30 and strategy.initial_cash == 5_000_000
    # settings 可以是回测库里存的 JSON 文本。
    text = record(settings=json.dumps(record()["settings"]))
    assert accounts.from_backtest(ctx, "lowvol2", text).settings["top_n"] == 30
    for bad in (record(id="xyz"), record(status="failed"), "not a dict"):
        with pytest.raises(accounts.AccountError):
            accounts.from_backtest(ctx, "bad", bad)


def test_backtest_records_read_only_from_result_store(tmp_path):
    """回测页结果库（同一建表语句）里只读取出已完成的回测；库不存在时当作没有。"""
    empty = accounts.BacktestRecords(tmp_path / "none")
    assert empty.get(KEY) is None and empty.completed() == []
    folder = tmp_path / "platform" / "backtests"
    folder.mkdir(parents=True)
    with sqlite3.connect(folder / "index.sqlite3") as db:
        db.execute(SCHEMA)
        for i, status in enumerate(("complete", "failed", "complete")):
            db.execute(
                "INSERT INTO backtests (id, name, source, settings, status, created, finished, data_version)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    f"{i:024x}",
                    f"回测{i}",
                    "rank(close)",
                    json.dumps({"top_n": 10 + i}),
                    status,
                    float(i),
                    float(i),
                    "v",
                ),
            )
    before = (folder / "index.sqlite3").read_bytes()
    store = accounts.BacktestRecords(tmp_path / "platform")
    assert [r["name"] for r in store.completed()] == ["回测2", "回测0"]
    got = store.get(f"{0:024x}")
    assert got["source"] == "rank(close)" and got["settings"] == {"top_n": 10}
    assert store.get("f" * 24) is None
    assert (folder / "index.sqlite3").read_bytes() == before


def test_enable_disable_and_active_selection(tmp_path):
    ctx = context(tmp_path, NOW)
    assert accounts.active(ctx) == ([], [])
    accounts.create(ctx, "a", source="rank(close)")
    accounts.create(ctx, "b", source="rank(close)", enabled=False)
    (ctx.paper / "manual").mkdir()  # 手工 start 的账户（没有策略定义）不参与每日流水线
    (ctx.paper / "manual" / "manifest.json").write_text("{}")
    names = lambda pairs: [n for n, _ in pairs]  # noqa: E731
    assert names(accounts.active(ctx)[0]) == ["a"]
    assert accounts.active(ctx, ["b"]) == ([], [])
    assert accounts.active(ctx, ["zzz"])[1] == ["zzz"]
    accounts.set_enabled(ctx, "b", True)
    accounts.set_enabled(ctx, "a", False)
    assert names(accounts.active(ctx)[0]) == ["b"]
    with pytest.raises(accounts.AccountError):
        accounts.set_enabled(ctx, "manual", True)
    # 策略定义坏了：不当成停用静默跳过，交给阶段报错。
    (ctx.paper / "a" / "strategy.json").write_text("{")
    assert names(accounts.active(ctx)[0]) == ["a", "b"]
    assert accounts.describe(ctx.paper / "a") == dict(strategy=None, plan=None)
    described = accounts.describe(ctx.paper / "b")
    assert described["strategy"]["enabled"] is True
    assert described["plan"]["next_signal_day"] is None


def test_create_is_atomic(tmp_path, monkeypatch):
    """2026-11-14 审查：建账户中途失败不能留下没有策略定义的半成品目录。"""
    ctx = context(tmp_path, NOW)

    def broken(path, value):
        raise OSError("disk full")

    monkeypatch.setattr(accounts, "write_json", broken)
    with pytest.raises(OSError):
        accounts.create(ctx, "half", source="rank(close)")
    assert not (ctx.paper / "half").exists()
    assert list((ctx.ops / "tmp").iterdir()) == []
    monkeypatch.undo()
    accounts.create(ctx, "half", source="rank(close)")
    assert (ctx.paper / "half" / "strategy.json").is_file()
    assert [n for n, _ in accounts.bound(ctx)] == ["half"]
    # 改名前一刻被别人建了同名（非空）目录：409，不覆盖。
    (ctx.paper / "race").mkdir()
    (ctx.paper / "race" / "keep").write_text("x")
    with pytest.raises(accounts.AccountError) as error:
        accounts.create(ctx, "race", source="rank(close)")
    assert error.value.status == 409 and (ctx.paper / "race" / "keep").exists()
