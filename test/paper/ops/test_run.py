"""运行阶段（2026-11-14 新增，模拟盘每日运行）：非交易日退出 0、已过时段退出 0、没有账户干净跳过、
阶段锁、等首个目标的新账户跳过、按目标成交并结算、dry-run 不写文件、同一天重跑从日志恢复。

行情来自假客户端（固定价格、成交量递增），时钟是假时钟；不联网。
"""

import hashlib
import json
from datetime import time, timedelta

import pytest

from ashare.paper.__main__ import read_state
from ashare.paper.ops import accounts
from ashare.paper.ops.context import Exit, stage_lock, write_json
from ashare.paper.ops.run import parse_until, run

from .helpers import FakeClient, FakeClock, client_factory, context, weekday_calendar

TODAY = "20261116"  # 周一
SYMBOLS = ["600000.SH", "000001.SZ"]
CAL = weekday_calendar("20261101", "20261231")


def setup_account(ctx, name="demo", weights=None, enabled=True):
    accounts.create(ctx, name, source="rank(close)", capital=100, enabled=enabled)
    if weights is not None:
        write_json(
            ctx.paper / name / "ops" / "targets" / f"{TODAY}.json",
            dict(
                version=1,
                at="09:15:00",
                until="14:56:50",
                rebalance_seconds=30,
                protection_ticks=2,
                weights=weights,
                signal_day="20261113",
                execution_day=TODAY,
            ),
        )
        write_json(
            ctx.paper / name / "ops" / "plan.json",
            dict(last_signal_day="20261113", pending_execution_day=TODAY),
        )


def go(ctx, client=None, **options):
    client = client or FakeClient(ctx.clock)
    client.universe = SYMBOLS
    seen = []
    result = run(
        ctx,
        client_factory=client_factory(client, seen),
        lock=ctx.paper / "tushare.lock",
        calendar=CAL,
        **options,
    )
    return result, client, seen


def test_no_accounts_weekend_and_late_start_exit_zero(tmp_path):
    ctx = context(tmp_path, "2026-11-16T08:55:00+08:00", tmp_path / "missing")
    result, client, seen = go(ctx)
    assert result.code == Exit.OK and result.message == "没有启用的账户，跳过"
    assert seen == []  # 不连行情
    setup_account(ctx, weights={SYMBOLS[0]: 0.5})
    ctx.clock.value = ctx.clock.value.replace(day=14)  # 周六
    result, client, seen = go(ctx)
    assert (result.code, result.message) == (Exit.OK, "非交易日") and seen == []
    ctx.clock.value = ctx.clock.value.replace(day=16, hour=17, minute=0)
    result, client, seen = go(ctx)
    assert (result.code, result.message) == (Exit.OK, "已过运行时段") and seen == []
    # 日历没覆盖今天：失败，不猜。
    ctx.clock.value = ctx.clock.value.replace(year=2027, month=1, day=4, hour=8)
    assert go(ctx)[0].code == Exit.FAILED
    assert (
        json.loads((ctx.ops / "state" / "run.json").read_text())["status"] == "failed"
    )


def test_waiting_account_skipped_and_lock(tmp_path):
    ctx = context(tmp_path, "2026-11-16T08:55:00+08:00")
    setup_account(ctx, "new")  # 没有目标、也没有账本
    setup_account(ctx, "off", weights={SYMBOLS[0]: 0.5}, enabled=False)
    result, client, seen = go(ctx)
    assert result.code == Exit.OK and result.details["skipped"] == {
        "new": "等待首个目标"
    }
    assert seen == [] and not (ctx.paper / "new" / "manifest.json").exists()
    with stage_lock(ctx, "run"):
        assert go(ctx)[0].code == Exit.BUSY
    assert go(ctx, accounts=["zzz"])[0].code == Exit.USAGE


def test_trades_to_targets_settles_and_records(tmp_path):
    ctx = context(tmp_path, "2026-11-16T09:20:00+08:00")
    setup_account(ctx, weights={SYMBOLS[0]: 0.4, SYMBOLS[1]: 0.4})
    result, client, seen = go(ctx, until=time(15, 30), poll_seconds=600)
    assert result.code == Exit.OK, result.message
    assert result.details["accounts"] == {"demo": "已结算"}
    assert seen == [ctx.paper / "tushare.lock"] and client.closed
    directory = ctx.paper / "demo"
    assert (directory / "manifest.json").is_file()
    assert (directory / "days" / f"{TODAY}.json").is_file()
    state = read_state(directory)
    positions = {p["symbol"]: p["total"] for p in state["account"]["positions"]}
    # 100 万 × 40% ÷ 10 元 = 4 万股（整手）。
    assert positions == {SYMBOLS[0]: 40_000, SYMBOLS[1]: 40_000}
    assert state["account"]["day"] == int(TODAY)
    marker = json.loads((directory / "ops" / "runs" / f"{TODAY}.json").read_text())
    assert marker["targets"] == f"{TODAY}.json" and marker["settled"] is True
    assert marker["error"] is None and marker["symbols"] == 2
    plan = json.loads((directory / "ops" / "plan.json").read_text())
    assert plan["last_execution_day"] == TODAY and plan["pending_execution_day"] is None
    assert (directory / "latest.json").is_file()
    assert json.loads((ctx.ops / "state" / "run.json").read_text())["status"] == "ok"
    # 当天已结算：重跑不再启动这个账户。
    again, _, seen = go(ctx, until=time(16, 0))
    assert again.code == Exit.OK and again.details["skipped"] == {"demo": "今天已结算"}
    assert seen == []


def journal_events(directory):
    return [
        json.loads(line)["body"]
        for line in (directory / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]


def test_restart_same_day_recovers_from_journal(tmp_path):
    """上午跑到一半进程没了，再启动从日志恢复，不重复建仓。

    2026-11-14 审查：原来只断言「已结算、持有 4 万股」，账本整个丢掉重建也能通过。现在还断言
    第二次运行没有新增买单、现金不变、事件序号接着第一次往下排。
    """
    ctx = context(tmp_path, "2026-11-16T09:20:00+08:00")
    setup_account(ctx, weights={SYMBOLS[0]: 0.4})
    directory = ctx.paper / "demo"
    first, client, _ = go(ctx, until=time(10, 0), poll_seconds=300)
    assert first.details["accounts"] == {"demo": "未结算（次日开盘前补结算）"}
    state = read_state(directory)
    held = {p["symbol"]: p["total"] for p in state["account"]["positions"]}
    assert held[SYMBOLS[0]] == 40_000
    cash = state["account"]["cash"]
    orders = [(o["id"], o["side"], o["qty"]) for o in state["orders"]]
    assert orders == [(1, "buy", 40_000)]
    events = journal_events(directory)
    second, _, _ = go(ctx, client=client, until=time(15, 30), poll_seconds=600)
    assert second.details["accounts"] == {"demo": "已结算"}
    state = read_state(directory)
    assert {p["symbol"]: p["total"] for p in state["account"]["positions"]} == held
    assert state["account"]["cash"] == cash
    after = journal_events(directory)
    assert after[: len(events)] == events  # 第一次的事件原样保留
    assert [e["seq"] for e in after] == list(range(1, len(after) + 1))
    assert len(after) > len(events)  # 第二次确实接着写了（盯市、结算）
    assert [
        (o["id"], o["side"], o["qty"]) for o in state["orders"]
    ] == orders  # 没有新单


def digest_tree(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_dry_run_restores_journal_and_reports_changes(tmp_path):
    """2026-11-14 审查：演练要在已有账本上恢复，给出当天真实的调仓差额；账户目录一个字节都不动。"""
    ctx = context(tmp_path, "2026-11-16T09:20:00+08:00")
    setup_account(ctx, weights={SYMBOLS[0]: 0.4})
    go(ctx, until=time(10, 0), poll_seconds=300)  # 正式运行：买进 4 万股 S0
    path = ctx.paper / "demo" / "ops" / "targets" / f"{TODAY}.json"
    write_json(
        path,
        json.loads(path.read_text()) | dict(weights={SYMBOLS[0]: 0.4, SYMBOLS[1]: 0.4}),
    )
    before = digest_tree(ctx.paper / "demo")
    result, _, _ = go(ctx, until=time(11, 0), poll_seconds=600, dry_run=True)
    assert result.code == Exit.OK, result.message
    outcome = result.details["accounts"]["demo"]
    # 权益约 99.99 万，40% ÷ 10 元按整手向下取整是 39900 股；已有的 S0 不再买。
    assert outcome["holdings"] == {SYMBOLS[0]: 40_000, SYMBOLS[1]: 39_900}
    assert outcome["changes"] == {SYMBOLS[1]: 39_900}
    assert digest_tree(ctx.paper / "demo") == before


def test_execution_day_mismatch_is_rejected(tmp_path):
    """2026-11-14 审查：目标文件只认 execution_day 等于今天的。"""
    ctx = context(tmp_path, "2026-11-16T09:20:00+08:00")
    setup_account(ctx, weights={SYMBOLS[0]: 0.4})
    path = ctx.paper / "demo" / "ops" / "targets" / f"{TODAY}.json"
    write_json(path, json.loads(path.read_text()) | dict(execution_day="20261113"))
    result, _, _ = go(ctx, until=time(10, 0), poll_seconds=600)
    assert result.code == Exit.FAILED
    assert result.details["accounts"]["demo"] == "打开失败：ValueError"
    assert not (ctx.paper / "demo" / "manifest.json").exists()


class SleepyClock(FakeClock):
    """第一次 sleep 时机器合盖睡了一个周末：墙上时间前进 3 天，单调时钟不走（系统挂起）。"""

    def __init__(self, now):
        super().__init__(now)
        self.slept = False

    def sleep(self, seconds):
        if not self.slept:
            self.slept = True
            self.value += timedelta(days=3)
        super().sleep(seconds)


def test_deadline_is_wall_clock_after_system_sleep(tmp_path):
    """2026-11-14 审查：周五启动的运行器睡过周末醒来，按单调时钟还没到截止，会拿周五的目标在周一
    交易。截止改按墙上时间：醒来已过周五 15:30 就收尾，不在周一下单。"""
    ctx = context(tmp_path, "2026-11-13T09:20:00+08:00")  # 周五
    ctx.clock = SleepyClock("2026-11-13T09:20:00+08:00")
    accounts.create(ctx, "demo", source="rank(close)", capital=100)
    for day, weights in (
        ("20261113", {SYMBOLS[0]: 0.4}),
        ("20261116", {SYMBOLS[1]: 0.4}),
    ):
        write_json(
            ctx.paper / "demo" / "ops" / "targets" / f"{day}.json",
            dict(version=1, at="09:15:00", weights=weights, execution_day=day),
        )
    result, client, _ = go(ctx, until=time(15, 30), poll_seconds=600)
    state = read_state(ctx.paper / "demo")
    assert state["account"]["day"] == 20261113
    assert result.details["accounts"] == {"demo": "未结算（次日开盘前补结算）"}
    assert all(p["total"] == 0 for p in state["account"]["positions"])
    assert client.calls.count("rt_k") <= 1


def test_dry_run_writes_nothing(tmp_path):
    ctx = context(tmp_path, "2026-11-16T09:20:00+08:00")
    setup_account(ctx, weights={SYMBOLS[0]: 0.4})
    before = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    result, client, seen = go(ctx, until=time(10, 0), poll_seconds=600, dry_run=True)
    assert result.code == Exit.OK and "rt_k" in client.calls
    after = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    # 只多了阶段锁文件（和它的目录），没有账本、日报、运行记录或状态文件。
    added = [str(p) for p in after if p not in before]
    assert added == ["paper/.ops/locks", "paper/.ops/locks/run.lock"]
    outcome = result.details["accounts"]["demo"]
    assert outcome["changes"] == {SYMBOLS[0]: 40_000}  # 新账户：从现金建仓


def test_parse_until():
    assert parse_until("17:00") == time(17)
    for bad in ("08:30", "25:00", "soon"):
        with pytest.raises(ValueError):
            parse_until(bad)
