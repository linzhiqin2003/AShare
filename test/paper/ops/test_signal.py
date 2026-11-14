"""信号阶段（2026-11-14 新增，模拟盘每日运行）：调仓日判定、目标文件与直接调用选股函数逐位一致、
幂等、补发、冻结已开始的成交日、没有启用账户时干净跳过。数据是只读的合成数据目录。
"""

import json

import numpy as np
import pytest

from ashare.paper.ops import accounts
from ashare.paper.ops import signal as signal_mod
from ashare.paper.ops.context import DATA_LOCK, Exit, stage_lock, write_json
from ashare.paper.ops.signal import signal
from ashare.paper.ops.strategy import (
    LivePanel,
    parse_source,
    pool_fields,
    pool_row,
    runner_safe,
    signal_alpha,
    target_weights,
)
from ashare.paper.runner import load_targets
from test.fixtures.synthetic_datadir import cached_datadir

from .helpers import calendar, context

SOURCE = "rank(-ts_std(close, 20))"


@pytest.fixture(scope="module")
def info():
    return cached_datadir()


@pytest.fixture
def setup(tmp_path, info):
    day = info.dates[-1]  # 20260130（周五）
    ctx = context(tmp_path, f"{day[:4]}-{day[4:6]}-{day[6:]}T18:40:00+08:00", info.root)
    # 合成日历在最后一天之后没有开市日：补一个下周一，月末判断才有下一开市日。
    cal = calendar([*info.dates, "20260202"], "20261231")
    panel = LivePanel(info.root)
    write_json(
        ctx.ops / "state" / "after-close-done.json",
        dict(day=day, panel_version=panel.version),
    )
    return ctx, cal, day


def targets(ctx, name, day="20260202"):
    return json.loads(
        (ctx.paper / name / "ops" / "targets" / f"{day}.json").read_text()
    )


def test_no_enabled_accounts_skips_cleanly(tmp_path):
    """用户还没开账户：不读日历、不要求盘后结果，退出 0（2026-11-14 用户要求）。"""
    ctx = context(tmp_path, "2026-11-13T18:40:00+08:00", tmp_path / "missing")
    result = signal(ctx)
    assert result.code == Exit.OK and result.details == {"skipped": True}
    accounts.create(ctx, "off", source=SOURCE, enabled=False)
    assert signal(ctx).code == Exit.OK
    assert not (ctx.paper / "off" / "ops" / "targets").exists()
    state = json.loads((ctx.ops / "state" / "signal.json").read_text())
    assert state["status"] == "ok" and state["message"] == "没有启用的账户，跳过"


def test_first_signal_matches_direct_selection_parity(setup, info):
    ctx, cal, day = setup
    accounts.create(
        ctx, "demo", source=SOURCE, settings={"universe": "ALL", "top_n": 11}
    )
    result = signal(ctx, calendar=cal)
    assert result.code == Exit.OK, result.message
    assert result.details == dict(
        execution_day="20260202",
        accounts={"demo": "已生成 11 只目标（建仓；周期末）"},
    )
    written = targets(ctx, "demo")
    # 直接调用同一套函数算一遍。
    panel = LivePanel(info.root)
    panel.ensure(pool_fields("ALL"))
    alpha = signal_alpha(panel, parse_source(SOURCE)[0].expression)
    row = len(panel.dates) - 1
    raw = target_weights(alpha[row], pool_row(panel, "ALL", row), 11, panel.symbols)
    expected, moved = runner_safe(raw)
    assert moved > 0  # 11 只时 1/11 的十进制和超过 1，挪过末位
    assert written["weights"] == expected and written["nudged"] == moved
    assert list(written["weights"]) == list(raw)
    assert {k: v for k, v in written.items() if k != "weights"} == dict(
        version=1,
        at="09:15:00",
        until="14:56:50",
        rebalance_seconds=30,
        protection_ticks=2,
        signal_day=day,
        execution_day="20260202",
        panel_version=panel.version,
        strategy=written["strategy"],
        reason="建仓；周期末",
        nudged=moved,
    )
    path = ctx.paper / "demo" / "ops" / "targets" / "20260202.json"
    assert load_targets(path)["weights"] == expected  # 运行器认这个文件
    plan = json.loads((ctx.paper / "demo" / "ops" / "plan.json").read_text())
    assert (
        plan["last_signal_day"] == day and plan["pending_execution_day"] == "20260202"
    )
    assert (
        plan["next_signal_day"] == "20260202"
    )  # 日历里只剩这一天，它是二月最后一个开市日
    # 重跑：已是最新，文件不动，信号记录不重复。
    before = path.read_bytes()
    again = signal(ctx, calendar=cal)
    assert again.details["accounts"] == {"demo": "已是最新"}
    assert path.read_bytes() == before
    lines = (ctx.paper / "demo" / "ops" / "signals.jsonl").read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["count"] == 11


def test_non_rebalance_day_writes_no_targets(setup, info):
    ctx, cal, day = setup
    accounts.create(ctx, "yearly", source=SOURCE, settings={"rebalance": "year"})
    previous = info.dates[-2]
    write_json(
        ctx.paper / "yearly" / "ops" / "plan.json",
        dict(last_signal_day="20251231", last_checked_day=previous),
    )
    result = signal(ctx, calendar=cal)
    assert result.details["accounts"] == {"yearly": "非调仓日"}
    assert not (ctx.paper / "yearly" / "ops" / "targets").exists()
    plan = json.loads((ctx.paper / "yearly" / "ops" / "plan.json").read_text())
    assert plan["last_checked_day"] == day and plan["last_signal_day"] == "20251231"


def test_missed_period_end_is_caught_up(setup, info):
    """机器停了几天、漏过了周期末：用最新数据补一次调仓，并在理由里写明漏的是哪天。"""
    ctx, cal, day = setup
    accounts.create(ctx, "weekly", source=SOURCE, settings={"rebalance": "month"})
    december = [d for d in info.dates if d.startswith("202512")]
    write_json(
        ctx.paper / "weekly" / "ops" / "plan.json",
        dict(last_signal_day=december[0], last_checked_day=december[-2]),
    )
    result = signal(ctx, calendar=cal)
    assert result.code == Exit.OK
    reason = targets(ctx, "weekly")["reason"]
    assert reason == f"周期末；补发：{december[-1]}"


def test_refresh_recomputes_until_execution_starts(setup):
    ctx, cal, day = setup
    accounts.create(ctx, "demo", source=SOURCE)
    assert signal(ctx, calendar=cal).code == Exit.OK
    path = ctx.paper / "demo" / "ops" / "targets" / "20260202.json"
    stale = targets(ctx, "demo") | dict(panel_version="old", weights={"X": 1.0})
    write_json(path, stale)
    # 面板版本变了、成交日还没开始：重算覆盖。
    assert "已生成" in signal(ctx, calendar=cal).details["accounts"]["demo"]
    assert targets(ctx, "demo")["weights"] != {"X": 1.0}
    # 成交日已开始运行：不再改目标。
    write_json(path, stale)
    write_json(
        ctx.paper / "demo" / "ops" / "runs" / "20260202.json", dict(day="20260202")
    )
    result = signal(ctx, calendar=cal)
    assert result.details["accounts"]["demo"] == "成交日已开始运行，目标不再改"
    assert targets(ctx, "demo")["weights"] == {"X": 1.0}


def test_not_ready_unknown_account_and_lock(setup):
    ctx, cal, day = setup
    accounts.create(ctx, "demo", source=SOURCE)
    assert signal(ctx, ["nope"], calendar=cal).code == Exit.USAGE
    with stage_lock(ctx, DATA_LOCK):  # 与盘后阶段同一把锁
        assert signal(ctx, calendar=cal).code == Exit.BUSY
    write_json(
        ctx.ops / "state" / "after-close-done.json",
        dict(day="20260127", panel_version="x"),
    )
    result = signal(ctx, calendar=cal)
    assert (
        result.code == Exit.NOT_READY
        and not (ctx.paper / "demo" / "ops" / "targets").exists()
    )
    # 盘后结果是今天、但面板版本对不上（盘后之后面板又被改过）。
    write_json(
        ctx.ops / "state" / "after-close-done.json", dict(day=day, panel_version="x")
    )
    assert signal(ctx, calendar=cal).code == Exit.NOT_READY
    # 日历里没有下一开市日：失败，不猜成交日。
    write_json(
        ctx.ops / "state" / "after-close-done.json",
        dict(day=day, panel_version=LivePanel(ctx.data).version),
    )
    short = calendar(list(np.asarray(LivePanel(ctx.data).dates)), "20261231")
    assert signal(ctx, calendar=short).code == Exit.FAILED


def test_one_bad_account_does_not_block_others(setup):
    ctx, cal, day = setup
    accounts.create(ctx, "good", source=SOURCE)
    accounts.create(ctx, "bad", source=SOURCE)
    raw = json.loads((ctx.paper / "bad" / "strategy.json").read_text())
    (ctx.paper / "bad" / "strategy.json").write_text(json.dumps(raw | {"version": 9}))
    result = signal(ctx, calendar=cal)
    assert result.code == Exit.FAILED and "bad" in result.message
    assert result.details["accounts"]["good"].startswith("已生成")
    assert targets(ctx, "good")["weights"]


# ---------------------------------------------------------------- 2026-11-14 审查


def test_unexecuted_target_is_reissued(setup, info):
    """建仓目标的成交日没有运行（机器那天关着）：下一次出信号时补发，理由写明哪天没执行。"""
    ctx, cal, day = setup
    prev = info.dates[-2]
    accounts.create(ctx, "q", source=SOURCE, settings={"rebalance": "quarter"})
    write_json(
        ctx.paper / "q" / "ops" / "targets" / f"{day}.json",
        dict(version=1, at="09:15:00", weights={"X": 0.5}, signal_day=prev),
    )
    write_json(
        ctx.paper / "q" / "ops" / "plan.json",
        dict(last_signal_day=prev, last_checked_day=prev, pending_execution_day=day),
    )
    result = signal(ctx, calendar=cal)
    assert result.code == Exit.OK
    assert result.details["accounts"]["q"].startswith("已生成")
    assert targets(ctx, "q")["reason"] == f"补发：{day} 的调仓没有执行"


@pytest.mark.parametrize(
    "marker,reissued",
    [
        (dict(targets=None), True),  # 运行器开跑时还没有目标
        (dict(targets="TARGET", error="RuntimeError"), True),  # 运行出错
        (dict(targets="TARGET", error=None), False),  # 正常执行
    ],
)
def test_reissue_depends_on_run_record(setup, info, marker, reissued):
    ctx, cal, day = setup
    prev = info.dates[-2]
    accounts.create(ctx, "q", source=SOURCE, settings={"rebalance": "quarter"})
    write_json(
        ctx.paper / "q" / "ops" / "plan.json",
        dict(last_signal_day=prev, last_checked_day=prev, pending_execution_day=day),
    )
    if marker.get("targets") == "TARGET":
        marker = marker | dict(targets=f"{day}.json")
    write_json(ctx.paper / "q" / "ops" / "runs" / f"{day}.json", dict(day=day) | marker)
    result = signal(ctx, calendar=cal)
    assert (result.details["accounts"]["q"] != "非调仓日") == reissued


def test_no_target_for_execution_day_that_already_started_then_reissued(tmp_path, info):
    """隔夜信号没出成，运行器 08:55 无目标开跑；10:00 补跑 signal 不再给这一天写目标（运行器看不到），
    当晚的信号把这次调仓补到下一个交易日。"""
    ctx = context(tmp_path, "2026-02-02T10:00:00+08:00", info.root)
    cal = calendar([*info.dates, "20260202", "20260203"], "20261231")
    write_json(
        ctx.ops / "state" / "after-close-done.json",
        dict(day=info.dates[-1], panel_version=LivePanel(info.root).version),
    )
    accounts.create(ctx, "m", source=SOURCE)
    write_json(
        ctx.paper / "m" / "ops" / "runs" / "20260202.json",
        dict(day="20260202", started_at="2026-02-02T08:55:00+08:00", targets=None),
    )
    result = signal(ctx, calendar=cal)
    assert result.details["accounts"]["m"] == "成交日已开始运行，目标不再改"
    assert not (ctx.paper / "m" / "ops" / "targets").exists()
    plan = json.loads((ctx.paper / "m" / "ops" / "plan.json").read_text())
    assert plan["pending_execution_day"] == "20260202"
    # 当晚 20260202 收盘后（合成面板仍停在 20260130，这里只验证补发判断）。
    undone = signal_mod.unexecuted(ctx.paper / "m" / "ops", plan, "20260203")
    assert undone == "20260202"


def test_empty_selection_writes_no_target(setup, monkeypatch):
    """最新一行信号全是 NaN（字段当天缺）：不写目标（空目标会让运行器次日清仓），返回 NOT_READY。"""
    ctx, cal, day = setup
    accounts.create(ctx, "e", source=SOURCE)
    real = signal_mod.signal_alpha

    def nan_last(panel, expression):
        a = np.array(real(panel, expression))
        a[-1] = np.nan
        return a

    monkeypatch.setattr(signal_mod, "signal_alpha", nan_last)
    result = signal(ctx, calendar=cal)
    assert result.code == Exit.NOT_READY
    assert result.details["accounts"]["e"].startswith("未就绪：有效信号从")
    assert not (ctx.paper / "e" / "ops" / "targets").exists()
    assert not (ctx.paper / "e" / "ops" / "signals.jsonl").exists()


def test_partial_signal_drop_is_not_ready(setup, monkeypatch):
    """最新一行只剩一半股票有信号：低于 COVERAGE_FLOOR，不写目标。"""
    ctx, cal, day = setup
    accounts.create(ctx, "h", source=SOURCE)
    real = signal_mod.signal_alpha

    def half_last(panel, expression):
        a = np.array(real(panel, expression))
        a[-1, ::2] = np.nan
        return a

    monkeypatch.setattr(signal_mod, "signal_alpha", half_last)
    assert signal(ctx, calendar=cal).code == Exit.NOT_READY
    assert not (ctx.paper / "h" / "ops" / "targets").exists()


def test_check_selection_thresholds(monkeypatch):
    """阈值：有效信号 ≥ 上一行的 80%；入选 ≥ min(top_n, 上一行有效数) 的一半；两天都选不出是失败。"""
    pool = np.ones(10, dtype=bool)
    monkeypatch.setattr(signal_mod, "pool_row", lambda panel, universe, row: pool)
    check = signal_mod.check_selection
    alpha = np.array([[1.0] * 10, [1.0] * 8 + [np.nan] * 2])
    check(None, "ALL", alpha, 1, pool, 5, 3)  # 8/10 有效、入选 3 ≥ 2.5
    with pytest.raises(signal_mod.NotReady):
        check(None, "ALL", alpha, 1, pool, 5, 2)
    drop = np.array([[1.0] * 10, [1.0] * 7 + [np.nan] * 3])
    with pytest.raises(signal_mod.NotReady):
        check(None, "ALL", drop, 1, pool, 5, 5)
    with pytest.raises(ValueError, match="都选不出"):
        check(None, "ALL", np.full((2, 10), np.nan), 1, pool, 5, 0)
