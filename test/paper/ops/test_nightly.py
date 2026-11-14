"""nightly 与 status（2026-11-14 审查）：开盘前最后一次 nightly 仍没出成目标时，日志记 ERROR、写标记，
``daily status`` 列出「今天没有目标」；之后成功就清掉。阶段函数用替身，不联网。
"""

import json
import logging

from ashare.paper.ops import accounts
from ashare.paper.ops.context import Exit, StageResult
from ashare.paper.ops.nightly import FLAG, nightly, status

from .helpers import context, weekday_calendar

CAL = weekday_calendar("20261101", "20261231")


def stages(after_code=Exit.OK, signal_code=Exit.OK):
    calls = []

    def after_close(ctx, accept_verify=False):
        calls.append("after-close")
        return StageResult("after-close", after_code, "当天数据还没发布齐：daily")

    def signal(ctx):
        calls.append("signal")
        return StageResult("signal", signal_code, "以下账户的选股结果不可信：demo")

    return calls, dict(after_close_fn=after_close, signal_fn=signal, calendar=CAL)


def test_last_morning_attempt_not_ready_leaves_flag(tmp_path, caplog):
    ctx = context(tmp_path, "2026-11-16T07:30:00+08:00")  # 周一早上
    accounts.create(ctx, "demo", source="rank(close)")
    calls, fns = stages(after_code=Exit.NOT_READY)
    with caplog.at_level(logging.ERROR, logger="ashare.paper.ops"):
        results = nightly(ctx, **fns)
    assert calls == ["after-close"] and results[-1].code == Exit.NOT_READY
    assert any("no_targets_today" in r.getMessage() for r in caplog.records)
    flag = json.loads((ctx.ops / "state" / FLAG).read_text())
    assert flag["day"] == "20261116" and flag["stage"] == "after-close"
    assert status(ctx)["alerts"] == [
        "今天没有目标：after-close 阶段 not_ready，当天数据还没发布齐：daily"
    ]
    assert FLAG.removesuffix(".json") not in status(ctx)["stages"]
    # 人工修好后补跑成功：标记清掉。
    calls, fns = stages()
    assert [r.code for r in nightly(ctx, **fns)] == [Exit.OK, Exit.OK]
    assert calls == ["after-close", "signal"]
    assert not (ctx.ops / "state" / FLAG).exists() and status(ctx)["alerts"] == []


def test_signal_not_ready_in_morning_also_flags(tmp_path):
    ctx = context(tmp_path, "2026-11-16T07:30:00+08:00")
    accounts.create(ctx, "demo", source="rank(close)")
    _, fns = stages(signal_code=Exit.NOT_READY)
    nightly(ctx, **fns)
    assert json.loads((ctx.ops / "state" / FLAG).read_text())["stage"] == "signal"


def test_no_flag_in_evening_on_weekend_or_without_accounts(tmp_path):
    ctx = context(
        tmp_path, "2026-11-16T18:30:00+08:00"
    )  # 晚上还有 20:30 和 07:30 两次机会
    accounts.create(ctx, "demo", source="rank(close)")
    _, fns = stages(after_code=Exit.NOT_READY)
    nightly(ctx, **fns)
    assert not (ctx.ops / "state" / FLAG).exists()
    ctx.clock.value = ctx.clock.value.replace(day=14, hour=7)  # 周六早上：不开市
    nightly(ctx, **fns)
    assert not (ctx.ops / "state" / FLAG).exists()
    ctx.clock.value = ctx.clock.value.replace(day=16)
    accounts.set_enabled(ctx, "demo", False)  # 没有启用的账户：不用目标
    nightly(ctx, **fns)
    assert not (ctx.ops / "state" / FLAG).exists()


def test_stale_flag_from_another_day_is_not_an_alert(tmp_path):
    ctx = context(tmp_path, "2026-11-16T07:30:00+08:00")
    accounts.create(ctx, "demo", source="rank(close)")
    _, fns = stages(after_code=Exit.FAILED)
    nightly(ctx, **fns)
    assert status(ctx)["alerts"]
    ctx.clock.value = ctx.clock.value.replace(day=17)
    assert status(ctx)["alerts"] == []
