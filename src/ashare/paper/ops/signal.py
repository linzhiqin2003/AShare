"""信号阶段：对每个启用的账户，在最新面板上算出下一个开市日要达到的目标持仓。

前提：盘后阶段已通过，且面板最后一天就是它处理的交易日 E（否则 NOT_READY）。

每个账户：

- 判断 E 是不是调仓信号日：账户第一次出信号（建仓）、E 是调仓周期的最后一个交易日、或者上次
  检查以来漏过了某个周期末（机器停了几天）——漏过的就用 E 的数据补一次调仓，并在记录里注明；
- 是信号日：算目标，写 ``ops/targets/<成交日>.json``（运行器 ``--targets`` 认的格式，额外记信号日、
  面板版本、策略摘要），追加一行 ``ops/signals.jsonl``；
- 不是信号日：不写目标，运行器那天只盯市、结算（与回测「非调仓日保持股数」一致）；
- 两种情况都更新 ``ops/plan.json``（上次信号日、上次成交日、下次信号日、待执行的成交日）。

幂等：同一成交日的目标已用同一面板版本算过就跳过；面板版本变了（数据被刷新）且成交日还没开始
运行，重算并覆盖。

2026-11-14 审查后补的三条：

- 成交日已经开始运行（``ops/runs/<成交日>.json`` 存在）就不再给这一天写目标——运行器开跑时读一次
  目标，之后写的它看不到。这次调仓记为待执行，由下一条在下一个交易日补发；
- 上一次的调仓没有执行（待执行的成交日早于本次成交日，而那天没有运行记录、运行时没带目标或运行
  出错）：本次补发，理由里写明哪天没执行；
- 选股结果不对劲不写目标：0 只一律不写（写空目标等于让运行器次日清仓）；最新一行的有效信号数比
  上一行骤降、或入选只数明显少于预期，说明当天数据缺了一块，返回 NOT_READY 等下一次盘后重试。
"""

from __future__ import annotations

import hashlib
import json
import logging
import time as _time
from pathlib import Path

from .accounts import active
from .after_close import ready_day
from .calendar import TradingCalendar
from .context import (
    DATA_LOCK,
    Busy,
    Context,
    Exit,
    StageResult,
    event,
    read_json,
    record,
    stage_lock,
    write_json,
)
from .strategy import LivePanel, load, pool_fields, pool_row, runner_safe, signal_alpha
from .strategy import target_weights as select

STAGE = "signal"
# 最新一行（股票池内）有效信号数至少是上一行的 80%。A 股每天上市、退市、停复牌带来的变动通常在
# 1%～3%；少两成以上几乎只会是当天某个字段整列缺失（数据没发布齐、被判可疑后清空）。
COVERAGE_FLOOR = 0.8
# 入选只数至少是 min(top_n, 上一行有效信号数) 的一半；并列时入选数只会多不会少，少一半说明这一行
# 能排序的股票所剩无几。
FILL_FLOOR = 0.5


class NotReady(RuntimeError):
    """这个账户今天的选股结果不可信，等数据补齐后重跑（不写目标）。"""


def digest(strategy) -> str:
    text = json.dumps(
        dict(source=strategy.source, settings=strategy.settings),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def signal(
    ctx: Context, accounts: list[str] | None = None, *, calendar=None
) -> StageResult:
    started = _time.perf_counter()
    try:
        with stage_lock(ctx, DATA_LOCK):
            result = _run(ctx, accounts, calendar)
    except Busy:
        return StageResult(STAGE, Exit.BUSY, "盘后或信号阶段已有实例在运行")
    except Exception as error:  # noqa: BLE001
        logging.getLogger("ashare.paper.ops").exception("signal_failed")
        result = StageResult(STAGE, Exit.FAILED, f"意外错误：{type(error).__name__}")
    return record(ctx, result, started)


def _run(ctx, names, calendar):
    accounts, unknown = active(ctx, names)
    if unknown:
        return StageResult(STAGE, Exit.USAGE, "没有这些账户：" + "、".join(unknown))
    if not accounts:
        # 还没开账户（或全部停用）：机制照常可用，这一步干净跳过。
        event("signal_no_accounts")
        return StageResult(
            STAGE, Exit.OK, "没有启用的账户，跳过", None, dict(skipped=True)
        )
    calendar = calendar or TradingCalendar.from_store(ctx.data)
    day = calendar.latest_closed(ctx.now())
    ready = ready_day(ctx)
    if day is None:
        return StageResult(STAGE, Exit.FAILED, "交易日历没有覆盖今天")
    if ready.get("day") != day:
        return StageResult(STAGE, Exit.NOT_READY, f"等待 {day} 的盘后数据", day)
    panel = LivePanel(ctx.data)
    if str(panel.dates[-1]) != day or panel.version != ready.get("panel_version"):
        return StageResult(
            STAGE, Exit.NOT_READY, "面板与盘后结果不一致，先重跑盘后阶段", day
        )
    execution = calendar.next_open(day)
    if execution is None:
        return StageResult(
            STAGE, Exit.FAILED, f"交易日历里没有 {day} 之后的开市日", day
        )
    outcomes, failed, waiting = {}, [], []
    for name, directory in accounts:
        try:
            outcomes[name] = signal_account(
                directory, panel, calendar, day, execution, ctx
            )
        except NotReady as error:
            event(
                "signal_not_ready",
                logging.WARNING,
                account=name,
                day=day,
                reason=str(error),
            )
            outcomes[name] = f"未就绪：{error}"
            waiting.append(name)
        except Exception as error:  # noqa: BLE001 - 一个账户出错不影响其他账户
            logging.getLogger("ashare.paper.ops").exception(
                "signal_account_failed account=%s", name
            )
            outcomes[name] = f"失败：{type(error).__name__}"
            failed.append(name)
    if failed:
        code, message = Exit.FAILED, "以下账户出信号失败：" + "、".join(failed)
    elif waiting:
        code, message = (
            Exit.NOT_READY,
            "以下账户的选股结果不可信：" + "、".join(waiting),
        )
    else:
        code, message = Exit.OK, "目标已生成或已是最新"
    return StageResult(
        STAGE, code, message, day, dict(execution_day=execution, accounts=outcomes)
    )


def signal_account(
    directory: Path, panel, calendar, day: str, execution: str, ctx
) -> str:
    strategy = load(directory)
    if not strategy.enabled:
        return "已停用"
    ops = directory / "ops"
    plan = read_json(ops / "plan.json", {}) or {}
    rebalance = strategy.settings["rebalance"]
    period_end = calendar.is_period_end(day, rebalance)
    if period_end is None:
        raise ValueError("交易日历没覆盖到本调仓周期末")
    if plan.get("last_signal_day") == day:
        # 同一天重跑：沿用当时的理由，下面按目标文件判断是否已是最新。
        reasons = [plan.get("last_reason") or "重跑"]
    else:
        reasons = []
        if not plan.get("last_signal_day"):
            reasons.append("建仓")
        if period_end:
            reasons.append("周期末")
        checked = plan.get("last_checked_day")
        if checked and checked < day:
            missed = [
                d
                for d in calendar.open_between(checked, day)
                if calendar.is_period_end(d, rebalance)
            ]
            if missed:
                reasons.append("补发：" + "、".join(missed))
    undone = unexecuted(ops, plan, execution)
    if undone and not any(undone in r for r in reasons):
        reasons.append(f"补发：{undone} 的调仓没有执行")
    nxt = calendar.next_period_end(day, rebalance)
    target_path = ops / "targets" / f"{execution}.json"
    existing = read_json(target_path)
    if not reasons:
        write_json(
            ops / "plan.json",
            plan | dict(last_checked_day=day, next_signal_day=nxt),
        )
        event("signal_hold", account=directory.name, day=day)
        return "非调仓日"
    fingerprint = digest(strategy)
    if (
        existing
        and existing.get("signal_day") == day
        and existing.get("panel_version") == panel.version
        and existing.get("strategy") == fingerprint
    ):
        # 目标已写好；计划文件可能在上次写目标后中断，补齐。
        write_json(
            ops / "plan.json",
            plan
            | dict(
                last_checked_day=day,
                last_signal_day=day,
                last_reason=existing.get("reason"),
                pending_execution_day=execution,
                next_signal_day=nxt,
            ),
        )
        return "已是最新"
    if (ops / "runs" / f"{execution}.json").exists():
        # 运行器开跑时读一次目标：这时再写它看不到。记为待执行，下一个交易日由 unexecuted 补发。
        reason = "；".join(reasons)
        write_json(
            ops / "plan.json",
            plan
            | dict(
                last_checked_day=day,
                last_signal_day=day,
                last_reason=reason,
                pending_execution_day=execution,
                next_signal_day=nxt,
            ),
        )
        event(
            "signal_frozen",
            logging.WARNING,
            account=directory.name,
            execution_day=execution,
            reason=reason,
        )
        return "成交日已开始运行，目标不再改"

    universe = strategy.settings["universe"]
    top_n = strategy.settings["top_n"]
    panel.ensure(pool_fields(universe))
    alpha = signal_alpha(panel, strategy.expression())
    row = len(panel.dates) - 1
    pool = pool_row(panel, universe, row)
    weights = select(alpha[row], pool, top_n, panel.symbols)
    check_selection(panel, universe, alpha, row, pool, top_n, len(weights))
    safe, moved = runner_safe(weights)
    target = dict(
        version=1,
        **strategy.execution,
        weights=safe,
        signal_day=day,
        execution_day=execution,
        panel_version=panel.version,
        strategy=fingerprint,
        reason="；".join(reasons),
        nudged=moved,
    )
    write_json(target_path, target)
    with (ops / "signals.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                dict(
                    signal_day=day,
                    execution_day=execution,
                    count=len(safe),
                    panel_version=panel.version,
                    strategy=fingerprint,
                    reason=target["reason"],
                    at=ctx.now().isoformat(timespec="seconds"),
                ),
                ensure_ascii=False,
            )
            + "\n"
        )
    write_json(
        ops / "plan.json",
        plan
        | dict(
            last_checked_day=day,
            last_signal_day=day,
            last_reason=target["reason"],
            pending_execution_day=execution,
            next_signal_day=nxt,
        ),
    )
    event(
        "signal_written",
        account=directory.name,
        day=day,
        execution_day=execution,
        count=len(safe),
        reason=target["reason"],
    )
    return f"已生成 {len(safe)} 只目标（{target['reason']}）"


def unexecuted(ops: Path, plan: dict, execution: str) -> str | None:
    """上一次调仓的成交日（早于本次成交日）没有按目标执行时返回那一天，否则 None。"""
    pending = plan.get("pending_execution_day")
    if not pending or pending >= execution:
        return None
    marker = read_json(ops / "runs" / f"{pending}.json")
    if (
        not marker
        or marker.get("targets") != f"{pending}.json"
        or marker.get("error") is not None
    ):
        return pending
    return None


def _valid(alpha_row, pool) -> int:
    import numpy as np

    return int((np.isfinite(np.asarray(alpha_row, dtype=float)) & pool).sum())


def check_selection(panel, universe, alpha, row, pool, top_n, selected) -> None:
    """选股结果的健全性检查（2026-11-14 审查）：不对劲就抛 NotReady / ValueError，不写目标。"""
    valid = _valid(alpha[row], pool)
    previous = _valid(alpha[row - 1], pool_row(panel, universe, row - 1)) if row else 0
    if selected == 0 and valid == 0 and previous == 0:
        raise ValueError("策略最近两个交易日都选不出股票")
    if previous and valid < previous * COVERAGE_FLOOR:
        raise NotReady(f"有效信号从 {previous} 只降到 {valid} 只")
    expected = min(top_n, previous or valid)
    if selected == 0 or selected < expected * FILL_FLOOR:
        raise NotReady(f"只选出 {selected} 只（预期约 {expected} 只）")
