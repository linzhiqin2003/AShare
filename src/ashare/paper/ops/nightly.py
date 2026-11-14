"""``daily nightly``：盘后阶段通过后接着出信号；开盘前最后一次仍没出成目标时留下明显记录。

调度在 18:30、20:30 和次日 07:30 各跑一次 nightly。07:30 那次是开盘前最后的机会（2026-11-14
审查）：今天开市、有启用的账户、而盘后或信号阶段仍没成功时，日志记 ERROR，并写
``.ops/state/no-targets.json``（今天、停在哪个阶段、原因）；``daily status`` 会把它列在最前面。
之后任何一次 nightly 成功就删掉这个标记。
"""

from __future__ import annotations

import json
import logging
from datetime import time

from .accounts import active
from .calendar import TradingCalendar
from .context import Context, Exit, StageResult, event, read_json, write_json

# 早于这个钟点跑的 nightly 是开盘前最后一次（集合竞价 9:15 开始，运行器 8:55 启动）。
LAST_CHANCE = time(9, 0)
FLAG = "no-targets.json"


def nightly(
    ctx: Context,
    *,
    accept_verify: bool = False,
    after_close_fn=None,
    signal_fn=None,
    calendar: TradingCalendar | None = None,
) -> list[StageResult]:
    if after_close_fn is None:
        from .after_close import after_close as after_close_fn
    if signal_fn is None:
        from .signal import signal as signal_fn
    results = [after_close_fn(ctx, accept_verify=accept_verify)]
    if results[0].code == Exit.OK:
        results.append(signal_fn(ctx))
    _flag(ctx, results[-1], calendar)
    return results


def _flag(ctx: Context, last: StageResult, calendar) -> None:
    path = ctx.ops / "state" / FLAG
    now = ctx.now()
    today = now.strftime("%Y%m%d")
    if last.code == Exit.OK:
        if path.exists():
            path.unlink()
        return
    if now.time().replace(tzinfo=None) >= LAST_CHANCE or not active(ctx)[0]:
        return
    try:
        calendar = calendar or TradingCalendar.from_store(ctx.data)
        is_open = calendar.is_open(today)
    except Exception:  # noqa: BLE001 - 日历读不了就当作开市，宁可多报
        is_open = None
    if is_open is False:
        return
    event(
        "no_targets_today",
        logging.ERROR,
        day=today,
        stage=last.stage,
        status=last.code.name.lower(),
        message=last.message,
    )
    write_json(
        path,
        dict(
            day=today,
            stage=last.stage,
            status=last.code.name.lower(),
            message=last.message,
            at=now.isoformat(timespec="seconds"),
        ),
    )


def status(ctx: Context) -> dict:
    """各阶段最近一次的结果；今天开盘前没出成目标时 ``alerts`` 里写明。"""
    folder = ctx.ops / "state"
    stages = (
        {
            p.stem: json.loads(p.read_bytes())
            for p in sorted(folder.glob("*.json"))
            if p.name != FLAG
        }
        if folder.is_dir()
        else {}
    )
    alerts = []
    flag = read_json(folder / FLAG)
    if flag and flag.get("day") == ctx.now().strftime("%Y%m%d"):
        alerts.append(
            f"今天没有目标：{flag.get('stage')} 阶段 {flag.get('status')}，"
            f"{flag.get('message')}"
        )
    return dict(alerts=alerts, stages=stages)
