"""运行阶段：交易日开盘前启动运行器，跑到 ``--until``（默认 17:00）后退出。

- 非交易日（按原始库的交易日历，含节假日）直接退出 0；已过 ``--until`` 也直接退出 0；
- 每个启用的账户一个 ``LiveRunner``（与 ``python -m ashare.paper start`` 同一个类、同样的账户
  恢复方式），当天有 ``ops/targets/<今天>.json`` 就带上它，没有就只盯市和结算；还没开始运行、
  当天也没有目标的新账户跳过；
- 所有账户共用一个 Tushare 出口（``TushareClient``：本机独占锁、rt_k 令牌桶），按账户轮流推进；
  一个账户出错只停它自己；
- 运行器从日志恢复，所以同一天重跑是安全的；阶段锁防止两个实例同时跑；
- 截止按墙上时间判断（``ctx.now() >= 今天的 --until``）：机器合盖睡眠时单调时钟不走，按单调时钟
  计时会让进程醒来后跑进下一个交易日、拿着旧目标交易（2026-11-14 审查）；
- 只用 ``execution_day`` 等于今天的目标文件；
- 当天已经结算过（``days/<今天>.json`` 存在）的账户不再启动；
- 每个账户在 ``ops/runs/<今天>.json`` 记开始、结束时间、用的目标文件和结算结果；
- ``--dry-run``：已有账本的账户先把账本复制到临时目录再恢复（与正式运行同样的持仓和现金），
  在副本上推进；结果里给出每个账户当天的持仓变化（股数），账户目录一个字节都不写。
"""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
import time as _time
from contextlib import ExitStack
from datetime import datetime, time
from pathlib import Path

from .accounts import active
from .calendar import TradingCalendar
from .context import (
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
from .strategy import load

STAGE = "run"
UNTIL = time(17, 0)
POLL_SECONDS = 6.0
# rt_k 按账号每分钟 50 次（Tushare 官方，不分进程）。录制任务全市场 6 秒一轮、每轮 3 次，约 30 次/
# 分钟；模拟盘所有账户共用这一个令牌桶，取 15 次/分钟，合计 45，给录制任务的重试留 5 次余量。
# 一个账户 6 秒一轮正好 10 次/分钟；账户多了每个账户轮询变慢（两个账户约 8 秒一轮），总量不变。
RT_K_PER_MINUTE = 15


def default_client(lock: Path, clock):
    from ..source import TushareClient

    return TushareClient(lock, clock=clock, per_minute=RT_K_PER_MINUTE)


def default_lock() -> Path:
    # 与 ``python -m ashare.paper start`` 默认的出口锁同一个文件：两者不会同时取数。
    return Path(__file__).resolve().parents[4] / ".cache/paper-tushare.lock"


def run(
    ctx: Context,
    accounts: list[str] | None = None,
    *,
    until: time = UNTIL,
    dry_run: bool = False,
    poll_seconds: float = POLL_SECONDS,
    client_factory=None,
    lock: Path | None = None,
    calendar: TradingCalendar | None = None,
) -> StageResult:
    started = _time.perf_counter()
    try:
        with stage_lock(ctx, STAGE):
            result = _run(
                ctx,
                accounts,
                until,
                dry_run,
                poll_seconds,
                client_factory or default_client,
                lock or default_lock(),
                calendar,
            )
    except Busy:
        return StageResult(STAGE, Exit.BUSY, "运行阶段已有实例在运行")
    except Exception as error:  # noqa: BLE001
        logging.getLogger("ashare.paper.ops").exception("run_failed")
        result = StageResult(STAGE, Exit.FAILED, f"意外错误：{type(error).__name__}")
    if dry_run:
        return result
    return record(ctx, result, started)


class Plan:
    """一个账户今天要做什么。"""

    def __init__(self, name: str, directory: Path, today: str):
        self.name, self.directory = name, directory
        self.strategy = load(directory)
        self.manifest = directory / "manifest.json"
        path = directory / "ops" / "targets" / f"{today}.json"
        self.target_path = path if path.is_file() else None
        self.started = self.manifest.is_file()
        self.settled = (directory / "days" / f"{today}.json").is_file()

    def targets(self) -> dict | None:
        if self.target_path is None:
            return None
        from ..runner import load_targets

        targets = load_targets(self.target_path)
        day = self.target_path.stem
        if json.loads(self.target_path.read_bytes()).get("execution_day") != day:
            raise ValueError(f"目标文件的成交日不是 {day}")
        return targets


def _run(ctx, names, until, dry_run, poll_seconds, client_factory, lock, calendar):
    accounts, unknown = active(ctx, names)
    if unknown:
        return StageResult(STAGE, Exit.USAGE, "没有这些账户：" + "、".join(unknown))
    now = ctx.now()
    today = now.strftime("%Y%m%d")
    if not accounts:
        # 还没开账户（或全部停用）：不读日历、不连行情，干净跳过。
        event("run_no_accounts", day=today)
        return StageResult(
            STAGE, Exit.OK, "没有启用的账户，跳过", today, dict(skipped=True)
        )
    calendar = calendar or TradingCalendar.from_store(ctx.data)
    status = calendar.is_open(today)
    if status is None:
        return StageResult(STAGE, Exit.FAILED, "交易日历没有覆盖今天", today)
    if not status:
        return StageResult(STAGE, Exit.OK, "非交易日", today, dict(skipped=True))
    if now.time().replace(tzinfo=None) >= until:
        return StageResult(STAGE, Exit.OK, "已过运行时段", today, dict(skipped=True))
    plans, skipped = [], {}
    for name, directory in accounts:
        try:
            plan = Plan(name, directory, today)
        except (OSError, ValueError, KeyError) as error:
            skipped[name] = f"策略定义读不了：{type(error).__name__}"
            continue
        if plan.settled:
            skipped[name] = "今天已结算"
        elif not plan.started and plan.target_path is None:
            skipped[name] = "等待首个目标"
        else:
            plans.append(plan)
    if not plans:
        return StageResult(
            STAGE, Exit.OK, "今天没有需要运行的账户", today, dict(skipped=skipped)
        )
    end = datetime.combine(now.date(), until, now.tzinfo)
    event(
        "run_start",
        day=today,
        accounts=[p.name for p in plans],
        until=until.isoformat(),
        dry_run=dry_run,
    )
    outcomes, failed = _drive(
        ctx,
        plans,
        today,
        end,
        dry_run,
        poll_seconds,
        client_factory,
        lock,
        calendar,
    )
    outcomes.update(skipped)
    code = Exit.FAILED if failed else Exit.OK
    message = "以下账户运行出错：" + "、".join(failed) if failed else "运行结束"
    return StageResult(STAGE, code, message, today, dict(accounts=outcomes))


# 账本以外的东西：行情录制、日报和我们自己的运行记录不复制进演练副本。
_NOT_JOURNAL = ("quotes", "reports", "ops", "latest.json", "strategy.json")


def _open(plan: Plan, stack: ExitStack, dry_run: bool):
    """打开账户：与 ``python -m ashare.paper start`` 相同的建账与恢复方式。

    演练时已有账本的账户在临时副本上恢复：持仓、现金与正式运行一致，写入只落在副本里。
    """
    from paperbroker import Broker

    from ..profiles import configuration, from_native, instrument
    from ..storage import ParquetStore

    targets = plan.targets()
    wanted = list((targets or {}).get("weights", {}))
    cash = plan.strategy.initial_cash
    journal = None if dry_run else plan.directory
    if plan.started:
        manifest = json.loads(plan.manifest.read_bytes())
        symbols = list(dict.fromkeys(manifest["symbols"] + wanted))
        cash, config = from_native(manifest["config"])
        config.instruments.update(
            {s: instrument(s) for s in symbols if s not in config.instruments}
        )
        if dry_run:
            copy = Path(stack.enter_context(tempfile.TemporaryDirectory())) / "journal"
            shutil.copytree(
                plan.directory, copy, ignore=lambda d, names: _ignored(plan, d, names)
            )
            journal = copy
    else:
        symbols = wanted
        config = configuration(symbols)
    store = None if dry_run else ParquetStore(plan.directory / "quotes")
    broker = stack.enter_context(
        Broker(symbols, cash, config, journal=journal, quote_store=store)
    )
    return broker, store, targets


def _ignored(plan: Plan, directory: str, names: list[str]) -> list[str]:
    if Path(directory) != plan.directory:
        return []
    return [n for n in names if n in _NOT_JOURNAL]


def _holdings(broker, report) -> dict[str, int]:
    positions = report(broker, [])["account"]["positions"]
    return {p["symbol"]: int(p["total"]) for p in positions}


def _drive(
    ctx, plans, today, end, dry_run, poll_seconds, client_factory, lock, calendar
):
    from ..runner import LiveRunner, report
    from ..schedule import Schedule
    from ..source import LiveSource, SourceError

    clock = ctx.clock
    schedule = Schedule(calendar.rows())
    outcomes, failed, runners, opening = {}, [], {}, {}
    with ExitStack() as stack:
        client = stack.enter_context(client_factory(lock, clock))
        for plan in plans:
            try:
                broker, store, targets = _open(plan, stack, dry_run)
                source = LiveSource(client, list(broker.symbols), interval=poll_seconds)
                runners[plan.name] = (
                    plan,
                    broker,
                    LiveRunner(
                        broker,
                        source,
                        schedule,
                        clock,
                        store,
                        targets,
                        None if dry_run else plan.directory / "reports",
                    ),
                )
                if dry_run:
                    opening[plan.name] = _holdings(broker, report)
                if not dry_run:
                    write_json(
                        _marker(plan, today),
                        dict(
                            day=today,
                            started_at=ctx.now().isoformat(timespec="seconds"),
                            targets=plan.target_path.name if plan.target_path else None,
                            symbols=len(broker.symbols),
                        ),
                    )
            except Exception as error:  # noqa: BLE001
                logging.getLogger("ashare.paper.ops").exception(
                    "run_open_failed account=%s", plan.name
                )
                outcomes[plan.name] = f"打开失败：{type(error).__name__}"
                failed.append(plan.name)
        interval = min(
            (r.source.interval for _, _, r in runners.values()), default=poll_seconds
        )
        # 墙上时间：睡眠醒来已过截止（或已是另一天）就立刻收尾，不再推进。
        while runners and ctx.now() < end:
            before = clock.monotonic()
            for name in list(runners):
                runner = runners[name][2]
                try:
                    runner.step()
                except SourceError as exc:
                    # 与 LiveRunner.run 相同：可恢复的数据源问题记日志后下一轮再试。
                    event(
                        "runner_source_failure",
                        logging.ERROR,
                        account=name,
                        reason=str(exc),
                    )
                except Exception as error:  # noqa: BLE001
                    logging.getLogger("ashare.paper.ops").exception(
                        "runner_failed account=%s", name
                    )
                    outcomes[name] = f"运行出错：{type(error).__name__}"
                    failed.append(name)
                    _finish(ctx, runners.pop(name), today, dry_run, report, error)
            clock.sleep(max(0.1, interval - (clock.monotonic() - before)))
        for name in list(runners):
            entry = runners.pop(name)
            outcomes[name] = _finish(ctx, entry, today, dry_run, report)
            if dry_run:
                after = _holdings(entry[1], report)
                before = opening.get(name, {})
                changes = {
                    s: after.get(s, 0) - before.get(s, 0)
                    for s in dict.fromkeys([*before, *after])
                    if after.get(s, 0) != before.get(s, 0)
                }
                outcomes[name] = dict(
                    status=outcomes[name], holdings=after, changes=changes
                )
    return outcomes, failed


def _marker(plan: Plan, today: str) -> Path:
    return plan.directory / "ops" / "runs" / f"{today}.json"


def _finish(ctx, entry, today, dry_run, report, error=None) -> str:
    plan, broker, runner = entry
    settled = runner.settled_day == today
    try:
        if broker.journal:
            broker.checkpoint()
        result = report(broker, runner.events)
        if not dry_run:
            from paperbroker.journal import canonical, durable_write

            durable_write(plan.directory / "latest.json", canonical(result))
    except Exception:  # noqa: BLE001
        logging.getLogger("ashare.paper.ops").exception(
            "run_finish_failed account=%s", plan.name
        )
    if not dry_run:
        marker = _marker(plan, today)
        write_json(
            marker,
            (read_json(marker, {}) or {})
            | dict(
                finished_at=ctx.now().isoformat(timespec="seconds"),
                settled=settled,
                error=type(error).__name__ if error else None,
            ),
        )
        if plan.target_path is not None:
            path = plan.directory / "ops" / "plan.json"
            write_json(
                path,
                (read_json(path, {}) or {})
                | dict(last_execution_day=today, pending_execution_day=None),
            )
    event("run_account_finished", account=plan.name, day=today, settled=settled)
    return "已结算" if settled else "未结算（次日开盘前补结算）"


def parse_until(text: str) -> time:
    value = time.fromisoformat(text)
    if value <= time(9, 0):
        raise ValueError("--until 必须晚于 9:00")
    return value
