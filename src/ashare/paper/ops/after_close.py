"""盘后阶段：增量同步 → 确认当天数据齐 → 增量重建面板 → 数据校验。

每一步失败都停下，不进入下一步，状态文件记下停在哪一步：

1. 同步交易日历（含未来 40 天所在年份），定出要处理的交易日 E（已收盘的最近一个交易日）；
2. E 已处理过且面板版本没变：直接结束（幂等）；
3. 增量同步（与 ``python -m ashare.data sync`` 同一函数、同样的默认窗口和刷新规则，终点是 E）；
4. E 的核心日表（日线、复权因子、每日指标、涨跌停价、ST）必须都已完整入库，否则返回
   NOT_READY（Tushare 当天还没发布；数据层会把空表或骤降的分区记为可疑，下次重抓）；
5. 增量重建面板，面板最后一天必须是 E；
6. ``verify``：与上次通过时的计数比，硬性检查（清单、分区覆盖、主键、开盘掩码、公司行为对账）
   的不一致数变多、这次没有检查、或第一次出现就带不一致，都失败；其余检查变多只记警告。第一次
   运行以当前计数为基线。
   ``--accept-verify`` 在人工核对后把当前计数记为新基线。
"""

from __future__ import annotations

import asyncio
import logging
import time as _time
from datetime import timedelta

from .calendar import TradingCalendar
from .context import (
    DATA_LOCK,
    LOG,
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

STAGE = "after-close"
# 面板的股票字段来自这些按日分区的表；当天任何一张没齐就不建面板。
REQUIRED = ("daily", "adj_factor", "daily_basic", "stk_limit", "stock_st")
HARD_CHECKS = (
    "manifest_files",
    "unique_keys",
    "open_masks_match_panelbt",
    "action_reconciliation",
)
HARD_PREFIXES = ("partition_coverage_",)
SYNC_WINDOW_DAYS = 61  # 与 ``python -m ashare.data sync`` 的默认起点相同
CALENDAR_AHEAD_DAYS = 40


def default_source(store):
    from ashare.data.sources import Tushare

    # 与 ``python -m ashare.data sync`` 的默认值相同。
    return Tushare(
        rate=300, concurrency=24, state_path=store.root / "account.json", budget=3000
    )


def _hard(name: str) -> bool:
    return name in HARD_CHECKS or name.startswith(HARD_PREFIXES)


def compare_checks(baseline: dict, checks: dict) -> tuple[list[str], list[str]]:
    """返回 (硬性检查的问题, 其他检查里变多的)；没有基线（第一次运行）时都为空。

    硬性检查的问题有三种（后两种是 2026-11-14 审查补的）：不一致数比基线多；基线里有、这次没出现
    （``verify`` 在前提不满足时不写这一项，等于没检查）；基线里没有、这次第一次出现就带不一致。
    """
    if not baseline:
        return [], []
    hard, soft = [], []
    for name in sorted(baseline):
        if _hard(name) and name not in checks:
            hard.append(f"{name}: 这次没有检查")
    for name, check in sorted(checks.items()):
        failed = check["failed"]
        before = baseline.get(name)
        if before is None:
            if failed > 0:
                text = f"{name}: 新出现 {failed}"
                (hard if _hard(name) else soft).append(text)
            continue
        if failed > before:
            (hard if _hard(name) else soft).append(f"{name}: {before} → {failed}")
    return hard, soft


def after_close(
    ctx: Context,
    *,
    accept_verify: bool = False,
    force: bool = False,
    datasets: list[str] | None = None,
    source_factory=default_source,
    sync_fn=None,
    build_fn=None,
    verify_fn=None,
) -> StageResult:
    started = _time.perf_counter()
    try:
        with stage_lock(ctx, DATA_LOCK):
            result = _run(
                ctx,
                accept_verify,
                force,
                datasets,
                source_factory,
                sync_fn,
                build_fn,
                verify_fn,
            )
    except Busy:
        return StageResult(STAGE, Exit.BUSY, "盘后或信号阶段已有实例在运行")
    except Exception as error:  # noqa: BLE001 - 任何意外都记状态并以失败退出
        logging.getLogger("ashare.paper.ops").exception("after_close_failed")
        result = StageResult(STAGE, Exit.FAILED, f"意外错误：{type(error).__name__}")
    return record(ctx, result, started)


def _run(
    ctx, accept_verify, force, datasets, source_factory, sync_fn, build_fn, verify_fn
):
    from ashare.data.store import Store

    if sync_fn is None:
        from ashare.data.sync import sync as sync_fn
    if build_fn is None:
        from ashare.data.panel import build as build_fn
    if verify_fn is None:
        from ashare.data.verify import verify as verify_fn

    now = ctx.now()
    store = Store(ctx.data)
    state_path = ctx.ops / "state" / "after-close-done.json"
    done = read_json(state_path, {}) or {}

    async def synchronize(end_calendar, end_data, *, data: bool):
        async with source_factory(store) as source:
            today = now.date()
            await sync_fn(
                source,
                store,
                (today - timedelta(days=10)).isoformat(),
                end_calendar,
                ["trade_cal"],
            )
            if not data:
                return
            await sync_fn(
                source,
                store,
                (today - timedelta(days=SYNC_WINDOW_DAYS)).isoformat(),
                end_data,
                datasets,
            )

    # 1. 日历
    ahead = (now.date() + timedelta(days=CALENDAR_AHEAD_DAYS)).isoformat()
    try:
        asyncio.run(synchronize(ahead, None, data=False))
    except Exception as error:  # noqa: BLE001
        LOG.exception(
            "calendar_sync_failed"
        )  # 数据层的异常只含数据集、分区和类别，不含密钥
        event("calendar_sync_failed", logging.ERROR, error=type(error).__name__)
        return StageResult(
            STAGE, Exit.FAILED, f"交易日历同步失败：{type(error).__name__}"
        )
    calendar = TradingCalendar.from_store(ctx.data)
    day = calendar.latest_closed(now)
    if day is None:
        return StageResult(STAGE, Exit.FAILED, "交易日历没有覆盖今天")

    # 2. 幂等
    current = _panel_version(ctx)
    if (
        not force
        and not accept_verify
        and done.get("day") == day
        and done.get("panel_version") == current
    ):
        return StageResult(STAGE, Exit.OK, "当天已完成", day, dict(skipped=True))

    # 3. 增量同步
    iso = f"{day[:4]}-{day[4:6]}-{day[6:]}"
    try:
        asyncio.run(synchronize(ahead, iso, data=True))
    except Exception as error:  # noqa: BLE001
        LOG.exception("sync_failed")  # 数据层的异常只含数据集、分区和类别，不含密钥
        event("sync_failed", logging.ERROR, day=day, error=type(error).__name__)
        return StageResult(
            STAGE, Exit.FAILED, f"增量同步失败：{type(error).__name__}", day
        )

    # 4. 当天核心表是否齐
    manifest = store.manifest()
    missing = [
        name
        for name in REQUIRED
        if manifest.get(f"{name}/{day}", {}).get("status") != "complete"
    ]
    if missing:
        event("data_not_ready", day=day, missing=missing)
        return StageResult(
            STAGE,
            Exit.NOT_READY,
            "当天数据还没发布齐：" + "、".join(missing),
            day,
            dict(missing=missing),
        )

    # 5. 面板
    try:
        meta = build_fn(store, incremental=True)
    except Exception as error:  # noqa: BLE001
        LOG.exception("build_failed")  # 数据层的异常只含数据集、分区和类别，不含密钥
        event("build_failed", logging.ERROR, day=day, error=type(error).__name__)
        return StageResult(
            STAGE, Exit.FAILED, f"面板构建失败：{type(error).__name__}", day
        )
    version = str(meta["version"])
    last = _panel_last_day(ctx)
    if last != day:
        return StageResult(
            STAGE, Exit.FAILED, f"面板最后一天是 {last}，不是 {day}", day
        )

    # 6. 校验
    try:
        report = verify_fn(store)
    except Exception as error:  # noqa: BLE001
        LOG.exception("verify_failed")  # 数据层的异常只含数据集、分区和类别，不含密钥
        event("verify_failed", logging.ERROR, day=day, error=type(error).__name__)
        return StageResult(
            STAGE, Exit.FAILED, f"数据校验出错：{type(error).__name__}", day
        )
    checks = report["checks"]
    baseline_path = ctx.ops / "state" / "verify-baseline.json"
    baseline = (read_json(baseline_path, {}) or {}).get("failed", {})
    hard, soft = compare_checks(baseline, checks)
    if soft:
        event("verify_warning", logging.WARNING, day=day, increased=soft)
    if hard and not accept_verify:
        return StageResult(
            STAGE,
            Exit.FAILED,
            "数据校验的不一致变多：" + "；".join(hard),
            day,
            dict(increased=hard, warnings=soft),
        )
    write_json(
        baseline_path,
        dict(
            day=day,
            panel_version=version,
            accepted=bool(hard and accept_verify),
            failed={k: v["failed"] for k, v in checks.items()},
        ),
    )
    write_json(
        state_path,
        dict(
            day=day,
            panel_version=version,
            finished_at=ctx.now().isoformat(timespec="seconds"),
        ),
    )
    return StageResult(
        STAGE,
        Exit.OK,
        "盘后数据已就绪",
        day,
        dict(panel_version=version, warnings=soft, accepted=hard),
    )


def _panel_version(ctx: Context) -> str | None:
    try:
        import ashare.data

        return str(ashare.data.open(ctx.data).panel([]).version)
    except Exception:  # noqa: BLE001 - 还没有面板
        return None


def _panel_last_day(ctx: Context) -> str | None:
    import ashare.data

    dates = ashare.data.open(ctx.data).panel([]).dates
    return str(dates[-1]) if len(dates) else None


def ready_day(ctx: Context) -> dict:
    """信号阶段用：最近一次通过的盘后结果（交易日、面板版本）。"""
    return read_json(ctx.ops / "state" / "after-close-done.json", {}) or {}
