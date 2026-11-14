"""模拟账户的建立、列表、启用与停用；账户目录 ``<账户根>/<名称>/``。

账户目录里我们只写 ``strategy.json`` 和 ``ops/``（信号、目标、运行记录）；``manifest.json``、
事件日志、快照、行情录制和日报由运行器（``paperbroker``）在第一个运行日建立。

从回测建账户用一条 JSON 约定（「回测记录」），不依赖回测页的代码::

    {"id": "<24 位十六进制>", "name": "…", "source": "<表达式>",
     "settings": {<回测页设置>}, "status": "complete", "data_version": "…"}

默认的读取方式是只读打开回测页结果库 ``<平台目录>/backtests/index.sqlite3`` 的 ``backtests`` 表
（``ashare.platform.backtest.jobs`` 的存储格式；只读，不启动它的工作进程）。也可以把这条 JSON
存成文件交给 ``account create --backtest-file``。
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path

from ashare.platform.backtest import settings as contract

from .context import Context, read_json, write_json
from .strategy import NAME, Strategy, load, parse_source

BACKTEST_ID = r"[0-9a-f]{24}"


class AccountError(ValueError):
    """面向用户的一句话错误（不含路径）。"""

    def __init__(self, message: str, status: int = 422):
        super().__init__(message)
        self.status = status


def account_dir(ctx: Context, name: str) -> Path:
    if not isinstance(name, str) or not NAME.fullmatch(name):
        raise AccountError(
            "账户名称只能用字母、数字、汉字、点、下划线和短横线，最长 64 个字符"
        )
    return ctx.paper / name


def create(
    ctx: Context,
    name: str,
    *,
    source: str,
    settings: dict | None = None,
    display_name: str | None = None,
    capital: float | None = None,
    origin: dict | None = None,
    enabled: bool = True,
) -> Strategy:
    """新建账户目录并写入策略定义；同名账户已存在时拒绝（409）。"""
    directory = account_dir(ctx, name)
    try:
        normalized = contract.normalize(settings or {})
        if capital is not None:
            normalized = contract.normalize(normalized | {"capital": capital})
    except ValueError as error:
        raise AccountError(str(error)) from None
    try:
        parse_source(source)
    except ValueError as error:
        raise AccountError(f"表达式不能用于模拟账户：{error}") from None
    label = (display_name or name).strip()[:80] or name
    strategy = Strategy(
        name=label,
        source=source,
        settings=normalized,
        initial_cash=float(normalized["capital"]) * 10_000,
        enabled=enabled,
        created_at=ctx.now().isoformat(timespec="seconds"),
        origin=origin or {"kind": "expression"},
    )
    if directory.exists():
        raise AccountError("已有同名账户", 409)
    # 先在 .ops/tmp 里建好整个目录再改名过去：中途失败不会留下没有策略定义的半成品账户
    # （2026-11-14 审查）。.ops 和账户目录在同一个文件系统上，改名是原子的。
    staging = ctx.ops / "tmp"
    staging.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".create-", dir=staging))
    try:
        write_json(work / "strategy.json", strategy.to_dict())
        (work / "ops").mkdir()
        if directory.exists():
            raise AccountError("已有同名账户", 409)
        try:
            os.rename(work, directory)
        except OSError:
            # 改名前一刻被别人建了同名目录（非空时 rename 失败）。
            raise AccountError("已有同名账户", 409) from None
    finally:
        if work.exists():
            shutil.rmtree(work, ignore_errors=True)
    return strategy


def from_backtest(
    ctx: Context,
    name: str,
    record: dict,
    *,
    capital: float | None = None,
    display_name: str | None = None,
    enabled: bool = True,
) -> Strategy:
    """按「回测记录」约定建账户：表达式、设置原样沿用，资金可另给（万元）。"""
    import re

    if not isinstance(record, dict) or not re.fullmatch(
        BACKTEST_ID, str(record.get("id", ""))
    ):
        raise AccountError("回测记录格式不正确")
    if record.get("status", "complete") != "complete":
        raise AccountError("只能从已完成的回测建账户")
    settings = record.get("settings")
    if isinstance(settings, str):
        settings = json.loads(settings)
    return create(
        ctx,
        name,
        source=record.get("source", ""),
        settings=settings,
        display_name=display_name or record.get("name"),
        capital=capital,
        origin=dict(
            kind="backtest", id=record["id"], data_version=record.get("data_version")
        ),
        enabled=enabled,
    )


def set_enabled(ctx: Context, name: str, enabled: bool) -> Strategy:
    directory = account_dir(ctx, name)
    if not (directory / "strategy.json").is_file():
        raise AccountError("这个账户没有策略定义", 404)
    raw = json.loads((directory / "strategy.json").read_bytes())
    raw["enabled"] = bool(enabled)
    strategy = Strategy.from_dict(raw)
    write_json(directory / "strategy.json", strategy.to_dict())
    return strategy


def bound(ctx: Context) -> list[tuple[str, Path]]:
    """有策略定义的账户（名称、目录），按名称排序；只有手工 ``start`` 账户的不算。"""
    if not ctx.paper.is_dir():
        return []
    return [
        (p.name, p)
        for p in sorted(ctx.paper.iterdir())
        if NAME.fullmatch(p.name) and (p / "strategy.json").is_file()
    ]


def active(ctx: Context, names: list[str] | None = None):
    """要处理的账户：(账户列表, 不认识的名称)。只剩停用账户时列表为空；读不了的策略
    定义算作在列（交给阶段报错），不当成停用而静默跳过。"""
    accounts = bound(ctx)
    unknown = sorted(set(names or ()) - {n for n, _ in accounts})
    if names:
        accounts = [(n, p) for n, p in accounts if n in names]
    selected = []
    for name, directory in accounts:
        try:
            if not load(directory).enabled:
                continue
        except (OSError, ValueError, KeyError, TypeError):
            pass
        selected.append((name, directory))
    return selected, unknown


def describe(directory: Path) -> dict:
    """账户列表要的策略摘要与运行计划；读不到时给 None，不抛错。"""
    try:
        strategy = load(directory)
    except (OSError, ValueError, KeyError, TypeError):
        return dict(strategy=None, plan=None)
    s = strategy.settings
    plan = read_json(directory / "ops" / "plan.json", {}) or {}
    return dict(
        strategy=dict(
            name=strategy.name,
            universe=s["universe"],
            rebalance=s["rebalance"],
            top_n=s["top_n"],
            trade_at=s["trade_at"],
            initial_cash=strategy.initial_cash,
            enabled=strategy.enabled,
            origin=strategy.origin.get("kind"),
        ),
        plan=dict(
            last_signal_day=plan.get("last_signal_day"),
            last_execution_day=plan.get("last_execution_day"),
            next_signal_day=plan.get("next_signal_day"),
            pending_execution_day=plan.get("pending_execution_day"),
        ),
    )


# ------------------------------------------------------------------ 回测记录


class BacktestRecords:
    """只读打开回测页结果库；库不存在或表结构不对时当作没有回测。"""

    COLUMNS = ("id", "name", "source", "settings", "status", "data_version", "finished")

    def __init__(self, root: Path | None = None):
        if root is None:
            from ashare.platform.config import platform_dir

            root = platform_dir()
        self.path = Path(root) / "backtests" / "index.sqlite3"

    def _query(self, sql, params=()):
        if not self.path.is_file():
            return []
        uri = f"file:{self.path}?mode=ro"
        try:
            with sqlite3.connect(uri, uri=True, timeout=5) as db:
                db.row_factory = sqlite3.Row
                return [dict(r) for r in db.execute(sql, params).fetchall()]
        except sqlite3.Error:
            return []

    def _record(self, row: dict) -> dict:
        try:
            settings = json.loads(row["settings"])
        except (TypeError, ValueError):
            settings = None
        return dict(
            id=row["id"],
            name=row["name"],
            source=row["source"],
            settings=settings,
            status=row["status"],
            data_version=row.get("data_version"),
            finished=row.get("finished"),
        )

    def get(self, key: str) -> dict | None:
        rows = self._query(
            f"SELECT {', '.join(self.COLUMNS)} FROM backtests WHERE id=?", (key,)
        )
        return self._record(rows[0]) if rows else None

    def completed(self, limit: int = 100) -> list[dict]:
        rows = self._query(
            f"SELECT {', '.join(self.COLUMNS)} FROM backtests "
            "WHERE status='complete' ORDER BY finished DESC LIMIT ?",
            (limit,),
        )
        return [self._record(r) for r in rows]
