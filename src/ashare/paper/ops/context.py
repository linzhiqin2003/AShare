"""运行环境：目录、时钟、退出码、阶段锁、日志与阶段状态文件。

目录（都不入库）：

- 数据根 ``$ASHARE_DATA_DIR``（缺省为仓库下的 ``data/``），原始库与面板在 ``<数据根>/cn``；
- 账户根 ``$ASHARE_PAPER_DIR``（缺省 ``<数据根>/paper``），每个子目录一个账户；
- 运行目录 ``<账户根>/.ops``：``logs/<日期>.log`` 文本日志，``state/<阶段>.json`` 每个阶段最近
  一次的结果，``locks/<阶段>.lock`` 防重复启动。目录名以点开头，不会被当成账户。
"""

from __future__ import annotations

import json
import logging
import os
import time as _time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from pathlib import Path

from paperbroker.journal import durable_write

from ..market import TZ
from ..source import Clock, SystemClock

LOG = logging.getLogger("ashare.paper.ops")


class Exit(IntEnum):
    """各阶段的退出码；调度器据此判断要不要稍后重试。"""

    OK = 0  # 完成，或今天无事可做（非交易日、已完成、已过时段）
    FAILED = 1  # 出错；不会在半成品状态下进入下一步
    USAGE = 2  # 参数错误
    BUSY = 3  # 同一阶段已有实例在运行
    NOT_READY = 4  # 数据还没发布或上游阶段没完成，稍后重跑


@dataclass
class StageResult:
    stage: str
    code: Exit
    message: str
    day: str | None = None
    details: dict = field(default_factory=dict)


def data_root(value=None) -> Path:
    from ashare.data.store import data_root as resolve

    return resolve(value)


def paper_root(value=None, data=None) -> Path:
    if value:
        return Path(value)
    configured = os.environ.get("ASHARE_PAPER_DIR")
    if configured:
        return Path(configured)
    return data_root(data) / "paper"


@dataclass
class Context:
    data: Path
    paper: Path
    clock: Clock

    @classmethod
    def create(cls, data=None, paper=None, clock=None) -> Context:
        resolved = data_root(data)
        return cls(resolved, paper_root(paper, resolved), clock or SystemClock())

    @property
    def ops(self) -> Path:
        return self.paper / ".ops"

    def now(self) -> datetime:
        return self.clock.now().astimezone(TZ)


# 盘后与信号共用一把锁（2026-11-14 审查）：信号读的面板不能正被盘后阶段重建。
DATA_LOCK = "nightly"


class Busy(RuntimeError):
    pass


@contextmanager
def stage_lock(ctx: Context, name: str):
    """非阻塞排他锁；已被占用时抛 ``Busy``。锁文件里写当前进程号，便于排查。"""
    import fcntl

    path = ctx.ops / "locks" / f"{name}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise Busy(name) from None
        handle.seek(0)
        handle.truncate()
        handle.write(f"{os.getpid()}\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    durable_write(
        path, (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
    )


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_bytes())
    except FileNotFoundError:
        return default


def record(ctx: Context, result: StageResult, started: float) -> StageResult:
    """把阶段结果写进 ``state/<阶段>.json``（最近一次）并记一行日志。"""
    payload = dict(
        stage=result.stage,
        code=int(result.code),
        status=result.code.name.lower(),
        message=result.message,
        day=result.day,
        finished_at=ctx.now().isoformat(timespec="seconds"),
        seconds=round(_time.perf_counter() - started, 3),
        details=result.details,
    )
    write_json(ctx.ops / "state" / f"{result.stage}.json", payload)
    level = logging.INFO if result.code in (Exit.OK, Exit.NOT_READY) else logging.ERROR
    LOG.log(
        level,
        json.dumps(
            dict(event="stage_finished", **{k: payload[k] for k in payload}),
            ensure_ascii=False,
            sort_keys=True,
        ),
    )
    return result


def configure_logging(ctx: Context) -> None:
    """标准错误 + ``logs/<日期>.log``。同一进程再次配置（换了目录）时替换掉上次加的处理器。"""
    root = logging.getLogger()
    for handler in [h for h in root.handlers if getattr(h, "_ashare_ops", False)]:
        root.removeHandler(handler)
        handler.close()
    folder = ctx.ops / "logs"
    folder.mkdir(parents=True, exist_ok=True)
    form = logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    stream = logging.StreamHandler()
    daily = logging.FileHandler(folder / f"{ctx.now():%Y-%m-%d}.log", encoding="utf-8")
    for handler in (stream, daily):
        handler.setFormatter(form)
        handler._ashare_ops = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    root.setLevel(logging.INFO)


def event(name: str, level: int = logging.INFO, **values) -> None:
    LOG.log(
        level,
        json.dumps(dict(event=name, **values), ensure_ascii=False, sort_keys=True),
    )
