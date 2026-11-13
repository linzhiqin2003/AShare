"""因子试验的台账：直接用 `ashare.ledger.Ledger`（与 `ashare.ml` 的账本同一种 JSONL 格式、文件锁、追加写，
不依赖可选依赖组 ml），
一行一个事件。

- ``event="trial"``：评估了一个候选（``kind="factor"``）。``trial_id`` = 去重键（规范形
  摘要）+ 数据版本 + 评估定义（标签、成本、调仓、股票池、切分）的哈希；不含搜索策略和
  适应度，所以同一候选在不同搜索里只算一次试验、可以互相复用结果。粗筛掉的、判为与已入选
  因子重复的、求值失败的（``status="failed"``）都照样是一行 trial。
- ``event="cache_hit"``：断点续跑或另一次搜索遇到账本里已有的候选，直接用记下的指标。
- ``event="manual"`` 不存在：手工研究用 `record_trial` 记成普通的 trial（``source="manual"``）。
- ``event="holdout_unlock"``：显式解锁留出段，一次搜索只能一次。

`Ledger.trials()` 数不同 trial_id 的个数：跨多次搜索、跨搜索与手工、和机器学习的试验
（若指到同一个文件）一起累计。默认位置 ``ASHARE_MINING_DIR``，否则数据根目录下的
``mining/``（不入库）；可以指到私有仓库。
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
from pathlib import Path

from ...ledger import Ledger
from ...ledger import code_version as _code_version

__all__ = [
    "Ledger",
    "code_version",
    "default_root",
    "open_ledger",
    "trial_id",
    "record_trial",
]


@functools.cache
def code_version():
    """仓库提交号，以及 `ashare.factor` 目录下工作区是否有改动。"""
    return _code_version(Path(__file__).resolve().parents[1])


def default_root():
    """台账与指纹的默认目录：``ASHARE_MINING_DIR``；否则数据根目录（``ASHARE_DATA_DIR``，
    默认仓库下的 ``data/``）下的 ``mining/``。"""
    if os.environ.get("ASHARE_MINING_DIR"):
        return Path(os.environ["ASHARE_MINING_DIR"])
    data = (
        os.environ.get("ASHARE_DATA_DIR")
        or Path(__file__).resolve().parents[4] / "data"
    )
    return Path(data) / "mining"


def open_ledger(ledger=None, root=None):
    """Ledger 对象、路径或 None（默认 ``<root>/ledger.jsonl``）→ Ledger。"""
    if isinstance(ledger, Ledger):
        return ledger
    if ledger is not None:
        return Ledger(ledger)
    return Ledger(Path(root if root is not None else default_root()) / "ledger.jsonl")


def digest(value):
    text = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def trial_id(key, data_version, definition):
    """一个候选在一份数据、一套评估定义下的试验编号。key：`dedup_key` 的结果。"""
    return digest(["factor-trial", 1, key, str(data_version), definition])


def record_trial(
    ledger, expr, report, *, data_version, definition=None, tags=(), source="manual"
):
    """把一次手工评估（`FactorReport`）记进台账，返回 trial_id。

    expr：表达式（`Expr`）或因子名字符串；definition：评估定义（缺省只用名字区分）。
    这样手工试过的因子同样计入「到今天为止一共试了多少个」。"""
    from ...factor.expression import Expr
    from .space import dedup_key

    key = dedup_key(expr) if isinstance(expr, Expr) else f"name:{expr}"
    tid = trial_id(key, data_version, definition or {})
    led = open_ledger(ledger)
    led.append(
        {
            "event": "trial",
            "kind": "factor",
            "source": source,
            "trial_id": tid,
            "key": key,
            "expr": str(expr),
            "data_version": str(data_version),
            "definition": definition or {},
            "metrics": report.to_dict()["segments"] if report is not None else None,
            "status": "ok",
            "tags": list(tags),
            "code": code_version(),
        }
    )
    return tid
