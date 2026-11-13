"""试验账本：每次训练追加一行 JSON，记下复现和多重检验校正需要的一切。

一行一个事件：`"trial"`（真的训练了一次）或 `"cache_hit"`（同样的定义重跑、直接返回
缓存）。`trial_id` 是定义的哈希（数据版本、特征、标签、预处理、股票池、切分、模型参数、
种子、代码版本），不含线程数；`trials()` 数的是不同的 trial_id，这是「到今天为止一共
试了多少组」——超参搜索的每一组都会入账，不只最好的那组。
"""

from __future__ import annotations

import fcntl
import functools
import json
import os
import subprocess
import time
from pathlib import Path

from .store import default_root


@functools.cache
def code_version():
    """仓库提交号和工作区是否有改动（拿不到时为 None）。"""
    here = Path(__file__).resolve().parent
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=here,
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", "."],
            cwd=here,
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return {"git": head.stdout.strip(), "dirty": bool(dirty.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"git": None, "dirty": None}


class Ledger:
    """追加写的 JSONL 账本。path 默认 `<ASHARE_ML_DIR 或 data/ml>/ledger.jsonl`。"""

    def __init__(self, path=None):
        self.path = Path(path) if path is not None else default_root() / "ledger.jsonl"

    def append(self, record):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **record},
            ensure_ascii=False,
            default=_default,
            allow_nan=True,
        )
        with open(self.path, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        return record

    def records(self):
        if not self.path.exists():
            return []
        out = []
        with open(self.path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue  # 半行（写入中断）跳过
        return out

    def trials(self, *, since=None, tag=None):
        """不同定义的试验组数；since：只数这个时间（ISO 字符串）之后的；tag：只数带此标签的。"""
        ids = set()
        for r in self.records():
            if r.get("event") != "trial":
                continue
            if since and r.get("time", "") < since:
                continue
            if tag and tag not in r.get("tags", []):
                continue
            ids.add(r["trial_id"])
        return len(ids)

    def find(self, trial_id):
        return [r for r in self.records() if r.get("trial_id") == trial_id]


def _default(value):
    import numpy as np

    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)
