"""追加写的 JSONL 试验台账（`ashare.ml` 与因子挖掘共用，只用标准库）。

一行一个 JSON 事件；写入时加文件锁、``fsync``。上一次写入若在半行处中断，下一次追加前
先补一个换行，坏行只丢它自己、不吞掉后面的记录；读取时跳过坏行并计数
（``skipped_lines``）。`trials()` 数不同 ``trial_id`` 的 ``event="trial"`` 记录。
"""

from __future__ import annotations

import fcntl
import json
import os
import subprocess
import time
from pathlib import Path


def code_version(directory):
    """directory 所在仓库的提交号，以及 directory 之下工作区是否有改动（拿不到时为 None）。"""
    env = dict(os.environ, GIT_OPTIONAL_LOCKS="0")
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=directory,
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", "."],
            cwd=directory,
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return {"git": head.stdout.strip(), "dirty": bool(dirty.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"git": None, "dirty": None}


class Ledger:
    """追加写的 JSONL 台账，path 为文件路径。"""

    def __init__(self, path):
        self.path = Path(path)
        self.skipped_lines = 0

    def append(self, record):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **record},
            ensure_ascii=False,
            default=_default,
            allow_nan=True,
        )
        with open(self.path, "ab+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0, os.SEEK_END)
            if f.tell() > 0:
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    f.write(b"\n")  # 上一条在半行处中断：先结束那一行
            f.write((line + "\n").encode())
            f.flush()
            os.fsync(f.fileno())
        return record

    def records(self):
        self.skipped_lines = 0
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
                        self.skipped_lines += 1  # 半行（写入中断）跳过
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
