"""给平台 spawn 的 tab 工作进程插闸门（只在测试里用）。

`Manager` 用 `self.context.Process(target=worker_main, args=...)` 起进程。测试把
`manager.context` 换成 `GatedContext`：进程仍是真正的 spawn 子进程，只是入口换成
`gated_worker_main`，它在子进程里给 `runner.calculate_many` 和结果队列加上闸门，
再调用原来的 `runner.worker_main`。

闸门全部用一个目录里的文件表示，子进程轮询文件是否存在。不用 multiprocessing 的
Event / Barrier：它们基于 Condition，被取消（SIGTERM 整个进程组）的子进程如果正
等在上面，会在计数里留下永远醒不来的「睡眠者」，之后测试进程 `set()` 时卡死
（2026-11-14 写这组测试时实际踩到：取消后放行闸门，测试挂满 240 s 超时）。

- `arrive-<job>-<pid>`：子进程进入计算（信号已求值、回测开始前）时创建；
- `parties`：几个子进程的到达文件都出现后才一起往下走，证明它们确实同时在算；
- `hold` 文件存在时，进入计算的子进程停在闸门前，直到 `release` 文件出现；
- `results_before_hold`：子进程只放行前 k 个结果，之后的结果和完成消息等 `release`。
"""

import multiprocessing as mp
import os
import threading
import time
from pathlib import Path

POLL = 0.002
LIMIT = 120.0


def _await(predicate):
    deadline = time.monotonic() + LIMIT
    while not predicate():
        if time.monotonic() > deadline:
            raise TimeoutError("测试闸门在期限内没有放行")
        time.sleep(POLL)


class Gates:
    def __init__(self, folder, *, parties=0, results_before_hold=None):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.parties = parties
        self.results_before_hold = results_before_hold

    def hold(self):
        (self.folder / "hold").touch()

    def release(self):
        (self.folder / "release").touch()

    def held(self):
        return (self.folder / "hold").exists()

    def released(self):
        return (self.folder / "release").exists()

    def arrived(self):
        """[(job, pid)]，按到达文件名排序。"""
        return sorted(
            (name.split("-")[1], int(name.split("-")[2]))
            for name in (p.name for p in self.folder.glob("arrive-*"))
        )

    def wait_arrivals(self, count, timeout=LIMIT):
        deadline = time.monotonic() + timeout
        while len(self.arrived()) < count:
            if time.monotonic() > deadline:
                raise TimeoutError(f"{count} 个工作进程没有在期限内进入计算")
            time.sleep(POLL)
        return self.arrived()

    def arrive(self, job):
        (self.folder / f"arrive-{job}-{os.getpid()}").touch()
        if self.parties:
            _await(lambda: len(self.arrived()) >= self.parties)
        if self.held():
            _await(self.released)


class GatedContext:
    def __init__(self, gates):
        self.real = mp.get_context("spawn")
        self.gates = gates
        self.processes = []

    def Queue(self):  # noqa: N802 - 与 multiprocessing 上下文同名
        return self.real.Queue()

    def Process(self, target, args):  # noqa: N802
        process = self.real.Process(
            target=gated_worker_main, args=(self.gates, target, *args)
        )
        self.processes.append(process)
        return process


class _HeldQueue:
    """前 k 个结果照常发出；之后的结果和完成消息等测试放行。"""

    def __init__(self, queue, gates):
        self._queue, self._gates = queue, gates
        self._lock = threading.Lock()
        self._sent = 0

    def put(self, item):
        kind = item[1]
        if kind in ("result", "complete"):
            with self._lock:
                held = self._sent >= self._gates.results_before_hold
                if kind == "result":
                    self._sent += 1
            if held:
                _await(self._gates.released)
        self._queue.put(item)

    def __getattr__(self, name):
        return getattr(self._queue, name)


def gated_worker_main(gates, target, incoming, outgoing, *rest):
    from ashare.platform import runner

    original = runner.calculate_many

    def calculate_many(runs, panel, job, out, parallel):
        gates.arrive(job)
        return original(runs, panel, job, out, parallel)

    runner.calculate_many = calculate_many
    if gates.results_before_hold is not None:
        outgoing = _HeldQueue(outgoing, gates)
    target(incoming, outgoing, *rest)
