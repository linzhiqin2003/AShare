"""并发测试的闸门工具：用可观察的状态确认竞态窗口真的打开，不靠 sleep 赌时序。

- `wait_for`：轮询一个可观察条件直到成立（超时即失败），只用来「等到某个状态」，
  不用来「等一段时间让别的线程大概跑完」。
- `parked_at`：读线程当前的 Python 栈，判断它是否停在某个函数的某一行
  （例如阻塞在 `with lock:` 上）；C 层的阻塞（拿锁、等事件）没有自己的 Python 帧，
  所以最内层帧停在调用它的那一行。
"""

import inspect
import sys
import threading
import time

import pytest


def wait_for(predicate, *, timeout=30.0, message="条件在期限内没有成立"):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            pytest.fail(message)
        time.sleep(0.002)


def line_of(function, text):
    """函数源码里包含 `text` 的那一行的行号；找不到就报错，防止产品代码改了而测试悄悄失效。"""
    lines, first = inspect.getsourcelines(function)
    hits = [first + i for i, line in enumerate(lines) if text in line]
    assert len(hits) == 1, f"{function.__qualname__} 里应恰有一行包含 {text!r}"
    return hits[0]


def innermost(thread):
    return sys._current_frames().get(thread.ident)


def parked_at(thread, function, text, samples=3):
    """线程连续几次采样都停在 function 中包含 text 的那一行（阻塞在 C 层调用上）。"""
    line = line_of(function, text)
    code = function.__code__
    for _ in range(samples):
        frame = innermost(thread)
        if frame is None or frame.f_code is not code or frame.f_lineno != line:
            return False
        time.sleep(0.001)
    return True


def in_stack(thread, name):
    frame = innermost(thread)
    while frame is not None:
        if frame.f_code.co_name == name:
            return True
        frame = frame.f_back
    return False


class Recorder:
    """线程安全的事件记录，用来断言进入顺序。"""

    def __init__(self):
        self._lock = threading.Lock()
        self.events = []

    def add(self, *event):
        with self._lock:
            self.events.append(event)

    def count(self, kind):
        with self._lock:
            return sum(1 for e in self.events if e[0] == kind)

    def kinds(self):
        with self._lock:
            return [e[0] for e in self.events]


class Worker(threading.Thread):
    """记下返回值或异常的线程；join 超时直接失败（防死锁挂住整个测试进程）。"""

    def __init__(self, target, *args, **kwargs):
        super().__init__(daemon=True)
        self._call = (target, args, kwargs)
        self.result = self.error = None
        self.finished = threading.Event()

    def run(self):
        target, args, kwargs = self._call
        try:
            self.result = target(*args, **kwargs)
        except BaseException as error:  # noqa: BLE001 - 交给主线程断言
            self.error = error
        finally:
            self.finished.set()

    def get(self, timeout=30.0):
        self.join(timeout)
        if self.is_alive():
            pytest.fail(f"线程 {self.name} 在 {timeout}s 内没有结束（死锁？）")
        if self.error is not None:
            raise self.error
        return self.result
