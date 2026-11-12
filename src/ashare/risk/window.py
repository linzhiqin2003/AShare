"""残差历史窗口（双份环形缓冲）与特质方差的预计算结果。"""

from typing import NamedTuple

import numpy as np


class WindowVariances(NamedTuple):
    base: np.ndarray
    corr0: np.ndarray
    corrnw: np.ndarray
    count: np.ndarray


class ResidualWindow:
    """最近 capacity 天残差的环形缓冲；每天写两份，任意不超过 capacity 天的窗口
    都是按时间排列的连续视图，不必每天把整段历史拼成新数组。"""

    def __init__(self, n, capacity):
        self.capacity = capacity
        self.rows = np.full((2 * capacity, n), np.nan)
        self.latest = -1
        self.first = None  # 第一次推入的日期；追加时可以从窗口起点而不是第 0 天开始

    def leaving(self, day):
        """推入 day 时将被覆盖（出界）的那一行；没有则为 None。"""
        if self.first is None or day - self.capacity < self.first:
            return None
        return self.rows[day % self.capacity]

    def push(self, day, residual):
        if self.first is None:
            if day < 0:
                raise ValueError("残差必须逐日连续推入")
            self.first, self.latest = day, day - 1
        if day != self.latest + 1:
            raise ValueError("残差必须逐日连续推入")
        slot = day % self.capacity
        self.rows[slot] = residual
        self.rows[slot + self.capacity] = residual
        self.latest = day

    def window(self, first, last):
        """第 first..last 天（含）的残差，按时间排列的连续视图。"""
        length = last - first + 1
        if (
            self.first is None
            or not self.first <= first <= last <= self.latest
            or length > self.capacity
        ):
            raise ValueError("请求的残差窗口不在缓冲内")
        if self.latest - self.capacity >= first:
            raise ValueError("请求的残差窗口已被覆盖")
        end = last % self.capacity + self.capacity + 1
        return self.rows[end - length : end]
