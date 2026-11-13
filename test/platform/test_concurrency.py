"""并发自动调节：按核数、内存定全机预算，按正在跑的任务平分。"""

import pytest

from ashare.platform import config
from ashare.platform.config import job_concurrency, ops_threads, parallel_budget

GIB = 2**30


@pytest.mark.parametrize(
    ("cores", "memory", "expected"),
    [
        (15, 24 * GIB, 4),  # 本机：核和内存都够，封顶 4
        (6, 64 * GIB, 2),  # 核少：每 3 核一个
        (32, 9 * GIB, 2),  # 内存紧：(9 − 6) / 1.5
        (32, 6 * GIB, 1),  # 内存只够面板：至少 1
        (2, 4 * GIB, 1),
        (16, None, 4),  # 拿不到内存时按核数
    ],
)
def test_parallel_budget_follows_cores_and_memory(monkeypatch, cores, memory, expected):
    monkeypatch.delenv("ASHARE_PLATFORM_VARIANTS", raising=False)
    monkeypatch.setattr(config.os, "cpu_count", lambda: cores)
    monkeypatch.setattr(config, "_physical_memory", lambda: memory)
    assert parallel_budget() == expected


def test_environment_overrides(monkeypatch):
    monkeypatch.setenv("ASHARE_PLATFORM_VARIANTS", "3")
    monkeypatch.setenv("ASHARE_PLATFORM_OPS_THREADS", "2")
    assert parallel_budget() == 3
    assert ops_threads() == 2
    monkeypatch.setenv("ASHARE_PLATFORM_VARIANTS", "99")
    assert parallel_budget() == 8


def test_ops_threads_default(monkeypatch):
    monkeypatch.delenv("ASHARE_PLATFORM_OPS_THREADS", raising=False)
    monkeypatch.setattr(config.os, "cpu_count", lambda: 15)
    assert ops_threads() == 8
    monkeypatch.setattr(config.os, "cpu_count", lambda: 4)
    assert ops_threads() == 4


@pytest.mark.parametrize(
    ("budget", "variants", "others", "expected"),
    [
        (4, 4, 0, 4),
        (4, 8, 0, 4),
        (4, 2, 0, 2),  # 不超过变体数
        (4, 8, 1, 2),  # 两个 tab 同时跑，平分
        (4, 8, 3, 1),
        (1, 8, 2, 1),  # 至少 1
    ],
)
def test_job_concurrency_splits_budget(budget, variants, others, expected):
    assert job_concurrency(budget, variants, others) == expected
