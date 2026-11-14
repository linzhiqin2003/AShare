"""原生内核释放 GIL、多个 Python 线程同时调用同一内核的结果与串行逐位相同。

释放 GIL 的判据：主线程调用内核期间，另一个纯 Python 计数线程能前进多少，与主线程
`time.sleep` 同样时长时前进多少之比。持有 GIL 的 C 调用（对照组用 `sorted`）期间计数线程
完全停住，比例接近 0；释放 GIL 的内核比例明显大于 0。比的是同一测试内的比例，与机器快慢无关。
来源：2026-11-14 并发测试补全（Rust 内核计算期间释放 GIL）。
"""

import sys
import threading
import time

import numpy as np
import panelbt as bt
import panelops as po
import pytest

from ashare.factor import groupbt
from bench.compare.common import inputs
from bench.platform.synthetic import panel

from .sync import Worker


class Spinner:
    def __init__(self):
        self.count = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._spin, daemon=True)

    def _spin(self):
        while not self._stop.is_set():
            self.count += 1

    def __enter__(self):
        self._thread.start()
        while self.count == 0:
            time.sleep(0.001)
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(10)

    def progress(self, call):
        before, start = self.count, time.perf_counter()
        call()
        return self.count - before, time.perf_counter() - start


def progress_ratio(call, rounds=3):
    """内核期间计数线程的进度 ÷ 同样时长 sleep 期间的进度，取几轮的中位数。"""
    ratios = []
    # 主线程每次回到解释器都可能把 GIL 让给计数线程最多一个切换间隔（默认 5 ms）；
    # 调小切换间隔，让这部分「边界赠送」远小于内核本身的时长。
    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-4)
    try:
        with Spinner() as spinner:
            for _ in range(rounds):
                busy, elapsed = spinner.progress(call)
                idle, _ = spinner.progress(lambda e=elapsed: time.sleep(e))
                ratios.append(busy / max(idle, 1))
    finally:
        sys.setswitchinterval(interval)
    return float(np.median(ratios)), elapsed


@pytest.fixture(scope="module")
def kernels():
    rng = np.random.default_rng(11)
    x = rng.normal(size=(2400, 1500))
    ret = rng.normal(0, 0.01, (2400, 1500))
    weights = rng.random((2400, 1500))
    market, target, costs, config = inputs(2400, 1500, interval=1)
    source = panel(n=300, full=True)
    alpha = rng.normal(size=source.shape)
    return {
        "panelops.ts_std": lambda: po.ts_std(x, 20),
        "panelops.ts_rank": lambda: po.ts_rank(x, 20),
        "panelbt.run_weights": lambda: bt.run_weights(weights, ret),
        "panelbt.run": lambda: bt.run(market, target, costs, config),
        "groupbt.run": lambda: groupbt.run(alpha, source, "ALL", sub_universes=False),
    }


def _long_enough(call, minimum=0.05):
    """重复调用直到单次总时长至少 minimum 秒，让比例有分辨率。"""
    start = time.perf_counter()
    call()
    once = time.perf_counter() - start
    repeat = max(1, int(np.ceil(minimum / max(once, 1e-6))))
    return lambda: [call() for _ in range(repeat)]


def test_detector_sees_gil_held_by_c_call():
    """对照组：`sorted` 整段持有 GIL，计数线程完全停住；证明判据能识别出没释放 GIL。"""
    data = np.random.default_rng(1).integers(0, 1 << 40, 600_000).tolist()
    ratio, _ = progress_ratio(_long_enough(lambda: sorted(data), 0.1))
    assert ratio < 0.05, ratio


@pytest.mark.parametrize(
    "name",
    [
        "panelops.ts_std",
        "panelops.ts_rank",
        "panelbt.run_weights",
        "panelbt.run",
        "groupbt.run",
    ],
)
def test_kernel_releases_gil(kernels, name):
    po.set_num_threads(1)
    try:
        ratio, elapsed = progress_ratio(_long_enough(kernels[name]))
    finally:
        po.set_num_threads(0)
    # 内核大部分时间不持有 GIL；输入检查、结果组装那一小段除外，留足余量。
    assert ratio > 0.2, (
        f"{name}: 计数线程进度只有 sleep 时的 {ratio:.3f}（{elapsed:.3f}s）"
    )


@pytest.mark.parametrize("threads", [1, 4])
def test_concurrent_panelops_calls_match_serial(threads):
    """几个 Python 线程同时调用 panelops（共享同一个 Rayon 池），结果与串行逐位相同。"""
    rng = np.random.default_rng(23)
    x = rng.normal(size=(1200, 400))
    x[rng.random(x.shape) < 0.05] = np.nan
    y = rng.normal(size=x.shape)
    calls = [
        lambda: po.ts_std(x, 20),
        lambda: po.ts_mean(x, 60),
        lambda: po.ts_rank(x, 10),
        lambda: po.ts_corr(x, y, 20),
        lambda: po.cs_rank(x),
        lambda: po.ts_sum(x, 30, exact=True),
    ]
    po.set_num_threads(threads)
    try:
        expected = [call() for call in calls]
        barrier = threading.Barrier(len(calls))

        def together(call):
            barrier.wait(30)
            return call()

        for _ in range(3):
            workers = [Worker(together, call) for call in calls]
            for worker in workers:
                worker.start()
            for worker, want in zip(workers, expected, strict=True):
                got = worker.get(60)
                assert got.tobytes() == want.tobytes()
            assert not barrier.broken
    finally:
        po.set_num_threads(0)


def test_concurrent_trade_mode_runs_match_serial():
    """成交模式（含现金、手数）几条线程同时跑，各自结果与串行逐位相同。"""
    cases = [inputs(600, 300, interval=i, variant=i) for i in (1, 3, 5, 7)]
    expected = [bt.run(*case) for case in cases]
    barrier = threading.Barrier(len(cases))

    def together(case):
        barrier.wait(30)
        return bt.run(*case)

    workers = [Worker(together, case) for case in cases]
    for worker in workers:
        worker.start()
    for worker, want in zip(workers, expected, strict=True):
        got = worker.get(60)
        for name in ("nav", "cash", "turnover", "commission"):
            a, b = getattr(got, name), getattr(want, name)
            assert np.asarray(a).tobytes() == np.asarray(b).tobytes(), name
