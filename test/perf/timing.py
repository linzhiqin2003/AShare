"""性能守卫的测量工具：同一测试内交替测量两组调用，只比比例。

`paired_ratio(a, b)`：每一轮紧挨着各测一次 a 和 b（轮与轮之间交换先后），算这一轮的
b / a，最后取各轮比值的中位数。相邻两次测量承受几乎相同的机器负载，负载的起伏在
比值里抵消；中位数再挡掉个别被抢占的轮次。单次调用太短时在一个样本里重复几次，
让每个样本至少约 `minimum` 秒，避免计时分辨率和调度抖动主导结果。

2026-11-14 第一版用「两组各自取中位数再相除」，在负载 40–60 的机器上（另外三条测试线
同时在跑）滚动算子的 T / 2T 比偶尔算出 < 1（如 T 37 ms、2T 21 ms），改成逐轮配对。
"""

import time
from statistics import median


def _repeated(call, minimum):
    start = time.perf_counter()
    call()
    once = time.perf_counter() - start
    count = max(1, int(minimum / max(once, 1e-6)) + 1) if once < minimum else 1

    def run():
        for _ in range(count):
            call()

    return run, count


def paired_ratio(a, b, *, rounds=9, minimum=0.03):
    """返回 (b/a 各轮比值的中位数, a 每次耗时中位数, b 每次耗时中位数)。"""
    a, count_a = _repeated(a, minimum)
    b, count_b = _repeated(b, minimum)
    a()
    b()
    ratios, times_a, times_b = [], [], []
    for index in range(rounds):
        order = (a, b) if index % 2 == 0 else (b, a)
        measured = {}
        for call in order:
            start = time.perf_counter()
            call()
            measured[call] = time.perf_counter() - start
        ta, tb = measured[a] / count_a, measured[b] / count_b
        times_a.append(ta)
        times_b.append(tb)
        ratios.append(tb / ta)
    return median(ratios), median(times_a), median(times_b)


def describe(names, ratio, a, b):
    return (
        f"{names[1]}/{names[0]} = {ratio:.2f}"
        f"（{names[0]} {a * 1e3:.1f} ms，{names[1]} {b * 1e3:.1f} ms）"
    )
