"""bench/ml 共用：真实数据路径、峰值驻留采样、负载记录、SVG 小工具。"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "bench" / "e2e"))
from common import default_data, task_info  # noqa: E402

sys.path.pop(0)

DATA = Path(os.environ.get("ASHARE_BENCH_DATA", default_data()))
# 缓存和账本：主目录 data/dev/ml（不入库）
ML_ROOT = Path(os.environ.get("ASHARE_ML_DIR", DATA / "dev" / "ml"))
RESULTS = Path(__file__).with_name("results")

try:
    import ctypes

    _LIB = ctypes.CDLL("/usr/lib/libproc.dylib")
except OSError:  # 非 macOS
    _LIB = None


def footprint():
    """物理内存占用（macOS phys_footprint）：不含可回收的文件映射页。

    驻留内存（RSS）会把读过的内存映射特征库文件页也算进去；footprint 更接近
    「这个进程真正占用、别人拿不走的内存」。两者都记。

    注意 macOS 的 libmalloc 会把释放掉的大块内存留在自己的缓存里、继续算进 footprint，
    逐折反复分配大矩阵时 footprint 只涨不跌；测量时设环境变量 `MallocLargeCache=0`
    关掉这个缓存，footprint 才是真实的存活内存（bench/ml 的脚本都这样跑）。"""
    if _LIB is None:
        return 0
    buf = (ctypes.c_uint64 * 40)()
    if _LIB.proc_pid_rusage(os.getpid(), 0, ctypes.byref(buf)) != 0:
        return 0
    # rusage_info_v0：16 字节 uuid 之后依次是 user、system、pkg_idle、interrupt、
    # pageins、wired、resident、phys_footprint……
    return int(buf[2 + 7])


class Peak:
    """后台线程每 20 ms 采一次驻留内存，记下峰值（相对开始时的增量和绝对值）。"""

    def __enter__(self):
        self.base = task_info()[0]
        self.peak = self.base
        self.fbase = footprint()
        self.fpeak = self.fbase
        self.stop = False
        self.load = os.getloadavg()
        self.t0 = time.perf_counter()

        def loop():
            while not self.stop:
                self.peak = max(self.peak, task_info()[0])
                self.fpeak = max(self.fpeak, footprint())
                time.sleep(0.02)

        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.t0
        self.stop = True
        self.thread.join()
        self.peak = max(self.peak, task_info()[0])
        self.load_end = os.getloadavg()

    def record(self, **extra):
        return {
            "seconds": round(self.seconds, 3),
            "peak_rss_gb": round(self.peak / 2**30, 3),
            "peak_delta_gb": round((self.peak - self.base) / 2**30, 3),
            "peak_footprint_gb": round(self.fpeak / 2**30, 3),
            "footprint_delta_gb": round((self.fpeak - self.fbase) / 2**30, 3),
            "load_start": [round(x, 2) for x in self.load],
            "load_end": [round(x, 2) for x in self.load_end],
            **extra,
        }


def save(name, payload):
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=str))
    return path


def log(*parts):
    print(time.strftime("%H:%M:%S"), *parts, flush=True)
