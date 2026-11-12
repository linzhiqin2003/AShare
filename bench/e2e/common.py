"""端到端剖析的共用工具：计时、CPU 比、内存峰值与缺页采样、运行环境和结果摘要。"""

import ctypes
import gc
import hashlib
import os
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def default_data():
    """工作区没有 data/ 时取主检出的 data/（git 公共目录的上一级）。"""
    if (ROOT / "data/cn/panel/current.json").exists():
        return ROOT / "data"
    common = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    return Path(common).parent / "data"


class _TaskInfo(ctypes.Structure):
    _fields_ = [
        ("virtual", ctypes.c_uint64),
        ("resident", ctypes.c_uint64),
        ("total_user", ctypes.c_uint64),
        ("total_system", ctypes.c_uint64),
        ("threads_user", ctypes.c_uint64),
        ("threads_system", ctypes.c_uint64),
        ("policy", ctypes.c_int32),
        ("faults", ctypes.c_int32),
        ("pageins", ctypes.c_int32),
        ("cow_faults", ctypes.c_int32),
        ("messages_sent", ctypes.c_int32),
        ("messages_received", ctypes.c_int32),
        ("syscalls_mach", ctypes.c_int32),
        ("syscalls_unix", ctypes.c_int32),
        ("csw", ctypes.c_int32),
        ("threadnum", ctypes.c_int32),
        ("numrunning", ctypes.c_int32),
        ("priority", ctypes.c_int32),
    ]


try:
    _LIBPROC = ctypes.CDLL("/usr/lib/libproc.dylib")
except OSError:  # 非 macOS：只报告 0，计时不受影响
    _LIBPROC = None


def task_info():
    """当前驻留字节、累计缺页与页入数（macOS libproc）。"""
    info = _TaskInfo()
    if _LIBPROC is None or _LIBPROC.proc_pidinfo(
        os.getpid(), 4, 0, ctypes.byref(info), ctypes.sizeof(info)
    ) != ctypes.sizeof(info):
        return 0, 0, 0
    return info.resident, info.faults, info.pageins


class _Sampler(threading.Thread):
    def __init__(self, interval=0.005):
        super().__init__(daemon=True)
        self.interval, self.peak, self.stop = interval, task_info()[0], False

    def run(self):
        while not self.stop:
            self.peak = max(self.peak, task_info()[0])
            time.sleep(self.interval)


# 计时区内生成结果摘要（SHA-256 约 2.8 GB/s，一张全尺寸面板约 80 ms）的耗时，
# 由 measure 从墙上时间和 CPU 时间里扣掉。
EXCLUDED = [0.0]


def measure(fn, repeat=3, setup=None):
    """重复执行，返回最小值、中位数、CPU/墙上比、峰值驻留增量、缺页和负载。"""
    runs, value = [], None
    for _ in range(repeat):
        if setup is not None:
            setup()
        value = None
        gc.collect()
        base, faults, pageins = task_info()
        sampler = _Sampler()
        sampler.start()
        load = os.getloadavg()[0]
        excluded = EXCLUDED[0]
        cpu, start = time.process_time(), time.perf_counter()
        value = fn()
        wall = time.perf_counter() - start
        cpu = time.process_time() - cpu
        excluded = EXCLUDED[0] - excluded
        wall, cpu = wall - excluded, cpu - excluded
        sampler.stop = True
        sampler.join()
        resident, faults2, pageins2 = task_info()
        runs.append(
            {
                "wall": wall,
                "cpu": cpu,
                "peak_delta_mb": (max(sampler.peak, resident) - base) / 2**20,
                "faults": faults2 - faults,
                "pageins": pageins2 - pageins,
                "load": load,
                "excluded_digest": excluded,
            }
        )
    walls = [r["wall"] for r in runs]
    return {
        "repeat": repeat,
        "wall_min": min(walls),
        "wall_median": statistics.median(walls),
        "cpu_ratio": statistics.median(r["cpu"] / max(r["wall"], 1e-9) for r in runs),
        "peak_delta_mb": max(r["peak_delta_mb"] for r in runs),
        "faults": int(statistics.median(r["faults"] for r in runs)),
        "pageins": int(statistics.median(r["pageins"] for r in runs)),
        "load": statistics.mean(r["load"] for r in runs),
        "runs": runs,
    }, value


class Phases:
    """逐段计时器：同一名字多次累加，用于拆分一个阶段内部的时间。"""

    def __init__(self):
        self.wall, self.count = {}, {}

    def wrap(self, name, fn):
        def timed(*args, **kwargs):
            start = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                self.add(name, time.perf_counter() - start)

        return timed

    def add(self, name, seconds):
        self.wall[name] = self.wall.get(name, 0.0) + seconds
        self.count[name] = self.count.get(name, 0) + 1

    def report(self):
        return {
            k: {"seconds": round(v, 6), "calls": self.count[k]}
            for k, v in sorted(self.wall.items(), key=lambda kv: -kv[1])
        }


def digest(*arrays):
    """逐位摘要：数组字节加形状和类型，NaN 的位模式也计入；耗时不计入 measure。"""
    start = time.perf_counter()
    try:
        return _digest(arrays)
    finally:
        EXCLUDED[0] += time.perf_counter() - start


def _digest(arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:24]


def stats(a):
    a = np.asarray(a, dtype=np.float64)
    finite = np.isfinite(a)
    return {
        "finite": int(finite.sum()),
        "sum": float(a[finite].sum()) if finite.any() else None,
    }


def environment(threads):
    """提交号、负载、开机时长、机器与线程设置。"""

    def run(*command):
        try:
            return subprocess.run(
                command, cwd=ROOT, capture_output=True, text=True, timeout=10
            ).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            return ""

    boot = run("sysctl", "-n", "kern.boottime")
    boot_seconds = None
    if "sec = " in boot:
        boot_seconds = int(boot.split("sec = ")[1].split(",")[0])
    return {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "commit": run("git", "rev-parse", "HEAD"),
        "dirty": bool(run("git", "status", "--porcelain", "--untracked-files=no")),
        "load": list(os.getloadavg()),
        "uptime_hours": None
        if boot_seconds is None
        else round((time.time() - boot_seconds) / 3600, 2),
        "cpu": run("sysctl", "-n", "machdep.cpu.brand_string"),
        "perf_cores": run("sysctl", "-n", "hw.perflevel0.physicalcpu"),
        "efficiency_cores": run("sysctl", "-n", "hw.perflevel1.physicalcpu"),
        "memory_gb": round(int(run("sysctl", "-n", "hw.memsize") or 0) / 2**30, 1),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "threads": threads,
    }
