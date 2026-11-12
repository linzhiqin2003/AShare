"""阶段 1：打开面板、读取字段、内存映射缺页与内存带宽参照。

全量建面板、追加一天和校验不在这里重跑（耗时与写盘量大），数字取自
docs/DATA_NOTES.md，时间去向在 docs/PERF_PROFILE.md 里按代码分析。
"""

import fcntl
import os

import numpy as np
import panelops
from common import digest, measure

import ashare.data


def run(ctx):
    cases = []
    root = ctx["data"]
    data = ashare.data.open(root)
    fields = list(data._meta["fields"])
    path = data._path
    nbytes = sum((path / f"{name}.npy").stat().st_size for name in fields)

    def case(name, stats, scale, summary=None):
        cases.append({"case": name, "scale": scale, **stats, "summary": summary})

    stats, _ = measure(lambda: ashare.data.open(root), ctx["repeat"](5))
    case("open", stats, {})
    stats, panel = measure(lambda: data.panel(fields), ctx["repeat"](5))
    case("panel_handle_all_fields", stats, {"fields": len(fields), "bytes": nbytes})
    shape = panel.shape

    close = path / "close.npy"
    size = close.stat().st_size

    def first_touch():
        # 每次新建映射：文件已在页缓存，测的是映射缺页（软缺页）开销。
        return float(np.load(close, mmap_mode="r").sum())

    stats, total = measure(first_touch, ctx["repeat"](3))
    case("mmap_first_touch_close", stats, {"bytes": size}, {"sum": total})
    mapped = np.load(close, mmap_mode="r")
    mapped.sum()
    stats, _ = measure(lambda: float(mapped.sum()), ctx["repeat"](3))
    case("mmap_warm_touch_close", stats, {"bytes": size})
    stats, _ = measure(lambda: np.load(close), ctx["repeat"](3))
    case("np_load_read_close", stats, {"bytes": size})

    def uncached():
        # F_NOCACHE 尽量绕过页缓存；已缓存的页仍可能命中，只作冷读的下界参考。
        fd = os.open(path / "volume.npy", os.O_RDONLY)
        try:
            fcntl.fcntl(fd, 48, 1)
            buffer = bytearray(64 << 20)
            view, total = memoryview(buffer), 0
            while count := os.readv(fd, [view]):
                total += count
            return total
        finally:
            os.close(fd)

    stats, _ = measure(uncached, ctx["repeat"](2))
    case("nocache_read_volume", stats, {"bytes": (path / "volume.npy").stat().st_size})

    pe = panel["pe_ttm"]
    stats, converted = measure(
        lambda: np.ascontiguousarray(pe, dtype=np.float64), ctx["repeat"](3)
    )
    case(
        "float32_to_float64_pe_ttm",
        stats,
        {"shape": shape},
        {"digest": digest(converted)},
    )
    del converted

    source = np.ascontiguousarray(panel["close"])
    target = np.empty_like(source)
    target.fill(0.0)
    stats, _ = measure(lambda: np.copyto(target, source), ctx["repeat"](5))
    case("memcpy_prefaulted_1thread", stats, {"bytes": source.nbytes})
    stats, _ = measure(lambda: np.array(source, copy=True), ctx["repeat"](5))
    case("memcpy_fresh_allocation_1thread", stats, {"bytes": source.nbytes})
    stats, _ = measure(lambda: np.empty_like(source).fill(1.0), ctx["repeat"](5))
    case("fresh_allocation_fill", stats, {"bytes": source.nbytes})
    stats, _ = measure(
        lambda: panelops.elementwise("add", source, 0.0), ctx["repeat"](5)
    )
    case("parallel_read_write_add0", stats, {"bytes": 2 * source.nbytes})
    return cases
