"""因子挖掘的 GPU 粗筛（可选依赖 ``gpu``：macOS 装 ``mlx``，Linux 装 ``mlx[cuda12]``）。

两阶段：GPU 上 float32 整代粗筛，只决定候选送不送 CPU；送去的由 CPU 按 float64 复算，
进台账、统计检验和对外显示的数字全部来自 CPU。用法见 docs/GPU_MINING.md：

```python
from ashare.factor.mining import mine
result = mine(panel, split=split, strategy=Evolution(), screen="gpu")
```

导入本包不加载 MLX；第一次构造 `Screener`（或 ``mine(screen="gpu")``）时才探测设备，
不可用时抛 `GpuUnavailable`。
"""

from .backend import RANGE_HIGH, RANGE_LOW, SUPPORTED, Backend, GpuUnavailable, probe
from .kernels import KERNEL_ID
from .screen import (
    DEFAULT_CHUNK,
    GPU_SCREEN_T_MARGIN,
    GpuScreen,
    Outcome,
    Screener,
    gate_threshold,
    pipeline,
)


def available():
    """(可用, 原因)：这台机器上能不能做 GPU 粗筛。只探测，不编译内核。"""
    try:
        probe()
    except GpuUnavailable as exc:
        return False, str(exc)
    return True, None


__all__ = [
    "DEFAULT_CHUNK",
    "GPU_SCREEN_T_MARGIN",
    "KERNEL_ID",
    "RANGE_HIGH",
    "RANGE_LOW",
    "SUPPORTED",
    "Backend",
    "GpuScreen",
    "GpuUnavailable",
    "Outcome",
    "Screener",
    "available",
    "gate_threshold",
    "pipeline",
    "probe",
]
