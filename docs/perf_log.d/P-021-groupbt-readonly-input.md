# P-021 分组回测只读 mmap 因子免拷贝

- 提交：未提交（基于 `a5ea630`）。
- 手段：减少分配。连续 float64 因子窗口只有在底层缓冲区也不可写时才直接借用；仅设
  `flags.writeable=False` 的 NumPy 数组仍复制，防止预先存在的可写别名污染结果。
- 基准：真实数据，TOP1800 样本内带 TOP800，加验证不带子池；行业中性化，记录持仓。
  同进程交替 7 次，冻结旧 Python 和 Rust 包与当前包使用相同 NumPy；不含读面板和表达式。

| 项目 | 之前 | 之后 |
|---|---:|---:|
| 单组中位 | 275.9 ms | 272.2 ms |
| tracemalloc 分配峰值 | 247.1 MiB | 148.3 MiB |

耗时变化很小，保留理由是明确省掉约 99 MiB 的因子窗口副本。分配量不是进程 RSS，
不含 Rust 分配和 mmap 驻留，不能当作整体内存指标。

验证：只读 mmap、可写 mmap、copy-on-write mmap、只读 NumPy 视图的可写别名、
非连续输入均有测试；`memoryview.toreadonly()` 也必须继续追到真实所有者，不能把只读
包装误当作不可写缓冲区。所有公开输出和持仓逐位比较。仅检查 flags，或在只读
memoryview 处停止追溯，两种变异均被别名测试抓到。最终全量与平台验收见分组共享报告。

复跑：`python -m bench.groupbt.shared --data real --workers 1 --scenario single -o .cache/groupbt-fusion/no-copy-paired.json`。
参考包须先保存到该工具的 `--reference` 目录，真实信号和测量结果只写 `.cache/`。
