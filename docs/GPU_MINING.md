# 因子挖掘的 GPU 粗筛（`ashare.factor.mining.gpu`）

选型与探针测量见 `docs/GPU_PROBE.md`；本文是正式实现的用法、精确性约定和测量。

## 用法

```bash
uv sync --extra gpu          # macOS 装 mlx（Metal），Linux 装 mlx[cuda12]；默认安装不带
```

```python
from ashare.factor.mining import Evolution, Split, mine
from ashare.factor.mining.gpu import GpuScreen, available

ok, why = available()                         # 只探测设备，不编译内核
result = mine(panel, split=Split.standard(), strategy=Evolution(), screen="gpu")
result = mine(..., screen=GpuScreen(margin=0.5, chunk=64, memory_limit=8 * 2**30))
print(result.summary["gpu"])                  # 设备、内核编号、粗筛 / 复算 / 判为不过线的个数、耗时
```

`screen="cpu"`（默认）与加入 GPU 之前逐位相同。GPU 不可用（没装 MLX、看不到 GPU、内核编译失败）时
`mine(screen="gpu")` 抛 `GpuUnavailable`，不退回 CPU。

## 精确性约定

两阶段：GPU 上 float32 只决定一个候选送不送 CPU；送去的由 CPU 按 float64 复算并照常判定（过线的回测）。
进台账、统计检验和结果里的数字全部来自 CPU。

- **门槛**：GPU 训练段 t 值 ≥ `min(screen_t, z₁₋α) − margin` 的送 CPU（`gate_threshold`）。
  `margin` 默认 `GPU_SCREEN_T_MARGIN = 0.3`：探针在真实面板一代 256 个上测得三种 GPU 写法的最大 t 误差
  0.25（MLX 自定义内核 0.062），0.3 大于全部实测。z₁₋α 一项保证训练段 IC 单侧 p ≤ α 的候选也全部复算——
  粗筛掉的候选的 p 值照样进 Holm / BH / BY 计数，Student t 的单侧 p 不小于正态的，t < z₁₋α 的 p 一定大于 α，
  所以不复算的候选不改变任何计数。α ≥ 0.5 时全部复算。
- **一律交给 CPU 的**：GPU t 不是有限数；算子或参数 GPU 不支持（`ts_median`、`min_periods ≠ window`、
  整数或广播字段、常量根）；某个乘除结果超出 float32 安全范围（有限值绝对值 > 1e15、非零值 < 1e-15、非零
  相乘下溢成 0，字段上传时同样检查）；排序主要由舍入决定（每天排好序的因子值里，非零相邻差不超过 1e-6 倍
  绝对值的占比 > 0.2，例如数学上恒为常数的 `ts_corr(low, -low)`：float32 与 float64 的排序都是噪声，小面板上
  t 差 1.24）；GPU 上出错。
- **台账**：没送复算的记一次 `stage="gpu_screen"`、`metrics=None` 的试验，`gpu` 字段只有设备、内核编号
  （`KERNEL_ID`，内核正文的摘要）、边际、门槛、决定和原因，没有任何 float32 数字；之后复用台账时这些记录
  按当次的设置在 CPU 上重算。送了复算的记录与纯 CPU 路径相同，只多一个 `gpu` 字段。
- **可复现**：每个线程的累加顺序、线程组内规约顺序固定，平均秩对排序是否稳定不敏感；同一设备上重复跑逐位相同。
  跨设备（Metal 与 CUDA）末位可能不同，不影响口径。

语义与 panelops 对齐（`packages/panelops/SPEC.md`）：非有限即缺失、窗口含今天且要满窗、常数窗口逐位判定
（标准差恰为 0，z 分、相关、偏度为缺失）、平均秩、Rank IC 在交集上分别求秩。

## 结构

| 模块 | 内容 |
|---|---|
| `kernels.py` | 内核正文（一份，`source(body, "metal" / "cuda")` 换占位符）：窗口算子（每格一个线程）、相关、EMA（每列一个线程）、延迟与差分、逐元素算子链解释器、一天一个线程组的 Rank IC；`KERNEL_ID` |
| `backend.py` | `Backend` 接口与 `MLXBackend`（`mx.fast.metal_kernel` / `mx.fast.cuda_kernel`）；`probe()` 运行时探测，失败抛 `GpuUnavailable` |
| `screen.py` | `Screener`：字段只上传一次、整批共享子树（`engine._schedule`）、按设备内存分批（`engine._split`，每张面板 4·T·N 字节，默认预算为可用内存的一半）、逐元素链融合计划、只取回日 IC；`pipeline`：GPU 粗筛第 k+1 块时后台线程在 CPU 上复算第 k 块（默认每块 64 个） |
| `search.py` 的 `mine(screen=...)` | 门槛、复算、台账字段、`summary["gpu"]` |

`mine` 里 GPU 只算到验证段起点（`rows=v`）：粗筛只看训练段的 t 值。

### 内核要点

- **窗口算子**：每格一个线程（T·N 个线程），窗口内按「今天、昨天……」顺序直接求和，不用前缀和（float32 前缀和
  在 2430 行上误差太大）。偏度先除以标准差再立方（离差的立方在 1e13 量级上就溢出）。
- **逐元素链**：一个预编译的解释器内核，程序（后缀指令）作为数据传入，所以任何形状的算子链都不用重新编译；
  一个线程一格、寄存器栈求值，中间结果不落显存，同时做范围检查。只有一个消费者、消费者也是逐元素的节点并进
  消费者的程序；共享子树仍只算一次。
- **Rank IC**：一天一个线程组（1024 线程、32 KB 共享内存，截面 ≤ 8192 只）。因子值在共享内存里双调排序
  （步长 < 8 在寄存器里换，8–255 用 SIMD shuffle，≥ 256 才过栅栏），二分出并列组求平均秩；收益只在整面板排序
  一次，每个因子只做「交集有效」的前缀和；Pearson 在组内规约。不写任何 (T, N) 中间面板，不按因子调 argsort。
  截面超过 8192 只时退回 argsort 写法。

## 测试

`test/factor/gpu/`（没装 MLX 或没有 GPU 时 GPU 部分 skip，CPU 部分照跑）：

| 文件 | 守住什么 |
|---|---|
| `test_gpu_differential.py` | 每个算子、逐元素链、Rank IC 对 panelops float64：缺失位置逐格一致；容差（`MEAN_TOL` 1e-5、`SPREAD_TOL` 1e-3、`DELTA_TOL` 3e-3、排名不同的格子 ≤ 0.1%、Rank IC 日值 ≤ 1e-3，探针实测放宽一个量级）；数据含停牌、上市前、退市后、常数窗口、大量并列、正负无穷、零；截面宽度跨 32、1024、8192 |
| `test_gpu_screen.py` | 整代粗筛：t 值误差 × 1.2 ≤ 边际（两块合成面板上 300 个随机表达式）；共享、分批、逐个求值逐位相同；重复跑逐位相同；不支持、超范围、舍入并列都交给 CPU；`rows` 前缀与整段一致 |
| `test_gpu_mine.py` | `screen="cpu"` 与改动前的冻结摘要逐位相同（`fixtures/mining_cpu_digest.json`，只在生成它的平台上比）且不加载 MLX；选择不变性（四组策略、种子、门槛：候选、适应度、入选、汇总、台账、指纹与收益序列文件全同，且 GPU 真的判掉了一些、也送了一些）；GPU 不可用时报错；台账里没有 float32 数字、之后可在 CPU 上复用；GPU 给不出有限 t 时一律复算 |

变异检查（每个都先确认能被抓到）：标准差的 ddof、Rank IC 的平均秩、取大的缺失传播、门槛的符号、复算判定、
z₁₋α 一项、CPU 路径的计数、舍入并列检查、范围检查、非有限 t 不复算——共 10 个，全部被抓到。

## 性能（本机 Apple M5 Pro，Metal）

真实面板 2016-01 至 2025-12（2430 天 × 5923 只），float32 每张面板 57.6 MB。本机实测设备内存带宽 265 GB/s
（1 GB 数组 `a + 1`）。机器同时有别的任务（1 分钟负载 5–8），测量都 `nice -n 15`、一次只跑一组。出处
`bench/gpu/results/screen_profile_metal_real.json`、`mine_e2e_real.json`。

**整代粗筛**（`bench/gpu/screen_profile.py`，探针的真实种群 256 个，不足的用默认空间随机表达式补；整段 2430 行）：

| 一代 | 耗时 | 每个表达式 | 按带宽算的下限 | 设备内存峰值 |
|---|---|---|---|---|
| 256 | 3.0 s | 11.7 ms | 0.39 s | 4.1 GB |
| 1024 | 13.0 s | 12.7 ms | 1.42 s | 7.7 GB（分 2 批） |
| 4096 | 58.8 s | 14.4 ms | 5.34 s | 8.1 GB（分 7 批） |

探针的同一代：CPU 4 线程 26.1 s，探针 GPU 写法 6.0 s。合入审查时用全部核心复测 CPU（14 线程，机器负载约 14）：11.8 s，所以对满核 CPU 约快 4 倍。代变大并不摊薄：每个表达式的 Rank IC 是固定开销、
占大头，共享子树只省求值（256 个里去重后 534 个算子节点，融合后要落显存的 517 个）。

**分阶段**（256 个，每个节点、每个根同步计时）：

| 阶段 | 秒 | 实际带宽（按「读一遍输入、写一遍输出」的最少字节） |
|---|---|---|
| 字段上传（一次，8 个字段） | 0.18 | — |
| 求值（全部算子） | 0.67 | 105 GB/s，峰值的 40% |
| 　其中 ts_corr / ts_zscore / ts_std | 0.14 / 0.12 / 0.05 | 窗口内重复读，算力为主 |
| Rank IC（256 个根） | 2.03 | 16 GB/s；每个约 8 ms，其中排序约 4.3 ms、二分求秩约 1.9 ms（单独测内核） |
| 取回日 IC | 0.003 | — |

改动过程（同一代 256 个）：照搬探针内核并加上范围检查 9.1 s → Rank IC 改成一天一个线程组 7.9 s → 范围检查并进
逐元素链内核 3.0 s → 窗口算子改为每格一个线程 2.7 s。把多个因子叠起来一次 argsort 没有用：MLX 对
(2430, 5923) 排一次 6.0 ms，叠 16 个平均每个 6.07 ms，排序本身是算力瓶颈，所以改为不需要 argsort 的
单线程组内核。

**`mine` 端到端**（`bench/gpu/mine_e2e.py`：`Split.standard()`、Evolution(256, 1 代)，即初代随机 256 个 +
一代后代 256 个，适应度默认要回测，CPU 4 线程）：

| | 初代 | 第 1 代 | 合计 | 粗筛掉 / 回测 |
|---|---|---|---|---|
| CPU 粗筛 | 41.0 s | 53.3 s | 95.4 s | 384 / 128 |
| GPU 粗筛 + CPU 复算 | 22.0 s | 36.7 s | 59.6 s | 384 / 128 |

GPU 一共粗筛 512 个用 6.3 s（只算到验证段起点）；送 CPU 复算 138 个（其中 8 个是舍入并列），复算加回测共
55.7 s（后台线程，与 GPU 粗筛重叠）；入选、显著性计数、回测名单与纯 CPU 完全相同。端到端只快 1.6 倍，因为
时间已经主要花在过线候选的 CPU 回测上（每代 25% 过线），GPU 帮不上。

**离下限还差多少、瓶颈在哪**：粗筛 3.0 s 对按带宽算的 0.39 s，差 7.7 倍，其中 Rank IC 占 2.0 s。Rank IC
是算力瓶颈不是带宽瓶颈：每天约 5900 个值的排序在一个线程组里做（双调排序 91 步，其中 15 步要过栅栏），再加
每个值两次二分。再往下压的办法和代价：

- 排序时把列号一起带着走，排完按位置直接得到秩，省掉二分（约 1.9 ms/个，预计 Rank IC 再快 20%）；共享内存
  装不下键和列号两份，过栅栏的步骤要拆成两趟交换，内核复杂度翻倍，约半天加测试；
- 先把当天有效的值压到前面、按有效个数（58% 的天不超过 4096）选排序长度，预计排序省 25%，同样约半天；
- 求值部分已到带宽的 40%，剩下主要是窗口算子在窗口内的重复读，改成滑动更新会换来 float32 累积误差，探针
  已经否掉。
- 端到端要再快，得减少 CPU 回测（例如每代只回测前 k 个，见 `GPU_PROBE.md` 第六节的前 k 名模式）——那会改变
  选择结果，不在本次「选择不变」的范围内，需要用户定。

## CUDA

本机没有 N 卡。在一台 NVIDIA Linux 机器上（x86_64 或 aarch64、glibc ≥ 2.35、驱动 ≥ 550、算力 ≥ SM 7.5、
Rust stable、uv），仓库根目录：

```bash
bash bench/gpu/verify_cuda.sh               # 合成面板（与真实面板同规模）
REAL=1 ASHARE_DATA_DIR=... bash bench/gpu/verify_cuda.sh   # 另测真实面板
```

它记录显卡、驱动、glibc；`uv sync --all-packages --group dev --extra gpu`；确认 MLX 在 GPU 上、全部内核能编译；
跑 `test/factor/gpu`（差分、整代粗筛、选择不变性、确定性）；合成面板 256、1024 个一代的计时与分阶段耗时；
汇总写到 `bench/gpu/results/verify_cuda.json`（步骤状态、测试计数和失败名单、带宽、耗时），任何一步失败非零退出。

预期：测试全过（冻结摘要那 4 个 skip）；每代耗时取决于显存带宽和排序算力，数据中心卡应明显快于本机。

可能与 Metal 不同的地方：

- 内核正文没在 NVRTC 上编译过。占位符换成 CUDA 写法（`blockIdx`/`threadIdx`、`__shared__`、`__syncthreads()`、
  `__shfl_*_sync`、`__any_sync`、`isfinite`、`sqrtf`、`logf`），假定 warp 为 32；第一次跑可能要修语法。
- 若 MLX 的 CUDA 编译开了快速数学（`--use_fast_math`），`isfinite` 可能被优化掉，差分测试的缺失位置会不一致——
  这正是差分测试要抓的。
- 独立显存：字段经 PCIe 上传一次（8 个 float32 字段约 0.46 GB）；设备内存预算取 `device_info()` 给出的一半。
