# 因子挖掘 GPU 加速探针（2026-11-14）

> 正式实现见 `docs/GPU_MINING.md`（`ashare.factor.mining.gpu`）。本文是选型时的测量，数字保留原样。

测量与选型，不是正式实现。代码在 `bench/gpu/`，结果在 `bench/gpu/results/`（下文每个数字后面括号里是出处文件）。
本机 Apple M5 Pro（15 核、24 GiB 统一内存），同时有别的任务：所有测量用 `nice -n 15`、CPU 对照 4 线程、不并行跑，负载如实记在每个结果文件的 `load*` 字段里（本轮 1 分钟负载约 4–14）。

## 结论

| 问题 | 结论 |
|---|---|
| 1. 时间花在哪 | 现行 CPU 粗筛一代 256 个 26.1 s：求值 14.0、Rank IC 12.1。算子里 `ts_corr` 一项 5.1 s |
| 2. CPU 整代合图 | 子树重复率 29%，整代一张图只快 9%（23.8 s），峰值 72 张面板、驻留 8 GB。不值得单做 |
| 3. GPU | 直接用框架算子写比 CPU 还慢（MLX 58 s、PyTorch 48 s）；滑动窗口和 Rank IC 写成自定义内核后 MLX 6.0 s、PyTorch 7.7 s，是 4 线程 CPU 的 4.4 倍 / 3.4 倍 |
| 4. float32 | 前 10%、前 32 名单与 float64 完全一致；门槛 0.5–3.0 没有一个跨线。误差最大的是数值退化的表达式。建议门槛下放 0.3 |
| 5. CUDA | 代码路径已写好（MLX 自定义内核同一份正文，CUDA 由 `mx.fast.cuda_kernel` 编译），本机测不了，一条命令 `bash bench/gpu/run_cuda.sh` 复现 |
| 6. 路线 | 推荐 MLX + 自定义内核，一套接口覆盖 Metal 与 CUDA；PyTorch 只作 CUDA 不行时的备选。整代收益取决于过线率，见第六节 |

## 一、现行做法的时间分解

真实面板 2016-01 至 2025-12，2430 天 × 5923 只。种群是真实 `Evolution(population=256, generations=2, seed=0)`（适应度 = 粗筛 t 值）第 2 代的 256 个后代，全部是交叉、变异的产物，平均 4.7 个节点（`population_real.json`）。

| 一代 256 个，4 线程，三次中位数 | 秒 |
|---|---|
| 读字段（一次） | 0.48 |
| 准备股票池、h=5 标签（一次） | 0.99 |
| 表达式求值（现行 `evaluate_many`，默认内存上限分 6 批） | 14.0 |
| Rank IC（每个 47 ms） | 12.1 |
| **一代合计** | **26.1** |
| 生成种群时的三代实际耗时 | 29.2、27.6、27.3 |

出处 `cpu_profile_real.json`。参照：`MINING_NOTES.md` 第七节在 15 线程下粗筛 100 个随机表达式 5.09 s，折合 256 个约 13 s（种群不同，只作量级参照）。

整代一张图逐节点计时（`median_ops_seconds`，一代合计 12.0 s）：

| 算子 | 秒 | 节点数 |
|---|---|---|
| ts_corr | 5.10 | 80 |
| ts_rank | 1.22 | 48 |
| ts_decay_linear | 1.17 | 33 |
| ts_zscore | 0.98 | 48 |
| ts_min / ts_std / ts_max | 0.71 / 0.69 / 0.52 | 33 / 33 / 25 |
| ts_mean | 0.36 | 21 |
| 其余（逐元素、延迟、截面排名） | 1.3 | 214 |

## 二、CPU 整代合图

| | 值 |
|---|---|
| 算子节点：逐个求值 / 整代去重后 | 753 / 534（重复 29%） |
| 逐个 `evaluate` | 27.6 s |
| 现行 `evaluate_many`（6 批，批内共享） | 26.1 s |
| 整代一批（全共享） | 23.8 s，峰值 72 张 float64 面板，进程驻留 8.0 GB |

合成数据同规模：35.9 / 33.4 / 30.6 s（`cpu_profile_synthetic.json`）。共享只省求值那一半里的一部分，Rank IC 一点不省；为 9% 付 8 GB 内存不划算。

## 三、GPU 两条路线（Metal 实测）

算子：滑动均值、标准差、z 分、最大最小、时序排名、线性衰减、滑动相关、差分、延迟、截面排名与标准化、逐元素，以及逐日 Rank IC；覆盖默认搜索空间的全部算子。语义按 `panelops`：非有限即缺失、窗口含今天且要满窗、常数窗口逐位判定（std 恰为 0，z 分、相关为缺失）、平均秩。求值顺序与释放规则照搬 `engine._schedule`（整代共享子树），每个根在设备上做股票池遮蔽和 Rank IC，只把 256 × 2430 个日 IC 取回主机，t 值在 CPU 上用 float64 算。

两种写法：

- 框架算子（`gpu_ops.py`）：窗口算子写成 W 个行切片逐项相加，排名用排序 + 前后累积极值，全用 MLX / PyTorch 自带算子。
- 自定义内核（`fused.py`）：每个窗口算子一个内核、一趟读写；Rank IC = 因子一次 argsort + 二分求平均秩 + 收益的排序整段只做一次（交集上的秩由累计个数换算）+ 每天一个 SIMD 组做 Pearson。同一份内核正文给 MLX 的 `metal_kernel`、`cuda_kernel` 和 PyTorch 的 `torch.mps.compile_shader` 用。

| 一代 256 个（真实面板） | 每代中位数 | 首代（含编译） | 字段上传（一次） | 设备内存峰值 | 重复跑逐位相同 |
|---|---|---|---|---|---|
| CPU 现行，4 线程 | 26.1 s | — | — | 进程 8.0 GB（整代一批时） | 是 |
| MLX，框架算子 | 58.1 s | 57.0 s | 0.13 s | 7.3 GB | 是 |
| PyTorch MPS，框架算子 | 48.1 s | 51.4 s | 0.16 s | 驱动 10.0 GB | **否** |
| MLX，自定义内核 | **6.0 s** | 6.05 s | 0.12 s | 4.7 GB | 是 |
| PyTorch MPS，自定义内核 | 7.7 s | 7.7 s | 0.37 s | 驱动 5.0 GB | 是 |

出处 `gpu_mlx_gpu_real.json`、`gpu_torch_mps_real.json`、`gpu_mlx-fused_gpu_real.json`、`gpu_torch-fused_mps_real.json`。合成数据同规模 MLX 自定义内核 5.7 s，CPU 33.4 s（`gpu_mlx-fused_gpu_synthetic.json`）。

- 框架算子慢在内存带宽：窗口 W 就是 W 趟整面板读写。单个滑动均值 W = 5 / 20 / 60：切片写法 5.0 / 14.4 / 42.2 ms，单内核 1.0 / 1.1 / 4.3 ms，两者逐位相同（`kernel_probe_mlx_metal.json`）。
- 自定义内核后 MLX 的一代里 Rank IC 占 4.36 s（每个 17 ms），全部窗口算子合计约 0.8 s。Rank IC 的下限是每个因子一次 argsort（约 7 ms）。第一版内核每天一个线程串行求秩，Rank IC 要 13.5 s（`gpu_mlx-fused_gpu_real_serial-rankic.json`），改成排序后按格子并行才降下来。
- PyTorch 自定义内核的窗口算子比 MLX 慢 2–4 倍（z 分 1.03 vs 0.25 s）：MLX 把窗口长度等作为模板常量编译，PyTorch 的 `compile_shader` 只能当运行时参数传。
- 设备内存峰值约 4.7 GB 主要是整代共享时同时活着的中间面板（与 CPU 的 72 张同源，float32 约 4.1 GB）。
- 依赖体积（本 worktree 的 site-packages）：MLX 237 MB（`mlx` + `mlx-metal`）；PyTorch 579 MB，另带 sympy 29 MB、networkx 8 MB 等。导入耗时 MLX 0.03 s、PyTorch 1.2–1.4 s。

MLX 的 CUDA 后端（官方文档 <https://ml-explore.github.io/mlx/build/html/install.html>，2026-11-14 时为 0.32.4）：`pip install mlx[cuda12]`（或 `mlx[cuda13]`，要驱动 ≥ 580），要求 Linux、glibc ≥ 2.35、NVIDIA 驱动 ≥ 550.54.14、算力 ≥ SM 7.5。仓库 `mlx/backend/cuda/` 下有 sort、scan、indexing、reduce、自定义内核（`mx.fast.cuda_kernel`）的实现；公开 issue 里缺的主要是量化矩阵乘等与本项目无关的算子，另有安装时 glibc 版本和 `cudaMemAdvise` 导入报错的反馈。本项目用到的 argsort、take_along_axis、cumsum、cummax 与自定义内核在 CUDA 上都有实现，但都未在本机验证。

## 四、float32 对选择的影响

同一代 256 个，GPU float32 日 IC 在 CPU 上算 t 值，对比 CPU float64（`gpu_*_real.json` 的 `float32_vs_float64`、`margin_real.json`）：

| | MLX 内核 | PyTorch 内核 | 框架算子（两家相同算法） |
|---|---|---|---|
| t 值误差 中位 / 90% / 99% / 最大 | 1.4e-5 / 1.1e-4 / 0.028 / 0.062 | 1.4e-5 / 1.1e-4 / 0.056 / 0.114 | 1.4e-5 / 1.1e-4 / 0.056 / 0.250 |
| 前 26 个（10%）、前 32 个名单重合 | 100%、100% | 100%、100% | 100%、100% |
| 门槛 0.5、1.0、1.5、2.0、3.0 跨线的 | 0 | 0 | 0 |

误差最大的几个都是数值退化的表达式，float64 的结果同样是舍入噪声：`vwap / (high * (vwap / high))` 恒等于 1、`ts_corr(abs(vwap), vwap)` 恒等于 1（vwap 为正），以及对 5 日开盘价标准差再求时序排名的一个。去掉这三个，最大误差 0.050，来自 `amount / (close - vwap)` 的相消。日 IC 缺失位置不一致的 814 格全部出自前两个退化表达式。

保险边际（门槛 1.0，下放 δ 后 float64 合格的漏掉几个、多复核几个）：

| δ | 0 | 0.1 | 0.25 | 0.5 |
|---|---|---|---|---|
| 真实面板：漏 / 多复核 | 0 / 0 | 0 / 1 | 0 / 1 | 0 / 4 |
| 合成噪声面板（MLX 内核）：漏 / 多复核 | 0 / 0 | 0 / 12 | 0 / 21 | 0 / 41 |

建议 δ = 0.3：大于本轮所有后端（包括退化表达式）的最大误差 0.25，真实面板只多复核约 1 个。合成面板是纯噪声，t 值都挤在门槛附近，多复核的就多。δ 写成可配置常量，随正式实现的差分测试一起维护：每次换后端或换内核都在同一份种群上重测最大误差，δ 至少取它的 1.2 倍。

算子级误差（`check_ops_*_real.json`，真实面板最后 600 天；合成数据见 `check_ops_*_synthetic.json`）：缺失位置全部一致；相对误差（相对该输出的中位数量级）均值、最大最小、延迟、线性衰减、逐元素 ≤ 1e-6，标准差、z 分、相关 ≤ 1e-4，差分 ≤ 3e-4（相消）；时序、截面排名有 0–0.016% 的格子与 float64 不同（float64 下不相等、float32 下相等，于是并列取平均）；Rank IC 日值最大差 4e-4。

## 五、CUDA 验证步骤

一台 NVIDIA Linux 机器（x86_64 或 aarch64，glibc ≥ 2.35，驱动 ≥ 550，算力 ≥ SM 7.5，装好 Rust stable 和 uv），在仓库根目录：

```bash
bash bench/gpu/run_cuda.sh
```

它依次：打印显卡；`uv sync --all-packages --group dev --group gpu-probe-cuda`（`mlx[cuda12]` + PyPI 上默认带 CUDA 的 torch，已写进 `pyproject.toml` 和 `uv.lock`）；确认 torch 看得到 CUDA、MLX 能在 GPU 上算；三个后端的算子差分检查；CPU 现行做法与整代共享（合成数据同规模，生成种群和 float64 参考）；三种写法的整代粗筛。预期：

- `check_ops_*_synthetic.json`：所有算子 `missing_mismatch` 为 0，误差量级与本机相同；
- `gpu_mlx-fused_gpu_synthetic.json`：`bitwise_same_as_first` 全为 true，`float32_vs_float64` 里前 32 名重合 1.0、各门槛跨线 0；
- 每代耗时：本机自定义内核 5.7 s，数据中心显卡带宽高一个数量级，预计明显更快，以实测为准。

可能与 Metal 不同的地方：

- **内核编译**：`mx.fast.cuda_kernel` 走 NVRTC，正文由 `fused._prelude` 把线程下标、`isfinite`、NaN 和 SIMD 规约换成 CUDA 写法（规约用 `__shfl_down_sync`，假定 warp 为 32）。这部分本机没跑过，第一次跑可能要修语法。
- **末位差异**：nvcc 默认允许乘加融合，Metal 也会融合，但两边融合的位置不同，结果在末位上不同；SIMD 规约的树形不同也一样。同一设备上可复现，跨设备不逐位相同——这不影响口径，因为进台账的数字全部来自 CPU。
- **确定性**：MLX 没有全局开关，本探针的内核每个线程的累加顺序固定、规约树固定，排名对排序是否稳定不敏感（并列取平均秩），所以结果可复现（Metal 上 5/5 次逐位相同）。PyTorch 在 MPS 上用框架算子时重复跑**不逐位相同**（两次都不同），CUDA 上要 `torch.use_deterministic_algorithms(True)`，排序用 `stable=True`。
- **显存**：数据中心卡是独立显存，字段经 PCIe 上传一次（8 个 float32 字段约 0.46 GB）。整代共享的峰值约 4–5 GB，24 GB 的卡够用；面板再大要按显存上限分批（见第六节）。
- **PyTorch 的 CUDA 路径**：本探针只有框架算子写法（`--backend torch --device cuda`）。自定义内核在 PyTorch CUDA 上要走 `torch.compile`（Triton）或 `cpp_extension.load_inline`（要 nvcc），没有实现。

## 六、建议

**路线**：MLX + 自定义内核，作为唯一的 GPU 后端；CPU（`panelops` float64）作参考实现和唯一的数字来源。理由：本机最快（6.0 s vs PyTorch 7.7 s）；Metal 和 CUDA 用同一套 API、同一份内核正文；依赖小（237 MB vs 600 MB 以上）、导入快、结果可复现。风险是 MLX 的 CUDA 后端较新，先在云上跑 `run_cuda.sh` 验证；不行再给 CUDA 补一个 PyTorch + Triton 的实现，接口不变。

**整代收益取决于过线率**。GPU 只省掉没过线的候选的 CPU 粗筛：本轮第 2 代过线 145/256（57%，进化选出来的种群过线率高），过线的要在 CPU 上用 float64 重算（每个约 102 ms，4 线程），一代仍有约 15 s CPU 工作。所以：

- 门槛模式（t_gpu ≥ screen_t − δ）：粗筛本身 26 s → 6 s，加上复核约 15 s；把复核与 GPU 粗筛流水线重叠后一代约 15 s。
- 前 k 名模式（例如每代前 32 个送 CPU；本轮 float32 与 float64 的前 32 名完全一致）：复核约 3.3 s，一代约 6 s，比现行快 4 倍左右。代价是过线但不在前 k 的候选没有 CPU 指标（台账里照样记一次试验）。两种模式都做成选项。

**模块划分**（`ashare.factor.mining.gpu`，可选依赖 `gpu`：macOS 装 `mlx`，Linux 装 `mlx[cuda12]`）：

| 模块 | 职责 |
|---|---|
| `backend.py` | 后端接口：`upload(panel, fields, universe, returns)`、`evaluate(roots) -> 日 IC (k, T) float64`、`memory_budget()`、`device`、`kernel_id`（内核源码哈希，进台账） |
| `kernels.py` | 内核正文一份，`metal` / `cuda` 两个前缀生成器；默认搜索空间之外的算子（中位数、EMA、偏度、斜率、argmax 等）补齐，补不了的让整棵表达式退回 CPU |
| `screen.py` | 整代求值：复用 `engine._schedule` 共享子树，复用 `engine._split` 按设备内存分批（每张面板按 4·T·N 字节算，预算默认取可用显存/统一内存的一半）；数据只上传一次，每代只取回日 IC |
| `search.py` 接入 | `mine(..., gpu=GpuScreen(mode="threshold"/"topk", margin=GPU_SCREEN_T_MARGIN))`；GPU 只决定送不送 CPU；送的走现有 `fe.screen`（float64）重判 `screen_t`；没送的照样记一次试验，阶段记为 `gpu_screen`，float32 t 值只作诊断字段、不进任何统计、复用台账时按本次门槛在 CPU 上重算 |
| `test/factor/mining/gpu/` | 差分测试（见下） |

精确性：

- GPU 只做粗筛，进台账和统计检验的数字全部来自 CPU float64 复核；`GPU_SCREEN_T_MARGIN = 0.3` 写成常量（依据第四节），台账记录用的边际和 `kernel_id`。
- 差分测试逐项守住语义：每个算子在带停牌、上市前、常数窗口、大量并列、正负无穷的小面板上对 `panelops`，缺失位置必须完全一致；数值容差按第四节实测再放宽一个量级（均值类 1e-5、标准差和相关 1e-3、差分 3e-3，排名不同的格子 ≤ 0.1%），Rank IC 日值 ≤ 1e-3（退化表达式除外，单列成测试）；再加一个门禁：固定种子的整代种群上，门槛下放 δ 后 float64 合格的一个不漏。
- 可复现：每个后端测「同一输入、同一设备跑两遍逐位相同」；PyTorch 后端打开 `use_deterministic_algorithms`。
- 数值退化的表达式（恒等于常数等）在搜索空间的去重里删掉，比靠边际兜底更干净。

性能：

- 字段、股票池、标签只上传一次；每代只传表达式、取回 k × T 个日 IC；所有中间结果留在设备上。
- 整代成批、共享子表达式；按设备内存分批，批间不共享（与 CPU 的 `_split` 同一规则）。
- CPU 复核与 GPU 粗筛重叠：一代切成若干块（例如每块 64 个），GPU 粗筛第 k+1 块时 CPU 复核第 k 块。跨代不能重叠：下一代要用这一代的适应度。

工作量（机器等待 + 每个交付物的分钟数，含自测）：

| 交付物 | 分钟 | 机器等待 |
|---|---|---|
| 后端接口 + MLX 内核迁到正式模块、补齐其余算子 | 90 | 构建与测试约 5 分钟 |
| 差分测试、可复现测试、边际门禁 | 45 | 小面板，单次 < 1 分钟 |
| 接入 `mine`（两种模式、台账字段、块内流水线） | 60 | 真实面板端到端一次约 10 分钟 |
| 云上 CUDA 验证与修正（`run_cuda.sh`） | 30 | 开机、依赖与 Rust 构建、跑完约 30 分钟 |
| 若 MLX CUDA 不可用：PyTorch + Triton 后端 | 90 | 云上约 20 分钟 |
| 独立复核（两到三轮） | 60 | — |

## 没做完或做不到的

- CUDA 全部未实测；`fused.py` 的 CUDA 内核正文没有编译过。
- PyTorch 在 CUDA 上的自定义内核（Triton / load_inline）没写。
- CPU 对照只测了 4 线程（机器负载高，按要求限制）；15 线程的数字只有 `MINING_NOTES.md` 的旧参照。
- 种群只有一代（第 2 代后代）；初代随机种群的过线率会低得多，GPU 的收益会更大，没有单独量。
- 探针用整段当训练段算 t 值，与 `mine` 里只用训练段的口径不同，两边对比时口径一致，不影响结论。
