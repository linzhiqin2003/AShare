# panelops

独立的 T × N 面板算子库，Python 运行依赖只有 NumPy，不导入 ashare、panelbt 或 paperbroker。Rust 内核使用 Rayon 并释放 GIL，所有正式代码禁止 unsafe；许可证 Apache-2.0。

在仓库根目录运行 `make dev`。需要 Python 3.13、uv 和 Rust 编译器；也可用 maturin 构建 wheel。

```python
import numpy as np
import panelops as po
x = np.array([[10., 20.], [11., 19.], [12., 21.]])
r = x / po.ts_delay(x) - 1
score = po.cs_rank(po.ts_mean(r, 2, min_periods=1), normalize=True)
targets = po.targets(score, k=1)
```

- 时序：delay、delta、sum、mean、std、var、zscore、median、mad、count、sumif、ffill、min、max、rank、quantile、skew、kurt、product、argmin/argmax、idxmin/idxmax（并列默认取最近一次，`ties="earliest"` 取最早一次，同 `numpy.argmax`）、corr、cov、decay_linear、ema、时间回归 slope/residual/rsquare、双序列 regression_slope/regression_residual，统一加 `ts_` 前缀。
- 截面：rank、zscore、winsor_quantile、winsor_mad、group_demean、neutralize、scale、demean，统一加 `cs_` 前缀。
- 逐元素：`elementwise(op, ...)` 支持 gt/ge/lt/le/eq/ne、and_/or_/not_、where、maximum/minimum、sign、power/signed_power、sqrt、exp、clip、is_nan、fill_nan。
- 评估：ic、rank_ic、rank_ic_many（多个因子共用一次收益排序）、factor_autocorr、group_returns、group_turnover、targets。

输入严格要求 C 连续 float64 二维数组；股票池掩码为同形状 bool，行业编号为 int64。类型或布局不符报错，不隐式转换。不修改输入，每次返回新数组。NaN 和无穷视为缺失；掩码同时作用于历史观测和当日输出。窗口包含当天，默认 min_periods 等于窗口大小。不允许并发修改正在计算的输入。

排名取平均名次；标准差默认 ddof=1，截面标准化默认 ddof=0；零方差相关和标准化返回 NaN。EMA 是忽略缺失的递推平均，历史长度无限，min_periods 统计从开头以来的全部有效样本。中性化带截距，支持加权、共线暴露、单只股票行业。完整语义见英文 `SPEC.md`。

收益与分数由调用方对齐，评估和目标权重均不做日期移位。稀疏目标为 days、indptr、cols、weights 四个数组，支持空目标行；不依赖回测包。

性能表见 `docs/PANELOPS_BENCHMARKS.md`，输出以外的增量内存、实际参与运算的线程数及环境元数据在 `bench/panelops/results/`。基准对比 NumPy、bottleneck、pandas、polars 宽表/长表、Rust 单线程/默认线程；当前机器有其他任务并行，应在空闲时用 `make panelops-bench` 重测。没有对应算子的库标为不可用；pandas rolling.apply 回调单独注明。仓库只保留一份紧凑冻结结果，运行输出写到被忽略的 `.cache/panelops/`。`make panelops-bench` 重跑代表配置并比较；完整矩阵用 `bench/panelops/run.py --full`。

`panelops.build_id` 在构建时对 Rust 内核和 Python 源码求 SHA-256，不靠手工版本号识别实现变更。

`make panelops-crosscheck` 运行 pandas/alphalens 核对；`make panelops-mutate` 做真实源码变异并恢复源码、重跑检查。`set_num_threads(1)` 设单线程，`set_num_threads(0)` 恢复可用 CPU 数；不同线程数逐位一致。

**与 bottleneck 的精度对比。** `ts_sum`、`ts_mean`、`ts_std`、`ts_var`、`ts_sumif` 保留补偿求和、事件重建和误差界；把依赖数据的分支改为位选择后，单线程速度与 bottleneck 持平（4000×5500、W=20、缺失 5%，忙机交替测：sum 76.3 对 76.3 ms，std 89.0 对 92.0 ms，sumif 94–98 对 113.6 ms；默认线程 12–17 ms）。在 `test/panelops/test_exact_reference.py` 的对抗语料上与精确有理数比较，约 5.3 万个有值格子里 bottleneck 超出 2e-10 容差的：sum 7344、std 12808、var 11733、sumif 7271 格，去掉 2^200 那一列后相对误差仍可达 1e23；panelops 为 0。1e6 连续 200 行后转为 0.1，bottleneck 在全是 0.1 的窗口上给出 0.014 的标准差，panelops 给 0。

**精确的滑动求和。** `ts_sum`、`ts_mean`、`ts_sumif` 在退化窗口上给精确值（全为 0 的窗口给正零，全相等的窗口均值恰为该值）。不超过 3 行的窗口逐窗口直接求和，结果只取决于窗口本身。更长的窗口默认仍是增量算法，最后一位可能随更早的行变化；需要两个相同的窗口给出逐位相同的和时（例如对两个窗口和求相关之前）传 `exact=True`：

```python
s = po.ts_sum(x, 20, exact=True)   # 每个窗口的正确舍入和
```

<!-- BENCHMARK_SUMMARY:START -->
## 实测摘要

4000×5500，窗口 20，缺失 5%；三次中位数，毫秒。完整线程与内存口径见项目基准文档。

| Operator | Fastest single baseline | Rust 1 | Rust default |
|---|---:|---:|---:|
| ts_mean | bottleneck: 72.34 | 66.60 | 16.82 |
| ts_std | bottleneck: 75.91 | 80.42 | 16.06 |
| ts_quantile | bottleneck: 612.83 | 490.54 | 51.49 |
| cs_neutralize | numpy_1: 3395.70 | 311.76 | 45.98 |
| rank_ic | polars_1: 1129.12 | 517.21 | 55.65 |
| targets | numpy: 514.03 | 254.42 | 29.75 |
<!-- BENCHMARK_SUMMARY:END -->
