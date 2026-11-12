# panelops 与 ashare.factor 第二轮记录

本轮以独立审查报告为依据。第一轮「34 个算子全部由 Rust 默认线程胜出」只适用于多线程，不能推出单线程通过；本轮用单线程作为主要口径。最终逐项时间、额外内存、两种选用口径见 [基准表](PANELOPS_BENCHMARKS.md)，唯一冻结结果为 `bench/panelops/results/baseline.json`。日志、试验副本、逐次采样都在被忽略的 `.cache/` 中。

## 审查逐项处理

| 审查项 | 处理 | 改动及守护测试 |
|---|---|---|
| P0-1 浮点常数零方差 | 已复现并修复 | `time.rs` 维护有效值相等的运行长度；截面和直接参考相对第一个有效值中心化。std/var 精确 0，相关/偏峰度/zscore/常数自变量回归缺失；`test_exact_float_constants_and_piecewise_constants`，真实复权因子常数窗口核对 |
| P0-2 尖峰滑出后污染 | 已复现并修复 | 追踪重建后的量级与舍入误差；危险时重建当前窗口。`test_adversarial_windows`，另加多量级正负抵消的浮点展开重建测试 |
| P0-3 阈值与容差、cross 无守卫 | 已复现并修复 | 由 2e-10 比较容差留出 1e-11 内部预算；二阶、四阶、cross、残差消去分别估界，统计自实际重建起的更新数。`test_adversarial_windows` 的六档水平跳变；R34/N02/N03 变异 |
| P1-1 库名撞名 | 已复现并修复 | `[lib] name="panelops_ext"`，Python 仍 `panelops._core`；workspace 构建及全新目录检查 |
| P1-2 单线程、整面板临时内存 | 内核已重写；门槛未全部达到 | 时序按连续列块逐行读入/滑出/输出；枚举入口、单态化；不再有整份输出临时面板。逐个未达标算子的数字、尝试、原因见基准表，不能称为全部满足合并门槛 |
| P1-3 常数性能悬崖 | 已复现并修复 | 常数窗口直接给精确结果，不逐格扫描；下方保留同机前后数字 |
| P1-4 缓存旧结果、损坏 | 已复现并修复 | build.rs 把 Rust/Python 源码指纹导出为 build_id；键另含因子求值实现指纹。npy 配校验和侧车、损坏重算、原子写入、死进程 tmp 清理、命中未命中都只读；`test_cache_corruption_axes_build_id_and_readonly`、`test_stale_temporary_cleanup_preserves_live_writer` |
| P1-5 测试及存活变异 | 已补齐并重做 | 620 行对抗语料、603 行随机面板、2051 列分块、线程逐位一致、切换无缺失快路径的前视探测；原 R/P 编号映射到新内核，新增 N/E 编号，明细如下 |
| P1-6 min_periods、字段与单位 | 已修复 | 首批滚动因子 W≥20 使用 ceil(.8W)，可按名字覆盖；采用数据层 total_market_cap/turnover/dividend_yield_ttm，比例不除 100。EP 说明亏损公司缺失，股息率不将缺失视作零；`test_authoritative_fields_minimum_counts_and_registry` |
| P2-1 独立签名、layout | 已修复 | 32 个公开时序函数各有签名与文档，无关参数拒绝；layout 仅保留内部兼容入口；参数测试 |
| P2-2 product 对数误差 | 已修复 | 全部窗口按时间顺序直接相乘，W=1 逐位原值（含 -0.0、次正规数），零优先；test_single_observation_product_preserves_finite_bits 与随机/极端乘积测试 |
| P2-3 起始行影响低位 | 已明确保留 | SPEC 说明截掉前缀可改变容差内低位；周期兜底重建仍存在。修改未来必须逐位不变，专门性质测试守住 |
| P2-4 EMA min_periods | 已补文档 | 计数是开头以来的有效样本，不是最近 W 行；默认 W；lookback=None 表示无限历史；EMA 缺失与参数测试 |
| P2-5 节点早校验与默认规范化 | 已修复 | window=20.0 建节点即拒绝；先补默认值再哈希，参数排序；`test_parameters_defaults_nested_lookback_and_long_chain`、`test_clip_parameter_order_is_canonical` |
| P2-6 求值器内存与摘要 | 已修复 | 标量广播、摘要/字段/回看缓存、迭代求值、CSE 引用计数释放；800 节点链峰值保留面板数≤3，公共子表达式只算一次 |
| P2-7 只读缓存和 tmp | 已修复 | 见 P1-4；保留活跃写进程的 tmp，避免误删并发写入 |
| P2-8 截面分配、重复排序 | 已优化 | 排名使用线程本地连续键值对；去极值选择算法与暂存；自相关/换手短 lag 复用排序/标签，长 lag 限制缓存；`test_lagged_sort_reuse_boundaries_and_common_universe` |
| P2-9 README 单线程 | 已补齐 | 两个 README 摘要由冻结结果生成，同时列 Rust 1 与默认线程 |
| P2-10 笔记单线程口径 | 已改写 | 本文与基准表替代第一轮仅多线程的结论 |
| P2-11 Registry 与字段来源 | 已补齐 | `names()`、`using_field()`；FIELDS 从数据层 metadata.fields 生成快照，`field_schema.py` 检查或重新生成，46 个名称/单位与当前代完全一致；不引入文件读取到因子层运行时 |
| P2-12 按分数加权/行业内目标 | 本轮不做 | 审查明确为后续扩展；当前 CSR 目标等权，分组收益支持给定权重，行业去均值/中性化已有；不冒充目标按分数加权 |

三条 P0、六条 P1 的旧行为均已复现；没有以“复现不出来”关闭的问题。P2 中的接口/文档问题通过源码和小面板复现，P2-12 是功能边界而非缺陷。历史脚本使用的副本路径已替换为本 worktree，未改审查原件。

## 数值方案和遍历取舍

- sum/mean/count/sumif 维护按列排列的状态数组，逐行更新；补偿加法加事件触发浮点展开，512 行周期兜底。额外试验发现 `[2^200,2^100,1,-2^200,-2^100]` 的多次消去会吞掉小量，因此不能仅把已有补偿求和重新扫一遍。
- std/var/zscore 只维护所需二矩，相对锚点求差；保存最近实际重建行，不能假设样本不足期间每个周期都重建。常数判定不靠 epsilon，也不把微小但真实的方差裁成零。
- corr/cov/回归与偏峰度维护所需配对/高阶矩及误差界；cross 和残差也受守卫保护。危险窗口用稳定的直接计算重建。时间回归使用原始交易行位置，缺失不压缩时间。
- decay 对主和及补偿分别更新，维护加权误差预算；EMA/ffill 为按列递推状态。delay/delta 对连续行切片做一次 zip 遍历。
- extrema 用两趟分块前缀/后缀法（并非单调队列），因此相应变异检查的是窗口块到期、前后缀合并、并列最新位置；rank 小窗口直接比较计数；分位数 W≤64 用固定有序缓冲，长窗口用确定性 treap。
- 单线程也按最多 1024 列切块，避免宽状态冲出缓存；默认线程按 64 列对齐。无缺失/无掩码路径只去掉有效性判断，算术顺序相同。没有 unsafe。
- 截面排名排序连续 `(sortable_f64_bits, column)`，避免比较器反复跨向量取值；±0 统一排序键。中性化先做加权行业去均值，再分解 K 个暴露（FWL、归一化带主元重正交 QR），不构造行业哑变量大矩阵。共线暴露、单股/双股行业与加权正交由测试覆盖。
- 独立参考是逐窗口 Python/numpy/math.fsum，生产实现不调用参考；全部维持 rtol=atol=2e-10，未放宽断言或跳过测试。

## 新增算子和叶子

新增 32 个算子，总计 66 个（32 时序、8 截面、6 评估、20 新逐元素；原有基本算术另计）。

| 组 | 新增内容及约定 |
|---|---|
| 比较 | gt/ge/lt/le/eq/ne 输出 1.0/0.0，任一侧缺失传播；Expr.__eq__ 保持结构相等 |
| 逻辑/选择 | and_/or_/not_ 的有限非零为真，缺失传播；where 只要求条件和被选分支有效 |
| 数学 | maximum/minimum、sign、power、signed_power、sqrt、exp、clip；非法定义域或非有限结果转缺失 |
| 缺失 | is_nan 把非有限输入视为缺失；fill_nan 只在原值缺失时取替代值；股票池外始终缺失 |
| 回归/矩 | ts_rsquare 为对交易行时间回归的 R²；ts_var 与 std 同 ddof 默认 1；ts_zscore 为当日值滚动标准化 |
| 稳健统计 | ts_median 是 q=.5 的线性分位数；ts_mad 是未乘一致性系数的中位绝对偏差 |
| 条件统计 | ts_count 数有限非零，min_periods 数所有有限值；ts_sumif 数有限配对，条件假仍计入 min_periods，只是不贡献和 |
| 填充/位置 | ts_ffill 只向前、limit 为交易行龄；ts_idxmax/idxmin 为名义窗口内从 1 起位置，idx=W-arg，含缺失/前部空位，并列取最新 |
| 截面 | cs_scale 使有限值绝对和为 1，零和缺失；cs_demean 常数行精确 0 |
| 市场叶子 | field(name, broadcast=True) 接收 T 或 T×1 序列广播为 T×N；普通字段仍为 T×N。因子层接受数据层数值/布尔 dtype 并转换；panelops 自身仍严格 float64/C 连续 |

完整英文约定见 `packages/panelops/SPEC.md`。首批因子字段来自数据层权威表，价格统一使用 close×adj_factor，因子层不移动信号日期。

## 设计疑问和明确边界

| 原要求/问题 | 处理和理由 |
|---|---|
| “单线程不慢于最快基线超过 10%；做不到逐个说明” | 报告所有超标项并保留原容差；不能用默认 15 线程的成绩替代单线程门槛。实际后端选择只在 NumPy/Rust 中比较，固定选择保持线程逐位一致。 |
| “峰值内存、输出之外增量内存” | 统一记录进程 ru_maxrss 和扣除输入基线/输出后的 RSS 增量。该差分是受分配器与惰性页影响的下界估计，不冒称精确 malloc 峰值；另给所有 Rust 内核的工作区公式。精确分配峰值尚未使用分配器插桩测量。 |
| “精度不靠限定输入量级”与 float64 有限范围 | 未设置业务量级上限；对抗语料包含成交额和 1e12 尖峰。中间幂溢出 float64 时按 SPEC 可产生 NaN/inf，有限容差不是任意精度算术承诺；近数值秩阈值的 QR/SVD 差异也保留。 |
| “FIELDS 同一处定义”与“因子层不读文件” | 运行时使用从 metadata 生成的字段快照；工具只读 metadata 并验证/输出新快照，合并后可由数据层构建流程调用；不在求值器内读取文件。 |
| “缓存实现指纹”与表达式层也可改变结果 | 除扩展 build_id 外，键还含求值器/表达式源码指纹、轴内容、股票池内容；避免只修 Python 求值器仍读旧缓存。 |
| 全局 worklog 要求与本任务禁止目录外写入 | 不写工作区外日志，工作记录留在本文。OneContext 历史检索工具无法打开数据库，未修改外部技能或历史文件。 |
| CSR targets 额外内存 | 当前每个调仓日的选择向量保留候选容量，大量调仓日时额外内存较高；真实 RSS 数字与上界另列。它不影响已消除的时序整输出临时面板，后续可用两遍计数/直接填 CSR 降低。 |
| 额外要求未包含 cummax/cumsum、跨因子批量 DAG、增量尾部更新 | 本轮不扩展；当前 CSE 限于一次求值，最终结果可磁盘缓存。没有把“已有时序回归”错误计作新增算子。 |
| 两个基准规模和完整 W/缺失网格 | 重跑命令保留 --full；本轮冻结的是用户指定的代表配置，常数性能另测 1000×1000。没有声称本轮重测完整第一轮网格。 |

## 共享根文件

本轮仅再改 `Makefile:31`：`make panelops-bench` 运行输出改到忽略目录，并自动与唯一冻结基线比较。`Cargo.toml`、`Cargo.lock`、根 `pyproject.toml`、`uv.lock` 的工作区改动仍是第一轮留下的内容，本轮没有新增改动。包内 `packages/panelops/pyproject.toml` 增加构建脚本的 uv cache-key，包 Cargo.toml 改扩展库名；不涉及其他会话文件。

## 打包时额外发现的问题

独立 sdist→wheel 试验发现：maturin 会把 Python 源码移到 sdist 根的 `python/`，Cargo manifest 仍在 `packages/panelops/`。原指纹脚本漏掉这个布局的 Python 文件。本轮改为定位真实 Python 源码根、按统一逻辑路径排序后哈希，缺少 Python 源码直接使构建失败；新增 workspace/sdist 两种布局下改 Python/内核源码必须改变指纹的测试，并独立安装 wheel 核对。

macOS 裸 `cargo build --workspace` 没有动态查找 Python 符号的链接参数，会同时在现有 panelbt 和 panelops 扩展上报 undefined symbols；没有再出现 output filename collision。使用与 maturin 一致的 `RUSTFLAGS="-C link-arg=-undefined -C link-arg=dynamic_lookup"` 后 workspace 构建通过。没有为此改动其他会话的构建文件；正常 `make dev && make check` 不需要这个额外设置。

## 条件算子的补充基准

最后检查发现，正态分布条件几乎全是非零，`where` 基准只走真分支，不能代表条件选择。正式冻结值因此把 and_/or_/not_/where 的条件改为各约一半的 0/1，并保留 5% 缺失，四个算子的全部基线统一重测。平衡条件下原 Rust where 单线程 81.1 ms，NumPy 38.0 ms；随后用位选择消除不可预测分支，保持只检查被选分支的缺失语义，补 N35 变异，并重跑最终整批 Rust 基准。没有把原来偏向单分支的数字用作胜出依据。


最终边界补充：product 的单观察值特例在入口走 delay(0)，保留 -0.0 位模式；fill_nan 对无掩码的两个数组使用连续块 zip 遍历，一次验证切片边界，补跨 65536 元素块及线程逐位一致测试。N35–N38 分别破坏 where 位选择、product 单值路径、fill_nan 位选择和 fill_nan 列块边界，均要求真实测试失败。

最后仅 is_nan 增加无掩码数组的连续切片快路径。它的两项 Rust 测量单独重跑，其余算法未变，沿用此前完整批次；冻结 JSON 保留各条实际构建指纹，未把两批伪装为同一构建。N39 破坏这条快路径的有效性判定。真实核对另包含 close 对 adj_factor 的相关、协方差、斜率和残差，直接覆盖审查的常数自变量案例。

## 验证结果

最终扩展 build_id：`1ca347f61d4319b1edd90c3ee66b14d50a21d7f93e7386bf80b917d137c5475b`。工作区、全新目录、sdist wheel 源码指纹一致。

工作区与全新目录的 make dev && make check 都通过；ruff、格式、clippy、Rust 测试、Python 测试全部执行。末尾原始输出：

工作区：
```text
test/panelops/test_streaming.py ..........................               [ 79%]
test/test_engine.py ................                                     [ 83%]
test/test_inputs.py ....................................                 [ 95%]
test/test_properties.py .....                                            [ 96%]
test/test_random.py .                                                    [ 96%]
test/test_rules.py ..........                                            [100%]

============================= 324 passed in 25.91s =============================
```

全新目录：
```text
test/panelops/test_streaming.py ..........................               [ 79%]
test/test_engine.py ................                                     [ 83%]
test/test_inputs.py ....................................                 [ 95%]
test/test_properties.py .....                                            [ 96%]
test/test_random.py .                                                    [ 96%]
test/test_rules.py ..........                                            [100%]

============================= 324 passed in 31.88s =============================
```

## 对抗语料的逐算子误差

620×12 面板、W=2/5/20/60/250，全部时序算子均与逐窗口参考核对。另有多量级消去、603 行随机面板及线程/掩码快路径切换测试。下表为各窗口的最大值；容差占用是 |误差|/(2e-10+2e-10·|参考|)，≤1 才通过。绝对误差和相对误差的最坏点未必相同。偏度真值接近零，因此最坏相对误差 3.67 不代表违反绝对容差。

| 算子 | 最大相对误差 | 最大绝对误差 | 最大容差占用 |
|---|---:|---:|---:|
| ts_delay | 0 | 0 | 0 |
| ts_delta | 0 | 0 | 0 |
| ts_sum | 2.67943e-12 | 0.000976562 | 7.40453e-05 |
| ts_mean | 2.67939e-12 | 6.10352e-05 | 7.40443e-05 |
| ts_std | 2.20787e-13 | 0.000329018 | 0.00110226 |
| ts_min | 0 | 0 | 0 |
| ts_max | 0 | 0 | 0 |
| ts_rank | 0 | 0 | 0 |
| ts_quantile | 5.96272e-14 | 6.10352e-05 | 0.000137181 |
| ts_argmin | 0 | 0 | 0 |
| ts_argmax | 0 | 0 | 0 |
| ts_product | 0 | 0 | 0 |
| ts_corr | 1.01573e-11 | 1.35891e-13 | 0.000339728 |
| ts_cov | 1.01573e-11 | 4.1943e+06 | 0.00386572 |
| ts_decay_linear | 1.17866e-10 | 2.28882e-05 | 0.000588741 |
| ts_ema | 0 | 0 | 0 |
| ts_skew | 3.66667 | 1.37668e-13 | 0.000200416 |
| ts_kurt | 2.69458e-10 | 1.23457e-13 | 0.000335043 |
| ts_slope | 2.59214e-12 | 0.00109863 | 0.000823701 |
| ts_residual | 4.33323e-12 | 0.00250244 | 0.0198071 |
| ts_regression_slope | 1.01579e-11 | 363 | 0.00114766 |
| ts_regression_residual | 1.76051e-11 | 0.000854492 | 0.00196592 |
| ts_var | 4.41549e-13 | 3.0199e+08 | 0.00220774 |
| ts_zscore | 5.80044e-13 | 5.36238e-14 | 0.00015706 |
| ts_rsquare | 5.18441e-12 | 1.58984e-13 | 0.00039746 |
| ts_median | 5.96272e-14 | 6.10352e-05 | 0.000137181 |
| ts_mad | 3.38438e-16 | 6.10352e-05 | 1.69219e-06 |
| ts_count | 0 | 0 | 0 |
| ts_sumif | 5.94859e-13 | 0.000976562 | 0.000129002 |
| ts_ffill | 0 | 0 | 0 |
| ts_idxmin | 0 | 0 | 0 |
| ts_idxmax | 0 | 0 | 0 |

## 真实面板只读核对

只读 mmap 当前代 `8c770ce48a062445b10440733250765443eee612ffed243ff6ea4b39b2340568`，原面板 [4824, 5923]，随机种子 0 抽 600 列；adj_factor/close/amount、32 个时序算子、W20/W250，加 close 对 adj_factor 的 8 组特定配对核对，共 200 组通过。取 22 个日期做直接逐窗口对拍，包含 127/128、249/250、511/512 及整个历史的等距点；原生计算运行完整历史。不是逐格直接参考覆盖全部 4824 天。

最大的容差占用 0.0534295。完整历史另检查全部有限常数窗口，adj_factor 的 1,484,012 个 W20 窗口全部满足精确 0/缺失约定。真实观察值没有入库。修复前使用当时当前代 694c70…，修复后数据层已发布新代；不把两代统计量差异当作性能或覆盖率变化。

## 外部交叉核对

1060 组基线适配器数值核对通过；pandas 224 组通过，最大绝对误差 5.73452e-12。Alphalens 0.5.1：20 天×30 证券，rank IC 最大绝对误差 5.55112e-17，五组收益 1.11022e-16，五组换手全部通过。

差异逐条约定：Alphalens IC 是 Spearman（对应 rank_ic），Pearson 单列为 ic；分组收益比较设置 demeaned=False；qcut 遇重复边界会报错，本库按列号打破并列；样本数不能被 q 整除时插值边界与 ordinal 分组不同；缺失收益不改变本库分桶，仅影响组内均值；pandas 偏度/峰度默认无偏修正，本库按 SPEC 使用有偏总体矩。没有改变断言去迁就差异。

## 常数性能悬崖前后

1000×1000、单线程，单位 ms；修复前为同一审查脚本预热后单次计时，修复后为三次中位数，两次负载不同。代表配置正式基准全部为三次中位数。

| 算子/窗口 | 前：随机/常数/分段 | 后：随机/常数/分段 |
|---|---:|---:|
| ts_std / 20 | 16.1 / 332.4 / 311.0 | 2.9 / 2.7 / 2.7 |
| ts_std / 250 | 23.2 / 971.9 / 28.1 | 3.1 / 2.4 / 2.8 |
| ts_corr / 20 | 16.4 / 382.6 / 358.1 | 14.3 / 10.3 / 9.9 |
| ts_corr / 250 | 23.7 / 1379.2 / 31.2 | 18.9 / 8.9 / 19.7 |
| ts_skew / 250 | 17.7 / 756.6 / 42.5 | 15.8 / 7.4 / 54.1 |
| ts_slope / 250 | 23.0 / 1356.6 / 30.2 | 35.7 / 8.6 / 18.4 |
| ts_mean / 250 | 5.4 / 5.3 / 5.3 | 1.9 / 1.9 / 2.0 |

## 变异检查

114 处，全部 KILLED；真实修改源文件，原生/包装层修改均重建并确认 build_id 改变。先 make check 全绿，再测试变异；恢复后源字节哈希与 build_id 回到对照，make check 全绿。分阶段执行：先 99 处，发现 P09 存活后补测试；最终改写影响到的内核重新执行相关变异，并增加 N27–N39 和构建指纹变异。各条源指纹保存在冻结结果，不把不同阶段伪装成同一构建。

旧审查 R02/R03/R36 的“永不重建”映射为同时禁用周期与事件重建；仅禁用周期、保留新的正确事件恢复可以是等价变异，不能要求它必然算错。旧 P16 原本等价；现在求值器不重复套掩码，裸字段测试使忽略叶子掩码成为可观察错误。extrema 已改两趟法，对应验证其块到期条件而非不存在的队列出队。

| 编号及变异 | 结果 | 失败证据 |
|---|---|---|
| E01 gt dispatch | KILLED | `test/panelops/test_streaming.py::test_elementwise_independent_reference[gt]` |
| E02 ge dispatch | KILLED | `test/panelops/test_streaming.py::test_elementwise_independent_reference[ge]` |
| E03 lt dispatch | KILLED | `test/panelops/test_streaming.py::test_elementwise_independent_reference[lt]` |
| E04 le dispatch | KILLED | `test/panelops/test_streaming.py::test_elementwise_independent_reference[le]` |
| E05 eq dispatch | KILLED | `test/panelops/test_streaming.py::test_elementwise_independent_reference[eq]` |
| E06 ne dispatch | KILLED | `test/panelops/test_streaming.py::test_elementwise_independent_reference[ne]` |
| E07 and_ dispatch | KILLED | `test/panelops/test_streaming.py::test_elementwise_independent_reference[and_]` |
| E08 or_ dispatch | KILLED | `test/panelops/test_streaming.py::test_elementwise_independent_reference[or_]` |
| E09 not_ dispatch | KILLED | `test/panelops/test_streaming.py::test_elementwise_independent_reference[not_]` |
| E10 where dispatch | KILLED | `test/panelops/test_streaming.py::test_elementwise_independent_reference[where]` |
| E11 maximum dispatch | KILLED | `test/panelops/test_streaming.py::test_elementwise_independent_reference[maximum]` |
| E12 minimum dispatch | KILLED | `test/panelops/test_streaming.py::test_elementwise_independent_reference[minimum]` |
| E13 sign dispatch | KILLED | `test/panelops/test_streaming.py::test_elementwise_independent_reference[sign]` |
| E14 power dispatch | KILLED | `test/panelops/test_streaming.py::test_elementwise_independent_reference[power]` |
| E15 signed_power dispatch | KILLED | `test/panelops/test_streaming.py::test_elementwise_independent_reference[signed_power]` |
| E16 sqrt dispatch | KILLED | `test/panelops/test_streaming.py::test_elementwise_independent_reference[sqrt]` |
| E17 exp dispatch | KILLED | `test/panelops/test_streaming.py::test_elementwise_independent_reference[exp]` |
| E18 clip dispatch | KILLED | `test/panelops/test_streaming.py::test_elementwise_independent_reference[clip]` |
| E19 is_nan dispatch | KILLED | `test/panelops/test_streaming.py::test_elementwise_independent_reference[is_nan]` |
| E20 fill_nan dispatch | KILLED | `test/panelops/test_streaming.py::test_elementwise_independent_reference[fill_nan]` |
| N01 sum incoming update | KILLED | `test/panelops/test_adversarial.py::test_adversarial_windows[2-sum]` |
| N02 std outgoing square | KILLED | `test/panelops/test_adversarial.py::test_adversarial_windows[2-std]` |
| N03 std incoming square | KILLED | `test/panelops/test_adversarial.py::test_adversarial_windows[2-std]` |
| N04 paired incoming update | KILLED | `test/panelops/test_adversarial.py::test_adversarial_windows[2-corr]` |
| N05 paired outgoing update | KILLED | `test/panelops/test_adversarial.py::test_adversarial_windows[2-corr]` |
| N06 decay outgoing update | KILLED | `test/panelops/test_adversarial.py::test_adversarial_windows[2-decay_linear]` |
| N07 std constant proof | KILLED | `test/panelops/test_adversarial.py::test_adversarial_windows[2-std]` |
| N08 constant predictor proof | KILLED | `test/panelops/test_adversarial.py::test_adversarial_windows[2-corr]` |
| N09 cross error guard | KILLED | `test/panelops/test_adversarial.py::test_adversarial_windows[5-slope]` |
| N10 column block omitted | KILLED | `test/panelops/test_adversarial.py::test_streaming_column_boundaries_and_prefix` |
| N11 extrema outgoing boundary | KILLED | `test/panelops/test_adversarial.py::test_adversarial_windows[2-min]` |
| N12 rank equal count | KILLED | `test/panelops/test_adversarial.py::test_adversarial_windows[2-rank]` |
| N13 long quantile order statistic | KILLED | `test/panelops/test_adversarial.py::test_adversarial_windows[250-quantile]` |
| N14 MAD deviations | KILLED | `test/panelops/test_adversarial.py::test_adversarial_windows[2-mad]` |
| N15 count nonzero | KILLED | `test/panelops/test_adversarial.py::test_adversarial_windows[2-count]` |
| N16 sumif missing condition | KILLED | `test/panelops/test_adversarial.py::test_adversarial_windows[2-sumif]` |
| N17 ffill limit boundary | KILLED | `test/panelops/test_adversarial.py::test_adversarial_windows[2-ffill]` |
| N18 rsquare square | KILLED | `test/panelops/test_adversarial.py::test_adversarial_windows[2-rsquare]` |
| N19 idx position origin | KILLED | `test/panelops/test_adversarial.py::test_adversarial_windows[2-idxmin]` |
| N20 variance output | KILLED | `test/panelops/test_adversarial.py::test_adversarial_windows[2-var]` |
| N21 zscore missing today | KILLED | `test/panelops/test_adversarial.py::test_adversarial_windows[5-zscore]` |
| N22 scale absolute norm | KILLED | `test/panelops/test_operators.py::test_random_cross_sections_and_evaluation` |
| N23 demean centering | KILLED | `test/panelops/test_adversarial.py::test_cross_section_ddof_and_new_conventions` |
| N24 median quantile | KILLED | `test/panelops/test_adversarial.py::test_adversarial_windows[2-median]` |
| N25 unmasked std outgoing update | KILLED | `test/panelops/test_adversarial.py::test_dense_fast_path_switch_cannot_change_past_bits` |
| N26 unmasked std incoming update | KILLED | `test/panelops/test_adversarial.py::test_dense_fast_path_switch_cannot_change_past_bits` |
| N27 rank ring outgoing count | KILLED | `test/panelops/test_streaming.py::test_random_long_windows_rebuilds` |
| N28 long quantile tree expiration | KILLED | `test/panelops/test_adversarial.py::test_adversarial_windows[250-quantile]` |
| N29 sum output mask | KILLED | `test/panelops/test_adversarial.py::test_adversarial_windows[2-sum]` |
| N30 neutralization intercept | KILLED | `test/panelops/test_edge_cases.py::test_neutralization_singletons_all_missing_and_zero_columns` |
| N31 IC date alignment | KILLED | `test/panelops/test_operators.py::test_random_cross_sections_and_evaluation` |
| N32 CSR row pointer | KILLED | `test/panelops/test_edge_cases.py::test_reference_and_production_empty_target_rows` |
| N33 sum rebuild excludes today | KILLED | `test/panelops/test_adversarial.py::test_adversarial_windows[2-sum]` |
| N34 cancellation expansion disabled | KILLED | `test/panelops/test_adversarial.py::test_multiple_scales_cancel_without_losing_small_sum` |
| N35 where bit selection | KILLED | `test/panelops/test_streaming.py::test_elementwise_independent_reference[where]` |
| N36 product singleton signed zero | KILLED | `test/panelops/test_streaming.py::test_single_observation_product_preserves_finite_bits` |
| N37 fill_nan bit selection | KILLED | `test/panelops/test_streaming.py::test_elementwise_independent_reference[fill_nan]` |
| N38 fill_nan block boundary | KILLED | `test/panelops/test_streaming.py::test_fill_nan_contiguous_blocks_and_threads` |
| N39 is_nan contiguous validity | KILLED | `test/panelops/test_streaming.py::test_elementwise_independent_reference[is_nan]` |
| P01 EMA default alpha | KILLED | `test/panelops/test_adversarial.py::test_adversarial_windows[2-ema]` |
| P02 autocorrelation lag | KILLED | `test/panelops/test_edge_cases.py::test_autocorr_pairwise_ranks_and_turnover_zero_lag` |
| P03 cache data version | KILLED | `test/factor/test_factor.py::test_cache_and_common_subexpressions` |
| P04 cache build identity | KILLED | `test/factor/test_factor.py::test_cache_and_common_subexpressions` |
| P05 cache axes | KILLED | `test/factor/test_round2.py::test_cache_corruption_axes_build_id_and_readonly` |
| P06 cache mask | KILLED | `test/factor/test_factor.py::test_cache_and_common_subexpressions` |
| P07 cache universe | KILLED | `test/factor/test_factor.py::test_cache_and_common_subexpressions` |
| P08 arithmetic nonfinite output | KILLED | `test/factor/test_factor.py::test_missing_and_no_shift` |
| P09 unordered canonical parameters | KILLED | `test/factor/test_round2.py::test_clip_parameter_order_is_canonical` |
| P10 nested lookback max not addition | KILLED | `test/factor/test_round2.py::test_parameters_defaults_nested_lookback_and_long_chain` |
| P11 delay lookback | KILLED | `test/factor/test_factor.py::test_hash_and_lookback` |
| P12 reversal sign | KILLED | `test/factor/test_factor.py::test_each_builtin_hand_calculated` |
| P13 unadjusted prices | KILLED | `test/factor/test_factor.py::test_hash_and_lookback` |
| P14 momentum skipped month | KILLED | `test/factor/test_factor.py::test_each_builtin_hand_calculated` |
| P15 Amihud signed return | KILLED | `test/factor/test_factor.py::test_each_builtin_hand_calculated` |
| P16 bare field mask | KILLED | `test/factor/test_round2.py::test_bare_field_respects_mask` |
| P17 constant hash type | KILLED | `test/factor/test_factor.py::test_expression_equality_obeys_hash_contract` |
| P18 fingerprint omits Python | KILLED | `test/panelops/test_build_id.py::test_fingerprint_tracks_python_in_workspace_and_sdist` |
| P19 fingerprint omits core | KILLED | `test/panelops/test_build_id.py::test_fingerprint_tracks_python_in_workspace_and_sdist` |
| R01 sum outgoing value | KILLED | `test/panelops/test_adversarial.py::test_adversarial_windows[2-sum]` |
| R02 sum never rebuild (periodic and event) | KILLED | `test/panelops/test_adversarial.py::test_adversarial_windows[2-sum]` |
| R03 paired moments never rebuild | KILLED | `test/panelops/test_adversarial.py::test_adversarial_windows[2-std]` |
| R04 moment outgoing coefficient | KILLED | `test/panelops/test_adversarial.py::test_adversarial_windows[2-corr]` |
| R05 ordered min_periods | KILLED | `test/panelops/test_adversarial.py::test_adversarial_windows[2-quantile]` |
| R06 rank missing today | KILLED | `test/panelops/test_adversarial.py::test_adversarial_windows[2-rank]` |
| R07 time rank minimum ties | KILLED | `test/panelops/test_adversarial.py::test_adversarial_windows[2-rank]` |
| R08 cross rank normalization | KILLED | `test/panelops/test_operators.py::test_invariances` |
| R09 extrema earliest ties | KILLED | `test/panelops/test_adversarial.py::test_adversarial_windows[2-argmin]` |
| R10 argmax position instead of age | KILLED | `test/panelops/test_adversarial.py::test_adversarial_windows[2-argmin]` |
| R11 decay rebuilt weights reversed | KILLED | `test/panelops/test_adversarial.py::test_adversarial_windows[2-decay_linear]` |
| R12 decay incoming weight | KILLED | `test/panelops/test_adversarial.py::test_adversarial_windows[2-decay_linear]` |
| R13 EMA swapped alpha | KILLED | `test/panelops/test_adversarial.py::test_adversarial_windows[2-ema]` |
| R14 regression predictor variance | KILLED | `test/panelops/test_adversarial.py::test_adversarial_windows[2-slope]` |
| R15 regression residual sign | KILLED | `test/panelops/test_adversarial.py::test_adversarial_windows[5-residual]` |
| R16 winsor lower bound | KILLED | `test/panelops/test_operators.py::test_random_cross_sections_and_evaluation` |
| R17 MAD consistency multiplier | KILLED | `test/panelops/test_operators.py::test_random_cross_sections_and_evaluation` |
| R18 negative group valid | KILLED | `test/panelops/test_operators.py::test_random_cross_sections_and_evaluation` |
| R19 unweighted design | KILLED | `test/panelops/test_operators.py::test_random_cross_sections_and_evaluation` |
| R20 rank IC returns unranked | KILLED | `test/panelops/test_operators.py::test_random_cross_sections_and_evaluation` |
| R21 group boundaries | KILLED | `test/panelops/test_edge_cases.py::test_signed_zero_tie_order` |
| R22 turnover wrong retained group | KILLED | `test/panelops/test_operators.py::test_random_cross_sections_and_evaluation` |
| R23 targets k+1 | KILLED | `test/panelops/test_edge_cases.py::test_signed_zero_tie_order` |
| R24 targets fraction floor | KILLED | `test/panelops/test_operators.py::test_random_cross_sections_and_evaluation` |
| R25 targets reversed tie break | KILLED | `test/panelops/test_edge_cases.py::test_signed_zero_tie_order` |
| R26 omitted final streamed row | KILLED | `test/panelops/test_adversarial.py::test_adversarial_windows[2-sum]` |
| R27 delta reversed | KILLED | `test/panelops/test_adversarial.py::test_adversarial_windows[2-delta]` |
| R28 ignore historical mask | KILLED | `test/panelops/test_adversarial.py::test_adversarial_windows[2-min]` |
| R29 kurtosis excess omitted | KILLED | `test/panelops/test_adversarial.py::test_adversarial_windows[5-kurt]` |
| R30 skew central moment | KILLED | `test/panelops/test_adversarial.py::test_adversarial_windows[5-skew]` |
| R31 covariance ddof | KILLED | `test/panelops/test_adversarial.py::test_adversarial_windows[2-cov]` |
| R32 quantile interpolation | KILLED | `test/panelops/test_adversarial.py::test_adversarial_windows[2-quantile]` |
| R33 product sign ignored | KILLED | `test/panelops/test_adversarial.py::test_adversarial_windows[2-product]` |
| R34 variance error guard removed | KILLED | `test/panelops/test_adversarial.py::test_adversarial_windows[2-std]` |
| R35 cross zscore ddof | KILLED | `test/panelops/test_adversarial.py::test_cross_section_ddof_and_new_conventions` |
| R36 higher moments never rebuild | KILLED | `test/panelops/test_adversarial.py::test_adversarial_windows[5-skew]` |

## 基准与存储

66 个算子的六类实现及线程、输出之外内存完整表见 [PANELOPS_BENCHMARKS.md](PANELOPS_BENCHMARKS.md)。`make panelops-bench` 默认输出 `.cache/panelops/latest.json` 并与唯一冻结基线比较；`--full` 才重跑完整规模/窗口/缺失矩阵。`field_schema.py METADATA --generate` 输出字段快照，不写数据层。

一次性探索 JSON、full.log、旧复核日志、绝对路径记录、preflight/quantile_probe/reconcile 均不再入库。以下清单是相对 git 的全部未忽略新增文件（包括上一轮未提交文件），而非把旧文件冒称为本轮新增。

| 新增文件 | 字节 |
|---|---:|
| `bench/panelops/additional_baselines.py` | 12819 |
| `bench/panelops/baselines.py` | 28230 |
| `bench/panelops/cliff.py` | 1523 |
| `bench/panelops/cross_baselines.py` | 7793 |
| `bench/panelops/crosscheck.py` | 8346 |
| `bench/panelops/field_schema.py` | 1039 |
| `bench/panelops/mutation_cases.py` | 15476 |
| `bench/panelops/mutations.py` | 7849 |
| `bench/panelops/numerical_audit.py` | 7063 |
| `bench/panelops/report.py` | 20019 |
| `bench/panelops/results/baseline.json` | 189955 |
| `bench/panelops/run.py` | 15714 |
| `bench/panelops/verify_baselines.py` | 6025 |
| `crates/panelops-core/Cargo.toml` | 106 |
| `crates/panelops-core/src/element.rs` | 8952 |
| `crates/panelops-core/src/evaluation.rs` | 6075 |
| `crates/panelops-core/src/lib.rs` | 26729 |
| `crates/panelops-core/src/time.rs` | 52120 |
| `docs/PANELOPS_BENCHMARKS.md` | 22815 |
| `docs/PANELOPS_NOTES.md` | 36261 |
| `packages/panelops/Cargo.toml` | 329 |
| `packages/panelops/README.md` | 5372 |
| `packages/panelops/README.zh-CN.md` | 3790 |
| `packages/panelops/SPEC.md` | 12641 |
| `packages/panelops/build.rs` | 1866 |
| `packages/panelops/python/panelops/__init__.py` | 504 |
| `packages/panelops/python/panelops/_numpy.py` | 9403 |
| `packages/panelops/python/panelops/api.py` | 21154 |
| `packages/panelops/python/panelops/py.typed` | 0 |
| `packages/panelops/python/panelops/reference.py` | 13575 |
| `packages/panelops/src/lib.rs` | 12618 |
| `src/ashare/factor/__init__.py` | 698 |
| `src/ashare/factor/engine.py` | 10034 |
| `src/ashare/factor/expression.py` | 10833 |
| `src/ashare/factor/library.py` | 6613 |
| `test/factor/test_factor.py` | 9068 |
| `test/factor/test_round2.py` | 9854 |
| `test/panelops/test_adversarial.py` | 7619 |
| `test/panelops/test_build_id.py` | 2185 |
| `test/panelops/test_edge_cases.py` | 8718 |
| `test/panelops/test_operators.py` | 14133 |
| `test/panelops/test_streaming.py` | 6678 |

新增文件共 42 个，642,594 字节；其中 bench/panelops 未忽略文件 321,851 字节。

## 第二轮复核后的返工

依据第二轮复核报告（3 条必修、9 条建议）。开始时 build_id `1ca347f6…`，`make check` 324 通过。逐位快照脚本（2425 组：对抗面板有无掩码、随机 5% 缺失、平衡 0/1 条件、无缺失面板；全部时序算子 × W=1/2/5/20/60/250 × min_periods 1/⌈W/2⌉/W，加 ddof=0 与全部截面算子）在修改前后各跑一次比较哈希，脚本与输出不入库。

### 必修 1：sumif 条件分支

- 复现（4000×5500，W=20，min_periods=10，5% 缺失，单线程，5 轮交替取中位数，负载 13）：条件全真 117 ms、一半真 239 ms、正态 118 ms；bottleneck 组合写法 99 ms。
- 修改：`time.rs` 的 sum/mean/count/sumif 共用 `summand`，有效性和条件都用位选择（保留原值位模式或给 +0.0），不再有依赖数据的分支。四处（周期重建、滑出、进入、事件重建）都走同一函数。逐位快照 2425 组与修改前全部一致。
- 修改后（同配置，负载 9.9）：sumif 全真 79 ms、一半真 80 ms、正态 80 ms；bottleneck 92 ms。顺带 sum 从 78 降到 61 ms、mean 到 62 ms（缺失 5% 的有效性分支原来也有误预测代价），与 bottleneck（62 ms）持平。
- 其他算子的同类分支（2000×2000 单线程探测：正态 5% 缺失 / 50% 随机缺失 / 一半与上一值相等的整数 / 一半真条件，`branch_probe1`）：
  - std/var/zscore：运行长度计数 `if last == x` 在并列多时误预测（std 正态 17.6 ms，并列 31.0 ms）；滑入/滑出的有效性也是分支。改为 `advance_run` 位选择和 `keep_bits`，并列与正态都降到约 12.5–13 ms。另试过把「样本不足」的提前 `continue` 改成位选择，配对 A/B 交替（4 轮）显示正态输入反而慢 30%（13.6→17.7 ms），已撤回；50% 随机缺失这种输入在真实面板上不会出现（缺失按停牌、上市前成段出现）。
  - min/max/arg*/idx*：前缀、后缀和合并三处的「有效且更优」改为位选择，输出门限也是。A/B（负载 24）：min 正态 31.0→15.6 ms、50% 缺失 61.4→39.9 ms；argmax 22.5→15.7 ms。
  - ema/ffill：有效性分支改为位选择；50% 缺失 16.3→4.9 ms，正态不变。
  - sum/mean/count/sumif 的输出门限（样本数是否达到 min_periods）也改为位选择：50% 缺失 17.5→10.2 ms。
  - 不处理的：配对矩与回归（corr/cov/slope 等）每格工作量大，50% 缺失只慢 5% 左右；rank 环形缓冲与 product 的分支是算法本身的比较；decay_linear 在整数并列输入上慢 3 倍不是分支问题，而是窗口和恰好为 0 时触发事件重建（见建议 6）。
  - 以上每一步都用逐位快照核对 2425 组结果与修改前一致。

### 必修 2：sum/mean/sumif 事件重建的精确参考

- 新增 `test/panelops/test_exact_reference.py`：每个有限浮点数是 2 的幂分之一的整数，按列取公共指数后用 Python 大整数前缀和得到每个窗口的精确幂和，统计量由精确整数一次舍入得到（开方的取精确平方舍入一次后再开方）。不调用内核和 `reference.py`。抽 400 个格子与逐窗口 `Fraction` 直接计算核对 3649 个值，全部在 1–4 ulp 内（`exact_sanity.py`）。
- 后来为建议 6 加了两列台阶序列（见下），共 19 列。覆盖 16 个算子：sum、mean、sumif、decay_linear（同一事件+浮点展开机制），以及 std、var、zscore、corr、cov、skew、kurt、slope、residual、regression_slope、regression_residual、rsquare（矩守卫与直接两遍兜底）。语料 620×19，含 1e12 尖峰、1e15 接 1e-3、四档水平跳变、`[2^200, 2^100, 1, -2^200, -2^100]` 循环、1e6 转 0.1、1e8/1e-8 混合、近常数、整数并列，3% 掩码，W=2/5/20/60/250。缺失位置要求逐格一致。
- 容差：除下面五个算子外一律 SPEC 的 rtol=atol=2e-10。cov、slope、residual、regression_slope、regression_residual 在专门构造的大动态范围列（尖峰、尖峰转常数、多量级循环、1e8/1e-8 混合）里、且窗口内仍有 ≥1e6 的值时，允许另加 `1e-12 × M`（M 为窗口内的中心化量级，定义见 SPEC）；尖峰滑出窗口后回到原容差，同一测试守住。
- 运行 2.7 秒。变异：M02（事件阈值 1e-11→1e-3）被 `test_exact_reference_adversarial_windows[sum]` 抓到。

### 必修 3：偏度、峰度的独立对照

- 新增 `test/panelops/test_independent.py`，期望值全部不经过 `reference.py`：
  - 1, 2, 4, 8 的手算分数：m2=115/16、m3=405/32、m4=25141/256，有偏偏度 0.6568077344996993（40 位十进制核对）、有偏峰度 −14534/13225；同时写出无偏（pandas/Excel 的调整 Fisher-Pearson）定义的值，断言结果与有偏定义一致、与无偏定义相差 0.1 以上；平移、取负、2 的幂缩放下不变；中间有缺失时跳过而不是补零。
  - 80×4 整数并列加缺失的滚动窗口，逐窗口用 `Fraction` 算偏度、峰度。
- 「只与自家参考对拍」的清单（修改前）：时序的 delay、delta、min、max、rank、quantile、argmin/argmax、idxmin/idxmax、product、ema、median、mad、count、ffill，以及第一轮内核层面的偏度、峰度和 sum 一族（必修 2 已由精确参考覆盖）；截面全部 8 个；评估 6 个（ic、rank_ic、factor_autocorr、group_returns、group_turnover、targets）只有零散的手算小例子，主体与 `reference.py` 对拍；逐元素 20 个与 `reference.elementwise`（标量规格）对拍。其中已有手算小例子的：idx*、ffill、count、sumif、mad、rsquare、cs_rank 并列、ic 对齐、targets 指针。
- 补上的独立对照（同一文件）：时序 16 个按 SPEC 原文逐格写的 Python 循环（W=1/3/7/50，min_periods 三档，rank 两种归一、quantile 三个 q、ffill 有无 limit）；截面 rank/zscore/demean/scale/winsor_quantile/winsor_mad/group_demean 逐行按 SPEC 计算，neutralize 用 `Fraction` 正规方程精确求解加权残差；评估 6 个按 SPEC 原文实现；逐元素 20 个各给 1–3 个字面值（含定义域外、溢出、缺失、掩码）。
- 联合变异（内核与 `reference.py` 同时改）：J02 峰度去 −3、J05 count 改数全部、J06 EMA 默认 alpha 改 1/(W+1)、J07 rank 单值归一改 0，结果见变异表。

### 建议 3、4、5、7

- 3：`Expr.canonical()` 和 `__str__` 改为显式栈：canonical 后序遍历，`str` 逐个输出片段最后一次拼接（深链不反复复制子串）。`test/factor/test_rework.py` 用 3000 层链检查；另用脚本把新实现与旧递归实现（逐字复制）在 14 个内置因子、一个中性化表达式和 500 个随机表达式上逐个比较，canonical 与文本全部相同（`expr_equiv.py`）。
- 4：`test_cache_key_tracks_evaluator_implementation` 把 `_implementation_id` 换成另一个值后必须不命中缓存、重算结果相同、缓存目录多出一份。M21 被它抓到。
- 5：`test/panelops/test_constant_cost.py`：std/var/zscore/corr/cov/skew/kurt/slope/rsquare，2400×384、W=250、单线程，常数输入与随机输入交替各跑 5 次取最小值，常数不得超过随机的 4 倍。本机实测常数输入只有随机的 0.2–0.8 倍；修复前的悬崖是 40–60 倍。M04 被它抓到（见变异表）。
- 7：同一份源码两次构建出的 `.so` 字节不同（链接产物含构建路径、时间等不可复现字节），但 `build_id` 只由源码决定。核对「导入的扩展是不是这份源码构建的」要看源码哈希加 `build_id`，不能看 `.so` 哈希；本轮的变异脚本和快照脚本都是这样核对的。

### 必修 1 补充：修改前后（同一时段 A/B）

4000×5500、W=20、min_periods=10、5% 缺失、单线程；原内核（本轮开始时的 `time.rs`）与最终内核两个 `.so` 交替换入、每次新进程跑 2 轮、共 3 次，取最小值的中位数；负载 14，同机有其他任务。

| 算子 | 原 | 现 | bottleneck（同轮） |
|---|---:|---:|---:|
| ts_sum | 80.0 | 63.8 | 62.9 / 64.4 |
| ts_mean | 86.6 | 66.5 | |
| ts_std | 105.9 | 75.3 | 76.5 / 78.5 |
| ts_var | 106.9 | 73.6 | |
| ts_zscore | 121.5 | 91.0 | |
| ts_sumif（条件一半真） | 239.6 | 80.7 | |
| ts_min | 167.6 | 77.2 | |
| ts_argmax | 115.7 | 81.8 | |
| ts_ffill | 29.8 | 19.3 | |

单进程内交替（`perf.py`，7 轮，负载 13.4）：sum 76.3 对 bottleneck 76.3；mean 79.9 对 78.3；std 89.0 对 92.0；var 89.8 对 96.7；sumif 全真/一半真/正态 96.8/94.1/97.6 对 bottleneck 组合写法 113.6。默认线程：sum 11.7、mean 13.6、std 12.8、var 12.7、sumif 14.9。

### 建议 1、2：SPEC 与边界核实

- SPEC 改写：其余算子在对抗输入下仍按精确值 rtol=atol=2e-10；cov、slope、residual、regression_slope、regression_residual 在窗口内量级相差悬殊时放宽为 `2e-10·(1+|精确值|) + 1e-12·M`（M 的定义写在 SPEC），尖峰滑出窗口后回到原容差。界由必修 2 的测试守住；基线下这些格子的误差是 eps×M 的 46–1370 倍（W=2 regression_slope 最大），离 1e-12·M（约 4500 倍）有三倍以上余量。
- 实测并写入 SPEC：次正规数和 1e-310 量级序列上 zscore 为 ±inf，skew、kurt、corr、rsquare 为 NaN，std、var 为 0（复核只提了前几个，std/var 给 0 是本轮补测发现的）；product 中途溢出给 inf；cs_scale 绝对值和溢出整行 NaN；cs_demean 先以第一个有限值为锚中心化，同号的 1e308 量级能算，正负 1e308 相消使均值溢出时整行 NaN；cs_zscore 平方和溢出整行 NaN。
- 核实时发现 NumPy 后端在这几种溢出行上与 Rust 不一致（scale 给 0、demean 给 ±inf、zscore 给 0），已在 `_numpy.py` 改为同样整行 NaN，`test_cross_section_overflow_rows_are_missing` 两个后端都测。
- dtype：因子层两种叶子都接受布尔、整数、浮点（含 float16/32）并转 float64，`dtype="int64"` 的分组叶子只收整数；panelops 本身仍严格 float64。SPEC 已统一写明。`engine.py` 里市场叶子原来那句 dtype 检查在转换之后、永远不会触发，报错信息也写成「float64 序列」，改为只核对形状、信息改为「长度 T 或 T×1 的数值序列」。

### 建议 6：在复核语料上看不出作用的守卫

结论：三处都构造出了需要它们的输入，全部保留，输入已进测试；没有删除任何守卫。

- M01 sum 的补偿求和：去掉补偿后，最坏误差界是 1022 次舍入 × 0.5 ulp × 峰值，而事件检查只保证峰值 < 5630×(|值|+1)，所以理论上 1.3e-9×(|值|+1) 的误差可以不触发事件。构造：W=2，值在 2843 附近，每行选尾数让「两数相加落到更粗一档网格」的平局按偶数规则向上舍入，509 行后转为 0.01、0.01。峰值 5686 低于这一行的事件门槛 5742，无补偿时 sum/sumif 误差 2.3e-10 > 容差 2.04e-10；有补偿时 2.2e-13。`test_sum_compensation_needed_between_rebuilds`。
- M03 std 误差界计入重建以来的更新次数：二矩状态是普通加减（无补偿），误差随更新次数累积。构造：W=2，锚点 0，±A（A≈2^20）交替，每行选尾数让 s2 的加法平局向上舍入，509 行累积约 510 个半 ulp；最后一对是 ±0.0145A（均值 0）。只按 W 估界不会重建，var 相对误差 2.68e-10；按更新次数估界会重建，误差 0。`test_variance_bound_counts_updates_since_rebuild`。
- M09 高阶矩守卫：均值相对锚点偏移 m/σ 在约 3 到 75 之间时，只有这条守卫会触发（二矩守卫在 m/σ≈37–75 才触发），峰度消去误差约 eps×(m/σ)^4。构造：每 17 或 23 行抬升 25–45 个噪声单位的台阶，加入精确参考语料（两列）。放宽到 1e-5 后 kurt 在 W=4–12 上 47–295 格超容差（最大占用 9.2），原守卫 0 格。偏度只到 0.19 占用，三次方不够敏感，由同一守卫的峰度失败抓住。
- M04 std 常数判定 `>=` 改 `>`：不是守卫。运行长度从序列开头累计、不受窗口截断，常数窗口里 run 很快超过 count，`>` 同样成立；只有暖机阶段 run 恰好等于 count 的几行多做一次重建。实测 2400×384、W=250 常数输入在这个变异下仍是 3 ms（随机 3 ms），结果逐位相同，属于等价变异，不能拿它当性能测试的靶子。性能测试改用「整个去掉常数判定」的变异检验（见变异表 M04b/M04c）。
- M08 cross 守卫 1e-11→1e-3：仍存活，判为在 SPEC 界内等价。二矩守卫已保证 eps·|a0·a1|/c ≤ √(eps·a0²/c · eps·a1²/c) ≤ 1.25e-12·√(vx·vy)，所以 cross 的消去误差先被 vx、vy 两条守卫限住；cross 守卫只在窗口内有尖峰时改变结果（精确参考下 slope 在 W=5 超原容差的格子从 7 增到 21），这些格子在两种构建下都在窗口尺度界 1e-12·M 之内（基线 46·eps·M，变异后 60·eps·M），与基线差距太小，不能写成稳定的测试。保留这条守卫（它给弱相关窗口提供相对精度），不删。
- 另记：decay_linear 在整数并列输入上比正态输入慢 3 倍（2000×2000：30 → 92 ms）。原因是窗口和恰好为 0 时，`峰值·eps > |和|` 的事件条件成立，每格做一次 O(W) 重建。这不是分支问题，也不是复核列出的守卫；要改需要换事件判据并证明不丢精度，本轮没有做，列为待办。

### 变异检查（本轮）

做法：直接改工作区真实源码 → `uv sync` 重建 → 核对 `build_id` 变了且扩展从本工作区导入 → `pytest -x test/panelops test/factor` → 恢复字节；全部结束后核对源码哈希和 `build_id`（`e3e7c50a…`）回到对照。先跑不改动的对照（311 passed）。前两批（第二批 20 处）在中间版本上跑，最后一批 15 处在最终源码上重跑关键项。

| 编号 | 变异 | 结果 | 抓到它的测试 |
|---|---|---|---|
| M01 | sum 滑入/滑出不做补偿 | KILLED | test_sum_compensation_needed_between_rebuilds |
| M02 | sum 事件阈值 1e-11→1e-3 | KILLED | test_exact_reference_adversarial_windows[sum]；单独跑 [mean]、[sumif] 也各自失败 |
| M03 | std 误差界不计更新次数 | KILLED | test_variance_bound_counts_updates_since_rebuild |
| M04 | std 常数判定 >= 改 > | 存活（等价，见建议 6） | — |
| M04b | 去掉 std 常数捷径 | KILLED | test_constant_input_costs_no_more_than_random[std] |
| M04c | 去掉矩算子常数捷径 | KILLED | test_constant_input_costs_no_more_than_random[corr] |
| M08 | cross 守卫 1e-11→1e-3 | 存活（SPEC 界内等价，见建议 6） | — |
| M09 | 高阶矩守卫 1e-11→1e-5 | KILLED | test_exact_reference_adversarial_windows[kurt]（[skew] 单独跑存活，三次方不够敏感） |
| M21 | 缓存键去掉求值器指纹 | KILLED | test_cache_key_tracks_evaluator_implementation |
| J02 | 峰度去 −3，内核与参考同改 | KILLED | 精确参考 [kurt]；只跑 test_independent.py 也被手算值测试抓到 |
| J05 | count 改数全部，内核与参考同改 | KILLED | test_cross_section_ddof_and_new_conventions（手算值） |
| J06 | EMA 默认 alpha 1/(W+1)，api 与参考同改 | KILLED | test_order_and_recursive_operators_follow_spec_text[ema] |
| J07 | rank 单值归一改 0，内核与参考同改 | KILLED | test_order_and_recursive_operators_follow_spec_text[rank] |
| B01 | sumif 位选择忽略条件 | KILLED | test_cross_section_ddof_and_new_conventions |
| B02 | sum 位选择忽略有效性 | KILLED | test_adversarial_windows[2-sum] |
| B03 | 运行长度把每个值都当重复 | KILLED | test_adversarial_windows[2-std] |
| B04 | 运行长度更新忽略有效性 | KILLED | test_exact_reference_adversarial_windows[std] |
| B05 | 极值前缀并列取旧位置 | KILLED | test_adversarial_windows[2-argmax] |
| B06 | ema/ffill 从缺失值取状态 | KILLED | test_adversarial_windows[2-ema] |
| B07 | sum 输出门限差一 | KILLED | test_adversarial_windows[2-sum] |
| B08 | 极值合并忽略后缀有效性 | KILLED | test_adversarial_windows[2-min] |

复核人 31 处里原来存活的 9 处：M01、M02、M03、M09、M21、J02 现在都被抓到；M04、M08 判为等价并给出理由；M19（列块宽度减 1）结果与分块无关，仍属等价，已有线程/分块逐位一致测试守住不变性。B 组是本轮新写的位选择代码各自的破坏。第一轮的 114 处变异没有整批重跑。

### 结果一致性、基准冻结与验证

- 逐位一致：最终构建（`e3e7c50a…`）的 2425 组快照与本轮开始时（`1ca347f6…`）全部相同。明确改了行为的只有：NumPy 后端 cs_scale/cs_demean/cs_zscore 在中间量溢出（正负 1e308 相消、绝对值和或平方和溢出）的行上改为整行 NaN（与 Rust 一致）；市场叶子的报错信息。`canonical()`/`str()` 的输出与旧实现逐个相同。
- 冻结基线 `bench/panelops/results/baseline.json` 仍只有一份：全部 68 个条目的 rust_1/rust_default 用最终构建重测；ts_sum/mean/std/var/sumif/count 的全部基线库重测，并新增 ts_sumif、ts_count「条件一半真」两档（键多了 condition 字段，旧条目视为正态档）；其余算子的非 Rust 行沿用上一次冻结。测量时负载 10–14。按验收口径 Rust 单线程最快 67/68、在 110% 内 68/68（ts_std 比 bottleneck 慢 5.9%），默认线程 68/68。基准表和两个 README 摘要由 `report.py` 重新生成，另加「精度保障的代价」一节。
- 根目录共享文件本轮没有改动。

工作区 `make dev && make check`（ruff、格式、clippy、cargo test、pytest）末尾：
```text
test/panelops/test_streaming.py ..........................               [ 82%]
test/test_engine.py ................                                     [ 86%]
test/test_inputs.py ....................................                 [ 95%]
test/test_properties.py .....                                            [ 97%]
test/test_random.py .                                                    [ 97%]
test/test_rules.py ..........                                            [100%]

============================= 379 passed in 34.14s =============================
```

全新目录（`rsync -a --exclude /target --exclude /.venv --exclude /.cache --exclude /data --exclude /.git --exclude '*.so'`，先确认副本里没有 `.so`）`make dev && make check` 末尾，收集 379 条，`build_id` 同为 `e3e7c50a…`：
```text
test/panelops/test_streaming.py ..........................               [ 82%]
test/test_engine.py ................                                     [ 86%]
test/test_inputs.py ....................................                 [ 95%]
test/test_properties.py .....                                            [ 97%]
test/test_random.py .                                                    [ 97%]
test/test_rules.py ..........                                            [100%]

============================= 379 passed in 35.19s =============================
```

| 本轮新增文件 | 字节 |
|---|---:|
| `test/panelops/test_exact_reference.py` | 13761 |
| `test/panelops/test_independent.py` | 17616 |
| `test/panelops/test_constant_cost.py` | 1696 |
| `test/factor/test_rework.py` | 3097 |

本轮修改：`crates/panelops-core/src/time.rs`、`packages/panelops/python/panelops/_numpy.py`、`packages/panelops/SPEC.md`、两个 README、`src/ashare/factor/expression.py`、`src/ashare/factor/engine.py`、`bench/panelops/run.py`、`bench/panelops/report.py`、`bench/panelops/results/baseline.json`、两份 docs。试验脚本（快照、A/B、构造、变异、精确参考核对）和日志放在工作区之外，没有入库。
