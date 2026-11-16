# C8 实现与验收记录

依据 DESIGN 6.10.2 / 4.19。2026-11-16 按审查决定修订：先逐字同步主干 d402835 的 6.10.2，再修改代码；现行规则见下方「第二次独立复核修复」；后文 R2 和逐行规则保留为历史记录。仅在本 worktree 开发；真实 metrics 通过显式 root 写指定外置目录，小时、分钟源只读。真实研究仅运行挖掘期，设定与台账均位于被忽略的 tmp/，不访问主目录 data/mining/。本任务禁止 worktree 外写入，故不更新全局 worklog。

## 2026-11-16 第二次独立复核修复（现行）

先通读只读 `.review/recheck_c8.md`。29 个新增最小回归在修改前 **29 failed**（`tmp/c8-daily-recheck-red.log`），随后先改 DESIGN 6.10.2，再改代码。onecontext 检索无相关历史，依据用户决议与两份只读审查。

- **P0-2 登记边界**：`metrics.py` 中 `REGISTERED_SWITCHES` 逐合约登记，BTCUSDT / ETHUSDT 都为 `1709510400000`（2024-03-04 00:00 UTC）。逐行对齐只取登记值；日中位数只核对，不再推断或移动边界。明确矛盾报 `REGISTERED_ALIGNMENT`，未登记合约报 `UNREGISTERED_SWITCH`；登记值必须在 UTC 日界。
- K=2 按排序后的已观测行**跨午夜累计**，缺口不清零（保守处理），灰区、空值、同向观测清零。切换前后各 24 小时闭区间内单个明确反向也报 `SWITCH_WINDOW`。不在窗口内的孤立反向仍保留并打质量位。独立 verify 另写登记表、中位数和循环规则，不调用生产实现。
- 日中位数最少双边可比数仍为 120，严格阈值 .01/.1 与逐行质量阈值 .05 不变。无法形成明确日证据时采用登记值，保留旧 `INHERITED_ALIGNMENT=2` 名称与字段，含义改为“当日证据不足、采用登记对齐”。不再要求相邻日替它推断。此处明确取代前两轮的未知日继承和推断切换规则，旧测试随设计替换断言，重复、空串、缺口及逐行质量检查保留。
- **P0-1 缓存**：`ledger.py` 检查完成项结构、统计量/归一化参数类型、内外公式/检验/数据/完整设定哈希及当前请求；`data_versions` 重算数据摘要，日期与评估签名也核对。错误缓存不返回、不进入 BH、不制造名额，追加 `cache_warning / CACHE_INCONSISTENT` 并重新计算。最新候选坏掉时也不能退回更老缓存。私有文件主动整体篡改不在威胁模型内，不做加密认证，已写入 TIMING_EVAL。
- **P1-1 身份**：`settings.py` 的检验哈希纳入三段规范日期。修改挖掘起点三次计三个检验，三个 p 的 BH 与 stats.adjust 精确一致。只有同检验最近一次有完整身份的评估与当前数据版本不同时才标记 `recomputation`；同身份同数据命中缓存不追加记录，失败重试/坏缓存修复/同数据 seed 修改不是数据复算。缓存返回的标志描述原结果生成那次计算，不代表本次又运行了复算。
- DESIGN 4.19 的设定/台账说明同步上述用户已决定的身份规则，避免旧“公式哈希计次数”措辞与接口冲突。

阶段性实测：冻结的外置误差数组预检，BTC 30 段、ETH 25 段明确反向证据，跨日后最长均 1，长度 >=2 均 0；切换前后 24 小时内明确反向均 0。最少非零双边可比数 126/162，120 门槛不改变历史直接判定天数。K 是对同一历史选出的工程门槛，不提供未来误报概率保证。最终 build/verify 数字与逐位对照见下方验收记录。

开发补充回归：`test_cache_consistent_inner_outer_still_checks_current_settings` 先 **1 failed**，移除提前跳过旧完整设定的分支；同一检验的内外设定虽彼此一致，只要与当前请求不符也记录告警，不能静默略过。

开发回归：`test_cache_warning_accepts_unestimable_nan_result` 先失败，确认严格内容版本 JSON 编码不能处理统计量 NaN；告警指纹改用稳定的允许 NaN 的 JSON 编码，真实数据内容编码保持不变。一次替换误伤 sync 的缩进，被四份测试的导入检查立即拒绝（`tmp/c8-daily-recheck-green2.log`），修复后继续全套导入及原有 sync 回归；未跳过任何用例。首轮其余失败是旧推断规则断言和不完整的人造缓存夹具，已按新合同更新，精确错误原因断言保留。

收尾新增回归 `test_revision_compares_latest_attempt_not_any_old_version[cache/failed_attempt]` 先 **2 failed**：v1 成功后，v2 的坏缓存修复或失败重试仍因历史存在 v1 被永久标记复算。改为只比较最近一次同检验、同公式的评估数据版本；完成项必须先通过完整结构校验，started/failed 提供最近实际尝试的版本，告警不参与。新增逆序变正序的变异保护这个条件；代码改变后检查点指纹拒绝继续旧变异批次，最终重新从变异 0 起跑。

`test_mutation_anchors_are_unique_and_compile` 先 **1 failed**，抓到复算重构后失效的变异锚点；修正锚点并把全部锚点唯一性与可编译性纳入变异 0，避免末批才发现基础设施失效。

`test_malformed_numeric_cache_is_miss_not_validator_exception` 先 **1 failed**：JSON 大整数令 NumPy 数值类型检查抛 TypeError。结构校验现在把类型、值域和溢出异常视为无效缓存，仍按未命中告警并重新计算，不让校验器异常阻断评估。

### 资源限制与复跑

共享负载下，首次变异总运行在 840 秒被停止，当时 18 项已被测试抓到；该未完成运行不作为最终验收。`bench/mutations_c8.py` 改为每批最多 10 项，逐项原子保存 tmp 检查点；源码、测试及脚本的 SHA-256 指纹必须完全相同，结果还必须是既定变异顺序的前缀。指纹或顺序不符拒绝续跑；`test_mutation_resume_refuses_changed_source_and_nonprefix_results` 先失败后通过。首次或代码改变后用 `--restart` 从变异 0 开始，后续运行相同命令（不带 restart）续跑；中间批次退出 75 并明确显示进度，全部完成才退出 0 并写公开结果。完整原测试树和原生扩展检查一直保留。

C6 首轮已完成 6 个整组及 ETH 两个 in 组各 24 项；两个 in 组在 840 秒被停止。保留逐持有期检查点，仅续算两个组剩余的 30 日持有期共 24 项。没有降低 4000 次模拟或改变种子、功效计算；最终仍对 288 项、27 个输入数组和 8 项 Holm 逐位比较。分组重启耗时单独报告，不能当作全部 C6 耗时。

### 第二次复核真实数据验收

已发布规则 `crypto-metrics-v4-registered`，内容版本 `b76976005658bee3f08206b7232a26ca00c1d4fc8a463c1839b877928dabeb7e`；分钟版本仍冻结为 `07d208e2077d6e88d0f776b5ec65b85496a2674f970ae644fceebd50700cccbd`。与 `837d9ce2cc38ecaa780cefbc712511821e89825d723e5224fe069c436b15b8f9` 相比，全部 4076 原档、137 个 parquet 分区的引用、SHA-256 和字节数完全相同，**1,173,114 行、11 个字段（含 null 位置）逐位一致**。版本变化只来自规则、登记表与核对阈值元数据。

| 指标 | BTCUSDT | ETHUSDT |
|---|---:|---:|
| 行数 | 651974 | 521140 |
| 切换时刻（UTC 毫秒） | 1709510400000 | 1709510400000 |
| 明确日证据 / 采用登记值的未知日 | 2003 / 263 | 1575 / 235 |
| TAKER_MISMATCH 行数 | 8886 | 7063 |
| TAKER_MISMATCH / 全部行 | 1.3629377859% | 1.3552980005% |
| 明确反向行数 / 跨日最长连续长度 | 30 / 1 | 25 / 1 |
| 切换前后 24h 明确反向行 | 0 | 0 |
| 独立 verify 差异 | 0 | 0 |
| 原字符串完全重复行 | 75255 | 0 |
| 含空值行 / 空值单元格 | 92272 / 227486 | 92241 / 227436 |
| 原档缺口段 / 缺行 | 156 / 634 | 10 / 140 |

两币仍在 2024-03-04 00:00 UTC 切换。每天的双边样本数、两种中位数、对齐和证据不足状态、逐行质量比例，以及逐年比例、两段误差分位数都在 `bench/results/crypto_metrics_validation.json`。来源历史冻结至 2026-11-14，本轮复用原档，不把新增日期混入与旧版本的逐位对照。

构建内部耗时 **104.080 s**（命令整体 110.725 s）；独立 verify 内部累计 **873.366 s**（BTC / ETH 两条命令分别 439.333 / 450.003 s，均未触及 840 s）。本轮无下载，沿用旧下载统计；原档占 **47,565,401 B**，parquet **55,203,251 B**，含历史版本、审计与报告的数据根共 **197,504,258 B**。

真实挖掘演示在 `tmp/c8-recheck-timing-demo/`：三项检验、6 条台账事件，全为 mining；真实 validation/confirmation 调用均 0。7 日动量 / 30 日动量 / 7 日资金费均值的 n 为 1933/1910/1934，p 为 .9172709323/.5401152212/.1762061985，BH q 为 .9172709323/.8101728318/.5286185954。剔除各层耗时字段后，全部统计量与上一轮精确相等；首次实际计算 131.364 s（命令 137.206 s）。

C6 与原始 `tmp/c8-c6-before` 比较：27 个输入数组的 dtype、shape、bytes 全相同；8/8 组 JSON、288 项 4000 次模拟结果和 8 项 Holm 全部逐位相同。两个超时组的剩余部分续算整体 252.878 s，这不是本次全套 C6 总耗时。

本轮无待用户决定的设计问题。既定限制：登记值需人工维护，K=2 和最少 120 样本来自现有历史、不是未来误报保证；私有台账没有加密认证，不防主动整体篡改；前视自检是有限截断检查，不是任意程序的数学证明。

### 本轮修复位置与回归

| 问题 | 实现位置 | 主要回归（均在 test/crypto/test_c8_recheck.py） |
|---|---|---|
| P0 切换边界 | metrics.py 的 REGISTERED_SWITCHES/alignment_days/build；verify_metrics.py 的独立 _alignment/verify | test_registered_boundary_rejects_2355_and_0005、test_reverse_run_crosses_midnight、test_switch_window_rejects_single_reverse、test_registered_constant_checked_by_daily_median、test_build_unregistered_contract_rejected_before_raw_access、test_verify_rejects_resigned_switch_registry |
| P0 缓存伪造 | timing/ledger.py 的 valid_result_record/cached/mining_summary；study.py 结果身份 | test_cache_identity_mismatch_recomputed_with_warning、test_incomplete_cache_is_miss_with_warning、test_review_minimal_forged_cache_not_returned_or_used_by_bh、test_cache_consistent_inner_outer_still_checks_current_settings、test_latest_invalid_cache_recomputes_instead_of_using_older_record、test_malformed_numeric_cache_is_miss_not_validator_exception |
| P1 检验身份/复算 | timing/settings.py 的 test_hash；study.py 最近评估版本比较 | test_each_stage_dates_part_of_test_identity、test_three_mining_starts_are_three_attempts_and_bh、test_same_data_failed_retry_is_not_revision、test_same_test_and_data_with_changed_seed_is_not_revision、test_revision_compares_latest_attempt_not_any_old_version |

新增复核文件 42 项用例；连同原有 C8 测试共 152 项。最终 `pytest -q -n 1 test/crypto test/paths` **617 passed、0 skipped**，64.17 s；只有既有 Starlette/httpx 弃用警告 1 条。`ruff check .`、`ruff format --check .`（845 文件）、`test/paths/test_static.py` 和 `git diff --check` 全通过。负例均保留具体错误码/match；独立 verify 报告差异项的原回归仍在。

本轮改动：上述 5 个实现文件；4 份 C8 测试（新增 test_c8_recheck.py，其余随现行规则更新）；bench/mutations_c8.py、bench/crypto_metrics_validation.py 及 bench/results/ 下五份验收 JSON；DESIGN、CRYPTO_DATA、TIMING_EVAL、C8_NOTES 四份文档。paths.py 仍只有原 C8 的 `crypto_metrics` 一行登记，本轮没有追加其他数据源改动。

最终变异结果：**43 / 43 killed，0 invalid，0 survived**。变异 0 与隔离副本对照各 **152 passed**（27.87 s / 23.26 s），所有源码导入位置均验证属于本 worktree；隔离副本单独验证源码路径。五批耗时分别 143.529 / 201.698 / 112.363 / 106.722 / 51.568 s，均低于 840 s；公开报告保存同一源码/测试指纹及每项实际失败测试名。新增覆盖登记赋值与核对、跨午夜、闭区间切换窗口、内外缓存身份和完整结构、阶段日期、数据版本变化及最近一次评估比较。最终 66 个 pytest.raises 均指定具体 match，没有用任意 ValueError 代替目标原因。

公开汇总：bench/results/crypto_metrics_validation.json、timing_demo.json、c8_c6_bitwise.json、mutations_c8.json、c8_checks.json；metrics 和演示文件的叶值均为数字，不含逐行行情。临时最终汇总脚本从 tmp 启动时显式传入 PYTHONPATH=$PWD:$PWD/src，使 bench 和当前源码均可定位；最终源代码与测试均通过，上述验收数字已核对。

## 2026-11-16 第一次独立审查修复（历史）

已先通读只读审查 `.review/review_c8.md`。旧实现上新增最小复现得到 **20 failed / 2 passed**（`tmp/c8-review-red.log`）：9 个入口/类型组合、伪造哈希缓存、代码/参数可替换、4 个非日界切换、K 边界、最少样本边界、换台账位置和 horizon 计数。

- **P0-1**：`definition.py` 冻结实际执行的规范代码与参数，哈希直接取字节码、常量、名字/参数结构和参数值。`Study.evaluate/confirm` 在缓存/名单之前要求精确 `Factor` 类型；普通 callable、伪造属性、覆盖执行方法的子类均以 `FACTOR_TYPE` 拒绝。公式名、局部名重命名仍不产生新身份。
- **P0-2**：UTC 日中位数仍使用严格 .01/.1。每日选定方向之后（包括继承日），检查双边有限的连续明确反向证据；K=2 即拒绝全次 build。按日排序的连续记录计数，不跨日；同向、灰区、不可比行都打断连续段。独立 verify 用 Python 循环重新实现，未调用生产判定。
- **P1-1**：研究设定增加 `ledger` 并计入哈希；相对路径仅在加载文件时按文件目录解析，`from_dict` 必须绝对路径，统一 resolve。Study 显式位置不同报 `LEDGER_PATH`；默认只接受 `None`，空字符串也拒绝。改变设定里的台账位置产生新设定哈希。历史记录缺少检验身份时 `LEDGER_IDENTITY` 拒绝，不能静默重置历史闸门。
- **P1-2**：`definition_hash` 为公式，`test_hash` 为公式 + 标的 + UTC 调仓钟点 + 持有期 + 固定标签版本。台账次数、BH、验证名单和闸门均按检验身份；1/2/7 日计 3 次，三个合成 p 值的 BH 与 stats.adjust 精确一致。同检验数据修订后只复算，started 和结果标记 `recomputation=true`，不增加名额；缓存不追加记录。
- **P2**：三份 C8 测试的负例全部指定错误码或具体 match；并发闸门捕获也核对原因。独立验证报告测试核对 `normalized_fields` 或 `quality_summary` 差异项，不能靠不相干 ValueError 通过。
- DESIGN 只清理 6.10.2 末尾“逐行全部对上才算通过”这一处残留表述；其余 d402835 同步内容不动。K/最少样本补充来自本次用户明确审查指令，写入代码和本文。

### K 和样本数依据

用冻结全历史误差数组做预检，再由全历史 build 与独立 verify 重新复算验证。阈值均为严格 <.01 / >.1：BTC 明确反向证据 30 段，ETH 25 段，连续长度分布都只有 **1**（BTC 的长度 1 频数 30，ETH 为 25；长度 >=2 的频数均 0）。取 K=2 是大于观测最长孤立段的最小整数，不排除孤立异常，却拒绝审查 01:00/11:00/13:00/23:00 这些最短仍有 12 行的日内切换。回归仅合成同长度片段，不把真实行作为夹具。

非零双边可比样本数分位点 [min,p01,p05,p50,p95,max]：BTC **[126,280,288,288,288,288]**，ETH **[162,288,288,288,288,288]**；两币各 128 个零样本日。取 **120**，即 10 小时的五分钟观测量，作为保守的最低证据门槛，低于真实最小非零值且留少量缺行空间。它是明确的工程门槛，不声称是统计最优值；未来不足 120 的日子进入未知日继承规则。119/120 的两侧边界均回归测试。`crypto-metrics-v3-daily-guarded` 将 K、最少样本及三误差阈值一并纳入内容版本。

### 开发中新增的先红后绿回归

- `test_old_formula_only_ledger_is_rejected_not_reset`：旧台账被当作零尝试，先 1 failed 后拒绝隐式迁移。
- `test_definition_hash_stable_after_execution_and_recreation`：直接 marshal code object 的引用标志使执行前后哈希变化；改为明确的执行语义字段序列化，保留实际代码绑定。
- `test_nonfinite_error_breaks_opposite_evidence`：无穷误差不能充当双边可比反向证据；与上一项先 2 failed 后修复。
- `test_explicit_empty_ledger_is_not_default`：`ledger or default` 接受显式空路径，先 1 failed 后改为仅 None 取默认。
- 缓存回归中的不可估计 p=NaN 用 `np.testing.assert_equal` 精确比较所有字段（含缺失位置），避免 Python 的 NaN!=NaN 造成夹具误报；没有放宽数值容差。
- 静态检查发现三处导入排序，修正后 ruff check / format --check 全部通过；代码测试同时保留构建保护。

### 本轮最终验收

- `pytest -q -n 1 test/crypto test/paths`：**578 passed**，131.39 秒（进程墙钟 133.736 秒），0 skipped；含 C8 **113 项**（metrics 49、timing 28、本轮审查回归 36）及既有 C6 77 项。唯一警告为已有 Starlette/httpx 弃用提示。`test/paths/test_static.py` 随整套路径检查通过；ruff check、format --check（844 文件）、git diff --check 通过。
- P2 额外核查：三份 C8 测试共有 **62 处** raises，全部指定 match；用审查的无关基础设施 ValueError 替换对齐函数后，**6 个负例全部拒绝该错误原因，错误接受数为 0**。
- 变异：先跑原样 **变异 0：113 passed**（51.70 秒），隔离副本 **113 passed**（70.70 秒）；**33/33 killed，invalid=0**。四个关键模块来自本 worktree；隔离阶段确认替代源码位置，保留原生扩展构建保护。覆盖执行代码/哈希绑定、精确 Factor 类型、不可变执行体、持续反向与 K 上下边界、两份最少样本判定、独立 verify 的持续反向、台账位置绑定、检验身份计数，以及全部原有必测点。内部耗时 757.700 秒，含启动的进程墙钟 773.834 秒，未超过 840 秒限时。
- C6 从改动前冻结输入重新计算 **8/8 组、288 项**；27 个数组的 dtype/shape/bytes、全部结果 JSON 字节及 8 项 Holm float64 字节完全相同。4000 次模拟和功效计算均保留，未使用中断续算。本轮 1543.358 秒，原基线 1401.518 秒；共享机器穿插测试，不作性能比较。
- 真实 timing 仅挖掘期：`tmp/c8-review-timing-demo/study.json` 明确登记同目录 ledger，6 条记录全为 mining，3 个公式/3 个检验；验证/确认调用均 **0**。三个因子的统计量（包括 p、q）与上一版逐值相同，耗时 98.810 秒。旧的公式身份台账保留，不自动迁移或覆盖。

真实 metrics 新版 **`837d9ce2cc38ecaa780cefbc712511821e89825d723e5224fe069c436b15b8f9`** 已写入指定外置根并更新 latest。规则为 `crypto-metrics-v3-daily-guarded`，冻结分钟版本 `07d208e2077d6e88d0f776b5ec65b85496a2674f970ae644fceebd50700cccbd`。原档与 **137 个 Parquet 分区均与 R2 逐字节相同**；变化来自规则、阈值与日反向证据统计，旧版保留。

| 当前全历史结果 | BTCUSDT | ETHUSDT |
|---|---:|---:|
| 原档 / 行数 | 2266 / 651974 | 1810 / 521140 |
| 独立 verify 差异 | 0 | 0 |
| 切换 UTC 时刻 | 2024-03-04 00:00 | 2024-03-04 00:00 |
| 直接判定日 / 未知并继承日 | 2003 / 263 | 1575 / 235 |
| 反向证据段数 / 最长段 | 30 / 1 | 25 / 1 |
| 最少非零双边可比样本 | 126 | 162 |
| 重复行 / 空值行 | 75255 / 92272 | 0 / 92241 |
| 空单元格 / 原始缺行 | 227486 / 634 | 227436 / 140 |
| TAKER_MISMATCH 行数 | 8886 | 7063 |
| 偏差占全部行 | 1.362938% | 1.355298% |

切换 epoch 为 **1709510400000 ms**。行数、偏差比例、继承日与 R2 相同；按年比例、每日中位数、每日反向证据以及各段误差分位数均在 `bench/results/crypto_metrics_validation.json`。原始归档截至 2026-11-14，4076 份、47,565,401 bytes；本轮复用原档，新增下载 0。历史下载耗时仍为约 1074.937 秒（旧中断段按 840 秒上限记）。build 内部 49.602 秒（含启动/分钟版本读取的进程墙钟 58.346 秒），独立 verify 合计 342.285 秒；两币报告严格核对版本/覆盖后合并，完整通过、差异 0。Parquet 55,203,251 bytes；数据根总占用 185,194,648 bytes，含旧审计与所有版本报告。

### 修复位置与主要回归

| 审查项 | 修复位置 | 回归测试（`test/crypto/`） |
|---|---|---|
| P0-1 | timing/definition.py、study.py | `test_uncontrolled_callable_rejected_before_cache_and_lists`、`test_forged_hash_cannot_poison_legitimate_cache`、`test_hash_follows_actual_executable_even_after_private_replacement` |
| P0-2 | metrics.py、verify_metrics.py | `test_non_midnight_switch_rejected_from_minute_volumes`（1/11/13/23 时）、`test_real_longest_isolated_reverse_and_k_boundary`、`test_build_and_independent_verify_reject_sustained_reverse` |
| P1-1 | timing/settings.py、ledger.py、study.py | `test_ledger_path_cannot_reset_gates`、`test_editing_configured_ledger_changes_settings_hash`、`test_explicit_empty_ledger_is_not_default` |
| P1-2 | timing/settings.py、ledger.py、study.py、inference.py | `test_label_horizons_are_three_tests_and_bh`、`test_test_identity_includes_entire_label_definition`、`test_data_revision_recomputes_same_test_without_increment`、`test_validation_list_binds_label_horizon` |
| P2 | test_metrics_c8.py、test_timing_c8.py、test_c8_review.py | `test_alignment_rejects_contradictions_and_unidentified`、`test_independent_verifier_rejects_resigned_daily_evidence` 等全部负例 |
| 最少样本 | metrics.py、verify_metrics.py | `test_minimum_paired_daily_samples`（119/120） |

### 未决项与限制

审查要求无未完成项；原来两条待澄清（DESIGN 末尾措辞、最少可比样本数）均已处理。K=2 与 120 是按截至上述日期的分布设定的工程门槛，未来出现持续反向或无法继承的日子会拒绝构建，不能自动放宽。Factor 限制用于受控研究接口，仍不是针对能修改私有对象/进程的恶意 Python 的操作系统沙箱；有限前视截断不能数学证明任意程序永远无前视。公式身份依赖冻结 venv 的字节码格式，解释器升级可能改变身份。C6 模型/模拟网格假设不变；未使用真实 C8 验证/确认期。全局 worklog 因本任务禁止 worktree 外写入未更新，阶段成果记录于本文。

## 环境与约定

- 已读 CLAUDE.md、设计两节和 C6/C7 实现。onecontext session 检索 C8/crypto_metrics/crypto_eval/分钟线无命中。
- make dev 使用本 worktree .venv，CARGO_HOME、UV_PYTHON_INSTALL_DIR、MPLCONFIGDIR 指向 .cache，cargo 限制 2 任务。所有重任务保留 nice -n 15；本沙箱返回 setpriority: Operation not permitted，另限制数值线程和任务数。
- paths.py 第 41 行只新增 `"crypto_metrics": Dataset(False),`，不修改 signals.py / spec/。与 D2 的合并由用户处理。

## 设计解释

- metrics 的内容版本包含原档 SHA-256/CHECKSUM、分钟版本、阈值、月分区和质量汇总，不含耗时。先下载、后判定、全部通过才更新 latest；失败可独立审计，不生成“勉强可用”的版本。原始时间缺口和规范化窗口缺口分别记录。
- 两种误差均按 `abs(实际比值−复算比值)/max(abs(复算比值),1e-12)` 定义；复算使用币本位成交量。零买量但有卖量仍可比较；零卖量、无成交、缺分钟为不可比。生产用 Polars 分组，独立校验用五根精确分钟网格和 `math.fsum`，不调用生产解析/对齐函数。
- 防火墙严格排除 `available_at >= cutoff`。公开 view 只开放挖掘；验证/确认 view 必须先经过名单预约。标签退出价可读阶段末端开盘，但专用读取器不暴露给因子。
- C8 的退出边界允许等于阶段结束；C6 原有严格小于边界保持不变。C8 校准传入 `end+1ms`，且另外屏蔽挖掘前预热。观察和模拟同用连续日历，h 日标签掩码不再次裁切单日收益。
- 定义哈希归一化源码名、参数名、局部名和文档，再加入冻结参数；不证明任意数学等价公式。缓存加入数据版本和评估设定；尝试数现按公式加标的、调仓钟点、持有期和标签口径构成的检验身份计。补登记名单哈希不使挖掘缓存失效。
- 验证/确认失败也消耗预约；整段操作持文件锁，损坏台账拒绝继续。BH 和 Holm 调用已有 stats，NaN 保留名额；某定义最新尝试失败后不会重用旧 p 值。
- 确认组合用挖掘期日频均值/样本标准差、固定名单等权、缺值零分、仓位 ±1；比较同费用的持有及 C3 趋势规则。趋势名义上限固定为 1，区别于 C3 示例的 2，此项偏差已明确公开。
- C6 基准脚本只增加可选持有期列表及检查点回调，默认仍按 7/1/30 顺序执行原数值核。新兼容脚本逐组 840 秒超时、最多 2 组并发，不降低 4000 次模拟或删掉功效计算。

## 回归来源与先红后绿

两份新测试模块的模块注释和具体测试注释注明来源。任务书要求的四类异常、一次切换正例、矛盾/二次切换/全未知/阈值附近反例均使用合成 ZIP/CSV；存储与独立验证端到端也只写 pytest 临时目录。

开发中发现并先确认失败再修复的问题包括：Polars 字典 select、可变参数改变定义、NumPy 隔离名字空间懒加载、整月缺档的误差数组形状、首尾都无锚点、规范化切换缺口、读取时原档损坏、零买量仍应对账、挖掘前预热污染校准、可重设的 view 上界、中断 sync 漏记 pending、未发布库状态、失败修订复用旧 BH p 值、重复 CSV 表头/多余单元格、独立解码忽略显式 UTC 偏移。最后三项 CSV 回归旧代码 3 失败、修复后 3 通过。红/绿日志保存在 tmp，不包含逐行真实行情。

其余验证覆盖：下一日收盘和封存期读取拒绝、全样本均值隐性前视、只读截止时刻、真实读取适配器接合成版本、手算跨界标签、名单限额和哈希、两次调用拒绝、阶段日期锁定、并发闸门、失败仍占名额、定义去重、BH 一致、固定 Holm 名额、组合手算和实际 panelbt 合成回测。

最终复核再补三项先红后绿回归：变异脚本复制测试树后使原生扩展 build_guard 在隔离根报错，现保留原测试树/cwd 和构建校验，只隔离 ashare 源码，并记录 stderr；公开的定义哈希和 Study 设定引用原本能重新赋值，现改为只读属性，防止冒用名单身份或换掉日期设定。没有关闭校验或把导入失败计作变异捕获。

另补 CHECKSUM 二进制标记回归：下载/构建和原档审计均允许 `sha256sum` 的 `*文件名`，已发布版本的独立校验原先误拒绝；合成归档先失败后修复。这只修改独立校验模块，不改变真实原档审计路径或已完成的 16 项变异目标分支。

## R2 历史：2026-11-16 按 UTC 日修订

- 仅双边可比行参与当天两个中位数；一个 `<0.01` 且另一个 `>0.1` 才直接判定。等号为未知。未另加未获指定的样本数阈值：非空双边样本即可求中位数，样本数每天披露；零样本为未知。本次非空日最少双边可比数为 BTC 126、ETH 162。
- 未知日（包括完全缺失日）只从前后两个直接判定且方向相同的日子继承。首尾未知、两侧冲突、全未知或第二次切换拒绝发布。中位数未满足先进入未知日规则；有一致双侧锚点时允许继承，不能将这些日子称为直接通过中位数阈值。
- 所有行按 `create_time` 的 UTC 日共用对齐；切换报告为该日 00:00，即使该日缺少 00:00 行。ETH 03-03 23:55 按 03-03 处理，旧逐行括号废止。全历史切换不是硬编码日期。
- `NULL_VALUE=1`、`INHERITED_ALIGNMENT=2`、`TAKER_MISMATCH=4` 可组合。所选对齐可复算且相对误差 `>0.05` 才打偏差位，等号不打；偏差行原值保留。不可比行不打偏差位，也不称为匹配通过。
- 日历、偏差比例和年统计均按原始 UTC 日期。`mismatch_ratio` 分母为全部行，`comparable_mismatch_ratio` 分母为所选对齐可复算行；无可比行时比例存 0 并同时明确 compared=0。
- 新定义 `crypto-metrics-v2-daily`；原档格式没改，旧 v1 下载计划可直接重建。新内容哈希包含规则、三阈值、每一天统计和年统计，旧内容版本不会按新规则解释。
- 独立校验继续独立解码、`math.fsum` 聚合和 Python `statistics.median` 日判定；与生产 Polars/NumPy 路径分开。行、位和计数精确核对；误差分布及比例的浮点比较允许 1e-12 相对/绝对差，避免两种累加顺序的舍入差，不改变对齐阈值。偏差明细保存在外置版本验证文件的 taker_mismatches，验证错误另存 details。

### 本轮回归与变异

旧实现上先跑修订测试：19 failed / 19 passed（39.55 秒）；修改后该组 38 passed（115.45 秒）。新增单次日界切换、中午切换拒绝、第二次切换、未知日双侧冲突/边缘无锚点、严格阈值、双边可比筛选、离群行保留与质量位、缺午夜行、整天缺行、v1 原档迁移，以及重签哈希后的日统计/年统计/行质量篡改拒绝。重复、空串、缺口和 CHECKSUM 回归保留。完整验收数字在下方本轮结果中更新。复核另发现独立验证未检查 Parquet 字段类型、未复算分段误差分位数；新增两项重签哈希回归先 2 failed（61.95 秒），修复后 2 passed（46.12 秒）。

分段验收回归：本轮 build+verify 合并命令在 840 秒上限终止，但 build 已完成（233.447 秒）。改用同一版本按币独立验证，每段仍限 840 秒，最后强制检查完整币种覆盖、内容版本和分钟版本再合并。分币报告独立命名并标注 complete；部分成功不写全库成功报告。旧接口先 1 failed（6.56 秒），新增范围/合并回归后 2 passed（9.49 秒）。没有删币、删行或跳过比对来缩短验证。

### 阈值依据与解释

旧逐行分布的尾部不能与错误对齐干净分离（历史分布见下）。新日级阈值按审查决定固定为 .01 / .1；逐行质量位独立用 .05，三个阈值不可混用。复算误差公式保持不变。全部实际可复算日（包含继承日）的日中位数分布将在正式版本质量统计中留存；已知预检查里，所选对齐的日中位数仍可能超过 .01，这些日子需要一致双侧锚点继承，不能静默改阈值或删行。

旧规则排查曾从 BTC、ETH 原始 ZIP 各抽 8 个离群点，用 Decimal 重算分钟量；16/16 与生产聚合一致。它支持“逐行异常来自已下载源对象”的结论，没有作为日级规则硬编码日期的依据。

### R2 最终结果（历史）

正式内容版本：`7aab0c3e5935c44baabe37b0ab978410e882b8b593e67294e07a2db52f8c7370`。构建使用已冻结的 4076 份日档（截至 2026-11-14）；未重新下载、未删行。独立验证完整通过，差异 0；日统计、年统计和全历史计数相符。

| 项目 | BTCUSDT | ETHUSDT |
|---|---:|---:|
| 行数 | 651974 | 521140 |
| 完全相同的重复行 | 75255 | 0 |
| 空值行 | 92272 | 92241 |
| 空单元格 | 227486 | 227436 |
| 直接判定日 | 2003 | 1575 |
| 未知日 / 继承日 | 263 | 235 |
| 所选对齐可复算行 | 614668 | 483877 |
| TAKER_MISMATCH 行 | 8886 | 7063 |
| 偏差占全部行 | 1.362938% | 1.355298% |
| 偏差占可复算行 | 1.445658% | 1.459668% |

两币切换时刻均为 `2024-03-04 00:00:00 UTC`（1709510400000 ms）。每天统计见公开 JSON 的 `symbols.<标的>.days`；两币均有 128 个零双边可比日，其余未知日因中位数未直接满足阈值而继承。

| UTC 年 | BTC 偏差行 / 全部行比例 | ETH 偏差行 / 全部行比例 |
|---|---:|---:|
| 2020 | 4 / 0.011395% | 无归档 |
| 2021 | 239 / 0.228376% | 0 / 0.000000% |
| 2022 | 628 / 0.597412% | 591 / 0.562215% |
| 2023 | 500 / 0.475660% | 530 / 0.504200% |
| 2024 | 891 / 0.846307% | 975 / 0.926093% |
| 2025 | 3821 / 3.634997% | 2030 / 1.931181% |
| 2026 | 2803 / 3.060578% | 2937 / 3.206892% |

日中位数的分布（非空双边样本日，包含继承日；所选对齐列）：

| 币 / 对齐段 | p50 | p90 | p99 | max | 另一对齐中位数的最小值 |
|---|---:|---:|---:|---:|---:|
| BTC / 0 | 2.4323413e-09 | 6.1874455e-05 | 0.0027937499 | 0.028583287 | 0.19185134 |
| BTC / 1 | 0.00027286143 | 0.01236731 | 0.027237892 | 0.044841372 | 0.23449269 |
| ETH / 0 | 3.0829808e-09 | 7.7018824e-05 | 0.0025252811 | 0.022119824 | 0.17615173 |
| ETH / 1 | 0.00032746954 | 0.010230331 | 0.023196655 | 0.085886872 | 0.15884216 |

所选日中位数的尾部仍会超过 .01，不能称为全体日子直接分离；直接判定和双侧继承数量明确分开。错误对齐日中位数的最小值仍高于 .1。逐行 .05 用于披露质量，不放宽日判定。

相关测试 `pytest -q -n 1 test/crypto test/paths`：**542 passed**，100.02 秒，0 skipped；包含 C8 77 项（metrics 49、timing 28）和既有 C6 77 项。一条 Starlette/httpx 弃用警告与此次改动无关。`test/paths/test_static.py` 随完整路径测试通过；ruff check 与 format --check（843 文件）通过。

下载沿用上轮 4076 份原档，47,565,401 bytes，历史下载约 1074.937 秒（上轮超时段按 840 秒上限计）。本轮 build 233.447 秒；独立分币 verify 为 BTC 276.784 秒、ETH 172.815 秒，合计 449.599 秒。137 个月分区，Parquet 55,203,251 bytes；数据根总占用 172,885,156 bytes，含原档、旧失败审计、两个分币报告和完整验证报告。首次合并命令的 840 秒中断另记，不伪装成成功验证耗时。

变异检查 **23/23 被捕获**，0 个导入/语法错误冒充捕获，耗时 121.755 秒。变异 0 与隔离对照各 77 项通过，四个模块导入路径均来自本 worktree，原生扩展 build_guard 保留。覆盖日中位数（含均值替换）、只用双边可比、两个严格阈值、双侧继承/边缘约束、单次切换、UTC 日界、质量位阈值、窗口结束可知、精确去重，以及原有防火墙、前视、跨界、阶段一次性、名单哈希与定义去重保护。

C6 本轮重新计算后逐位通过：27 个输入数组 dtype/shape/bytes 相同；8/8 组、288 项 JSON 结果字节相同；8 项 Holm float64 字节相同。每格仍为 4000 次模拟，保留功效计算。旧基线 1401.518 秒；本轮从冻结输入到最后组输出的墙钟 3659.291 秒，包含两组 840 秒中断后的检查点续算，以及穿插的 metrics/测试验收，不能作为纯计算性能比较。C6 本轮输入和明细只在 tmp/c8-daily-c6-after；公开文件只记统计与逐位比较。

最终 git diff --check 通过；分支仍为 agent/codex/c8-metrics-timing、HEAD 仍为 f5d5955，未提交、未推送。DESIGN 6.10.2 再次逐字核对与 d402835 一致。没有真实 C8 验证/确认调用，没有读写主目录 data/mining/。

### R2 当时的限制与待清理文字（前两项本轮已解决）

- 按要求原样同步的 DESIGN 6.10.2 最后一条仍写“逐行全部对上才算通过”，与该节新增质量位规则冲突。本轮遵从用户明确决定：独立校验要求全部行的归一化、质量位和统计正确，偏差行本身不使验证失败。建议主干后续清理旧句；本 worktree 没有擅改同步原文。
- 设计未给“可比行太少”的具体最小样本数，当前按明示中位数规则计算，并披露每天双边样本数；不添加第三个判定阈值。有限样本和中位数不能证明日内每个瞬间的时间语义，少数反向行按质量信息保留。
- AST 隔离和有限截断自检用于防常见研究误用，并非恶意 Python 安全沙箱，也不能数学证明任意因子无前视。数学等价公式自动去重不在本实现能力内。推断沿用 C6 工作模型与有限网格近似。
- 真实择时演示仍只有挖掘期，未重新调用真实 C8 验证或确认；原演示三个因子不依赖 metrics。
- nice 调整被本执行环境拒绝，保留 nice 调用并限制数值线程 1、下载 ≤4、重计算 ≤2。此次未改 Rust，仅执行任务指定的 Python 静态检查和相关测试。

## 文件清单

| 范围 | 文件 |
|---|---|
| metrics 新增 | `src/ashare/data/crypto/metrics.py`、`verify_metrics.py` |
| timing 新增 | `src/ashare/factor/timing/__init__.py`、`settings.py`、`definition.py`、`view.py`、`inference.py`、`ledger.py`、`study.py`、`backtest.py` |
| 现有源码调整 | `src/ashare/paths.py`（只加一行）、`src/ashare/factor/crypto_eval.py`（可选 boundary） |
| 新增验收脚本 | `bench/c8_c6_compat.py`、`crypto_metrics_validation.py`、`timing_demo.py`、`mutations_c8.py` |
| 现有脚本调整 | `bench/crypto_factors_report.py`（可选检查点）、`bench/mutations_crypto_factors.py`（两处锚点同步） |
| 新增合成测试 | `test/crypto/test_metrics_c8.py`、`test_timing_c8.py`、`test_c8_review.py` |
| 文档 | 新增 `docs/C8_NOTES.md`、`TIMING_EVAL.md`；更新 `docs/DESIGN.md`（同步 6.10.2 后仅清理一处授权残留措辞）、`CRYPTO_DATA.md`、`DATA_LOCATIONS.md` |
| 公开汇总 | `bench/results/crypto_metrics_validation.json`、`timing_demo.json`、`c8_c6_bitwise.json`、`mutations_c8.json`、`c8_checks.json` |

## 原逐行规则的历史证据（已由 d402835 取代）

以下是规则修订前的验收记录，不代表当前发布状态。原 match=.005 / reject=.05 逐行判定拒绝全历史，促成此次按日修订；历史错误阈值、候选括号和失败计数仅留作决策依据。新结果见上方本轮结果。

机器可读产物：`bench/results/crypto_metrics_validation.json`、`timing_demo.json`、`c8_c6_bitwise.json`、`mutations_c8.json`、`c8_checks.json`。最终数字汇总见下文。checks 的 exit_code 表示验收脚本是否执行完成；metrics 的 `ok=0,built=0` 单独明确数据验收失败。

最终相关测试 `pytest -q -n 2 test/crypto test/paths`：526 passed，210.09 秒，无跳过；其中 C8 新测试 61 项（metrics 33、timing 28），C6 原有 77 项包含在内。一条警告来自 Starlette/httpx 测试客户端的弃用提示。`test/paths/test_static.py` 随全套路径测试通过。ruff check、ruff format --check（843 文件）、git diff --check 均通过。

变异验收：16/16 被实际测试断言捕获，0 个导入/语法错误冒充捕获，耗时 307.480 秒。执行前核对四个关键模块均来自本 worktree，变异 0 和隔离对照各 60 项通过；原生扩展 build_guard 保留。覆盖对齐、第二次切换、可知时刻、冲突去重、阈值边界、防火墙、隐性前视、跨界标签、验证/确认一次闸门、名单哈希、定义缓存/名字归一化、参数冻结和两项公开身份保护。随后新增的 CHECKSUM 标记回归只涉及独立校验器；变异目标文件和被检测分支未再修改。

C6 真实逐位验收已通过：27 个输入数组 dtype/shape/原始字节相同；8/8 组 JSON 输出字节相同，共 288 项，每格 4000 次模拟，保留原功效计算；8 项 Holm 的 float64 字节相同。改动前计算 1401.518 秒、改动后 1769.529 秒，耗时受共享机器负载影响，不作为性能比较。前后输入冻结同一个小时/季度信号版本，身份文件和数值明细仅在 tmp；公开报告只含比较计数。

全历史独立审计完成（`ok=0, built=0`）：

| 项目 | BTCUSDT | ETHUSDT |
|---|---:|---:|
| 日档 | 2266 | 1810 |
| 去重后行数 | 651974 | 521140 |
| 完全相同的重复行 | 75255 | 0 |
| 至少一个空值的行 | 92272 | 92241 |
| 空单元格 | 227486 | 227436 |
| 缺口段 / 缺行 | 156 / 634 | 10 / 140 |
| 至少一种对齐可复算 | 614671 | 483879 |
| 两种对齐均不匹配的行（下界） | 85503 | 81403 |
| 明确投票序列的方向变化次数 | 85 | 65 |
| 按候选分段对账的不一致行 | 86004 | 81975 |

候选切点均为 UTC 2024-03-04 00:00:00（1709510400000 ms）。BTC 的局部证据区间是该单点；ETH 为 2024-03-03 23:55:00 至 2024-03-04 00:00:00。两币全历史严格判定都失败，因此以上不是已发布版本的切点。按候选切点分段的**所选对齐**相对误差分布如下；另一个对齐的全套分位数也在 JSON 中，分位顺序为 min/p01/p50/p90/p99/p999/max。

| 币 / 段 | p50 | p90 | p99 | p999 | max | 另一对齐 p50 |
|---|---:|---:|---:|---:|---:|---:|
| BTC 旧段（结束） | 2.59463e-9 | .00128936 | .0254047 | .163412 | 170.303 | .378972 |
| BTC 新段（开始） | .000521222 | .0191494 | .0835407 | .278328 | 12.5837 | .429436 |
| ETH 旧段（结束） | 2.56407e-9 | .00282024 | .0339087 | .171478 | 143.444 | .376004 |
| ETH 新段（开始） | .000560907 | .0164299 | .0732394 | .243589 | 27.6857 | .395768 |

4076 份原档全部通过独立 SHA-256/CHECKSUM 核验，压缩原档 47,565,401 bytes。下载约 1074.937 秒（首次中断按 840 秒上限计、续跑实测 234.937 秒）；严格 build 38.393 秒，独立 verify 原档审计 181.039 秒。外置 metrics 总占用 106,142,729 bytes，包含原档、清单和逐行审计；合格 Parquet 为 0 bytes。确定不一致至少 166906 行，逐条误差和相反投票切换清单只写外置 `verification/raw-audit.json`。按候选分段的 167979 个不一致是诊断数量，不与上述下界混称。

真实择时演示仅 BTCUSDT 挖掘期 2020-01-01 至 2025-05-01，持有 7 天、4000 次模拟；临时设定与台账在 `tmp/c8-timing-demo/`，C8 验证/确认调用均为 0。三个定义共耗时 98.443 秒，每个完成 11 个前视截断检查。未读取主目录 mining，也未使用不合格 metrics。

| 定义 | 有效样本 | t | 保守 p | BH q |
|---|---:|---:|---:|---:|
| 7 日动量 | 1933 | .166641 | .917271 | .917271 |
| 30 日动量 | 1910 | .829656 | .540115 | .810173 |
| 7 日资金费均值 | 1934 | −1.380314 | .176206 | .528619 |

