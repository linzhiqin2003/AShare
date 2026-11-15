# C4：两钱包实现记录

## 先写的确定性反例（实现前）

以下数值全部来自合成输入的手算，费用未说明时为零；数量乘数为 1。
来源：DESIGN 4.17 C4 验证①–⑭，以及设计审查 P1-1–5、P2-7–11。

| 编号 | 输入与独立期望 |
|---|---|
| ① | 初始 W=100，100 买 1，收盘 150，IMR=.2：U=50、IM=30、可划=70；请求出 100 实出 70，差额 30。 |
| ② | W=100，100 买 1，收盘 80，IMR=.375：U=−20、IM=30、可划=50。 |
| ③ | 初始 10，100 买 1，下一开盘 90，MMR=.01：无入金 E=0≤MM=.9，开盘强平；盘前入 10 后 E=10，保留 1。 |
| ④ | 初始 10，100 买 1，同一收盘资金费 10，E=0，收盘强平；下一行入 10 后仍保留该强平记录，不能回溯救援。余额正好零未穿仓，允许入金。 |
| ⑤ | 两边初始 100/100，空仓互转 +30/−20：总权益一直 200，总收益一直 0，每行两边现金流之和为 0。 |
| ⑥ | 合约初始 10，100 买 1，下一开盘 80 穿仓；再请求现货转入 50：两边实际流均为 0，付款方现金不减。 |
| ⑦ | 初始 1000，成交价 100/120/130/140/150/160，目标 1/2/1/−1/0/1：均价 100/110/110/140/0/160；W=1000/1000/1020/1050/1040/1040；U=0/20/20/0/0/0。 |
| ⑧ | 初始 100，100 买 1，费率 .01，收盘 110，资金费 2：W=97、U=10、权益 107。 |
| ⑨ | 初始 100，100 买 10，下一开盘 80：已实现 −200、缺口补记 +100；W=U=权益=0，缺口 100，不是入金。 |
| ⑩ | W=300，100 买 10，下一行 IMR 从 .1 升 .2，上一结算 100、下一标记开盘 150：IM=200、可划=100，不能取旧率或当前开盘。 |
| ⑪ | 两钱包 50/100，价 100，目标现货 +1/合约 −1：现货实际 .5，follow_spot 合约实际 −.5（合约内部乘数 .1），残余 0。 |
| ⑫ | 现货现金 100、成交 110、目标 1、步长 .001：实际 .909，现金 .01；不是 .909090…；非加密市场仍保留原结果。 |
| ⑬ | 首行无上一根已完成价：任何一腿非零目标报错；首行零目标允许。新上市列同理。 |
| ⑭ | 多组随机输入逐行独立验证 W+ΣQ(M−e)=权益；手算结构同①–⑩，不以权益倒推出 W。 |

## 实现中的选择

1. **原记账与 W/U 分离**。Rust `wallets::Ledger` 挂在启用交易所强平的输出上，成交、交割、资金费、清算、缺口补记分别更新；绝不通过 `权益−U` 倒算 W，也不回写原有成交判断。一般期货及现金/保证金账户不分配该账本。配置现金利息/融资利息时照原符号进 W，以保持守恒。
2. **动态流与旧静态流区分**。`flow=None` 保留旧路径；显式 `flow=0` 也要检查静态流冲突。检查配置的全部行，不只当行。负请求受新限额截断，旧静态流保持原限制。差额列沿用“实际−请求”，拒收正入金时为负数。
3. **浮点边界**。独立语料发现结算现金比 W 低几个 ulp 时，全部提现会制造负现金。可划限额另与 `max(0, cash−IM)` 取小；精确算术下不改变设计公式，仅消除凭空负债。没有放宽断言、调整权益或引入经济金额容差。
4. **统一币单位与可用价格**。目标按基础币数或合计权益权重；有现货列就用已完成现货收盘，两腿共价。支持 `coin_units` 和乘数，包括千币报价；首行、新上市无价时拒绝非零目标。下架残仓没有信号价时，诊断用上一已知持仓估值，不能拿这个回退去下新目标。
5. **成对操作**。复制全部 State 数组冻结决策，在任何一腿推进前确定两边共同接受的金额。失败后整个协调器锁死，不尝试事后继续或暴露单边结果。`follow_spot` 使用成交后持有量，并按目标比例缩放；合约成交不足保留残余。日频关闭两边顺延，公告退出为例外。
6. **严格残余阈值**。阈值单位 USDT，比较各币绝对名义之和；默认 1e−8，显式放开仍标 `valid=False`。不把多币方向相抵作为对冲成功。救援定义为现货到合约的非零请求未足额到账；“随后强平”只记录时间先后。
7. **快照**。新合约账本使用内核格式 4；其他账户继续原格式 3，旧格式 2/3 可读、字节兼容。两钱包格式 1 使用校验和、完整行情指纹、时间/单位/目标/策略参数指纹；恢复先建临时对象。策略协议用 `config()` 覆盖全部固定参数，`state()/restore()` 覆盖全部可变决策状态；普通函数可运行，无协议时不能快照。带协议函数的闭包用状态交换包装隔离，普通有状态对象深拷贝。
8. **参考实现**。`reference_wallets.py` 自己维护未结转的 W、数量、开仓均价和标记 U；从设计文字计算风险/划转。开仓缩放枚举可行数量格点，不复用内核二分；不导入内核或 `reference_crypto.py`。保留原 500 组合成语料，新增 120 组双币/三币、十进制 .1/.001 格点、日频 0–5 时调仓、跨币受阻与现货下架语料；另有 W+U、千币单位和双边基差手算。
9. **性能选择**。保留每行复制状态、Python 回调、先现货后合约的语义。原生模拟器拥有不可变的公司行为索引，仅在构造时验证整段单调性，避免每次 `step/state` 再扫描 T 行；仍执行形状检查。成交后直接使用既有只读持仓视图，减少重复状态读取。
10. **实数据隔离**。通过 `rsync -a` 把源数据复制到忽略的 `tmp/crypto`，只读副本。脚本默认相对 `data/crypto`，接受参数/环境变量；公开产物只含汇总数字、输入版本指纹和合成测试，不包含行情序列。没有改数据层或 C5 准备脚本。

11. **返工后的数量边界**。数量相减、容量与全平判断使用 `1e-9 × step` 容差，币数换张数使用 `1e-9` 张的合法整数边界；现金比较保持严格。参考实现用相邻格点距离独立判断，未复制 Rust 算法。
12. **MarginBand 语义扩充（审查 P2-6）**。默认 `pre_fund=True, buffer=.02`；在调仓行，先冻结目标，用上一结算价、当前保证金率算新目标 IM，取新旧 IM 较大值，再加预计手续费及新目标名义×buffer。请求取带规则与资金缺口的较大值，受现货现金限额约束；`pre_fund=False` 恢复原带规则。这是 DESIGN 中 MarginBand 的显式扩充，按任务要求不改 DESIGN。
13. **跨币退出与残余归因**。下架例外传币列掩码，非调仓小时只改变涉及币的合约目标；新增取整/成交不足残余数组和最大值诊断。比例 1 下二者和为总残余，比例不同还含策略主动敞口。

## 设计问题

未修改 DESIGN.md。以下选择补足设计未指定处，避免隐式推断：

- **旧强平快照不可重建均价**：C1 快照缺原始均价。按返工要求保留旧恢复/继续/重新编码路径，账本标为不可用；全部新增钱包状态字段（含 notional/initial_margin）和显式 `flow=` 报错，不影响旧字段。旧强平快照不再从兼容语料排除。
- **重复币的合约选择**：每个钱包每币一列。季度对冲可选一个具体季度，内核照常交割/下架；不自动选择同币多个季度，也不在 C4 重做 C3 的多列换月路由。需在调用前明确合约，否则报错。
- **初始余额必须各自为正**：沿用 `Config.initial_cash > 0` 合同；未扩大到零初始钱包。之后可以划到零，非爆仓零余额钱包可入金。
- **C1 的空仓费用负债边界**：交割费或配置利息在没有持仓可强平时造成的负现金，原模型保留负债并锁存破产，不生成缺口补记。C4 沿用它，W+U 等于原负权益；只有实际 `bankruptcy_deficit` 事件才补记到 W。不能为让所有破产状态都等于零而改变第一批输出。
- **划转模型没有交易所原文证据**：公式只标未验证假设，不声称相对实际 `maxWithdrawAmount` 更保守。IM×1.5 是收紧这一模型的敏感性，不能当作真实交易所保证金制度。
- **快照中的用户状态**：要求 JSON 可序列化且 `state/restore` 完备；外部 I/O、副作用、隐含全局状态不在协议保证内。恢复的函数包装仍串行调用原函数，不能和原函数并发执行。
- **现货最小名义、固定数量快照、买入扣现金费用**沿用第一批的已知偏差。C4 修复 shares 格点和完整零头卖出，没有悄悄更换历史规则。
- **资金费例子的含义**：原例子主要残余来自保证金不足，不能归为本金小或取整。现采用调仓前补资、5 倍杠杆和现金储备；额外按最小下单名义延后小额增仓，否则即使资金充裕也会留下未成交整张。理论调仓名义仍为权益×h，不在延后行生成新目标。实际行情可能跳空超过费用/缓冲估算，因此补资不是无条件成交保证；例子独立断言未成交整张残余为零，仍如实报告取整敞口。
- **性能对比口径**：当前 C3 `crypto_run.simulate` 在无事件/换月时实际走一次原生 `run` 的快速路径，并非全部逐行。两钱包要执行 59,903 次用户决策边界及 119,806 次原生 step，因此同时报告单账户逐行基线、净 Python 额外耗时和 profile；不把批量原生耗时冒充协调层耗时。

## 回归与验收

### 先红后绿

实现前写①–⑭，旧版本 33 项失败记录在 `bench/results/crypto_wallets_initial_red.txt`。
另外开发中发现的错误分别先加测试，在未修代码上确认失败，再修复：

| 来源 | 回归与处理 | 失败记录 |
|---|---|---|
| 独立语料 seed 5 | 全额提现的 ulp 差制造假爆仓；现金安全上限 | `crypto_wallets_roundoff_red.txt` |
| 首次真实脚本调用 | `bench/profile.py` 遮蔽标准库 profile；导入前移除脚本目录 | `crypto_wallets_cli_red.txt` |
| 全平规则复核 | 容量先取整把 .15 零头截成 .1；完整关闭判断先于格点 | `crypto_wallets_oddlot_red.txt` |
| 最终下架边界复核 | 缺信号的残仓产生 NaN 暴露、无法快照；仅估值回退 | `crypto_wallets_boundaries_red.txt` |
| 最终兼容摘要复核 | 初版摘要漏比旧 crypto coverage；改为摘要全部诊断/coverage 键，并加摘要本身的反例 | `crypto_wallets_compat_counter_red.txt` |
| 最终十进制格点复核 | .07/.01 的 ceil 误进位、.29 格点少买一格；初版改为零起点格点及商的一 ulp；本轮改为步长相对容差，现金检查始终严格 | `crypto_wallets_final_boundary_red.txt` |
| 最终诊断复核 | 合并输出遗漏 C3 清算费/缺口/时间等字段；补全相同语义 | 同上 |
| 最终输出结构复核 | 空批次遗漏已启用 W/U 的 `(0,T)` 数组；按市场启用标记构造 | `crypto_wallets_empty_batch_red.txt` |
| 最终差异复核 | 市场标记误加到 Targets；移除无关字段、只保留 Market 标记 | `crypto_wallets_marker_red.txt` |
| 最终快照协议复核 | deepcopy(function) 仍指向同一闭包；隔离恢复状态 | 同上 |

最初手算夹具使用了只清零合约费用的旧 ZERO，现货默认费用仍非零；明确清零两类费用，手算期望未改。Rust 编译中遗漏新增字段的解构已修正并通过编译回归，不影响运行期模型。

### 审查返工第 1 轮：先红后绿

完整阅读 `.review/review_c4.md` 后，先保存当前版本 9 项失败在
`bench/results/crypto_wallets_review_red.txt`，再改生产代码。手算期望未随实现修改。

| 意见 | 固定反例与期望 | 新回归（`test/test_crypto_wallets_review.py`） |
|---|---|---|
| P1-1 | 步长 .1，1→.8 应持有 .8、最终权益 1080；.1 经 1.5 拆股且容量 .15 应全平、现金 1005 | `test_review_p1_1_decimal_reduction_native_and_independent_oracle`（原生/参考两例）、`test_review_p1_1_decimal_capacity_full_close_after_split` |
| P1-2 | .3/.1 必须为 3 张；价格 100→200 后合计权益仍 200 | `test_review_p1_2_decimal_coin_contract_integer_boundary`（两种 hedge） |
| P1-3 | BTC 行 25 退出，ETH 行 24 受阻的 5 张不能重试；最终权益 300 | `test_review_p1_3_daily_delist_only_rehedges_affected_coin` |
| P2-4 | 自定义策略 size .5 改 1，恢复必须拒绝 | `test_review_p2_4_snapshot_checks_fixed_strategy_config` |
| P2-5 | 合成旧强平快照恢复后权益 100/100/120，旧格式重新编码字节相同；新账本字段/flow 报错 | `test_review_p2_5_old_liquidation_snapshot_legacy_path` |
| P2-6 | 原反例末行应持现货 1、空合约 1000×.001；两次预补资手算 2 与 30.4 | `test_review_p2_6_prefund_rebalance_target_before_spot_execution`、`test_review_prefund_hand_amounts_off_switch_and_cash_cap` |

另测严格现金上限、真实非整数不得进位、缺 config 协议拒绝快照，以及最小订单名义造成的
整张不足必须计入成交不足残余。旧快照夹具 `test/fixtures/c4_legacy_snapshot.txt` 由冻结的
旧扩展生成，只含合成输入，不依赖本地缓存才能测试。

补资估算的防御性单元测试补充：无仓且目标为零的未上市列，其 NaN 结算价不能使其他币的补资
计算失效；有仓或目标非零而缺结算价时须在任一腿执行前报错。先以独立 Python 补资方法测试
记录 2 项失败在 `crypto_wallets_prefund_boundary_red.txt`（第一轮时该方法不调用 Rust；
第三轮新增只读成交判定后，测试提供真实模拟器作为查询对象），回归位于
`test/test_crypto_wallets_prefund_boundary.py`。 独立参考也单独以手算 16 核对，失败记录在 `crypto_wallets_reference_prefund_red.txt`。这里直接向估算方法/参考对象注入缺价；正常 Market 构造器已拒绝缺失 futures settlement，不能据此声称有效市场会出现该故障。

### 独立语料与兼容性

500 组 × 12 行合成随机语料逐行比较两腿权益、持仓、W/U、划转、合计权益和残余；测试最后断言全部要求分支出现。计数：

| 分支 | 次数 |
|---|---:|
| 现货到合约 / 合约到现货 | 1186 / 1172 |
| 截断 / 完全被拒 | 1509 / 584 |
| 救援成功 / 救援失败 | 32 / 24 |
| 爆仓后入金被拒 | 128 |
| 反手 / 交割 / 下架 | 358 / 149 / 130 |
| 有残余暴露 | 4789 |

新增 120 组双币/三币十进制日频语料，分支实际计数：`{"multiple_coins": 120, "step_0.1": 60, "step_0.001": 60, "daily_trade_hour": 120, "spot_delist": 120, "unrelated_blocked_not_retried": 120, "prefunded_target": 432}`。每组逐行比较两腿权益、持仓、流、合计权益和残余，末尾断言所有分支非零。

`bench/crypto_wallets_compat.py` 从修改前冻结的 Python 包和原生扩展分别启动进程，比较现金、保证金、持仓期货、加密合约各 100 组：`run/run_batch/step/snapshot` 共 1,600 组输出逐位一致，400 个旧快照（含 100 个强平快照）重新编码字节与恢复输出一致，400 个旧 NPZ 迁移和 DataFrame 列集一致。新 W/U 另测单次、批次、逐步、快照恢复，不纳入旧字段摘要。

### 变异与完整回归（第一轮返工结果）

- `ruff check .`、`ruff format --check .`、`cargo fmt --all --check`、`cargo clippy --workspace --all-targets -j 2 -- -D warnings` 通过。
- 全部 Rust：33 项通过，0 失败。
- `pytest -q test/test_*.py test/crypto test/backtest test/factor/test_groupbt*.py -n 4`：1761 项通过，0 跳过，40.28 秒；12 个已有 groupbt 空切片均值告警，未改断言。
- 原审查六项对应 9 个先失败反例；补资估算的 3 个防御性缺价边界、旧快照保证金元数据占位零值的 1 个边界（`crypto_wallets_legacy_metadata_red.txt`），也均先失败再修。
- 旧版/新版四类账户各 100 组，`run/run_batch/step/snapshot` 共 1,600 项输出逐位一致；400 个旧快照（含 100 个强平快照）恢复和重新编码通过，400 个旧 NPZ/列集兼容。
- Rust 源指纹：`2ec28f041f6685d6ed372ca66d4b071f34da66c4f2df47b7e663ed90aedde03c`。每处 Rust 变异都重建、确认工作树导入与指纹；恢复后重建并再次通过功能回归。

第一轮最终 **22/22** 处变异被测试抓住，构建/语法/导入失败不计为抓住。明细：

| 变异 | 失败测试数 |
|---|---:|
| `spot_decimal_lattice_uses_legacy_rounding` | 6 |
| `dynamic_flow_after_open_check` | 2 |
| `unrealized_profit_transferable` | 1 |
| `im_current_mark_not_previous_settlement` | 2 |
| `im_previous_rule_row` | 2 |
| `reduction_changes_entry` | 26 |
| `reversal_keeps_old_entry` | 24 |
| `omit_deficit_credit` | 8 |
| `internal_flow_counted_as_return` | 3 |
| `unequal_transfer_amounts` | 8 |
| `bankrupt_accepts_deposit` | 1 |
| `follow_target_not_actual_fill` | 3 |
| `omit_snapshot_entry` | 2 |
| `reference_wrong_funding_direction` | 2 |
| `review_p1_1_subtraction_one_ulp_only` | 3 |
| `review_p1_1_capacity_no_quantity_tolerance` | 1 |
| `review_p1_2_raw_contract_truncation` | 4 |
| `review_p1_3_global_exit_override` | 2 |
| `review_p2_4_ignore_fixed_config` | 1 |
| `review_p2_6_skip_prefunding` | 3 |
| `review_p2_5_reject_legacy_liquidation_snapshot` | 2 |
| `prefund_unused_missing_mark_poison` | 1 |

### 真实例子和手算

只读副本范围 UTC `[2020-01-01 00:00, 2026-10-31 23:00)`，59,903 小时。
初始 30,000 USDT，现货/合约 24,000/6,000，杠杆 5，目标保证金率 .5、现金储备率 1，
h=.4。每日 UTC 00:00 决策，每次新目标的每腿名义=合计权益×h；采用
`follow_spot` 和默认开启预补资的 `MarginBand(.5,.25,1)`。
新增整张名义小于最小下单名义 1.5 倍时，整组调仓延后，防止一腿小额加仓被过滤；
该判断只用上一结算价，不能保证未知大跳空下成交。BTC 本次延后 162 次，
单腿目标名义峰值约 16,007.44，低于第一档 50,000。

| IM 划出倍数 | 年化 | 资金费收入 USDT | 费用 USDT | 最大回撤 | 划转 | 救援失败 / 强平 | 取整残余峰值 USDT | 成交不足残余峰值 USDT | valid |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 4.3048% | 11132.667640 | 1085.894123 | 1.0863% | 29 | 0 / 0 | 108.973050 | 0.000000 | True |
| 1.5 | 4.3048% | 11132.667640 | 1085.894123 | 1.0863% | 30 | 0 / 0 | 108.973050 | 0.000000 | True |

两种限额下现金都能支持完整对冲，故实际持仓和收益相同，划转次数不同。
残余限额为 250 USDT；另外逐行断言取整残余不超过一张合约名义、成交不足残余小于 1e−7，
并断言第一档未超和 `valid=True`，不靠放宽残余限额隐藏未成交整张。
费用含 commission/tax/other_fee/slippage_cost/impact_cost；清算/交割费用不重复计。
原例子的 196.05/219.19 残余主要是保证金不足，该结果已由本表替代。

手算全部从原始输入独立计算：

- 新目标预补资：固定 .01 BTC，上一原始标记收盘对应新目标 IM=.2×名义、手续费=.0004×名义、缓冲=.02×名义，初始合约资金=.1×名义。期望划转 34.93222992000000，实际 34.93222992000000 USDT。
- 带规则救援（单独固定币数场景，显式关闭预补资以隔离带规则）：从原始成交 open、标记 close、资金费递推得 W=149.56692731413088、U=-64.41001329140003、名义=354.5600132914；`.5×名义−(W+U)`=92.12309262296917，实际相同，差 0.0。
- 2024-06-01 资金费日：固定空 .01 BTC，08/16/次日00 三笔 `.01×Σ(原始费率×关联标记价)`，收入 0.20488769058528228 USDT，内核同值；另核实持仓始终 −.01 BTC。未从被测权益/资金费数组生成期望。

### 性能

BTC+ETH 59,903 小时；构造市场和读数据不计入，协调器构造、推进、结果诊断计入。
第三轮返工后重新测量，每项 3 次取中位数：

| 指标 | 结果 |
|---|---:|
| 两钱包一次跑完 | 3.270490 秒 |
| C3 `crypto_run.simulate` 原生快速路径 | 0.007974 秒 |
| 单账户 Simulator 逐行循环 | 0.224172 秒 |
| 两钱包总开销/行 | 54.596 微秒 |
| 扣除两个单账户循环后的额外开销/行 | 47.112 微秒 |
| 相对单账户原生快速路径 | 410.15 倍 |

原始实现 11.1876 秒，曾发现每次 step/state 扫描完整公司行为索引的 O(T²) 问题，
已移到构造时验证。返工新增逐行残余归因，当前 profile 的主要额外耗时是冻结状态、
Python 协调以及小数组转换；原生 step 只占较小部分，细项见 performance JSON。
“额外开销”是 `(两钱包−2×单账户循环)/行数`，包含查询与调用差异，非精确的纯 Python CPU 计数。
与一次原生 run 相比仍明显慢于 5 倍，原因是保留逐小时回调、冻结决策和双边同步边界。
原始对照保留 `crypto_wallets_performance_before.json`。

## 本地工作日志

本任务明确禁止修改 worktree 外文件，因此不写全局 worklog，阶段日志在本项目内记录。

- [16:30] AShare C4 — 已完成 Rust 动态流与 W/U、两钱包协调器、独立 500 组语料、1,600 项逐位比较和真实例子；补齐提现浮点、全平零头、下架估值与快照闭包回归，变异与最终验收进行中。

- [16:47] AShare C4 — 最终功能代码通过 1,739 项 Python、33 项 Rust 和全仓库静态检查；真实样本两钱包 2.851 秒，BTC 例子年化 7.0799%/7.3025%，两次独立手算吻合。最终变异复测与下游回归继续执行。
- [16:57] AShare C4 — 最终 14/14 变异被抓住，恢复源码后功能回归通过，全部旧计数纳入后的 1,600 项兼容对比通过；扩展回归 963 过、1 项 ps 权限受限，测试服务清理 SIGINT 同样被沙箱拒绝，已明确报告。

- [17:40] AShare C4 返工 — 六项审查反例先失败后修复；1,757 项相关 Python、33 项 Rust、ruff/clippy 通过，400 个旧快照恢复逐位一致；真实例子已消除成交不足残余，20 处变异执行中，补资缺价边界待源码恢复后修复。
- [18:04] AShare C4 返工 — 六条审查意见及补资缺价、旧快照元数据边界均完成；22/22 变异被抓住，全量相关回归、旧强平快照逐位兼容和原始输入手算通过，真实例子两组 valid=True、成交不足残余为零。

## 文件清单

- Rust：`crates/panelbt-core/src/wallets.rs`（新增），以及 `lib.rs`、`crypto.rs`、`futures.rs`；绑定 `packages/panelbt/src/lib.rs`。
- Python 内核封装：`packages/panelbt/python/panelbt/api.py`、`models.py`、`step.py`；独立参考新增 `reference_wallets.py`。
- 应用协调：新增 `src/ashare/backtest/crypto_wallets.py`；`crypto.py` 仅增加导出和现货标记。
- 合成验证：`test/test_crypto_wallets.py`、`test/test_crypto_wallets_contract.py`、`test/test_crypto_wallets_oracle.py`、`test/test_crypto_wallets_compat.py`、`test/generators_wallets.py`；返工新增 `test/test_crypto_wallets_review.py`、`test/test_crypto_wallets_prefund_boundary.py` 和合成旧快照 `test/fixtures/c4_legacy_snapshot.txt`。
- 实验：`bench/crypto_wallets_example.py`、`bench/crypto_wallets_compat.py`、`bench/mutations_crypto_wallets.py`。
- 规范与说明：`packages/panelbt/SPEC.md`、`docs/CRYPTO_BACKTEST.md`、本记录。
- 结果：`bench/results/crypto_wallets_{compat,corpus,example,performance,performance_before,validation}.json`、`mutations_crypto_wallets.json`；`crypto_wallets_*_red.txt` 均为合成反例/导入诊断的失败记录，已去掉机器路径。


## 第二轮返工记录

四项审查意见均先补确定性测试，再改实现。基线得到 **10 个失败**（四项新增测试展开
9 个用例，加原闭包隔离测试 1 个）；日志为 `bench/results/crypto_wallets_review2_red.txt`。
零缓冲用例分别检验原生协调器和独立参考实现，期望来自手算，不由对拍结果推导。

| 意见 | 修复位置与行为 | 红测试 |
|---|---|---|
| P1 退出币合约漏重试 | `crypto_wallets.py::_exit_target/step/snapshot/restore` 增加逐币退出状态，两腿持仓及挂起订单都清除才释放；日频仅更新退出币的合约列，恢复后也继续退出 | `test_review2_exit_retries_blocked_future_only_until_both_legs_close`，含中途快照恢复 |
| P1 零缓冲入账舍入 | `crypto_wallets.py::_prefund` 按内核结算现金核验实际浮点加法，向上修正请求后仍按现货可划转额截断；参考实现按自己的 W 入账和持仓估值独立核验 | `test_review2_zero_buffer_prefund_credits_seven_contracts_hand_expected`，原生/参考两个用例 |
| P1 策略状态误进指纹 | `crypto_wallets.py::_strategy` 仅以策略身份及 `config()` 固定参数作指纹，可变状态从快照载入；恢复仍隔离调用方策略 | `test_review2_restore_advanced_policy_checks_config_not_mutable_state`；原 `test_function_protocol_restore_does_not_mutate_callers_strategy` 删除手工清零，保留并检验推进后的调用方计数不变 |
| P2 旧快照展示失败 | `models.py::_MissingWallet` 为旧状态/结果单独过滤 dataclass 字段并提供对应 repr，显式访问账本字段仍报错，普通 State/Result 元数据不变 | `test_review2_legacy_dataclass_display_omits_unavailable_fields`，状态/结果 × repr/asdict 四个用例，同时检查 print |

### 实现中的选择与设计问题

- **零缓冲不是放宽资金约束**：手算 0.7 BTC / 0.1 = 7 张，IM=3.5，理论补资=3.5−1.05=2.45。
  实际浮点 IM=3.5000000000000004，而 1.05+2.45=3.5。修正后的请求为
  2.4500000000000006，实际入账 3.500000000000001，持仓为 0/−7/−7，权益始终 101.05。
  全部请求仍在两腿推进前冻结，未调整 Rust 的现金/保证金比较。
- **可划转额仍是硬上限**：另加 `test_review2_zero_buffer_rounding_topup_never_exceeds_spot_cash`。
  若现货只有 2.45，即使只差一个 ULP，也只能划转 2.45，合约仍为 −6 张，记录
  `reason=capacity`、`rescue_failed=True`，现货现金为零。不能用负现金掩盖不足。
- **退出意图属于快照状态**：已生效退出按币保留，不依赖现货尚有持仓，也不启用日频普通单全局重试。
- **策略参数与状态分工**：固定参数由 `config()` 覆盖；`state()/restore()` 只负责可变状态的恢复，
  不要求调用方先倒退状态。第三轮明确：`TWALLET1` 从未发布，不兼容任何缺少必需字段的
  开发期两钱包快照，也不迁移包含 `initial_state` 的旧指纹；统一明确拒绝。
  C4 前内核格式 2/3 快照兼容不受影响。
- **旧账本不可重建的限制仍在**：展示省略新增字段，不伪造零值；直接读取 W/U、可划转额、IM
  等字段及显式 `flow=` 仍报错。首次改造字段拦截时产生自递归，现有四个展示回归和两个
  旧快照访问回归先抓到该错误，修正为从类型读取拦截集合；失败记录同附红日志末尾。
- 不修改 DESIGN。补资浮点核验与退出意图持久化补足原有语义，没有增加资金容差。

### 回归、变异与复跑

- `make dev` 初次因网络限制无法查询构建依赖；使用本 worktree 缓存，`UV_OFFLINE=1` 重建成功。
  测试与每个变异均通过工作树导入检查、Rust 源指纹检查。
- `cargo test -j 2 --workspace`：**33 通过**；相关 Python 全量命令同上：**1771 通过**，
  0 跳过、0 失败，33.50 秒；12 个已有 groupbt 空切片告警。
- 钱包测试独立运行 **94 通过**；变异恢复后功能组 **90 通过**（不含四项摘要兼容单测）。
- `ruff check .`、`ruff format --check .`（786 文件）、`cargo fmt --all --check`、
  `cargo clippy --workspace --all-targets -j 2 -- -D warnings` 全通过。
- 重新运行旧接口比较：**1600 项输出逐位一致**，**400 个旧快照**恢复/重新编码一致（含
  100 个旧强平快照），400 个旧 NPZ/列集一致。

| 本轮新增变异 | 被预期断言抓到的测试数 |
|---|---:|
| `review2_exit_drops_flat_spot_blocked_future` | 4 |
| `review2_zero_buffer_skip_actual_credit_check` | 2 |
| `review2_fingerprint_includes_mutable_state` | 2 |
| `review2_legacy_display_includes_missing_fields` | 4 |

新增 **4/4** 捕获，追加到 `bench/results/mutations_crypto_wallets.json`，累计 **26/26**。
既有 22 处沿用第一轮实跑记录，本轮没有把构建、导入或语法失败计作捕获；每次恢复源码后再测通过。

真实数据只读副本上的例子、性能与原始输入手算均已重跑，更新两个结果 JSON。
两档可划转额因子 1/1.5 的最终 NAV 仍为 1.3340385137811117 / 1.334038513781113，
年化均为 4.3048%，划转 29/30 次，救援失败及强平均为零，`valid=True`。
最大取整残余均 108.9730502 USDT，成交不足残余均为零；上文例子结果表继续成立。
预补资 34.93222992、救援 92.12309262296917、资金费日收入 0.20488769058528228 USDT，
从原始输入独立手算与引擎结果一致。

BTC/ETH 的 59,903 小时中位耗时为 **2.808039 秒**，单账户原生快速路径 **0.006810 秒**，
单账户逐行循环 **0.203812 秒**；扣除两个循环后的协调额外开销 **40.072 微秒/行**。
相对一次原生 run 为 412.33 倍，profile 仍指向冻结状态、Python 小数组操作和双边逐行调用，
不是 Rust 成交内核变慢。共享机器上的计时只是本轮测量，不将与第一轮的差异归因于优化。

所有重命令均带 timeout、nice -n 15，pytest 最多 4 worker、cargo 最多 2 job。
沙箱拒绝 nice 的优先级调整（命令仍执行）；onecontext 历史库无法打开，本轮依据审查原文、
设计与代码验证。禁止 worktree 外写入的约束继续优先，工作日志留在本文件。

- [18:36] AShare C4 第二轮返工 — 四项反例先红后修，33 项 Rust、1771 项 Python、4 个新变异、旧快照兼容及真实例子/手算全部通过；结果与性能 JSON 已更新。

## 第三轮返工记录

先保留中断前的测试和实现，依据 `.review/REWORK3.md` 继续验收。修复前新增回归得到
**20 失败、1 通过**，原始失败输出保存在 `bench/results/crypto_wallets_review3_red.txt`。
已通过的一项是旧 `initial_state` 指纹拒绝测试；此次保留它并把拒绝原因写清楚。

| 意见 | 修复位置与行为 | 红测试（`test/test_crypto_wallets_review3.py`） |
|---|---|---|
| P1 扣费后的预补资舍入 | `crypto.rs::OpeningBudget` 的 `project/allows` 同时供真实成交与 `opening_feasible` 使用；PyO3/`step.py` 导出只读查询，`TwoWallet._prefund` 按该查询修正请求；参考实现独立检查扣费后的余额 | `test_review3_prefund_fee_subtraction_hand_expected`；`test_review3_fraction_prefund_random_prices_fees_multipliers_targets`，两者均含原生/参考实现 |
| P1 旧两钱包快照丢失退出意图 | `TwoWallet.restore` 要求完整且精确的字段集合，删除 `exit_pending` 默认空值路径；仍由快照保存逐币退出意图 | `test_review3_snapshot_requires_every_schema_field_even_at_day_zero`（15 字段）；`test_review3_cancelled_delist_snapshot_preserves_exit_or_rejects_missing_field` |

### 实现中的选择与设计问题

- **预补资估算不再充当成交判定**：Python 保留新旧 IM、费用、名义缓冲的估算作为起点，
  是否足额由 Rust 与真实成交共用的判定决定，包含入账、减仓、篮子和逐单开仓检查。
  原有篮子与逐单检查的浮点运算顺序不同，分别原样保留，未引入现金容差。
- **只读查询边界**：`Simulator.opening_feasible(target, prices, flow=...)` 使用调用方给定的
  已知参考价及本行生效费用/保证金率，不偷看当前开盘价；不裁剪 flow，也不保证流动性、
  数量过滤、最小名义、滑点、跳空或强平压力下成交。协调器在查询前执行现货可划转额上限。
- **浮点步进与硬上限**：请求不足时向上步进到能改变入账余额的下一浮点值，避免从零请求
  逐个遍历 subnormal。上限内仍无法通过时保留未满足请求、记录 capacity，绝不让现货负债。
  `test_review3_zero_request_steps_credited_cash_without_subnormal_loop` 固定检验零请求边界。
- **独立手算**：4 bp 反例的一张合约 IM=.5、费用=.0004，最终权益应为
  100.15/100.1496/100.1496，持仓应为 0/−1/−1；两个实现都用此期望验证。
  Fraction 性质测试 RNG=34017、每个实现 2048 例，覆盖随机整数/十进制价格、五档乘数、
  三档保证金率、目标张数及 0–10 bp 的全部费率；精确计算理论张数与补资，严格断言
  全额张数、划转不超上限、两腿流互相抵消。仅补资浮点表示与精确有理数比较有 ULP 界。
- **性质测试找到的新反例**：case 22：价格 20、乘数 1、目标 2 张、费用 9 bp、IMR=1/10，
  初始合约现金 2.8。精确需求 1009/250=4.036，扣除实际 binary64 初始现金的精确补资为
  173951535607185433/140737488355328000；旧原生与独立参考均只开 1 张，修后均开 2 张。
- **边界一致性额外验证**：`test_review3_rust_preview_matches_fill_on_adjacent_float_budgets`
  在 128 组双币场景、上下相邻浮点预算上共做 256 次比较，含已有仓、减仓、反手、按张费用；
  查询不改变快照，判定与实际是否完整成交严格一致，且通过/拒绝两分支都出现。
- **旧两钱包与旧内核快照分开处理**：未发布的 TWALLET1 旧 schema、缺字段、未知字段及
  旧 initial_state 指纹统一拒绝，不尝试重建退出任务。正常快照在公告后来撤销时仍完成
  原退出任务，最终权益 300，不多收 5 的资金费。C4 前内核格式 2/3 的恢复、展示及旧数值
  兼容路径完全保留。不修改 DESIGN。

### 回归、变异与复跑

- `UV_OFFLINE=1 make dev` 使用本 worktree 的 venv 和缓存重建成功；Rust 源指纹为
  `5bd1622c2ae3252068c9b35e50ae7d8f79a4ac7c68201dbf3e4457b20f026755`。
- `cargo test -j 2 --workspace`：**33 通过**；相关 Python 全量命令同前：**1795 通过**，
  0 跳过、0 失败，47.71 秒，12 个已有 groupbt 空切片告警。第三轮新增 **24 项**测试。
- `ruff check .`、`ruff format --check .`（787 文件）、`cargo fmt --all --check`、
  `cargo clippy --workspace --all-targets -j 2 -- -D warnings` 全通过。
- 再次对冻结旧扩展比较：**1600 项旧输出逐位相同**，**400 份旧内核快照**恢复及重新编码
  通过（含 100 份旧强平快照），400 个旧 NPZ/列集兼容。`repr/print/asdict` 兼容保持。

| 本轮新增变异 | 捕获的预期失败数 |
|---|---:|
| `review3_preview_omits_opening_fee`（Rust 预判漏开仓费） | 4 |
| `review3_missing_exit_schema_defaults_empty`（缺退出字段静默补空） | 2 |

新增 **2/2** 被抓，报告累计 **28/28**，其中前 26 处保留各自历史源指纹和实跑结果；
`tested_this_run` 明确列出本轮的两处，没有把历史结果当成本轮重跑。Rust 变异前后均重建，
每轮校验导入路径及源指纹；构建、导入、语法失败不计捕获。恢复源和扩展后 **114 项功能
测试通过**（8.11 秒），包含全部第三轮测试，不含四项兼容摘要单测。

真实数据副本重新跑例子、两档可划转额敏感性和性能，结果更新到对应 JSON。
两档最终 NAV 为 1.3340385137811117 / 1.334038513781113，年化均 4.3048%，划转 29/30 次，
救援失败及强平均零、`valid=True`；取整残余峰值均 108.9730502 USDT、成交不足残余均零。
预补资 34.93222992、救援 92.12309262296917、资金费日收入 0.20488769058528228 USDT，
原始输入独立手算与引擎结果全部一致。上文例子结果表仍成立。

59,903 小时 BTC/ETH 两钱包三次中位 **3.270490 秒**，单账户快速路径 **0.007974 秒**、
单账户逐行 **0.224172 秒**，扣除两个逐行循环后额外开销 **47.112 微秒/行**。
相对单账户快速路径 **410.15 倍**；profile 仍以 Python 决策、冻结状态、小数组及双边逐行
调用为主。共享机器上的时长变化不单独归因于这次修复。

所有长命令使用 timeout，重活 nice -n 15，pytest 最多 4 worker、cargo 最多 2 job；
沙箱仍拒绝 nice 调整优先级但允许执行命令。onecontext 数据库仍无法打开，依据审查原文
与实际工作区继续；遵守不写 worktree 外文件，阶段工作日志保留在这里，未提交、未推送。

- [19:11] AShare C4 第三轮返工 — 两项 P1 先红后修，Rust 共用成交判定消除扣费舍入少开张，严格拒绝不完整 TWALLET1；1795 项 Python、33 项 Rust、2 项新变异、旧接口逐位兼容和真实例子手算通过，结果 JSON 已更新。
