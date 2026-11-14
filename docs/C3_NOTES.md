# C3 实现记录

范围：币安市场档案、独立现货/合约账户调度、C2 限流与停机修复、离线验收及真实例子。
不修改 DESIGN，不提交、不推送。真实数据只读复制到忽略的 `tmp/crypto`。

## 实现中的选择

- 费用唯一依据为 BINANCE.md「回测口径」。规则证据覆盖不到的区间必须显式 `assume=`；历史清算费不能伪装成已核实。
- 日频普通目标不顺延；下架清仓单独逐小时生成零目标，只覆盖对应列。
- 半截线保留真实 OHLC；随后连续缺口由此前半截线自证停机，恢复行终止推断。异常倒置时间戳保留并单列，不充当停机证据。
- **证据区间**：风险参数分字段记录。只有 BINANCE.md 明示的维持率连续性跨区间使用；档位上限、最高杠杆的孤立观测仅覆盖相应小时，清算费快照仅覆盖观测日。未核实的中间变化不猜日期。`previous` 在过去空档向前延续、早于全部观测时取最早值，逐段标假设；完全未知品种必须提供数值、理由和出处。费用切换日是任务明确指定的默认模型假设，不另外要求确认。
- **数量与输入**：C2 的 `volume_engine/funding/settlement` 直接传给内核，不重乘乘数。日频参数必须有日期标签；Polars Date、numpy 时间/整数毫秒都显式支持。季度普通张数目标路由也按新旧乘数换算；换月专用行保持报价单位数量，只在调仓整点尝试。
- **缺口**：自证停机和公告暂停直接持平；其他缺口默认拒绝，显式允许才持平。倒置时间戳保留实际价格供质检，但回测须显式允许并按前值估值；首行无过去估值仍拒绝。资金费缺值不能被 `allow_missing` 放行。
- **现货转换**：使用已知且生效的事件，固定旧计量单位接续新行情，不虚构卖出/买入/现金。转换链和目标币不能重复占两个列；旧数量格点不能落在新币规则格点时拒绝。日线仍由数据层逐实际交易对给出，跨迁移信号需调用方明确接续。
- **现货目标**：第一批只开放权重目标；现有内核的现货 `shares` 会绕过步长，因此不在此接口默默开放。合约 `shares` 仍是严格检查过滤器的整数张。
- **CMS 续跑**：按查询范围、币种和目录分进度，成功处理整页才推进页码；完成整次扫描后重抓首页。正文保存与事件导入分完成状态，原文的首次观察时间保留；中断不能制造重复修订，正文退回旧版本仍是新的观察修订。
- **真实例子**：参数预先固定为 20/60/120 日投票、30 日波动率、20% 年化波动预算均分给 BTC/ETH；初始 1,000 USDT，合约目标总名义 2 倍、现货 1 倍。没有通过搜索参数挑收益。Quant 对照改用同一 BTC 单币策略、01:00 调仓与 1 倍目标上限，末次共同清仓对齐比较边界；只读外部源码、禁写 pyc。

## 设计问题

- 全局工作日志目录在授权 worktree 之外，本任务禁止越界写入；完成记录保留在本文件，未修改全局 FORMAT.md。
- **原始数据并非全部完整**：除任务书列出的时间戳形态，现货每列还有 10 个非自证缺口（共 20 小时），只能登记为“已观察到的归档缺口、研究显式持平”，不能声称是已确认停机。这些例外进入副本的冻结输入及公开的元数据汇总。半截线之后的 5 段/11 小时则自动识别，不需要例外表。每列另外 1 个倒置时间戳，回测的缺失诊断含它，合计 21 个异常/未解释小时。
- **月归档缺天**：U 本位成交 K 线无时间戳异常和缺口，标记价/指数价月归档却有整天遗漏。共补到 29 份官方日归档，逐份核验 CHECKSUM，只填原来不存在的整日，不覆盖月归档已存在的行；补后 6 组 U 本位序列均无缺口。这是完整月文件存在并不保证内部完整的边界，未更改 C2 通用同步的分区优先级。
- **近期资金费不能凭间距填 interval**：原始副本的 11 月 REST 资金费没有归档历史间隔证据。严格构建终点取 2026-10-31 23:00 UTC，排除右边界为 11 月 1 日零时、缺 interval 的最后一行。原始 11 月记录保留，不能声称完成到 11 月 14 日的严格回测。`bench/crypto_prepare.py` 由实际归档最后间隔计算这个边界，不硬编码币种或日期。
- **历史风险规则不完整**：2021-05-19 手算中的 0.4% 维持率、1.25% 清算费是明确外推，不能声称完全复原当天账户规则。C1 对任意盘中强平行保守记档位覆盖不确定；因此手算例显式允许 1 行 `tier_cap_exceeded`，策略两个账户均为 0 行。
- **10 倍口径要含开仓价差**：成交价高于标记开盘时，初始现金仅为成交名义 / 10 加手续费仍不足以按标记权益开足仓。手算资金改为 `Q × (mark_open/10 + entry×fee + entry−mark_open)`，开仓后恰为标记名义的 10 倍；普通开仓费和开仓浮亏均已覆盖。
- `nice -n 15` 在本沙箱返回 `setpriority: Operation not permitted`；所有重活仍通过该前缀启动，但不能声称成功降低优先级。编译设置 `CARGO_BUILD_JOBS=2`，pytest 最多 4 worker；没有写主目录 venv、原始库或 Quant。

## 回归与验收

每项错误先补离线回归并保留 `bench/results/crypto_c3_*_red.txt`（已去除本机路径）。没有放宽既有数值断言、没有新增跳过。注册表原来的“四种档案精确集合”因本任务新增两个档案而失败，改为“六种档案精确集合”，没有改成子集断言。

| 来源 / 错误 | 回归与红灯 |
|---|---|
| 任务书：连续 CMS 429、续跑重抓首页 | `test_cms_long_backoff_recovers_and_is_configurable`、`test_cms_interrupted_pages_resume_without_fetching_page_one`；initial red |
| 任务书：半截线、±1 秒、倒置时间戳、停机缺口 | spot 毫秒/微秒与 usdm 合成形态，独立 verify 标记变异、截断前缀；initial red |
| 任务书：新档案/API 不存在 | 4 个档案用例、8 个适配接口用例；initial / adapter red |
| 自检：盘中强平计数名误写 | `test_liquidation_diagnostics_intraday_not_lost`；extra red |
| 自检：迁移不能兑付现金或忽略比例 | `test_migration_carries_residual_without_cash_sale`；extra red |
| 自检：正文落盘后事件未完成，续跑漏导 | `test_announcement_resume_after_body_saved_before_events`；resume red |
| 自检：恢复逻辑漏掉正文回退、重复部分已写修订 | `test_cms_body_reversion_is_a_new_observed_revision`、`test_cms_partial_revision_resume_keeps_first_observation_time`；reversion / partial_revision red |
| 真实例子：numpy 毫秒被当 datetime；接口扩展类型 | `test_numpy_daily_dates_from_dataset_are_accepted`、numpy datetime64 / Polars Date；numpy_dates / date_types red |
| 真实手算：只成交 995 张却按 1,000 张手算 | `test_hand_example_funds_execution_to_mark_loss`，合成成交 100/标记 90；hand red |
| 自检：季度普通张数目标路由忘记乘数换算 | `test_quarter_roll_preserves_units_with_different_multipliers`；roll red |
| 自检：现货 shares 绕过步长、转换格点不兼容 | 两个显式拒绝回归；spot_lots red |

额外合同用例覆盖：日线唯一错位、非 trade_hour 不重试、换月日历落点与实际成交、top_n 真时间窗口/截断/同币合并/USDT 保证金/历史下架、未来事件版本不改变旧成交、只改下架列、未变现不计现金、缺数据报交易对与时段、第一档超限默认异常、风险假设边界、日期分段费率与敏感性、C2 两合约生命周期、资金费和交割字段直连、最小下单过滤、信号不读当前小时。原有 C1 的 200 项内核强平/资金费/数量等测试原样合跑。

## 首次交付检查（返工前）

- 本 worktree 独立 `.venv` 经 `make dev` 构建；相关 `make check` Python 检查完成：Ruff lint 通过、771 个文件格式通过、`git diff --check` 通过。
- crypto 数据层、C1 内核、C3 适配、现有市场注册表、全部 backtest 和 step 回归：**472 passed（10.03 秒，4 workers，全部离线）**。未改 Rust，未重复无关的 Rust 检查。
- 12 处真实源码变异 **12/12 被杀死**，没有把语法/导入失败算作成功；变异前已确认 Python 与原生扩展都来自本 worktree，恢复后基线再次通过。覆盖唯一移位、日频不顺延、公告可知时间、下架覆盖买入、选币窗口、费率切换、规则假设、换月落点、分档有效性、迁移比例、停机证据和公告页码恢复。
- 交付文件审计无本机绝对路径、无真实行情文件；真实副本和完整核验明细仅在忽略目录。最终验收汇总见 `bench/results/crypto_c3_checks.json`。

## 文件清单

- 档案：`packages/panelbt/python/panelbt/markets/binance.py`；注册于同目录 `__init__.py`。
- 回测：`src/ashare/backtest/crypto.py`、`crypto_schedule.py`、`crypto_run.py`、`crypto_migration.py`。
- C2 修复：`src/ashare/data/crypto/sources.py`、`sync.py`、`announcements.py`、`formats.py`、`panel.py`、`verify.py`、`verify_raw.py`、`verify_contract.py`、`timeutil.py`、`dataset.py`。
- 测试：`test/crypto/test_backtest.py`、`test_binance_profile.py`、`test_c3_data_fixes.py`、`test_c3_review.py`；更新 `test/test_markets_hkus.py` 的精确注册表集合。
- 实验：`bench/crypto_prepare.py`、`crypto_examples.py`、`mutations_crypto_backtest.py`；结果为 `bench/results/crypto_data_validation.json`、`crypto_examples.json`、`mutations_crypto_backtest.json`、`crypto_c3_checks.json` 和上述红灯记录。
- 文档：本文件、`docs/CRYPTO_BACKTEST.md`、`docs/CRYPTO_DATA.md`。没有修改 DESIGN.md、BINANCE.md、C1 内核、Quant 或现有共享合成夹具。

## 真实结果

数据副本覆盖 2020-01-01 00:00 至 2026-10-31 23:00 UTC（右端不含）：59,903 小时 × 4 列 = 239,612 格，955 份归档，verify 0 错误。数据层 6,642 条诊断保留在忽略目录；公开汇总无逐行行情。现货 BTC/ETH 各 6 根半截线、1 根倒置时刻、1 根容忍范围内的边界偏差；U 本位三类 K 线均未发现同类时间戳形态。

| 策略 | 最终净值 | 年化 | 最大回撤 | 净资金费 / 初始资金 | 费用 / 初始资金 | 强平 |
|---|---:|---:|---:|---:|---:|---:|
| BTC/ETH 永续趋势 | 2.251745 | 12.6033% | 18.6597% | 22.9415% | 9.7068% | 0 |
| BTC/ETH 现货只做多 | 2.633912 | 15.2145% | 13.5104% | 0% | 13.7742% | 0 |

合约 8,394 个输入格使用早期资金费价格近似。策略两行 `valid=True`，无第一档超限；现货的 20 段缺失/异常区间和 10 段自证停机均明示。规则假设区间数分别为 22、4；这些数字是输入使用范围，不是付费笔数。

| 手算 | 独立计算 | 引擎 | 结论 |
|---|---:|---:|---|
| 2021-05-19 BTC 强平价，04:00 小时 | 38,731.15662650602 | 38,731.156626506025 | 一致；真实日最低标记价 29,563.37，触线 |
| 普通强平手续费 | 15.492462650602407 | 15.492462650602413 | 一致 |
| 清算费（剩余权益封顶） | 139.43216385541587 | 139.43216385542314 | 一致；现金 0、穿仓缺口 0 |
| 2024-06-01 三笔净资金费 | 0.20488769058528228 | 0.20488769058528228 | 一致；最终现金 1001.671222809415 |
| 合成季度到期最终现金 | 99.7601 | 99.76010000000001 | 08:00 交割落 07:00；交割费 0.0199、资金费 0 |

BTC 手算初始现金 4,323.461016 USDT，实际开仓 1,000 张 × 0.001 = 1 BTC；开仓手续费 17.153016。盘中强平先于当天首次后续资金费结算，实际资金费为 0；与 C1 历史手算结果一致。真实复制样本没有季度，因此季度手算明确使用合成夹具。

## Quant 对照

同一 BTC 策略、同一输入数据，2,495 个日决策，最后共同清仓。C3 最终净值 **1.9458397245**，Quant **1.9865057860**，差 **−0.0406660616**；资金费占初始资金分别 **21.1201% / 22.8963%**，交易费 **10.3928% / 11.8474%**。两边没有强平，Quant 没有资本约束违规。因此强平/穿仓规则是模型差异，本例没有触发，不能把净值差归因给未发生的事件。

| 项目 | C3 | Quant account.py |
|---|---|---|
| 资金费价格 | 官方关联标记价，早期右边界 mark open | 成交 K 线 open，缺小时用决策执行价兜底 |
| 强平 | 标记 OHLC 触线插值、正常平仓费 + 独立清算费 | 成交 OHLC 不利极值、用维持率当清算费 |
| 穿仓 | 缺口单列、余额清零并锁存爆仓 | 合约负现金截断，没有独立缺口账 |
| 钱包 | 独立单合约账户全部权益作抵押 | 两钱包，决策后按名义 × margin 重设合约钱包 |
| 交易费 | 4bp → 假设切换后 5bp | 对照配置固定 5bp |
| 下单 | 费前权益、整数张、历史过滤器 | 连续数量、迭代费后权益、没有最小单规则 |
| 估值 | 逐小时标记结算、mark open 定权重张数 | 日决策之间按成交执行价结转 |

没有改 Quant、没有调参数追求对齐。完整参数和结果在 `bench/results/crypto_examples.json`。

## 本地工作日志

- [06:22] AShare — 完成 C3 数据修复、档案与账户适配阶段；331 项 crypto 测试通过，真实副本 239,612 格/955 份归档 verify 零错误，12 项源码变异全杀死。全局日志受本任务目录限制，记录暂存本文件。

- [06:38] AShare — C3 收尾验收完成：472 项相关测试、Ruff 与 12/12 变异检查全部通过；交付两账户趋势结果、真实强平/资金费手算、合成交割及只读 Quant 对照，数据终点与假设详列本文件。

## Claude 第一轮返工

九条意见全部接受。新增 `test/crypto/test_c3_review.py`，首批 14 个反例均先在返工前实现失败（`bench/results/crypto_c3_r1_red.txt`）；继续排查发现正常标记线反证后，后续缺口又恢复错误停机推断，另补 1 条红灯（`crypto_c3_r1_extra_red.txt`）。没有放宽原有断言。

| 审查项 | 修复位置与行为 | 新回归（同文件） |
|---|---|---|
| P0-1 | `crypto_run.simulate` 由整个逐时实际交易对集合决定事件路径，按当前执行符号和 asof 退出；`crypto_market` 缺公告检查也看实际符号及当时版本 | `test_r1_p0_1_migrated_delist_and_future_announcement_invariance`；`test_r1_p0_1_missing_notice_checks_actual_execution_symbol` |
| P0-2 | `panel.build_instrument` 仅接受成交半截线证据；正常标记线清除整段推断资格。`verify.verify_frame` 独立搜索原始成交前驱与正常标记时间轴，检查反证 | `test_r1_p0_2_normal_market_data_refutes_inferred_halt`（2 形态）；`test_r1_p0_2_independent_verify_rejects_false_halt`；`test_r1_p0_2_counterevidence_breaks_entire_gap_chain` |
| P0-3 | `crypto_market` 验证到期结算价、来源、事件引用；缺行情持平只改行情，不覆盖有效到期结算 | `test_r1_p0_3_missing_trade_bar_preserves_delivery_price`；`test_r1_p0_3_unproven_expiry_price_is_rejected`（缺价格/缺来源） |
| P0-4 | `announcements._drain_article` 按 SHA/首次观察时间持久化待导入版本，先恢复旧正文再请求新版本，逐事件幂等；已观察修订不回退首发时间 | `test_r1_p0_4_resume_consumes_old_body_before_new_revision`，包括网络请求前检查旧版本已入账 |
| P1-5 | `crypto_market` 交割费按 `times + HOUR`，普通费仍按 `times`，诊断分别标左右边界 | `test_r1_p1_5_delivery_fee_uses_right_boundary` |
| P1-6 | `dataset.daily/signal_frame` 默认拒绝异常价；显式 carry 与小时持平一致；`CryptoMarket.daily` 和例子使用该视图，原始异常留在 read | `test_r1_p1_6_daily_signal_excludes_invalid_close` |
| P1-7 | `crypto_examples.summary` 总费用不重复加 clearance_fee，另列清算费分项 | `test_r1_p1_7_summary_counts_clearance_once` |
| P1-8 | `crypto_examples.funding_check` 独立读取冻结原始费率、关联价、乘数，核对三次结算时刻、逐小时现金流及实际持仓 | `test_r1_p1_8_raw_funding_oracle_catches_adapter_scale`，适配后资金费乘十必失败 |
| P2-9 | `binance._observations/rules` 保存真实生效时刻、整点应用时刻和近似标志；`crypto_market` 合并风险近似使用区间 | `test_r1_p2_9_risk_rounding_metadata_reaches_diagnostics` |

新增选择：观察到正文变化之后，修订统一采用首次观察该版本的时刻（比依赖可能倒退的 CMS 更新时间更保守）。风险近似区间表示本次回测使用该条取整规则的范围，另附真实生效与应用时刻；不暗示整个范围都产生额外延迟。非停机证据支持的缺口继续默认拒绝，显式 allow_missing 的持平仍是研究假设。

返工后的真实重跑：`crypto_prepare.py --data tmp/crypto --accept-spot-gaps` 完成 build/verify，239,612 格、955 份归档、0 个错误、6,642 条诊断。新数据版本为 `592e490b1d0b50ca14c51035be2d373370bc53a3df8a3f77ce83dd1a6735c194`；变更来自停机证据说明改为明确的成交半截线来源，样本区间不变。没有重新下载或修改主目录原始库。

`crypto_examples.py` 含只读 Quant 对照完整重跑：两个趋势策略、强平手算、资金费现金、季度交割与 Quant 既有数字均逐值不变。策略没有强平或季度交割，清算费重复计数与交割费切换修复不改变这些趋势结果；现货倒置时间戳不处在所用日收盘，修正后的日线 high/low 不用于本例信号。新增清算费分项（两策略均为零）及独立资金费核对元数据；原始三笔记录独立计算 10 张 × 0.001 乘数的净资金费仍为 0.20488769058528228 USDT，08/16/次日00 点逐小时现金流和持仓均通过。

本轮最小反例的关键数值：转换后新币在第 3 小时按公告卖出，追加未来原币公告不改变历史；缺成交线的季度到期价保留 199，最终现金 99.7601；费率切换边界开仓费 0.04、交割费 0.05；强平汇总费用从错误的 32.63930722891555 修正为 18.574297188754965；异常日收盘从 10000 回到已知前值 101，原值仍留在原始视图。

- [07:11] AShare — C3 第一轮九条审查问题修复并补齐 15 个离线回归；487 项相关测试通过，真实副本 build/verify 零错误，例子与 Quant 重跑完成，原有汇总数字不变。

最终验收：本轮相关测试 **487 passed（12.47 秒，4 workers，全部离线）**；Ruff lint、772 个文件格式检查及 git diff --check 通过。原 12 项变异和新增对应九条修复的 9 项变异共 **21/21 被杀死**；没有将语法/导入失败算作成功，变异前核验所有相关模块和原生扩展均来自本 worktree，恢复后再次跑基线并逐文件核验 SHA。结构化逐项记录为 `bench/results/crypto_c3_review_r1.json`，通用验收汇总 `crypto_c3_checks.json` 已更新。全部真实例子与报告只有汇总数字，无逐行行情、无本机绝对路径。

- [07:13] AShare — C3 第一轮返工收尾：21/21 源码变异被抓到，恢复源码后 487 项测试再次通过；逐项验收及真实重跑结果已写入 bench/results 和 C3_NOTES.md，未提交、未推送。

## Claude 第二轮返工

完整复核意见中的两项缺陷和快慢路径覆盖建议均接受。新增 `test/crypto/test_c3_review_r2.py` 共 12 个用例：修复前 **6 failed、6 passed**，失败日志为 `bench/results/crypto_c3_r2_red.txt`。6 个失败分别覆盖空仓挂起买单、另一列仍有买入/减仓/卖出目标的三种组合，以及构建器、独立校验器的停机正例边界。已有卖出行为和四组快慢路径对照是新增保护用例，修复前即通过，没有伪称失败。

| 审查项 | 修复位置与行为 | 回归测试 |
|---|---|---|
| P1 挂起买单 | `crypto_run.simulate` 的下架覆盖条件加入 `state.pending_cols`；空仓也覆盖该列零目标，保留其他列的挂起目标；已有减仓/卖出继续逐小时退出 | `test_r2_p1_delist_cancels_flat_pending_buy`；`test_r2_p1_cancel_only_delisted_pending_buy`（买入/减仓/卖出）；`test_r2_p1_delisted_pending_reduction_still_exits`（减仓/清仓） |
| P2 反证边界 | `panel.build_instrument` 仅在成交缺口中让正常标记线清除证据；`verify.verify_frame` 独立要求正常标记时间严格晚于前驱半截成交线小时 | `test_r2_p2_normal_mark_before_gap_does_not_refute_build`；`test_r2_p2_independent_verify_accepts_mark_before_gap` |
| P0-1 补覆盖 | 不改执行语义；测试监测真实 `bt.run` / `bt.Simulator` 入口，覆盖顺延开/关、迁移后旧币公告不退出新币，以及新币下架退出与显式退出计划 | `test_r2_p0_1_migration_fast_slow_bitwise`（4 组）；所有逐行数组比较原始字节，成交、事件和内核诊断完全相同 |

本轮实现选择：下架列的挂起买入被零目标替换；已有挂起减仓按照下架要求继续卖至零。其他列的顺延目标原值保留。停机证据只覆盖半截成交小时之后至恢复交易之前的连续缺口；同小时正常标记不构成反证，缺口中途的正常标记仍使此后缺口失去推断资格。没有新增设计问题，也没有改 `DESIGN.md`。

- [07:28] AShare — C3 第二轮修复挂起买单与停机反证边界；12 个新增用例完成先红后绿及路径覆盖，499 项相关测试、Ruff 通过；正在重跑真实副本与变异验收。

真实副本再次执行 `crypto_prepare.py --data tmp/crypto --accept-spot-gaps`：**239,612 格、955 份归档、6,642 条诊断、0 个错误**，面板版本仍为 `592e490b1d0b50ca14c51035be2d373370bc53a3df8a3f77ce83dd1a6735c194`，未新增下载。随后完整重跑例子和只读 Quant 对照，`crypto_examples.json` 与第一轮报告逐值相同，包括强平、原始资金费独立核对和合成交割。现货期末净值 **2.633912380760807**、年化 **15.214524%**、最大回撤 **13.510447%**；永续期末净值 **2.251744618123367**、年化 **12.603304%**、最大回撤 **18.659712%**。本样本 BTC/ETH 日频策略没有现货下架挂单，U 本位也没有此次边界形态，因此本轮修复不改变实盘归档例子的数字。

最小复现则按预期从现金 7,492.5、净值 0.74925 修正为全程空仓、现金 10,000、净值 1。半截线同小时正常标记的正例通过严格构建和独立校验；第一轮的“缺口内有正常标记”反例仍被拒绝。

第二轮最终验收：**499 passed（11.79 秒，4 workers，全部离线）**；Ruff lint、773 个文件格式检查和 `git diff --check` 通过。变异扩展至 **24/24 被抓到**：新加的“不取消挂起买单”被四个回归抓到，构建器和独立校验器各自的同小时反证变异分别被对应正例抓到；原 21 项仍全部被抓到。变异前检查本 worktree 导入，变异后逐文件 SHA 恢复核验及基线复测通过。结构化记录为 `bench/results/crypto_c3_review_r2.json`，总验收汇总及变异结果同步更新。

- [07:31] AShare — C3 第二轮返工完成：499 项相关测试、24/24 源码变异、Ruff 全通过；真实副本 239,612 格/955 归档零错误，例子与 Quant 数字逐值不变；修复和红灯记录已落盘，未提交、未推送。
