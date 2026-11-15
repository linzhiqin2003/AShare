# C5：加密货币推断延伸

## 实现中的选择

- 严格版默认关闭推断，保留原有字段和数值；按 `last_archive_time + interval * 1h - 1h` 约束仍需覆盖的永续，再与请求终点取小。已经覆盖到已知退出边界的列不截短其他列。终点是 `[start,end)` 的右端，不是最后一行左端。
- 推断版独立写 `latest_inferred.json`，默认读取仍走 `latest.json`；明确的版本 SHA 可读取任一历史版本。推断输入与元数据冻结选项、请求终点、fundingInfo 抓取时间、原始字节哈希和各列截断原因。
- 先求生命周期内仍需覆盖的归档序列的共同完整小时右端，再检查 REST，最晚检查到该列已知退出边界。窗口外的坏记录不能影响窗口内结果；第一处异常或末笔缺失时保守退到上一笔合格结算的右端，不使用下一笔未知结算前的空白小时。快照不支持当前间隔时保留严格上限。
- 只推断 interval；费率和结算标记价来自实际 REST。新质量位 `4096`（`funding_interval_inferred`）仅标发生推断结算的行；规范化资金费记录 `source=inferred_continuation`。严格面板不增加列，旧质量位含义不变。
- fundingInfo 按 HTTP 响应原始字节存档，`updateTime` 不作判断。默认 8h 只适用于未列出的最后归档间隔为 8h 的交易对；其他间隔必须有一致的列表项。
- 月归档优先，只为月归档完全没有记录的日子尝试日归档，复用 CHECKSUM 校验；月内部分小时缺口继续走原有缺失处理，不用重叠日文件覆盖。取消 K 线 REST 补尾。
- 对账只比较结算时刻、interval、rate、price 的经济值及结算集合，忽略来源标签；报告存在数据根的 `funding_reconciliations/`，verify 列出相关报告但不把已报告修订当作构建错误。旧版本只读保留。
- 遵照任务书第 2 项的特定许可，`src/ashare/backtest/crypto.py` 仅新增一行 `inferred_funding_intervals` 诊断，沿用其现有窗口和区间工具；这是与 C4 唯一需要协调的文件。不改其他 C4 文件或内核。

## 设计问题

- DESIGN 同时描述“第一笔异常之前”和“前一笔结算右边界”，两者之间可能相差数小时。按较保守的后一种解释实现；严格版沿用 C3 的既有公式，不把严格历史缩短。
- fundingInfo 的当前状态及相邻 REST 间距无法排除短暂改频后恢复、恰好漏掉中间新增结算。实现仅满足显式推断假设，不声称恢复历史完整性；后续归档对账保留纠错轨迹。官方接口文档没有提供 updateTime 的历史证明承诺。
- C3 已观察到月 ZIP 内遗漏整日，原同步只在整个月不存在时找日文件。C5 把完整缺日修复纳入通用同步；完整日缺口才能无重叠地补档，部分日缺口仍保留原诊断与例外机制。
- 机器限制：`nice -n 15` 返回 `setpriority: Operation not permitted`，各重活仍使用该前缀；构建使用 `CARGO_BUILD_JOBS=2`、pytest 最多 4 worker。OneContext 搜索失败（历史数据库无法打开），以仓库设计与审查 6/12 为依据。任务明确禁止写 worktree 外，工作记录保存在本文件，不改全局日志或技能文件。

## 回归与验收

新增用例全部使用独立合成常数，不读取真实副本、不联网。红灯记录见 `bench/results/crypto_c5_red.txt`。

- 旧 HEAD 隔离源码上先跑 C5 初始用例：18 failed；日归档补缺/原样快照回归：2 failed、既有 CHECKSUM 防线 1 passed。
- 开发自检发现覆盖外坏 REST 导致多截断，先复现 1 failed，再改为先求共同归档上限；测试来源写在函数文档中。
- 旧离线 HTTP 夹具首次重同步失败，原因是模拟服务忽略 `startTime`、仍返回全部历史；将夹具改为遵守请求范围，保留原断点续传断言，并由独立增量用例断言只请求末记录之后。

- 开发自检发现“归档更正但缓存 REST 费率仍旧”会使严格构建失败，先复现 1 failed，再增加归档费率优先的显式来源冲突诊断；进一步用 1e-15 的修订证明冲突识别不能套用现金计算的容差，先红再改为经济字段精确比较。
- 独立核验补上原始快照内容/时间/哈希对账，三个篡改用例在修复前失败。原始行情与冻结内容校验没有被放宽。
- `make dev` 使用 worktree 独立 venv 完成。相关 `make check`：Ruff lint 通过、779 个文件格式检查通过、`git diff --check` 通过；未改 Rust，不重跑无关的编译检查。
- 全部 crypto（数据/C1/C3）及市场注册、backtest、step 回归：**526 passed**，4 workers，12.08 秒，零跳过。命令：`nice -n 15 .venv/bin/pytest -q test/crypto test/test_crypto.py test/test_crypto_review.py test/test_crypto_review_intent.py test/test_markets_hkus.py test/backtest test/test_step.py -n 4`。
- 真实源码变异 **12/12 被杀死**，没有语法/导入/收集失败冒充成功，恢复后基线通过；变异前确认 Python 模块和 panelbt 原生扩展均来自本 worktree。覆盖间距、默认 8h 适用范围、缺价、推断末笔截断、原有 schedule 末笔检查、经济差异、构建和读取默认关闭、独立终点公式、归档费率纠错、日归档无重叠与 CHECKSUM。见 `bench/results/mutations_crypto_recent.json`。

## 真实副本验收

原始库以 `rsync -a` 复制至忽略目录 `tmp/crypto`，只在副本上增量 sync，跳过已抓公告、未使用 refresh；本次新增下载 19,555,112 字节。旧计划、配置与最新版本引用在副本的 `before_c5.*.json` 留档。两版使用同一新同步计划，不为严格版改写请求终点。

| 项目 | 严格版 | 推断版 |
|---|---|---|
| UTC 起点 | 2020-01-01 00:00 | 2020-01-01 00:00 |
| UTC 右端（不含） | 2026-10-31 23:00 | 2026-11-15 00:00 |
| 小时 × 列 | 59,903 × 4 | 60,240 × 4 |
| 行数 | 239,612 | 240,960 |
| verify | 0 错误 / 963 份归档 | 0 错误 / 963 份归档 |
| 版本前缀 | a36fc5385ccf | e191300cdfe9 |

严格版与复制来的 C3 原版 **全部列逐位一致**（浮点按 uint64 位模式核对，同时完整 IPC 字节哈希一致）。推断版在严格区间内也逐位相同，新增 337 小时；BTC、ETH 永续各 43 笔推断结算，结算时刻从 11-01 00:00 至 11-15 00:00。最后可用日归档为 11-14，未补墙钟最近几小时。完整逐格校验报告仅保存在副本，公开产物只有汇总。

同一策略参数、初始资金 1,000 USDT 的例子重跑：

| 策略 / 指标 | 严格版 | 推断版 |
|---|---:|---:|
| 永续趋势最终净值 | 2.251744618 | 2.283899854 |
| 永续年化 | 12.603304% | 12.760818% |
| 永续最大回撤 | 18.659712% | 18.659712% |
| 永续净资金费支付（USDT） | 229.414952 | 228.920694 |
| 永续费用（USDT） | 97.068294 | 97.927782 |
| 现货只做多最终净值 | 2.633912381 | 2.614532203 |
| 现货年化 | 15.214524% | 14.999708% |
| 现货最大回撤 | 13.510447% | 13.510447% |
| 现货费用（USDT） | 137.741942 | 138.907620 |

两版均 valid、无强平、无第一档超限；现货资金费为零，强平、资金费、合成交割手算例均通过。历史输入相同，差异来自新增区间的价格、日线信号/交易与 REST 资金费结算；年化分母随样本长度变化。永续净支付减少约 0.494258 USDT、费用增加约 0.859488 USDT；现货费用增加约 1.165678 USDT。没有做 C4 两钱包例子，也未改策略参数。

## 文件清单

- 数据实现：`src/ashare/data/crypto/{archive,sync,funding,build,panel,dataset,__main__,verify,verify_raw}.py`；新增 `recent.py`（构建条件）、`verify_recent.py`（独立条件）、`reconcile.py`（版本对账）。`sources.py` 的现有 CHECKSUM/资金费限速复用，无修改；`verify_contract.py` 沿用既有合约校验。
- C3 最小对接：`src/ashare/backtest/crypto.py` 一行读取诊断。
- 运行与变异：`bench/crypto_prepare.py`、`bench/crypto_examples.py`、新增 `bench/mutations_crypto_recent.py`。
- 离线测试：新增 `test/crypto/test_recent.py`、`test_recent_sync.py`、`test_c5_review_r1.py`、`test_c5_review_r2.py` 及独立旧版摘要 `test/fixtures/crypto/c5_strict_hashes.json`；更新 `test_pipeline.py` 的公开快照与分页 HTTP 夹具。
- 文档：`docs/CRYPTO_DATA.md`、本文件；未修改 DESIGN。
- 公开验收汇总：`bench/results/crypto_c5_{strict,inferred,strict_bitwise,comparison,examples_strict,examples_inferred,checks}.json`、`crypto_c5_red.txt`、`crypto_c5_review_r1_red.txt`、`crypto_c5_review_r2_red.txt`、`mutations_crypto_recent.json`。不含逐行行情、接口原文、密钥或本机绝对路径。

## 第一轮审查返工

完整复核 `.review/review_c5.md`，四条均成立。以下回归先在返工前的 C5 源码运行，**11 failed**，再修改生产代码；失败摘要见 `bench/results/crypto_c5_review_r1_red.txt`。前面的 526 项 / 12 项变异记录为首次交付结果，本节及最新汇总替代其验收结论。

| 审查项 | 修复位置与结果 | 新回归（`test/crypto/test_c5_review_r1.py`） |
|---|---|---|
| P1-1 退出列截短面板 | `recent.py` 按当时可知的事件界定活跃覆盖区间；`build.py` 传入起点和事件；`verify_recent.py` 独立重放事件版本并核对实际共同终点，过早截断也报错 | `test_ended_column_does_not_cap_panel`（严格/推断 × 结算退出/下架，4 项）；`test_verifier_rejects_ended_column_truncation`（2 项） |
| P1-2 新归档仍沿用旧 REST 价格 | `sync.py` 比较前后冻结计划的月归档 key/哈希，重新请求新增归档月份；整个请求区间替换，冻结新快照并保留旧对象；`reconcile.py` 读取新冻结输入 | `test_sync_refetches_prices_when_archive_published`：合成 HTTP 同步链路，100→120；断言新资金费 0.00012、价格差异报告、旧版输入/面板不变，以及下一次同步不重复抓这个月 |
| P2-3 月尾缺笔逃过审计 | `archive.py` 从文件月份得出完整声明范围，`sync.py` 冻结覆盖引用；`reconcile.py` 按声明月份审计，不按末条观测或新严格版终点收缩 | `test_reconciliation_audits_declared_month_tail`：七月末 16:00 缺笔，严格版 15:00 截断成功，仍报告该笔 `archive=null`；八月零点不误报，重复构建成功 |
| P2-4 REST 输入未与原文核对 | `sync.py` / `build.py` 冻结 `funding_price_refs`；`verify_raw.py` 独立核对引用、原毫秒时刻、费率、标记价和记录集合 | `test_verify_rest_input_against_frozen_response`（时刻、费率、价格，3 项）：同时改变输入与面板，逐行核算可通过，原文核对必须拒绝 |

严格不变测试不再调用当前 `build_frame` 得期望值。`test/fixtures/crypto/c5_strict_hashes.json` 是合成场景在 HEAD `04c37867d641fd51bc1af09cce54fdd8435f4958` 隔离源码上产出的逐字段 SHA-256：浮点按原始位模式，其余列包含 dtype 和 null。覆盖单列 23h、双列中一列 20h 退出但共同终点 48h 的结算/下架场景。测试本身不调用 git，也不依赖真实数据；`test_recent.py::test_strict_unchanged_default_and_separate_latest` 同样使用此独立摘要。

返工自检再发现一个 P1-2 边界：若后续同步把请求起点移到月中，按新请求窗口裁剪 REST 重取会留下月初旧价格，但对账仍声明整月覆盖。新增 `test_sync_archive_refetch_uses_full_declared_month` 先得到 **1 failed**，再将重取范围改为完整声明月份（不超过抓取时刻），并增加对应源码变异。此次共新增 12 个回归实例，全部先红再绿。

### 返工中的选择与设计问题

- 覆盖需求按退出执行前一小时已经可知的有效事件版本确定；请求起点前已退出的列按起点快照判定。事前撤销/延期不提前豁免；不以当前生命周期标签倒推历史，也不改原事件执行器及 `funding_schedule`。原设计的全局取小公式没有显式排除退出列，本次作生命周期范围解释，未改 DESIGN。
- 归档发布后自动重取其声明的完整月份（最晚到抓取时刻），范围可以略大于旧推断时段，确保归档内所有采用的 REST 结算价来自新快照。已发布且哈希不变的旧月份继续增量，不循环重抓；服务器尚未发布的价格修订仍无法凭 ZIP 发现。末笔覆盖与缺价规则不放宽。
- 对账使用 `[月初,下月初)`，而面板结算发生在行的右边界，必须区分这两个边界约定。月底缺失报告为 `archive=null`；来源标签变化仍不算差异，报告不阻塞严格构建。
- 新输入标 `recent.source_contract=1`，增加 `funding_price_refs` 和 `funding_archive_coverage`，并保留 raw_version。旧版本没有引用，仍可按原有保证读取复现；要获得新的 REST 输入原文保证须重建。无原始对象的合成/旧输入只能按观测范围对账，不能声称整月审计。
- 这轮没有新增 backtest 或 C4 文件改动；前述 `crypto.py` 单行诊断仍是唯一协调点。OneContext 历史库本轮仍无法打开，工作日志只写本文件，遵守不写 worktree 外的限制。

### 返工回归与验收

- 全部 crypto 及市场注册、backtest、step 回归：**538 passed**，4 workers，9.15 秒，零失败/跳过。命令与首次交付相同；返工新增 12 个实例均有修复前失败记录。
- Ruff lint 与 **780 个文件**格式检查通过，`git diff --check` 通过。使用本 worktree venv；未改 Rust 或任何 C4 内核文件。
- `bench/mutations_crypto_recent.py`：**19/19 被杀死**，无语法/导入/收集错误冒充成功，源码逐项恢复后基线通过。原 12 项之外增加：退出列资金费上限、退出列价格上限、新归档沿用旧 REST、月尾缩小审计、新请求起点裁掉整月重取、跳过 REST 原文核对、校验器放行严格版提前截断。变异前检查导入路径均属本 worktree，结果见 `bench/results/mutations_crypto_recent.json`。

最终源码上，在同一个 `tmp/crypto` 副本重新执行 sync（无 refresh、跳过公告）、严格与推断构建、verify 和两版例子。此次同步下载 19,547,820 字节；资金费月归档新增/变更均为 0，所以真实样本未触发“新月档发布后价格修订”，该路径由上述合成 HTTP 链路验收。

| 最终验收项 | 严格版 | 推断版 |
|---|---|---|
| UTC 右端（不含） | 2026-10-31 23:00 | 2026-11-15 00:00 |
| 行数 | 239,612 | 240,960 |
| verify | pass，0 错误，963 份归档 | pass，0 错误，963 份归档 |
| 差异报告数 | 0 | 0 |
| 版本前缀 | 5ae4f191bced | bf894f27e183 |
| 永续最终净值 | 2.251744618123367 | 2.283899853755265 |
| 永续净资金费支付（USDT） | 229.41495192123983 | 228.92069388136807 |
| 永续费用（USDT） | 97.068293651 | 97.927781976 |
| 现货最终净值 | 2.633912380760807 | 2.6145322027303073 |
| 现货费用（USDT） | 137.74194243920002 | 138.90761996970002 |

严格面板再次与原 C3 版本全部字段逐位一致，完整 IPC SHA-256 仍为 `207a0523c53f4fbc668e3ebc2b1ee2c95cdb8a723792e60c26ce040c0423b9ae`。推断面板与返工前 C5 推断版也逐位一致，严格窗口前缀相同；仍延伸 337 小时，BTC/ETH 各 43 笔推断结算。两版例子的所有汇总和手算检查（除数据版本 ID）与返工前完全一致；两版均 valid、无强平、无第一档超限。严格与推断的差异仍来自延伸区间的价格、信号/交易、资金费及年化样本长度。

公开汇总已经更新为本轮最终结果：`crypto_c5_checks.json`、`crypto_c5_comparison.json`、两版 prepare/examples、严格逐位报告及变异报告。原始数据、冻结输入、逐格校验只在忽略目录 `tmp/crypto`；检查全部 33 个新增/修改文件，无本机绝对路径、真实逐行行情或 worktree 外文件变更。未提交、未推送。

## 第二轮审查返工

完整读取 `.review/review_c5_r2.md`。上一轮四条修复获确认；本轮新增 P1 是将 `strict_end >= stop` 错当成“归档已经证明没有新结算”，导致请求 21h 或已知退出 21h 时漏掉 REST 中的 20h 结算。默认严格构建未改。

- `recent.py` 删除这项提前跳过。推断模式始终从归档末笔检查到有效上限，再决定是否延伸或截断；只有退出之后不再要求覆盖。fundingInfo 缺失/不匹配时先把有效上限限制在严格上限内，仍检查此范围内的 REST。
- `verify_recent.py` 独立扫描相同范围的原始记录，推断模式不再按严格上限或 fundingInfo 授权状态跳过检查；不复用构建 helper。构建与校验保留各自的末笔覆盖判断。
- 新增 `test/crypto/test_c5_review_r2.py`：`test_rest_inside_short_horizon_stops_extension` 验证两个反例均截到 16h；`test_verifier_rejects_unchecked_rest_inside_short_horizon` 独立造出漏费面板，并用 20h/4h 归档对照确认该笔应为 0.0001；`test_rest_after_effective_end_preserves_exemption` 确认有效终点之外的坏间距不截短。三组均覆盖 fundingInfo 匹配、缺失和不匹配。
- 先在修复前源码跑出 **12 failed、6 passed**，再修改生产代码；失败摘要在 `bench/results/crypto_c5_review_r2_red.txt`。修复后上述 18 项通过，上一轮回归和原 C5 测试也通过。
- 保守选择：未获 fundingInfo 支持时不扩大检查/构建到严格上限之外；但不能忽略严格上限内已经观察到的结算。这符合推断选项的条件限制，不借此改默认严格版。生命周期豁免与结算记录检查分开处理，不改 DESIGN 或 C4 文件。
- OneContext 仍报历史数据库无法打开；本轮记录只写本文件，遵守不写 worktree 外的限制。

第二轮验收：全部 crypto 及相关市场/backtest/step 回归 **556 passed**（4 workers，10.85 秒，零跳过）；Ruff lint 与 **781 个文件**格式检查、`git diff --check` 通过。源码变异 **22/22 被杀死**，新增构建跳过短窗口、校验跳过退出前窗口、校验跳过 fundingInfo 未确认窗口三个变异；导入均来自本 worktree，无语法/导入错误冒充成功，源码恢复后基线通过。结果写入原有 `bench/results/` 汇总；本轮只改数据判据、测试、变异脚本和文档，没有增加 backtest/C4 改动。

真实副本本轮使用同一冻结同步计划，重新构建严格/推断版、verify 并重跑两版例子，未额外 sync。两版均 **0 错误、963 份归档**；严格右端仍为 **2026-10-31 23:00 UTC**（239,612 行），推断为 **2026-11-15 00:00 UTC**（240,960 行）。严格版与原 C3 所有字段及浮点位模式一致，IPC 哈希仍为 `207a0523c53f4fbc668e3ebc2b1ee2c95cdb8a723792e60c26ce040c0423b9ae`；两版内容版本 ID 与第一轮相同。例子完整输出（包括手算核对）与第一轮完全相同：永续最终净值严格/推断为 **2.251744618123367 / 2.283899853755265**，现货为 **2.633912380760807 / 2.6145322027303073**；资金费、费用等详见上一轮表及本轮重写的公开汇总。35 个新增/修改文件扫描无本机绝对路径或真实逐行行情；没有写 worktree 外文件，未提交、未推送。
