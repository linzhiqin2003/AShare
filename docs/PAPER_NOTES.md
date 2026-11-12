# paperbroker 第三轮修复记录

2026-11-12。范围为 P1/P2；没有提交、推送、修改 panelbt/tradecore/数据层，没有读取密钥或调用需密钥的接口。先阅读 paper_review2.md、paper_exp2 脚本及本地项目约定；OneContext 查询无历史命中。本轮原始运行记录在被忽略的 `.cache/round3/`，不作为入库产物。第一、二轮的代码和守护测试继续保留；旧格式、旧基准、旧输出表由本记录替代。

## 审查逐项回应

测试简称：core3 = `packages/paperbroker/tests/test_round3.py`；app3 = `test/paper/test_round3.py`；core2/app2 = 对应目录的 `test_revision.py`；wide = 包内 `test_wide.py`；journal = 包内 `test_journal.py`。

| 编号 | 处理与位置 | 守护测试（可按名称检索） |
|---|---|---|
| N1/P0 | 已修。runner 盘前查 `actions_through` 之后至今天的每个交易日；core 按证券保存最后除权日，按日期补做，同证券连续日期不再被每日布尔标记挡住。查询失败停在盘前，成功进度已落 WAL，重启从未完成日期续做 | app3 `test_outage_actions_in_date_order` 四场景；`test_outage_query_failure_blocks_trading_and_resumes_cursor`；core3 `test_historical_action_idempotence_and_consecutive_dates` |
| N2/P1 | 已修。根 release profile 开 overflow-checks，包含生成机器码的绑定包；删除两子包无效 profile。扩展捕获陷阱并恢复状态，抛 OverflowError；Journal.feed 先试跑，失败不写行情和 WAL | core3 `test_release_extension_dividend_overflow_is_atomic`、`test_feed_overflow_preflight_does_not_poison_wal`，均经实际 release Python 扩展 |
| N3/P1 | 已修。runner 按证券检查 15:00 截点，默认 15:03 后继续结算；过期证券代理量为 0，保留最后标价，日报列 `stale_close_symbols`。CLI 可设 `--settlement-deadline` | app3 `test_close_deadline_is_per_security`：一只/全部冻结、截止前后、录制代理量和日报标价 |
| N4/P1 | 已修内存和缓存累积。core 换日删除终态订单、保持全局订单 ID；history/pending/daily_states 只留当日；日报一次归档；快照校验后只留 N 份；行情按保留交易日及恢复锚点清理；gc CLI、status 分项；恢复流式扫描 WAL | core3 `test_bounded_snapshots_archived_events_and_gc`；app3 `test_reports_not_rewritten_status_throttled_gc_cli`；40 日/全日实测见后文。历史日报仍增长，不能同时满足“永远保留日报”和“除日志外总量恒定”，见设计疑问 |
| P2-1 | 已修。consume 将接收时间回拨的 ValueError 转安全的 SourceError，run 退避等时钟追上 | app3 `test_clock_rollback_run_survives_and_targets_back_off` |
| P2-2 | 已补四个缺口：盘口价在量为零的开/收盘双侧锁；无日志接收时间倒退；未结算跨日；两天不同累计量基线 | core2 `test_auction_limit_lock` 12 格；core3 `test_receipt_backwards_without_journal_and_unsettled_rollover`；app2 `test_two_days_baseline_report_and_calendar_refresh`；6 个存活变异均重做 |
| P2-3 | 已加强。60 种子×8 日，限价 60% 取对应涨跌停价，深度高比例为零，增加竞价步骤。匹配时统计有效挂单实际被锁，不再把输入行计成覆盖 | wide `test_wide_random_differential`，18 格与各档案成交计数均断言 >0；计数口径见后文 |
| P2-4 | 已修。每个输出有持久化单调 seq，events(after=seq) 严格取后续；归档按需读取；README 更新至少一次说明 | core3 `test_bounded_snapshots_archived_events_and_gc`，跨日/重启/边界游标 |
| P2-5 | 已修。profiles.py 识别 688/689；其他代码段核对表见后文 | app3 `test_code_profiles` 17 参数用例 |
| P2-6 | 已修。主板卖单余数等于可卖持仓余数，或为整手；科创/北交 step=1 不足起点余额须一次卖完；target_weights 保留可卖零头 | core3 `test_mainboard_residual_lot`、`test_step_one_residual_and_target_reconciliation`、`test_target_sells_residual_without_dropping_it` |
| P2-7 | 按要求不改。配置/既有证券档案冻结，只能追加证券；无 v1/v2→v3 迁移工具；明确拒绝旧格式，不能静默重算旧账 | manifest/版本拒绝测试；README |
| P2-8 | 已修。完全相同价量盘口、相同时间的普通行可一次升级为竞价行；须处于竞价阶段且该场尚未完成；其他同时间行情仍拒绝 | app3 `test_early_close_quote_can_be_enriched_once` |
| P2-9 | 按要求不改。CostModel/LotRule 已按证券抽象；SimpleCosts/SimpleLots 仍为临时规则，tradecore 接入与回测对账属 P3 | 档案/费用/T+1 现有测试继续运行 |
| P2-10 | 已清理。bench/paper 只留两份 JSON 及生成脚本。变异/构建验证脚本移至 test/paper，冒烟/盘后核对入口移至 ashare.paper；一次性结果删除，结论记在本文 | 文件清单检查 |
| P2-11 | 已修。宽随机混入主板、科创、北交、ETF，逐批显式传接收时间；不同成交路径、各档案成交都有非零断言 | wide，同 P2-3 |
| P2-12 | 已修。已有日报不重写，账户内归档/日报用硬链接去重；status 默认每 30 秒至多写一次 | app3 `test_reports_not_rewritten_status_throttled_gc_cli` 检查 mtime 不变 |
| P2-13 | 已修。在提交目标订单前记尝试时间；即使拒绝也按 rebalance_seconds（默认 30 秒）退避 | app3 `test_clock_rollback_run_survives_and_targets_back_off`，65 秒恰好尝试 3 次 |

## 协议与操作

状态、manifest、WAL、快照版本现在为 3；行情数组仍为 `int64[N,18]`，Parquet 列格式仍为 2。每证券 corporate_dates、actions_through、event_sequence、order_offset 都是状态的一部分，同输入序列保持确定性。当天 pending/history 在换日清掉；需跨日可靠消费时持久化 seq，使用 `events(after=seq)`。没有 journal 时只可读取当天事件；游标不能替消费者保证外部操作恰好一次。

`Journal.apply/feed` 都先在临时内核试跑，之后行情文件落盘、WAL 落盘、真实执行。macOS 使用 F_FULLFSYNC。溢出是原子异常拒绝（OverflowError），不是业务 rejected 输出；坏输入不能成为无法重放的历史事实。绑定回滚包括状态与输出队列。

默认保留最新 3 个校验过的快照、最近 20 个有录制的交易日；`snapshot_retention>=1`，`quote_retention_days=0` 关闭行情清理。`checkpoint()` 只清旧快照；`journal.gc()` 另清所有保留恢复锚点之前、超过保留日数的行情。WAL 不删除。删除行情前先持久化 recovery_floor；全部保留快照损坏后会明确失败，不能假装可从缺行情的 WAL 重放。完整从头回放须单独保留录制/备份。恢复流式校验日志，不把全年日志一次读入内存。

```sh
uv run python -m ashare.paper start --directory data/paper-demo --full-market \
  --settlement-deadline 15:03 --snapshot-retention 3 --quote-retention-days 20
uv run python -m ashare.paper status data/paper-demo
uv run python -m ashare.paper gc data/paper-demo --snapshot-retention 3 --quote-retention-days 20
uv run python -m ashare.paper.smoke --directory data/paper-smoke
uv run python -m ashare.paper.verify_live --directory data/paper-smoke --day 20261112
uv run python -m ashare.paper import-raw /path/to/raw/20261112 --day 20261112 --output /path/to/imported
uv run python -m ashare.paper replay /path/to/imported/quotes --day 20261112 \
  --script /path/to/imported/script.json --output /path/to/result.json
```

`--dry-run` 使用内存账户、不创建账户录制/日报目录。目标文件：`{"version":1,"at":"09:35:00","until":"14:56:50","rebalance_seconds":30,"protection_ticks":2,"weights":{"EXAMPLE":1.0}}`。卖出未成交不提前使用其现金；按比例预算、多轮重算；停牌/T+1/封板/保护价都可能造成目标差额。

导入原始录制按 t_recv 排序；trade_time 是逐证券快照时间，用于日期、新鲜度和委托相对行情顺序；t_send 为本机请求起点；t_recv 为本机响应接收时刻，驱动事件时钟。t_recv−t_send 为请求耗时，t_recv−trade_time 是两类时钟偏差，不等同于纯网络延迟。保存三个时间；盘后才产生的 auction_o/c 不回填到盘中撮合。全市场限速每轮 6 秒、rt_k 桶默认 40/min，其他接口各 400/min；40203 识别限频。单出口锁仅本进程/本机可协调，运行时其他机器不可同时用同一账号取数。

## 证券档案核对

以下是当前**申报数量/报价单位**分类，不能据此宣称已覆盖完整费用、订单上限、所有基金 T+0 规则。

| 代码段 | 档案 | 买入起点 / 步长 | 报价单位 | 卖出余额 |
|---|---|---|---|---|
| 688/689.SH（含存托凭证） | star-v1 | 200 / 1 | 0.01 | 不足 200 的可卖余额一次卖完 |
| 300/301/302.SZ | cash-equity-v1 | 100 / 100 | 0.01 | 整手，或一次包含全部不足 100 的零头 |
| 920/8xx/4xx.BJ | bse-v1 | 100 / 1 | 0.01 | 不足 100 的可卖余额一次卖完 |
| 沪 ETF：51/52/56/58.SH | fund-v1 | 100 / 100 | 0.001 | 同主板零头口径 |
| 沪 LOF：50.SH | fund-v1 | 100 / 100 | 0.001 | 同上 |
| 深 ETF：15.SZ（含 158/159） | fund-v1 | 100 / 100 | 0.001 | 同上 |
| 深 LOF：16.SZ（160001–169999） | fund-v1 | 100 / 100 | 0.001 | 同上 |
| 其他普通沪深股票 | cash-equity-v1 | 100 / 100 | 0.01 | 同主板 |

核对资料：[上交所交易业务零股说明](https://www.sse.com.cn/lawandrules/guide/stock/jyglywznylc/tz/c/c_20230209_5716007.shtml)、[深交所交易规则解读](https://www.szse.cn/www/investor/index/update/t20230706_601604.html)、[深交所证券代码区间](https://www.szse.cn/marketServices/technicalservice/doc/P020240510623040090566.pdf)、[深交所基金业务指南](https://docs.static.szse.cn/www/lawrules/service/fund/W020240619367937222417.pdf)、[上交所统计年鉴中的 688/689 编码](https://www.sse.com.cn/aboutus/publication/factbook/documents/c/10170562/files/da4f403a0b7443fa9b2ba2e002d38f10.pdf)。北交 100/1 口径按本任务已确认规则测试；`.BJ` 统一分类。17.SZ 非上市开放基金申赎代码不当成 LOF；完整证券主数据和基金差异需 P3 校准。纯包不猜代码，调用方可明确传 Instrument。

## 复现脚本修复结果

只改 reviewer 脚本副本的路径断言，原文件未改。`x_outage.py` 原场景 11 日净值 999982.5，跨停机除权后错误降为 961477.5；修复后输出：

```text
11-11 settled: 20261111 | A shares 7000 | equity 999982.5
11-13 after restart: settled 20261113
A shares 16000 | mark 4.5 | equity 999977.5
```

查除权日期包括 20261112，先补 7000→14000 股及现金 7000 元，再正常调仓多买 2000 股；净值只少这次新订单 5 元费用。`x_overflow.py` 原来第 9 次分红负数回绕，修复后：

```text
RESULT: trapped at dividend #9: OverflowError: accounting overflow; input not applied
```

另一个真实扩展测试让卖出入账触发普通 i64 加法溢出，确认不是只靠分红显式 checked_add。发生异常前后的 dump、WAL 字节、行情文件集合完全相同，关闭重启 dump 一致。发布配置失效变异由这个测试抓住。

## 长期体积与内存实测

运行：`uv run --no-sync python -m bench.paper.retention`。5500 只、每天 4 批、200 笔订单，连续 40 次结算（合成 day ID，无假装实际日期），使用与应用相同的 ParquetStore/F_FULLFSYNC。每天清缓存/垃圾并读取 Mach task_info 的**当前 RSS**；不是累计峰值，不调用被沙箱禁止的 ps。单位均为十进制 MB，目录体积为文件逻辑字节，硬链接去重；并行机器负载会干扰耗时和内存。原始精确数字在 benchmark-results.json。

| 结算日 | 单快照 MB | 账户总 MB | WAL MB | 历史日报 MB | 当前 RSS MB |
|---|---:|---:|---:|---:|---:|
| 1 | 3.472 | 5.438 | 0.051 | 0.526 | 181.010 |
| 2 | 3.482 | 9.576 | 0.103 | 1.057 | 202.457 |
| 3 | 3.482 | 13.714 | 0.154 | 1.589 | 220.922 |
| 4 | 3.482 | 14.380 | 0.206 | 2.120 | 238.436 |
| 5 | 3.496 | 15.052 | 0.257 | 2.653 | 263.848 |
| 6 | 3.498 | 15.728 | 0.309 | 3.187 | 281.838 |
| 7 | 3.498 | 16.404 | 0.361 | 3.722 | 301.744 |
| 8 | 3.498 | 17.066 | 0.413 | 4.256 | 318.669 |
| 9 | 3.500 | 17.729 | 0.465 | 4.791 | 335.839 |
| 10 | 3.508 | 18.403 | 0.516 | 5.331 | 353.288 |
| 11 | 3.510 | 19.080 | 0.568 | 5.871 | 361.185 |
| 12 | 3.510 | 19.755 | 0.620 | 6.411 | 364.151 |
| 13 | 3.510 | 20.423 | 0.672 | 6.951 | 367.182 |
| 14 | 3.510 | 21.088 | 0.724 | 7.492 | 366.051 |
| 15 | 3.510 | 21.754 | 0.776 | 8.032 | 343.032 |
| 16 | 3.510 | 22.419 | 0.828 | 8.572 | 348.652 |
| 17 | 3.510 | 23.085 | 0.880 | 9.113 | 357.892 |
| 18 | 3.510 | 23.751 | 0.931 | 9.653 | 360.366 |
| 19 | 3.510 | 24.416 | 0.983 | 10.193 | 304.005 |
| 20 | 3.510 | 25.082 | 1.035 | 10.733 | 310.821 |
| 30 | 3.510 | 31.003 | 1.554 | 16.136 | 348.701 |
| 40 | 3.510 | 36.925 | 2.072 | 21.539 | 329.531 |

每天 history=1204、daily_states=1，第三天起快照恒为 3 个；20 天后行情恒为 80 份/1.469 MB，快照合计 10.529 MB。第 20/40 天单快照完全相同 3,509,745 字节。RSS 预热后在约 300–350 MB 波动。目录第 20/40 天仍从 25.082 MB 增至 36.925 MB，增量主要是保留的日报（每个约 0.540 MB）和不可清的 WAL；不能把该总量称为恒定。

全市场完整一天另实测 5500 只×2400 批（6 秒/批）+200 笔订单：2603 条 WAL，**866604 字节**；Parquet **44119200 字节**；含录制、强制落盘、快照、结算 **71.418 秒**。合成行情高度可压缩，这个 Parquet 体积不能用来预估真实全市场硬盘需求。这里只统计一次原始流的持久化，不是联网等待耗时。

## 宽随机覆盖

运行 `uv run --no-sync pytest packages/paperbroker/tests/test_wide.py -q -s`。每个候选计数必须是已经接受、还挂着、时间及限价可执行、有正参与率额度的订单；排除前序订单已经耗尽额度的情形，并验证本订单没有成交。表不是输入行计数。连续竞价的已知零价/零量同时也受盘口容量约束，所以**不声称仅移除封板判断就一定成交**；该双重防线与竞价可成交的独立反例分别由定点测试/双侧变异守住。

| 阶段 | 方向 | 缺盘口 (-1) | 价为 0 | 价在、量为 0 |
|---|---|---:|---:|---:|
| open_auction | buy | 2 | 4 | 11 |
| open_auction | sell | 3 | 5 | 5 |
| continuous | buy | 94 | 91 | 216 |
| continuous | sell | 69 | 63 | 118 |
| close_auction | buy | 8 | 18 | 26 |
| close_auction | sell | 8 | 6 | 12 |

总接受 16406、拒绝 37119、成交 15405、公司行为 209、结算 480、零股接受 1949。主板/科创/北交/基金成交分别 3654/2992/4535/4224；买入竞价/盘口/退回最新价 2891/2496/4098，卖出 1627/1656/2637。18 格、4 档案及原来的覆盖项均保留大于零断言。没有缩小种子数或事件范围、跳过失败测试。

## 设计疑问、边界及未验证项

1. “账户目录除日志外不随天数线性增长”与“往日输出留日报”同时要求，数学上无法同时严格满足。选择保留用户要求的审计日报，清掉重复状态、过期缓存和内存历史；明确报告总量真实增长，不把日报伪装成 WAL 或删历史。若要求总硬盘恒定，须另定日报外部归档/保留期限；本轮没有擅自删除日报。
2. “WAL 是事实来源”同时要求删除旧行情：WAL 只含引用，清掉录制后不能从零重放。选择保留全部 WAL 和至少一个有效恢复锚点，删除前公布恢复下界；完整历史重放依赖外部备份。3 快照损坏/硬盘丢失不由当前 GC 自动解决。
3. 跨停机日先结算上个未处理日、再按日期补公司行为，查询失败不交易。没有当天盘中录制的停机区间不能凭日线虚构成交；除权处理忽略延迟到账/上市日、碎股补偿，延续研究模型。
4. 截止前等齐收盘行，截止后停滞证券无竞价成交、按旧价估值，日报明确列出。收盘竞价使用累计量差代理仍不是逐笔成交事实，今天收盘前后的实录尚未验证，不声称真实市场联调完成。
5. 对溢出采用原子异常，不把财务错误饱和裁切或无声继续；Rust 的默认 panic hook 在人为极端输入测试中会向 stderr 打印固定的算术错误位置，不含密钥。真实程序 run 捕获后告警退避。
6. 配置及档案冻结、旧格式不能自动迁移；tradecore/P3 对账、P4 守护进程、完整交易所订单/禁撤单/T+0 基金规则仍在范围外。POSIX 持久化已测，Windows 未测；没有物理断电试验。
7. 基准同时受其他本机任务影响；合成报价压缩率、委托密度不能代表实盘吞吐上限。恢复校验 WAL 时间仍随日志长度增长，但扫描内存不随长度增长。events(after) 会按需加载历史日报，调用方主动索取全部历史时，其返回列表本身会占用相应内存。

## 基准与共享文件

本轮重测 5500 行/批（7 组×100 次，中位数）：Python 循环 1.669 ms、NumPy 向量化 0.125 ms、Rust 0.064 ms；三者先校验保留行情相同。Python/NumPy 基线只做行情接收，不能声称已实现完整撮合。4800 批×5500（2640 万行）+400 笔订单：三次 0.799/0.771/0.756 秒，中位数 0.771 秒，400 笔全部成交。脚本 `bench/paper/benchmark.py` 保留长期测试字段，`retention.py` 更新长期/持久化测量，`coverage.py` 重跑合成覆盖及外置夹具端到端，但 JSON 只保存合成覆盖计数。

相对第三轮开始保存的副本，根共享文件仅 `Cargo.toml` 变更：删除原第 15 行 `[profile.release.package.paperbroker-core]`，原第 16 行 `overflow-checks = true` 变为当前第 15 行，归入第 11 行 `[profile.release]`，覆盖绑定与其生成代码。Cargo.lock、pyproject.toml、uv.lock 本轮逐字节未变，Makefile 未改。git diff 看到的其他共享差异来自前两轮，不能算成本轮新改。两子包各自无效 release profile 已删除，干净构建没有这些警告。

## 数据与夹具检查

真实夹具不在仓库。测试先取 ASHARE_DATA_DIR/fixtures/paper/tushare，未设置时从测试文件父目录逐级查找第一个 data/fixtures/paper/tushare/INDEX.json；配置了环境变量但路径不存在时明确 skip，不偷偷改用别处。该机器使用外置的 22 个文件，原位置没有重新创建。全新副本验证显式设置 ASHARE_DATA_DIR，主 worktree 验证祖先查找。

扫描 test/、bench/、docs/、packages/ 全部非缓存源码/文档/数据候选：未发现与 22 份夹具相同内容的文件、完整 response JSON 或 Parquet/CSV/NPY/NPZ/GZIP 数据文件。JSON 中没有接口 data.items 或真实逐证券快照序列；手写测试中的几行示例数值保留。coverage-results 已去除旧 E2E 财务汇总，仅包含合成测试计数。忽略目录 .cache 内仅有开发临时产物，不是公开仓库内容；没有把用户的原始录制复制进代码目录。

## 变异重跑证据

`uv run --no-sync python test/paper/mutations.py`：先运行未修改代码的完整 make check；每项单独注入；Rust 项构建 release 扩展，检查实际导入路径属于本 worktree、SHA256 等于 target 产物且不同于基线；恢复源码后重建并校验哈希回到基线，才运行下一项。仅 pytest 已收集用例退出 1 且明确 FAILED 才算捕获，编译/收集错误不算。22/22 捕获，恢复后完整检查通过。本轮重跑的是指定 6 个存活项 + 16 个新增项；**没有宣称本轮重新跑过全部 45 项**，第一轮 19 项/复核其他 39 项通过来自审查人的独立结论和保留的原测试。

首次尝试的 T2 只替换了去重条件，没有同时恢复旧的布尔写入方式，因历史日的布尔值仍为 false，不能制造“连续日被挡”的预期错误。补齐该变异后原四场景测试能捕获；另外保留第一次改法作为 T16，新增直接重复提交历史除权日的测试，证明确实会漏掉幂等保护。没有删除存活证据或放宽断言。

基线与最终恢复扩展 SHA256：`19ca9a220daaa2be3247656b7e70483eb5943d27189dcc72a17242c3ab89930f`。22 项的具体失败 node ID、各次扩展/失败输出 SHA256 记录在忽略目录 `.cache/round3/mutation-results.json`；可用保留脚本重做，不将一次性 JSON 放入 bench。

| ID | 注入错误 | 实际失败的测试（省略参数） | 结果 |
|---|---|---|---|
| R3 | 竞价封板忽略卖一量为零 | `test_auction_limit_lock` | 捕获（exit 1） |
| R6 | 内核接收时间允许回拨 | `test_receipt_backwards_without_journal_and_unsettled_rollover` | 捕获（exit 1） |
| R10 | 上一日未结算可换日 | `test_receipt_backwards_without_journal_and_unsettled_rollover` | 捕获（exit 1） |
| R12 | 跨日不清收盘基线 | `test_two_days_baseline_report_and_calendar_refresh` | 捕获（exit 1） |
| D9 | 两侧同时允许未结算换日 | `test_receipt_backwards_without_journal_and_unsettled_rollover` | 捕获（exit 1） |
| D10 | 两侧同时忽略盘口量为零 | `test_auction_limit_lock` | 捕获（exit 1） |
| T1 | 停机后仅查今天除权 | `test_outage_actions_in_date_order` | 捕获（exit 1） |
| T2 | 恢复旧的每日布尔公司行为锁 | `test_outage_actions_in_date_order` | 捕获（exit 1） |
| T3 | 关闭实际 release 溢出检查 | `test_feed_overflow_preflight_does_not_poison_wal` | 捕获（exit 1） |
| T4 | feed 删除试跑，坏输入先落盘 | `test_feed_overflow_preflight_does_not_poison_wal` | 捕获（exit 1） |
| T5 | 超过截止仍等待全部收盘行 | `test_close_deadline_is_per_security` | 捕获（exit 1） |
| T6 | 不删除旧快照 | `test_bounded_snapshots_archived_events_and_gc` | 捕获（exit 1） |
| T7 | 不清旧行情 | `test_bounded_snapshots_archived_events_and_gc` | 捕获（exit 1） |
| T8 | 输出序号不递增 | `test_bounded_snapshots_archived_events_and_gc` | 捕获（exit 1） |
| T9 | 游标包含边界旧事件 | `test_bounded_snapshots_archived_events_and_gc` | 捕获（exit 1） |
| T10 | 主板零股只能全部清仓 | `test_mainboard_residual_lot` | 捕获（exit 1） |
| T11 | 内存历史永久保留 | `test_bounded_snapshots_archived_events_and_gc` | 捕获（exit 1） |
| T12 | 早到收盘行不能升级竞价 | `test_early_close_quote_can_be_enriched_once` | 捕获（exit 1） |
| T13 | 目标拒绝后不记尝试时间 | `test_clock_rollback_run_survives_and_targets_back_off` | 捕获（exit 1） |
| T14 | 每轮强制写 status | `test_reports_not_rewritten_status_throttled_gc_cli` | 捕获（exit 1） |
| T15 | 过期证券仍有竞价代理量 | `test_close_deadline_is_per_security` | 捕获（exit 1） |
| T16 | 历史除权日可重复应用 | `test_historical_action_idempotence_and_consecutive_dates` | 捕获（exit 1） |

## 最终验收与真实录制

主 worktree 最终执行 `make dev && make check` 成功；全新目录通过 rsync 创建，排除 target/、.venv/、.cache/、.git、*.so、*.pyc 和 Python 缓存，然后独立构建并执行同样命令。新增最后一个历史除权幂等测试后也更新副本重跑；最终逐文件比较 Python/Rust/TOML 代码，副本与交付工作树完全相同。两处均 180 个 Python 测试通过，跳过 0 个；Rust 7 个测试（paperbroker-core 6、panelbt-core 1），ruff/check/format、cargo fmt、clippy 均通过。没有无效子 profile 警告。

主 worktree make check 末尾原始输出：

```text
packages/paperbroker/tests/test_engine.py ..........                     [ 73%]
packages/paperbroker/tests/test_journal.py ........                      [ 77%]
packages/paperbroker/tests/test_revision.py ........................     [ 91%]
packages/paperbroker/tests/test_round3.py ...............                [ 99%]
packages/paperbroker/tests/test_wide.py .                                [100%]

paper fixture/test skips: 0
============================= 180 passed in 44.69s =============================
```

全新目录末尾原始输出：

```text
packages/paperbroker/tests/test_wide.py .                                [100%]

paper fixture/test skips: 0
============================= 180 passed in 45.42s =============================
```

本轮 make dev 后再次核对原生扩展实际导入路径在本 worktree，SHA256 仍为变异基线 19ca9a220daaa2be3247656b7e70483eb5943d27189dcc72a17242c3ab89930f，且等于 target/release/libpaperbroker_native.dylib；不存在恢复源码但继续加载变异扩展的残留。未修改 panelbt；两个包仍由 make dev 正常构建，独立 wheel 的顺序构建脚本保留于 test/paper/verify_build.py，本轮未另重跑该脚本，不冒充第二轮碰撞验证为新证据。

收尾于 2026-11-12 08:13:39 本地检查用户指定的 data/live/raw/20261112：目录存在，*.json.gz 数为 **0**。尚未到 09:05 录制启动时间，所以本轮没有实际原始录制可导入；没有编造盘前返回、trade_time 延迟、旧日期行或倒量观察。导入器的三个时间和离线回放仍由合成原始录制测试守住；真实 15:00 截止行为要等收盘后的数据，按任务要求不等待。真实接口冒烟、真实十分钟采集、原始录制实盘行为、P3 对账、P4 常驻均未验证。

## 文件清单及大小

以下为当前 git 尚未跟踪、在本任务允许范围内的全部文件（包括前两轮交付尚未提交的文件）；星号表示第三轮新出现的路径，其中 4 个为从 bench/paper 移动的脚本。单位为字节，排除本地扩展和缓存。根共享文件改动单独列于上文。第三轮新建代码为 coverage.py、retention.py、两份 test_round3.py；移动为 smoke.py、verify_live.py、mutations.py、verify_build.py。

| 文件 | 字节 |
|---|---:|
| bench/paper/benchmark-results.json | 18736 |
| bench/paper/benchmark.py | 4203 |
| bench/paper/coverage-results.json | 2917 |
| * bench/paper/coverage.py | 1452 |
| * bench/paper/retention.py | 5017 |
| crates/paperbroker-core/Cargo.toml | 152 |
| crates/paperbroker-core/src/lib.rs | 38789 |
| docs/PAPER_NOTES.md | 28113 |
| packages/paperbroker/Cargo.toml | 334 |
| packages/paperbroker/README.md | 13595 |
| packages/paperbroker/README.zh-CN.md | 5131 |
| packages/paperbroker/pyproject.toml | 693 |
| packages/paperbroker/python/paperbroker/__init__.py | 433 |
| packages/paperbroker/python/paperbroker/_core.pyi | 529 |
| packages/paperbroker/python/paperbroker/api.py | 17598 |
| packages/paperbroker/python/paperbroker/journal.py | 13894 |
| packages/paperbroker/python/paperbroker/py.typed | 0 |
| packages/paperbroker/python/paperbroker/reference.py | 19332 |
| packages/paperbroker/python/paperbroker/replay.py | 1133 |
| packages/paperbroker/src/lib.rs | 3385 |
| packages/paperbroker/tests/conftest.py | 250 |
| packages/paperbroker/tests/test_engine.py | 20118 |
| packages/paperbroker/tests/test_journal.py | 7066 |
| packages/paperbroker/tests/test_revision.py | 10264 |
| * packages/paperbroker/tests/test_round3.py | 8578 |
| packages/paperbroker/tests/test_wide.py | 16185 |
| src/ashare/paper/__init__.py | 72 |
| src/ashare/paper/__main__.py | 8164 |
| src/ashare/paper/importer.py | 6357 |
| src/ashare/paper/market.py | 6290 |
| src/ashare/paper/profiles.py | 1484 |
| src/ashare/paper/runner.py | 23464 |
| src/ashare/paper/schedule.py | 1720 |
| * src/ashare/paper/smoke.py | 253 |
| src/ashare/paper/source.py | 13057 |
| src/ashare/paper/storage.py | 2702 |
| * src/ashare/paper/verify_live.py | 3270 |
| test/paper/__init__.py | 0 |
| test/paper/conftest.py | 600 |
| * test/paper/mutations.py | 13312 |
| test/paper/test_end_to_end.py | 6135 |
| test/paper/test_revision.py | 19238 |
| * test/paper/test_round3.py | 12000 |
| test/paper/test_runner.py | 8593 |
| test/paper/test_sources.py | 6517 |
| * test/paper/verify_build.py | 4517 |
