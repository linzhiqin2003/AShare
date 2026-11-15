# C7：BTC、ETH 分钟线实现与验收

本任务仅新增文件，未改已有实现、DESIGN、小时数据或路径入口；未提交、未推送。真实 ZIP、Parquet、逐小时行情差异仅位于被忽略的 `tmp/crypto_minute/`，公开结果只有汇总数字及合成测试。

## 实现与合并事项

- 新增 `src/ashare/data/crypto/minute.py`：同步、整理、内容版本、Polars 月文件切片读取、小时聚合、状态和四个 CLI 子命令。
- 新增 `src/ashare/data/crypto/verify_minute.py`：标准库 CSV 独立解析、源字段逐项对账、整月网格与缺口独立重建、内容哈希核对，以及固定严格小时版本的逐小时比较。
- 下载和存储复用 `PublicClient.archive` / `RawStore`，锁和 JSON 发布复用 `lock` / `atomic_json`，归档路径由既有 `archive_key` 的 1h 路径转换为 1m；时间范围复用 `timeutil`。没有另建下载器或存储层。
- 所有数据入口接受显式 `root`。**D1 合并后改为 `ashare.paths.dataset_dir("crypto_minute")`**：目前 `minute.data_root` 与独立验证入口各有一个任务书指定的暂定默认值。后者可复用路径入口，但不复用整理判据。
- “只新增文件”和“CRYPTO_DATA.md 加一节”冲突，按前者执行：完整使用文档为 `docs/CRYPTO_MINUTE.md`，新增 `docs/C7_CRYPTO_DATA.patch` 提供原文件的追加章节。已通过 `git apply --check`，由审查人在合并时应用；本工作区不改原文件。
- 全局 worklog 位于本项目外，与本任务禁止外写冲突；本次阶段记录保存在 `docs/C7_WORKLOG.md`，未外写。onecontext 检索尝试失败（历史数据库无法打开），实现以 DESIGN 和当前源码为依据。

## 保守选择与设计边界

1. 仅支持任务指定的八条序列。同步默认从 2020-01-01 起，终点是 UTC 今天零时。窗口要求整日，不抓最新未完成日，不用 REST；未发布的 CHECKSUM 返回 404 记 `missing`，其他失败阻止构建。
2. 月档只要某日有一根分钟就排除该日的日档。源档的重复和倒序均报错；先拒绝再排序合并，不以排序掩盖原始问题。新月档发布会替代相关日档的引用，旧版本和原始对象不动。
3. 开盘时间按量级判毫秒/微秒，支持两市场的两种单位，不按市场或年份猜。单档混用单位、无法无损转毫秒、网格不齐、越界都拒绝。13 位/16 位是明确接受范围，其他量级不猜测。表头按内容识别；空档、错误列数、空值、非有限数、负数拒绝。CSV 的收盘时间仍在 ZIP 中，但不额外引入小时库的异常收盘时间推断规则，因为本节只要求开盘时间网格和原始选定字段。
4. `quality.gaps` 对完整日历月逐段记空洞，不填值、不删异常、不推断维护。样本窗口和最近未完成月的未请求部分同样是该版本未覆盖区间，结合 `requested_days` 解读，不能把这些段误报为交易所缺数。每月质量记录始终满足“实有行数 + 缺口分钟数 = 日历月分钟数”。
5. 版本覆盖原档哈希/CHECKSUM、Parquet 字节哈希、全部质量记录与人工说明、请求日和定义版本。全局锁串行化同步/构建，Parquet 原子替换，最后发布版本和指针。被中断时不会把半截结果发布成新版本；旧读者仍可固定旧版本。
6. 读取只加载请求涉及的月，也校验这些月引用的原始 ZIP；不全库读取。元数据哈希始终校验，未读分区的物理文件由 `verify` 全检。不存在的月分区报错，不让“未下载”表现为无交易。
7. `verify.ok` 表示本地来源与产物一致，允许已记录缺口；`hourly_matches` 单独表示跨来源完全可对账。CLI 非零用于本地校验失败，自动对账验收必须额外读取 `hourly_matches`。官方来源冲突原样报告，不能静默改成相等。
8. 成交线比较 OHLC、量、额、主动买入额；标记价比较 OHLC。小时面板只保留指数 close，因此指数 open/high/low 无法用现有面板核验；成交笔数、主动买入量也没有小时对应字段。其分钟源值由独立 CSV 对账，求和规则由合成测试覆盖。
9. 对账的价格必须精确相等。量的每一处非零差异仍写入完整报告；只将不超过 `1e-8 + 1e-12 × abs(小时值)` 的差异归类为浮点舍入。真实源冲突另用 Decimal 原文求和排除了浮点解释。此容差仅用于诊断分类，不修改数据，不放宽既有测试断言。
10. `quality_notes` 可补充人工来源说明，会改变版本；它不允许覆盖自动计算的缺口。全量只保留最新内容时约 2 GB，未来上游修订、旧版本留存会额外占空间；没有自动 GC，以保留可复现性。

官方格式来源：[Binance Public Data README](https://github.com/binance/binance-public-data/blob/master/README.md)。它说明月/日归档、K 线列、现货微秒、CHECKSUM 以及归档可能修订；本实现实际以值域、文件内容和校验和为准。

## 回归与测试

测试全部离线，ZIP 在临时目录由合成数值生成，不引入真实行情夹具。覆盖毫秒/微秒、两种表头、未知/混合单位、非网格、倒序/重复、逐段缺口、月度优先、CHECKSUM、重试退避、中断续传、跨月只读所需文件、原档与 Parquet 篡改、OHLC/五种数量字段聚合、内容版本及旧版本复现、价格类型、CLI 和只读性。

独立性验证会把整理器的解析、缺口、哈希和聚合函数替换成“一调用就失败”的函数，验证器仍通过。另有六种重新计算内容哈希后的恶意产物测试，防止验证器只查哈希而不查语义：缺口删除、价格改变、行遗漏、重复、schema 更改、CHECKSUM 引用损坏。

开发自审发现的两处续传错误及一处性能外推错误先在修复前跑红，未放宽断言：

| 问题 | 旧行为 | 修复 | 证据 |
|---|---|---|---|
| 跨月续跑抹掉旧失败 | 3 月失败后只同步 4 月，会清掉失败并允许构建 | 仅清除本次实际重试月份的失败 | `bench/results/crypto_minute_resume_red.txt` |
| 失败月遗忘原请求 | 月档失败后缩短窗口续跑，丢掉之前请求的其他日 | 下载前先保存该分区请求日，后续取并集 | `bench/results/crypto_minute_request_red.txt` |
| 缓存时间误当下载吞吐 | 样本重跑命中缓存，仍能给出偏快的全量下载估计 | 实际下载量不足以覆盖样本时，网络外推返回 null | `bench/results/crypto_minute_bench_red.txt` |

变异检查 `bench/mutations_crypto_minute.py` 先确认导入来自本 worktree，再跑基线和恢复测试，逐处实际改源码并确保恢复原字节。**9/9 被抓到**：时间戳换算、缺口不记录、重复不报错、日度覆盖月度、开盘取末根、收盘取首根、量改为均值、独立验证忽略缺口、清空旧失败。语法或收集错误不计杀死。详见 `bench/results/mutations_crypto_minute.json`。

最终新增测试 **35 项通过**（32 项数据与 CLI、3 项验收脚本回归），结果见 `bench/results/crypto_minute_tests.txt`；与既有加密数据目录合跑时 **278 项通过**（已包含当时的 32 项分钟测试，后增的 3 项验收脚本测试单独通过），结果见 `crypto_minute_crypto_tests.txt`。Ruff 检查及格式、diff 空白检查、文档补丁试用检查均通过；新增文件扫描未发现本机绝对路径或数据文件，已有受跟踪文件无改动。

## 小样本验收与对账

样本为 BTCUSDT、ETHUSDT 的八条序列各取 2021-05、2025-03 两个完整月，再取 2026-11-01 至 11-03 三天。共 **16 份月档 + 24 份日档 = 40 份 ZIP**，24 个整理分区，**748,800 行**；原始 SHA-256/CHECKSUM、CRC、独立解析及逐字段对账全部通过。

两历史月份没有缺分钟。最近月每序列 4,320 行，其 11-04 至 12-01 的未采样段各有一个质量记录，共 8 段。8 次月档 404 是当前月尚未发布，日档正常补入，无同步失败。

对账始终固定小时严格版本 `525fcae799e5a7012fa038f2bc0c5bc5e004f53c88f721eeca26486736c33567`，通过 `ashare.data.crypto.open(..., version=...)` 读取；未写小时库。分钟样本版本 `04b338c6fdc93ad5c826f4a692144564aaed82bc3f13b5be8fba4a582d86f52b`。

历史样本共对比 **11,904 个“序列 × 小时”**。所有成交价 OHLC、标记价 OHLC、指数 close 的最大绝对差均为 **0**。量字段有 698 个仅浮点舍入的差异；另有 **3 个小时、9 个字段的实质来源差异**，全部发生在 2021-05 的永续成交线。

以下是逐小时差值的字段最大值（价格列的四项均为 0；指数仅比较 close）。

| 序列 | 月份 | OHLC 最大差 | volume 最大差 | quote_volume 最大差 | taker_buy_quote_volume 最大差 |
|---|---|---:|---:|---:|---:|
| BTC 现货 | 2021-05 | 0 | 1.819e-12 | 5.960e-8 | 5.960e-8 |
| BTC 现货 | 2025-03 | 0 | 9.095e-13 | 5.960e-8 | 2.980e-8 |
| ETH 现货 | 2021-05 | 0 | 2.910e-11 | 5.960e-8 | 2.980e-8 |
| ETH 现货 | 2025-03 | 0 | 7.276e-12 | 1.192e-7 | 2.980e-8 |
| BTC 永续 | 2021-05 | 0 | **317.982** | **15,748,016.85846** | **2,448,062.03699** |
| BTC 永续 | 2025-03 | 0 | 3.638e-12 | 4.768e-7 | 4.768e-7 |
| ETH 永续 | 2021-05 | 0 | **2,156.282** | **8,762,980.53111** | **2,250,792.28900** |
| ETH 永续 | 2025-03 | 0 | 1.164e-10 | 2.384e-7 | 1.192e-7 |
| BTC、ETH 标记价 | 两个月 | 0 | — | — | — |
| BTC、ETH 指数价 | 两个月 | close = 0 | — | — | — |

原因追溯：`bench/crypto_minute_source_audit.py` 从固定小时版本关联的原始归档重新核对 CHECKSUM，小时面板的 9 个值都等于其小时原档；以分钟 CSV 的十进制文本用 Decimal 求和，9 个差异仍成立。因此定位为**官方 1m 与 1h 原始归档不一致**，不是单位识别、漏分钟、浮点累加或小时面板整理造成。交易所内部为何出现该修订差异尚无证据，不推断更具体原因，不选一份覆盖另一份。

最近三日共 576 个“序列 × 小时”没有严格小时面板覆盖，明确列为 `hourly_unavailable`，不能宣称已完成跨来源对账；本地独立校验通过。完整逐条差异在被忽略的 `tmp/crypto_minute/reconciliation.json`，公开汇总为 `bench/results/crypto_minute_sample.json` 和 `crypto_minute_source_audit.json`。

同样窗口再次同步：**0 份归档下载、0 下载字节**，只复查 8 个未发布的月档，耗时约 8.54 秒。结果见 `bench/results/crypto_minute_resume.json`。

## 性能与全量外推

本 worktree 执行 `make dev` 建自己的 venv，Rust 构建 `CARGO_BUILD_JOBS=2`；重任务使用 `nice -n 15` 和 `timeout`，pytest 并行最多 4。沙箱拒绝再次调整 nice 值，但进程实际继承的优先级经 `os.getpriority` 确认为 **15**。没有安装系统级软件。

| 阶段/产物 | 本次实测 |
|---|---:|
| 首次同步（含 CHECKSUM、月档解码和日档请求） | 51.065 秒 |
| 整理写 Parquet / 发布版本 | 0.455 秒 |
| 全量独立校验 | 2.390 秒 |
| 独立校验 + 小时库读取 + 对账 | 2.996 秒 |
| 原始 ZIP | 27,524,683 B（27.52 MB） |
| Parquet | 24,938,252 B（24.94 MB） |
| 两者合计 | 52,462,935 B（52.46 MB） |

以上保留首次实际下载的性能结果。缓存续跑只记录续传耗时，验收脚本不会用缓存耗时外推下载；`--skip-sync` 若读到缓存同步报告，下载外推为 null。

从 2020-01-01 到 2026-11-15 UTC（右端不含）八条序列共有 28,915,200 个日历分钟格。用两个完整月的压缩率外推体量、首次同步的字节吞吐外推下载、全样本行数外推整理/校验：

| 全量估计 | 中心估计 |
|---|---:|
| 原始 ZIP | 1.068 GB |
| Parquet | 0.964 GB |
| 合计 | **2.032 GB（1.89 GiB）**，另加少量清单/元数据 |
| 同步 | **约 33.0 分钟** |
| 整理 | 约 17.6 秒 |
| 独立校验 | 约 92.3 秒 |
| 同步 + 整理 + 校验 | 约 34.9 分钟 |

这些是同机器、同带宽、串行访问、没有重试/限流的线性外推，不是全量实测或时限承诺。下载时间受请求延迟、带宽、后期交易量与压缩率影响最大，可按约 20–60 分钟安排首轮操作；若仍用单命令 840 秒上限，分月/年执行或断点续跑。初次需要约 2.1 GB，保留修订和临时空间时建议预留 3 GB 以上。外置盘写速尚未实测，最终全量由审查人合入后执行。

## 新增文件清单

- `bench/crypto_minute_sample.py`
- `bench/crypto_minute_source_audit.py`
- `bench/mutations_crypto_minute.py`
- `bench/results/crypto_minute_bench_red.txt`
- `bench/results/crypto_minute_crypto_tests.txt`
- `bench/results/crypto_minute_request_red.txt`
- `bench/results/crypto_minute_resume.json`
- `bench/results/crypto_minute_resume_red.txt`
- `bench/results/crypto_minute_sample.json`
- `bench/results/crypto_minute_source_audit.json`
- `bench/results/crypto_minute_tests.txt`
- `bench/results/mutations_crypto_minute.json`
- `docs/C7_CRYPTO_DATA.patch`
- `docs/C7_NOTES.md`
- `docs/C7_WORKLOG.md`
- `docs/CRYPTO_MINUTE.md`
- `src/ashare/data/crypto/minute.py`
- `src/ashare/data/crypto/verify_minute.py`
- `test/crypto/test_minute.py`
- `test/crypto/test_minute_bench.py`

## 审查修正（2026-11-15，审查人）

- 独立审查 P2-1：`reconcile_hourly` 在没有任何小时被比较时也报 `matches=True`；改为要求 `compared_hours > 0`，回归测试 `test_review_empty_reconciliation_is_not_a_match`（先在旧代码上失败）。
- 独立审查 P2-2：`bars` 的终点不含没有测试守住；补 `test_review_bars_end_is_exclusive_within_and_across_months`，把 `< end` 变异为 `<= end` 后该测试失败。
- 文档补丁已并入 `docs/CRYPTO_DATA.md`；交接用的补丁和工作日志不入库。
