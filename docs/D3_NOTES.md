# D3 筹码与 ETF 专题接入笔记

2026-11-16；基线 `2a14e70`，工作分支 `agent/codex/chips`。没有提交、推送、全量下载或正式面板重建。

## 结论与停止边界

**继续不建立 `chip_*` 面板字段。2026-11-16 全量复核已否定 `his_low/his_high` 是历史极值，
原验收判据作废；目前最支持前复权收盘候选分母，但固定基准、特殊证券尺度和跨快照不变性仍未证实。**
下文早期样本及第一次审查保留为工作记录；最新依据见文末「全量独立复核与判据修订」，
当前程序将极值比对标为「诊断（不判定）」，不计入正式失败数；复权基准固定为「未证实」，
列明所缺发布证据。历史记录中的 pass/fail 已撤销其判定含义，详见文末合并前修复记录。
登记了 `cyq_perf` 原始数据集、同步计划和独立核对；没有改 `FIELDS`、股票面板构建、
`platform/language.py` 的字段表。因此也没有可报告的 language.py 修改行号。

ETF 专题独立完成：`etf_basic` 的 L/D/P 三次请求合并为一个 `all` 分区，
`etf_share_size` 按交易日分区。基金面板增加 `total_share`（份）、`total_size`（元），
另存 `table_etf_basic.parquet`；原 `total_shares` 来自 fund_share，保持原值。

## 筹码真实核对

原始响应、逐行结果全部在忽略的 `tmp/d3/`，不写主库。可复跑程序为
`bench/d3_real_check.py`，汇总为 `bench/results/d3_real_check.json`。
采用账号环境变量、最多 4 个并发；初次探针的全局间隔至少 0.21 秒，复跑客户端为每接口
250 次/分钟、预算 700 次。分页满 6000 行继续，直到不足一页；没有截断长历史。

- 固定随机种子 31415，从 2007-01-04 至 2017 年底上市的当前股票中抽 210 只，
  再并入 2026-11-10、11、12 三天实施除权的 27 只，去重共 235 只。
- 样本日：2018-01-02/03/04；2021-01-04/05/06；2024-01-02/03/04；2026-11-10/11/12。
- 每年实际可比股票都超过 200 只，共 2,679 个证券日；日线按上市起点请求。
  **审查修正：旧实现只检查首日及现有报价的因子连接，不能证明中间/尾部完整，原“缺失 0”完整性结论撤回。**
  下表数值仍是当次已有行情上的误差诊断，不是通过逐日完整性证明后的统计。
  样本是当前股票加近期除权股票，有幸存者偏差，仅作接口核对，不用于因子研究。
- 每条使用该响应 `_fetched_at` 的上海日期，选不晚于该日的最新复权因子 `adj_F`。
  截至被核对交易日的全部日线，用 `low/high × adj_s / adj_F` 计算上市以来极值。
  缺上市历史、缺日线对应因子、无效抓取时间一律不能通过；本次修复另加逐日日历与停牌核对，见审查修复记录。
- 半个 0.1 元价位（0.05 元）是初始报告阈值；另列 0.10 元和 1% 的诊断，**没有以放宽阈值通过验收**。

| 年份 | 股票 | 证券日 | 低高两端误差均≤0.05元 | 均≤0.10元 | 低高隐含基准相差≤1% | 低价绝对误差中位数 | 高价绝对误差中位数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2018 | 202 | 606 | 54 | 208 | 123 | 0.085763 | 0.151637 |
| 2021 | 224 | 672 | 72 | 212 | 128 | 0.090000 | 0.172042 |
| 2024 | 233 | 699 | 60 | 189 | 117 | 0.092632 | 0.169179 |
| 2026 | 234 | 702 | 13 | 56 | 26 | 1.145813 | 0.591868 |
| 合计 | 235（并集） | 2679 | 199（7.43%） | 665（24.82%） | 394（14.71%） | — | — |

“隐含基准”比较 `(his_low / 自算低价) / (his_high / 自算高价)`。若只有抓取基准日不同，
同一证券日的两端应有近似相同的乘数；2,285 条连 1% 的一致性也不满足，不能只换一个基准日。

### 差异模式

以下是核对结论的少量数值例子，不保存整份行情响应：

| 股票 / 日 | 源低价 / 自算低价 | 源高价 / 自算高价 | 含义 |
|---|---|---|---|
| 000338.SZ / 20180102 | 0.8 / 0.892981 | 36.0 / 9.091432 | 两端乘数分别约 0.896、3.960；改基准日不能同时解释 |
| 002158.SZ / 20180102 | 1.2 / 1.376314 | 36.3 / 19.484315 | 高价差 16.82 元，远大于 0.1 元网格 |
| 002139.SZ / 20261112 | 3.0 / 0.764882 | 22.8 / 22.711073 | 高价接近，但低价明显不是上市以来最低价 |

2026 年的 702 条中，407 条源低价比上市以来最低价高超过 0.5 元，317 条源高价低于
上市以来最高价超过 1 元。全样本最大低价、高价绝对误差分别约 29.31、365.69 元。
把历史交易日的极值窗口扩到抓取日也不能统一解释：只有 782/2679 条高价落在 0.1 元内。

成本价确有明显网格离散，但这不足以解释上述大偏差。部分历史行的高价接近后来才出现的
价位、近期低价又丢掉早年低点，提示源可能使用不同计算窗口、分批生成范围或不同复权算法；
**这些只是待证假设**。本次没有证明源使用哪一种，也不能仅凭两个极值反推出成本价的正确换算。
官方[筹码接口说明](https://tushare.pro/document/2?doc_id=293)没有给出足以复原这些差异的生成算法。

**最终采用的换算：无。** `close_t × adj_t / adj_F` 仅作为核对候选，不作为已验证的筹码分母。
原始成本保持源值并注明基准未核实；胜率原始单位 `%`，核对时除以 100 检查 [0,1]。
不能声称已经证明不同真实抓取基准下的成本比值一致。

## 已落地代码与回归

- `catalog.py`：股票组登记 `cyq_perf`，按交易日、20180102 起、收盘后、成本元/胜率%；
  ETF 两表只进入 FUND_DATASETS，覆盖起点与实测单位见 FUND_CB_DATA。
- `chip_verify.py` / `verify.py`：原始胜率范围、成本正值、分位单调、覆盖起点与逐字段非空覆盖；
  `verify` 每年首三个已有分区、最多 200 只股票做基准核对，并报告缺历史，失败不被自动豁免。
- 原有 Store 的内容哈希不含抓取时间。重复抓取值不变时只更新清单，文件及文件内 `_fetched_at`
  保持原样；内容变化才切换修订。测试直接检查路径、字节、mtime、版本与文件抓取时间。
- `fund_sync.py`：ETF 主表三种状态全部成功后才写一个分区；中途失败保留旧完整修订。
- `fund_schema.py` / `fund_panel.py`：新增两个基金字段与 ETF 主表；输入版本从名字前缀判断改为
  FUND_DATASETS 成员判断，保证 ETF 变化触发新基金代次，股票、期货原始版本仍隔离。
- `fund_verify.py`：主表/份额证券日覆盖、跨源份额一致率（仅报告）、主键与分区完整性、
  面板单位逐格复算、次日可知元数据、主表内容一致性。
- `test/data/test_fund_fixtures.py` 仅增加 ETF 合成表，已有合成基金、债券数值未改。
- 开发发现：核对脚本最初把 Polars `is_between` 的字符串日期当列名；已改显式 `pl.lit`，
  `test_real_probe_date_literals_regression` 覆盖边界。新的核对与 ETF 回归先在旧代码运行，
  因模块/入口尚不存在出现 2 个收集错误，再实现；未放宽旧断言或新增跳过。

## 测试、变异与未决项

独立 venv 由 `make dev` 创建；CARGO_HOME、UV_PYTHON_INSTALL_DIR、MPLCONFIGDIR 指向本 worktree。
重命令均通过 `nice -n 15` 启动，pytest 并发≤2、cargo jobs=2。当前沙箱的 nice 会报告
`setpriority: Operation not permitted`，命令仍执行；没有尝试提权或修改共享机器设置。

`bench/chips_legacy.py` 使用修改前完整 src 副本和当前代码，分别构建两份相同合成原始库：
股票 48 个原字段、基金 16 个原字段、可转债 22 个原字段的 dtype、形状、字节 SHA-256 全部相同，日期和证券轴相同。
结果 `bench/results/chips_legacy.json`；这项检查不读取真实面板。

完整测试与变异结果在本节收尾时追加。`bench/mutations_chips.py` 只改 tmp 源码副本，
先验证变异 0 导入当前 worktree，之后每例再核对副本导入位置；收集错误不能算杀死变异。

按硬门槛保留的未决项：

1. 确定 cyq_perf 的真实计算窗口、复权算法与基准，解释上述异常；目前不能发布成本比值。
2. chip_* 字段、平台解析求值、不同抓取基准快照的比值不变性以及对应比值分母变异，全部延后。
3. ETF 次日可知是至少一天的滞后约定；跨境品种更新可能更晚，无法还原逐笔首次发布时间。
4. 全量同步、正式面板重建由用户合并后执行。

## 工作日志

- [02:22] AShare — 完成 D3 真实筹码基准核对：235 只、2679 条，核对失败，停止发布筹码字段；ETF 两表单位与覆盖起点完成实测。
- [02:22] AShare — ETF 专题接入和独立校验完成；合成旧新面板股票 48、基金 16 个原字段逐位一致，产物见 bench/results/。

任务明确只改 worktree，且沙箱不允许写 `~/worklog`，所以本次日志留在此处，没有改动项目外文件。

补充回归的首轮变异 0 抓到测试自身的错误前提：夹具中 159001.SZ 上市前 `total_shares`
本来就是 NaN，不能断言全部旧份额都是有限值。现改为保存改动前整张旧字段并逐字节比较，
再核对已上市且有份额的指定证券；保留“新增专题份额缺失不能前填”的断言。这不是放宽数据断言。

### 最终检查记录

- `test/data` + `test/platform`，`-n 2`：**1550 passed，7 skipped，2 warnings，552.12 秒**。
  7 个跳过都是原有 Databento 可选依赖未安装；2 条警告为 FastAPI/Starlette 的 TestClient 弃用提示。
  没有新增跳过。完整集合之后补加的边界、缺失值和报告集成回归，由下列定向集合覆盖。
- 当前 D3 定向集合（筹码核对 + ETF）最终重跑：**23 passed，20.89 秒**；变异 0 确认导入当前 worktree。
- `make check` 的静态部分全部通过：`ruff check .`、`ruff format --check .`、
  `cargo fmt --all --check`、`cargo clippy --workspace --all-targets -j 2 -- -D warnings`。
  本次没有额外运行全项目 pytest 或 cargo test；数据与平台测试已经单独完整运行。
- 定向变异 **9/9 killed**：抓取基准错误、极值分母错误、胜率漏 /100、覆盖起点错误、
  取消分位单调检查、ETF 份额漏 ×10000、规模漏 ×10000、次日可知错误、ETF 内容遗漏版本依赖。
  成本/收盘比值分母及不同真实基准快照不变性的变异明确为 deferred，没有假称通过。
- 逐位对照增加可转债：股票 **48**、基金 **16**、可转债 **22** 个原有字段全部相同。
- 汇总：`bench/results/d3_validation.json`；变异明细：`bench/results/mutations_chips.json`。

改动文件清单：

- `src/ashare/data/catalog.py`、`chip_verify.py`、`verify.py`、`fund_sync.py`、`fund_schema.py`、`fund_panel.py`、`fund_verify.py`。
- `test/data/test_chips.py`、`test_etf_data.py`、`test_fund_fixtures.py`。
- `bench/d3_real_check.py`、`chips_legacy.py`、`mutations_chips.py`。
- `bench/results/d3_real_check.json`、`chips_legacy.json`、`mutations_chips.json`、`d3_validation.json`。
- `docs/D3_NOTES.md`、`DATA_NOTES.md`、`FUND_CB_DATA.md`。

- [02:27] AShare — D3 收尾：数据与平台 1550 通过、7 个既有可选依赖跳过，静态检查通过；9 个已实现路径变异全部捕获，筹码字段因真实基准核对失败保持未发布。

## 独立审查修复（2026-11-16）

审查原文只读：`.review/review_d3.md`。本轮不联网下载筹码，
不访问用户正在下载的 staging/cyq；等用户通知完成后，才用全量数据只读重跑按年、按股票的误差分析。
独立审查与原诊断都不能解释筹码基准，继续不发布 `chip_*`。

### 先复现，再修复

修改业务代码前运行新增复现：**10 failed、23 deselected（15.63 秒）**，日志 `tmp/d3/review_red.log`。
四个同分区案例在旧代码上未抛异常（两表 × 空/骤降），两个分状态案例未抛异常，两个空面板案例
错误返回 pass，两个行情缺日案例错误返回 history_complete。没有把收集错误算成复现成功。

- `Store.guard_partition_refresh` 复用期货同分区刷新规则到两张 ETF 表：非空完整修订遇空响应，
  或旧行数达到门槛且新行数低于 65%，告警、抛错、保留旧内容；确认状态不能豁免。
- `fund_sync` 在 L/D/P 合并前分别检查。ETF 主表 `drop_check_min_rows=1`，D/P 少于 100 行时
  也保护；份额日分区沿用 100 行门槛。P 的 20→0/5 即使被 L 增长抵消，仍拒绝发布。
- `check_etf_panel` 拒绝空原始专题表，拒绝整列全 NaN/无有限值；NaN 相等只适用于局部缺值。
- `basis_rows` 按上市区间枚举日历中的开市日；日历本身须覆盖所有自然日且无冲突，缺行情必须
  有同股同日全日停牌证据（沿用面板 S + 空/全天定义）。缺内部日或尾部日、临时停牌、未知日历、
  未知上市区间、缺因子均不能给完整结论。输出 `history_note`、缺日数量/例子；
  `audit_chip_basis` 返回 `insufficient_history`，明确“不足以核实”，并计入失败。
- 小量核对脚本新增 `--history-root` 只读日历/停牌证据；没有证据时不能产生完整/通过结论。
  本轮未运行真实核对脚本；旧 `d3_real_check.json` 是历史产物，其 history_incomplete=0 不再作为证据。
  该汇总新增 `history_completeness_notice` 明示这一限制，原有误差数字保持原样。

复现来源写在对应测试 docstring；正例覆盖完整历史、分区抓取时间、全日停牌，反例覆盖日历缺日/冲突、
缺上市区间、临时停牌/复牌、中间普通日/极值日、尾部截断、ETF 空响应/缩水、空面板。
最终测试与变异结果在本轮验证完成后追加。

实现检查另发现上市前报价会污染极值：`test_basis_excludes_prices_before_listing_interval`
先确认失败（1 failed，`tmp/d3/listing_red.log`），再把极值输入限制在上市起点以后；配套独立变异覆盖。

### 本轮回归与变异

新增 **32** 个用例（计入参数化），D3 定向集合共 **55** 项。主要回归位置：

| 文件 | 回归测试 | 用例数 |
|---|---|---:|
| test_etf_data.py | test_successful_empty_or_collapsed_refresh_keeps_revision | 4 |
| test_etf_data.py | test_successful_status_shrink_keeps_complete_table | 2 |
| test_etf_data.py | test_verify_rejects_empty_source_and_empty_panel | 2 |
| test_etf_data.py | test_verify_rejects_all_nan_fields | 1 |
| test_etf_data.py | test_successful_empty_share_sync_warns_and_keeps_revision | 1 |
| test_chips.py | test_basis_rejects_internal_gap_and_truncated_tail | 4 |
| test_chips.py | test_basis_requires_complete_calendar / test_basis_calendar_conflict_is_insufficient | 3 + 1 |
| test_chips.py | test_basis_only_full_day_suspension_explains_gap | 5 |
| test_chips.py | test_basis_requires_known_listing_interval | 4 |
| test_chips.py | test_audit_reports_missing_evidence_as_insufficient | 4 |
| test_chips.py | test_basis_excludes_prices_before_listing_interval | 1 |

`bench/mutations_chips.py` 本轮先跑变异 0：**55 passed，74.92 秒**，核实导入
`src/ashare/data/chip_verify.py` 来自本 worktree。随后 **18/18 killed**（原 9 项 + 新 9 项），
新增覆盖空响应、同分区保护入口、缩水、分状态缩水、空原始表、全 NaN 面板、历史缺日、
日历连续性、上市区间。逐例断言失败明细在 `bench/results/mutations_chips.json`，
日志在 `tmp/d3/mutations/`；没有收集错误冒充杀死变异。未发布筹码比值字段的两项变异仍为 deferred。

`make check` 静态部分四项全部通过：Ruff 检查、Ruff 格式（833 文件）、Cargo fmt、
Clippy（workspace、all-targets、`-j 2`、`-D warnings`）。本轮没有更改平台字段表。

- [11:38] AShare — 修复 D3 审查 P0/P1；10 项旧代码复现失败后完成保护，新增 32 项回归；变异 0 的 55 项通过，18/18 变异捕获，静态检查通过。完整数据/平台测试继续执行。

完整集合 `test/data` + `test/platform`（`-n 2`，临时目录位于 worktree/tmp）最终：
**1586 passed、7 skipped、2 warnings，852.36 秒**。7 个跳过均是原有 Databento 可选依赖未安装，
2 条警告均为原有 Starlette TestClient 弃用提示；没有新增跳过、放宽断言或未通过的测试。
命令在 14 分 30 秒上限前完成，没有中断或重跑完整集合。日志 `tmp/d3/review_full_tests.log`，
结构化汇总 `bench/results/d3_validation.json`。`git diff --check` 通过。

本轮修复涉及 `store.py`、`catalog.py`、`fund_sync.py`、`fund_verify.py`、`chip_verify.py`；
回归在 `test_chips.py`、`test_etf_data.py`；核对/变异脚本在 `d3_real_check.py`、`mutations_chips.py`；
更新本笔记、`DATA_NOTES.md`、`FUND_CB_DATA.md` 及三份 D3 验证 JSON。没有改平台字段表或筹码面板，
没有提交/推送，没有访问用户的筹码暂存目录。全量核对仍等用户通知下载完成后进行。

- [11:53] AShare — D3 审查修复完成验证：数据/平台 1586 通过，7 个既有依赖跳过；55 项变异 0 通过、18/18 变异捕获、四项静态检查通过。旧真实样本完整性结论已撤回并标注，筹码字段继续不发布。

## 全量独立复核与判据修订（2026-11-16）

以下数字整理自只读独立审查 `.review/recheck_d3.md`，不是本轮重新计算的结果。
本轮没有联网，没有读取或修改 `data/staging/cyq`，
没有改主库。DESIGN 只修改 6.1.5；**`his_low/his_high` 历史极值判据正式作废**，
不将旧程序极值不匹配等同于成本分母被否定，也不将时间连续性等同于固定基准已证实。

### 核对范围及按年误差

全量 9,786,688 行、2,150 个交易日，原分区和合并文件逐日行数及两组内容摘要一致，重复主键 0。
成功映射 5,633 个代码，有效价格/因子可比 9,605,105 行；未映射北交所旧代码 248 个、181,286 行，
另 297 行缺有效价格或因子，均排除在误差分母之外。按日历、上市区间和停牌证据证明历史完整的
可比记录为 7,005,201 行，其余仅作已有历史诊断。主库面板对原始分区核对 46,693,566 个有限价格值、
16,062,483 个有限因子值，差异均为 0。

以下 F 是主库截至 2026-11-13 最新有效因子，源抓取于 11 月 16 日；**本地 F 不是已证实的源基准**。
误差为源边界与截至当日 `low/high × adj_s / adj_F` 历史极值的绝对差（元）；两端匹配指均 ≤0.05 元，
仅保留作反证诊断，不再作为验收标准。

| 年份 | 可比行数 | F 误差 P50 低/高 | F 误差 P90 低/高 | 两端匹配 |
|---|---:|---:|---:|---:|
| 2018 | 765,116 | .199/.160 | 1.817/20.516 | 4.958% |
| 2019 | 828,222 | .188/.170 | 1.718/20.890 | 4.819% |
| 2020 | 892,951 | .191/.181 | 1.857/19.776 | 4.679% |
| 2021 | 1,011,690 | .200/.188 | 2.327/15.099 | 4.496% |
| 2022 | 1,129,226 | .200/.199 | 2.245/13.767 | 4.590% |
| 2023 | 1,232,366 | .197/.208 | 2.196/15.535 | 4.663% |
| 2024 | 1,285,713 | .170/.218 | 1.451/14.956 | 4.632% |
| 2025 | 1,313,256 | .179/.218 | 1.494/11.835 | 4.363% |
| 2026 | 1,146,565 | .381/.317 | 6.461/16.988 | 2.673% |

全体 F 两端匹配 4.393%（历史完整子集也仅 5.730%）；不复权 0.141%、本地首日因子 0.110%、
交易日因子 1.471%。F 的全体 P50 为 .200/.207、P90 为 2.125/16.154 元。
将窗口延伸到最新日也仅 2.700%（含未来行情，只作诊断），收盘极值 0.062%；
250/500/750/1000/1250/1500/2000 个有效报价滚动窗口分别为
0.534/0.846/1.245/1.600/1.957/2.303/2.985%，没有统一窗口。
各板块及近 1/5/20/60 日没有除权的样本也普遍不符，不能归因于少数除权日错位。

### 按股票的边界与尺度结构

下表为「源高端 − 当时 F 基准历史最高价」的年度中位数（元），保留符号。

| 股票 | 观察数 | 低/高边界取值数 | 2018 高端误差 | 2026 高端误差 | 模式 |
|---|---:|---:|---:|---:|---|
| 603444.SH | 2,150 | 1/1 | +365.687 | +7.868 | 八年多边界固定 |
| 601888.SH | 2,150 | 1/2 | +309.211 | +1.397 | 早年高端超过当时极值 |
| 603396.SH | 2,150 | 1/1 | +151.071 | −.711 | 固定高端后来低于历史最高 |
| 300295.SZ | 2,139 | 4/3 | +.309 | −87.791 | 2026-05-21 收缩 |
| 300659.SZ | 2,076 | 20/31 | −.078 | −54.958 | 同日收缩 |
| 002642.SZ | 2,143 | 8/4 | +.282 | −53.418 | 同日收缩 |
| 600519.SH | 2,150 | 1/2 | +1,626.075 | +36.594 | 2026-06-29 高端下降 |

2026-05-21 全市场有 **1,405 只股票边界收缩**，不符合各股独立窗口自然到期的解释。
断点后 120 条记录，300659 的 250 日窗口两端差 ≤0.10 元有 117/120，
300295、002642 同窗口均 0/120，120–2000 日窗口也不能统一解释七只股票。
600519 上市早于主库起点，尤其不能核实完整低端。源边界更像算法价格范围或网格边界，
但尚未证明是当日非零筹码分布的支撑边界。

`R_t = cost_t / (close_t × adj_t / adj_F)` 的相邻交易日 `100 × |Δln R|`：

| 样本 | 对数 | 中位成本 P50/P90 | 加权成本 P50/P90 | 中位成本/不复权收盘 P50/P90 |
|---|---:|---:|---:|---:|
| 因子变化 ≤1bp | 9,560,671 | 1.455/4.764 | 1.396/4.461 | 1.455/4.764 |
| 因子变化 >1bp | 34,497 | 1.419/4.570 | 1.357/4.313 | 2.327/23.259 |
| 因子增加 >5% | 4,910 | 1.794/5.448 | 1.715/5.323 | 31.994/49.412 |

大除权回归斜率均约 −.0026，年度比值中位数：中位成本 1.0035–1.0404，加权成本 1.0120–1.0740。
另 137 次跨停牌有效观察的大因子变化，两个比值变动 P50 为 4.875、5.008，不能由连续交易日样本替代。
这些支持前复权方向；但 `Δln R = Δln cost − Δln(close × adj_t)`，固定 `adj_F` 完全抵消，
**时间连续性识别不了整段常数倍偏差，不能确定源基准日**。

这种偏差确有证据：胜率在 49.5%–50.5% 时，600165.SH 的 21 条记录成本/候选收盘中位数
1.3406、绝对价差中位数 .9194 元；000908.SZ 的 19 条为 1.4452、1.1016 元。
有效胜率记录排除价差 ≤.10 元后，**19,046 条胜率与中位成本方向矛盾**，其中 10,585 条价差 >收盘的 5%。
不能全归因于普通取整，须区分成本基准、胜率算法和特殊公司行为。

此外胜率 >100 有 **7,286 条、2,463 只股票**，范围 100.01%–100.61%；
非正成本 **138 条、10 只股票**；分位倒序与成本越出源边界均 0 条。
因此连 `chip_winner` 也继续不发布，先明确异常隔离规则，不能静默截断到 [0,1]。

### 发布前仍缺的证据

`close × adj_t / adj_F` 是目前最支持的候选分母，尚未最终采用。
须取得正常和异常证券少量 `cyq_chips` 分布、查明特殊公司行为口径，并取得
**跨实际除权的同一证券日两份抓取快照**，证明按各自抓取基准换算后的比值不变。
旧样本与全量的 54,079 条重合记录、9 个数值字段完全一致，但没有覆盖不同基准，不能完成这项证明。
北交所旧代码映射、主库以前的完整历史也仍未解决。此轮按用户要求只整理既有证据，不请求新数据。

## 第二次审查 P1 修复（首次空日和逐日有效覆盖）

- `store.py`：覆盖期内首次空响应、旧零行 → 零行记 `suspect / empty_unverified`，告警「待核实」；
  同哈希和旧人工确认不能绕过。`complete()` 对旧版零行也重新检查上市证据，`refresh=False` 仍可补抓；
  批次重核随主表更正撤回空分区完成状态。
- 新增 `etf_coverage.py`：用 ETF 主表上市日期及基金主表退市日期判断应在市集合。
  只有明确区间证明当天无在市 ETF，空日才是 `complete / no_listed_etf`；接口覆盖前不要求数据。
  主表缺失、未知/非法日期、只有当前 D 状态均不足以证明历史缺席，不能豁免。
  `fund_sync.py` 将 ETF 主表放在日分区前完成，避免正常同步依赖完成顺序。
- `check_partitions` 检查有效日期内每个分区完成、非零行、两字段各有在市 ETF 有限值；
  窗口取基金行情范围，不能用份额表自身首尾缩窗掩盖缺日，也不检查日历中的未来整年。
  `check_etf_panel` 同时逐日检查原表和面板，不再以全表任意一个有限值代替覆盖。
  两项检查都由上市区间判断合法空日，不走人工例外清单。
- **取消份额表旧修订不足 100 行的缩水豁免**：`drop_check_min_rows=1`。
  早期小分区也代表同一交易日完整集合，99→1 没有合理豁免依据。保留严格 65% 阈值，
  只比较同一日不同修订，不比较相邻交易日；期货的原阈值不变。

旧代码先验证失败：新增初版 16 项为 **15 failed / 1 passed**（`tmp/d3/recheck_red.log`）；
随后把两个 verify 分开测，未改 verify 时独立 **12 failed**（`tmp/d3/recheck_verify_red.log`）。
定向运行又发现逐日错误明细会挤掉「整列全空」诊断，保留原断言，改为先列汇总再列逐日明细。
完整测试、静态检查和变异结果见后续验证记录及 `bench/results/d3_validation.json`。

- [13:22] AShare — 修复第二次审查 P1：ETF 空日待核实、旧零行补抓、逐日有效覆盖与小分区缩水保护；新增 28 项回归，变异 0 的 83 项通过，四项静态检查通过。已整理独立全量筹码证据，撤销历史极值判据，继续不发布字段；完整回归和变异运行中。

### 第二次审查的回归和变异结果

新增测试均在 `test/data/test_etf_coverage.py`，共 **28** 项（含参数化），每项注明复核或实现检查来源：

| 测试名 | 用例数 | 目的 |
|---|---:|---|
| test_empty_etf_partition_remains_unverified | 3 | 首次空、旧零行、旧确认零行保持待核实 |
| test_small_etf_snapshot_shrink_is_rejected | 1 | 99→1 同日刷新拒绝 |
| test_empty_etf_partition_is_retried_without_refresh | 2 | 关闭刷新后补抓新旧零行 |
| test_verify_rejects_each_empty_etf_day | 8 | 两个 verify 独立检查首尾空日/未建分区，不走豁免 |
| test_verify_rejects_each_nan_etf_day | 4 | 两个 verify 独立检查单字段整行 NaN |
| test_proven_no_listed_etf_allows_empty_days | 2 | 上市前/明确退市后合法空日正例 |
| test_current_delisted_status_is_not_historical_absence | 1 | 当前 D 状态不足以解释历史空日 |
| test_etf_coverage_positive | 1 | 正常逐日覆盖通过 |
| test_empty_etf_revalidated_when_listing_evidence_changes | 1 | 主表更正后撤回旧完成状态，不采信人工确认 |
| test_unknown_listing_date_does_not_prove_absence | 3 | 未知、非法日期不足以豁免 |
| test_etf_source_coverage_start_is_respected | 1 | 接口覆盖前不要求数据 |
| test_etf_partition_window_uses_fund_daily | 1 | 不缩掉首尾缺日，也不检查未来日历 |

`bench/mutations_chips.py` 先核实导入本 worktree，再跑变异 0：**83 passed，101.90 秒**。
随后 **25/25 killed**，全部由测试断言捕获，无收集错误或无效变异。新增 7 项覆盖首次空日待核实、
旧零行补抓、小分区缩水、逐日分区覆盖、逐日原表有限值、逐日面板有限值、未知上市证据不能豁免。
成本字段尚未发布，对应分母与真实跨基准快照不变性两项仍 deferred。
结果保存在 `bench/results/mutations_chips.json`，逐例日志在 `tmp/d3/mutations/`。
变异只操作 worktree/tmp 内源码副本，子进程有 600 秒上限，逐例顺序执行。

静态检查全部通过：`ruff check .`、`ruff format --check .`（835 文件）、
`cargo fmt --all --check`、`cargo clippy --workspace --all-targets -j 2 -- -D warnings`。
完整数据与平台测试拆为两组顺序运行，均 `-n 2`、14 分 30 秒上限，测试临时文件在 worktree/tmp。

`test/data`：**1027 passed、7 skipped，772.05 秒**。7 项均为既有 Databento 可选依赖未安装，
没有新增跳过或放宽断言；日志 `tmp/d3/recheck_data.log`。

`test/platform`：**587 passed、2 warnings，188.52 秒**；警告是既有 Starlette TestClient 弃用提示。
两组共 **1614 passed、7 skipped、2 warnings**，顺序运行累计 960.57 秒，单组均在上限内。
平台日志 `tmp/d3/recheck_platform.log`；结构化汇总 `bench/results/d3_validation.json`。

本轮修复位置：`etf_coverage.py`、`store.py`、`catalog.py`、`fund_sync.py`、`fund_verify.py`；
回归 `test_etf_coverage.py`；变异脚本及结果更新。文档更新 `D3_NOTES.md`、DESIGN 6.1.5、
`DATA_NOTES.md`、`FUND_CB_DATA.md`。未修改平台字段表、筹码面板或正式数据；未提交/推送。

- [13:53] AShare — 第二次 D3 审查修复完成：新增 28 项回归，数据/平台 1614 通过、7 个既有依赖跳过；变异 0 的 83 项通过、25/25 变异捕获、四项静态检查通过。全量复核结论已入笔记和 DESIGN 6.1.5，筹码字段继续不发布。


## 合并前复核修复（recheck2_d3.md，2026-11-16）

来源为只读 `.review/recheck2_d3.md`。本轮仅用合成夹具，不联网、不读取或改写
`data/staging/cyq`，继续不发布 `chip_*`。OneContext 本次因数据库打不开未能检索，
以审查原文和本 worktree 的代码为依据。

### P1：极值诊断与复权基准判定彻底分开

- `chip_verify.py::audit_chip_basis` 固定返回 `unverified / 未证实`，检查数、失败数均为 0，
  通过率为空。列明三项缺证：少量正常/异常证券的 `cyq_chips` 分布、特殊公司行为成本尺度口径、
  跨实际除权的同一证券日两份抓取快照。极值恰好相符也不能证实基准。
- 极值结果独立为 `diagnostic / 诊断（不判定）`，用 `within_tolerance`、`outside_tolerance`
  保留数量，保留逐行观察值、候选值、抓取日因子及历史完整性；没有 `failed` 或 `pass_rate`。
  原行级 `passed` 更名为 `within_tolerance`，避免继续使用验收语义。
- 缺数据、缺样本或缺历史通过 `data_status` 单列；日历、上市区间和全日停牌证据仍逐日检查。
  历史不完整的行既不计为容差内，也不计为容差外，单独列 `history_incomplete` 和缺日原因。
- `verify.py` 不再把极值结果送进 `Audit`。JSON 顶层分别给出 `chip_adjustment_basis` 与
  `chip_extrema_diagnostic`；Markdown 分别显示状态和缺证说明。胜率、正成本、分位单调、
  非空与覆盖等原始检查继续参加正式失败计数。
- `bench/d3_real_check.py` 同步取消极值的 pass/fail 汇总，使用诊断状态与容差计数。
  `bench/results/d3_real_check.json` 的早期 2,679 条数值没有重算或改动，仅改为诊断标签，
  并明确旧历史完整性计数未重新验证，不能据此发布字段。

修复前先运行新增回归：**9 failed、36 deselected，10.45 秒**，日志
`tmp/d3/recheck2_red.log`。仅将 `his_high` 从 8 改成 28，旧基准由 pass 变 fail；
落盘 `verify` 夹具只变边界时，正式失败总数从 **11 变成 12**（其中 11 是夹具的其他检查）。
新断言要求完整 `checks` 和基准报告逐项相同，并同时证明诊断数字仍随边界改变。
本轮检查还发现 `Infinity` 不能安全序列化为标准 JSON，先复现再修复为诊断中的空值。

既有测试仅调整已作废的判定语义；逐日完整性、缺日数量、极值计算、抓取时间因子、
覆盖和原始值范围断言保留。没有跳过或放宽任何有效检查。

### P2：文档位置可移植

两处本机绝对路径改为 `.review/review_d3.md`、`data/staging/cyq`，并加回归防止重新引入。
本轮没有修改 DESIGN（6.1.5 已明确撤销旧判据）、平台字段表或面板字段。

### 新增回归与变异范围

新增测试位于 `test/data/test_chips.py`，共 **9** 项（含参数化），每项注明来源：

| 测试名 | 用例数 | 目的 |
|---|---:|---|
| test_chip_boundaries_are_diagnostic_only | 2 | 匹配/不匹配都未证实，诊断保留数字及缺证清单 |
| test_chip_basis_stays_unverified_without_samples | 2 | 没数据/没样本不能形成通过或失败 |
| test_chip_diagnostic_nonfinite_boundaries_are_json_safe | 3 | null/NaN/Infinity 边界可安全报告 |
| test_verify_boundary_changes_do_not_change_failures | 1 | 改边界不影响正式失败数和基准报告，Markdown 显示缺证 |
| test_d3_notes_use_repository_relative_paths | 1 | 笔记无本机绝对路径 |

变异脚本增加四项：把基准恢复为极值 pass/fail、把诊断恢复为 pass/fail、把极值重新计入
正式失败数、去掉无穷值的 JSON 清理。沿用其余 25 项，总计 29 项；先核实 worktree 导入位置，
跑未改动的变异 0，再顺序执行源码副本变异。成本/收盘字段和实际跨除权快照不变性仍 deferred。

- [15:19] AShare — 合并前 P1/P2 修复已落地；9 项新增回归先在旧代码失败，极值仅作诊断、基准固定未证实，路径及非有限值报告同步修复；正在运行变异及完整检查。


### 合并前验证结果

变异 0：**92 passed，50.89 秒**，核实导入的是本 worktree 的 `src/ashare/data/chip_verify.py`。
随后 **29/29 killed**，四项新增变异分别被状态断言、正式失败数 11→12 的断言、
标准 JSON 序列化断言捕获；全部无收集错误或无效变异。
结果 `bench/results/mutations_chips.json`，运行日志 `tmp/d3/recheck2_mutations.log`，
逐例断言日志 `tmp/d3/mutations/`。

`make check` 的四项静态检查全部通过：Ruff 检查、格式检查（835 文件）、
`cargo fmt --all --check`、`cargo clippy --workspace --all-targets -j 2 -- -D warnings`。
日志 `tmp/d3/recheck2_static.log`。命令均保留 `nice -n 15`；沙箱拒绝调整优先级，
没有提权重试。完整回归按 data/platform 两组顺序执行，均为 `-n 2`，单组 870 秒上限。


`test/data`：**1036 passed、7 skipped，418.94 秒**。跳过均为既有 Databento 可选依赖未安装，
没有新增跳过；日志 `tmp/d3/recheck2_data.log`。

`test/platform`：**587 passed、2 warnings，183.67 秒**。警告均为既有 Starlette TestClient
弃用提示；日志 `tmp/d3/recheck2_platform.log`。

合计 **1623 passed、7 skipped、2 warnings**，两组测试耗时累计 **602.61 秒**；
新测试恰好比上一轮增加 9 项，全部通过。`git diff --check` 通过。
汇总为 `bench/results/d3_validation.json`。本轮变更限于 `chip_verify.py`、`verify.py`、
`test_chips.py`、两个核对/变异 bench 脚本及三份结果、`DATA_NOTES.md`、本笔记。
前几轮 ETF 修复与完整性检查保持生效；股票/基金/可转债面板构建未改动。

尚缺的发布证据仍为分布、特殊公司行为口径和跨实际除权快照。本轮解决报告与判定的隔离，
不声称已经验证复权基准或筹码换算；`chip_*` 继续不发布。没有联网、主库/全量 staging
读写、提交或推送；工作日志依任务只改 worktree 的约束继续留在本笔记。

- [15:35] AShare — 合并前 P1/P2 修复完成：新增 9 项回归先红后绿；数据/平台共 1623 通过、7 项既有依赖跳过，变异 0 的 92 项通过、29/29 变异捕获、四项静态检查通过。极值仅作诊断、基准未证实，笔记路径已修正，继续不发布筹码字段。
