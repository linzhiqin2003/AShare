# C2 实现记录

任务：`ashare.data.crypto`，DESIGN 4.17/6.10 和三轮设计审查。未改 DESIGN.md，未改 C1/C3、未提交、未推送。实际数据仅在本 worktree 的 tmp；不用主目录 venv/data，不安装系统软件。

## 文件清单

- `src/ashare/data/crypto/`：`__init__.py`、`__main__.py`；采集 `sources.py`、`archive.py`、`sync.py`；解析 `formats.py`、`timeutil.py`、`delivery.py`；存储 `storage.py`；事件 `announcements.py`、`events.py`；档案 `instruments.py`、`mappings.json`；面板 `funding.py`、`panel.py`、`build.py`、`dataset.py`、`coverage.py`；独立校验 `verify.py`、`verify_contract.py`、`verify_raw.py`。
- `test/crypto/`：`__init__.py`、`conftest.py`、`test_announcements.py`、`test_collection.py`、`test_events.py`、`test_formats.py`、`test_fixture_layouts.py`、`test_panel.py`、`test_pipeline.py`、`test_sources.py`、`test_verify.py`。
- `test/fixtures/crypto/`：`generate.py`、`inputs.json`、`panel.parquet`、`panel.csv`、`arrays.npz`；`generate_formats.py` 生成 `raw/` 下六个合成 CSV（`spot_ms`、`spot_us`、`usdm_header`、`usdm_no_header`、`funding`、`mark`）及格式说明 `provenance.json`；`announcements/waves.json` 为自撰 FAKE 公告，旧路径不代表真实内容。
- `docs/CRYPTO_DATA.md`：接口、全部字段、生命周期约定、覆盖矩阵、命令及限制；`docs/C2_NOTES.md`：选择、设计问题、回归来源与验收记录。
- `.gitignore`：增加根级 `/tmp/`，防止真实下载数据进入版本控制。

## 实现中的选择

1. **按数据版本复现**：原始内容、冻结清单、输入快照、月面板都保留。追加更晚的原始行与 `known_at` 更晚的事件不影响已有历史行；交易所修补旧归档则产生新版本，旧回测依旧可读，不能把两种场景混为同一个承诺。
2. **事件可知时间（R1 修正）**：有更新证据取 CMS/标题/正文的最晚更新时间，日期精度不足取当日结束；没有更新证据取首发时间并标 `known_at_basis=publish_no_update_evidence`，承认可能存在未标注静默修改。两种时间证据均缺失才取抓取时刻。仍标 `version_unknown`；后续观察到的正文修改不能回写首发，必要时以观察时刻新增版本。人工补入也要记录 known_at 的依据。自然语言不明确的迁移、取消、多个时刻进入待审队列。重复抓取不能覆盖历史版本。
3. **公告与成交分开**：当前 exchangeInfo 的 SETTLING 状态不直接生成历史退出；需要事件。连续三小时零成交冻结只标当前行，永不倒推下架。现货停止后保留未变现估值，不设置期货式 expires。
4. **资金费严格缺失**：连续性由逐笔 interval 字段决定，连区间首尾的漏笔都拒绝；2023-10-31 08:00 后缺官方关联价默认报错。早期只取精确右边界 mark open，不能用之后任意一个价格顶替。最后一行有事件但缺价时整幅矩形一起截断。
5. **规则证据与模型区分**：当前快照向前填用标 assumption；历史消失合约需要显式有出处的规则假设，不能按价位猜。后续快照规则变化追加到原档案，单位保持冻结，无法整除时报错。C2 保存过滤器原始/规范化信息，C3 负责费用、档位和风险规则的唯一考据表。
6. **来源覆盖逐项报告**：不因现货历史很长就承诺标记价、资金费、交割价同样完整。近期 REST K 线可补，但 REST 资金费没有历史间隔字段时，不凭当前 fundingInfo 或相邻时间差倒填；严格运行等月归档/独立证据。
7. **日线只用于信号/报表**：按 trade_hour 输出完整 24 小时，右边界 available_at，不在数据层移位；资金费现金仍按小时交出。额外 `last_funding_rate` 只提供左端已结算的费率。
8. **缺口可以显式建模但不能静默通过**：allow_missing 仅选择持平、不可交易；verify 仍需维护公告或精确例外及来源，否则失败。暂停与恢复按独立事件保存。
9. **迁移时点不臆造**：MATIC→POL 官方明确 1:1、旧现货停止与新现货恢复，但没有逐钱包到账的精确时间。映射表把新币恢复交易时间作为保守模型生效点并注明；不自动据此完成 C3 的资产转换/交易。
10. **采样和环境**：独立 `.venv` 由 `make dev` 建立；重活均尝试 `nice -n 15`，但当前沙箱返回 setpriority 不允许，不能在这里修改系统优先级；pytest 最多 4 worker。onecontext 查 crypto/币安/资金费历史为零条，依据现有设计、三轮审查和只读经验文件实施。
11. **公开仓库夹具（R2 修正）**：原任务的“真实小样本入夹具”由本轮明确要求取代。真实归档和公告只留在忽略的 tmp，公开测试用独立生成的合成值及自撰文字，保留实测格式，不提交原始行、真实文件 SHA 或公告段落。

## 设计问题

### D1：交割价接口 deliveryTime 与实际交割时刻不同

2026-11-15 实测官方 `/futures/data/delivery-price?pair=BTCUSDT` 的 18 条记录都给交割日 **00:00 UTC**；官方季度规格要求 **08:00 UTC**。因此不能把接口字段当实际事件时刻直接减一小时，也不能用标记收盘补掉匹配失败。具体价格只保留在被忽略的原始库。

处理：实际时刻由明确交割事件给出；仅当事件为同日 08:00 时允许匹配该日 00:00 官方记录，保存 `delivery_record_time` 原值。独立校验另写同一事实判定。回归测试在修复前失败，修复后检查 07:00 行交割。

### D2：未上市 settlement 占位与“追加数据完全不改历史”的边界

DESIGN 要求上市前填首次标记收盘。若一个版本根本没观察到该交易对上市，无法提前生成这个正值。处理：不能为尚无首次标记价的期货列补造占位，构建报错；追加数据不改历史的保证针对原有已可构建列/固定单位及同一原始修订。新增未来上市的列，旧版仍按原列集读取，不把列集变化当作历史数值一致测试。

### D3：归档首次观测不等于精确挂牌时间

现货 exchangeInfo 没有历史上市时间，归档目录也可能缺早期数据。处理：首个归档观测作为可用覆盖起点，写 `listing_basis=first_archived_observation` 与源 URL；不声称恢复真实首笔挂牌秒。若研究需要 min_age 的精确上市边界，用户/C3 应补有来源的 listed_at。

### D4：近期资金费 REST 无历史间隔，不能承诺同步到现在就可回测到现在

REST 能给实际费率/关联价，但不能单凭它确定漏笔还是改变间隔。当前 fundingInfo 也不能用于历史。原始下载保留 REST，严格构建仍需要归档 interval 或独立规则证据；不以降低校验要求兑现不成立的覆盖承诺。

### D5：最终网页没有更新证据时，不能一概把历史公告推迟到抓取日

DESIGN 6.10 的“版本未知、不倒填最终内容”没有单列完全缺少更新证据时的处理。初版采用抓取时间，会让历史下架公告普遍在回测期间不可知，破坏 4.17 的现货退出调度。按 Claude 返工 R1 与用户明确要求，无更新证据时允许用首发时间，并以 `publish_no_update_evidence` 写明假设；有更新证据仍用最晚更新时间，不恢复无法取得的初版。未改 DESIGN.md。

## 开发回归与验收记录

| 来源 / 发现 | 回归 |
|---|---|
| R1-16、经验文件：现货微秒、UM 毫秒、表头不能按年份猜 | 六份实测格式的合成 CSV（R2 替换）、单位混杂/范围/表头错误、S3 多页与无进展 |
| R1-17、R2 N2、R3 V4：资金费原价不同于小时 open | 差异只诊断、现金公式、漏首/中/末笔、间隔字段、早期右端近似、末行截断 |
| R1-9、R2 N6：交割价直接结转 | 手算 199 与 mark_close=208.5，08:00 归 07:00 |
| 开发实测：deliveryTime 是日期标签 | `test_real_delivery_api_uses_midnight_date_label`，先失败后修复 |
| 开发实测：CMS JSON 使用 child | `test_cms_child_tree_regression`，先失败后修复 |
| 开发自检：页面更新时间证据不应丢失 | `test_migration_and_no_unproven_backdating`，改成最晚可证明时间，先失败后修复 |
| R1-10、R3 V3：僵尸尾部与修订前视 | 暂停恢复、零量不退出、现货未变现、OMG 修订/取消、截断数据与事件一起比较 |
| R1-11/12/18、R2 N8/N9 | 固定单位、1000PEPE 价量换算、金额不乘倍率、非整点规则、TRADIFI/quote/margin 筛选 |
| 开发自检：冻结乘数错误地忽略后续规则快照 | `test_sync_appends_changed_snapshot_rules_without_reselecting_unit`，先失败后修复 |
| 开发自检：损坏内容地址对象续传未重写 | `test_corrupt_object_repaired_on_resume`，先失败后修复 |
| 开发自检：暂停持平的校验却读当小时 mark | 暂停恢复测试增加 verify pass，先失败后修复；校验独立使用上一估值 |
| 真实 E2E 截断：首次事件前 null 推断导致追加后历史列类型变化 | `test_future_first_event_does_not_change_prefix_schema`，先失败后固定全部数值与可空事件字段类型，再通过真实截断核对 |
| 6.10 独立性 | 独立 CSV 解码、12 类直接产物变异、禁止 verifier 导入构建/解析/事件业务函数 |
| 联合旧数据层测试发现 pytest 模块同名 | 新目录成为独立测试包；`test_crypto_and_existing_data_modules_collect_together` 保证四个同名新旧模块同时收集，首次联合运行已复现错误 |
| 命令与存储 | 全离线模拟 HTTP 的 sync→build→verify、重新同步跳过 ZIP、旧版本读取、校验失败/限速/超时/预算 |

真实样本数据、日志和报告在 `tmp/e2e/`，原始格式探针在 `tmp/real-formats/`、`tmp/probe/`，均被 `/tmp/` 忽略。首轮真实同步约 21.3 MB，6 列×1,464 小时，54 个归档，独立校验 0 错误；价格诊断只作复核。没有全历史下载。

## Claude 返工第 1 轮：公告 known_at（P1）

修改 `announcements.py` 的首发回退、依据标记、正文/标题更新格式、CMS 文本节点 HTML 实体；保留列表的 `releaseDate` 给解析器使用。列表补充字段不写回原文对象，避免 hash 变化造成虚假修订；真实观察到的静默修改仍以较晚抓取时刻新增版本，标 `observed_revision`，不改旧版。

检查了 `tmp/probe/waves.json` 和 e2e 内容库中的公告，去重后只有同一篇 WAVES 正文。其正文有日期精度的更新标记，CMS `lastUpdateTime=0`。当时旧代码已能识别该标记；e2e 的自动事件仍记 2026 年，是早期提取结果被原文 hash 相同的续跑逻辑保留。R1 使用过的真实节点夹具已在 R2 替换为自撰合成文字，只保留结构与更新短句式。括号、冒号、大写、实体空格等其他形式也是合成兼容性测试，不宣称有其他真实格式样本。

新增 `test_announcements.py` 共 **22 个离线用例**：首发两字段及空值回退、仅全缺时抓取回退、7 种更新文本、标题/正文/CMS 多更新时间取最晚、秒级时间、日期精度与抓取上界、CMS 树中的更新标记、自动现货退出可知时点、未来重复抓取不改事件、观察修订保留历史、补列表字段不伪造修订。R2 更换正文后继续保留这些断言。

- 先在旧实现运行首批 21 例：**20 failed、1 passed**，日志 `tmp/review-r1-before.log`；真实 WAVES 时间已识别，但缺 `known_at_basis`，真实采集测试同样阻止缺标记回归。
- 开发中发现“给原文添加列表日期会改变 hash”问题，先加第 22 例复现 `imported=1` 而预期 0，日志 `tmp/review-r1-cache-before.log`，再改为仅传给解析器。
- 最终 `nice -n 15 .venv/bin/pytest -q test/crypto -n 4`：**89 passed**（3.13 秒），无跳过，日志 `tmp/review-r1-tests-final.log`。全仓 `ruff check .`、`ruff format --check .`（748 文件）及 `git diff --check` 通过。沿用 worktree 自有 `.venv`；nice 仍受沙箱 setpriority 限制。

离线重建脚本为 `tmp/rebuild_review_r1.py`：读取旧冻结清单中的原始正文与当时抓取时间；新建 `tmp/e2e-review-r1/`，复用相同归档/REST 对象，重新提取 5 个自动事件，保留季度交割的人工事件，移除人工 WAVES 事件。它不是在旧事件账本里倒写时间。原始数据版本仍为 `bb9a952def22012c5fd519a71ca24551c993e50d9de2495184877a9d4c15f886`，新增下载 **0 字节**。产出已存在时脚本拒绝覆盖；下面的 build/verify 可以重复运行：

```bash
nice -n 15 .venv/bin/python -m ashare.data.crypto --root tmp/e2e-review-r1 build
nice -n 15 .venv/bin/python -m ashare.data.crypto --root tmp/e2e-review-r1 verify
```

新面板版本 `792d023e55b5a59b99e787948b94db12727586f767f7ac9b8eb17b713de1a933`；2024-05/06，6 列、8,784 格，54 个归档；**verify pass、0 错误**，490 条资金费价格诊断。新旧面板逐列比较，仅 `event_version` 不同，所有价格、资金费、结算和其他字段逐项相等。旧目录的 18 个事件/面板/计划/最新版本文件 SHA 全部未变。

两条 WAVES 自动事件共有字段：`published_at=1717383610753`（2024-06-03 03:00:10.753 UTC），`known_at=1717459200000`（2024-06-04 00:00 UTC），`known_at_basis=latest_update_evidence`、`manual=false`、`version_unknown=true`、`supersedes=null`、`settlement_price=null`。只有更新日期，取次日零时；原文并没有给结算价。

| instrument / kind | effective_at（UTC） | event_id |
|---|---|---|
| `usdm:WAVESUSDT` / settlement | 2024-06-11 09:00 | `f65faeeefe07417a8b6dd2900ba8da7e:usdm:WAVESUSDT:settlement` |
| `spot:WAVESUSDT` / delist | 2024-06-17 03:00 | `f65faeeefe07417a8b6dd2900ba8da7e:spot:WAVESUSDT:delist` |

出处仍为 [币安原公告](https://www.binance.com/en/support/announcement/detail/f65faeeefe07417a8b6dd2900ba8da7e)。WAVES 永续 6-11 08:00 行使用 `mark_open_approximation`，BTC 季度 6-28 07:00 行使用 `delivery_rest`，结算值均与各自原始来源核对一致。原六列面板没有 WAVES 现货行情，现货验证覆盖真实公告提取与 asof 可知时点，未声称完成它的逐小时卖出回测。摘要及完整事件字段在 `tmp/e2e-review-r1/summary.json`，CLI 日志为 `tmp/review-r1-{build,verify}.log`。

## Claude 返工第 2 轮：公开仓库夹具（P1）

本轮只改夹具、测试和文档，未改变 `src/ashare/data/crypto/` 的解析或事件规则。共 15 个文件：

- 替换 `raw/spot_ms.csv`、`spot_us.csv`、`usdm_header.csv`、`usdm_no_header.csv`、`funding.csv`、`mark.csv` 及 `provenance.json`。
- 替换 `announcements/waves.json`；新增 `test/fixtures/crypto/generate_formats.py`，离线生成上述 8 个文件。
- 新增 `test/crypto/test_fixture_layouts.py`；调整 `test_formats.py`、`test_announcements.py`、`test_events.py` 的合成值、虚构交易对与断言。
- 更新本文件及 `docs/CRYPTO_DATA.md`，不再把待入库夹具描述成真实归档或原文。

合成价格以 100 为固定起点，数量以 10/5 为起点，费率用人工指定的三个常量；只依赖标准库和生成器内的常量，不读取真实行情，也没有用真实值平移或缩放。元数据只列格式来源类型、数据集、文件名模式、结构特征和 `synthetic=true`，不含真实文件 SHA、校验文本或原始行。公告使用自撰文字和 FAKE/DUMMY/MOCK/SAMPLE，原路径只为兼容测试，文章 ID 和首发时分秒也为合成值。

核对点（逐行逐列与替换前记录的纯格式信息比较，6/6 一致）：

| 检查项 | 结果 |
|---|---|
| 列数与顺序 | 五种 K 线格式均 12 列；资金费 3 列；列序完整相同 |
| 表头 | 现货两份、旧 U 本位无表头；其余有表头；表头字段和大小写逐字一致 |
| 时间戳 | 毫秒 13 位、现货微秒 16 位；K 线开收时间均检查原生单位，小时步长及右边界减 1 |
| 现货数值 | OHLC、量、金额、主动买入量/额均 8 位小数 |
| U 本位成交数值 | OHLC 2 位、量/主动买入量 3 位、金额/主动买入额 5 位；count/ignore 整数 |
| 标记价数值 | OHLC 通常 8 位，保留 9 个整数价格格的位置；量/金额/ignore 为整数零、count 为 3600 |
| 资金费 | calc_time 为毫秒整数，interval 为整数 8，费率固定 8 位；另构造 0/1/2/3/5 ms 抖动，解析回整点且归档原值保留 |
| 编码、分隔与行尾 | ASCII/UTF-8 兼容、无 BOM、逗号分隔、无引号、LF，最后一行有换行 |
| 行数 | 每份总计 20 行；有表头为 19 条数据，无表头为 20 条数据 |
| 公告结构 | JSON 字符串 body、node/child/text 嵌套、标题形式、毫秒 publishDate、lastUpdateTime=0；保留必要更新/下架/结算短句式，5 个事件含 2 个 FAKE 事件 |

资金费抖动是对已知格式能力的合成覆盖，不声称与原样本的逐笔时间偏移相同。完整小数位布局比较报告在 `tmp/review-r2-format-comparison.json`；生成器重复运行，8 个输出文件逐字节相同。C1/C3 共用的输入、Parquet、CSV 面板和 NPZ 未改。

新增 7 个回归先在旧夹具上全部失败，日志 `tmp/review-r2-before.log`。最终 `nice -n 15 .venv/bin/pytest -q test/crypto -n 4 --basetemp=tmp/pytest-review-r2-final` 为 **96 passed**（4.13 秒），无跳过；全仓 Ruff 检查、格式检查（750 个文件）及 `git diff --check` 通过。日志 `tmp/review-r2-tests-final.log`。nice 受沙箱 setpriority 限制的情况同前轮。

对 `test/`、`docs/`、`src/` 下 658 个未被忽略的文件扫描：旧样本 346 个价格文本、117 条完整 CSV 行、27 个长度至少 100 字符的公告文本节点均无匹配。问候语标记只命中拒绝它的负向测试断言；没有公告内容。小时 CSV 行模式只命中本轮五份已核对的合成 K 线。额外检查交割和下架价格数字，文档已改为核对结论，未保留价格行。扫描报告 `tmp/review-r2-content-audit.json`。

未重新下载或重建真实 e2e；保留 R1 已通过的 8,784 格、54 份归档、0 校验错误，以及审查人独立重算的 BTC 两个月 OHLC、标记价和 183 笔资金费一致结论。原有 tmp 中 311 个文件逐一核对 SHA 未变，忽略规则仍覆盖探针、归档和真实面板；本轮仅在那里新增审计报告和测试临时文件。未提交、未推送。

## 本地工作日志

本任务明确禁止写 worktree 外文件，当前沙箱也未授权 `~/worklog/codex/`。因此本次没有越界写全局日志；在本文件保留完成记录，供用户之后统一归档，未修改全局 FORMAT.md。

- [03:49] AShare — 完成 C2 加密货币原始采集、版本事件、月面板、独立校验与共享夹具；66 项离线测试通过，6 列两个月真实端到端 8,784 格、54 份归档、0 校验错误，历史截断与排名一致。

最终检查：全仓 `ruff check .`、`ruff format --check .` 通过；`pytest -q test/crypto test/data -n 4` 为 **1,011 passed、7 skipped**（旧 Databento 可选依赖缺失），新增 crypto 67 项全部通过。`git diff --check` 通过；共享 Parquet 与重建结果逐项相等，NPZ 形状和手算交割值核对通过。

- [03:56] AShare — C2 收尾验收完成；新旧数据层联合测试 1,011 通过，7 个旧可选依赖测试跳过，全仓 ruff 通过，真实面板版本 c9e3f5e5… 与截断历史一致。

- [04:10] AShare — 完成 Claude C2 R1 公告 known_at 返工：无更新证据采用标明假设的首发时间，真实 WAVES 原文及 22 个回归已补齐；89 项离线测试、全仓 Ruff 和缓存重建 verify 通过，自动 WAVES 已知时间为 2024-06-04 UTC，旧版本不变。

- [04:21] AShare — 完成 Claude C2 R2 公开夹具返工：六份 CSV 与公告改为可复现的独立合成样本，逐格格式一致；96 项离线测试及全仓 Ruff 通过，658 文件扫描无原始行情/公告长文本，原有 tmp 311 文件未变。
