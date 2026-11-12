# 数据层交付笔记（第三、四轮，2026-11-12）

本轮只修改数据层允许的目录；没有提交、推送或修改其他 worktree。真实全历史沿用已有原始库，Tushare 和 BaoStock 新增真实调用均为 **0**。以下数字标为“实测”的均来自现有库或本轮执行，不把文档推测当成接口事实。

## 第四轮（2026-11-12）：差异化分红口径与测试缺口

本节是第四轮的全部结论；以下第三轮各节保留原文，凡与本节冲突的以本节为准（来源计数、公司行为口径、变异）。本轮没有下载数据，没有调用任何需要密钥的接口；真实原始库只读，面板建在系统临时目录、验证后删除，`data/cn` 下无新增或变化文件（`current.json` 仍指向 `8c770ce4…`）。

### 第一部分：差异化分红（来源 5）

**现象。** 有回购专户的公司分红时，专户股份不参与分配。持有人每股实得公告值，交易所按摊薄后的虚拟比例计算除权参考价，复权因子的变化因此比公告值小。第三轮的裁决在偏差超过 0.5% 时整格换成合成事件（`split=因子比`、`cash=0`、来源 3），同一种事件按回购比例大小被切成两种记法。

**实现。**（`panel_actions.choose_action`）
1. 候选顺序不变：去重全部、仅现金、仅送转、逐条，0.5% 内对上的照旧采信，来源 1/2 不变。
2. 都对不上、且同日有满足时点要求的记录（公告日不晚于记账日；挪日后的记录同样要求公告日不晚于目标日）时，试「公告值按同一比例 k 摊薄」：前收 P、因子比 a、公告每股现金合计 c、公告每股送转合计 s=split−1，参考价 `(P−k·c)/(1+k·s)`，`k = P·(a−1)/(a·c+P·s)`（`panel_actions.dilution_ratio`；无现金时 `k=(a−1)/s`，不需要前收）。k∈[k_min,1) 就采信：`split`、`cash_dividend` 记**公告值**，`action_source=5`。
3. 其余（k 越界、没有同日记录、只有晚公告记录）照旧按因子合成，来源 3。
4. `k_min` 默认 0.85，`build(action_k_min=…)`、CLI `--action-k-min`；与 `action_tolerance` 一样进入版本哈希，写进元数据 `action_k_min`。
5. 构建期对账：来源 5 的格子核对「k 在 [k_min,1)，且按 k 摊薄后的参考价比与因子比在容差内」；其余来源照旧。元数据 `diagnostics.actions` 新增 `dilution_min`、`diluted`（格数、k<0.90 格数、k 分位、前 20 个样例），`source_counts` 扩到 0–5。
6. **独立校验。** `verify_panel` 重写，不再导入构建期的 `prior_prices`/`reconciliation_check`：前收逐行推进、公告值隐含比写成 `split/(1−cash/P)`、k 写成 `(a−1)/(a·c/P+s)` 后再由参考价反算摊薄比，另外核对「来源 1/2/3/5 必有事件、来源 0/4 必无事件」。构建与校验的 k 公式一起改错时，由手算用例（三格真实数字，用分数手算的 k 写死在测试里）抓到，见变异 N11。

**含义（字段说明、`DESIGN.md` 6.1 已写）。** 来源 5 记的是持有人实得的公告值；按复权因子算的收益比按面板事件算的收益低 `(1−k)` 乘以分配比例（中位 0.78%）；下游 `panelbt` 按面板事件记账，现金流、红利税都按公告值计。

**真实数据结果。** 来源 3 → 5 的格子正好 **72** 格（与复核人的 72 格逐格一致：抽查其列出的前 25 格全部在内，年份分布 2007:2、2019:11、2020:7、2021:6、2022:2、2023:9、2024:12、2025:10、2026:13 完全相同），含现金 56 格；没有同日多条不同记录的格子。

|k 区间|格数|
|---|---:|
|[0.85,0.90)|1|
|[0.90,0.93)|8|
|[0.93,0.96)|15|
|[0.96,0.97)|8|
|[0.97,0.98)|24|
|[0.98,0.99)|16|

最小 0.8752、10 分位 0.9233、中位 0.9731、90 分位 0.9828、最大 0.9888。

**k<0.90 只有一格：600382.SH 2024-07-12**（前收 3.74、派现 0.1828、因子比 1.044690、k=0.8752）。低价、小额派现时 k 对参考价的舍入很敏感：交易所参考价取到 0.01 元，±0.005 元的舍入让 k 变化 ±0.005/0.1828≈±0.027，区间 [0.848,0.903] 覆盖 0.90，与回购专户 10% 上限相容；离线无法确认是否另有大股东放弃分红，保留来源 5。

**k_min 的结论：维持 0.85。** 理由：真实命中最小 0.875，0.85 给小额派现的参考价舍入留出约 0.03 的余量；k_min 以下最近的是 600025.SH 2019-07-05（k=0.842）和 2020-06-19（0.833），两格即使计入舍入（±0.028）也到不了 0.90，回购专户解释不了，保持来源 3 合成更稳妥；再往下是 2007 年股改期和 000409.SZ 2017（0.71）这类别的原因。改成 0.90 会把 600382.SH 一格挤回合成，改成 0.80 会收进 600025 两格而没有依据，都不比 0.85 好。

**读 `action_source` 的地方。** 数据层内部：`panel_actions.reconcile`（写入与计数）、`panel.build`（字段与元数据）、`verify_panel.audit_panel`（独立核对，按来源分支）、`catalog`（字段说明）。`bench/data/round3/report_real.py`（来源计数扩到 0–5；它自带的公告值对账不计来源 5，由 verify 核对）。测试：`test_round3_actions.py`、`test_round3_increment.py`、本轮两个新文件。文档：本笔记、`DESIGN.md` 6.1。`src/ashare` 的其他模块（因子、风险、组合、模拟盘）、`packages/`、`research/` 均不读这个字段（全文检索 0 处）；`panelbt` 只读 `split`/`cash_div`。根目录 `CHANGELOG.md`、`CLAUDE.md` 只提到字段名、不涉及编号，不在本轮范围，未改。

### 第二部分：补测试与变异

新增两个测试文件 `test/data/test_round4_actions.py`（16 条）、`test/data/test_round4_alias_increment.py`（8 条），数据层 94 → **118** 条。夹具全部手写合成；三格手算用例只用了真实案例的四个数字（前收、因子比、公告送转、公告现金）。

变异脚本在临时目录（`r4_mutate.py`，不入库）：克隆只含 `src/ashare` 与 `test/data`，先确认导入路径在克隆里，先跑不改动的对照（118 通过），每处跑**完整** `test/data`，并判断失败的是不是对应的那条测试；结束后源文件逐字节还原、再跑一次 118 通过。

复核人 24 处（源码漂移处手工对应：X6 的 `bad` 表达式多了一项、X20 改为常量 `SNAPSHOT_MAX_AGE`、X21 版本哈希多了 `action_k_min` 一行；其余原样命中）：**24/24 抓到，且都由对应断言失败。**

|编号|变异|抓到它的测试|
|---|---|---|
|X1|仅现金排到全部前|test_candidate_order_and_tolerance_boundary|
|X2|容差 ×4|同上（另有手算的来源 5 用例）|
|X3|挪日 ±3→±1|test_suspension_shift_and_fail_closed、test_shift_requires_known_target|
|X4|挪日不查公告已知|test_shift_requires_known_target|
|X5|记账不查公告已知|test_reconcile_guard_holds_without_upstream_filter（上游过滤被替换成不过滤的版本，记账层守卫单独守住）|
|X6|对账不查有事件无因子变化|test_reconciliation_catches_bad_ratio_and_unbacked_event|
|X7|因子噪声 0.1%→1%|test_factor_noise_threshold|
|X8|前收不错位|test_suspension_shift_and_fail_closed 等 4 条|
|X9|来源恒为 1|test_named_action_cases 等 6 条|
|X10|平局取较晚日|test_shift_tie_prefers_earlier_day|
|X11|dividend_rows 不过滤晚公告|test_late_announcement_filtered_by_dividend_rows|
|X12|去重保留最晚公告|test_economic_dedup_key_and_candidate_order|
|X13|2014-02-21 前不强制来源 3|test_industry_source_forced_before_sw2014|
|X14|换码优先级反转|test_alias_priority_before_and_after_switch|
|X15|接缝有公司行为不拦|test_alias_seam_action_and_missing_quote_rejected|
|X16|接缝价不连续不拦|test_alias_prices_factors_all_datasets|
|X17|因子比例取倒数|同上等 3 条|
|X18|分红不按全列去重|test_alias_dividend_full_column_dedup（相同记录只记一次；同主键不同条款两条都留）|
|X19|换码上下文变化不重建|test_alias_context_change_forces_full_rebuild|
|X20|快照日龄 40→4000|test_snapshot_age_warning_and_strict|
|X21|容差不进版本哈希|test_tolerance_and_k_min_enter_version|
|X22|月定稿宽限期 3→0|test_month_finality_report_window_and_refresh_zero|
|X23|GC 多留一代|test_rename_preserves_incremental_and_gc 等 2 条|
|X24|verify 用 50% 容差|test_verify_fails_on_bad_panel（篡改已发布面板的现金，偏差约 2%）|

第一部分新逻辑 13 处、第三部分 4 处：**17/17 抓到，都由对应断言失败。**

|编号|变异|抓到它的测试|
|---|---|---|
|N1|k 分子 (a−1)→(a−0.99)（构建）|手算用例 test_differential_dividend_keeps_announced_values|
|N2|k 分母去掉 a·c 的 a（构建）|同上|
|N3|k_min 下界放宽 0.01|test_dilution_bounds_and_order（k=0.849 必须合成）|
|N4|上界 1→1.05|同上（k=1.02 必须合成）|
|N5|摊薄候选排到普通候选前|同上（偏差 0.3% 必须走来源 1），另 2 条|
|N6|时点检查（记账层与 dividend_rows 两处守卫同时去掉）|test_differential_requires_known_record 等 4 条|
|N7|来源编号 5→2|手算用例|
|N8|k_min 不进版本哈希|test_tolerance_and_k_min_enter_version|
|N9|verify 的来源 5 k 判定失效|test_dilution_bounds_and_order、test_verify_fails_on_bad_panel|
|N10|构建期矩阵版 k 分子改错|test_differential_requires_known_record（构建对账拦截）|
|N11|构建两处与 verify 两处的 k 一起改错|手算用例（k 用分数手算写死）|
|N12|构建期对账不查来源 5 的 k 范围|test_build_reconciliation_rejects_out_of_range_k|
|N13|verify 不查来源与事件一致|test_verify_source_event_consistency|
|P1|权重非年月分区不整体重建|test_weight_revision_in_generic_partition|
|P2|行业比较不转字符串|test_industry_null_typed_column_increment（`SchemaError`）|
|P3|接缝缺行情静默跳过|test_alias_seam_action_and_missing_quote_rejected|
|P4|行业混用计数取反|test_industry_mixing_counts_foreign_codes|

P1 第一次跑**存活**：我当时把「无下划线的分区名」先折成空串，空串加 `01` 恰好排到最前、等价于整体重建，于是删掉守卫的变异和原代码等价。改回直写 `rsplit` + 显式守卫后单独重跑，抓到。

### 第三部分：小问题的处理

1. **增量的两个布局分支**（`panel_increment.py`）。指数权重分区名解析不出「代码_年月」（如通用的 `all` 分区）时，修订后整体重建（`reuse_prefix` 返回 0），不再把 `all01` 当日期。行业表新旧两份按列比较前一律转成字符串，整列为空、被存成 Null 类型的分区不再让 anti-join 报 `SchemaError`。测试：`test_weight_revision_in_generic_partition`、`test_industry_null_typed_column_increment`（两处都与独立全量逐字节比较）。
2. **换码接缝缺行情改为报错。** 理由：接缝两天的行情决定旧码因子的缩放基准，缺一边时原来按 1 处理并照样改代码，旧段复权因子会静默错位；真实三只接缝都有行情，报错不影响现有构建。原始库根本不覆盖接缝日（两天的 daily 分区不是 complete）时库里至多只有一边的代码，不需要基准，照旧跳过——小样本库、只下载近几年的库都属于这种情形。报错信息提示核对 `catalog.SYMBOL_ALIASES` 的日期。
   **出现新的换码股票时**：
   1. 信号：构建日志出现「daily 有 N 行证券不在证券表，疑似换码」（来自 `diagnostics.joins.daily.unknown_symbol_rows`，本轮起构建时告警）；或者证券表里出现一只上市日很晚、却有多年回写历史的新码。
   2. 量切换日：旧码最后一个有行情的交易日记为 `old_last`，其后新码第一个有行情的交易日记为 `switch`（不要用新码回写历史的最早日期）。
   3. 核对接缝：旧码 `old_last` 收盘价等于新码 `switch` 的昨收（差不超过 1 分）；`switch` 当天两只代码都没有实施的公司行为。不满足时构建会拒绝，需要人工确认。
   4. 在 `catalog.SYMBOL_ALIASES` 加一条 `{old, new, old_last, switch}`，补 `DATA_NOTES` 的换码表。
   5. 改 `catalog.py` 会改变 `code_version`，下次构建自动全量；之后跑 `verify`，核对 `joins.daily.unknown_symbol_rows` 回到 0。
3. **说明文字。** 写进了 `split`/`cash_dividend` 的 `known_at`（`catalog.ACTION_RULE`）、`action_source` 的 `known_at`、元数据 `pit_caveat`、`industry` 的 `known_at`：
   - 派现小于最小报价单位、交易所不除权时按无事件处理（来源 4 或 0），持有人实得的小额现金不计（复核测得每格影响多数 <0.1%）。
   - 合成事件（来源 3）：等效参考价再投资，不扣红利税、没有现金流、持股会变成非整数。
   - 「收盘后可知」对两种用途的含义：按事件记账（`panelbt` 在除权日开盘先派现再调股数）不受影响；把 `split`/`cash_dividend`/`action_source` 当作当日盘前可用的信号（事件驱动策略）就是前视，信号要至少错一天。
   - 行业回填后同一天可能混有 SW2014 与 SW2021 两套编码；构建时计算并写进 `metadata.diagnostics.industry_mixing`（实测数见下）。
4. **已知限制：挪日窗口 ±3 个交易日。** 603014.SH 2025-10-24、603221.SH 2026-08-12 的因子变化与分红记录相隔 4 个交易日，超出窗口，变成「否决（来源 4）+ 合成（来源 3）」：按因子的收益仍对，只是现金被记成送转，红利税和现金流丢失。不放宽窗口：窗口越大，股改期相邻事件错配的风险越高（复核数到同一证券 3 个交易日内两次显著因子变化 46 处）。

### 第三部分附带：`--strict` 复用已建代时也检查快照日龄

补 X20 的测试时发现：版本哈希不含 `strict`，先非严格构建、再 `--strict` 构建时直接返回已存在的那一代，超过 40 天的快照不报错。现在复用已建代前读它的 `diagnostics.index_snapshots`，日龄超过 40 天就拒绝；阈值抽成 `panel.SNAPSHOT_MAX_AGE`。

### 第四部分：真实原始库验证

做法同复核的 `r3_fullbuild.py`：原始库符号链接只读引用、清单导出到临时目录，面板写在临时目录，结束删除（脚本 `r4_real.py`，不入库）。负载：全程 load average 7–8（同机有别的任务）。

|项目|结果|第三轮|
|---|---|---|
|全量到 20261110|42.4 s|40–44 s|
|追加 20261111（复用 4,823 行）|9.1 s|4–8 s|
|全量到 20261111|39.8 s，峰值 RSS 4.77 GB|39.8 s|
|增量 vs 全量|版本号相同，46 个字段逐字节相同，`actions` 诊断相同|同|
|与已发布 `8c770ce4…` 比较|只有 `split`、`cash_dividend`、`action_source` 三个字段不同；差异格 72 格，全部是来源 3→5，其余 43 个字段逐字节相同|–|
|全历史 verify|60.0 s；`action_reconciliation` 检查 53,487、失败 0；`factor_change_has_panel_event` 49,966、失败 0|52 s|

新面板版本 `738d8a03…`（只在临时目录）。来源计数：0: 28,521,003；1: 49,248；2: 1,384；3: 458（原 530）；4: 387；5: 72。构建期对账 53,487 格失败 0，`over_one_percent` 0。行业混用：`industry_mixing.foreign_code_cells` = **4,639**（与复核一致），2014-02-21 前全部 3,352,090 个回填格另计。

追加一天 9.1 s 比第三轮自述的 4.1 s 慢、与复核在负载下测得的 8.4 s 相当；构建多出的工作只有行业混用计数（读 2021-08-02 前的行业与来源矩阵），全量耗时没有可见增加。verify 比 52 s 多 8 s，负载也更高（7–8 对 3–6）；独立实现的前收逐行推进是新增的主要开销，逐年块只 250 行左右。

**`make check`。** 工作区：450 通过、1 跳过（缺 pandas）。全新目录（rsync 锚定排除 `/target /.venv /.cache /data /.git`，核对数据层收集 118 条）：427 通过、24 跳过（23 条模拟盘真实夹具不在副本里、1 条缺 pandas），共 451 = 主干 427 + 本轮 24。

`raw` 中 `factor_change_has_event`（源因子变化但无实施事件）4,549 条与第三轮相同，属源数据缺口，不受本轮影响。

## 第三轮：结果与复跑入口

真实范围为 2007-01-04 至 2026-11-11，4,824 × 5,923，46 个字段。公司行为偏差超过 1%：**515 → 0**；构建和独立全历史 verify 的对账失败均为 **0**。行情、复权因子、行业与指数的稠密数组仍为 C 连续 NPY；时间切片使用只读 mmap。

持久数据根用 `ASHARE_DATA_DIR` 指定，下面的命令不会重新下载历史。先在 shell 中设置这个变量，再执行，避免在同一条环境赋值命令里提前展开空变量。

```sh
make dev && make check
python -m ashare.data build-panel --incremental
python -m ashare.data verify
python -m ashare.data status
python -m ashare.data gc
python bench/data/round3/measure.py build --root "$ASHARE_DATA_DIR" --end 20261110 --record full-build.json
python bench/data/round3/measure.py build --root "$ASHARE_DATA_DIR" --end 20261111 --incremental --record incremental-build.json
python bench/data/round3/compare_real.py --root "$ASHARE_DATA_DIR"
python bench/data/round3/measure.py verify --root "$ASHARE_DATA_DIR" --record full-verify.json
python bench/data/round3/report_real.py --root "$ASHARE_DATA_DIR"
python bench/data/benchmark.py
python bench/data/round3/mutations.py
python bench/data/round3/fresh_check.py
```

真实构建/基准入口必须显式传 `--root`；测试和变异完全用临时目录，子进程移除真实密钥。测试进程解析根路径时另外拦截仓库及其祖先下的真实 `data/`，防止漏传 `tmp_path`。临时全量比较副本只共享原始 Parquet，清单独立、GC 不清共享 raw；退出时删除副本。

审计文件在 `$ASHARE_DATA_DIR/cn/reports/round3/`：`before-actions.json`、`full-build.json`、`incremental-build.json`、`bitwise-real.json`、`full-verify.json`、`final-real.json`、`mutation-results.json`、`repository-audit.json`、`make-check.log`、`fresh-check.log`。完整逐年校验报告在 `cn/reports/verify.json` 和 `verify.md`。仓库只保留脚本与 `bench/data/baseline.json` 一份小基线。

## 逐条处理结果

表内测试路径均相对于 `test/data/`。原审查复现脚本带外部路径或写入真实数据根的，改用相同输入条件的临时目录测试复现；没有直接运行这些危险入口。

|任务条目|处理与代码|守护测试/实测证据|
|---|---|---|
|一.1 因子比、隐含比和 0.5% 容差|已修：`panel_actions.py`；盘后因子对账，容差可配置|`test_round3_actions.py` 的逐格算式；全历史 53,487 格|
|一.2 经济条款去重|已修：`panel_events.dividend_rows`，证券/除权日/送转/税前现金，数值 1e-6 取整；不改 raw|`test_economic_dedup_key_and_candidate_order`，R1/R2|
|一.3 候选顺序与停牌错日|已修：全部、仅现金、仅送转、逐条；相差至多 3 个交易日、公告已知时移到因子变化日|候选顺序、现金/送转/单条回退测试；停牌日期移动测试；实测移动 11 条|
|一.4 等效合成与缩股|已修：无匹配候选时 cash=0、split=a，允许小于 1|招商银行、000887 两例；R4/R6|
|一.5 否决不属于持有人的事件|已修：因子无显著变化且无法匹配，split=1、cash=0，来源 4|002157、002142；R5|
|一.6 来源枚举与元数据|已修：`action_source` uint8，0–4；元数据全历史来源计数|`test_quote_tradable_and_integer_storage`；来源实测表|
|一.7 默认失败、显式放行、verify|已修：`--allow-unreconciled` 默认关；`verify_panel.py` 重算隐含比；报告 `action_reconciliation`|破坏裁决输出须报错，显式放行仍计失败；R8；独立真实普查|
|一.8 九组夹具|已修：`test_round3_actions.py` 人工摘录必要数字，重复行 fixture 只留 2×7 字段|9 个参数化案例；没有整份响应|
|一.9 前后结果/年度合成/20格手算|完成|下列表格及 `final-real.json`；修前普查确实复现 515/198/317|
|二.1 换码|已修：`catalog.SYMBOL_ALIASES`、`aliases.py`；所有证券表型数据统一规范化，原始文件不改|全证券数据集映射、因子比例、接缝失败测试；9,051 行四字段零差异|
|二.2 行业缺口|已修：旧版缺口用 SW2021 的历史区间填充；`industry_source` uint8；字段元数据内有覆盖表|`test_industry_backfill_source`；R11/R12；下列更新表|
|二.3 月初沿用|已修：`month_final` 必须有行；默认沿用最近已知快照，记录日龄，超过40天告警；`--strict` 才拒绝|空月、跨月补抓再增量测试；R13/R14；完整权重缺失仍有明示诊断|
|二.4 十处存活测试|已修：退市/补录/中间插列/行业/月权重修订均增量对独立全量；has_quote 与候选回退逐格断言|N8/N8b/N9/N9b/N3a/N3b/N19/N17/N12c/K2 均捕获|
|二.5 测试不得触碰真实根|已修：`conftest.py` 临时环境；`store.data_root` pytest 保护；删除旧 round2 变异和复现入口|真实默认根解析直接报错；R18；全新目录默认测试根未创建|
|二.6 只读/旧读者/GC|已修：SQLite ro、投影到系统临时目录；旧代删除后新取字段明确要求 reopen，已取得 mmap 可读；原始写者共享锁/GC 非阻塞独占锁|`test_read_only_and_expired_reader`、`test_gc_does_not_remove_live_write`；R16/R17|
|二.7 tradable、停牌与 st_known|已修：`tradable = has_quote & volume>0`；停牌两类不一致写入字段说明；移除无信息的 st_known|零量/无行情/非上市逐格测试；R15/N19/N17|
|二.8 S-2|已修：接口令牌在 HTTP 并发槽前等待；sync 按接口分 worker 组|`test_rate_wait_does_not_hold_other_api_and_corrupt_ledger`，慢接口不占唯一 HTTP 槽|
|二.8 S-6|已修：批次结束按日期重判骤降；`confirm-partition KEY --reason 说明` 核对哈希后确认异常|逆序/正序写入均标 suspect，确认后完成；`test_suspect_confirmation_order_and_old_revision`|
|二.8 S-7|已修：旧 SW2014 行业默认冻结，人工 --force 可刷|`test_zero_month_legacy_freeze_and_repeated_rate`|
|二.8 2.6-1 指数调样滞后|已标注：月末快照下一交易日起可用，可能晚于实际生效日两到三周|`catalog.py` 的权重和成分 known_at；不把未来快照提前|
|二.8 2.6-4 增量诊断只有尾段|已修：所有事件与计数重核全历史，缓存只优化原始事件读取；`action_reference` 由新的 `actions` 对账诊断替代|每个修订场景同时比较完整 actions 诊断与矩阵；缓存哈希修订测试|
|二.8 2.6-5 涨跌停舍入|保留源价，字段注明 1 分误差；公式审计仍严格列出，不为了通过放宽|既有标量/Polars 公式测试；完整报告保留不一致样例|
|二.8 2.6-6 小因子跳变|保留源因子；无声明且 ≤0.1% 的跳变不造事件；仍受 0.5% 总对账约束|微小噪声测试；实测剩余 2,325 个微小变化无事件|
|三.1 bench 清理及路径|完成：只保留7个脚本、1份基线 JSON；旧记录移至数据目录审计档案|无日志/profile/一次性JSON留在 bench；文档与 bench 无本机绝对路径|
|三.2 真实数据不进 git|完成：删除 round2-industry/probe/行业整响应等；仅手写的两行事件与公开错误消息|全仓库无 Parquet/NPY/CSV/Arrow、无完整 fields/items JSON；见卫生检查|
|三.3 dtype、节省量|完成：所有 bool 1字节；来源uint8、行业uint32、上市年龄uint16；浮点类型不变|整数类型测试、真实逐字段清单；下列建议|
|三.4 结束只留一代|完成：默认 keep=1；缓存跟代次生命周期，失败 staging 清理|GC 实测；无 building/raw tmp/投影残留|
|验收与全新目录|完成：本地与全新副本 make dev && make check；没有放宽断言或跳过|162通过、0跳过；变异基线与恢复94通过|

### S-8 的其余小项

- 已修：`account.json` 单独锁文件、临时文件 fsync 后原子替换；半截 JSON 可恢复。只持久化时间调度，不把本进程降速永久写回。假传输和损坏账本测试守住。
- 已修：SQLite 设置 `synchronous=FULL`、`fullfsync=ON`、`checkpoint_fullfsync=ON`。未做真实断电试验，不能声称所有硬件断电情形都已验证。
- 已修：内容回到旧哈希时重用旧 Parquet，旧 `_fetched_at` 不变。测试把内容 1→2→1 并比较最初文件逐字节。
- 不修成“自动猜测上限”：若服务端真实单页上限悄悄低于请求值且不报告总数，短页仍无法区分“结束”或“截断”。现有全历史没有被截断的证据；继续用页分布、跨表覆盖和缺失样例审计。未经接口实测不降低页大小，也不凭空承诺不会漏页。
- 保留分区异常时 fail-fast：权限、每日配额和未知源错误要停止这次运行，已完成分区可续传；避免在明确失败后继续消耗调用预算。频率超限仍按接口等待重试，不属于此类立即失败。

## 公司行为口径和真实结果

全部数字为本轮实测。对账用上一笔有行情的收盘价（跨停牌查找，只用于核算，不填价格面板）；因子比仍是**相邻交易日**的比值。纯股数合成 cash=0 时 θ=split，前收可代数约去，历史首段没有前收也能核验。公告晚于除权日的声明不参与当时候选；可能由源因子合成等效事件。

因子来源是盘后数据，`split`/`cash_dividend`/`action_source` 的可知时刻明确为“收盘后因子对账”。合成事件是按参考价再投资的等效记账，不是假称配股免费获股；不能把这些盘后判决作为当天盘前交易信号。价格本身与源因子不被裁决逻辑修改。

|来源|含义|格数|
|---:|---|---:|
|0|无事件（含非上市格）|28,521,003|
|1|分红表原样|49,248|
|2|经济去重/候选裁决/错日对齐后|1,384|
|3|复权因子合成|530|
|4|声明事件被否决，无持有人事件|387|

修前偏差超过1%为515格（偏多198、偏少317）；修后0格。53,487格因子变化/事件参与总不变量检查，0失败；其中显著因子变化49,966格，缺少面板事件0格。2,325个无事件微小因子变化仍原样保留，误差在容差内。事件来源1/2/3合计51,162格。

实施记录55,835行；同日多条1,399组/2,811行；经济去重删1,388行；实施公告日缺失0，一般公告日ann_date缺失6；公告晚于除权日1条；不在证券主表3,131条（包括非本面板证券）；非交易日除权0条；停牌日期对齐移动11条。这些诊断都针对完整构建范围，增量与全量相同。

|合成年份|格数|
|---|---:|
|2007|131|
|2008|37|
|2009|33|
|2010|28|
|2011|23|
|2012|11|
|2013|19|
|2014|19|
|2015|8|
|2016|11|
|2017|13|
|2018|15|
|2019|26|
|2020|28|
|2021|16|
|2022|21|
|2023|21|
|2024|22|
|2025|27|
|2026|21|

### 20格独立手算

由 `report_real.py` 直接读取发布数组，用 Decimal 逐格代入 `a=新因子/旧因子`、`θ=P×split/(P−cash)`；实现不调用候选裁决函数。前9行是指定案例；另11行按来源补样。下表误差是 `θ/a−1`，百分比只为展示而舍入；完整精度在 `final-real.json`。

|证券/日期|前收P|旧→新因子|split|cash|θ|偏差|来源|
|---|---:|---|---:|---:|---:|---:|---:|
|688766.SH 20230614|176.3|1.407→2.095|1.49|0.0|1.490000000|0.06826%|2|
|002346.SZ 20140321|21.05|1.025→2.05|2|0.0|2.000000000|0.00000%|2|
|601088.SH 20170710|22.29|1.326→1.53|1|2.97|1.153726708|-0.01035%|2|
|002157.SZ 20231208|2.63|16.264→16.264|1|0.0|1.000000000|0.00000%|4|
|689009.SH 20260807|42.6|1.033→1.064|1.03000968|0.0|1.030009681|0.00000%|3|
|002142.SZ 20171116|17.58|1.972→1.972|1|0.0|1.000000000|0.00000%|4|
|600036.SH 20130905|10.66|3.824→3.897|1.01908996|0.0|1.019089958|-0.00000%|3|
|000887.SZ 20070330|3.83|6.449→2.106|0.326562258|0.0|0.326562258|-0.00000%|3|
|301393.SZ 20261110|53.68|1.0223→1.0252|1|0.15|1.002802167|-0.00345%|2|
|000001.SZ 20070620|28.69|25.015→27.524|1.1|0.009|1.100345176|0.00412%|1|
|000002.SZ 20070516|22.97|45.366→68.511|1.5|0.15|1.509859772|-0.02146%|1|
|000006.SZ 20070529|28.48|5.387→5.454|1|0.35|1.012442232|0.00048%|1|
|000001.SZ 20081031|11.31|27.524→35.906|1.3|0.0335|1.303862014|-0.05153%|2|
|000002.SZ 20190815|27.13|142.668→148.412|1|1.045102|1.040065405|-0.01883%|2|
|000012.SZ 20070619|16.59|8.336→8.568|1|0.45|1.027881041|0.00486%|2|
|000004.SZ 20070528|12.52|3.908→4.064|1.03991812|0.0|1.039918117|0.00000%|3|
|000009.SZ 20080516|14.33|2.854→3.674|1.28731605|0.0|1.287316048|0.00000%|3|
|000011.SZ 20091103|14.52|2.086→2.899|1.38974113|0.0|1.389741131|0.00000%|3|
|000033.SZ 20160415|10.38|1.978→1.978|1|0.0|1.000000000|0.00000%|4|
|000035.SZ 20121203|12.25|2.306→2.306|1|0.0|1.000000000|0.00000%|4|

## 换码和行业

实测发现，当前 raw 同时有旧码历史和新码回写历史，并非新码在换码前全部为空。三只旧码的9,051行均与新码历史重叠，因此不能直接重命名后堆叠：接缝前优先旧码、之后优先新码，再按数据集键去重。新码“切换日”定义为旧码最后报价之后的新码首个报价，不把源回写历史的最早日期误认成换码日。原始目录及哈希不变。

|旧→新|旧码末日→新码接续首日|历史行|旧段因子乘数|前收/新昨收|接缝因子比|旧码涨跌停行/映射后缺失|
|---|---|---:|---:|---|---:|---|
|000022.SZ→001872.SZ|20181220→20181226|2618|1|16.46/16.46|1.0|2911/0|
|000043.SZ→001914.SZ|20191213→20191216|3012|1.0001200048|20.25/20.25|1.0|3150/0|
|300114.SZ→302132.SZ|20250214→20250217|3421|1|72.18/72.18|1.0|3510/0|

三处接缝原始实施分红为空，参考价连续；最终 split=1、cash=0。旧码全部9,051行的 close/previous_close/volume/amount 对统一列逐格比较，四项均0差异。原审查指出的001914旧段3,012个报价日涨跌停缺口已经补齐；全体旧码限价记录映射后缺失0。

行业来源：0缺失；1为SW2014当时口径（2014-02-21至2021-07-30）；2为SW2021当时口径（2021-08-02起）；3为历史回填/重述。2014版发布前已有的旧版历史也标3，避免把重述误称当时分类。区间两端均含，一级冲突仍构建失败。两套源都没有归属时保持0，不猜行业。

|日期|上市|回填后缺失|占比|
|---|---:|---:|---:|
|20070104|1411|10|0.7087%|
|20130104|2472|18|0.7282%|
|20140220|2516|17|0.6757%|
|20140221|2516|0|0.0000%|
|20190102|3567|0|0.0000%|
|20210104|4182|1|0.0239%|
|20210730|4474|1|0.0224%|
|20210802|4476|17|0.3798%|
|20261111|5580|0|0.0000%|

此表也保存在 `field("industry")["coverage"]` 和面板字段元数据。严格研究只使用来源1/2；来源3明确带后见之明。SW2014历史接口在第二轮已取得7,558行并保存原始库，本轮只读这些证据，没有新增探查，不把旧分类历史缺口补成无标记的“当时口径”。

## 全历史校验摘要

完整规则、逐年逐板块/是否ST分组、覆盖起点和逐年行数、每类失败样例都保存在 `cn/reports/verify.json` / `verify.md`。报告区分原始库版本与发布面板版本。下面是当前完整报告的原始计数；非零项属于源表审计，不能读成“数据全部干净”。

|规则|检查数|失败数|通过率|
|---|---:|---:|---:|
|action_reconciliation|53,487|0|100.000000%|
|adjustment_continuity|15,622,094|3,683|99.976424%|
|daily_coverage|4,824|0|100.000000%|
|daily_vs_adj_factor|15,629,710|1,749|99.988810%|
|daily_vs_daily_basic|15,629,710|90,785|99.419151%|
|daily_vs_stk_limit|15,629,710|91,438|99.414973%|
|ex_dividend_reference|50,056|906|98.190027%|
|factor_change_has_event|54,254|4,549|91.615365%|
|factor_change_has_panel_event|49,966|0|100.000000%|
|manifest_files|46,449|0|100.000000%|
|market_cap|15,538,925|473|99.996956%|
|missing_quote_has_suspension|16,055,326|52,534|99.672794%|
|no_post_delist_rows|15,629,710|0|100.000000%|
|ohlc_high|15,629,710|0|100.000000%|
|ohlc_low|15,629,710|0|100.000000%|
|partition_coverage_adj_factor|4,824|0|100.000000%|
|partition_coverage_daily|4,824|0|100.000000%|
|partition_coverage_daily_basic|4,824|0|100.000000%|
|partition_coverage_fut_daily|4,824|0|100.000000%|
|partition_coverage_margin_secs|4,824|0|100.000000%|
|partition_coverage_stk_limit|4,824|0|100.000000%|
|partition_coverage_stock_st|4,824|0|100.000000%|
|price_limits|16,000,287|452,575|97.171457%|
|suspension_no_trade|512,733|15,594|96.958651%|
|unique_keys|76,258,233|3,622|99.995250%|
|unlimited_sessions|8,858|0|不适用|

解释与代表性样例：

- `action_reconciliation`和`factor_change_has_panel_event`均0失败，才是公司行为面板的验收项。`factor_change_has_event`检查**原始分红表**，漏配股等仍会失败；这与合成事件填补后面板通过并不矛盾。
- `ex_dividend_reference`的906项是源声明值对参考价的审计，包括差异化分红与不属于持有人的记录；不能再次把它们直接相加覆盖已裁决面板。
- `price_limits`总失败452,575，其中448,027项源昨收非正数，正参考价失败仅4,548/15,521,460。例20070104的000009源昨收为0但限价非0；另有S股名称覆盖缺口、ST切换、20191028科创板无限制哨兵，以及半分舍入差1分。
- `unique_keys`计数包含原始分红多版本等，样例000703.SZ 20070330同报告期实施记录；不静默删原始修订。公司行为另按经济条款去重并对账。
- 行情对daily_basic缺90,785、对stk_limit缺91,438；这些格子不填市值或公式限价。另有行情却无adj_factor 1,749；缺因子不推测造值。
- 上市无行情且无全天停牌记录52,534；有行情且标全天停牌15,594，已写字段说明。tradable以行情且量>0判定，不用is_suspended覆盖成交事实。
- BaoStock本轮没有重联网；以前300对价格/2天状态抽样仍保留，但原始输入指纹变化后被明确标为stale，不能冒充本轮全历史的独立验证。

|年份|正参考价涨跌停检查|不一致|一致率|
|---|---:|---:|---:|
|2007|323,008|2,440|99.244601%|
|2008|360,006|286|99.920557%|
|2009|374,967|39|99.989599%|
|2010|430,977|273|99.936656%|
|2011|510,931|333|99.934825%|
|2012|565,281|265|99.953121%|
|2013|563,930|276|99.951058%|
|2014|569,703|365|99.935932%|
|2015|569,574|42|99.992626%|
|2016|641,302|24|99.996258%|
|2017|743,025|23|99.996905%|
|2018|816,881|5|99.999388%|
|2019|884,521|36|99.995930%|
|2020|944,934|23|99.997566%|
|2021|1,059,547|4|99.999622%|
|2022|1,169,469|42|99.996409%|
|2023|1,257,137|27|99.997852%|
|2024|1,287,814|12|99.999068%|
|2025|1,313,305|16|99.998782%|
|2026|1,135,148|17|99.998502%|

2020-08创业板、2023-04主板注册制、2026-07主板ST限制切换仍沿用已核验规则与测试；本轮不改公式。原始源链接留在 `rules.SOURCES` 与校验报告，本轮未重新联网确认规则。2.6-5要求的1分只写作下游价格判定建议，审计本身保持原断言。

## 性能与输出体积

本机macOS arm64，其他任务同时运行；数字包含文件缓存和后台任务影响，不能承诺固定速度。全量基准用独立清单和共享只读raw重建，逐字段哈希比较，副本删除。所有峰值RSS按macOS字节计。

|任务|构建/校验秒|调用入口总秒|峰值RSS字节|输出字节|
|---|---:|---:|---:|---:|
|全量至11-10|31.230|32.859|4,453,220,352|6,720,175,477|
|追加11-11，复用4823行|4.110|5.834|2,051,899,392|6,721,564,316|
|独立全量至11-11|42.913|42.913|4,325,228,544|6,721,568,255|
|全历史verify|49.292|49.305|2,966,077,440|—|

46个NPY的SHA256全部一致，记录在 `bitwise-real.json`；不同构建的元数据计时、reused_rows以及缓存文件行排列不属于数组逐位一致的范围。全量小于120秒，每日增量小于10秒，verify峰值小于4GB。

最初全历史重对账的增量用时19.01秒；剖析显示慢在数千个事件小分区读取。当前代内的 `_events_dividend.parquet`、`_events_index_weight.parquet` 按分区哈希增补，缓存原始行，不缓存裁决结果；增量仍完整对账，诊断不只算尾段。最初给数千个LazyFrame各加常量列产生了昂贵优化树（一次全量113秒），改为8个线程并行读小文件再concat。事件裁决CPU剖析约1.34秒，没有值得新增Rust包的CPU热点；本轮没有改crates/packages。

T=4800、N=5900的全规模合成散布基线（两种方法各独立进程）：

|方法|秒|峰值MiB|
|---|---:|---:|
|numpy|0.089637|722.05|
|polars|10.086156|1422.77|

继续使用NumPy下标散布；Polars用于读取、连接和分块表达式。未把Python循环模拟Rust并声称测过Rust；结论来自真实热点分布与上述基线。

### 每字段体积

每格数28,572,552，下面是数据字节；每个NPY另有128字节头。bool一律1字节而非8字节。所有既有浮点字段保持第二轮类型不变。

|字段|dtype|数据字节|NPY字节|
|---|---|---:|---:|
|action_source|uint8|28,572,552|28,572,680|
|adj_factor|float64|228,580,416|228,580,544|
|amount|float64|228,580,416|228,580,544|
|can_buy_open|bool|28,572,552|28,572,680|
|can_sell_open|bool|28,572,552|28,572,680|
|cash_dividend|float64|228,580,416|228,580,544|
|close|float64|228,580,416|228,580,544|
|delist|bool|28,572,552|28,572,680|
|dividend_yield|float32|114,290,208|114,290,336|
|dividend_yield_ttm|float32|114,290,208|114,290,336|
|float_market_cap|float64|228,580,416|228,580,544|
|float_shares|float64|228,580,416|228,580,544|
|free_shares|float64|228,580,416|228,580,544|
|has_quote|bool|28,572,552|28,572,680|
|high|float64|228,580,416|228,580,544|
|industry|uint32|114,290,208|114,290,336|
|industry_source|uint8|28,572,552|28,572,680|
|is_listed|bool|28,572,552|28,572,680|
|is_st|bool|28,572,552|28,572,680|
|is_suspended|bool|28,572,552|28,572,680|
|listing_age|uint16|57,145,104|57,145,232|
|low|float64|228,580,416|228,580,544|
|lower_limit|float64|228,580,416|228,580,544|
|member_csi1000|bool|28,572,552|28,572,680|
|member_csi2000|bool|28,572,552|28,572,680|
|member_csi300|bool|28,572,552|28,572,680|
|member_csi500|bool|28,572,552|28,572,680|
|member_csi_all|bool|28,572,552|28,572,680|
|open|float64|228,580,416|228,580,544|
|pb|float32|114,290,208|114,290,336|
|pe|float32|114,290,208|114,290,336|
|pe_ttm|float32|114,290,208|114,290,336|
|previous_close|float64|228,580,416|228,580,544|
|ps|float32|114,290,208|114,290,336|
|ps_ttm|float32|114,290,208|114,290,336|
|split|float64|228,580,416|228,580,544|
|total_market_cap|float64|228,580,416|228,580,544|
|total_shares|float64|228,580,416|228,580,544|
|tradable|bool|28,572,552|28,572,680|
|turnover|float32|114,290,208|114,290,336|
|upper_limit|float64|228,580,416|228,580,544|
|volume|float64|228,580,416|228,580,544|
|vwap|float64|228,580,416|228,580,544|
|weight_csi1000|float64|228,580,416|228,580,544|
|weight_csi2000|float64|228,580,416|228,580,544|
|weight_csi300|float64|228,580,416|228,580,544|
|weight_csi500|float64|228,580,416|228,580,544|
|weight_csi_all|float64|228,580,416|228,580,544|

- 成分标记等11个bool合计314,298,072字节；相对float64少2,200,086,504字节。保持字节bool可直接交给接受bool数组的Rust接口，未做位压缩。
- 来源各uint8（两个合计57,145,104字节）；上市年龄uint16（57,145,104字节，范围不足会报错）；行业保留801xxx原整数编码，uint32是无需字典重编码的最小类型（114,290,208字节）。行业及年龄相对此前float64共节省285,725,520字节，新增来源抵消一部分。
- 如果以后把行业重编码成≤255个分类的字典uint8，可再省85,717,656字节，但必须同时发布编码字典、变更下游解释，不能只cast成uint8。本轮不改变行业编码语义。
- 非价格float64的量额、市值、股本、5个指数权重共12个字段若改float32，可再省1,371,482,496字节（约1.277GiB）。float32相对精度约1e-7：大市值/金额可能有万元级绝对舍入，大股本不能逐股精确表达。价格、派现、split、复权因子继续建议float64。
- mmap对任一dtype本身都可零拷贝；Rust接收端也必须接受对应dtype。若已有PyReadonlyArray<f64>入口，传float32需要转换并复制，不能把“省磁盘”说成自动零拷贝。本轮只落实bool与小整数，没有改浮点。

## 变异检查

同一份源码和测试先全绿，临时副本逐项替换，再恢复；没有修改工作区源码，不把导入/语法错误算捕获。另一次开发中的运行碰到主进程同时修改缓存源码，最终哈希保护按设计拒绝该次结果；下面是冻结源码后完整重跑的30项。基线与恢复均94项通过。

|编号|变异|结果|失败测试|
|---|---|---|---|
|N8|退市变化不回退|捕获|test_increment_revision_branches[delist]|
|N8b|补录旧证券不回退|捕获|test_increment_revision_branches[backdated_security]|
|N9|月权重修订不回退|捕获|test_cross_month_missing_weight_then_increment|
|N9b|行业修订不回退|捕获|test_increment_revision_branches[industry]|
|N3a|不试单条|捕获|test_economic_dedup_key_and_candidate_order|
|N3b|任取第一条|捕获|test_economic_dedup_key_and_candidate_order|
|N19|has_quote 恒真|捕获|test_quote_tradable_and_integer_storage|
|N17|恢复无信息 st_known|捕获|test_quote_tradable_and_integer_storage|
|N12c|同窗口重复减速|捕获|test_zero_month_legacy_freeze_and_repeated_rate|
|K2|新证券只追加末列|捕获|test_increment_revision_branches[middle_column]|
|R1|去重忽略现金|捕获|test_economic_dedup_key_and_candidate_order|
|R2|不做百万分位取整|捕获|test_economic_dedup_key_and_candidate_order|
|R3|优先送转丢现金|捕获|test_economic_dedup_key_and_candidate_order|
|R4|禁用因子合成|捕获|test_named_action_cases[689009.SH-20260807-42.6-1.0300096805421104-records4-expected4]|
|R5|不否决重整事件|捕获|test_named_action_cases[002157.SZ-20231208-2.63-1.0-records3-expected3]|
|R6|缩股截为一|捕获|test_named_action_cases[000887.SZ-20070330-3.83-0.3265622577143743-records7-expected7]|
|R7|不移停牌事件|捕获|test_suspension_shift_and_fail_closed|
|R8|不拦截对账失败|捕获|test_suspension_shift_and_fail_closed|
|R9|不拼旧代码|捕获|test_alias_prices_factors_all_datasets|
|R10|因子不对齐|捕获|test_alias_prices_factors_all_datasets|
|R11|取消行业回填|捕获|test_industry_backfill_source|
|R12|回填冒充当时可知|捕获|test_industry_backfill_source|
|R13|空月定稿|捕获|test_zero_month_legacy_freeze_and_repeated_rate|
|R14|月初拒绝沿用|捕获|test_cross_month_missing_weight_then_increment|
|R15|tradable 忽略零量|捕获|test_quote_tradable_and_integer_storage|
|R16|GC 无写者互斥|捕获|test_gc_does_not_remove_live_write|
|R17|源目录读取需要写权限|捕获|test_read_only_and_expired_reader|
|R18|测试不拦真实根|捕获|test_read_only_and_expired_reader|
|R19|事件缓存忽略修订哈希|捕获|test_event_cache_hash_invalidation[dividend]|
|R20|诊断写入非标准NaN|捕获|test_missing_reference_diagnostics_are_standard_json|

N3a/N3b按新裁决函数重定位单条回退，语义对应原审查；N17旧字段已删除，变异改为重新公开一个无信息st_known，专门的删除契约测试抓到它。R17同时验证只读目录和**可写目录上只读连接拒绝DDL**，不依赖chmod是否能约束高权限进程。新增缓存R19把哈希比较改为仅看分区名，测试立即发现修订丢失。R20确保没有前收时诊断写null而非非标准NaN；当前元数据和报告6份JSON均经严格解析。

## 验证、仓库卫生与共享文件

本工作区 `make dev && make check`：162通过、0跳过。全新临时目录独立 `uv sync --all-packages` 后同样162通过、0跳过，完整日志见数据目录；fresh脚本退出删除副本，默认测试数据根未被触碰。

本轮没有改根目录共享文件（`Makefile`、`pyproject.toml`、`uv.lock`）。git diff中它们原有的改动属于前轮：Makefile第9/10行显式把 `src/ashare/data test/data bench/data` 交给ruff；依赖行仍为已有httpx/polars/pyarrow及data-verify组。根.gitignore按用户要求不改，合并时仍须把data/改成/data/；不要因此漏加数据层代码。

删除的bench一次性记录和历史笔记归档到数据目录 `reports/archive-data-layer`，不是git内容。原有round2-industry完整响应已经删除，旧响应的多余备份也清掉。fixtures仅保留：手写两行分红387字节、用户提供的限频错误245字节、说明325字节；九个案例数字直接写在测试中。没有测试需要成批真实响应，因此没有条件跳过，也没有联网上游的测试。

本轮新建文件（重写已有文件不在此表；完整目录职责见处理表）：

|文件|字节|
|---|---:|
|src/ashare/data/aliases.py|4,783|
|src/ashare/data/panel_cache.py|2,213|
|test/data/test_round3_actions.py|6,336|
|test/data/test_round3_alias_industry.py|4,556|
|test/data/test_round3_increment.py|6,908|
|test/data/test_round3_storage.py|7,457|
|test/data/fixtures/README.md|325|
|bench/data/baseline.json|1,005|
|bench/data/round3/compare_real.py|2,126|
|bench/data/round3/fresh_check.py|1,771|
|bench/data/round3/inspect_real.py|7,269|
|bench/data/round3/measure.py|2,272|
|bench/data/round3/mutations.py|9,852|
|bench/data/round3/report_real.py|8,180|

完整仓库卫生审计和最终磁盘占用在下节更新。密钥检查只输出命中数量，不输出搜索模式或匹配内容；每次写夹具前先检查密钥不在内容中。

## 设计疑问与保留限制

1. 原文“任何一天的值必须那天能看到”与“旧版行业缺口用SW2021回填”：按本轮明确决定允许行业重述，但来源3不可用于严格当时可知研究；2014版发布前所有分类也标3。来源1/2是分类体系口径，并非声称供应商最初发布版本已被完整保存。
2. 原文“前收”遇停牌没有前日行情：只在核算中找上一笔有行情收盘；矩阵价格缺失不前填。无历史前收的纯股数合成按代数约消验证；有现金但无法确定前收时不采信现金候选。
3. 原文“新代码换码前是空的”：本轮raw实测存在新码回写历史，因此优先级拼接并去重；证券身份用新代码稳定表达，名称快照仍不是历史名称。接缝参考价异常或存在实施事件会失败，不能任意抹平未来发现的公司行为。
4. 原文“行情停牌”与“当天是否能成交”：源停牌记录和行情并不完全一致，因此保留独立字段，新增tradable作为日频代理；不声称当天任意时刻都能成交。
5. 原文“只留一代”与“打开的句柄固定版本”：选择允许旧代清理；已取得mmap继续可读，新读取给出明确reopen错误。证券投影在系统临时目录生成，只随Panel对象生命周期存在，不要求数据根可写。
6. 原文“全量/增量一致”不要求构建时间、reused_rows或缓存序列化顺序一致；46个矩阵逐字节一致，完整公司行为诊断在测试中一致。所有事件仍全历史重核，缓存只减少小文件IO；追加或修订分区哈希会失效。
7. 原文“无意义存储”：保留源真实修订和manifest.json迁移回退；面板只一代，事件缓存约几MB且随代删除。没有按天存另一份7GB快照，也不把采集响应放git。
8. 无法仅凭本轮离线数据确认源每日指标中的PE/PB所依赖财务版本是否当时可知，历史原始修订也不能还原首次发布值；字段仍明确收盘后与修订风险。因子合成按用户指定收益口径，不替代真实配股现金流/税务模型。
9. 不放心的仍是源数据异常（上表非零项）、严格行业研究的重述缺口、月末快照延迟、源单页上限变更，以及未实测的断电情形。BaoStock历史抽样已标过期，没有把它包装成新验收结果。
10. 项目外全局worklog写入与本任务“只改worktree和指定数据目录”的范围冲突；按任务范围将阶段日志和可回溯产物集中在本笔记和数据目录报告，没有写项目外日志。

11. 4.11秒增量数字针对列集合不变的追加一天；证券更名/退市可复用前缀，新股插入中间列也在测试中与全量一致。但C连续T×N文件增加N必须重排旧矩阵，写入量接近一代全量，不能把已有列的速度承诺给新增列日。

## 最终卫生与磁盘占用（实测）

仓库检查113个非构建文件，数据目录文本类另查126个文件：真实密钥值命中0。仓库原始数据制品（Parquet/NPY/CSV/Arrow）0，完整fields/items响应JSON 0；夹具只有手写事件行和公开错误消息。严格JSON解析6份当前元数据/报告，无NaN或Infinity。发布面板的构建代码哈希等于最终源码。

|占用项|文件数|逻辑字节|说明|
|---|---:|---:|---|
|raw|46,455|2,686,486,175|保留6个既有旧修订，本轮未新增原始数据|
|panel|52|6,721,564,399|仅1代，含current指针、事件缓存和元数据|
|reports|142|6,547,289|报告、验收日志和前轮审计档案|
|其中事件缓存|2|2,769,556|已包含在panel合计，不重复相加|
|manifest|2|54,392,065|SQLite及原JSON迁移回退|
|根目录其他|5|57|账号节流状态、锁等|
|投影缓存|0|0|列投影已改为系统临时目录|

`.building-*`、raw中的`*.tmp`、临时全量对照目录、全新检查副本均0残留。SQLite WAL/SHM在进程关闭后也无残留。

本工作区 `make check` 末尾原始输出：

```text
test/data/test_store.py .....                                            [ 48%]
test/data/test_verify.py ...............                                 [ 58%]
test/test_engine.py ................                                     [ 67%]
test/test_inputs.py ....................................                 [ 90%]
test/test_properties.py .....                                            [ 93%]
test/test_random.py .                                                    [ 93%]
test/test_rules.py ..........                                            [100%]

============================= 162 passed in 43.19s =============================
```

全新目录末尾：

```text

============================= 162 passed in 45.24s =============================
Fresh copy: make dev && make check passed; default test data root untouched.
```

## 开盘可买、可卖掩码（性能优化 P-016，2026-11-12）

面板新增两个 1 字节布尔字段 `can_buy_open`、`can_sell_open`（共 48 个字段）：以当日开盘价成交能否买入、卖出，口径与 `panelbt.markets.ashare.tradability(open, high, low, upper_limit, lower_limit, is_suspended | ~has_quote)` 逐格相同，规则写在字段说明（`metadata.fields.*.known_at`）里。构建在 `panel_trade.py`（进入代码哈希），只算新增行，增量与全量逐位相同；`verify` 用 panelbt 的函数按年重算逐格比对（检查名 `open_masks_match_panelbt`）。回测适配直接读这两个字段，不再每次从 6 个字段现算。全历史验证、计时和留给负责人定的口径疑问见 `docs/perf_log.d/P-016-open-masks.md`。
