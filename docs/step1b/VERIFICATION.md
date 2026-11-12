# 1b 审查返工验证记录

本文件记录返工后的结果，替代初次交付的验证结论。逐条处理见 [REVIEW_FIXES.md](REVIEW_FIXES.md)。

## 当前目录与干净导出目录

`make dev SANDBOX=1 && make check SANDBOX=1` 在两处均返回 0。检查包括 Ruff、格式、cargo fmt、clippy -D warnings、cargo test 和 pytest。干净目录没有复制 .cache、target、.venv；从 git archive HEAD 加允许范围的未提交改动导出，使用共享 Cargo 下载缓存，测试不创建仓库 .cache。命令由 `bench/review_clean.py` 固化，验证完成后删除导出目录。

当前目录末尾原始输出：

```text
configfile: pyproject.toml
testpaths: test
plugins: platformdirs-4.14.5, hypothesis-6.170.15
collected 131 items

test/test_engine.py ................                                     [ 12%]
test/test_extended.py ....................                               [ 27%]
test/test_inputs.py ....................................                 [ 54%]
test/test_properties.py .....                                            [ 58%]
test/test_random.py .                                                    [ 59%]
test/test_review.py ...........................................          [ 92%]
test/test_rules.py ..........                                            [100%]

============================= 131 passed in 18.52s =============================
```

干净目录末尾原始输出：

```text
plugins: hypothesis-6.170.15
collected 131 items

test/test_engine.py ................                                     [ 12%]
test/test_extended.py ................s...                               [ 27%]
test/test_inputs.py ....................................                 [ 54%]
test/test_properties.py .....                                            [ 58%]
test/test_random.py .                                                    [ 59%]
test/test_review.py ...........................................          [ 92%]
test/test_rules.py ..........                                            [100%]

=========================== short test summary info ============================
SKIPPED [1] test/test_extended.py:518: could not import 'pandas': No module named 'pandas'
======================= 130 passed, 1 skipped in 18.87s ========================
```

干净环境未装可选 pandas，因此按任务要求 importorskip 该项；当前环境已装 pandas，该项实际通过。没有跳过任何内核、随机对拍或历史样本检查。

## 20000 组第一轮摘要

真实 faec31e 构建与当前各运行项目自己的 case(seed)，seed 0–19999；7437 组逐位相同，12563 组不同。每组摘要覆盖原有日序列、持仓、成交、退市事件和诊断；原始/修正摘要并存在 test/fixtures/legacy_sha256.npz。测试读取固定修正摘要，不会自动更新它。

另真实构建 faec31e 的“仅调整公司行为估值价”对照：8578 组受公司行为修正影响，9212 组受其他数量规则修正影响，两者重叠。2902 个公司行为变化种子没有原持仓在缺收盘价日发生行为的条件：缺成交价的盘中估值、同日退市、重建历史价也会受到影响。全部修正输出通过独立成交账本；账本验证给定成交后的钱，成交决策另由手算和申报/约束检查守护。

**未满足窄口径的“其余种子逐位一致”**：它与同时修复第一轮卖出下限、非零低目标、零头单的要求冲突。没有隐藏这部分差异，也没有为了相同摘要保留错误。设计疑问与处理理由在 STEP1_NOTES。

逐种子、逐字段、逐日的差异明细（解压后约 8 MB）没有入库，可用 `bench/review_legacy_build.py` 重新生成；汇总在 `bench/results/legacy-summary.json`。审查时另做了归因：把四条规则修正（公司行为估值、卖出下限、非零小目标、零头容差）逐条关掉重新构建，四条全关时 20000 组与第一轮逐位相同，所以差异全部来自这四条修正。下面列点名种子的首个变化日。

|种子|首个变化日|变化字段|分类|
|---:|---:|---|---|
|581|—|逐位相同|—|
|947|3|nav, cash, turnover, tax, other_fee, slippage_cost, positions, trades.qty, trades.tax, trades.other_fee, trades.slippage_cost|corporate_price_adjustment, closing_minimum_nonzero_residual_or_dust|
|978|4|nav, cash, turnover, commission, tax, other_fee, slippage_cost, n_blocked_sell, positions, trades.rows, trades.qty, trades.tax, trades.other_fee, trades.slippage_cost|closing_minimum_nonzero_residual_or_dust|
|1118|5|nav, cash, turnover, commission, tax, other_fee, slippage_cost, positions, trades.rows, trades.qty, trades.commission, trades.tax, trades.other_fee, trades.slippage_cost|corporate_price_adjustment, closing_minimum_nonzero_residual_or_dust|
|1175|11|nav, cash, turnover, tax, other_fee, slippage_cost, n_blocked_sell, positions, trades.qty, trades.tax, trades.other_fee, trades.slippage_cost|closing_minimum_nonzero_residual_or_dust|
|1406|3|nav, cash, turnover, commission, tax, other_fee, slippage_cost, positions, trades.rows, trades.qty, trades.tax, trades.other_fee, trades.slippage_cost|closing_minimum_nonzero_residual_or_dust|
|1605|2|nav, cash, turnover, commission, tax, other_fee, slippage_cost, n_blocked_buy, n_blocked_sell, positions, trades.rows, trades.qty, trades.commission, trades.tax, trades.other_fee, trades.slippage_cost, events.rows|closing_minimum_nonzero_residual_or_dust|
|3938|3|nav, cash, turnover, commission, tax, other_fee, slippage_cost, positions, trades.qty, trades.commission, trades.other_fee, trades.slippage_cost, trades.rows, trades.tax, events.price, events.proceeds, events.qty|corporate_price_adjustment, closing_minimum_nonzero_residual_or_dust|
|4865|21|nav|corporate_price_adjustment|

复现历史摘要：`CARGO_HOME="$PWD/.cache/cargo-home" .venv/bin/python bench/review_legacy_build.py`。脚本创建真实历史构建和公司行为单独对照、输出摘要/差异后删除历史检出；完整临时账本仅留忽略目录。

## 第一轮对当前配对计时

平台 macOS-26.6-arm64-arm-64bit-Mach-O，15 逻辑 CPU。审查 perf_pair.py 的同一输入生成方式，T=2000、N=2000、100 个目标、21 日调仓；两端 release/LTO，常驻进程，6 轮 AB/BA 交替，每场景每端 240 次。导入、数据构造和 Market 构造不计。机器有并行任务，load average 起止 [12.814453125, 11.6806640625, 10.84375] → [12.4287109375, 11.619140625, 10.82666015625]，不是 CPU 利用率。

|场景|第一轮最小/中位 ms|当前最小/中位 ms|最小值比率|≤1.05|
|---|---:|---:|---:|---|
|full_rules|3.357 / 4.658|1.481 / 2.119|0.4412|通过|
|no_delist_panel|1.764 / 2.521|1.587 / 2.184|0.8997|通过|
|prices_only|1.253 / 1.849|1.122 / 1.563|0.8955|通过|

原始每次耗时没有入库；make bench 会自动重跑此配对，历史目录用后删除。

## 4000×5500 完整规则与剖析

make bench 的完整 A 股规则整手 API 中位 3.783 ms（最小 3.450 ms），仍未达到 2 ms。内部阶段单独测量，不能把独立中位数相加当作一次真实调用。

|阶段|中位 ms|
|---|---:|
|准备|0.098|
|内核与绑定|3.566|
|结果组装|0.025|

独立一次性源码副本插入计时器，正式内核未加计时。20 次含仪表开销的中位耗时 3.918 ms；以下占比按各段中位数 / 总中位数计算。原始数据在 review-profile.json。

|阶段|仪表中位 ms|占比|
|---|---:|---:|
|flows|0.061|1.5%|
|corporate_actions_and_delists|1.057|27.0%|
|command_and_valuation|0.371|9.5%|
|execution|1.044|26.7%|
|close_and_daily_outputs|0.874|22.3%|
|未分段部分、计时及绑定开销|0.510|13.0%|

已消除整行退市回查、无行为日的多余计算和现金账户中的空头/强平分支。剩余主要是持仓证券跨面板访存、有效成交报价/现金二分预算以及收盘估值和必需结果写入。直接删掉事件检查或费用试算会改变语义。当前尚未引入按日公司行为索引；它可能减少无事件日的访存，但会增加 Market 构造时的预处理，并改变内部接口，仍需独立测量与账本验证。继续压低还可研究数据布局和报价路径。上述最小值是忙机观测下限，不是理论不可优化的证明，2 ms 缺口保留。

与初次 1b 固定基线另有退化标记：
- lots: 3.783 ms，-11.1%。
- fractional: 4.001 ms，+20.3%，REGRESSION >10%。
- batch: 585.695 ms，+8.3%。
- long_short: 4.845 ms，+34.8%，REGRESSION >10%。

## 健康语料覆盖与独立账本

独立账本不调用 reference 或 api._prepare，重算费用、现金、公司行为、空头、估值、NAV、时间加权收益、单位净值和贡献，并逐笔检查成交顺序、can_short、申报单位、成交量及开仓约束。20000 组原生成器 + 3000 组小比例现金流语料，416479 笔成交，236699 次开仓约束检查，0 个不符。覆盖计数是触发次数，不是不同账户数。

|3000 组新语料项|次数|
|---|---:|
|cases|3000|
|trades|61164|
|failures|0|
|opening_constraint_checks|33191|
|shares_target|30419|
|trade_band|792|
|dividend_tax|4922|
|cash_flow|24797|
|withdrawal_shortfall|1659|
|cash_interest|46779|
|impact|38439|
|per_share_fee|61164|
|commission_cap|32277|
|open_short|15217|
|cover_short|13241|
|margin_scaled|6297|
|borrow_cost|38449|
|short_dividend|4421|
|margin_call|4474|
|liquidation|4119|
|bankruptcy|121|
|financing_interest|4182|
|reduction_clipped|48|
|invalid_quote|1106|
|genuine_bankruptcy|121|
|reversal|5198|

正式测试仍保留原 2000 组与新增 1000 组逐日/逐笔对拍；覆盖必须命中，未缩小随机范围或放宽对拍断言。

## 重做的变异检查

40/40 被指定手算测试捕获，退出码均为 pytest 1；构建失败不计捕获。前 24 项复跑，25–31 是审查中存活的七处双边变异，32–39 覆盖本轮修复，40 是新增前缀约束反例。每次 Rust 变异后与恢复后均 make dev，conftest 核对源码指纹；基线全检和最终全检通过，源文件与 git diff 均恢复。

|编号|变异|范围|捕获测试|结果|
|---:|---|---|---|---|
|1|tax omitted|kernel|`test_engine.py::test_fees_and_tax_on_slipped_amount`|捕获（1）|
|2|can_sell ignored|kernel|`test_engine.py::test_can_sell_and_carry`|捕获（1）|
|3|cash scaling omitted|kernel|`test_engine.py::test_proportional_cash_scaling_before_column_order`|捕获（1）|
|4|commission floor omitted|kernel|`test_engine.py::test_min_commission_and_cash_scaling`|捕获（1）|
|5|dividend cash omitted|kernel|`test_engine.py::test_dividend_before_split_and_odd_buy`|捕获（1）|
|6|split omitted|kernel|`test_engine.py::test_dividend_before_split_and_odd_buy`|捕获（1）|
|7|carry touches all holdings|kernel|`test_engine.py::test_carry_only_blocked_security`|捕获（1）|
|8|volume cap omitted|kernel|`test_engine.py::test_volume_and_odd_lot_liquidation`|捕获（1）|
|9|sell slippage reversed|kernel|`test_engine.py::test_fees_and_tax_on_slipped_amount`|捕获（1）|
|10|delist recovery ignored|kernel|`test_engine.py::test_delist_prior_close_and_no_reentry`|捕获（1）|
|11|shares rounded|kernel|`test_extended.py::test_shares_bypass_lots_and_trade_band_exceptions`|捕获（1）|
|12|band omitted|kernel|`test_extended.py::test_shares_bypass_lots_and_trade_band_exceptions`|捕获（1）|
|13|dividend tax omitted|kernel|`test_extended.py::test_dividend_flow_interest_and_contribution`|捕获（1）|
|14|external flow omitted|kernel|`test_extended.py::test_dividend_flow_interest_and_contribution`|捕获（1）|
|15|cash interest omitted|kernel|`test_extended.py::test_dividend_flow_interest_and_contribution`|捕获（1）|
|16|impact omitted|kernel|`test_extended.py::test_impact_cash_and_panel_slippage`|捕获（1）|
|17|per-share commission omitted|kernel|`test_extended.py::test_per_share_commission_without_cap`|捕获（1）|
|18|commission cap omitted|kernel|`test_extended.py::test_commission_cap_directional_share_fees`|捕获（1）|
|19|can_short ignored|kernel|`test_extended.py::test_short_permission_dividend_borrow_cover_and_delist`|捕获（1）|
|20|short dividends omitted|kernel|`test_extended.py::test_short_permission_dividend_borrow_cover_and_delist`|捕获（1）|
|21|borrow fee omitted|kernel|`test_extended.py::test_short_permission_dividend_borrow_cover_and_delist`|捕获（1）|
|22|maintenance call omitted|kernel|`test_extended.py::test_margin_call_liquidate_record_and_bankruptcy`|捕获（1）|
|23|forced direction reversed|kernel|`test_extended.py::test_margin_call_liquidate_record_and_bankruptcy`|捕获（1）|
|24|initial margin uses maintenance|kernel|`test_extended.py::test_margin_scaling_financing_and_account_identity`|捕获（1）|
|25|M3 restricted interest base|dual|`test_review.py::test_restricted_proceeds_interest_bases`|捕获（1）|
|26|S28 cash interest includes proceeds|dual|`test_review.py::test_restricted_proceeds_interest_bases`|捕获（1）|
|27|S29 financing ignores restricted proceeds|dual|`test_review.py::test_restricted_proceeds_interest_bases`|捕获（1）|
|28|S11 reversal volume reset|dual|`test_review.py::test_reversal_shares_daily_volume`|捕获（1）|
|29|S12 longs open before shorts|dual|`test_review.py::test_opening_and_reduction_order`|捕获（1）|
|30|S13 covers precede long sales|dual|`test_review.py::test_opening_and_reduction_order`|捕获（1）|
|31|S19 unfilled plan forgotten|dual|`test_review.py::test_unfilled_liquidation_continues_without_maintenance_breach`|捕获（1）|
|32|split stale mark|dual|`test_review.py::test_suspended_action_preserves_wealth`|捕获（1）|
|33|dividend stale mark|dual|`test_review.py::test_suspended_action_preserves_wealth`|捕获（1）|
|34|cash account bankruptcy enabled|kernel|`test_review.py::test_cash_account_missing_mark_is_not_bankrupt`|捕获（1）|
|35|empty account bankrupt|dual|`test_review.py::test_empty_account_can_resume_after_deposit`|捕获（1）|
|36|initial margin repeats liquidation|dual|`test_review.py::test_liquidation_stops_at_filled_plan`|捕获（1）|
|37|partial sale minimum omitted|dual|`test_review.py::test_selling_minimum_and_nonzero_small_target`|捕获（1）|
|38|quantity dust tolerance omitted|dual|`test_review.py::test_split_dust_has_no_commission`|捕获（1）|
|39|forced day reducing row discarded|dual|`test_review.py::test_liquidation_day_honors_reducing_target`|捕获（1）|
|40|S15 prefix check bypassed|dual|`test_review.py::test_prefix_check_when_previous_fees_created_cash_deficit`|捕获（1）|

最终恢复的文件 SHA-256：

```json
{
  "crates/panelbt-core/src/lib.rs": "fbd8bde90b2bd5a1b7b97a877926089bee1e93ae02b745766605707d7d2cd434",
  "crates/tradecore/src/lib.rs": "abaec0ddcd5fa7ddf51512c1062fe32b1310408ede639b3896aae4d5be2a8d39",
  "packages/panelbt/python/panelbt/reference.py": "7e2f2527708171dc3a0b4fe82b242837a6856a77768c3d9fbdee98d22a136423"
}
```

## 独立安装

`uv build --package panelbt` 成功生成源码包与 cp313-abi3 macOS arm64 wheel，并从源码包构建 wheel。wheel/sdist 本机路径扫描为 0，15 条 RECORD 校验通过。新虚拟环境只装 panelbt 和 NumPy 2.1.0，以 Python 3.13.13 的 -I 模式执行中英文 README 各 3 段例子，均通过；包括港股式/美股式档案与参考实现核对。最低 NumPy 版本验证文件为 bench/results/review-install.json。wheel 不入库，dist 已忽略。

稳定 ABI 的上轮原生/abi3 成对性能数据保留在 bench/results/abi-*.json，返工没有更改 ABI 决策；这些历史性能数字不冒称本轮重新测量。当前 wheel 另在 Python 3.14 的全新环境运行 README，结果见 standalone-python314.json。

## 对比基线与范围

完整两种耗时口径、适用性与净值差额见 docs/BENCHMARKS.md；源数据 bench/results/compare.json，适配器分别位于 bench/compare。make bench / make bench-compare 只写 bench/results；make bench-docs 才生成文档。DESIGN 与返工起点逐字节相同。所有随机输入生成在计时外，含准备口径计各实现的输入校验/转换。

完整矩阵共 160 行：105 项实际计时、55 项不适用，无超时、无排除项。两种口径的原始样本、中位数、1000 组目标唯一性、近似批量的全部零费率控制、源码哈希以及生成表格的全部 160 行均已独立核对，结果见 comparison-audit.json。Backtrader 在 Python 3.13 直接运行成功，可选 vectorbt Rust 实际参与任务一和批量。

批量 T=1000、N=100、K=1000：panelbt 回测本身 56.287 ms、含准备 341.144 ms；近似 NumPy 分别 52.973 / 62.965 ms。结果组装已优化，但 Python 输入准备仍占明显时间，不能据执行口径声称整个调用更快。两种口径在独立忙机窗口测量，中位数倒置原样保留，不将其差值解释为输入准备耗时。

不提交、不推送、不发布；不读取密钥；未触碰指定并行任务目录。第一轮临时检出已删除。下一步仍需空闲机器重跑性能与真实券商/行情对账；独立成交账本和合成场景不能证明所有实盘微结构都可表达。
