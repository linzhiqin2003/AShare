# 1b 审查返工逐项记录

原始复现脚本的运行输出保存在忽略目录 `.cache/rework/`，不作为仓库产物。下面列文件名均相对于仓库。最终计数、计时和检查结果见 `VERIFICATION.md` 及 `bench/results/review-*.json`。

|审查项|处理|实现与守护测试|
|---|---|---|
|P0-1|已修|`panelbt-core` 现金账户编译期分支不判爆仓；保证金无历史收盘价使用有效成交价，估值仍未知则延后判定。`test_cash_account_missing_mark_is_not_bankrupt`、`test_margin_missing_mark_uses_execution_price`，两个实现分别跑。|
|P0-2|已修|公司行为调整估值价，重新建仓重建调整后的历史价；参考实现独立更新，`test_properties` 影子账本同步；`test_suspended_action_preserves_wealth`、`test_short_action_does_not_force_other_long`。|
|P1-1|已修|恢复买入分支的零估值诊断条件；`test_zero_mark_reduction_does_not_count_blocked_buy`，20000 组摘要回归。|
|P1-2|已修|保存试算的逐证券减仓量；持续追保只看计划未成交，数量和维持保证金比较带容差；`test_liquidation_stops_at_filled_plan`、`test_unfilled_liquidation_continues_without_maintenance_breach`。|
|P1-3|已修|空仓且无负债不爆仓；语料现金流改为初始资金 +3% / −5%；`test_empty_account_can_resume_after_deposit`；覆盖分别统计 `bankruptcy` 和 `genuine_bankruptcy`。|
|P1-4|已修|reference 不导入 `_prepare`，独立展开费用、申报单位、账户率和 classic 条件；`test_reference_does_not_call_prepare`；利息、反手共用量、开/减仓顺序和持续追保均有手算。|
|P1-5|已修|测试覆盖 JSON 只写 `tmp_path`；正式覆盖产物由独立审计脚本输出；干净目录检查命令见验证记录。|
|P1-6|已修|`tradecore::round_close(q, position, minimum, step)`；成交量截断后重新校验下限；保留同方向非零低目标，防止误清仓；`test_selling_minimum_and_nonzero_small_target`、`test_volume_cap_cannot_create_subminimum_partial_sale`。|
|P1-7|已修|目标差额相对容差、开仓零头容差；`test_split_dust_has_no_commission` 的 shares / weights 两种模式。|
|P1-8|5% 判据通过；2 ms 未达|现金账户编译期专门化、零公司行为快速跳过、标量校验减少冗余数组扫描；三场景配对脚本 `bench/review_perf.py`，阶段剖析 `bench/review_profile.py`。是否满足 5% / 2 ms 见实测表，不以本轮自身基线代替第一轮。|
|P1-9|已修|Rust 库名 `panelbt_ext`、PyO3 名 `_core`，Python 仍为 `panelbt._core`；清理旧 `target/{release,debug}/lib_core.*`。|
|P1-10|已修|用 faec31e 真实构建与自己的生成器生成 20000 组 SHA-256；原始与修正摘要并存，不删旧 30 组原始样本；差异逐种子、逐字段、逐日压缩保存。|
|P2-1|已修|删除无效 `suppress` 状态；强平清空 pending，只有新调仓行能重建计划。|
|P2-2|已修（补触发用例）|保留检查并补触发用例 `test_prefix_check_when_previous_fees_created_cash_deficit`：前日融券费导致现金 −5，第一笔开空只收 1，后续开空收 10；整篮可行而第一笔不可行。第 40 处双边变异守护。|
|P2-3|已修|强平日保留降低风险的目标，增加风险部分作废；`test_liquidation_day_honors_reducing_target`。|
|P2-4|已修|并入 P1-6，非零小目标不再当成清仓。|
|P2-5|已修|新增逐日 `n_reduction_clipped` 和覆盖计数；`test_reduction_clipped_and_invalid_quote_counted`。|
|P2-6|已修|新增逐日 `n_invalid_quote` 和覆盖计数，非正报价不进入现金约束试算；同上。|
|P2-7|已修|利息覆盖使用货币容差，裁剪计数使用数量容差；`test_coverage_is_stable_at_tiny_cash_residuals` 明确覆盖 1048、8257。|
|P2-8|已修（减少 Python 开销）|`_batch_result(record=False)` 一次组装连续日序列，移除 K 个中间 Result 对象；标量输入校验减少冗余扫描。保留 NumPy 校验与安全绑定，未以未经测量的 Rust 重写替代。分段实测见自家基准。|
|P2-9|已修|适配器分开 prepare/execute；两种口径均重复实测，全部净值核对；批量行情一次初始化、逐计划转换和预备输入执行分开。边界写在 BENCHMARKS。|
|P2-10|已修|`make bench`、`make bench-compare` 仅写 `bench/results`；`make bench-docs` 才更新 BENCHMARKS。DESIGN 与返工起点逐字节相同。|
|P2-11|已修|包内全部中文异常信息改英文，`rg` 检查 Python 源文件无中文。|
|P2-12|已修|Costs `kw_only=True`；原有位置参数测试改成同值关键字参数。|
|P2-13|已修|NumPy 下限 2.1；对比依赖保留单独组，dev 不安装该组，`--inexact` 保留已安装可选依赖；pandas 测试 `importorskip`。最低版本独立安装验证见记录。|
|P2-14|已修|build.rs 编入绑定/内核/tradecore 全部 Rust 源 SHA-256；conftest 比对实际导入扩展，不符提示 make dev；变异构建和每次恢复均校验。|
|P2-15|已修|删除 docs 下原始构建/运行/变异日志与源码副本，只留验证和小型汇总；运行日志留忽略目录。|
|P2-16|已修|A 股历史必填覆盖字段进入 FeeSchedule 数据 `required_overrides_before`，通用 MarketProfile 不再比较 ASHARE 对象身份。|
|P2-17|已修|summarize 返回破产/几何指标有效标记；负权益时不可定义的几何年化、Calmar 为 NaN，算术风险和回撤可用；`test_metrics_on_negative_equity`。|
|P2-18|不修，保留并说明口径|分红总额是有符号净额，逐证券现金流可重建；显式非零维持率使用保证金账户；现金不足不补单保持旧规则。SPEC 明确三点，避免把这些既定口径当成静默缺陷。|

## 兼容性要求冲突

“其余种子逐位一致”不能同时覆盖本轮明确要求修的第一轮申报下限错误和零头单错误。逐种子的差异明细没有入库（可用 `bench/review_legacy_build.py` 重新生成）；修正后每组输出的摘要（SHA-256 前 8 字节）在 `test/fixtures/legacy_sha256.npz`，第一轮的摘要可由 `faec31e` 重新生成。另真实构建“faec31e 只改公司行为估值”版本，分离公司行为影响与数量规则影响，汇总在 `legacy-summary.json`。除停牌持仓外，缺成交价时的盘中估值、同日退市回收、重新建仓重建历史价也必须调整；否则会再次使用除权前估值。所有 20000 组修正输出均经独立成交账本复核。

## 复现边界与审查中的数值更正

修改前已实际运行两个 repro 和 probes1–5；缓存日志保留原始输出。P0、P1 的计数/追保/空账户/申报/零头问题，P2 的强平日目标、无诊断截断和指标异常均复现。打包命名、依赖、无效状态、中文文本、样本数量和基准副作用通过读取相应源码及构建产物确认；干净目录失败的根因是测试写不存在的 .cache 路径，修复后以实际新目录验证，不宣称执行过修改前的第二次干净构建。

审查 probes2 D2 的文字预计融资利息为 5，但其原始实际输出是 4.285714。150 股多头与 60 股空头在 0.5/0.5 保证金下需要 1050，超过 1000 权益，先按 20/21 缩减后自有现金为 −3000/7，利息为 30/7。手算测试同时保留该原输入和无须缩减的 120/−60 输入（利息恰好为 2）。P2-2 的“等价检查”结论未成立：已构造前日融券费产生欠款的反例，并通过双边变异确认该检查确实阻止不合规的第一笔开仓。

## 依赖下限

NumPy 2.1 开始支持 Python 3.13，见 [官方 2.1 发布说明](https://numpy.org/doc/2.1/release/2.1.0-notes.html)。本包使用的 NumPy API 不需要 2.5；实际最低版本安装与 README 试验记录在验证文件。
