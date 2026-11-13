# 资产配置（`ashare.allocation`）

几个到几十个大类资产（ETF、指数、期货连续合约、可转债组合）怎么分配权重。输出每个调仓日一行权重，
直接交给 `panelbt` 回测或 `ashare.report.tearsheet` 出报告。本文只讨论回测，不是投资建议。

```python
import numpy as np
import ashare.allocation as aa

returns = np.random.default_rng(0).normal(0.0003, 0.01, (1000, 4))
res = aa.allocate(returns, aa.RiskParity(), rebalance=21)
print(aa.tearsheet(res, returns=returns))
```

## 1. 方法

| 方法 | 说明 |
|---|---|
| `FixedWeights(weights, redistribute=False)` | 固定权重；配合 `allocate(..., band=)` 做再平衡带 |
| `EqualWeight()` | 可用资产等权 |
| `InverseVolatility()` | 权重 ∝ 1 / σ |
| `RiskParity(budget=None)` | 风险贡献 ∝ 预算（默认相等），见第 3 节 |
| `MinVariance(max_weight=None)` | 全额投资、只做多；经 `ashare.portfolio.optimize` |
| `MeanVariance(risk_aversion=5, expected="mean")` | 同上；预期收益默认用窗口均值（噪声很大，仅作对照），可传函数 |
| `TargetVolatility(base, target=0.10, max_leverage=1.0)` | 叠加层：按事前波动缩放 base 的权重，上限 `max_leverage`；不足 1 的部分是现金，超过 1 需要回测里允许融资。事前波动为 0 时不缩放、不加杠杆，原样用基础配置并在诊断里标 `fallback` |
| `TrendFilter(base, window=200, redistribute=False)` | 叠加层：最近收盘低于 window 日均线的资产置 0（或把权重分给其余资产） |

协方差统一由 `Covariance(half_life=63, shrinkage="auto", target="constant_correlation")` 估计，内部用
`ashare.risk.statistical.shrinkage_components`（指数加权 + 收缩），展开成稠密矩阵。不收缩、不加权时等于样本协方差（测试守住）。

`allocate(returns, method, *, dates, symbols, rebalance="monthly", lookback=252, min_history=60, covariance, band)`：

- 风险预算 `RiskParity(budget={代码: 份额})` 按代码对齐，有资产尚未上市时其余资产按各自预算重新归一。
- `rebalance`：`"monthly"`、`"quarterly"`、`"weekly"`、`"daily"`、正整数（每 N 行）或行号序列。
- `min_history`：资产在窗口内少于这么多有效收益就不参与（权重 0，诊断里有 `available`）。
- `band`：上次调仓后按收益漂移到 t−1 收盘的权重与新目标的最大偏离不超过 band 时，这天不下单（`traded=False`）。
- 返回 `Allocation`：`rows`、`dates`、`weights`、`traded`、`diagnostics`；`dense()` 给 T×N（不调仓的行为 NaN，即
  `panelbt.Targets.from_dense` 的「不下单」）；`targets()`；`to_dict()` / `to_json()` 只含内置类型；`frame()`。
- `returns_market(returns)`、`run(res, ...)` 把收益表包成一个以收盘成交、零费用、可分割份额的行情做快速回测；
  真实回测用 `fund_market` / `cb_market` 建的行情和费用。

## 2. 时点

调仓行 t 是**成交行**；第 t 个调仓日的权重只用 `returns[:t]`，即 t−1 日收盘及以前的收益。`returns` 第 r 行是
r−1 收盘到 r 收盘的收益。月度调仓 = 每月第一个交易日 = `ashare.backtest.timing` 里信号日（上月最后一个交易日）
对应的成交日（`test_schedule_matches_timing` 守住两边一致）。

通用前视测试：把调仓行 t 及以后的收益整段换成随机数，各方法在 t 的权重（或数据无关方法的下单标记）必须不变。

## 3. 风险平价求解

正式实现（`riskparity.solve_risk_parity`）对 Spinu 的凸目标 ½ yᵀΣy − bᵀ log y 用牛顿法，Armijo 线搜索（牛顿减量
≤ 1e-10 时跳过线搜索直接走整步）；减量 ≤ 1e-26 或 max |y·∇| ≤ tol 时停，再按风险贡献占比与预算的最大绝对差
（residual）≤ 1e-8 确认。诊断：

| 字段 | 含义 |
|---|---|
| `converged` | 在**原协方差**上解到了等风险贡献 |
| `solved_with_ridge` | 原协方差上没解到（无下界或迭代用完），加了 ridge 才解到（此时 `converged` 为 False、`status` 为 `ridge`） |
| `status` | `converged` / `ridge` / `max_iter`（原协方差上迭代用完、加 ridge 也没解到） / `unsolved` |
| `original_status` | 原协方差上那一次求解的结果：`converged` / `unbounded` / `max_iter` |
| `residual`、`residual_original` | 在实际求解的矩阵、在原协方差上的残差（加了 ridge 时后者可能很大） |
| `ridge`、`iterations` | 用到的 ridge（× 平均方差）与迭代次数 |

`allocate` 汇总每个调仓日的 `status` 到 `settings["risk_parity_status"]`，有不是 `converged` 的调仓日时发 `RuntimeWarning`。

参考实现（`reference.risk_parity_reference`）是纯 Python 的循环坐标下降，两者独立写成，12 组随机协方差（含高相关、
负相关、条件数很大的）上权重相差不超过 1e-5（参考实现收敛较慢，误差在它那边），正式实现的风险贡献与预算相差不超过 1e-10。
先试过 SLSQP 和 least_squares 作参考，它们在负相关时会停在某资产权重为 0 的假解上，所以换成坐标下降。

病态输入：

| 输入 | 行为 |
|---|---|
| 协方差奇异，但不存在方差为 0 的长仓组合 | 直接有解，不加 ridge |
| 存在方差为 0 的长仓组合（如两两完全对冲） | 等风险贡献无解；依次在对角线加 1e-10、1e-8、1e-6、1e-4 × 平均方差重解（小 ridge 下病态到迭代不够就换更大的），`status = "ridge"`、`converged = False` |
| 含对冲组合的近奇异矩阵（如第三只 = −(前两只之和)，对角线只有 1e-14 的余量） | 原协方差上没有严格的零方差组合，但病态到迭代用完（`original_status = "max_iter"`）；同样回退到 ridge，`status = "ridge"`、`solved_with_ridge = True`、`converged = False`，`residual_original` 照报原协方差上的残差（这个例子约 0.007） |
| `max_iter` 太小 | 原协方差上迭代用完后也试 ridge；ridge 也解不到时交回原协方差上那次的结果，`status = "max_iter"`、`ridge = 0`，权重和为 1，`allocate` 发警告 |
| 两只完全相同的资产 | 平分 |
| 预算为 0 的资产 | 权重 0，其余按预算解 |
| 窗口内价格从不变动（波动为 0） | 风险平价、倒波动把它排除（诊断 `zero_volatility`）；全部为 0 时退回等权（诊断 `fallback`） |
| 历史不足 `min_history` | 不参与，上市后自动加入 |
| 对角线有 0 或负数的协方差直接传给求解器 | 报错并说明 |

## 4. 性能基线

`bench/allocation/perf.py`：30 个资产、2500 个调仓日（每天调仓、回看 252 日），3 次取中位数，测时负载约 9–10。

| 步骤 | 耗时 |
|---|---|
| 协方差估计 2500 次 | 0.60 s（每次约 0.24 ms） |
| 风险平价求解 | 每次 0.14 ms |
| 整条 `allocate(RiskParity)` | 1.13 s（零波动识别改成向量化之前 1.70 s） |

瓶颈是协方差估计，已是 numpy 向量化；没有必要上 Rust。

## 5. 真实数据例子：四只 ETF

`bench/allocation/etf_example.py`：沪深 300 ETF（510300.SH）、国债 ETF（511010.SH）、黄金 ETF（518880.SH）、
豆粕 ETF（159985.SZ，2019 年底上市，2020-07 起满足历史要求）。月度调仓（月初开盘成交），回看 252 日、至少 120 日。
`panelbt` 回测，行情和费用来自 `fund_market`：佣金万分之一、无最低、滑点万分之五、100 份一手、涨跌停不能成交，初始 100 万元。

- `equal`：可用资产等权。
- `fixed_60_40`：60% 沪深 300 + 40% 国债。
- `risk_parity`：等风险贡献。
- `risk_parity_vol6`：风险平价 + 目标波动 6%，杠杆上限 2；融资年利率 3%（按 252 日折算），初始保证金 50%、维持 25%。
- `risk_parity_vol8_unlevered`：原定方案，目标波动 8%、不加杠杆。

结果写到 `<数据 root>/cn/reports/allocation/`，可以用 `python -m ashare.report show <文件>` 查看，`--from 2021-01-04`
看共同区间。夏普给两种：无风险利率取 0，和扣掉货币 ETF（511880.SH，共同区间年化 1.63%）日收益的超额收益夏普。
换手是年化双边（买卖合计 / 净值；此前版本的文档多乘了 2）。

修数据（停牌期间的拆分、重复分红、代码复用）没有影响这五只基金：修复前后它们的拆分、分红、复权因子、行情逐格相同，下面的数字与修复前一致（换手除外）。

全样本（2015-01-05 至 2026-11-12；第一次配置 2015-08-03）：

| 配置 | 年化收益 | 年化波动 | 夏普（无风险 0） | 超额夏普 | 最大回撤 | 年化双边换手 |
|---|---|---|---|---|---|---|
| 等权 | 7.55% | 8.35% | 0.91 | 0.64 | −11.13% | 44% |
| 60/40 | 3.06% | 11.75% | 0.32 | 0.12 | −24.30% | 34% |
| 风险平价 | 5.25% | 2.98% | 1.73 | 0.96 | −4.75% | 54% |
| 风险平价 + 目标波动 6%（2 倍杠杆，融资 3%） | 6.92% | 5.26% | 1.30 | 0.86 | −9.78% | 121% |
| 风险平价 + 目标波动 8%（不加杠杆） | 5.25% | 2.98% | 1.73 | 0.96 | −4.75% | 54% |

四只都可用的共同区间（2021-01-04 起）：

| 配置 | 年化收益 | 年化波动 | 夏普（无风险 0） | 超额夏普 | 最大回撤 | 年化双边换手 |
|---|---|---|---|---|---|---|
| 等权 | 6.91% | 8.74% | 0.81 | 0.62 | −11.13% | 35% |
| 60/40 | 0.33% | 10.81% | 0.08 | −0.06 | −24.30% | 26% |
| 风险平价 | 5.04% | 2.58% | 1.92 | 1.29 | −2.87% | 40% |
| 风险平价 + 目标波动 6% | 7.12% | 4.98% | 1.41 | 1.08 | −4.60% | 93% |
| 风险平价 + 目标波动 8%（不加杠杆） | 5.04% | 2.58% | 1.92 | 1.29 | −2.87% | 40% |

共同区间风险平价平均权重：国债 78%、黄金 9%、沪深 300 7%、豆粕 6%；加杠杆版平均总敞口 1.85 倍。
目标波动 8% 不加杠杆与纯风险平价逐位相同：风险平价的事前波动只有约 3%，按 8% 放大会被杠杆上限 1 截住。

分解（共同区间，都与上表的风险平价对照）：

| 配置 | 年化收益 | 年化波动 | 夏普（无风险 0） | 超额夏普 | 最大回撤 |
|---|---|---|---|---|---|
| 风险平价 | 5.04% | 2.58% | 1.92 | 1.29 | −2.87% |
| 国债仓位换成 0 收益现金 | 2.40% | 2.42% | 0.99 | 0.32 | −3.07% |
| 只用三类风险资产（股票、黄金、豆粕）的风险平价 | 9.77% | 11.07% | 0.90 | 0.75 | −12.63% |
| 目标波动 6% 版，融资年利率改为 6% | 4.09% | 4.98% | 0.83 | 0.50 | −5.26% |
| 风险平价，初始资金 2 万元 | 3.80% | 2.29% | 1.64 | 0.93 | −2.64% |

### 怎么看这些数字

- **风险平价的成绩主要来自国债仓位和利率下行。** 平均 78% 放在国债上，把这部分换成 0 收益现金，年化从 5.04% 降到 2.40%，
  超额夏普从 1.29 降到 0.32：一半多的收益是国债给的。2021 年以来中国国债收益率一路下行，久期资产有持续的资本利得；
  利率走平或上行时这部分不会重现。
- **「夏普 1.92」是无风险利率取 0 的口径**，低波动、以债券为主的组合被这个口径抬得最多；扣掉货币 ETF 后是 1.29，等权从 0.81 降到 0.62。
- **加杠杆版的优势依赖融资成本。** 融资利率 3% 时年化 7.12%；改成 6%（更接近券商两融的常见水平，未核实）时降到 4.09%、超额夏普 0.50，
  低于不加杠杆的风险平价。换手也高一倍多。四只 ETF 是否都是两融标的未核实。
- **小资金配不到位。** 2 万元初始资金时国债 ETF 一手约 1.3 万元，权重达不到，年化 3.80%。
- **只用风险资产**的风险平价年化 9.77%、波动 11%、回撤 −12.6%，夏普与等权相当，说明低波动来自国债，而不是方法本身。
- **黄金这段时间特别强**，等权的收益相当一部分来自黄金，资产选择本身带有后见之明；四只都是现在还存续的基金。
- **60/40 被股票拖累**：沪深 300 在 2015 年和 2021 年后两段大跌，是这段 A 股行情的结果，不是 60/40 方法本身的普遍结论。
- 起点不同：豆粕 ETF 2020 年才进组合，所以单列了共同区间。
- 只有一条样本路径、四只资产、十一年，没有做多重检验或样本外检验；这些数字只说明工具链能跑通、量级合理。

## 6. 测试

- 交叉验证：风险平价与坐标下降参考（12 组随机协方差）；最小方差、均值方差与 SLSQP 参考；调仓行与直白参考；协方差与手写指数加权公式。
- 性质：各方法权重非负、有限、合计 ≤ 1（目标波动可 > 1 但 ≤ 上限）；目标波动在事后能把实现波动拉到目标附近；倒波动比例；趋势过滤方向；
  再平衡带控制下单次数；单资产、晚上市、零波动、全零、错误输入的报错；风险预算按代码对齐（前面有资产未上市时）；
  趋势过滤用 t−1 日收盘；近奇异的对冲组合原协方差上迭代用完后回退到 ridge（报 `solved_with_ridge`、照报原协方差残差）；
  ridge 也解不到时报 `max_iter`、交回原协方差上的结果；加了 ridge 不报 `converged`；事前波动为 0 时不加杠杆；
  零波动识别与 `np.nanstd` 一致。
- 前视：第 2 节的通用测试，对每个方法参数化。
- 输出：`to_dict()` 只含内置类型、`to_json()` 往返；`tearsheet` 能读。

## 7. 未完成与已知限制

- 可转债组合作为配置资产还没有接（强赎退出已在 `cb_market` 里处理，见 `docs/FUND_CB_DATA.md`）。
- 预期收益只有窗口均值，没有接因子或 Black-Litterman。
- 期货连续合约作为资产需要自己先算收益表。

## 8. 变异检查

在 `src`、`test` 的副本里逐条改坏（先确认 `ashare.__file__` 指向副本、副本未改时目标测试全过），每条都要有测试失败。
M2、M12、M14、M16 是这次修掉的错误的原写法，M20–M23 对应复核时存活的四处变异（补了测试），M13、M15、M17–M19、M24、M25 守住这次的新规则。
M26–M36 是第二轮收尾加的：M26、M31 是上一轮的原写法，M32、M33 是第二轮复核时存活的两处（N1、N3，补了测试），其余守住这轮的新规则。

| 编号 | 改动 | 抓住它的测试 |
|---|---|---|
| M1 | 股票版本号不再排除基金数据集 | `test_stock_and_futures_identical_with_fund_partitions` |
| M2 | 停牌期间的因子比按相邻日历行算（修复前的写法） | `test_split_during_suspension_lands_on_resumption` |
| M3 | 转股价只按生效日 | `test_conversion_price_as_known` |
| M4 | 强赎标记从公告当天起 | `test_call_and_remaining_size` |
| M5 | 可转债成交量不换算 | `test_derived_fields` |
| M6 | 校验把未解释问题判通过 | `test_verify_passes_and_fails` |
| M7 | 调仓日用到当天收益（前视） | `test_min_and_mean_variance_match_reference` |
| M8 | 风险平价不做 ridge 重试 | `test_risk_parity_singular_covariance` |
| M9 | 倒波动用方差 | `test_inverse_volatility_proportions` |
| M10 | 目标波动不封顶杠杆 | `test_target_volatility_realized` |
| M11 | 趋势过滤方向反了 | `test_trend_filter_direction` |
| M12 | 分红去重键带上除息日（修复前的写法） | `test_duplicate_dividend_counted_once` |
| M13 | 因子没动也照记分红 | `test_dividend_rejected_when_factor_does_not_move` |
| M14 | 复用代码的旧段不拆列 | `test_reused_codes_resolved` |
| M15 | 沿用价仍算行情 | `test_stale_bar_is_not_a_quote` |
| M16 | 校验只看相邻两天都有行情的行（修复前的盲区） | `test_split_during_suspension_lands_on_resumption` |
| M17 | 没卖掉的可转债仍按最后收盘了结 | `test_cb_redemption_and_exit` |
| M18 | 强赎公告后仍可买入 | `test_cb_redemption_and_exit` |
| M19 | 清仓不插入卖出行 | `test_cb_redemption_and_exit` |
| M20 | 强赎标记取最后一次实施公告 | `test_call_flag_uses_first_implementation_notice` |
| M21 | 风险预算按位置对齐 | `test_risk_budget_aligned_by_symbol` |
| M22 | 趋势过滤用 t−2 收盘 | `test_trend_filter_uses_previous_close` |
| M23 | fund_version 混入期货数据集 | `test_stock_and_futures_identical_with_fund_partitions` |
| M24 | 加了 ridge 仍报 converged | `test_risk_parity_singular_covariance` |
| M25 | 事前波动为 0 时用满杠杆 | `test_target_volatility_zero_ex_ante_uses_base` |
| M26 | 复用代码只搬有 `!1` 因子、六位代码自己没有因子的那几天（上一轮的写法） | `test_reused_old_life_rows_with_own_factor_move_to_seven_digit` |
| M27 | 不修补深市 2020-02-25 收盘 | `test_fix_closes_backfills_from_next_pre_close` |
| M28 | `fund_market` 不排除分级基金 | `test_graded_funds_excluded_from_market_and_verify` |
| M29 | 重复行情检查不判失败 | `test_verify_rejects_duplicate_quotes` |
| M30 | 七位 / 六位代码重叠不判失败 | `test_duplicate_check_requires_zero_seven_six_overlap` |
| M31 | 原协方差上迭代用完不回退 ridge（上一轮的写法） | `test_risk_parity_near_singular_hedge_falls_back_to_ridge` |
| M32 | 首选除息日错时不找与因子对得上的候选日（N1） | `test_dividend_lands_on_candidate_matching_factor` |
| M33 | 赎回价从最后交易日当天起作估值（N3，前视） | `test_cb_redemption_mark_starts_after_last_trade` |
| M34 | 校验不排除分级基金 | `test_graded_funds_excluded_from_market_and_verify` |
| M35 | 记错代码的行情：存续期外偶尔巧合也去掉 | `test_misfiled_quotes_outside_life_dropped` |
| M36 | 不去掉记错代码的行情 | `test_misfiled_quotes_outside_life_dropped` |
