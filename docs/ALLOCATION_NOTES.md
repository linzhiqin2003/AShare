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
| `TargetVolatility(base, target=0.10, max_leverage=1.0)` | 叠加层：按事前波动缩放 base 的权重，上限 `max_leverage`；不足 1 的部分是现金，超过 1 需要回测里允许融资 |
| `TrendFilter(base, window=200, redistribute=False)` | 叠加层：最近收盘低于 window 日均线的资产置 0（或把权重分给其余资产） |

协方差统一由 `Covariance(half_life=63, shrinkage="auto", target="constant_correlation")` 估计，内部用
`ashare.risk.statistical.shrinkage_components`（指数加权 + 收缩），展开成稠密矩阵。不收缩、不加权时等于样本协方差（测试守住）。

`allocate(returns, method, *, dates, symbols, rebalance="monthly", lookback=252, min_history=60, covariance, band)`：

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
（residual）≤ 1e-8 确认收敛。诊断给出 `iterations`、`converged`、`residual`、`ridge`。

参考实现（`reference.risk_parity_reference`）是纯 Python 的循环坐标下降，两者独立写成，12 组随机协方差（含高相关、
负相关、条件数很大的）上权重相差不超过 1e-5（参考实现收敛较慢，误差在它那边），正式实现的风险贡献与预算相差不超过 1e-10。先试过 SLSQP 和 least_squares 作参考，它们在负相关时会停在某资产权重为 0 的假解上，所以换成坐标下降。

病态输入：

| 输入 | 行为 |
|---|---|
| 协方差奇异，但不存在方差为 0 的长仓组合 | 直接有解，不加 ridge |
| 存在方差为 0 的长仓组合（如两两完全对冲） | 等风险贡献无解；依次在对角线加 1e-10、1e-8、1e-6、1e-4 × 平均方差重解，诊断里记下用了多大的 ridge，residual 按加了 ridge 的矩阵算 |
| 两只完全相同的资产 | 平分 |
| 预算为 0 的资产 | 权重 0，其余按预算解 |
| 窗口内价格从不变动（波动为 0） | 风险平价、倒波动把它排除（诊断 `zero_volatility`）；全部为 0 时退回等权（诊断 `fallback`） |
| 历史不足 `min_history` | 不参与，上市后自动加入 |
| 对角线有 0 或负数的协方差直接传给求解器 | 报错并说明 |

## 4. 性能基线

`bench/allocation/perf.py`：30 个资产、2500 个调仓日（每天调仓、回看 252 日），3 次取中位数，测时负载约 5.4。

| 步骤 | 耗时 |
|---|---|
| 协方差估计 2500 次 | 0.61 s（每次约 0.24 ms） |
| 风险平价求解 | 每次 0.135 ms |
| 整条 `allocate(RiskParity)` | 1.70 s |

瓶颈是协方差估计，已是 numpy 向量化；没有必要上 Rust。

## 5. 真实数据例子：四只 ETF

`bench/allocation/etf_example.py`：沪深 300 ETF（510300.SH）、国债 ETF（511010.SH）、黄金 ETF（518880.SH）、
豆粕 ETF（159985.SZ，2019 年底上市，2020-07 起满足历史要求）。月度调仓（月初开盘成交），回看 252 日、至少 120 日。
`panelbt` 回测，行情和费用来自 `fund_market`：佣金万分之一、无最低、滑点万分之五、100 份一手、涨跌停不能成交。

- `equal`：可用资产等权。
- `fixed_60_40`：60% 沪深 300 + 40% 国债。
- `risk_parity`：等风险贡献。
- `risk_parity_vol6`：风险平价 + 目标波动 6%，杠杆上限 2；融资年利率 3%（按 252 日折算），初始保证金 50%、
  维持 25%。

结果写到 `<数据 root>/cn/reports/allocation/`，可以用 `python -m ashare.report show <文件>` 查看，`--from 2021-01-04`
看共同区间。

全样本（2015-01-05 至 2026-11-12；第一次配置 2015-08-03）：

| 配置 | 年化收益 | 年化波动 | 夏普 | 最大回撤 | 年化双边换手 |
|---|---|---|---|---|---|
| 等权 | 7.55% | 8.35% | 0.91 | −11.13% | 88% |
| 60/40 | 3.06% | 11.75% | 0.32 | −24.30% | 67% |
| 风险平价 | 5.25% | 2.98% | 1.73 | −4.75% | 108% |
| 风险平价 + 目标波动 6% | 6.92% | 5.26% | 1.30 | −9.78% | 242% |

四只都可用的共同区间（2021-01-04 起）：

| 配置 | 年化收益 | 年化波动 | 夏普 | 最大回撤 |
|---|---|---|---|---|
| 等权 | 6.91% | 8.74% | 0.81 | −11.13% |
| 60/40 | 0.33% | 10.81% | 0.08 | −24.30% |
| 风险平价 | 5.04% | 2.58% | 1.92 | −2.87% |
| 风险平价 + 目标波动 6% | 7.12% | 4.98% | 1.41 | −4.60% |

风险平价平均权重：国债 75%、黄金 12.5%、沪深 300 8%、豆粕 4%；加杠杆版平均总敞口 1.85 倍。

### 怎么看这些数字

- **风险平价的高夏普主要来自国债。** 2015–2026 年中国国债收益率一路下行，久期资产有持续的资本利得；风险平价把 75%
  放在国债上，等于押注了这段利率下行。利率走平或上行时，这部分收益不会重现。
- **黄金这段时间特别强。** 等权组合的收益有相当一部分来自黄金（2019 年后的大涨），样本期内的资产选择本身就带有后见之明。
- **60/40 被股票拖累。** 沪深 300 在 2015 年和 2021 年后两段大跌，60% 股票的组合回撤和波动都最大；
  这是这段 A 股行情的结果，不是 60/40 方法本身的普遍结论。
- **目标波动最初没有效果。** 原定目标 8%、不加杠杆：风险平价的事前波动只有约 3%，按 8% 放大会被杠杆上限 1 截住，结果与
  纯风险平价完全相同。改成 6%、允许 2 倍杠杆后才有区别；收益的提高依赖 3% 的融资利率假设，换手也高一倍多。
- 起点不同：豆粕 ETF 2020 年才进组合，全样本的四种配置在前五年其实只有三只资产，所以单列了共同区间。
- 只有一条样本路径、四只资产、十一年，没有做多重检验或样本外检验；这些数字只说明工具链可以跑通、量级合理。

## 6. 测试

- 交叉验证：风险平价与坐标下降参考（12 组随机协方差）；最小方差、均值方差与 SLSQP 参考；调仓行与直白参考；协方差与手写指数加权公式。
- 性质：各方法权重非负、有限、合计 ≤ 1（目标波动可 > 1 但 ≤ 上限）；目标波动在事后能把实现波动拉到目标附近；倒波动比例；趋势过滤方向；
  再平衡带控制下单次数；单资产、晚上市、零波动、全零、错误输入的报错。
- 前视：第 2 节的通用测试，对每个方法参数化。
- 输出：`to_dict()` 只含内置类型、`to_json()` 往返；`tearsheet` 能读。

## 7. 未完成与已知限制

- 可转债组合作为配置资产还没有接（需先解决强赎退出，见 `docs/FUND_CB_DATA.md`）。
- 预期收益只有窗口均值，没有接因子或 Black-Litterman。
- 期货连续合约作为资产需要自己先算收益表。

## 8. 变异检查

在 `src`、`test` 的副本里逐条改坏（先确认 `ashare.__file__` 指向副本、副本未改时目标测试全过），每条都有测试失败：

| 编号 | 改动 | 抓住它的测试 |
|---|---|---|
| M1 | 股票版本号不再排除基金数据集 | `test_fund_isolation` |
| M2 | 拆分倍数取倒数（D / G） | `test_fund_panel_matches_reference` |
| M3 | 转股价只按生效日（不等公告） | `test_conversion_price_as_known` |
| M4 | 强赎标记从公告当天起 | `test_call_and_remaining_size` |
| M5 | 可转债成交量不从手换算成张 | `test_derived_fields` |
| M6 | 校验把未解释的问题判为通过 | `test_verify_passes_and_fails` |
| M7 | 调仓日用到当天收益（前视） | `test_no_lookahead`（8 个方法）、`test_min_and_mean_variance_match_reference`、`test_inverse_volatility_proportions` |
| M8 | 风险平价不做 ridge 重试 | `test_risk_parity_singular_covariance` |
| M9 | 倒波动用方差 | `test_inverse_volatility_proportions` |
| M10 | 目标波动不封顶杠杆 | `test_target_volatility_realized` |
| M11 | 趋势过滤方向反了 | `test_trend_filter_direction` |
