# 回测报告（`ashare.report`）口径说明

`build_tearsheet` 把一次 `panelbt` 回测整理成一个只含 JSON 基本类型的 dict；`render_html` 把它写成一个自包含、离线能打开的 HTML。本文是这份 dict 里每个量的定义，页面上显示的数都由它算出。

## 用法

```python
import ashare.data, ashare.report

result = panelbt.run(market, targets, costs, panelbt.Config(..., record_trades=True))
ashare.report.tearsheet(result, market, targets, dates, symbols,
                        data=ashare.data.open(), benchmark="000300.SH",
                        benchmark_kind="price", title="……", description="……",
                        rules="……", costs=costs, config=config).save("report.html")
```

- `dates`、`symbols` 是回测面板的行、列标签（`YYYYMMDD` 或 `YYYY-MM-DD`）。
- `data` 可省略；给了就从数据层取证券名称、行业（`industry` 加 `index_classify` 的申万一级名称）、行业换版日、停牌掩码和 `index_daily` 里的基准行情。
- 基准用指数代码时必须声明 `benchmark_kind`：`"price"`（价格指数，不含分红再投资）或 `"total_return"`（全收益）。数据层不记录指数是否含分红，所以由调用方说明，页面照写。
- 不经数据层时直接调 `build_tearsheet`，基准用 `Benchmark(values, name, kind)`，行业用 T×N 的码或名称矩阵加 `industry_names`，换版日用 `industry_switch` 显式传入。
- `costs`、`config` 只用来在页面脚注里写明费用设定，不参与计算。

## 前提与报错

报告只对下面这类回测成立；不满足时抛 `ReportError`，消息是中文说明，不会给出错的数：

| 情形 | 处理 |
|---|---|
| 没开 `record_trades`（没有持仓矩阵） | 报错：持仓、行业权重、未成交都需要它 |
| 批量回测结果（二维净值） | 报错：先取出单个策略 |
| 第一行已有成交或费用（`nav[0] ≠ 1`） | 报错：第一行是基准日，请往前多留一个交易日 |
| 做空（负目标权重、负持仓或空头市值） | 报错：持仓、行业权重和换手的口径只适用于纯多头 |
| 期货腿（`Market.futures`） | 报错：保证金和名义敞口下的换手、权重口径没有定义 |
| 资金进出（`cash_flow` 非零） | 报错：净值按「权益 ÷ 初始资金」，资金进出会让它失真 |
| 净值出现非正数或非有限值 | 报错 |
| 日期与结果、行情、行业、停牌掩码的形状对不上，日期不递增 | 报错 |
| 基准末尾缺失、`kind` 不是两种之一 | 报错 |

能正确处理的：没有基准（所有基准相关量为 `null`，页面不画基准）、基准晚于策略起步（见下）、历史不足一年（滚动指标为空）、每日调仓、持仓只数变化很大、空仓期、没有行业或名称信息。

## 通用约定

- 第一行是**基准日**：净值 1、尚未持仓，日收益记 0。「全区间」指去掉基准日后的 n 个交易日。
- 收益、权重、回撤都是小数（0.05 = 5%）；金额单位与回测相同（元）。
- 一年按 `periods_per_year`（默认 252）个交易日；年数 = n / 252。
- 日收益 `r_t` 取引擎的 `result.returns`（无资金进出时等于 `nav_t / nav_{t-1} − 1`）；`nav_t = result.nav_t / result.nav_0`。
- 基准净值 `B_t = 基准值_t / 基准值_s`，s 是基准第一个有效日；`b_t = B_t / B_{t-1} − 1`。中间缺的天按前值填充并在 `meta.benchmark.filled_days` 计数，页面脚注写明。
- 无风险利率默认 0；带 `_rf2` 的按 2%/年（每日 2%/252）。
- 样本不足或标准差为 0 时指标为 `null`，不是 0。

## summary

| 字段 | 公式 |
|---|---|
| `cumulative_return` | `nav_末 − 1` |
| `annual_return` | `∏(1+r)^(252/n) − 1`（几何） |
| `annual_volatility` | `std(r, ddof=1) × √252` |
| `sharpe`、`sharpe_rf2` | `mean(r − rf/252) / std(r, ddof=1) × √252` |
| `sortino`、`sortino_rf2` | `mean(r − rf/252) × √252 / sqrt(mean(min(r − rf/252, 0)²))`，下行偏差按全部日子求均值 |
| `max_drawdown` 及 `_peak`、`_trough`、`_recovery` | 最深一次回撤段的深度（≤ 0）、峰值日、谷底日、恢复日（未恢复为 `null`） |
| `calmar` | `annual_return / |max_drawdown|`；没有回撤时为 `null` |
| `monthly_win_rate` | 月收益 > 0 的月份占比 |
| `annual_turnover` | `Σ daily.turnover / 年数`（双边） |
| `annual_turnover_one_sided` | 上式 / 2 |
| `total_fees`、`total_fees_to_initial` | 佣金 + 印花税 + 其他费用合计，及其 ÷ 初始资金 |
| `total_slippage`、`total_slippage_to_initial` | 引擎的滑点成本 + 冲击成本合计（已含在成交价里，单列），及其 ÷ 初始资金 |
| `total_commission`、`total_stamp_tax` | 分项 |
| `total_dividend_gross`、`total_dividend_tax` | 税前现金分红、红利税合计 |
| `rebalances` | 调仓计划的行数（`targets.days` 的个数） |

基准与相对指标（无基准时全为 `null`）。基准晚于策略起步时，这一组都在**重叠区间**（基准第一个有效日 s 之后）上算，`benchmark_start` 记下 s：

| 字段 | 公式 |
|---|---|
| `benchmark_cumulative_return` | `B_末 − 1` |
| `benchmark_annual_return`、`_volatility`、`_sharpe`、`_max_drawdown` | 同策略口径，用 b |
| `excess_annual_return` | 策略年化 − 基准年化（算术差，重叠区间） |
| `excess_annual_return_geometric` | `(1+策略年化)/(1+基准年化) − 1` |
| `tracking_error` | `std(r − b, ddof=1) × √252` |
| `information_ratio` | `mean(r − b) × 252 / tracking_error` |
| `beta` | `cov(r, b) / var(b)`（样本协方差与方差） |
| `monthly_win_rate_vs_benchmark` | 月收益 > 基准月收益的月份占比（只数两边都有的月份） |

## daily（列式，长度 T）

`date`、`nav`、`benchmark_nav`、`return`、`benchmark_return`、`drawdown`（`nav_t / max_{s≤t} nav_s − 1`）、`benchmark_drawdown`、`holdings`（收盘持股数 > 0 的证券只数）、`cash_weight`（现金 / 权益）、`turnover`（**双边**：引擎的 `result.turnover` 是单边，乘 2；即当日成交额 ÷ 交易前权益）。

## monthly、yearly

- 月收益 = 月末净值 / 上月末净值 − 1，第一个月以基准日为起点。
- 年收益 = 年末净值 / 上年末净值 − 1；`volatility`、`sharpe` 用该年日收益；`max_drawdown` 以年初为起点。基准那一年没有完整数据时为 `null`。
- 完整性用传入的交易日序列判断，不按日历日期猜：
  - 首月、首年：第一行是基准日（首个交易日的前一个交易日）。基准日与首个交易日同月（同年）时，`monthly.first_partial`（第一年的 `partial`）为真。
  - 末月、末年：区间最后一天之后、同月（同年）内还有交易日时，`monthly.last_partial`（最后一年的 `partial`）为真。交易日历由 `calendar` 传入（`tearsheet(data=...)` 取数据层的开市日历，含当年余下的未来日期）；日历覆盖不到的部分按「之后同月 / 同年是否还有工作日（周一到周五）」判断，节假日会被当成交易日，所以可能把其实已完整的月份标成不完整。`meta.completeness_basis` 记下用的是哪种依据（`calendar`、`calendar+weekday`、`weekday`）。
  - 页面在不完整的月份格子上加星号，并在脚注说明。

## drawdowns

把净值序列切成不重叠的回撤段：从一个峰值开始，到净值重新 **≥** 该峰值的那天结束；同一段只取最低点。按深度取前 5 段（`top_drawdowns`）。字段：`peak`、`trough`、`recovery`、`depth`（≤ 0）、`days_to_trough`（峰到谷的**交易日**数）、`duration_days`（峰到恢复的交易日数；未恢复的算到最后一天）、`recovered`。

## rolling

窗口 `rolling.window`（默认 252）个日收益，从第 252 行起每天一条（窗口恰好是第 1 到 252 个真实日收益）：`annual_return = ∏(1+r) − 1`（窗口一年，即滚动一年收益）、`volatility = std × √252`、`sharpe = mean / std × √252`、`beta`（窗口内含缺失基准时为 `null`；无基准时整列为 `null`）。历史不足一个窗口时各列为空。

## rebalances（每个调仓行一条）

| 字段 | 含义 |
|---|---|
| `date` | 调仓行（成交日） |
| `targets` | 目标里权重 > 0 的只数 |
| `buys` / `sells` | 当天有买入 / 卖出成交的只数（含留仓股票的加减仓） |
| `new_names` / `dropped_names` | 与调仓前一日收盘持仓比，新进 / 剔除的只数 |
| `turnover` | 当天双边换手；`window_turnover` 是到下次调仓前（含重试日）的合计 |
| `fees`、`slippage` | 当天到下次调仓前一天的费用、滑点合计（含重试成交） |
| `unfilled`、`unfilled_by_reason`、`unfilled_detail` | 见下 |
| `unfilled_filled_before_next_rebalance` | 上述未成交的股票里，下次调仓前补成交了的只数 |
| `blocked_buy_days` / `blocked_sell_days` | 引擎的 `n_blocked_buy` / `n_blocked_sell` 在调仓日到下次调仓前一天的合计 |
| `holdings_after`、`cash_weight_after` | 成交日收盘持仓只数、现金占比 |

**未成交的计数口径**：`unfilled` 按**只数**计，只看两类——新进股票（目标里有、调仓前没持有）在调仓日开盘不可买，剔除股票（调仓前持有、目标里没有）在调仓日开盘不可卖。「可买、可卖」就是回测用的 `market.can_buy` / `can_sell`。原因：停牌掩码为真或执行价缺失记 `suspended`，否则买入记 `limit_up`、卖出记 `limit_down`（不给停牌掩码时只按执行价缺失判断停牌）。留仓股票加减仓被挡不拆原因，只体现在 `blocked_*_days` 里。

页面上，平均调仓间隔不少于 12 个交易日（如月度）时，每次调仓画一组按原因分色的圆点；更密（周度、日度）时改成按月汇总的堆叠小柱（按月太窄时按季），高度是只数，悬停时读数给出该月各原因的只数。

引擎的「被挡」是**按天重试计数**的：一只股票连续 5 天卖不出去记 5 次，所以 `blocked_*_days` 不是笔数，也不能和 `unfilled` 直接比。页面上的未成交圆点和「停牌 / 涨停 / 跌停」计数用的是 `unfilled_by_reason`（只数）。

## industry_weights

每次调仓成交日收盘后的行业权重：权重 = 持股数 × 收盘价 / 权益，停牌或无行情的股票用最近一个收盘价，股票部分按比例缩放到引擎的 `long_value`，保证每列 `Σ 行业 + cash = 1`。调仓行超过 260 个（例如每日、每周调仓的长回测）时改按每月最后一个交易日取样，`sampling` 记为 `month_end`，页面脚注写明。

行业标签：0、空串、NaN 记为「未分类」；码经 `industry_names` 映射成名称。数据层的行业是时点口径，2021-08-02 起由申万 2014 换成申万 2021；`tearsheet(data=...)` 用面板的 `industry_source`（2 = 申万 2021 当时口径）第一次出现的日子作换版日，不从行业名去猜。直接调 `build_tearsheet` 时由调用方传 `industry_switch`，不传就不画那条线。

## holdings_latest、attribution

- 最后一天收盘持仓，按权重降序：`symbol`、`name`、`weight`、`shares`、`industry`。
- 年度归因（每年除以上年末权益）：`dividend_net`（税后分红）、`fees`（−费用）、`slippage`（−滑点与冲击）、`price`（权益变化 − 税后分红 + 费用 + 滑点，含现金利息），`total = price + dividend_net + fees + slippage`；`dividend_tax` 仅供参考、已含在 `dividend_net` 里。

## 净值图上的回撤标注

最深三次回撤直接标在净值轨道上：只在「峰到谷」这一段铺淡底；从峰到修复（未修复的到区间末日、末端带箭头）用标注下方一条细括线表示，括线两端有短竖线表示该端点在视图内。所选区间的起点处于回撤中时，左端不画竖线、淡底从区间起点铺到谷（谷已在区间之前则不铺）。标注（深度、见底和修复天数）贴在括线上方，与其他标注相交时依次试右对齐、居中，仍放不下就只写深度，再放不下就不标。

## 页面上的区间重算

切换「近 N 年」或选中一次回撤时，右侧指标在浏览器里按所选区间的日收益重算，公式与上面相同（`n` 为区间交易日数；区间不足 240 个交易日时显示区间收益而不是年化）。全区间直接显示 `summary` 里的数。区间的费用、滑点 = 区间内各调仓行的 `fees`、`slippage` 合计 ÷ 区间起点权益。

## HTML

- 完全自包含：原生 JS 加内联 SVG，系统字体栈，不引用任何外部网络资源；数据以列式 JSON 内嵌，按显示精度取整（净值、日收益 6 位小数，回撤、现金占比、换手 5 位），`summary` 原样嵌入。
- 打印（`@media print`）：A4 纵向单栏；指标和图都固定在全区间，读数取区间末日；交互控件隐藏；每张图不跨页切断。
- 性能（2026-11-13，本机，低波策略 11.9 年日频）：HTML 464 KB；`build_tearsheet` 约 0.03 s、`render_html` 约 0.01 s，`tearsheet(data=...)` 另花约 0.15 s 从数据层取名称、行业、停牌和基准；首屏脚本执行约 55 ms，切换区间约 4 ms，拖动十字线每次约 0.2 ms（均不含浏览器绘制）。全市场每周调仓 100 只（7.6 年、407 次调仓）的报告 469 KB。
