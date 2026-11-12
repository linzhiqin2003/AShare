# 回测报告（`ashare.report`）口径说明

`build_tearsheet` 把一次 `panelbt` 回测整理成一个只含 JSON 基本类型的 dict；`render_html` 把它写成一个自包含、离线能打开的 HTML；`Tearsheet.text()` 和 `python -m ashare.report show` 把它排成终端里看的文本（见「文本版与命令行」）。本文是这份 dict 里每个量的定义，页面和文本版上显示的数都由它算出。

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

## 文本版与命令行

给在终端里工作的人和 AI 代理看：跑完回测存一份报告数据，之后随时在终端里看，不用翻 JSON。

```python
ts = ashare.report.tearsheet(result, market, targets, dates, symbols, data=ashare.data.open(),
                             benchmark="000300.SH", benchmark_kind="price")
print(ts)                                    # 默认摘要，40 行以内
print(ts.text(["yearly", "warnings"]))       # 按小节取
print(ts.text("all", fmt="markdown"))        # 全部小节，Markdown 表格
ts.text("summary", fmt="json", lang="en")    # 所选小节的原始数值
ts.text(start="2020-01-01", end="2023-12-31")  # 只看一段，指标按页面公式重算
ts.save_json("report.json")                  # 存一次；Tearsheet.load("report.json") 读回
```

```
python -m ashare.report show report.json                         # 默认摘要
python -m ashare.report show report.json -s yearly -s warnings   # 也可写 -s yearly,warnings
python -m ashare.report show report.json -s all -f markdown
python -m ashare.report show report.json --from 2020-01-01 --to 2023-12-31
python -m ashare.report show report.json -f json --lang en
python -m ashare.report html report.json -o report.html          # 与 Tearsheet.save 写出的页面相同
```

小节：`summary`（默认）、`yearly`、`monthly`、`drawdowns`、`rolling`（每季最后一个交易日一行）、`rebalances`（汇总加最近 10 次）、`holdings`、`industry`、`costs`（费率设定、合计和分年归因）、`benchmarks`、`winrate`、`contributions`、`tails`、`concentration`、`warnings`，`all` 为全部。

约定：

- 每个数带单位或百分号，正收益写 `+`；日期用 ISO 格式；缺失写 `-`，不写 0；负号是 ASCII 的 `-`。
- 默认中文表头；`--lang en` 把表头和标签换成报告数据里的 JSON 键名（`annual_return`、`benchmark_return` 等），方便程序对照。提醒的正文仍是中文。
- 纯文本不用 ANSI 颜色、不用制表线字符，不读终端宽度；行宽不超过 100 列（中文按两列算）。列多的表先把列间距压到一个空格，仍放不下就保留第一列、其余列拆成几张表；长句按宽度折行，续行缩进两格。
- 默认摘要不超过 40 行：区间、基准，策略与基准的累计、年化、波动、夏普、最大回撤及起止日，年化超额、信息比率、跟踪误差、Beta，年化换手、费用和滑点占期初权益，分年收益与超额，最深三次回撤，提醒的代码。历史很长时分年表只留最近若干年，标题里写明更早的年数。
- 数字都取自 `build_tearsheet` 的输出，格式与页面相同（收益和波动 1 位小数、夏普等 2 位、换手 0 位、费用 2 位；取整规则照 JS 的 `toFixed`：对二进制精确值四舍五入，恰好一半时取绝对值大的一侧）。
- `fmt="json"` 输出 `{"range": …, 小节: …}`，数值不取整、不加单位，键名与报告数据一致。

### 自选区间（`--from/--to`）

与页面「切换区间」同一套公式（`window.metrics` 逐行照搬页面的 `metrics(a, b)`）：输入是嵌入页面的那份取整后的日度序列（`render.compact`），求和顺序相同；区间的基准日 a 是起始日之前的最后一个交易日，b 是不晚于截止日的最后一个交易日；全区间时与页面一样直接取 `summary`，所以逐位一致。区间不足 240 个交易日时与页面一样只给区间收益和区间超额，不给年化。测试里用 node 运行页面模板里的那段 JS，与 Python 版逐项比对（相对误差 ≤ 1e-13，只可能差在 `pow` 的最后一位）；另外核对公式路径（不取 `summary`）在全区间上与 `summary` 的显示值完全相同，数值差只来自嵌入时的取整（≤ 2e-6）。

自选区间时：分年表里被区间截断的年份按 `daily.nav` 重算区间内收益（标 `*` 并写明起止），其余列为 `-`；月度表、回撤、调仓、行业取样、滚动指标按日期筛选；最新持仓只有报告末日的数据；`benchmarks`、`winrate`、`contributions`（全区间那张表）、`tails`、`concentration` 是全区间的数，小节开头写明。

### 提醒（`warnings`）

只读报告数据，规则保守、带数字，不做主观评价。阈值在 `checks.py` 顶部：

| 代码 | 条件 | 写出的数字 |
|---|---|---|
| `short_history` | 交易日不足 `periods_per_year`（一年） | 交易日数 |
| `benchmark_price_index` | 基准是价格指数且有相对指标 | 分年归因里税后分红的平均年贡献（只取完整年份；无分红时只写一句） |
| `benchmark_late` | 基准第一个有效日晚于基准日 | 基准起始日 |
| `benchmark_filled` | 基准中间缺的天数 > 0 | 填充天数 |
| `partial_start` | 首月或首年不完整（判法同 `monthly.first_partial` 与 yearly 的 `partial`） | 首个交易日、首年交易日数 |
| `partial_end` | 末月或末年不完整 | 末日、末年交易日数 |
| `weekday_basis` | `meta.completeness_basis == "weekday"` | — |
| `unfilled` | 调仓日开盘未成交只次 ÷（新进 + 剔除只次）≥ 5% | 只次、比例、按原因、最多的一次、下次调仓前补成交只次 |
| `holdings_short` | 成交后持仓只数 < 目标的 80% 的调仓占有目标调仓的 ≥ 10% | 次数、比例、最少的一次 |
| `cash_high` | 首次调仓后现金占比的中位数 > 5% | 中位数、超过 10% 的天数占比、最高值及日期 |
| `drawdown_open` | 末日回撤 ≤ −5% | 峰值日、交易日数、当前深度、期间最深及日期 |

只有一年数据时 yearly 的 `partial` 无法区分首尾，`partial_end` 只在首年判为完整（基准日在上一年）时才报末年。

### 真实样例（沪深300 低波动 30 等权，月度调仓，2014-12-31 至 2026-11-12）

`print(ts)` 的输出：

```
沪深300低波动30等权（月度调仓）
区间 2015-01-05 至 2026-11-12（基准日 2014-12-31），2881 个交易日（11.4 年）
基准 沪深300（000300.SH），价格指数（不含分红）

指标         策略  沪深300
累计收益  +157.8%   +14.4%
年化收益    +8.6%    +1.2%
年化波动    16.4%    21.5%
夏普         0.59     0.16
最大回撤   -34.3%   -46.7%
最大回撤 2015-06-12 见顶，2015-08-25 见底，2018-01-23 修复
年化超额 +7.4%，信息比率 0.44，跟踪误差 13.8%，Beta 0.58
年化换手（双边） 242%，费用 / 期初权益 2.29%，滑点 / 期初权益 2.27%

分年收益（* 2026 至 11-12）
年份     策略  沪深300    超额
2015   +31.2%    +5.6%  +25.6%
2016    +0.7%   -11.3%  +12.0%
2017   +15.0%   +21.8%   -6.7%
2018   -14.0%   -25.3%  +11.3%
2019   +13.7%   +36.1%  -22.3%
2020    -4.6%   +27.2%  -31.8%
2021    +9.3%    -5.2%  +14.5%
2022    -1.4%   -21.6%  +20.3%
2023   +18.7%   -11.4%  +30.1%
2024   +36.5%   +14.7%  +21.8%
2025    +2.8%   +17.7%  -14.9%
2026*   +1.2%   -12.7%  +13.9%

最深三次回撤
  深度  峰值        谷底        修复          见底     持续
-34.3%  2015-06-12  2015-08-25  2018-01-23   51 天   640 天
-24.9%  2018-01-24  2020-03-23  2023-03-20  522 天  1248 天
-15.3%  2025-07-22  2026-06-30  未修复      226 天   317 天

提醒：benchmark_price_index、partial_end、unfilled（--section warnings 看详情）
```

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

## 补充内容（数据层，页面暂不显示）

`build_tearsheet` 默认还输出下面五块（`extras=False` 时不算，输出与加入这些内容之前完全相同）；原有的键和数值逐位不变，页面不嵌入这五块（`render.EXTRA_KEYS`），HTML 逐字节不变。文本版各有对应小节。

### benchmarks：对比指数

`build_tearsheet(..., benchmarks=[Benchmark(...), ...])`，或 `tearsheet(..., data=..., benchmarks=[("000905.SH", "price"), ("000852.SH", "price", "中证1000")])`。每条指数一项：`name`、`symbol`、`kind`、`kind_label`、`filled_days`、`start`（第一个有效日），`cumulative_return`、`annual_return`、`annual_volatility`、`sharpe`、`max_drawdown`，策略相对它的 `strategy_annual_return`（重叠区间上的策略年化）、`excess_annual_return`、`tracking_error`、`information_ratio`、`beta`，以及 `yearly`（每年的指数收益，缺期初时为 `null`）。口径与 summary 的基准一组完全相同（同一套函数、同样只在重叠区间上算）；测试里把主基准再作为对比指数传一次，各项与 summary 逐位相同。没有对比指数时为 `[]`。

### win_rates：胜率

每条指数（主基准在前，`main: true`）一项；没有任何基准时为 `null`。

- 按年：每年策略收益与指数收益（年末 ÷ 上年末 − 1，与 yearly 同一口径）比，策略严格更高记一年跑赢。不完整的年份（yearly 的 `partial`）列在 `partial_years`、不计入；指数缺期初的年份也不计入。`years_won`、`years_counted`、`years_ratio`，逐年明细在 `yearly`（`strategy`、`benchmark`、`excess`、`won`、`partial`）。
- 滚动持有期（`rolling` 的 `1y`、`3y`、`5y`）：取月末行（首月不完整时不用基准日、末月不完整时不用末日；指数有效之前的点去掉），任意相隔 12k 个月的两点构成一段，策略区间收益 > 指数区间收益记跑赢。给出段数 `windows`、跑赢段数 `won`、比例 `ratio` 和超额最差的一段 `worst`（起止日、策略、指数、超额）。样本不够一段时为 `null`。

### contributions：个股贡献

引擎的 `result.contribution` 只有全区间每只证券的合计，没有逐日、逐年的拆分，所以用持仓和价格重建：

- 估值价：收盘价；收盘价缺失（停牌）时沿用上一个估值价，并按当天的现金分红和送转调整——与引擎的 `last` / `past_close` 相同。市值 = 持股数 × 估值价。
- 逐笔现金流：成交（买入为 −股数 × 成交价，卖出为 +；成交价已含滑点），费用（佣金 + 税 + 其他费用），税后分红（除权日前一天收盘持股 × 每股分红，红利税按当天的总税额在当天的正分红之间按比例分摊），退市清算所得。
- 区间 (a, b] 上一只证券的 `price` = 期末市值 − 期初市值 + 成交现金流 + 退市所得；`contribution` = `price` + `dividend` − `fees`。占比除以期初权益（全区间是初始资金，分年是上年末权益）。`holding_days` 是区间内收盘持股 > 0 的交易日数。

输出 `total`（全区间）和 `yearly`（每年一项，`period` 为年份，`partial` 同 yearly）：涉及证券数 `securities`、贡献最高的十只 `top` 和最低的十只 `bottom`（不与 `top` 重复）、合计 `total_pnl`、`total_price`、`total_dividend`、`total_fees`、`cash_interest`、`equity_change`。恒等式「各股贡献之和 + 现金利息 = 权益变化」（即价格 + 分红 − 费用 + 利息 = 权益变化）在测试里对全区间和任意子区间都核对；全区间的逐只贡献与引擎的 `result.contribution` 逐只相等（误差 ≤ 1e-6 元，测试含停牌期间除权的情形）。

### tails：收益分布与尾部

- `daily_skew`、`daily_kurtosis`、`monthly_skew`、`monthly_kurtosis`：总体偏度 m3 / m2^1.5 与超额峰度 m4 / m2² − 3（m_k 为 k 阶中心矩，除以 n）；样本少于 3 个或方差为 0 时为 `null`。
- `var_95`、`cvar_95`、`var_99`、`cvar_99`：历史法，基于全部日收益。VaR 为第 ⌈n·(1 − 置信度)⌉ 小的日收益（经验分布的左逆）；CVaR 为不高于它的日收益的均值。都写成收益（亏损为负）。⌈⌉ 用十进制精确值算，避免浮点下 100 × (1 − 0.95) = 5.0000000000000044 被进位成 6。
- `worst_days`：最差的 5 天（日期、收益、当天基准收益）；`worst_months`：最差的 5 个月（月份、收益、当月基准收益）。
- `longest_losing_months` 及 `_start`、`_end`：最长的连续月收益 < 0 的月数和起止月。

### concentration：集中度

逐日（有持仓的交易日）：前十大持仓权重之和、有效持仓数 1 / Σ(wᵢ / Σw)²、第一大行业的权重。权重口径与 `holdings_latest`、`industry_weights` 相同（持股数 × 估值价，按引擎的 `long_value` 缩放后除以权益）。`top10_weight`、`effective_holdings`、`top_industry_weight` 各给 `mean`（有持仓的日子的均值）、`max` 与 `max_date`、`latest`（末日，空仓为 `null`）；没有行业数据时 `top_industry_weight` 为 `null`。逐日序列在 `daily` 里（留给以后的页面）。

### 多策略对比：`compare`

```python
cmp = ashare.report.compare([ts_a, ts_b, ts_c], names=["低波", "红利", "动量"])
print(cmp)                  # 指标并排、分年收益并排、日收益相关系数
cmp.text("markdown"); cmp.data
```

区间取交集：共同的基准日（各报告第一行中最晚的一个）到共同的最后一天，报告区间与交集不同的逐个写明；各策略基准不同时也写明。指标用 `window.metrics`（页面公式）在交集上重算，交集正好是某个策略的全区间时取它的 summary。分年收益：整年都在交集里的取报告自己的 yearly，被截断的年份用嵌入的净值重算并标 `*`。相关系数是共同交易日上日收益的 Pearson 相关。

## HTML

- 完全自包含：原生 JS 加内联 SVG，系统字体栈，不引用任何外部网络资源；数据以列式 JSON 内嵌，按显示精度取整（净值、日收益 6 位小数，回撤、现金占比、换手 5 位），`summary` 原样嵌入。
- 打印（`@media print`）：A4 纵向单栏；指标和图都固定在全区间，读数取区间末日；交互控件隐藏；每张图不跨页切断。
- 补充内容不嵌入页面：加了它们之后，同一份回测写出的 HTML 与之前逐字节相同。
- 补充内容的耗时（2026-11-13，本机负载约 12，低波样例，三次取中位数）：`build_tearsheet` 0.037 s → 0.153 s（`extras=False` / 默认）；多出的部分主要是行业编号映射（约 0.05 s）、逐日集中度（约 0.05 s）和个股贡献重建（约 0.04 s）。`tearsheet(data=..., benchmarks=[两条指数])` 另花约 0.4 s 从数据层取指数（单次实测）。文本版默认摘要约 0.009 s、全部小节约 0.019 s（负载约 25）。

## 文本版与补充内容的验证

- 对拍：页面模板里的 `pct`、`num` 和 `metrics(a, b)` 用 node 直接运行，与 Python 版逐项比较（node 不可用时这两项测试跳过）；个股贡献与引擎的 `result.contribution` 逐只比较，并与一个逐日逐只模拟的直白参考实现比较（含停牌期间的分红和送转）；胜率、尾部、集中度、对比指数、多策略对比各有不复用正式代码的直白参考实现（尾部的偏度、峰度与 scipy 比较）。
- 排版：五种回测（有 / 无基准、基准晚起步、26 年长历史、不足一年）× 中英文，默认摘要不超过 40 行，全部小节的每一行不超过 100 列，不含 ANSI 和制表线字符。
- 变异检查（逐处改坏、跑 `test/report`、还原；先确认导入的是 worktree 里被改的文件），16 处全部被抓到：

| 文件 | 改坏的地方 |
|---|---|
| `fmt.py` | `toFixed` 半数取整改成银行家舍入 |
| `window.py` | 区间 beta 的协方差除以 n 而不是 n − 1 |
| `window.py` | `--from` 的基准日取到起始日本身 |
| `window.py` | 区间年化的指数用 n + 1 |
| `checks.py` | 未成交提醒的阈值翻倍 |
| `checks.py` | 现金占比统计不跳过首次调仓之前 |
| `checks.py` | 未修复回撤的峰值日取第一个而不是最后一个 |
| `extras.py` | 停牌期间估值价不随分红调整 |
| `extras.py` | 红利税不分摊到个股 |
| `extras.py` | 不完整年份计入胜率 |
| `extras.py` | VaR 分位用浮点算 ⌈n·(1 − 置信度)⌉ |
| `extras.py` | 有效持仓数不按股票总权重归一 |
| `sections.py` | 默认摘要的行数预算多给 6 行 |
| `compare.py` | 对比时被截断的年份不重算 |
| `text.py` | 宽表不压缩列间距 |
| `render.py` | 补充内容嵌入页面 |
- 性能（2026-11-13，本机，低波策略 11.9 年日频）：HTML 464 KB；`build_tearsheet` 约 0.03 s、`render_html` 约 0.01 s，`tearsheet(data=...)` 另花约 0.15 s 从数据层取名称、行业、停牌和基准；首屏脚本执行约 55 ms，切换区间约 4 ms，拖动十字线每次约 0.2 ms（均不含浏览器绘制）。全市场每周调仓 100 只（7.6 年、407 次调仓）的报告 469 KB。
