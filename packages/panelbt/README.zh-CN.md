# panelbt

市场无关的日频组合回测引擎，运行依赖只有 NumPy；Rust 内核通过 `tradecore` 与其他交易引擎共享逐笔成交原语。许可证 Apache-2.0。

## 最小示例

```python
import numpy as np
import panelbt as bt

prices = np.array([[10.0, 20.0], [11.0, 19.0], [12.0, 21.0]])
market = bt.Market(prices, prices, prices > 0, prices > 0)
targets = bt.Targets.from_dense([[0.5, 0.5], [np.nan, np.nan], [0, 0]])
costs = bt.Costs(commission_rate=0, commission_min=0,
                 fee_rate_buy=0, fee_rate_sell=0, slippage=0)
config = bt.Config(initial_cash=1000, min_qty=0, qty_step=0)
result = bt.run(market, targets, costs, config)
np.testing.assert_allclose(result.nav, [1.0, 1.025, 1.125])
```

目标在给定行成交，内核不会移动信号。整行 NaN 不调仓，全零行清仓。
`Targets.from_dense(..., kind="shares")` 直接指定股数，不按申报单位取整；仍受现金、保证金和可交易性约束。
负目标要求 `Config(margin=MarginModel(allow_short=True, ...))`。

## 纯权重记账

`run_weights(target, fwd_ret, can_buy=None, can_sell=None)` 和
`run_weights_batch(fwd_ret, targets, ..., keep_weights=True)` 是同一 Rust 引擎
中的纯权重模式。输入目标已对齐成交日，由调用方负责信号错位。该模式没有现金、股数、
手数、费用、分红账本或价格带来的权重漂移。每日调回目标，不能增减的股票冻结在前一日
权重，其余按多、空两边剩余额度缩放；盈亏为持仓权重与持有期收益的点积，换手为权重
变化绝对值之和，买卖两边都算。NaN 当零，无穷值沿用 `nan_to_num` 口径。

`GroupWeightTargets(labels, long_group, short_group=-1, long_budget=1,
short_budget=0)` 让多条组合共享分组标签，不展开每组完整目标矩阵。
`keep_weights=False` 只保留日序列；批量模式也支持逐组合选择是否保留持仓。
计算期间释放 GIL，调用方须保证输入不被并发修改。借用的数组视图临时设为只读，
最后一个重叠调用结束后恢复原有标志，异常退出也恢复；这不能锁住同一内存的既有别名。
旧 NumPy 实现保存在
`panelbt.reference_weights`，用于逐位验证。

`run_group_weights` 在同一次调用中逐日完成中性化、稳定分组、同一个权重账本的推进
和风格暴露；必须显式传入信号与成交起始行，以及价格历史和窗口内因子、对数市值、
行业、股票池。该入口仅返回日序列，不输出每组的完整持仓面板。截面代码静态复用
`panelops-core`，Python 运行依赖仍只有 NumPy。归约保留 NumPy 的顺序，点积使用
当前 NumPy 的 float64 内核；有符号零也纳入逐位对照。`run` / `run_batch` 的
现金和股数成交模式仍采用原有记账规则。

`run_group_weights_batch(..., variants=[{"rebalance": 1}, {"rebalance": 3,
"weighted": True, "record_holdings": True}])` 在一次释放 GIL 的调用中共享逐日
中性化、排名、目标和价格历史，各调仓周期独立记账。其余参数与 `run_group_weights`
相同，调仓、权重组合和持仓记录选项只放在 `variants` 中。返回顺序与输入相同，输出
数组彼此独立；相同调仓周期还会复用共同账本。目标工作区最多保留 64 日。

不同信号可通过 `exposure_cache=GroupExposureCache(logcap, universes, close,
adjustment, signal_start)` 共享各股票池、日期、风险因子的截面均值，支持并发。
传入连续且符合内核类型的数组，仅在一次任务内使用，整个缓存生命期禁止修改这些数组。
复用时检查缓冲区身份、股票池顺序和历史起点。持仓加权求和仍保留全长 NumPy 求和树，
不能去掉零项后再归约，否则会改变舍入结果。

要求 NumPy >= 2.1。Rust 归约刻意复现当前 NumPy 的 float64 加法次序（八路累加、
128 元素叶块），这属于 NumPy 内部实现，并非其兼容性承诺。导入 `panelbt` 时用
32 条小探针与 `np.add.reduce` 逐位核对，长度覆盖 1、7、8、9、127、128、129、1000，
包含抵消、负零和 NaN。不一致就抛 `ImportError`，说明求和次序变了、需要更新 Rust
内核；不会静默放宽为近似一致。

## 市场规则

```python
from panelbt.markets import ASHARE, Board

market = ASHARE.market(prices, prices, high=prices, low=prices,
                      up_limit=prices * 1.1, down_limit=prices * 0.9)
costs = ASHARE.costs(["2025-01-02", "2025-01-03", "2025-01-06"],
                      commission_rate=0.00025, commission_min=5)
config = ASHARE.config(board=[Board.MAIN, Board.STAR])
```

`ASHARE.costs(dates)` 按日期展开 2005-01-24 起的 A 股印花税和中国结算过户费，全历史回测不用再手填费率：

| 生效日 | 印花税 | 过户费（双边） |
|---|---|---|
| 2005-01-24 | 双边 0.1% | 每股 0.001 元 + 成交金额 0.0255‰（未核实，保守，见下） |
| 2007-05-30 | 双边 0.3% | |
| 2008-04-24 | 双边 0.1% | |
| 2008-09-19 | 仅卖出 0.1% | |
| 2012-09-01 | | 每股 0.0003 元 + 成交金额 0.0255‰ |
| 2015-08-01 | | 成交金额 0.02‰ |
| 2021-11-15 | | 成交金额 0.025‰（北交所费率，保守） |
| 2022-04-29 | | 成交金额 0.01‰ |
| 2023-08-28 | 仅卖出 0.05% | |

2015-08-01 以前沪市按面值（即按股数）、深市按金额收过户费；引擎的费率只能随日期变、不能随证券变，所以两种都计入，是上限。每一条的出处和核实情况写在 `panelbt/markets/__init__.py` 里 `ASHARE` 上方的注释中。佣金由券商定，调用方显式给。覆盖某个方向过户费的一种写法（`fee_rate_buy` 或 `fee_per_share_buy`）时，表里同方向的另一种写法清零；全区间固定费率这样写：`ASHARE.costs(dates, tax_buy=0, tax_sell=0.001, fee_rate_buy=1e-5, fee_rate_sell=1e-5)`。`infer_board` 接受带或不带 `.SH`/`.SZ`/`.BJ` 的六位代码；有权威的板块列时优先用板块列。

自建档案直接用 `MarketProfile`（内置的港股、美股档案见下一节）。下面的港股式、美股式费率**仅用于示意，未核实，不是可信的市场默认值，使用前必须核实**。示例复用前面的 prices 和 targets。

```python
from panelbt.markets import FeeSchedule, MarketProfile

dates = ["2025-01-02", "2025-01-03", "2025-01-06"]
hk_style = MarketProfile("HK-style illustration", "HKD", 252,
    FeeSchedule((("2025-01-01", {"tax_buy": 0.001, "tax_sell": 0.001}),)))
hk_config = hk_style.config(lot_sizes=[100, 500], initial_cash=100000)
hk_costs = hk_style.costs(dates, commission_rate=0.001,
                         commission_min=0, slippage=0)
hk_result = bt.run(hk_style.market(prices, prices), targets, hk_costs, hk_config)

us_style = MarketProfile("US-style illustration", "USD", 252,
    FeeSchedule((("2025-01-01", {"fee_per_share_sell": 0.001}),)))
us_config = us_style.config(lot_sizes=[0, 0], initial_cash=1000)
us_costs = us_style.costs(dates, commission_rate=0, slippage=0,
    commission_per_share=0.005, commission_min=1, commission_max_rate=0.01)
us_result = bt.run(us_style.market(prices, prices), targets, us_costs, us_config)
```

直接给 `lot_sizes` 可表达每只证券不同的每手股数；0 允许小数股。无涨跌停的档案仅检查有效行情和停牌。

## 港股与美股档案

内置三份档案：香港本地账户、港股通（内地投资者经沪深港通买港股）、美股。按名字取，`costs`、`config`、`market` 的用法与 `ASHARE` 一样：

```python
import numpy as np
import panelbt as bt
from panelbt import markets

prices = np.array([[50.0, 8.0], [52.0, 8.2], [55.0, 7.9]])
dates = ["2026-03-02", "2026-03-03", "2026-03-04"]
targets = bt.Targets.from_dense([[0.5, 0.5], [np.nan, np.nan], [0, 0]])
hk = markets.get("hk_direct")             # 香港本地账户，港币
config = hk.config(lot_sizes=[500, 2000], h_share=[True, True], initial_cash=1e6)
result = bt.run(hk.market(prices, prices), targets, hk.costs(dates), config)
```

```python
import numpy as np
import panelbt as bt
from panelbt import markets

prices = np.array([[50.0, 8.0], [52.0, 8.2], [55.0, 7.9]])
dates = ["2026-03-02", "2026-03-03", "2026-03-04"]
targets = bt.Targets.from_dense([[0.5, 0.5], [np.nan, np.nan], [0, 0]])
sb = markets.get("hk_southbound")         # 港股通，以港币回测
eligible = np.array([[True, True], [True, False], [True, False]])  # 第二只被调出
config = sb.config(lot_sizes=[500, 2000], initial_cash=1e6)
market = sb.market(prices, prices, connect_eligible=eligible)
result = bt.run(market, targets, sb.costs(dates), config)
cny = markets.connect_home_equity(result.equity, result.cash, [0.92, 0.921, 0.919], 1e6)
```

```python
import numpy as np
import panelbt as bt
from panelbt import markets

prices = np.array([[50.0, 8.0], [52.0, 8.2], [55.0, 7.9]])
dates = ["2026-03-02", "2026-03-03", "2026-03-04"]
targets = bt.Targets.from_dense([[0.5, 0.5], [np.nan, np.nan], [0, 0]])
us = markets.get("us")                    # 整股，分红预扣 10%（中美协定）
config = us.config(n=2, initial_cash=1e5)
result = bt.run(us.market(prices, prices), targets, us.costs(dates), config)
```

- 费用表从 2010 年起按日期分段（港股通从 2014-11-17 起）：港股印花税、证监会征费、会财局征费、交易费、股份交收费、交易系统使用费；美股 SEC 规费和 FINRA TAF（只收卖方）。`profile.charges(dates)` 按名字列出每一项。
- 内核没有每笔固定金额、单项每笔下限和取整，这些按 `amount / fill_value` 折成比例费（默认 HK$20,000、US$10,000）。成交额等于它时等于逐笔规则的上限；固定费和下限在更大的单笔上多收、更小的单笔上少收；印花税取整余量在单笔超过 `fill_value` 一半时高于其平均值（约 HK$0.5）。改成策略的典型单笔，或 `None`。单项每笔上限忽略（多收）。
- 默认佣金是假设：港股本地每笔 HK$15，港股通万分之 3，美股 IBKR Fixed 每股 $0.005、下限 $1、上限 1%，都可覆盖。
- 分红扣税：港股本地非 H 股 0%、H 股 10%；可选标记外商投资企业 H 股（2026-09-01 起 20%）和中国居民企业红筹（`resident_enterprise`，10%，默认不启用，若实际被预扣则偏乐观）。不给 `h_share` 时全部按 10% 并告警。港股通 20%；美股 10%（`US_WITHHOLDING` 里有 30% 和 0%）。内核每天只有一个税率：把行情的 `cash_div` 传给 `config`，只在当天派息的证券里取税率；不传则取当天最高税率并告警。`net_dividends` 是不能做空的账户的精确替代。
- 退市：价格冻结到除牌日，除牌日了结。`delisting="zero"`（只做多时的默认）按 0：多头悲观、空头乐观；`"last"` 按最后收盘价：多头乐观、空头中性。保证金模型允许做空时必须显式选择。无涨跌停、无卖出锁定：用规则时写 `Rules(sell_lock=False)`。港股通不能做空、不能融资；覆盖 `fee_rate_buy/sell` 时 `fx_cost` 仍然叠加（`fx_cost=0` 去掉）。
- 保证金账户：`markets.reg_t_margin(financing_rate=...)`（Reg T）、`markets.hk_margin(...)`（比例由券商给）。
- 哪些是近似、哪些不模拟、哪些需要内核支持，以及待确认的默认值，见 AShare 仓库的 `docs/markets/PROFILES.md`。

## 账户、费用与结果

`MarginModel` 支持逐日保证金率、多空、融资、融券费、受限融券所得、追保和次日强平。
它是按权益定义的简化保证金模型，并非完整监管信用账本。
A 股监管担保物/负债公式与模型不同，`markets.ashare_margin` 要求调用方显式校准所有比率。

`Costs` 支持比例和按股佣金、佣金下限与上限、方向分开的逐日税和其他费用、面板滑点及成交量冲击。
`Config` 另有调仓带、分红税、资金进出和每期闲置现金利息。行情与目标必须对齐；使用期间不能通过别名修改借用的数组。

`nav` 是权益除初始资金；资金进出后的绩效使用 `returns` 和 `unit_nav`。
`contribution` 计证券价格盈亏、税后分红、成交费用及融券费；现金利息和融资利息另列。
`panelbt.metrics.summarize(result)` 用 NumPy 计算绩效，年交易期数可以指定。

`result.to_npz(path)` / `Result.from_npz(path)` 存取版本化数组和 JSON，不使用 pickle。
`result.to_frame(index=dates)` 仅在调用时导入可选 pandas。
`run_batch(..., threads=0)` 使用全部可用 CPU，批量共享行情；同一环境下，线程数不改变单策略浮点执行顺序。
开启 `record_trades=True` 会保存 T×N（批量 K×T×N）持仓及成交，较大的批量通常保持默认关闭。

## 期货对冲列

在 `Market(futures=Futures(...))` 中逐列标记期货，现货与期货在同一面板结算。期货权重表示名义金额除以权益，股数模式表示整数手数；开仓只支付费用并占用保证金。必须提供结算价、合约乘数、到期行和保证金率；手续费、交割费可分别按名义金额和按手设置。示意用法见 [英文 README](README.md#futures-hedge-columns)；示例费率并非市场默认值，使用前须核实。

`panelbt.futures.roll_targets` 按到期前交易日生成换月目标；`hedge_contracts` 按组合市值或贝塔计算对冲手数。`markets.cffex_index_futures` 只内置核实过的 IF/IH/IC/IM 合约乘数，费率及保证金率由调用方提供。

`hedge_contracts` 与引擎一样向零取整，敞口不足一手时给出 `HedgeRoundingWarning`。现金账户挂上期货后，股票部分仍按原来的现金账户规则成交：先从现金里预留期货保证金，持有过合约后才会追保。现金只约束保证金占用、不约束名义金额：保证金率 12% 时，现金账户可以持有约 8 倍权益的合约名义。期货亏损使现金为负后，股票卖出只要成交额够付自身费用就照常成交。费率接受标量、(1, N) 按合约、(T, 1) 按日和 T×N，一维数组因含义不明确而报错。到期行的 `settlement` 必须填交易所的交割结算价。引擎不区分平今仓手续费（中金所平今费率远高于普通平仓），同一天开平的换月会低估成本。

## 纯期货组合

`FuturesBook` 用逐合约、逐行的交易所参数（结算价、乘数、保证金率、手续费、最后交易行）建一个全是期货（也可以混现货）的市场，用同一个内核回测。它负责：期货公司条款 `FuturesAccount(margin_add, margin_scale, fee_scale)`；实物交割合约的最后交易行、或可选的最后持仓行之后，按当行结算价强平并计手续费和滑点；涨跌停掩码；三种目标写法：手数（`"lots"`）、名义市值 / 权益（`"notional"`）、保证金 / 权益（`"margin"`）。

```python
s = np.array([[100.0], [101.0], [99.0]])
book = bt.FuturesBook(
    settlement=s, close=s, exec_price=s, multiplier=[10.0], expiry=[3],
    margin_rate=0.1, fee_rate=1e-4,
    account=bt.FuturesAccount(margin_add=0.05, fee_scale=2.0),
    initial_cash=1e5,
)
book_result = book.run(np.array([[0.5], [np.nan], [np.nan]]), kind="margin")
```

盈亏一律按具体合约的结算价算；连续价格只拿来算信号，不要用它算收益（换月价差会被算成盈亏）。语义与理由见 SPEC 的「Stand-alone futures portfolios」（英文）。

## 逐日步进

`bt.Simulator` 用同一个内核一次推进一天，策略可以根据账户状态决定当天的目标。第 t 步给的目标是第 t 日成交后要持有的；第 t 步之前的 `sim.state` 只含第 t 日成交前已知的信息。同样的目标逐日喂进去，结果与 `bt.run` 逐位相同。

```python
import numpy as np
import panelbt as bt

prices = np.array([[10.0, 20.0], [11.0, 19.0], [12.0, 21.0]])
market = bt.Market(prices, prices, prices > 0, prices > 0)
costs = bt.Costs(commission_rate=0, commission_min=0,
                 fee_rate_buy=0, fee_rate_sell=0, slippage=0)
config = bt.Config(initial_cash=1000, min_qty=0, qty_step=0)

sim = bt.Simulator(market, costs, config)
for t in range(sim.n_days):
    state = sim.state                     # 现金、权益、持仓、持仓成本、建仓日……
    if t == 0:
        target = ([0, 1], [0.5, 0.5])     # (列号, 权重)；也可以给长度 N 的一行
    elif state.shares[0] and prices[t - 1, 0] < 0.9 * state.cost[0]:
        target = ([1], [0.5])             # 第 0 列亏损 10% 后退出
    else:
        target = None                     # 当天不给新目标
    sim.step(target)
result = sim.result()

blob = sim.snapshot()                     # 字节串，用 Simulator.restore 恢复
```

状态数组是只读的实时视图（要保留历史就 `copy()`；numpy 不允许把它改回可写，经由公开接口改不了模拟状态）。`step` 期间释放 GIL，不同的模拟可以在不同线程里推进；同一个模拟不能被两个线程同时推进或读取（会抛 `RuntimeError`）。每一步的固定开销是几微秒，不随证券数增长。止损这类固定模式用下面的声明式规则更快。

## 声明式止损和限价目标

止损、止盈、移动止损、最长持有期和限价单都在内核里执行，`run`、`run_batch` 和 `Simulator` 一样可用。日线不告诉你一天之内价格的先后顺序，所以每个说不清的情形都按对策略不利的一边处理：跳空穿过止损线按开盘价成交，同一天既碰止损又碰止盈按止损算，移动止损线只用到前一天为止的最高价。`fill_time` 说明 `exec_price` 在一天中的位置，必须给。

```python
import numpy as np
import panelbt as bt

o = np.array([[10.0], [10.4], [9.0], [9.2]])
h = np.array([[10.5], [10.9], [9.3], [9.6]])
l = np.array([[9.9], [10.1], [8.7], [9.0]])
c = np.array([[10.3], [10.2], [9.1], [9.5]])
ok = np.ones_like(o, dtype=bool)
bars = bt.Market(o, c, ok, ok, open=o, high=h, low=l)  # 成交价 = 开盘价

rules = bt.Rules(
    fill_time="open",        # exec_price 是开盘价
    stop_loss=0.08,          # 标量、每只证券一个值，或 T×N 面板
    trailing_stop=0.12,
    take_profit=0.30,
    max_hold=20,             # 行数
)
exit_config = bt.Config(initial_cash=1e5, rules=rules, record_trades=True)
exit_result = bt.run(bars, bt.Targets.from_dense([[1.0], [np.nan], [np.nan], [np.nan]]),
                     bt.Costs(), exit_config)
print([(t.day, t.reason.name, t.price) for t in exit_result.trades])
print(exit_result.diagnostics["stop_loss"])   # 各规则的计数在 diagnostics 里

# 限价目标：当天价格跌破 9.1 才买（最低价正好等于限价不算成交；
# Rules(limit_fill="touch") 可以改成碰到就算）。
limits = np.array([[np.nan], [np.nan], [9.1], [np.nan]])
weights = np.array([[np.nan], [np.nan], [1.0], [np.nan]])
limit_result = bt.run(bars, bt.Targets.from_dense(weights, limits=limits), bt.Costs(),
                      bt.Config(initial_cash=1e5, rules=bt.Rules(fill_time="open"),
                                record_trades=True))
```

规则平仓后，这一列要等到之后的目标行（`cooldown=None`，默认）或 `cooldown=k` 行之后才能重新开仓。`sell_lock=True`（对所有市场默认开启，对应 A 股 T+1；别的市场要设为 False）时当天买入的股份当天不会被规则卖出，平仓顺延到下一行。限价买单要当天最低价严格低于限价才成交（卖单对称）：最低价正好等于限价的日子，往往是挂在当天极值上没成交的单，算成交会明显高估低吸类策略。卖不出去（跌停、停牌）的平仓在之后各行按当天的 `exec_price` 重试。不用规则时结果与没有这项功能的版本逐位相同。市场里有期货列时，退出规则只作用于现货列（标量百分比只施加给现货列；每只证券或 T×N 的取值在期货列上必须是 NaN），限价目标直接拒绝。确切语义见 [SPEC.md](SPEC.md)（英文）。

## 输入检查

`Market(...)` 与 `Targets(...)` 默认逐格检查（价格为正或 NaN、掩码取值、送转为正、分红非负、期货结算价与费率、调仓日递增、权重有限、同一行列号不重复），检查在 Rust 内核里多线程完成。传 `validate=False` 只跳过这些逐格的内容检查：

```python
fast_market = bt.Market(prices, prices, prices > 0, prices > 0, validate=False)
fast_targets = bt.Targets.from_dense([[0.5, 0.5], [np.nan, np.nan], [0, 0]],
                                     validate=False)
fast = bt.run(fast_market, fast_targets, costs, config)
np.testing.assert_allclose(fast.nav, [1.0, 1.025, 1.125])
assert not fast.inputs_validated
```

适用于同一份行情已经检查过一次、之后反复重建，或输入来自上游已校验过的数据。形状、dtype、内存布局、各面板形状一致、目标的指针和行列号范围始终检查，所以未检查的输入不会越界访问，只是坏值会算出没有意义的结果。需要账户条款的检查（做空权限、整数手、现金账户权重和）对未检查的目标同样跳过。未检查的调仓日如果重复或倒序不会报错，后面那些行会静默地不执行。`result.inputs_validated` 说明这次回测（批量时为全部目标）用的行情和目标是否都检查过。

详细顺序、公式、监管来源、T+1 日频约定和不支持的功能见 [SPEC.md](SPEC.md)。多币种和日内多次成交尚不支持。

## 独立构建

需要 Python 3.13+ 和稳定版 Rust。源码包包含 Rust 路径依赖，可在本项目以外用 `uv build` 或 `pip wheel .` 构建；wheel 的运行依赖只有 NumPy。
不需要本项目其他模块、行情服务、凭据或本机绝对路径。当前不发布到 PyPI。
