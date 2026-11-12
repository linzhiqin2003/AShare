# panelbt

市场无关的日频组合回测引擎，运行依赖只有 NumPy；Rust 内核通过 `tradecore` 与其他交易引擎共享逐笔成交原语。许可证待定，尚未发布，也尚未授予发布许可。

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

## 市场规则

```python
from panelbt.markets import ASHARE, Board

market = ASHARE.market(prices, prices, high=prices, low=prices,
                      up_limit=prices * 1.1, down_limit=prices * 0.9)
costs = ASHARE.costs(["2025-01-02", "2025-01-03", "2025-01-06"],
                      commission_rate=0.00025, commission_min=5)
config = ASHARE.config(board=[Board.MAIN, Board.STAR])
```

只内置 A 股档案。下面的港股式、美股式费率**仅用于示意，未核实，不是可信的市场默认值，使用前必须核实**。示例复用前面的 prices 和 targets。

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
