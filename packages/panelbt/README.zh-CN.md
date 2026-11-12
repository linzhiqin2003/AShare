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

详细顺序、公式、监管来源、T+1 日频约定和不支持的功能见 [SPEC.md](SPEC.md)。多币种和日内多次成交尚不支持。

## 独立构建

需要 Python 3.13+ 和稳定版 Rust。源码包包含 Rust 路径依赖，可在本项目以外用 `uv build` 或 `pip wheel .` 构建；wheel 的运行依赖只有 NumPy。
不需要本项目其他模块、行情服务、凭据或本机绝对路径。当前不发布到 PyPI。
