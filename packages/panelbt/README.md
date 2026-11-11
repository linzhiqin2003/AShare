# panelbt

对齐面板上的只做多组合回测。运行依赖仅 numpy；Rust 内核独立于 Python，`run_batch` 用 rayon 并行，模拟期间释放 GIL。不会读取行情、联网或移动信号日期。

## 安装和最小例子

在工作区根运行 `make dev`。以下代码可直接用 `uv run --no-sync python` 执行：

```python
import numpy as np
import panelbt as bt

prices = np.array([[10., 20.], [11., 19.], [12., 18.]])
market = bt.Market(
    exec_price=prices,
    close=prices,
    can_buy=np.ones_like(prices, dtype=bool),
    can_sell=np.ones_like(prices, dtype=bool),
)
targets = bt.Targets.from_dense([
    [0.5, 0.5],          # 第 0 日成交
    [np.nan, np.nan],    # 不调仓
    [0., 0.],           # 第 2 日清仓
])
costs = bt.Costs(tax_sell=0.0005)
config = bt.Config(initial_cash=100_000, record_trades=True)
result = bt.run(market, targets, costs, config)
print(result.nav)
print(result.positions)
print(result.trades[-1])

batch = bt.run_batch(market, [targets, targets], costs, config, threads=0)
np.testing.assert_array_equal(batch.nav[0], result.nav)
np.testing.assert_array_equal(batch.nav[1], result.nav)
```

## 接口

- `Market(exec_price, close, can_buy, can_sell, volume=None, split=None, cash_div=None, delist=None)`：所有面板同为 `T×N`，按行遍历。构造时转换为 C 连续 float64/bool 并校验一次；已符合要求的数组直接借用。NaN 的语义见 SPEC。
- `Targets(days, indptr, cols, weights)`：调仓日严格升序；每个调仓行列号唯一，自动按列排序。权重非负、行和不超过 `1+1e-9`。`Targets.from_dense(W)` 中整行 NaN 表示不调仓，其他 NaN 当 0；全 0 行表示清仓。`targets.to_dense(T,N)` 可还原。
- `Costs(commission=2.5e-4, min_commission=5., tax_sell=0., tax_buy=0., transfer_fee=1e-5, slippage=5e-4)`：税率接受标量或长度 T 的序列。
- `Config(initial_cash=1e7, min_qty=100., qty_step=100., max_participation=0., carry_blocked=True, delist_recovery=1., cash_buffer=0., record_trades=False)`：申报单位接受标量或长度 N 的序列；`qty_step=0` 允许小数股，忽略对应 `min_qty`。
- `run(market, targets, costs=None, config=None) -> Result`。
- `run_batch(market, targets, costs=None, config=None, threads=0) -> Result`：前置 K 维，0 使用全部核，1 串行；相同环境与相同输入时不同线程数逐位一致。空列表产生 `(0,T)` 输出。
- `panelbt.reference.run`：同接口的逐股 Python 参考实现，用于核对规则。

`Result` 提供 `nav`（期末资产/初始资金）、`cash`、`turnover`、`commission`、`tax`、`transfer_fee`、`slippage_cost`、`n_blocked_buy`、`n_blocked_sell`，以及 `delist_events`。数值字段为长度 T 的 numpy 数组，受限笔数为 int64。开启记录后，`positions` 为 `(T,N)`，`trades` 为 `Trade` 列表，卖出股数为负，记录含滑点成交价与分项成本。批量的持仓为 `(K,T,N)`，成交/退市事件为每组一个列表。`diagnostics` 是实际分支执行计数，供覆盖检查使用。

## A 股规则

```python
from panelbt.rules.ashare import (
    Board, infer_board, lot_rules, stamp_tax_sell, tradability,
)
minimum, step = lot_rules([Board.MAIN, Board.CHINEXT, Board.STAR, Board.BSE])
assert minimum.tolist() == [100., 100., 200., 100.]
assert step.tolist() == [100., 100., 1., 1.]
```

`tradability` 接收同形状成交价、高低价、真实涨跌停价和停牌掩码；半分钱绝对容差。`stamp_tax_sell` 支持 2008-09-19 起的日期。代码推断支持沪深 A 股、科创板和北交所 920 与历史代码；旧北交所前缀也可能属于全国股转系统，必须结合历史证券主表确认。

## 使用边界

- 目标行就是成交行，信号错位由调用方负责。公司行为使用原始价格和原始股数，不能同时再传复权价格。
- `Market` 和 `Targets` 持有只读视图；调用方须保证底层数组在对象使用期间不变，尤其不能在释放 GIL 的回测期间从其他线程修改共享数组。
- 当日缺失成交价时不交易，估值只使用当时已知的收盘价；不能从未来回填。退市后不再重新进入。
- 全量校验放在 `Market` 构造阶段；批量研究请复用同一对象。记录全量持仓会占用 `K×T×N×8` 字节，默认关闭。

完整边界与记账定义见 [SPEC.md](SPEC.md)，实测、设计疑问和限制见 [STEP1_NOTES.md](../../docs/STEP1_NOTES.md)。
