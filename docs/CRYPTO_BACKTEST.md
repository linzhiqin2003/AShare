# 币安加密货币回测

`ashare.backtest.crypto` 把数据层的小时面板接到 `panelbt`。现货现金账户与 U 本位全仓账户各自运行；结果仍是 `panelbt.Result`。规则依据是 [BINANCE.md](markets/BINANCE.md)，数据字段见 [CRYPTO_DATA.md](CRYPTO_DATA.md)，实现选择与验收见 [C3_NOTES.md](C3_NOTES.md)。

## 最短例子

```python
import numpy as np
from ashare.data import crypto
from ashare.backtest.crypto import crypto_market, daily_targets, top_n

# 固定 version= 可以重现旧数据版本。
ds = crypto.open("tmp/crypto")
m = crypto_market(
    ds, ["BTCUSDT", "ETHUSDT"], "2024-01-01", "2024-07-01",
    kind="usdm", initial_cash=1_000, leverage=2,
    assume="conservative",  # 规则缺口取前后观测中更不利的一个，报告列出每段。
)
daily = m.daily(trade_hour=0)  # 调用数据层 daily，不在回测层重新聚合。
# 日线 day 是区间左端；用这一天收盘算出的目标在下一日 00:00 执行。
plan = daily_targets((
    ["2024-01-01", "2024-01-02"],
    [[0.3, 0.3], [0.0, -0.3]],
))
r = m.run(plan)
print(r.nav[-1], r.diagnostics["crypto"])
print(m.daily_result(r))
```

现货换成 `kind="spot"`，目标非负且总权重不超过 1；不需要合约风险规则 `assume`。现货只开放权重目标，避免内核 `shares` 模式绕过数量快照；合约可用 `shares`。账户不收资金费、不执行交易所合约强平。`r.nav` 是相对初始资金的净值，`r.equity`、`r.cash` 是 USDT 金额，两者不能直接混着比较。

`crypto_market` 的其他参数：`allow_missing=False`、`max_participation=0`（0 关闭成交量上限）、`slippage=0`、`fee_switch=None`、`config=None`。`config` 可传 `panelbt.Config` 配置现金缓冲、记录成交等；数量规则仍由档案决定，不允许把现货融资、借币或通用退出规则混入本接口。普通 `run` 不额外移动目标；`kind="shares"` 表示合约张数，默认 `weights` 表示名义 / 权益。

## 日频、小时与信号

- 一行是 UTC `[t, t+1h)`，在左端按成交 K 线 open 交易。资金费在右端收付，归本行交易、强平、到期之后仍持有的张数；先结算旧仓，再执行下一小时目标。
- `daily(trade_hour=h)` 的日线覆盖 `[d日 h, 次日 h)`，`available_at` 是右端。`daily_targets` 接受 `(days, D×N)`、日期索引的 pandas 宽表，或 `day/instrument/weight` 的 Polars 长表。日期标签可为零时日期或该日 `trade_hour`，不应传入已经错位的 `available_at`。它只移动一次。
- `DailyTargets` 自动绑定市场小时轴；只有 `trade_hour` 有普通目标，固定 `carry_blocked=False`。成交量受阻的日频目标等下一个调仓日，不会在 01:00、02:00 反复尝试。小时模式直接传 `T×N` 数组或 `panelbt.Targets`，全 NaN 行不调仓；是否顺延由配置决定。
- `m.signal_prices[t]` 只含上一根已完成小时的成交收盘价；上市前占位、已终止价格不进信号。首行没有前一小时则为 NaN。原始 `m.arrays` 包含本小时完整行情和事后资金费，不能整行直接喂给当小时信号。费率信号用数据层 `last_funding_rate`。
- 引擎始终小时推进。小时年化因子 8760；`daily_result` 输出完整日区间末值，日频报表用 365，不使用股票的 252。`m.daily()` 使用数据层 `invalid_prices="carry"` 信号视图；异常值另留在 `ds.read()`，不会通过日线重新进入信号。直接调用 `ds.daily()` 遇倒置时间戳默认拒绝。
- 季度链按币分组，`roll_before="3D"` 用真实时间减三天，再向下落到最近的 `trade_hour`。例如 6 月 28 日 08:00 到期、01:00 调仓，换月落 6 月 25 日 01:00。`run` 默认启用；传 `roll_before=None` 直接交易具体合约、等待交割。链只向更晚到期日切换，新合约须已有完成小时报价。换月独立行维持报价单位数量，旧列平仓、新列开仓；若旧仓受阻，可能短时重叠，诊断单列。正常目标在已经移到更远月的链上路由到新合约，不回买旧合约。

## 市场档案与数量

```python
from panelbt import markets
profile = markets.get("binance_usdm")
fees = profile.fee_terms(["2023-05-31T23", "2023-06-01T00"])
# taker == delivery == [0.0004, 0.0005]；provenance 给来源与可信度。
rules, provenance = profile.rules(
    ["BTCUSDT"], ["2021-05-19T00"], assume="previous", leverage=10,
)
```

两档案均为 USDT、无涨跌停、`sell_lock=False`。现货吃单费固定 0.1%；U 本位 2019-12-20 08:00 起 0.04%，假设 2023-06-01 零时起 0.05%。更早的合约费用报错，不能用 `assume` 绕过。普通费用按行左端成交时刻取值，交割费按行右端结算时刻取值，二者分别保留出处。例如 2023-05-31 23:00 开仓收 4bp，次日 00:00 交割收 5bp。普通手续费按张部分为 0。费用通过档案生成，不从当前快照另起一份表；`fee_switch="2024-01-01"` 可做切换日敏感性，变更影响的区间标假设。

维持率、第一档名义上限、最大杠杆、清算费分别按交易对和证据区间取值。BTC 维持率连续性有明确文档依据，档位上限的中间变更日期则没有；不能因为知道当前 300,000 就倒填历史。点观测只覆盖观测小时，清算费当前快照只覆盖观测日；未知切换期间默认拒绝。非整点生效取下一整点。`assume="conservative"`（例子用这个）先用推断区间（两端观测一致、档位接口 `updateTime` 回填，记为假设），再在剩下的缺口里取前后两个记录中更不利的一个（维持保证金率、清算费取高，第一档上限、最大杠杆取低）；`assume="previous"` 在区间外沿用前一个观测，早于全部观测则沿用最早一个。两者都不是历史规则已核实的声明。2026-11-15 调研补齐了 BTC 第一档（2019-11 起）、ETH 第一档（2020-05 起分段）和 2021-06 以前按公告的清算费，证据见 BINANCE.md 5.2、5.3。所有用到的交易对、参数、起止、来源和原因都进入诊断，未来日期也不会默认无限沿用。风险规则另保留 `effective_at` 原始时刻、`applied_at` 实际整点、`rounding` 与 `approximation`；这些规则的使用区间与下单规则近似一并进入诊断。

无证据的交易对（含多数历史季度）须明确写假设值及出处，例如：

```python
assume = {"rules": {"usdm:SYNTHUSDT_240628": {
    "maint_rate": 0.004, "tier_cap": 50_000,
    "max_leverage": 100, "clearance_fee_rate": 0.0125,
    "reason": "合成试验参数，不是历史交易所规则",
    "source_url": "https://example.test/synthetic",
}}}
```

期货每张的乘数由数据版本冻结：价格保持交易所报价单位；成交量用 `volume_engine`（张）；`funding` 已是每张 USDT，不能再乘一次乘数。`1000PEPE` 的报价和数量单位是千枚 PEPE，USDT 成交额、最小名义不能乘 1000。`qty_step/min_qty/min_notional` 为 T×N，最小名义按标记开盘价，减仓豁免、反手与全平零头遵循 C1。

现货使用每个交易对最早数量规则的一份固定快照，整个使用区间标假设；最小名义第一批不执行。现货手续费从现金扣；交易所实际买入扣币，因此本模型持币量与实际存在差异。内核费用不能分交易对，BTCUSDT 零费率活动窗口仍收 0.1%，偏悲观。

## 选币与退出

```python
selector = top_n(20, lookback="30D", min_age="30D")
coins = selector.select(ds, at="2024-06-01T00:00")
```

按币合并 USDT 永续的实际报价成交额、按币名稳定打破并列，只读取调仓时刻以前完成的小时。窗口按实际毫秒长度，`min_age` 按档案挂牌时间；报价、保证金都必须为 USDT，排除传统资产与季度。当前截面还须有上一完整小时的活跃、可交易观测；已停止交易的币排除，但以后才下架的币会进入其历史截面。对交易暂停采用保守的上一完成小时资格。数据层以首个归档观测推断挂牌的品种，其年龄也只是覆盖年龄，不声称准确恢复首次挂牌。

现货下架按每小时 `execution_symbol` 匹配转换链上的实际交易对，再取当时可知的公告版本；缺公告检查同样使用实际交易对。快路径和逐小时路径不因追加未来公告而改变历史成交。`known_at` 不晚于当前小时左端才生效；之后每小时给该列零目标，压过普通买入，并取消该列此前受阻顺延的买单，即使当时空仓且没有新目标。已有持仓和挂起卖单继续按零目标退出，受普通成交量和手续费约束。无关列保持原持仓及小时模式下的顺延买入/减仓/卖出目标；日频普通受阻单不会被顺便重发。停止交易后的残留仓位继续按最后成交价估值，`cash` 不增加，也不生成 `DelistEvent` 现金兑付。缺公告的已下架列直接报错。

已经可知且生效的 `migration` 事件按数据层 `replacement/ratio` 接新币行情。引擎列保留旧计量单位：每旧单位对应 `ratio` 个新报价单位，价格乘比例、数量/成交量除比例，实际执行交易对见 `m.arrays["execution_symbol"]`。转换本身不收费、不产生现金。转换链与新币不能同时放入两个独立列，避免重复暴露；缺新币档案或行情、固定旧数量格点不能合法落到新币格点时均报错。未来才知道的映射不倒改旧交易。日线由数据层逐实际交易对聚合，跨迁移的策略信号须按映射明确接续，不能把旧币末价冻结当持续信号。

合约到期/下架由数据层 `expires/settlement` 交给内核。08:00 交割落 07:00 行，settlement 直接是交割价；不会先结转标记收盘再重复计交割损益。

## 缺口、强平与结果有效性

未上市、已结束合约可有 NaN 标记价、0 资金费，并保留正的 settlement 占位。占位不代表历史交易机会。生命周期内缺价默认报出交易对和时段；显式 `allow_missing=True` 才按不可交易、前值估值处理。首行没有过去估值仍报错。到期行的有来源结算价独立于缺行情处理保留，缺结算价或来源时仍拒绝。缺资金费不能用这个开关当作 0。

数据层标出的“成交 K 线半截线 + 紧随缺口”或公告暂停，有独立停机依据，可直接持平并诊断；辅助线半截或缺失不能证明全市场停机，缺口期间的正常成交线/标记线是反证。半截成交线同小时的正常标记线发生在缺口之前，不算反证；恢复成交即结束该缺口。倒置收盘时间单列异常，不能当作停机证据，回测默认拒绝。正常收盘时间的归档缺口也不自动叫停机。数据层 `verify` 通过表示与证据及显式例外一致，不意味着这些例外已被还原成真实交易过程。

U 本位按标记价做开盘、成交后、盘中、收盘维持保证金检查。多列盘中同时走向各自不利极值，是联合压力情景，不能称收益下界；单列准确判断是否触及强平线，成交价仍近似。强平普通吃单费之外另扣清算费，按剩余正权益封顶；穿仓缺口单列并清零余额，爆仓锁存。只用第一档，任何 `tier_cap_exceeded` 默认抛出 `InvalidCryptoResult`，异常的 `result` 属性保留完整结果；研究可传 `allow_tier_exceeded=True`，此时 `valid=False`，不能把它作为有效第一档回测。C1 对盘中退出的档位覆盖采取保守计数，因此强平手算例也需显式放开。

资金费价格：官方关联标记价优先，早期取右边界标记 open 并标近似。资金费后用本小时 settlement 再检查保证金，存在几秒的估值差；交易所结算边界前后约数秒的收费不确定性不重放。

## 诊断字段

`result.diagnostics["crypto"]` 保留构造诊断并追加结果诊断：

| 字段 | 含义 |
|---|---|
| `rule_assumptions` / `rule_provenance` | 实际使用的规则假设区间 / 所有风险参数证据；半开 UTC 区间 |
| `fee_provenance` / `delivery_fee_provenance` | 左端成交费 / 右端交割费的费用段、来源、可信度及假设切换日 |
| `rule_approximation_intervals` | 非整点规则和旧仓豁免未建模的完整适用区间 |
| `funding_price_approximation_cells` | 用边界标记 open 近似资金费价格的输入格数，不等于实际持仓付费次数 |
| `settlement_price_approximation_cells` | 合约下架结算价近似格数 |
| `missing_intervals` / `halt_intervals` / `partial_halt_cells` | 未解释缺口/有证据暂停区间，以及半截线格数 |
| `liquidation_counts` | open / post_trade / intraday / close 账户事件数 |
| `liquidation_events` | 每列被平仓的小时、交易对和数量；阶段统计见上项 |
| `clearance_fee` / `bankruptcy_deficit` | 实际清算费 / 穿仓缺口 USDT 合计 |
| `unrealized` | 样本末仍未变现的现货数量及末价估值，不计现金 |
| `tier_cap_exceeded` / `valid` | 第一档覆盖超限行数 / 第一档模型是否有效 |
| `filtered_orders` | 数量取整、最小数量/名义等内核过滤计数 |
| `blocked_buy` / `blocked_sell` | 普通交易受阻次数 |
| `rolls` / `roll_overlap_events` | 实际换月尝试及旧新合约重叠次数 |
| `migrations` | 转换时点、可知时间、比例、来源与计量单位 |

`Result` 本身保留逐小时 `funding/clearance_fee/bankruptcy_deficit`、费用、现金、权益、贡献、coverage 和事件；`record_trades=True` 才记录完整成交和持仓矩阵。

## 可复现实验

```sh
make dev
# 先把原始库复制到 worktree；CRYPTO_RAW_SOURCE 由调用方提供，只读源目录。
rsync -a "$CRYPTO_RAW_SOURCE/" tmp/crypto/
nice -n 15 .venv/bin/python bench/crypto_prepare.py --data tmp/crypto \
  --repair-daily --accept-spot-gaps
nice -n 15 .venv/bin/python bench/crypto_examples.py --data tmp/crypto \
  --quant-account "$QUANT_ACCOUNT_PATH"
nice -n 15 .venv/bin/python bench/mutations_crypto_backtest.py
nice -n 15 .venv/bin/pytest -q test/crypto test/test_crypto*.py -n 4
```

数据默认路径 `data/crypto`，可用 `--data` 或 `CRYPTO_DATA_ROOT`；Quant 源文件通过 `--quant-account` 或 `QUANT_ACCOUNT_PATH` 提供。脚本不改 Quant，也禁止向其写 pyc。公开结果仅写 `bench/results/crypto_examples.json` 的汇总数字，不写行情逐行值。手算 2021-05-19 使用真实标记价；资金费日从冻结原始结算记录独立计算张数 × 乘数 × 官方关联标记价 × 费率，并核对 08/16/次日00 点及实际持仓；费用汇总中的 `other_fee` 已含清算费，只计一次，另列清算费分项；复制样本无季度合约，交割用 C2 合成夹具并明确标记。

趋势例使用 20/60/120 日动量符号投票、30 日样本标准差、365 日年化、20% 波动预算均分给 BTC/ETH。前 120 日空仓。合约允许多空、调仓目标总名义最多 2 倍；现货截掉负目标、总权重最多 1。盘中名义会随价格漂移，2 倍是调仓目标上限。初始资金 1,000 USDT；未做参数搜索，也没有样本外收益声明。资金费和费用占比均以初始资金为分母，正资金费表示净支付。

未实现：C4 两钱包划转、逐仓、完整维持档位、组合保证金、现货借币、币本位、ADL、闲置 USDT 收益。结果与 Quant 的两钱包引擎不同是模型口径差异，不以数值对齐为目标。
