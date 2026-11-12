# 境外期货：Databento 与 Bloomberg 适配器

> **未用真实账号验证。** 两个适配器都是照官方文档、官方 Python 客户端（Databento）的接口和返回形状写的，测试全部用合成数据和假客户端。拿到账号后先跑文末的「首次核对清单」，逐条对上公开数字以后再用于研究。

把境外期货（NYMEX WTI、ICE 布伦特、NYMEX 布伦特现金结算）的合约表、日线、分钟线、日历转成与境内期货相同的统一字段（`ashare.data.futures_schema`），面板、连续合约、`panelbt_inputs` 都不用改。设计见 `docs/DESIGN.md` 6.1.2。

## 最短用法

```bash
pip install "ashare[databento]"     # 或 uv sync --extra databento；Bloomberg 用 ashare[bloomberg]
export DATABENTO_API_KEY=...        # 只从环境变量读，不写进任何文件
```

```python
from ashare.data.sources.databento import DatabentoSource
from ashare.data.futures_foreign_databento import DatabentoFutures
from ashare.data.futures_foreign import build_panel, write_parquet

src = DatabentoSource()                                   # 默认预算 0：只估价，不取数
fut = DatabentoFutures(src, ["CL"], "20261102", "20261106", margin={"CL": 0.1})
print(fut.estimate())                                      # 各请求的估算费用（美元），不计费
src.budget_usd = 2.0                                       # 显式给预算才会取数（累计，超了就停）
fut.daily(); fut.contracts(); fut.minute("20261104", "20261104")
write_parquet(fut, "data/foreign/databento")               # 只存本地 data/，不进仓库
panel = build_panel(fut, "data/foreign/panel")             # 单独的数据根
panel.continuous("CL", rule="open_interest")              # 本仓库的换月规则
```

```python
from ashare.data.sources.bloomberg import BloombergSource  # 本机终端已登录，127.0.0.1:8194
from ashare.data.futures_foreign_bloomberg import BloombergFutures

fut = BloombergFutures(BloombergSource(), ["CL", "B"], "20251103", "20251107")
fut.daily(); fut.minute("20251105", "20251105")
```

接回测（与境内同一个入口；保证金、费用来源没有，必须显式给）：

```python
from ashare.data.futures_panelbt import panelbt_inputs
import panelbt as bt

inputs = panelbt_inputs(panel, products=["CL"],
                        missing={"margin_rate": 0.1, "fee_rate": 0.0, "fee_per_contract": 2.0})
market = bt.Market(**inputs["market"], futures=inputs["futures"])
```

`test/data/test_futures_foreign.py::test_adapter_to_continuous_to_panelbt` 用合成的 CL 月度合约链走通了这一整条路（适配器 → 面板 → 按持仓量换月的连续合约与换月日程表 → `panelbt` 输入数组 → `bt.Market`）。

## 模块

| 层 | 模块 | 做什么 |
|---|---|---|
| 取数 | `sources/databento.py` | 建 `databento.Historical`（密钥取自环境变量）；**每次取数前先 `metadata.get_cost`**，累计估算超预算抛 `BudgetExceeded`（带估算值），不发 `timeseries.get_range`；`DBNStore.to_df()` → polars，字段保留原名 |
| 取数 | `sources/bloomberg.py` | 同步 `Session`，`//blp/refdata` 的 `HistoricalDataRequest`、`ReferenceDataRequest`、`IntradayBarRequest`；响应用 `Message.toPy()` 原样转成字典 |
| 换算 | `futures_foreign.py` | 品种表、十年回绕、交易日归属、分钟线字段 `MINUTE`、日历、前结算、参数表、落盘 `write_parquet`、接面板 `build_panel` |
| 换算 | `futures_foreign_databento.py`、`futures_foreign_bloomberg.py` | 两家的原始表 → 统一字段；`DatabentoFutures`、`BloombergFutures` 是适配器（`contracts`、`daily`、`params`、`calendar`、`mapping`、`source_continuous`、`diagnostics`、`minute`） |

可选依赖：`ashare[databento]`（`databento>=0.90`）、`ashare[bloomberg]`（`blpapi>=3.24`，在 Bloomberg 的包索引 `https://blpapi.bloomberg.com/repository/releases/python/simple/` 上，`pyproject.toml` 已登记为显式索引）。都没装时 `make check` 照样全过；调用到取数层才报错并提示装哪个 extra。换算层不依赖官方库。

## 品种

| 统一品种 | 交易所 | 合约示例 | Databento | Bloomberg | 乘数 | 报价 | 交割 | 交易时段归属 |
|---|---|---|---|---|---|---|---|---|
| CL | NYMEX | `CL2612.NYMEX` | `GLBX.MDP3`，母代码 `CL.FUT`，原始代码 `CLZ6` | `CLZ6 Comdty`，通用 `CL1 Comdty` | 1000 桶 | 美元/桶，最小变动 0.01 | 实物 | 芝加哥 17:00 起归下一交易日 |
| B | ICE | `B2612.ICE` | `IFEU.IMPACT`，母代码 `BRN.FUT`，原始代码 `BRN FMZ0026!` | `COZ6 Comdty`，通用 `CO1 Comdty` | 1000 桶 | 美元/桶，0.01 | 现金（EFP 可交割，记 cash） | 伦敦 23:00 起归下一交易日 |
| BZ | NYMEX | `BZ2612.NYMEX` | `GLBX.MDP3`，母代码 `BZ.FUT` | 未登记（代码未核实） | 1000 桶 | 美元/桶，0.01 | 现金 | 同 CL |

品种规则来源见 `docs/markets/futures_rules.md` 第二部分。加品种只在 `futures_foreign.PRODUCTS` 加一行。

## 字段映射

### 合约表 `contracts`

| 统一字段 | Databento（`definition`） | Bloomberg（`ReferenceDataRequest`） |
|---|---|---|
| symbol | 品种 + `maturity_year`/`maturity_month`（缺失时解析 `raw_symbol`，以 `expiration` 为参照日解年份；两者不一致时以定义为准并记进诊断 `maturity_conflicts`） | 解析代码（以 `LAST_TRADEABLE_DT` 为参照日解年份） |
| exchange、product | 品种表 | 品种表 |
| name | 品种表名称 + 年月 | `NAME` |
| listing_date | `activation` 按交易时段归属的交易日 | `FUT_FIRST_TRADE_DT` |
| last_trade_date | `expiration`（最后可成交时刻）按交易时段归属的交易日 | `LAST_TRADEABLE_DT` |
| delivery_date | 空（来源没有） | `FUT_DLV_DT_LAST` |
| delivery_month | 年月 | 年月 |
| multiplier | `unit_of_measure_qty`（`unit_of_measure` = BBL；与品种表不同时记进诊断 `multiplier_mismatch`） | `FUT_CONT_SIZE` |
| quote_unit | 品种表（美元/桶） | 品种表 |
| tick_size | `min_price_increment` | `FUT_TICK_SIZE` |
| settlement_type | 品种表 | 品种表 |
| currency | `currency`（USD） | `CRNCY`（USD） |
| multiplier_source | `contract_table_unverified`（没有成交额可反查） | 同左 |

Databento 的定义：只要 `instrument_class == "F"`（价差是 S，代码带 `-`）；同一 `instrument_id` 取 `ts_recv` 最晚的一条，`security_update_action == "D"` 的删掉。`display_factor` 不是 1 的记进诊断 `display_factor`。

### 日线 `daily`

| 统一字段 | Databento | Bloomberg（`HistoricalDataRequest`，DAILY） |
|---|---|---|
| date | 交易所交易日：K 线按交易时段归属；结算价取 `ts_ref` 的 UTC 日期 | 响应里的 `date`（假定为交易所交易日，**未核实**） |
| open | 当日第一根小时线的开盘（含前一晚夜盘） | `PX_OPEN` |
| high、low | 当日各小时线的最高、最低 | `PX_HIGH`、`PX_LOW` |
| close | 当日最后一根小时线的收盘（最后一笔电子盘成交所在小时） | `PX_LAST` |
| settlement | `statistics`，`stat_type == 3`，`stat_flags` 第 0 位 = 1（final）且第 3 位 = 0（不是日内结算）；同一天多条取 `ts_recv` 最晚的；最后一条是删除（`update_action == 2`）则为空。`settlement="any"` 时也收初步结算价 | `PX_SETTLE` |
| previous_settlement | 同一交易所日历上一开市日该合约的结算价；新合约首日为空（来源没有挂牌基准价） | 同左 |
| volume | 默认 `volume="cleared"`：`stat_type == 6`（结算量，含大宗、EFP，与交易所日报口径一致）；`volume="electronic"`：小时线成交量之和（只含电子盘）。两者不混用，缺了就是空 | `PX_VOLUME` |
| amount | 空（来源没有成交额） | 空 |
| open_interest | `stat_type == 9` 的 `quantity`；有 `ts_ref` 用它的 UTC 日期，没有时归发布时刻所属交易日的上一开市日（**未核实**）；`INT64_MAX` 视为空 | `OPEN_INT` |

- 单位：价格保持来源报价单位（美元/桶），不换算成美分；成交量、持仓量都是手，单边。
- 为什么不用 `ohlcv-1d`：官方说明它按 **UTC 日** 切分；CME 的交易日从前一天 17:00（芝加哥）开始，UTC 日切分会把夜盘劈成两半。CME 17:00、ICE 23:00 都在整点，小时线能按交易时段干净切开，价钱只有分钟线的约六十分之一。
- 负价格保留：WTI 2020-04-20 结算价为负（CLK0，−37.63）。境内统一字段「价格只保留正数」针对的是源把 0 当占位，境外按真实值保留，个数记在诊断 `nonpositive_price_rows`。
- 只留 `[start, end]` 且在合约上市期内的行；对不上合约表的 `instrument_id`（价差、其他品种）计数在 `unknown_instrument_bar_rows`、`unknown_instrument_stat_rows`，上市期外丢掉的计数在 `outside_listing_rows`。

### 分钟线 `minute`（统一字段新增，`futures_foreign.MINUTE`）

| 字段 | 单位 | Databento（`ohlcv-1m`） | Bloomberg（`IntradayBarRequest`，1 分钟，TRADE） |
|---|---|---|---|
| ts | K 线开始时刻，UTC | `ts_event`（官方：区间开始） | `time`（官方示例：GMT；假定为开始时刻，**未核实**） |
| date | 所属交易日 | 按交易时段归属 | 同左 |
| symbol | 统一合约标识 | `instrument_id` → 合约表 | 请求的代码 → 合约表 |
| open、high、low、close | 报价单位 | 同名 | 同名 |
| volume | 手，只含电子盘 | `volume` | `volume` |

两家都是「这一分钟没有成交就没有这根 K 线」。分钟线另走一次费用闸门（`ohlcv-1m` 价钱高）。

### 逐日参数 `params` 与日历 `calendar`

- `params`：两家都没有交易所逐日保证金和费用。每个日线行给一行，`margin_long`/`margin_short` 是调用方给的 `margin`（标量或「品种 → 比例」，例如经纪商标准，自己写明来源），不给就为空；费用一律为空，回测里用 `panelbt_inputs(missing=...)` 或显式参数给（交易所费用调研见 `docs/markets/futures_rules.md`：ICE 布伦特 0.88 美元/手/边；CME 未核实）。
- `calendar`：两家都没有日历接口。开市日 = 该交易所**有 final 结算价**的交易日，区间内其余自然日为休市。整交易所一天都没有结算价才算休市；半天交易的节假日（感恩节次日等）照常开市。

## 交易日归属

`futures_foreign.trade_dates`：把 UTC 时刻换到交易所时区，本地时间 ≥ 切换点的归下一天，再顺延过周六周日，给了开市日集合时继续顺延到下一个开市日。

| 交易所 | 时区 | 切换点 | 例 |
|---|---|---|---|
| CME Globex（CL、BZ） | America/Chicago | 17:00 | 周日 17:00 开盘 → 周一；周四 17:30 → 周五；感恩节当天的成交 → 周五 |
| ICE Futures Europe（布伦特） | Europe/London | 23:00 | 周日 23:00 开盘 → 周一；周二 22:59 → 周二 |

夏令时由时区库处理（同一个 UTC 钟点在夏令时和冬令时归属不同，测试里各有一例）。结算价的交易日直接用 Databento `ts_ref`（官方：CME 的 TradingReferenceDate，只有日期精度，不换时区），与按时段归属的 K 线在同一天对齐。

## 合约标识与十年回绕

- Databento：`CLZ6`（CME 一位年份）、`CLZ16`、`BRN FMZ0026!`（ICE：品种空格补足 4 位 + F 期货 + M 月度 + 月份码 + 日 00 + 两位年份 + `!`）；价差（带 `-`）不认。连续合约记法 `CL.c.0`（按到期先后）、`CL.n.0`（按前一日持仓量）、`CL.v.0`（按前一日成交量）由 `parse_continuous` 解析。
- Bloomberg：`CLZ6 Comdty`、`CLZ16 Comdty`、一个字母的品种补空格（`W N9 Comdty` 这类）；通用代码 `CL1 Comdty` 由 `parse_generic` 解析。
- 年份：`resolve_year(数字, 月份, 参照日)` 取「交割月不早于参照日所在月」的最小年份，一位、两位年份同样处理。**参照日用最后交易日**：交割月最多比最后交易日晚两个月（WTI 一个月、布伦特两个月），所以任何年代都解得对（测试用随机年份对拍直白参考，含 2029→2030、1999→2000 跨界）。用观测日当参照日时，十年以上的远月合约与同名近月合约分不开，这是一位年份代码的固有歧义，所以 Databento 优先用定义里的 `maturity_year`。

## 连续合约口径

适配器只取**具体合约**，连续合约一律由本仓库的 `futures_continuous` 用同一套换月规则拼（`rule="open_interest"` / `"volume"`，只用前一交易日数据）。两家自带的连续序列只作对账参照，不进面板：

| 来源 | 记法 | 换月口径（官方文档） | 本仓库怎么用 |
|---|---|---|---|
| Databento | `CL.c.0` | 按到期先后，到期后的下一个交易日换（例：NG 10 月合约 9-28 收盘到期，9-29 起指向 11 月） | `reference_continuous=True` 时经 `symbology.resolve` 取逐日对应合约：`source_continuous()` 给三种规则，`mapping()` 里 `c.0` 记为 nearby |
| Databento | `CL.n.0`、`CL.v.0` | 按**前一日**收盘的持仓量 / 成交量排序；具体换月时刻文档没写（**未核实**） | `n.0` 记为 mapping 的 main，可用 `rule="source"` 对照 |
| Bloomberg | `CL1 Comdty`、`CO1 Comdty` | 默认在到期时换到下一合约，终端里可按代码改换月口径；**未核实** | `source_continuous()` 给通用代码的结算价（历史请求不说当天是哪个合约），只能按价格对账 |

Databento 的 `symbology.resolve` 返回区间 `d0`/`d1`，按「d0 含、d1 不含」理解（**未核实**），默认不调（`reference_continuous=False`）。

## 费用与密钥

- Databento 按量计费。`DatabentoSource.fetch` 先用**同样的参数**调 `metadata.get_cost`，累计估算（`spent_usd + 本次`）不超过 `budget_usd` 才调 `timeseries.get_range`；**预算为 0 时永远不取**（估算为 0 也不取）。超预算抛 `BudgetExceeded`，消息里有本次估算、已批准累计和预算。`fut.estimate()` 只估价、不计费。每次估价记在 `source.log`，也进诊断 `cost_log`。
- 官方说明：估价按请求范围，账单按实际发送字节；起止不对齐 10 分钟边界时估价可能偏高。所以预算闸门是上限保护，不是精确账单。
- 一次日线取数的请求：每个数据集 `definition`（[start, end+1)）、`statistics`（[start, end+6)，持仓量、结算量在下一个 UTC 日或周日发布）、`ohlcv-1h`（[start−3 天, end+1)，覆盖周日晚和前一晚开盘）。
- 密钥只从环境变量 `DATABENTO_API_KEY` 读，不写进任何文件、日志、诊断或报错；没设时报「环境变量没有设置」，不打印环境。Bloomberg 桌面 API 走本机终端，不需要密钥。
- 数据许可：Databento 称自身不加限制但须遵守原交易所许可，再分发要与交易所签约；Bloomberg 数据按终端协议只供本人使用。两者都只存本地 `data/` 下（不入库），测试夹具全部是合成数据。

## 测试证明了什么、没证明什么

测试文件：`test/data/test_futures_foreign.py`、`test/data/test_futures_foreign_sources.py`，夹具 `test/data/futures_foreign_fixtures.py`（合成的 CL 四个月份 + 布伦特三个月份，2026-10-01 至 12-31，含感恩节、圣诞、夏令时切换、周五周日发布、缺 `ts_ref`、删除记录、初步 / 日内结算价、价差和对不上合约表的记录）。

证明了：

- 在形状与官方一致的输入上，两个适配器的日线与夹具**独立算出**的期望（夹具按「前一开市日切换点到当天切换点」逐日循环生成，不经过适配器的 UTC 归属代码）逐位相同；同一份合成市场经两家的形状进来，统一日线、日历、合约表关键字段逐位相同；两家的分钟线逐位相同。
- 交易日归属与直白参考（「早于 D 当地切换点的最早开市日 D」）在随机时刻上一致；十年回绕与逐年枚举的参考在随机年份上一致。
- 默认预算下任何公开方法都在发请求前停下，`get_range` 调用数为 0；估价与取数用同一组参数；预算按累计计；缺可选依赖时报错写明装哪个 extra；密钥只取自环境变量，不出现在报错和日志里。
- 装了 `databento` 时：假客户端每个方法的参数名与官方 0.90.0 逐个相同；用 `databento_dbn` 把夹具编码成真正的 DBN 字节，经官方 `DBNStore.to_df()` 读出的表与假对象给的表在用到的列上逐位相同，整条适配器换算结果也相同。
- 走通适配器 → 面板 → 连续合约（与直白参考对拍）→ 换月日程表 → `panelbt_inputs` → `bt.Market`；面板版本随适配器输出内容变化。

没证明：真实服务返回的数据就是这个形状和含义（见下面的未核实清单）；`blpapi` 没有装（安装被本机权限拒绝），Bloomberg 一侧的方法名和响应形状只按文档核对过，假模块的签名没有与真库对拍。

### 变异检查

逐个改坏一处、跑上面两个测试文件，先确认测试导入的是 worktree 里被改坏的那份代码，并先跑不改动的对照（全过）。13 处全部被抓住：

| 改坏的地方 | 抓住它的测试 |
|---|---|
| 十年回绕：交割月等于参照月时也进位 | test_resolve_year_cases |
| 交易日切换点：整点不归次日 | test_trade_date_cases |
| 周日不顺延 | test_trade_date_cases |
| 休市日顺延方向反了 | test_trade_date_rolls_over_holiday |
| 前结算取下一开市日 | test_databento_daily_matches_independent_expectation |
| 默认预算 0 时估算为 0 也取数 | test_default_budget_never_fetches |
| 统计删除记录不生效 | test_databento_daily_matches_independent_expectation |
| 日内结算价不排除 | test_databento_daily_matches_independent_expectation |
| 初步结算价也收 | test_settlement_takes_final_not_preliminary_or_intraday |
| 持仓量缺 ts_ref 时归发布日本身 | test_databento_daily_matches_independent_expectation |
| 上市日错用到期时刻 | test_databento_daily_matches_independent_expectation |
| 收盘取第一根 K 线 | test_databento_daily_matches_independent_expectation |
| Bloomberg 收盘价错用结算价 | test_bloomberg_daily_matches_independent_expectation |

第一轮「十年回绕」漏网：原写法前后两个循环互相兜底，改坏一个另一个会纠正；简化成一步判断并补了「参照日就在交割月」的用例后抓住。

### 性能基线

交易日归属（100 万个分钟时刻，负载 8.4，三次中位数）：逐个 Python 的直白写法 1.47 秒，polars 向量化（正式实现）0.12 秒，结果逐位相同。合成夹具上两家适配器换算（三个月、两个品种、约 300 个合约日，含假客户端造表）：Databento 0.067 秒，Bloomberg 0.009 秒（负载 11）。

## 未核实清单

Databento：

1. `IFEU.IMPACT` 是否支持母代码 `BRN.FUT`（ICE 原始代码前有空格补位），`definition` 里 ICE 合约的 `asset`、`maturity_year`/`maturity_month` 是否有值。
2. ICE 原始代码写法 `BRN FMZ0026!` 是照文档里 2024 年 8 月合约 `BRN FMQ0024!` 推出来的。
3. 定义是否每个交易日重发：只在区间开始前发过一次定义的合约，`definition` 请求会取不到（适配器从 `start` 起取）。
4. CL、布伦特的 `display_factor` 与价格单位：假定 `to_df()` 的价格就是美元/桶。
5. 持仓量、结算量的 `ts_ref`：文档只对结算价写明 `ts_ref` 是交易日；没有 `ts_ref` 时的归属规则是推断。
6. CME 节假日前后的时段（感恩节等）合并到哪个交易日：按「休市日的成交顺延到下一开市日」处理。
7. `symbology.resolve` 是否计费、`d1` 是否不含；连续合约 `n`/`v` 的具体换月时刻。
8. `get_cost` 估价与实际扣费的差距。
9. 结算量是否含 TAS、大宗、EFP，以及与 CME 日报成交量是否一致。

Bloomberg：

1. `blpapi` 没装，只按 3.24.11 文档写；`Message.toPy()` 的响应字典形状（`securityData`、`fieldData`、`barData.barTickData`、`responseError`、`securityError`）照 HTTP API 文档和 pdblp 的请求日志推断。
2. 字段与覆盖名凭经验：`FUT_CHAIN`（元素 `Security Description`）、`INCLUDE_EXPIRED_CONTRACTS`、`FUT_CONT_SIZE`、`FUT_TICK_SIZE`、`FUT_FIRST_TRADE_DT`、`LAST_TRADEABLE_DT`、`FUT_DLV_DT_LAST`、`FUT_NOTICE_FIRST`、`CRNCY`、`PX_SETTLE`、`OPEN_INT`。Bloomberg 的字段字典（终端 FLDS）不公开。
3. 历史请求的 `date` 是否就是交易所交易日；`PX_LAST` 对期货是收盘价还是结算价；`OPEN_INT` 是否晚一天。
4. 已到期合约的代码是一位还是两位年份（两种都能解析）。
5. NYMEX 布伦特现金结算（BZ）的 Bloomberg 代码。
6. 通用代码的换月口径；分钟线 `time` 是开始时刻还是结束时刻。
7. 日期字段在 `toPy()` 里是 `datetime.date` 还是字符串（两种都接受）。

共同：

1. 日历由「有结算价的日子」推出，半天交易日照常算开市。
2. `activation` 可能早于真正的首个交易日；`listing_date` 只用于上市期过滤。
3. 合约表的 `delivery_date` 在 Databento 一侧为空。

## 首次核对清单

拿到 Databento 账号后，按顺序：

1. 预算 0 跑 `DatabentoFutures(DatabentoSource(), ["CL", "B"], "20261102", "20261106").estimate()`：看每个请求的估算，确认总额在几美元以内（不计费）。
2. `budget_usd=5` 取 CL 2026-11-02 至 11-06：
   - CLZ6、CLF7 每天的 `settlement` 与 CME 网站 Light Sweet Crude Oil「Settlements」页同一天的结算价逐个相同；
   - `open_interest`、`volume` 与 CME Daily Bulletin 同一交易日的持仓量、成交量相同（确认日期没有错一天）；
   - CLZ6 的 `last_trade_date` 等于 CME 产品日历上的最后交易日；`multiplier` = 1000，`tick_size` = 0.01，诊断里 `display_factor`、`maturity_conflicts`、`multiplier_mismatch` 为空；
   - 周一的 `open` 来自周日 17:00（芝加哥）以后的成交：`fut.minute("20261102", "20261102")` 的第一根 K 线在 2026-11-01 23:00 UTC 附近。
3. 同一周取布伦特（`IFEU.IMPACT`）：B2601 每天的结算价与 ICE Report Center 公布值相同，价格是美元/桶（不是美分）；母代码 `BRN.FUT` 能取到单月合约；原始代码写法与上面的推断一致。
4. 取 CL 2020-04-17 至 04-22：CLK0 2020-04-20 的结算价是 −37.63，`nonpositive_price_rows` > 0。
5. 节假日：取 CL 2026-11-25 至 11-30，确认感恩节（11-26）没有结算价、当天的成交归到 11-27，日历里 11-26 休市。
6. `reference_continuous=True` 取一个月：`CL.c.0` 在到期次日换月；`CL.n.0` 与 `panel.continuous("CL", rule="open_interest")` 逐日对应同一合约的比例（境内五个品种是 79%–99%）。

拿到 Bloomberg 终端后：

1. `BloombergSource().reference(["CL1 Comdty"], ["FUT_CHAIN"], overrides={"INCLUDE_EXPIRED_CONTRACTS": "Y"})`：看代码写法，确认已到期合约的年份位数。
2. `historical(["CLZ5 Comdty"], ["PX_SETTLE", "PX_LAST", "OPEN_INT", "PX_VOLUME"], "20251103", "20251107")`：结算价与 CME 公布值相同；`PX_LAST` 与结算价不同（是收盘价）；持仓量与 CME 日报同一天对上。
3. `reference(["CLZ5 Comdty", "COZ5 Comdty"], 合约表字段)`：最后交易日、合约规模与交易所公布一致；字段名都有返回（没有的会在 `fieldExceptions` 里）。
4. `intraday_bars("CLZ5 Comdty", 2025-11-04 23:00 UTC, 2025-11-05 22:00 UTC)`：时间是 UTC、是开始时刻，成交量之和与当天 `PX_VOLUME`（电子盘部分）接近。
5. 同一周比较两家（如果两家都有）：`a.daily().equals(b.daily())` 或逐字段差异。

## 出处（2026-11-13 读取，均未登录）

Databento（官方 Python 客户端 `databento` 0.90.0、`databento-dbn` 0.73.0，装在 worktree 的 `.venv` 里核对了方法签名和 `to_df()` 的列与类型）：

- OHLCV schema：`https://databento.com/docs/schemas-and-data-formats/ohlcv`（`ts_event` 为区间开始；`ohlcv-1d` 按 UTC 日；没有成交就没有 K 线；只含电子盘成交）
- Statistics schema：`https://databento.com/docs/schemas-and-data-formats/statistics`（`stat_type` 3 结算价、6 结算量、9 持仓量；`quantity` 不用时为 INT64_MAX；`update_action` 1 新增、2 删除）
- Instrument definitions：`https://databento.com/docs/schemas-and-data-formats/instrument-definitions`
- CME Globex MDP 3.0：`https://databento.com/docs/venues-and-datasets/glbx-mdp3`（`stat_flags` 各位含义；`ts_ref` = TradingReferenceDate；持仓量、结算量次日发布）
- ICE Europe Commodities：`https://databento.com/docs/venues-and-datasets/ifeu-impact`（原始代码示例 `BRN FMQ0024!`）
- 数据集列表：`https://databento.com/docs/venues-and-datasets`
- 符号体系：`https://databento.com/docs/standards-and-conventions/symbology`（`parent`、`continuous`，`c`/`n`/`v` 规则）
- `metadata.get_cost`：`https://databento.com/docs/api-reference-historical/metadata/metadata-get-cost`
- `symbology.resolve`：`https://databento.com/docs/api-reference-historical/symbology/symbology-resolve`

Bloomberg（`blpapi` Python 文档 3.24.11，包本身未安装）：

- `https://bloomberg.github.io/blpapi-docs/python/3.24.11/` 下的 `Session`、`SessionOptions`、`Request`、`Element`、`Event`、`Message` 页（方法签名、`Event` 常量、`toPy()`、默认 127.0.0.1:8194）
- `blpapi` 包索引：`https://blpapi.bloomberg.com/repository/releases/python/simple/`（2026-11-13 最新 3.26.10.1）
- 官方示例 `IntradayBarExample.py`（blpapi 3.16.1 发行包，第三方镜像 gemfury）：`barData` → `barTickData` 的 `time`、`open`、`high`、`low`、`close`、`volume`、`numEvents`，时间为 GMT
- `https://github.com/bloomberg/blpapi-http/blob/develop/doc/http-api-guide.md`：`HistoricalDataRequest`、`ReferenceDataRequest` 的请求元素与 `securityData`/`fieldData` 响应形状
- pdblp 教程 `https://matthewgilbert.github.io/pdblp/tutorial`：`HistoricalDataRequest` 的完整请求日志与响应嵌套；`FUT_CHAIN` 批量字段的元素名 `Security Description`

品种规则：`docs/markets/futures_rules.md` 第二部分（ICE 产品页、CFTC 存档的 NYMEX 文件）。

## 需要现有文件配合的地方（这次没改）

1. `futures_schema.py`：把分钟线表 `MINUTE` 挪进来（加进 `TABLES` 但不进 `REQUIRED_TABLES`），`conform` 就能直接检查分钟线；现在定义在 `futures_foreign.MINUTE`。
2. `store.py`：`Store` 只认 `market="cn"`，境外面板只好放在单独数据根的 `cn/futures_panel` 下；建议加一个 `market="intl"`（一行：允许的市场集合）。
3. `futures_panel.build_futures`：版本号由清单、代码、来源名组成，境外没有清单，现在靠适配器名字里带内容哈希；建议改成优先读适配器的 `content_version` 属性（一行 `getattr`）。另外 `CODE_FILES` 不含 `futures_foreign*`，只改换算代码、不改输出时面板版本不变（输出变了会随内容哈希变）。
4. `docs/FUTURES_DATA.md`「境外来源适配器的接口约定」写的是「价格为正」：境外按真实值保留负价（WTI 2020-04-20），建议在那里加一句。

已经改了的一处现有测试：`test/data/test_layers.py::test_source_independence` 只允许取数层导入标准库和 httpx、polars、pyarrow、baostock，这次在允许名单里加了 `databento`、`blpapi` 两个官方客户端（任务要求用官方库；只加名字，没有放宽其他检查）。
