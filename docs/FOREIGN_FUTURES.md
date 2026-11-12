# 境外期货：Databento 与 Bloomberg 适配器

> **未用真实账号验证。** 两个适配器都是照官方文档、官方 Python 客户端（Databento）的接口和返回形状写的，测试全部用合成数据和假客户端。拿到账号后先跑文末的「首次核对清单」，逐条对上公开数字以后再用于研究。

把境外期货（NYMEX WTI、ICE 布伦特、NYMEX 布伦特现金结算）的合约表、日线、分钟线、日历转成与境内期货相同的统一字段（`ashare.data.futures_schema`），面板、连续合约、`panelbt_inputs` 都不用改。设计见 `docs/DESIGN.md` 6.1.2。

## 最短用法

```bash
pip install "ashare[databento]"     # 或 uv sync --extra databento
export DATABENTO_API_KEY=...        # 只从环境变量读，不写进任何文件
# Bloomberg：blpapi 只在 Bloomberg 自己的包索引上，不是本仓库的可选依赖，需要时手动装
uv pip install --index-url https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi
```

```python
from ashare.data.sources.databento import DatabentoSource
from ashare.data.futures_foreign_databento import DatabentoFutures
from ashare.data.futures_foreign import build_panel, write_parquet

src = DatabentoSource()                                   # 默认预算 0：只估价，不取数
fut = DatabentoFutures(src, ["CL"], "20261102", "20261106", margin={"CL": 0.1})
print(fut.estimate())                                      # 各请求的估算费用（美元），不计费
src.budget_usd = 2.0                                       # 显式给预算才会取数（整份计划先估总价，超了一个都不取）
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
| 取数 | `sources/databento.py` | 建 `databento.Historical`（密钥取自环境变量）；**取第一个请求前先对整份计划 `metadata.get_cost`**，累计超预算抛 `BudgetExceeded`（计划总价、已花、差额），一个 `timeseries.get_range` 都不发；已取的表会话内缓存；`DBNStore.to_df()` → polars，字段保留原名 |
| 取数 | `sources/bloomberg.py` | 同步 `Session`，`//blp/refdata` 的 `HistoricalDataRequest`、`ReferenceDataRequest`、`IntradayBarRequest`；响应用 `Message.toPy()` 原样转成字典；`responseError`、`securityError`、`fieldExceptions` 都报 `BloombergError` |
| 换算 | `futures_foreign.py` | 品种表、十年回绕、交易日归属、分钟线字段 `MINUTE`、日历、前结算、参数表、落盘 `write_parquet`、接面板 `build_panel` |
| 换算 | `futures_foreign_databento.py`、`futures_foreign_bloomberg.py` | 两家的原始表 → 统一字段；`DatabentoFutures`、`BloombergFutures` 是适配器（`contracts`、`daily`、`params`、`calendar`、`mapping`、`source_continuous`、`diagnostics`、`minute`） |

可选依赖只有 `ashare[databento]`（`databento>=0.90`）。`blpapi` 只在 Bloomberg 的包索引上，若登记成可选依赖，`uv.lock` 就会依赖那台主机（锁文件解析、`uv sync --locked` 都要连它），所以不登记，需要时手动装：`uv pip install --index-url https://blpapi.bloomberg.com/repository/releases/python/simple/ blpapi`。都没装时 `make check` 照样全过；调用到取数层才报错并给出安装方法。换算层不依赖官方库。

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
| name | 品种表名称 + 年月 | `NAME`（缺了留空） |
| listing_date | `activation` 按交易时段归属的交易日 | `FUT_FIRST_TRADE_DT`（缺了留空） |
| last_trade_date | `expiration`（最后可成交时刻）按交易时段归属的交易日 | `LAST_TRADEABLE_DT` |
| delivery_date | 空（来源没有） | `FUT_DLV_DT_LAST`（缺了留空） |
| delivery_month | 年月 | 年月 |
| multiplier | `unit_of_measure_qty`（`unit_of_measure` = BBL；与品种表不同时记进诊断 `multiplier_mismatch`） | `FUT_CONT_SIZE` |
| quote_unit | 品种表（美元/桶） | 品种表 |
| tick_size | `min_price_increment` | `FUT_TICK_SIZE` |
| settlement_type | 品种表 | 品种表 |
| currency | `currency`（USD） | `CRNCY`（USD） |
| multiplier_source | `contract_table_unverified`（没有成交额可反查） | 同左 |

Databento 的定义：只要 `instrument_class == "F"`（价差是 S，代码带 `-`）；同一 `instrument_id` 取 `ts_recv` 最晚的一条，`security_update_action == "D"` 的删掉。`display_factor` 不是 1 的记进诊断 `display_factor`。请求窗口里同一个 `instrument_id` 对应几个代码（来源复用了 id）时报错，不合并成一个合约。

**乘数、最小变动价位、币种缺了就报错，不回退到品种表**（品种表的值只用来与来源比对）：Databento 的 `unit_of_measure_qty`、`min_price_increment`、`currency`，Bloomberg 的 `FUT_CONT_SIZE`、`FUT_TICK_SIZE`、`CRNCY`、`LAST_TRADEABLE_DT`。所以合约表里这几列一定来自来源本身。

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
| open_interest | `stat_type == 9` 的 `quantity`；有 `ts_ref` 用它的 UTC 日期，没有时归发布时刻所属交易日的上一开市日（**未核实**）；`≥ 2147483647` 视为空（v2 起未定义值是 `INT64_MAX`，DBN v1 是 int32 的 `2147483647`，官方 `to_df()` 读 v1 时不升级） | `OPEN_INT` |

- 单位：价格保持来源报价单位（美元/桶），不换算成美分；成交量、持仓量都是手，单边。
- 为什么不用 `ohlcv-1d`：官方说明它按 **UTC 日** 切分；CME 的交易日从前一天 17:00（芝加哥）开始，UTC 日切分会把夜盘劈成两半。CME 17:00、ICE 23:00 都在整点，小时线能按交易时段干净切开。
- 取数窗口比 `[start, end]` 宽（见「费用与密钥」），多取的几天只用来推开市日、算 start 当天的 `previous_settlement`、归属 start 那个交易日从上一开市日傍晚开始的 K 线，**不并进区间**：交易日落在 start 之前的 K 线归它们自己的那天、再被裁掉；end 傍晚之后归下一交易日的 K 线同样裁掉，end 那天的夜盘（上一开市日傍晚起）照常算进 end。
- 结算价完整性（`check_settlements`）：诊断 `settlement_missing` 给每个品种区间内结算价为空的行数。某个品种区间内一个结算价都没有时报错（来源的结算价口径与预期不同时，日历会推成全部休市，后面全错）；某个合约某天有成交（成交量 > 0 或有开盘价）却没有最终结算价，默认也报错，`require_settlement=False` 放行（比如 end 当天结算价还没公布）。
- 只留 `[start, end]` 且在合约上市期内的行；对不上合约表的 `instrument_id`（价差、其他品种）计数在 `unknown_instrument_bar_rows`、`unknown_instrument_stat_rows`，上市期外丢掉的计数在 `outside_listing_rows`。

### 负价格在各环节的表现

WTI 2020-04-20 的结算价是负数（CLK0，−37.63）。境内统一字段「价格为正」针对的是源把 0 当占位；境外是特许的例外，按真实值保留：

| 环节 | 负价格 |
|---|---|
| 适配器（两家）、`write_parquet` | 原样保留，个数记在诊断 `nonpositive_price_rows` |
| 面板 `build_panel`、`continuous(adjust="none")`、`roll_schedule` | 原样保留 |
| `continuous(adjust="difference")` | 正常（加减偏移，不受符号影响） |
| `continuous(adjust="ratio")` | **报错**，提示改用 `difference`：换月处新旧价格异号时比例因子变号，整段历史的符号会被翻过来 |
| `panelbt_inputs` | 原样放进数组 |
| `bt.Market` | **报错**（`exec_price: expected finite positive values or NaN`）：回测引擎不接受非正的执行价，含负价的区间不能直接回测，要么避开这段，要么自己换一个执行价口径 |

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
- `calendar`：两家都没有日历接口。开市日 = 该交易所**有 final 结算价**的交易日，区间内其余自然日为休市。整交易所一天都没有结算价才算休市；半天交易的节假日（感恩节次日等）照常开市。适配器内部的日历覆盖整个取数窗口（start 前约一周到 end 后几天），交出的日历裁到 `[start, end]`。区间里落在已知开市日范围之外的日子（比如 end 当天的结算价还没公布）记为休市，并列在诊断 `calendar_unknown_days` 里。

## 交易日归属

`futures_foreign.trade_dates`：把 UTC 时刻换到交易所时区，本地时间 ≥ 切换点的归下一天，再顺延过周六周日，给了开市日集合时继续顺延到下一个开市日。

**只在已知日历范围内顺延**：已知范围是开市日集合的最早、最晚一天（或显式传 `known=(首日, 末日)`）。自然交易日落在范围之外、或顺延越过最后一个开市日的，返回空，由调用方丢弃并计数（诊断 `bars_outside_calendar`），不会被顺延到范围的第一天。复核发现的原缺陷就是这里：开市日只覆盖 `[start, end]` 时，start 之前三天的 K 线被顺延进 start，start=2026-10-05 的 CLZ6 开盘价、电子盘成交量都错了。

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

- Databento 按量计费。`DatabentoSource.fetch_plan(请求列表)` 在取第一个请求前，先对计划里**全部**还没取过的请求用**同样的参数**调 `metadata.get_cost`（免费），已批准累计 `spent_usd` 加计划总价不超过 `budget_usd` 才开始逐个 `timeseries.get_range`；**预算为 0 时永远不取**（估算为 0 也不取）。超预算抛 `BudgetExceeded`：`plan_total`（完成这次取数还要多少）、`spent`（此前已批准）、`budget`、`shortfall`（至少还差多少）和逐个请求的估价，消息里写明「把 `budget_usd` 调到不低于 `spent + plan_total`」，照做后重试即可。适配器的日线一次就是一份计划（每个数据集三个请求），分钟线另一份。`fut.estimate()` 只估价、不计费。每次估价记在 `source.log`，也进诊断 `cost_log`。
- 已取到的原始表按请求参数缓存在 `DatabentoSource` 对象里：同一会话里重试、换个适配器取同样的请求，都不重取、不重计。请求发出后才失败（网络断了）时，那个请求的估价已经记进 `spent_usd`（发出就可能已计费，宁可多记），重试时它要重新过闸门。
- 预算和估价都必须是有限的非负数：预算是 NaN、无穷、负数在建对象和每次取数时都报 `ValueError`，是字符串、布尔、空值报 `TypeError`；`get_cost` 返回 NaN、负数、非数字同样报错，不取数。检查、记账和缓存在一把锁里，多线程共用一个对象时预算不会被并发越过。`spent_usd`、`budget_usd` 是公开属性，只应由本对象改；改坏了（NaN 等）下一次取数会报错。
- 不要 pickle 适配器或 `DatabentoSource`：官方客户端对象里带着密钥。
- 官方说明：估价按请求范围，账单按实际发送字节；起止不对齐 10 分钟边界时估价可能偏高。所以预算闸门是上限保护，不是精确账单。
- 一次日线取数的请求：每个数据集 `definition`（[start−7, end+1)）、`statistics`（[start−7, end+7)：开市日要覆盖整个 K 线窗口，start 当天要有前结算价，持仓量、结算量在下一个 UTC 日或周日发布）、`ohlcv-1h`（[start−5, end+1)：start 那个交易日从上一开市日傍晚开始，周末加休市日也够；上一开市日比这还早时报错，不悄悄少算几个小时）。分钟线 `minute(a, b)` 只接受落在 `[start, end]` 里的区间（否则不估价直接报错），取 [a 的上一开市日, b+1)。
- Bloomberg 的历史请求从 start 前 7 天取到 end（同样为了日历和 start 当天的前结算价），分钟线逐个合约取 [a 的上一开市日 UTC 0 点, b 次日 UTC 0 点)。
- 密钥只从环境变量 `DATABENTO_API_KEY` 读，不写进任何文件、日志、诊断或报错；没设时报「环境变量没有设置」，不打印环境。Bloomberg 桌面 API 走本机终端，不需要密钥。
- 数据许可：Databento 称自身不加限制但须遵守原交易所许可，再分发要与交易所签约；Bloomberg 数据按终端协议只供本人使用。两者都只存本地 `data/` 下（不入库），测试夹具全部是合成数据。

## 测试证明了什么、没证明什么

测试文件：`test/data/test_futures_foreign.py`、`test/data/test_futures_foreign_sources.py`、`test/data/test_futures_foreign_windows.py`（区间两端、费用闸门、完整性检查），夹具 `test/data/futures_foreign_fixtures.py`（合成的 CL 五个月份 + 布伦特五个月份，数据覆盖 2026-09-01 至 2027-01-15，比用例的区间两头都宽；含劳动节、感恩节、圣诞、元旦、夏令时切换、周五周日发布、缺 `ts_ref`、删除记录、初步 / 日内结算价、价差和对不上合约表的记录；定义每天 UTC 0 点重发）。

证明了：

- 在形状与官方一致的输入上，两个适配器的日线与夹具**独立算出**的期望（夹具按「前一开市日切换点到当天切换点」逐日循环生成，不经过适配器的 UTC 归属代码）逐位相同，**对七组不同的 start/end 各跑一遍**（周一、感恩节后的周五、月初且跨夏令时、圣诞前一天结束、圣诞后的周一且跨元旦、end 是数据最后一天、整段），两种成交量口径都比；start 当天的前结算价也对；分钟线跨四个交易日（含周一）与夹具的期望逐位相同；同一份合成市场经两家的形状进来，统一日线、日历、合约表关键字段逐位相同；两家的分钟线逐位相同。
- 交易日归属与直白参考（「早于 D 当地切换点的最早开市日 D」）在随机时刻上一致；十年回绕与逐年枚举的参考在随机年份上一致。
- 默认预算下任何公开方法都在发请求前停下，`get_range` 调用数为 0；估价与取数用同一组参数；预算按累计计；整份计划先估：预算只够前两个请求时一个都不取，报出的总价、差额对，照报错调预算后重试每个请求只取一次；中途网络失败后重试不重取已取到的；预算、估价为 NaN、无穷、负数、字符串、布尔都在取数前报错；八个线程共用一个只够一次的预算时只有一个取到；缺可选依赖时报错写明怎么装；密钥只取自环境变量，不出现在报错和日志里。
- 完整性：ICE 结算价没有 final 位时报错而不是「全部休市」；有成交缺结算价默认报错、放宽后诊断计数对；Bloomberg 历史请求的 `securityError`、两种请求的 `fieldExceptions`、`PX_SETTLE` 整列为空、某合约没有结算价、合约表缺必需字段都报错；定义里的乘数与品种表不同时用定义的值；定义缺乘数 / 最小变动 / 币种、一个 id 两个代码都报错；DBN v1 的未定义数量（真库编码、官方 `to_df()` 读出）当空；`minute` 区间外不花钱就报错；负价格走通适配器和面板，`ratio` 复权报错、`difference` 正常。
- 装了 `databento` 时：假客户端每个方法的参数名与官方 0.90.0 逐个相同；用 `databento_dbn` 把夹具编码成真正的 DBN 字节，经官方 `DBNStore.to_df()` 读出的表与假对象给的表在用到的列上逐位相同，整条适配器换算结果也相同。
- 走通适配器 → 面板 → 连续合约（与直白参考对拍）→ 换月日程表 → `panelbt_inputs` → `bt.Market`；面板版本随适配器输出内容变化。

没证明：真实服务返回的数据就是这个形状和含义（见下面的未核实清单）；`blpapi` 没有装（安装被本机权限拒绝），Bloomberg 一侧的方法名和响应形状只按文档核对过，假模块的签名没有与真库对拍。

### 变异检查

逐个改坏一处、跑上面三个测试文件（`-x`），先确认测试导入的是 worktree 里被改坏的那份代码，并先跑不改动的对照（全过）。28 处全部被抓住（第一轮 13 处，复核返工后加 15 处，含复核漏网的 M5、M10 和原缺陷 P0）：

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
| M5 定义乘数被忽略、一律取品种表 | test_definition_multiplier_is_used_not_product_table |
| M10 Bloomberg 历史请求不检查证券错误 | test_bloomberg_historical_security_error_raises |
| P0 已知日历范围外仍顺延（复核发现的原缺陷） | test_trade_dates_outside_known_calendar_are_dropped |
| P0 统计不往 start 前多取（日历只覆盖区间） | test_databento_daily_matches_independent_expectation |
| 闸门：缓存失效、重试重复取 | test_budget_is_cumulative |
| 闸门：去掉锁 | test_budget_holds_under_concurrency |
| 闸门：估价 NaN / 负数放行 | test_bad_cost_estimate_blocks_fetch |
| DBN v1 未定义数量当真值 | test_dbn_v1_undefined_quantity_read_by_official_library_is_missing |
| 一个 id 多个代码不报错 | test_instrument_id_reused_for_another_contract_raises |
| minute 区间外不拦 | test_minute_outside_adapter_range_raises_before_spending |
| 有成交缺结算价不报错 | test_traded_day_without_settlement_raises_unless_relaxed |
| Bloomberg `fieldExceptions` 被忽略 | test_bloomberg_field_exceptions_raise |
| Bloomberg 合约表缺字段不报错 | test_bloomberg_missing_contract_field_raises_not_falls_back |
| Bloomberg `PX_SETTLE` 整列为空不报错 | test_bloomberg_settle_column_all_empty_raises |
| `ratio` 复权不拦非正价格 | test_ratio_adjust_rejects_nonpositive_prices |

第一轮「十年回绕」漏网：原写法前后两个循环互相兜底，改坏一个另一个会纠正；简化成一步判断并补了「参照日就在交割月」的用例后抓住。这一轮「去掉锁」第一次漏网：并发测试里八个线程取的是不同请求，估价之后的「检查 + 记账」在 GIL 下本来就不交错；补了「八个线程同时取同一个请求」（没有锁时都在估价时看到没缓存、各取一次）后抓住，连跑三次都抓住，正常代码连跑五次都过。

### 性能基线

交易日归属（100 万个分钟时刻，负载 8.4，三次中位数）：逐个 Python 的直白写法 1.47 秒，polars 向量化（正式实现）0.12 秒，结果逐位相同。合成夹具上两家适配器换算（三个月、两个品种、约 300 个合约日，含假客户端造表）：Databento 0.067 秒，Bloomberg 0.009 秒（负载 11）。

## 未核实清单

Databento：

1. `IFEU.IMPACT` 是否支持母代码 `BRN.FUT`（ICE 原始代码前有空格补位），`definition` 里 ICE 合约的 `asset`、`maturity_year`/`maturity_month` 是否有值。
2. ICE 原始代码写法 `BRN FMZ0026!` 是照文档里 2024 年 8 月合约 `BRN FMQ0024!` 推出来的。
3. 定义是否每个交易日重发：官方通用说明是从 UTC 0 点起满 24 小时的请求含期间所有在场合约的定义（适配器从 start−7 起取）；夹具按每天重发造数据。
4. CL、布伦特的 `display_factor` 与价格单位：假定 `to_df()` 的价格就是美元/桶。
5. 持仓量、结算量的 `ts_ref`：文档只对结算价写明 `ts_ref` 是交易日；没有 `ts_ref` 时的归属规则是推断。
6. CME 节假日前后的时段（感恩节等）合并到哪个交易日：按「休市日的成交顺延到下一开市日」处理。
7. `symbology.resolve` 是否计费、`d1` 是否不含；连续合约 `n`/`v` 的具体换月时刻。
8. `get_cost` 估价与实际扣费的差距。
9. 结算量是否含 TAS、大宗、EFP，以及与 CME 日报成交量是否一致。
10. `IFEU.IMPACT` 的 `stat_flags`（final、日内结算位）和 `ts_ref` 含义：公开文档只在 GLBX.MDP3 页写了。ICE 的结算价若没有 final 位或没有 `ts_ref`，适配器会报「一个最终结算价都没有」，不会静默变空。
11. 服务端是否还会给 DBN v1（int32 数量）：0.90 客户端取的是 v3；读旧缓存时按 `≥ 2147483647` 当空处理。
12. CME 的 `instrument_id` 会不会复用：适配器按 id 取最后一条定义，请求窗口里一个 id 对应几个代码时报错；长区间回补跨年时要分段取并留意。

Bloomberg：

1. `blpapi` 没装，只按 3.24.11 文档写；`Message.toPy()` 的响应字典形状（`securityData`、`fieldData`、`barData.barTickData`、`responseError`、`securityError`）照 HTTP API 文档和 pdblp 的请求日志推断。
2. 字段与覆盖名凭经验：`FUT_CHAIN`（元素 `Security Description`）、`INCLUDE_EXPIRED_CONTRACTS`、`FUT_CONT_SIZE`、`FUT_TICK_SIZE`、`FUT_FIRST_TRADE_DT`、`LAST_TRADEABLE_DT`、`FUT_DLV_DT_LAST`、`CRNCY`、`PX_SETTLE`、`OPEN_INT`；`PX_OPEN`、`PX_HIGH`、`PX_LOW`、`PX_VOLUME` 没找到公开出处；`periodicitySelection = DAILY` 的元素名与取值也没找到公开出处。Bloomberg 的字段字典（终端 FLDS）不公开。字段名错了会进 `fieldExceptions`，现在直接报错；可选字段（`NAME`、`FUT_FIRST_TRADE_DT`、`FUT_DLV_DT_LAST`）没返回时留空。
3. 历史请求的 `date` 是否就是交易所交易日；`PX_LAST` 对期货是收盘价还是结算价；`OPEN_INT` 是否晚一天。
4. 已到期合约的代码是一位还是两位年份（两种都能解析）。
5. NYMEX 布伦特现金结算（BZ）的 Bloomberg 代码。
6. 通用代码的换月口径；分钟线 `time` 是开始时刻还是结束时刻（若是结束时刻，所有分钟线偏一分钟，ICE 每天 22:59–23:00 那根会被归到下一天，静默错位）。
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
3. 同一周取布伦特（`IFEU.IMPACT`）：B2601 每天的结算价与 ICE Report Center 公布值相同，价格是美元/桶（不是美分）；**诊断 `settlement_missing` 里 B 为 0**（ICE 的 `stat_flags`、`ts_ref` 口径与 GLBX 一致）；母代码 `BRN.FUT` 能取到单月合约；原始代码写法与上面的推断一致。
   - 再取一段跨年的长区间（如 2024 全年）看会不会报「同一个 instrument_id 对应几个合约代码」：会的话说明 CME 复用 id，要分段取。
4. 取 CL 2020-04-17 至 04-22：CLK0 2020-04-20 的结算价是 −37.63，`nonpositive_price_rows` > 0。
5. 节假日：取 CL 2026-11-25 至 11-30，确认感恩节（11-26）没有结算价、当天的成交归到 11-27，日历里 11-26 休市。
6. `reference_continuous=True` 取一个月：`CL.c.0` 在到期次日换月；`CL.n.0` 与 `panel.continuous("CL", rule="open_interest")` 逐日对应同一合约的比例（境内五个品种是 79%–99%）。

拿到 Bloomberg 终端后：

1. `BloombergSource().reference(["CL1 Comdty"], ["FUT_CHAIN"], overrides={"INCLUDE_EXPIRED_CONTRACTS": "Y"})`：看代码写法，确认已到期合约的年份位数。
2. `historical(["CLZ5 Comdty"], ["PX_SETTLE", "PX_LAST", "OPEN_INT", "PX_VOLUME"], "20251103", "20251107")`：结算价与 CME 公布值相同；`PX_LAST` 与结算价不同（是收盘价）；持仓量与 CME 日报同一天对上。
3. `reference(["CLZ5 Comdty", "COZ5 Comdty"], 合约表字段)`：最后交易日、合约规模与交易所公布一致；字段名都有返回（`fieldExceptions` 为空，否则现在会直接报错）。
4. `intraday_bars("CLZ5 Comdty", 2025-11-04 23:00 UTC, 2025-11-05 22:00 UTC)`：时间是 UTC、是开始时刻，成交量之和与当天 `PX_VOLUME`（电子盘部分）接近。再取 COZ5 一天，看伦敦 22:59 开始的那根（如有）归到哪天：`time` 若是结束时刻，它会显示成 23:00 并被归到下一天。
5. 同一周比较两家（如果两家都有）：`a.daily().equals(b.daily())` 或逐字段差异。

Bloomberg 一侧会**静默给出错数**的五种情形，首次核对必须逐条排除（能变成报错的已经改成报错）：

| 情形 | 现在的处理 | 怎么排除 |
|---|---|---|
| 字段名无效或不适用，进了 `fieldExceptions` | **已改为报错**（列出字段名） | 第 2、3 条跑通即排除 |
| 合约表字段缺失时回退到品种表 | **已改为报错**（乘数、最小变动、币种、最后交易日）；名称、上市日、交割日留空 | 第 3 条 |
| 期货的 `PX_LAST` 其实是结算价 | 无法在代码里判断 | 第 2 条：`PX_LAST` 与 `PX_SETTLE` 不同 |
| `OPEN_INT` 比交易所日报晚一天 | 无法在代码里判断 | 第 2 条：持仓量与 CME 日报同一天对上 |
| 分钟线 `time` 是 K 线结束时刻 | 无法在代码里判断 | 第 4 条：看 22:59 那根归哪天 |

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

## 后续（这一轮不做）

1. 分钟线表并入 `futures_schema`：把 `MINUTE` 挪进去（加进 `TABLES` 但不进 `REQUIRED_TABLES`），`conform` 就能直接检查分钟线；现在定义在 `futures_foreign.MINUTE`。
2. `Store` 加 `market="intl"`：`Store` 只认 `market="cn"`，境外面板只好放在单独数据根的 `cn/futures_panel` 下。
3. `build_futures` 的版本来源：版本号由清单、代码、来源名组成，境外没有清单，现在靠适配器名字里带内容哈希；建议改成优先读适配器的 `content_version` 属性。另外 `CODE_FILES` 不含 `futures_foreign*`，只改换算代码、不改输出时面板版本不变（输出变了会随内容哈希变）。

## 改了的现有文件

- `src/ashare/data/futures_continuous.py`：只在 `adjust_prices` 的 `ratio` 路径加了非正价格检查（报错并建议 `difference`）；境内价格都为正，输出逐位不变。
- `docs/FUTURES_DATA.md`「境外来源适配器的接口约定」的「价格为正」处加了一句：境外的真实负价格特许保留。
- `test/data/test_layers.py::test_source_independence`：取数层允许导入的名单里加了 `databento`、`blpapi` 两个官方客户端（只加名字，没有放宽其他检查）。
