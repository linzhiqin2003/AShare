# 数据源规范（v1，2026-11-15）

本规范规定**外部数据接入 ashare 的统一格式**。无论数据来自自己的数据库、别家数据商还是一份 CSV 导出，只要整理成符合本规范的**数据包**，就能用同一套工具校验，读出来直接交给因子引擎和 `panelbt`。

本文用「必须」「不得」「应当」「可以」四个词区分要求的强弱。凡是写了「必须」的地方，校验工具（第 8 节）都会检查。

## 1. 三层数据

| 层 | 内容 | 是否受本规范约束 |
|---|---|---|
| 来源层 | 原样保存的下载结果，例如 Tushare 原始库、币安日度归档 | 否，各适配器自定 |
| **标准层** | 长表 parquet 加一份清单 `datapackage.json` | **是：外部数据只从这一层进入** |
| 面板层 | 为计算速度建的 T×N 数组 | 否，由标准层构建，使用者不用管 |

已有的数据族（A 股、期货、港美股、基金可转债、加密货币）为了性能，内部格式保持不变。它们各配一个导出器，能产出标准层视图，第 10 节列出对照表。**新增的数据源一律先落到标准层。**

## 2. 数据包

```
<包目录>/
  datapackage.json            清单
  bars/2024.parquet           表名/分区.parquet
  bars/2025.parquet
  instruments/all.parquet
  calendar/all.parquet
```

- 一个包是**一个来源、同一口径**的一批数据，例如「某数据商的 A 股日线，2010–2026」。口径不同（复权与不复权、两个数据商）就分成两个包。
- 一张表在清单里是一个 resource，它的文件放在以表名命名的子目录里。
- 分区按时间划分，名字只能是 `YYYY`、`YYYY-MM` 或 `all`（不分区的小表）。一个分区就是一个文件，同一行不得同时出现在两个分区里。
- 文件只能是 parquet，压缩用 zstd 或 snappy。
- 文件写完就不再改。追加数据就新增分区文件，修正历史就换掉文件，两种情况都会得到一个新版本（第 7 节）。

清单采用 [Data Package v2](https://datapackage.org/) 的格式（`resources`、Table Schema 的 `fields` 和 `primaryKey`），本规范要求的扩展属性都用驼峰命名。因此，别处已有的 Data Package 工具也能读懂这份清单的结构。

Table Schema 的字段类型与 parquet 列类型一一对应：

| Table Schema `type` | parquet 列类型 |
|---|---|
| `date` | `date32` |
| `datetime` | `timestamp`，单位 ms 或 us，时区 UTC |
| `number` | `float64` |
| `integer` | `int64`（`quality` 列为 `uint32`） |
| `boolean` | `bool` |
| `string` | `string` |

一份清单的描述部分（`seal` 之前）大致如下：

```json
{
  "specVersion": "1",
  "name": "vendor-a-cn-daily",
  "title": "某数据商 A 股日线",
  "source": "某数据商，2026-11 导出",
  "licenses": [{"name": "proprietary"}],
  "symbology": "cn",
  "calendar": {"name": "SSE", "tz": "Asia/Shanghai", "sessionOpen": "09:30", "sessionClose": "15:00"},
  "resources": [
    {
      "name": "bars",
      "kind": "bars",
      "frequency": "1d",
      "schema": {
        "fields": [
          {"name": "symbol", "type": "string"},
          {"name": "date", "type": "date"},
          {"name": "open", "type": "number"},
          {"name": "close", "type": "number"},
          {"name": "volume", "type": "number"},
          {"name": "turnover_rate", "type": "number", "unit": "ratio", "knownAt": "end", "description": "换手率"}
        ],
        "primaryKey": ["symbol", "date"]
      }
    },
    {"name": "instruments", "kind": "instruments", "schema": {"fields": [{"name": "symbol", "type": "string"}, {"name": "asset_type", "type": "string"}, {"name": "currency", "type": "string"}], "primaryKey": ["symbol"]}},
    {"name": "calendar", "kind": "calendar", "schema": {"fields": [{"name": "date", "type": "date"}], "primaryKey": ["date"]}}
  ]
}
```

`kind` 取 `bars`、`instruments`、`calendar`、`corporate_actions`、`events` 之一。`events` 表示其他带 `available_at` 的时间序列，比如财务和公告。`seal` 会在每个 resource 下补上 `path`（文件列表）、`bytes`、`hash`、`rows`，并在顶层补上 `version` 和 `created`。

## 3. 时间

### 3.1 时间列（每张时间序列表恰好一种）

- **日频及更低频率**：列名为 `date`，类型为 Arrow `date32`，取交易所本地的交易日。夜盘归到下一个交易日。
- **日内频率和 24 小时市场**：列名为 `ts`，类型为 Arrow `timestamp`，单位用 `ms` 或 `us`，时区**必须**是 `UTC`。
- 以下写法一律拒收，因为都容易引起误读：
  - 不带时区的时间戳；
  - 整数形式的 epoch：分不清毫秒和微秒，币安归档在 2025 年就从毫秒改成了微秒；
  - `YYYYMMDD` 字符串。

### 3.2 区间

每行代表一个左闭右开的区间 `[开始, 开始 + 频率)`，时间列写区间的**开始**。日线的 `date` 就是那个交易日。频率写在 resource 的 `frequency` 属性里，可选值为 `1m`、`5m`、`15m`、`30m`、`1h`、`4h`、`1d`、`1w`、`1M`。

### 3.3 可知时点（防前视，本规范最重要的一条）

每个字段都要声明 `knownAt`，即这个值最早在什么时候能被看到：

| 写法 | 含义 | 典型字段 |
|---|---|---|
| `start` | 区间开始时已知 | 开盘价、前收、涨跌停价、当日停牌、复权因子 |
| `end` | 区间结束时已知 | 收盘价、最高价、最低价、成交量 |
| `end+<时长>` | 区间结束后再过一段时间才知道 | 下一交易日才公布的指数权重：`end+1d` |
| `column:<列名>` | 由同一行的某个时间列给出 | 财报、公告、事后修订：`column:available_at` |

- 日线的「开始」和「结束」按包的日历（3.4）换算成具体时刻。
- 对行情表，字段不写 `knownAt` 时，按第 5 节词典的默认值处理；不在词典里的字段必须写明。
- 非行情的时间序列表（公司行为、财务、公告、新闻）**必须**有 `available_at` 列，类型为 `timestamp` UTC，所有字段都声明为 `column:available_at`。
- `available_at` 不得早于该行所描述的事件或区间的开始。

### 3.4 日历

清单顶层写 `calendar`：

```json
"calendar": {"name": "SSE", "tz": "Asia/Shanghai", "sessionOpen": "09:30", "sessionClose": "15:00"}
```

- `tz` 用 IANA 时区名。
- 日频的包**必须**带 `calendar` 表（一列 `date`，列出全部交易日），这样才能区分「这天没成交」和「这天不开市」。
- 全天候交易的市场写 `"name": "24x7"`，不需要日历表。

## 4. 标识符

- 行情表和事件表都有 `symbol` 列（string）。在一个包里，一个 `symbol` 自始至终只指同一个可交易品种。
- 清单顶层的 `symbology` 声明代码写法，只能取下表登记过的值：

| 值 | 写法 | 例子 |
|---|---|---|
| `cn` | 代码.交易所 | `600000.SH`、`000001.SZ`、`430047.BJ`、指数 `000300.SH`、`H30269.CSI` |
| `hk` | 五位代码.HK | `00700.HK` |
| `us` | 交易代码 | `AAPL`、`BRK.B` |
| `futures` | 品种 + 年月.交易所 | `SC2612.INE`、`TA2611.CZCE`、`CL2612.NYMEX` |
| `crypto` | 市场:交易对 | `spot:BTCUSDT`、`usdm:BTCUSDT`、`usdm:BTCUSDT_240628` |
| `custom` | 自定 | 用 `custom` 时必须带 `instruments` 表 |

- 代码改名、退市后代码被复用这类情况，不得改写历史行的 `symbol`。旧代码和新代码是两个 symbol，二者的关系写在 `instruments` 表的 `successor` 列。

`instruments` 表（凡有行情表的包都应当带上）：

| 列 | 类型 | 必须 | 说明 |
|---|---|---|---|
| `symbol` | string | 是 | 主键 |
| `asset_type` | string | 是 | `stock`、`index`、`etf`、`fund`、`bond`、`convertible`、`future`、`option`、`spot`、`perpetual` |
| `currency` | string | 是 | 报价货币，ISO 4217 代码；加密货币写计价资产，如 `USDT` |
| `name` | string | 否 | |
| `list_date`、`delist_date` | date32 | 否 | 生命期，左闭右闭；行情行不得落在生命期之外 |
| `multiplier` | float64 | 否 | 合约乘数；股票不写，按 1 处理 |
| `tick_size`、`lot_size` | float64 | 否 | 最小价位、最小下单数量（按第 5 节的数量单位计） |
| `successor` | string | 否 | 改名或换代码后的新 symbol |

## 5. 字段与单位

### 5.1 行情字段词典

| 字段 | 类型 | 单位 | 默认 `knownAt` | 说明 |
|---|---|---|---|---|
| `open` `high` `low` `close` | float64 | 报价货币 / 数量单位 | `start`；其余三个为 `end` | **不复权**的原始成交价 |
| `previous_close` | float64 | 同上 | `start` | 交易所公布的前收（已按除权调整） |
| `vwap` | float64 | 同上 | `end` | |
| `volume` | float64 | 数量单位 | `end` | |
| `amount` | float64 | 报价货币 | `end` | 成交额 |
| `trade_count` | int64 | 笔 | `end` | |
| `open_interest` | float64 | 数量单位 | `end` | 单边口径 |
| `settle` | float64 | 报价货币 / 数量单位 | `end` | 结算价 |
| `adj_factor` | float64 | 无量纲 | `start` | 累计复权因子：后复权价 = 价格 × `adj_factor`，只有比值有意义，基准任选 |
| `limit_up` `limit_down` | float64 | 报价货币 / 数量单位 | `start` | 涨跌停价 |
| `suspended` | bool | | `start` | 当天停牌 |
| `taker_buy_volume` | float64 | 数量单位 | `end` | 主动买入量 |

### 5.2 单位

- **数量单位**按资产类型确定：股票、基金是 `share`，期货和期权是 `contract`，现货加密货币是 `coin`（基础资产），可转债是 `bond`（张）。包里的数量**必须**换算成这些基本单位。例如 A 股的「手」要乘以 100 换成股，「万张」要换成张。
- **金额**用报价货币的基本单位：元，不用千元或万元。
- **比例**用小数：5% 写 `0.05`。不用百分数，也不用基点。
- 字段的 `unit` 属性只能取以下几类：ISO 4217 货币代码（及 `USDT`、`USDC`）、上面列出的数量单位、`<货币>/<数量单位>`（价格）、`ratio`、`1`（无量纲）、`count`。
- 词典里的字段可以省略 `unit`，由 `instruments` 的币种和资产类型推出。

### 5.3 类型

- 价格、数量、金额用 `float64`，计数用 `int64`，标志用 `bool`，类别用 `string`。
- 不得使用 `float32`、`decimal` 或带字典编码的类别列。

### 5.4 扩展字段

- 词典以外的字段可以自由添加，但名字必须是小写 snake_case，必须声明 `unit`、`knownAt` 和 `description`。
- 扩展字段不得与词典里的名字同名而含义不同。

### 5.5 公司行为表 `corporate_actions`

| 列 | 类型 | 说明 |
|---|---|---|
| `symbol` | string | |
| `ex_date` | date32 | 除权除息日 |
| `kind` | string | `dividend`、`split`、`rights`、`spinoff`、`merger`、`other` |
| `split` | float64 | 每 1 股变成多少股，例如 10 送 5 写 1.5；没有就写 null |
| `cash_dividend` | float64 | 每股现金（除权前股本口径，报价货币） |
| `available_at` | timestamp UTC | 公告时刻 |
| `source` | string | 来源说明（自由文本） |

带了 `adj_factor` 的包，校验时会把它与公司行为表做比对，差异只报告、不判为失败。

## 6. 缺失、质量与已知缺口

- 缺失值**必须**写成 null。不得用 NaN、0、-1 或空字符串表示缺失。浮点列里出现 NaN 一律判为失败，适配器要先把它转成 null。
- **没有这一行**和**这一行的某个值为 null**是两回事：
  - 没有这一行，表示这个区间没有成交、停牌或尚未上市；
  - 有这一行但某个值为 null，表示有记录，只是这个字段缺失。
- 行情表不得补齐行，也不得把前值往后填。
- 主键是 `(symbol, date)` 或 `(symbol, ts)`，在包内**必须**唯一，并且在每个文件里按主键排序。
- 价格之间必须自洽：四个价格都不为 null 时，`low ≤ min(open, close)` 且 `max(open, close) ≤ high`。价格必须大于 0；允许负价的字段（例如 2020 年 4 月的原油）要声明 `"allowNegative": true`。数量和金额不得为负。
- 可选的 `quality` 列（`uint32` 位标志）：

  | 位 | 名字 | 含义 |
  |---|---|---|
  | 1 | `SUSPECT` | 来源可疑但保留，例如价格跳变无法解释 |
  | 2 | `CORRECTED` | 用另一个来源修正过 |
  | 4 | `SYNTHETIC` | 来源里没有这一行，是为了对齐生命期补的占位行，价格全部为 null |
  | 8 | `STALE` | 来源沿用了上一个区间的价格 |

  - 低 16 位保留给上表的公共含义，高 16 位留给各包自定义，自定义的含义在清单的 `qualityFlags` 里说明。
- 已知缺口：来源本身缺失的时段（比如交易所某天的归档缺失）**必须**写进清单的 `gaps`。这样使用者不会把缺口当成「没有成交」或「价格没变」。

  ```json
  "gaps": [{"table": "bars", "symbol": "spot:BTCUSDT", "start": "2021-04-25T04:00:00Z", "end": "2021-04-25T08:00:00Z", "reason": "归档缺失"}]
  ```

  不写 `symbol` 表示整张表都缺。`start` 和 `end` 是左闭右开的 UTC 时刻。

## 7. 版本与完整性

- 清单里每个文件都要记录 `path`、`bytes`、`hash`（形如 `sha256:<十六进制>`）和 `rows`。
- 包的 `version` = `sha256(规范化后的清单)`。计算时去掉 `version` 和 `created` 两个字段，键名排序后写成紧凑 JSON 再求哈希。所以任何一个文件改动、任何一处声明改动，版本号都会跟着变。
- 这些字段不用手写：`python -m ashare.data.spec seal <包目录>` 会计算并写入。使用者只需要写清单里描述性的部分。
- 读取时默认核对每个**被读到的**文件的哈希。核对失败就报错，不会退回旧数据。

## 8. 校验

`python -m ashare.data.spec check <包目录>` 按以下顺序检查。返回码：0 表示全部通过，1 表示有失败项，2 表示包结构坏到无法继续检查。

1. **结构**：清单符合 Data Package v2 加本规范的扩展；目录和文件名合法；没有清单里未登记的文件。
2. **完整性**：每个文件的字节数、哈希、行数与清单一致；`version` 正确。
3. **结构定义**：列名、列类型与清单一致；时间列的类型和时区符合 3.1；`unit` 和 `knownAt` 合法；词典字段的含义没有被改写。
4. **内容**：
   - 主键唯一且有序，没有 NaN，价格自洽，非负约束成立；
   - 日频行情的日期都在日历内，行情行都在生命期内；
   - `available_at` 不早于区间或事件开始；
   - 跨分区没有重复行，`gaps` 里的时段内确实没有行。
5. **一致性**（只报告，不判失败）：复权因子与公司行为是否对得上；相邻区间价格跳变过大的行。

校验报告是一个 JSON：逐项给出通过或失败，每种失败附最多 20 个例子（文件、行号、值）。

## 9. 接入方式

### 9.1 直接写文件

1. 按第 2–6 节写好 parquet 文件和清单的描述部分；
2. 运行 `python -m ashare.data.spec seal <包目录>`；
3. 运行 `python -m ashare.data.spec check <包目录>`。

`docs/examples/datapackage/` 里有一个最小的完整例子，数据是合成的。

### 9.2 写一个适配器

数据在数据库里，或者需要定期更新时，实现下面这个协议：

```python
class Adapter(Protocol):
    def describe(self) -> dict: ...                      # 清单的描述部分（不含文件和版本）
    def partitions(self) -> Iterable[tuple[str, str]]: ... # (表名, 分区名)
    def read(self, table: str, partition: str) -> pl.DataFrame: ...
```

`ashare.data.spec.build(adapter, 包目录)` 会逐个分区写文件、封存，然后校验，校验不过就不发布。已有分区内容没变时会跳过，不重写。发布是原子的：先在临时目录里写好，校验通过后才替换清单。

### 9.3 登记与读取

- 包可以放在任意目录。推荐放在 `data/ext/<包名>/`，这个位置由 `ashare.paths.dataset_dir("ext")` 解析，因此同样可以用 `LOCATION.json` 指到外置盘。
- `ashare.data.spec.open(包名或路径)` 返回的对象提供：
  - `read(table, start, end, symbols, fields)`：返回 Polars 长表；
  - `panel(fields, start, end, symbols, table="bars")`：返回 T×N 面板，接口与数据层的 `Panel` 一致（`dates`、`symbols`、`version`、`[字段]`），可以直接交给因子引擎。日频面板的时间轴用日历表，日内面板用规则的时间网格。没有行的格子填 NaN，另外附带一个由「这一格有没有行」得到的 `has_quote`；
  - `info()`：返回来源、范围、版本、字段和单位。
- 交给 `panelbt` 时，可交易掩码按以下规则得出：有 `suspended`、`limit_up`、`limit_down` 就用它们，没有就只用 `has_quote`。

## 10. 现有数据族对照

各数据族的内部格式不变，标准层视图由导出器生成（`python -m ashare.data.spec export <数据族> <输出目录>`）。

| 数据族 | 内部时间列 | 内部写法与本规范的差别 | 导出器 |
|---|---|---|---|
| A 股（`ashare.data`） | `date`，YYYYMMDD 字符串 | 时间写法不同；`action_source` 用数字枚举 | 第一批 |
| 加密货币小时线（`ashare.data.crypto`） | `time`，整数毫秒 | 时间写法不同；质量位不同（映射到 6 节的公共位 + 自定义位） | 第一批 |
| 加密货币分钟线 | `open_time`，整数毫秒 | 时间写法不同；按（交易对、市场、种类）三元组寻址 | 第一批 |
| 季度合约信号 | `time`，整数毫秒 | 同上 | 之后 |
| 境内期货 | `date`，YYYYMMDD | 时间写法不同 | 之后 |
| 港股、美股 | `date`，YYYYMMDD | 不是时点数据（`LOOKAHEAD_FIELDS`），导出时这些字段标成 `end+unknown`，校验只给警告 | 之后 |
| 基金、可转债 | `date`，YYYYMMDD | 时间写法不同；`action_source` 用数字枚举 | 之后 |
| 境外期货适配器 | 分钟线 `ts` 为 UTC 微秒 | 已经接近本规范 | 之后 |
| 实时录制（`data/live`） | 整数毫秒 | 属于来源层，不导出 | 不做 |

几个数据族之间的 `action_source` 枚举互不兼容（A 股 0–5、港美股 0–7、基金 0–6，同一个数字含义不同），所以标准层不用数字枚举，改为 `corporate_actions.kind` 加上自由文本 `source`。

## 11. 规范的变更

- 清单写 `"specVersion": "1"`。兼容的扩展（新增词典字段、新登记的代码写法）不改版本号；任何让旧包失效的改动都要升版本号，并且校验工具同时支持新旧两个版本至少一个发布周期。
- 改动先改本文，再改校验工具，两者在同一个提交里。
