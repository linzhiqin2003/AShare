# 币安数据层（C2 / C5）

入口为 `ashare.data.crypto`，实现 DESIGN 4.17 的数据合同和 6.10。它提供数据和数组，不导入 `panelbt`，不构造市场、不生成交易。USDT 现货与 U 本位 COIN 永续、季度合约是不同列，例如 `spot:BTCUSDT`、`usdm:BTCUSDT`、`usdm:BTCUSDT_240628`。不进入 A 股的 `DATASETS`。

## 数据集与覆盖矩阵

所有时间为 UTC。下表的历史起点是已核实的例子，不是所有交易对的保证；每次构建的 `meta.json.coverage` 单独列每列、每数据集的首末观测、记录数及来源。

| 数据集 | 原始来源 | 下载与覆盖 | 缺失处理 |
|---|---|---|---|
| 现货 `klines` 1h | vision monthly/daily；`/api/v3/klines` | 含历史下架交易对；上市覆盖起点取归档目录最早文件首条观测 | 生命周期内缺口默认报错 |
| U 本位 `klines` 1h | vision monthly/daily；`/fapi/v1/klines` | 永续与各期季度分别存放；BTC 永续原始成交可早于标记价覆盖 | 不因零成交回溯下架 |
| `markPriceKlines` 1h | vision monthly/daily；同名 fapi 接口 | BTC 已知归档从 2020-01 起；其他合约逐个实测 | 必需输入，无替造价格 |
| `indexPriceKlines` 1h | vision monthly/daily；同名 fapi 接口 | 按 pair 保存；季度默认使用同基础 pair，可在档案指定 `index_symbol` | 必需输入；与交割价的差异只作诊断 |
| `fundingRate` | vision **只有 monthly** | `calc_time, funding_interval_hours, last_funding_rate`；BTC/ETH 已知 2020-01 起 | 漏结算报错；不以相邻距离猜间隔 |
| 资金费关联标记价 | `/fapi/v1/fundingRate`，分页 | 从 2023-10-31 08:00 UTC 起优先用非空 `markPrice`；之前可为空 | 早期取结算整点标记 K 线 open 并标近似；后期缺官方价报错；最后一行缺价则整幅面板截一行 |
| 资金费当前间隔 | `/fapi/v1/fundingInfo` 公开 GET | 每次同步保存原始响应字节、抓取时间、内容 SHA-256 | 只辅助显式推断；updateTime 不证明历史 |
| 季度交割价 | `/futures/data/delivery-price?pair=…` | 本次 BTC 返回 18 条，2022-03 至 2026-06；接口实际范围可能变化 | 只用官方值；缺值报错，不能承诺全季度历史都可运行 |
| 交易对、过滤器 | spot/usdm `exchangeInfo` 每次完整快照 | 只代表采集时刻；历史全集另取 S3 目录；不以当前交易状态排除历史下架品种 | 历史无快照时需要逐项有来源的显式档案和规则假设 |
| 公告、事件 | 公开 CMS 列表及详情；人工补录 | 下架、币种动态、合约动态目录按页抓取；保存原文和待审队列 | 明确单句自动结算、明确交易对与唯一时刻的现货下架可提取；复杂迁移、延期、撤销等需核对后补录 |
| 币单位、转换表 | `mappings.json`、交易对档案、迁移事件 | 内置 1000PEPE 的逐条出处，以及 MATIC→POL 的比例和各市场时间 | 不对任意数字前缀猜倍率；钱包实际转换时刻未知时注明模型生效时刻 |

月归档优先；仅对月文件完全没有记录的日子尝试日归档（含月文件内部遗漏的整日），逐份核对 CHECKSUM。月内已有部分小时的日子不混入重叠日文件；原有缺小时继续显式处理。K 线、标记价、指数价不再用 REST 补最新几小时。资金费 REST 同时采集费率和关联价格，但没有历史间隔字段：默认严格构建仍等待月归档，显式推断版本见下节。原始采集覆盖与可运行模型覆盖因此不同。

事实依据：[BINANCE.md](markets/BINANCE.md)、[官方归档说明](https://github.com/binance/binance-public-data)、[官方市场数据接口](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data)。

## 命令和目录

在自己的 worktree 执行 `make dev`。根目录表示 crypto 数据库本身，不自动再加 `crypto/`。开发验证只写 worktree 的 `tmp/`，正式数据可由用户指定 `data/crypto/`。

先创建 `<root>/config.json`。必须显式选择交易对与区间，避免误触全量历史下载：

```json
{
  "start": "2024-05-01",
  "end": "2024-07-01",
  "instruments": ["spot:BTCUSDT", "spot:ETHUSDT", "usdm:BTCUSDT", "usdm:ETHUSDT"],
  "max_bytes": 30000000
}
```

```bash
.venv/bin/python -m ashare.data.crypto --root tmp/crypto sync
.venv/bin/python -m ashare.data.crypto --root tmp/crypto build
.venv/bin/python -m ashare.data.crypto --root tmp/crypto verify
```

`end` 不含；必须是已完成的整小时。`sync --refresh` 重新核对已完成对象，历史修补会产生新内容版本。默认按对象续跑；损坏对象重新下载。`--skip-announcements` 只用于已经人工导入事件或离线验收，返回摘要中的公告结果为 null，不能理解为“没有下架”。`verify --version <sha256>` 指定旧版，失败时退出码为 1；完整逐格诊断写入 `verification/<version>.json`。

请求均为公开 GET，不读密钥，不签名。HTTPX `trust_env=True` 读取 `HTTPS_PROXY`/`HTTP_PROXY`/`ALL_PROXY`，不打印环境值。连接超时 10 秒、请求超时 30 秒。普通接口最多四次尝试，429/418/5xx 遵从 `Retry-After`，超过 60 秒的等待要求退出留待续跑。C3 修复后 CMS 单独采用长指数退避：默认 8 次尝试、30 秒起、指数部分上限 240 秒；配置键 `cms_attempts/cms_backoff/cms_backoff_cap`。有 `Retry-After` 时至少等服务端要求，过长则退出；每次 sleep 分成至多 60 秒以便中断。资金费共享接口间隔 0.65 秒，其余 REST 0.25 秒，CMS 1 秒，归档 0.1 秒；串行运行，默认单次 HTTP 响应总量上限 40 MB。可调 `max_bytes`，不用全局依赖或系统服务。

公告抓取把查询范围、币种和目录组成进度键，保存在 `announcements/progress-*.json`。整页正文与事件处理完成才推进页码；中断于正文处理时复用已存列表页，正文按 SHA 与首次观察时间进入持久化导入队列；恢复时先消费已落盘的所有旧版本，再请求新正文。事件全部入账才标该观察版本完成，部分写入按版本幂等恢复。正文已变化时，修订以首次观察新正文的时间为 known_at，不因旧事件尚未入账而倒填首发。目录全部完成后，下次同步重新抓首页以发现新公告；未完成的同步从断点继续，增大页数上限也可以接着抓。

```text
<root>/
  config.json                 显式选定范围、档案、事件、例外
  manifest.sqlite3            独立清单、状态、修订引用
  objects/<sha256>             原始 ZIP、REST JSON、公告、目录响应
  raw_versions/<version>.json  冻结清单
  profiles.json               冻结计量单位及累计规则
  events/<version_id>.json     只追加的事件版本
  sync_plan.json              本次构建使用的冻结输入
  panels/<version>/           inputs.json、meta.json、YYYY-MM.parquet
  latest.json                 当前严格面板版本
  latest_inferred.json        当前显式推断面板版本
  funding_reconciliations/    推断与新归档的经济差异报告（只追加）
  verification/<version>.json 校验报告
```

每个 ZIP 核对同名 `.CHECKSUM` 的 SHA-256 **和文件名**；S3 ListObjects v1 使用 `NextMarker`，缺它时取最后 Key/Prefix，拒绝不前进的翻页。K 线、标记价、指数价虽然 ZIP 文件名相同，完整路径键不同，不会覆盖。REST 被后出的归档替代时，旧内容对象和旧面板版本仍在。同步计划固定 SHA，不随清单最新指针变化。

## 读取与数组

```python
from ashare.data import crypto

ds = crypto.open("tmp/crypto")               # 也可传 version=
hours = ds.read(start="2024-05-01", end="2024-06-01")
arrays = ds.arrays(
    ["open", "settlement", "funding", "mark_open", "qty_step", "tradable"],
    instruments=["usdm:BTCUSDT", "usdm:ETHUSDT"],
)
daily = ds.daily(trade_hour=1)
```

`read` 为 Polars 长表，按 `time, instrument` 排序；`arrays` 保留调用者的列顺序，返回 `time`、`instruments` 和各 T×N numpy 数组，检查矩形完整性。`profiles` 为该数据版本的档案；`inputs()` 含冻结的规范化原始行、事件、规则及原始库版本，供独立核对。读取检查分块和输入摘要，损坏立即报错。

`daily(trade_hour=h)` 输出完整的 `[d日 h, 次日 h)` 24 小时区间。`day` 为左端，`available_at` 为右端，只供信号与报表，不供交易引擎推进，也不在这里移位。日线 `funding_report_only` 仅是汇总，不能替代逐小时资金费。默认 `invalid_prices="raise"` 拒绝含倒置时间戳的日线；显式 `invalid_prices="carry"` 将生命周期内缺失/异常价格逐小时按之前有效收盘持平、成交量置零，再聚合，与小时回测一致。原始异常值仍由 `read()` 提供质检。`rank_coins` 是供 C3/测试核对的辅助函数，按真实时间窗口汇总已完成小时的永续 USDT 成交额、按币合并，按挂牌年龄过滤，绝不读取未来行；它不生成回测目标。

## C5：显式资金费推断延伸

```python
strict = crypto.build("tmp/crypto")
inferred = crypto.build("tmp/crypto", allow_inferred_funding=True)
ds = crypto.open("tmp/crypto", allow_inferred_funding=True)
report = crypto.verify("tmp/crypto", allow_inferred_funding=True)
# crypto.open("tmp/crypto") 仍读取严格版；固定 version= 可复现任一旧版。
```

```bash
.venv/bin/python -m ashare.data.crypto --root tmp/crypto build --allow-inferred-funding
.venv/bin/python -m ashare.data.crypto --root tmp/crypto verify --allow-inferred-funding
.venv/bin/python bench/crypto_prepare.py --data tmp/crypto --accept-spot-gaps --allow-inferred-funding --output bench/results/crypto_inferred_validation.json
.venv/bin/python bench/crypto_examples.py --data tmp/crypto --allow-inferred-funding --output bench/results/crypto_inferred_examples.json
```

`build` 的选项默认 False。低层 `build_frame` 继续检查调用方提供的完整区间，保留 `funding_schedule` 的首笔、中间、末笔覆盖检查；它不自动开启推断。正式构建入口把严格终点控制收回数据层，准备脚本不再缩短并覆盖同步计划的请求终点，因此可以先后构建两种版本。

设最后归档结算为 `a`、间隔为 `h` 小时，所有终点均指 `[start,end)` 的右端：

- 严格终点仍按 `a + h×1h − 1h` 约束，再与 requested_end 取小；仅纳入覆盖尚不足以支撑该列生命周期的永续。最后一行左端还要再减 1h。末行结算缺价格继续由原构建器截行。
- 推断先求 `D = min(requested_end, 各仍需覆盖的数据集归档中最后完整小时的右端)`；已经覆盖到已知退出边界的列不参与截短。从 `a` 起检查 REST：时间取整距离不超过 1 秒，逐笔间距等于 h、在 h 小时网格上，费率有限且结算标记价正且有限。第一笔异常之后的记录不再采用；只检查到 D 与该列退出边界的较小值，其后记录不影响窗口。
- fundingInfo 必须已经冻结：交易对在列表中时 `fundingIntervalHours == h`；不在列表中仅当 h=8 才可推断。缺快照或不匹配时该列保持严格上限。`updateTime` 原样存档而不作证据。
- 无异常且 D 之前应有的结算全部存在时取 D；若异常或末笔缺失，退到前一笔合格结算的右边界（可为 a）。各列取最小值。缺标记价不能以小时开盘近似延伸，即使时间早于官方价格历史起点也一样。

这只是历史间隔的**延续假设**。当前快照不能排除“临时改频又恢复、REST 恰好漏掉额外结算”，因此不会把推断版升级为严格证据。[官方 fundingInfo 文档](https://developers.binance.com/en/docs/catalog/core-trading-derivatives-trading-usd-s-m-futures/api/rest-api/market-data#get-funding-rate-info) 描述当前调整信息，没有提供历史间隔证明。

`meta.json.allow_inferred_funding` 与冻结输入的 `recent.allow_inferred_funding` 标版本类别；`recent.requested_end` 保留请求终点，`recent.funding_info` 包括 `key/fetched_at/sha256/records`，`recent.instruments` 记录归档锚点、间隔、接受条数、上限和截断原因。原始规范化资金费中新增 `source=inferred_continuation`；面板沿用字段，仅在相应结算行加质量位 4096。`crypto_market` 的 `inferred_funding_intervals` 按实际读取窗口列交易对与受影响小时 `[start,end)`；结算发生在这些小时的右边界。

覆盖上限仅约束生命周期内需要数据的时段。已知结算退出或下架后，旧列按原规则持平，不因停止发布行情/资金费而截短仍活跃的列。退出边界使用前一小时已经可知的事件版本（请求开始前已退出的列使用起点可知版本）；事前取消或延期不能提前免除覆盖要求。`funding_schedule` 的末笔覆盖检查仍保留。

推断模式中，覆盖豁免不等于已经证明归档之后没有结算：即使请求终点或退出边界早于下一个预期结算，也逐条检查归档末笔到有效终点内的 REST。例：最后归档 16h、间隔 8h、请求或退出在 21h，但 REST 有 20h 结算，必须退到 16h，不能让该笔资金费成为零。fundingInfo 缺失或不匹配时仍不允许延伸到严格上限之外，但上限以内已观测到的异常结算同样会缩短推断版；有效终点之后的记录不影响该列。默认严格构建保持原行为。

同步通常从旧资金费末记录之后增量请求；当新的月度资金费归档发布（或归档内容哈希改变）时，即使不加 `--refresh`，也会重新请求对应声明月份（最晚到抓取时刻），用新响应替换这段缓存并冻结新快照。新响应缺笔时不会用旧缓存补回。原始内容寻址对象、原始清单版本和旧面板保留；无新归档时不重复重取历史月份。日归档尚未发布的尾部不补。

新版本的 `recent.source_contract=1` 启用来源核对：`recent.funding_price_refs` 按交易对记录 REST 快照的 `key/sha256`，`recent.funding_archive_coverage` 记录月归档的 `key/sha256/start/end`。覆盖区间是文件声明的完整 `[月初,下月初)`，即使 CSV 缺末笔也不缩短。这些字段同时冻结在输入和元数据中，不新增面板列。旧版本缺少这些来源字段时仍可复现；它们不具备新增的 REST 输入独立对账保证，需要重建才能获得。

### 归档发布后的纠正与核对

再次 sync、build 后，严格版自然使用新归档和重新取得的对应区间 REST 结算价。构建会逐一对比旧推断版本与归档声明月份的交集，比较结算集合（能发现新增/消失的结算）、时刻、间隔、费率和关联结算标记价；来源标签变化不算差异，相同经济记录不写报告。月尾缺笔以 `archive: null` 记录，即使新严格版因该缺笔已经提前结束；下月零点不计入本月审计范围。没有原始对象引用的合成/旧输入只能按已观测记录范围核对。

差异写入 `<root>/funding_reconciliations/<hash>.json`，含旧推断版、新归档版以及差异值。`verify` 的 `funding_reconciliations` 同时列出涉及所查新版本或旧版本的报告；差异不改变通过/失败判定，避免已经纠正的数据永久无法构建。旧面板与旧原始对象不改写，旧研究固定其 SHA；需要使用纠正结果时明确选择新版本并重跑研究。完整差异只存数据目录，不进入公开汇总。

若新归档费率与缓存 REST 费率仍不一致，原始 REST 不改写：规范化归档记录标 `archive_rate_override=true`，资金费按归档费率乘 REST 关联标记价，verify 输出 `funding_rate_archive_override` 诊断。这不豁免归档连续性或关联价缺失；缺价仍不能造价。结算标记价并不存在于资金费 ZIP，实际是比较两个冻结版本的官方关联价，若交易所尚未修订 REST，就无法独立得知新结算价。

verify 从冻结 REST、原始快照和归档独立重算推断条件、生命周期覆盖需求与终点，不导入构建的 recent 判据。它核对 fundingInfo 内容/时间/哈希、REST 引用与原始清单，并逐条比较 REST 原时刻（毫秒）、费率、结算标记价及记录集合。仅验证原始对象哈希不足以证明构建输入正确；新核对也会拦截“输入和面板一起改错但原文未改”的情况。

## 每个面板字段

一行是 `[time, time + 1h)`，时间均为 UTC 毫秒。期货价格和 `volume` 以**交易所报价单位**表示：1000PEPE 的价格是每 1000 PEPE 的 USDT，数量单位也是 1000 PEPE，不先乘或除 1000 再传内核。

| 字段 | 含义 / 单位 |
|---|---|
| `time` | 小时左端 UTC 毫秒 |
| `instrument`, `market`, `symbol` | 唯一列标识、spot/usdm、交易所原代码 |
| `coin`, `coin_units` | 原生币与每个报价单位含多少币；1000PEPE 为 PEPE、1000 |
| `contract_type` | SPOT、PERPETUAL、CURRENT_QUARTER、NEXT_QUARTER |
| `multiplier` | 期货每张含多少报价单位；固定，不随规则变化重选 |
| `open/high/low/close` | 实际成交价格，USDT/报价单位 |
| `volume` | 报价单位成交量；普通 BTC 即 BTC，1000PEPE 即千枚 PEPE |
| `quote_volume`, `taker_buy_quote_volume` | USDT 成交额、主动买入成交额，**不乘 coin_units** |
| `volume_engine` | 期货 `volume / multiplier`，张；现货等于 volume |
| `mark_open/high/low/close` | 同小时标记 OHLC，价格单位与成交价相同 |
| `index_close` | 同小时价格指数收盘；季度按档案 index_symbol 取基础 pair |
| `funding_rate` | 右端结算费率，非结算行 NaN；这不是左端已知信号 |
| `last_funding_rate` | 左端之前或恰在左端已结算的最近费率，首次结算前 NaN；供信号使用 |
| `funding_price` | 这一笔资金费对应的官方标记价或注明的早期近似价 |
| `funding_interval_hours` | 归档的结算间隔，或显式推断沿用的间隔（quality 位 4096）；非结算行 NaN |
| `funding` | 每张 USDT = rate × funding_price × multiplier；正数为多头支付；无事件 0 |
| `funding_price_source` | none / rest / mark_open_approximation |
| `settlement` | 活跃期货为 mark_close；终止行直接使用交割/下架价，不再重复结转 mark_close |
| `settlement_source` | none / mark_close / delivery_rest / announcement / mark_open_approximation / pre_listing_placeholder / terminal_carry / missing_carry / unrealized_spot_carry |
| `delivery_record_time` | 官方交割价记录的原始 deliveryTime；可为当天 00:00 日期标签，不能当实际交割时刻 |
| `tradable` | 处于生命周期、非暂停、行情完整且该小时有量；不是保证订单一定成交 |
| `expires` | 期货在本行右端终止；现货下架不设置它，不兑付现金 |
| `lifecycle` | pre_listing / active / ended；暂停仍是 active，另有质量标记 |
| `qty_step`, `min_qty` | 期货为张，现货为报价单位；相互独立 |
| `min_notional` | USDT 最小名义，不乘 coin_units；现货首批执行器是否检查由 C3 的模型范围决定 |
| `rule_source`, `rule_source_url` | announcement / snapshot / assumption 及逐条来源 |
| `rule_known_at`, `rule_effective_at` | 规则可知与实际生效时间；非整点从下一整点采用 |
| `rule_approximation` | 非整点或旧仓豁免条款被简化，整个适用段标记，不能仅标一个小时 |
| `data_source`, `mark_source`, `index_source` | archive / rest / none；逐字段数据集来源 |
| `event_version` | 终止行实际使用的事件版本 ID；可回原文 |
| `quality` | 位标志，可并存，见下表 |

| 位 | 含义 |
|---:|---|
| 1 | 活跃期缺行情/标记/指数，以之前估值持平 |
| 2 | 当前小时 USDT 成交额为 0 |
| 4 | 已观测至少连续 3 小时零成交且价格不变；只标从第 3 小时起的当前行，不回溯 |
| 8 | 公告维护/暂停窗口 |
| 16 | 早期资金费关联价格近似 |
| 32 | 合约下架结算价取右端标记 open 近似 |
| 64 | 下单规则采用显式假设或最早快照向前沿用 |
| 128 | 生命周期外 |
| 256 | 成交 K 线来自 REST 补齐 |
| 512 | 原始 K 线早于正常右边界超过 1 秒结束，保留 OHLC，标 `partial_halt` |
| 1024 | 紧随成交 K 线半截线的连续缺口，且缺口内无正常成交/标记线反证；半截线同小时的标记线不算反证，恢复成交或缺口内出现反证即结束推断 |
| 2048 | 原始收盘时刻早于开盘，保留异常行并禁止交易，校验诊断 `invalid_close_time` |
| 4096 | 本行右边界资金费的间隔由延续推断，`funding_interval_inferred`；与关联价格近似位 16 分开 |

生命周期外约定：

- 未上市期货：成交 OHLC 和标记 OHLC 为 NaN，量为 0，`tradable=False`，`funding=0`；`settlement` 填上市后第一根可用标记收盘，必须正且有限。此占位值只满足 C1 的输入检查，不得作为历史可交易价格或信号。
- 到期/合约下架后：成交 OHLC 和标记 OHLC 为 NaN，量为 0，资金费为 0，`settlement` 固定为已了结价格，`expires=False`。了结发生在此前右端相等的那一行。
- 现货上市前价格 NaN；现货正式下架后 OHLC 保留最后价格、量为 0，标 `unrealized_spot_carry`，不制造现金、成交或交割。转换资产由 C3 读取迁移事件接续。
- 生命周期内缺数据默认错误。公告暂停、成交 K 线半截线之后紧随的缺口可以持平；指数或标记等辅助数据集的半截线不能证明全市场停机。缺口中出现正常成交线或标记线即构成反证，后续缺口不能重新沿用此前证据；反证时间必须晚于半截成交线所在小时，同小时的正常标记不否定随后缺口。恢复成交时结束该段推断。构建器维护成交证据状态，独立校验从原始成交时间轴及正常标记时间轴重新查证，不共享判据函数，不需要人工例外表。其他 `build --allow-missing` 保留的缺失格不自动获得解释，`verify` 仍要求公告窗口或 `config.exceptions` 中逐交易对、`start/end` 毫秒区间、`reason/source_url` 的明确例外，否则失败。倒置时间戳不构成停机证据；它保留在输入及面板质量标记里，verify 单列 review 诊断，回测默认拒绝。
- 输入未提供、无事件的 0、数值缺失的 NaN/质量位分别处理。资金费漏记录永远不能用 `--allow-missing` 放行。

## 事件和档案

`EventLedger.append` 接受 `event_id, instrument, kind, known_at, effective_at, source_url, original_text`；可选 `settlement_price, replacement, ratio, supersedes, status, manual, version_unknown, published_at, known_at_basis`。kind 为 delist/settlement/delivery/suspend/resume/migration；status 为 active/cancelled。所有字段进入内容 hash，修订必须以 `supersedes` 指向更早版本。旧 JSON 不回写 `superseded_by`；反向关系从新记录推导。

`asof(events, time)` 逐 event_id 取当时最新且未撤销版本。延期不改旧 effective_at；OMG 形态在测试里先保持 12-14 版本的 12-30，到 12-20 才能看到 1-31。自动公告的时间依据按以下顺序确定（C2 返工 R1）：

| 条件 | `known_at` | `known_at_basis` |
|---|---|---|
| CMS `lastUpdateTime` 或标题/正文有明确更新标记 | 所有更新证据中最晚的时间 | `latest_update_evidence` |
| 没有更新证据，有 `publishDate` / `releaseDate` | 首发时间，优先 `publishDate`；详情缺失时保留列表 `releaseDate` | `publish_no_update_evidence` |
| 首发与更新证据均缺失 | 抓取时刻 | `fetched_no_time_evidence` |
| 已保存旧正文，后来观察到正文改变，但没有更晚的版本时间证据 | 本次观察时刻，新增版本并指向旧版 | `observed_revision` |

只有日期的更新取次日 00:00 UTC；所有可知时间不晚于已抓到该内容的时刻。`published_at` 保留首发毫秒值，缺失为 null。自动解析始终保留 `version_unknown=true`，并不声称找回历史原文。**无更新证据时采用首发时间，是“未标注的静默修改可能存在”的显式假设**，用于避免历史回填的公告全部变成回测期间不可知；一旦有更新证据，不再回退首发。迟到的终止事件不能静默倒写已处理的行。

支持 ISO 日期的 `Updated on …`、`Update (日期)`、`(Updated 日期)`、`Update: 日期` 等标记，保留明确的时分秒，并处理 CMS `child` 节点和 HTML 空格实体。实测观察到正文的 `was updated on YYYY-MM-DD` 写法与 `lastUpdateTime=0`；验证结论在 C2_NOTES.md，原文仅在被忽略的 tmp 中。公开测试使用自撰合成公告；其余变体只有合成兼容性测试，不作为已观察到的币安格式事实。

原始正文 hash 不变时，续跑不会重新提取事件。因此解析规则修正后，旧事件不会自动被回写：应从缓存原文在新的数据根目录重新提取、构建并校验，旧根目录和固定面板版本继续保留。R1 的离线重建步骤与结果见 [C2_NOTES.md](C2_NOTES.md)。

`config.events` 可人工补入上述结构，必须带原文及出处，`manual=true`。复杂迁移、多个市场不同时间和取消条款进入 `announcements/pending/`，确认后再补入事件账本；不能将待审文本直接用作交易退出。合约下架价未知时只允许结算整点标记 open 的显式近似；季度则必须有官方交割价。

`config.profiles` 以 instrument 为键提供档案覆盖项。当前快照中消失的历史合约，必须明确 `base_asset/coin/coin_units/quote_asset/margin_asset/contract_type/underlying_type/source_url/rules`，不凭名称或价格量级猜规则。每条规则有 `known_at/effective_at/qty_step/min_qty/min_notional/source/source_url`。数量过滤器优先 MARKET_LOT_SIZE，零步长时用 LOT_SIZE。维持率、最高杠杆、档位上限、费率的历史考据归 C3，本层不把 exchangeInfo 的旧 maintMarginPercent 当第一档维持率。

乘数在首次建立档案时取步长能表示的最小十进制幂，保存到 profiles.json。以后快照规则变化只追加，不改变乘数。新步长无法被旧单位整数表示时拒绝续建，要求另建数据系列；旧版本始终可读。`mapping_at` 给出已知且已生效的迁移映射和数量比例；不改原始报价，不执行钱包转换。

## C1 / C3 共用夹具

`test/fixtures/crypto/inputs.json` 是完整合成来源，`panel.parquet` / `panel.csv` 是可读的小面板，`arrays.npz` 可直接读取 numpy 数组。运行 `python test/fixtures/crypto/generate.py` 可离线重建。合成内容不代表真实交易规则。

- 时间 2024-06-28 00:00–12:00 UTC，12 行、2 列。NPZ 列顺序固定为 `usdm:PERPUSDT`、`usdm:FUTUSDT_240628`；Parquet 是按 time/instrument 排序的长表。
- 永续 02:00 上市，乘数 0.01，步长 1 张、最小量 2 张、最小名义 5 USDT。前两行 settlement=103.5，标记价 NaN；04/08/12 时费率分别 +0.0001/−0.0002/+0.0003，关联价 105/109/113，资金费落 03/07/11 行，分别 +0.000105/−0.000218/+0.000339 USDT/张。
- 季度此前已挂牌，乘数 0.1，08:00 交割。07:00 行 mark_close=208.5，**settlement=199，expires=True**；08:00 起 settlement 保持 199，mark OHLC 为 NaN，无资金费，不可交易。从基准 200 持有一张到交割的最终价格盈亏为 −0.1 USDT。
- NPZ 包含 time/instruments/multiplier、成交 OHLC、volume_engine、settlement、标记 OHLC、funding、qty_step/min_qty/min_notional、tradable/expires/quality；其余来源、生命周期、已知费率等字段在 Parquet/CSV。

`raw/` 的六份 CSV **全部为合成数据**，保留实测归档的格式；价格、成交量和费率由独立固定常数生成，没有从真实数值平移、缩放或抽样。`provenance.json` 只记录 `synthetic=true`、数据集、文件名模式及格式特征，不保存真实归档 SHA、校验文本或原始行。可用 `.venv/bin/python test/fixtures/crypto/generate_formats.py` 离线重建；生成器不读取 tmp，不联网，不修改共用面板夹具。

| 文件 | 列 / 表头 / 数据行 | 时间戳 | 数值文本格式 |
|---|---|---|---|
| `spot_ms.csv` | 12 / 无 / 20 | 13 位毫秒 | OHLC、量、金额及主动买入量/额均 8 位小数 |
| `spot_us.csv` | 12 / 无 / 20 | 16 位微秒 | 同上 |
| `usdm_header.csv` | 12 / 有 / 19 | 13 位毫秒 | OHLC 2 位，量/主动买入量 3 位，金额/主动买入额 5 位 |
| `usdm_no_header.csv` | 12 / 无 / 20 | 13 位毫秒 | 同上 |
| `mark.csv` | 12 / 有 / 19 | 13 位毫秒 | OHLC 8 位，保留实测格式中的 9 个整数价格格；量/金额/ignore 为整数 0，count 为 3600 |
| `funding.csv` | 3 / 有 / 19 | 13 位毫秒 | 费率 8 位，interval 为整数 8；合成 calc_time 含 0/1/2/3/5 毫秒偏移 |

全部文件使用逗号、无引号、无 BOM 的 ASCII（兼容 UTF-8），LF 行尾且最后一行有换行。K 线的列顺序与完整表头文字保持一致：`open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore`；资金费为 `calc_time,funding_interval_hours,last_funding_rate`。毫秒/微秒的开收时间位数、小时步长、收盘为右边界减一个原生时间单位均由测试检查。

`announcements/waves.json` 仅保留原测试路径，**内容是自撰合成公告**，使用 FAKE、DUMMY、MOCK、SAMPLE 虚构资产。保留标题形式、毫秒 `publishDate`、零值 `lastUpdateTime`、JSON 字符串中的 `node/child/text` 树，以及更新标记、现货下架和自动结算的必要短句式；没有真实公告段落、文章 ID、来源响应 SHA。公告和 CSV 由同一个生成器生成，格式约束见 `test_fixture_layouts.py`。

## 格式与限制

- 表头看内容，不能按年份硬编码；现货允许毫秒/微秒，通过量级、整点和文件范围验证；U 本位只接受已核实的毫秒。每份文件检查时间递增、重复、close_time 与小时边界。收盘与正常边界相差不超过 1 秒允许；更早结束标半截线，收盘早于开盘标异常，越过右边界超过 1 秒仍拒绝。规范化原始行保留 `close_time/time_quality`，不静默丢行。资金费整点抖动仍容忍不超过 1 秒。
- 资金费归档字段不含结算价；官方关联价格与下一根 mark open 的偏差、是否超出下一根高低区间均是诊断。默认提示阈值为 1%，不作相等断言。
- 实测交割价接口 deliveryTime 是 UTC 00:00 日期标签。只有显式交割事件为同日 08:00 时才作日期匹配，保留 delivery_record_time；不以 00:00 生成交割事件。交割价与同小时指数收盘比较，只提示，不复算 30 分钟均价。
- 零成交/冻结可能是暂停或归档问题，只标质量，不生成历史下架；追加未来零成交不会回改前三天的退出。公布时间和生效时间分开。归档被交易所修补是新的内容版本，不承诺不同数据修订计算结果相同。
- 当前 CMS 不保证覆盖全部历史版本，通用自动解析不可能可靠还原所有自然语言；本次不声称已完成全部下架币历史事件库。
- 缺早期标记价的区间不能回测；没有季度历史官方交割价的区间不能通过。新发布 REST 资金费没有可信历史 interval 时，默认严格版仍等待归档；显式推断版接受下述可追溯假设。
- 现货手续费扣币、最小名义执行、完整维持保证金档位和两钱包调度不属于 C2。非整点终止默认拒绝，不能假装小时 OHLC 能重放分钟内退出。

## 验证与真实端到端

测试命令：`nice -n 15 .venv/bin/pytest -q test/crypto -n 4`。初次交付 67 项通过；当时与现有数据层合跑 1,011 项通过，7 项旧 Databento 测试因未安装可选依赖跳过。R1 公告回归、R2 合成夹具替换与运行结果见 [C2_NOTES.md](C2_NOTES.md)。全部离线，覆盖格式、校验和、S3/REST 分页、节流重试、内容修订、事件版本、生命周期、资金费漏笔与间隔、官方价格诊断、交割归属、暂停恢复、迁移与 1000PEPE、字段筛选、日线、历史截断、数组与不可变版本、独立校验的直接产物变异。

真实端到端取 2024-05-01 至 2024-07-01，BTC/ETH 现货与永续、WAVESUSDT 永续、BTCUSDT_240628 季度。原始同步约 21.3 MB，公告列表 48 页；只抓这段日期与这些币相关的正文。构建 1,464×6=8,784 格、2 个月分块，54 份 ZIP 通过官方 SHA 与独立 CRC/CSV 对账，verify 无错误。WAVES 6-11 09:00 结算落 08:00 行；BTC 季度 6-28 08:00 交割落 07:00 行，交割值与官方来源核对一致。原始数据和公告原文均留在被忽略的 tmp；公开仓库只保存合成夹具与验证结论。

此端到端通过表示数据与所声明模型一致，不表示历史规则全部已考据：五个当前快照品种的早期规则向前沿用、已消失季度的规则显式借用 BTC 过滤器，均标 assumption；WAVES 下架价使用边界 mark open 并标近似。490 条资金费价格诊断的最大绝对相对偏离为 0.04535%，没有超过 1% 提示阈值；没有交割/指数或现货/永续的超阈值差异。真实数据截到 6 月 1 日后，历史面板（含类型）及前 N 排名与完整数据逐项一致。初次交付面板版本为 `c9e3f5e59b07341fffd430f8333f1dd170354496253abff5592885c53496ca4c`。完整摘要放在本 worktree `tmp/e2e/summary.json`，不提交真实面板或原始响应。含单独格式、接口探针，开发阶段公开数据下载合计约 42 MB。

R1 使用相同原始数据在 `tmp/e2e-review-r1/` 离线重提取、重建，移除原样本中人工补录的 WAVES 事件。自动结算及现货下架事件的 `known_at` 均为 **2024-06-04 00:00 UTC**，依据为 `latest_update_evidence`；结算生效 6-11 09:00、现货下架生效 6-17 03:00 UTC。新面板版本 `792d023e55b5a59b99e787948b94db12727586f767f7ac9b8eb17b713de1a933`，8,784 格、54 个归档、verify 0 错误。未重新下载，旧目录的事件、计划及面板保持不变；现货 WAVES 事件已验证可知时间，但本次六列行情样本不含 WAVES 现货。

## 分钟线（C7）

BTCUSDT、ETHUSDT 的现货成交线与 U 本位永续成交线、标记价、指数价，通过独立模块 `ashare.data.crypto.minute` 同步、整理、校验和读取；不改变小时面板与回测引擎。归档原样核对 CHECKSUM，整理为每市场/交易对/种类/月一个 Parquet，记录逐段缺口，不补值；原档、整理结果、质量记录和定义共同进入内容版本。

入口为 `python -m ashare.data.crypto.minute sync|build|verify|status`；读取用 `bars(symbol, market, kind, start, end, *, root=None)`，返回按需读取的 Polars 表。现阶段默认 `data/crypto_minute`，D1 合并后统一路径；所有入口均可显式指定 `root`。

完整字段、续传与版本语义、独立校验及 1m→1h 对账用法见 [CRYPTO_MINUTE.md](CRYPTO_MINUTE.md)，样本验收与已确认的官方归档差异见 [C7_NOTES.md](C7_NOTES.md)。

## C6：季度合约仅信号数据集

`ashare.data.crypto.signals.SignalDataset` 与可交易面板独立，只提供季度收盘与质量位，不要求官方交割价格，也不能直接构造可交易市场。它不修改 `latest.json` / `latest_inferred.json`。

```sh
.venv/bin/python -m ashare.data.crypto --root tmp/crypto signals-sync \
  --start 2020-01-01 --end 2026-11-01
.venv/bin/python -m ashare.data.crypto --root tmp/crypto signals-verify
```

每期列举全部月归档并核对 CHECKSUM，已核对对象按内容摘要复用。上市优先有来源的 onboardDate/公告，否则以全部归档最早一条作保守代理，不能使用查询起点。交割从代码日期加08:00 UTC，与 deliveryDate 核对；已到期合约最后一根成交量大于0的K线，开盘必须位于[交割−24小时，交割]，否则报错。未到期合约登记 pending_not_expired，仍拒绝交割之后的成交；到期后必须补做下界核验。无上市来源的合约不进候选，构建报告保留排除清单。

目录为 `signals/raw/`（独立清单和原始对象）、`signals/versions/<sha>/`（冻结输入、规范化元数据、小时表）、`signals/latest.json` 与 `signals/verification/`。内容版本覆盖原始引用、完整解析结果、上市/交割时刻和来源、质量例外、小时表摘要及定义版本；读时同时校验原始对象、元数据和小时表，修改元数据须生成新版本。`signals-verify` 自行解压解析 CSV，独立核对连续性、质量、上市和交割，不调用生产构建器，也不调用交割价格接口。

零成交与连续冻结小时保留质量位，信号层屏蔽；缺小时只能通过带原因的 `--exceptions` 文件登记，仍为缺值。质量位与现有面板保持对应关系。定义、时点和因子使用详见 [CRYPTO_FACTORS.md](CRYPTO_FACTORS.md)。

同步在构建前保存 `signals/sync_plan.json`，构建失败也保留冻结引用；`signals-build [--exceptions <json>]` 可离线重建，完全不访问网络。按 DESIGN 4.18 v5.1，收盘晚于交割的行自动登记为 `post_delivery` 并从信号表剔除，原始字节和完整解析行保留。逐期排除计数、范围和原因随元数据进入内容版本，独立校验重新计数；调用方不能手工覆盖此例外。交割整点开盘的有成交K线可以用于交割核对，但其收盘在交割之后，仍不能进入信号表。上市与 deliveryDate 冲突照常报错。

```sh
nice -n 15 timeout 180 .venv/bin/python -m ashare.data.crypto --root tmp/crypto signals-build
nice -n 15 timeout 180 .venv/bin/python -m ashare.data.crypto --root tmp/crypto signals-verify
```

## 合约指标 metrics（C8）

`ashare.data.crypto.metrics` 是独立冷数据集 `crypto_metrics`，读取位置通过 `ashare.paths` 解析。显式 `root` 表示完整数据集目录；它不和小时或分钟数据共用根。首批 BTCUSDT、ETHUSDT，来源为币安 U 本位 `daily/metrics` 公开归档，只有日档。`sync` 列举全部已发布文件，最多 4 个下载连接；每份归档核对 CHECKSUM 后内容寻址保存。中断后重跑会复用已核对对象，`--refresh` 才复查上游修订。

```sh
python -m ashare.data.crypto.metrics sync --root '<metrics 数据集>' --workers 4
python -m ashare.data.crypto.metrics build --root '<metrics 数据集>' --minute-root '<分钟数据集>'
python -m ashare.data.crypto.metrics verify --root '<metrics 数据集>' --minute-root '<分钟数据集>'
python -m ashare.data.crypto.metrics status --root '<metrics 数据集>'
```

构建按月读取永续 1m K 线，分别按 `create_time` 为窗口结束和开始复算主动买卖比。对齐由代码中每个合约单独登记的切换常量决定：BTCUSDT、ETHUSDT 均为 **2024-03-04 00:00 UTC**，之前标窗口结束，之后标窗口开始。新合约缺少登记以 `UNREGISTERED_SWITCH` 拒绝；build 不从观测推断切换时刻。

每个 UTC 日用双边均可比行计算误差中位数，至少 120 行，某侧严格 <0.01、另一侧严格 >0.1 才形成明确证据。日证据必须与登记一致，否则报 `REGISTERED_ALIGNMENT`；未知日直接采用登记值，并设置证据不足质量位。连续 K=2 行明确反向（登记侧 >0.1，另一侧 <0.01）报 `SUSTAINED_OPPOSITE`，跨午夜累计；灰区、空值和同向观测中断，缺口不清零。切换前后各 24 小时的闭区间内，单行明确反向也报 `SWITCH_WINDOW`。矛盾提示可能出现新切换，需要人工复核登记；质量位不能豁免。灰区不被强行当成反向，允许保留。

按登记对齐复算，误差严格 >0.05 的行保留并标记 `TAKER_MISMATCH`，等于 0.05 不标记。两个中位数、双边可比数、当日证据不足状态、反向行数及连续长度，以及偏差比例均写入每日统计。整天缺行也在日历统计中保留，但不填充数据行。

```python
from ashare.data.crypto import metrics
frame = metrics.read(
    "BTCUSDT", "2024-03-01", "2024-04-01",
    root="<metrics 数据集>", version="<冻结内容版本>",
)
```

查询按归一化 `time` 的 `[start,end)` 返回 Polars。`time` 为五分钟窗口开始，`available_at = time + 5min`；**全部字段都在窗口结束才可知**，包括未平仓量快照。`create_time` 保留归档时刻，`alignment=0` 表示原始时刻标结束，`1` 表示标开始。质量位可按位组合：`NULL_VALUE=1` 表示至少一个空值，`INHERITED_ALIGNMENT=2` 表示所在日证据不足、采用登记对齐（保留旧位名兼容字段），`TAKER_MISMATCH=4` 表示所选对齐的逐行误差 >0.05。

| 字段 | 单位 / 口径 |
|---|---|
| `open_interest` | 币 |
| `open_interest_value` | USDT |
| `top_account_ratio` | 大户账户多空比 |
| `top_position_ratio` | 大户持仓多空比 |
| `global_account_ratio` | 全市场账户多空比 |
| `taker_buy_sell_ratio` | 主动买量 / 主动卖量（币本位量） |

内容版本冻结原始对象及 CHECKSUM、分钟源版本、阈值和定义、月度 Parquet、重复/空值计数、原始时间缺口、归一化窗口缺口和切换证据。旧版本不覆盖；最新指针只在完整构建成功后发布。读取校验关联原始对象、版本和 Parquet 哈希。没有归档的月份返回空表，不填行情。

独立 verify 重新解码原始 CSV，以另一份逐分钟累加代码对账，独立按登记常量赋值，重算日中位数、跨日反向段、切换窗口、空值、缺口及质量位，并核对每天和每年的汇总。通过表示归一化字段、判定和质量记录一致，不表示所有比值都无偏差。`verification/<版本>.json` 中 `details` 是验证错误，`taker_mismatches` 列出所有偏差行；这些明细仅存数据根，不应加入仓库。公开验收 JSON 只含统计量。`mismatch_ratio` 以全部行作分母，`comparable_mismatch_ratio` 以所选对齐可复算行作分母；每天和每年均按原始 `create_time` 的 UTC 日期归属。

规则定义为 `crypto-metrics-v4-registered`。原始 CSV 格式未变，允许复用旧 v1/v2/v3 下载计划；旧版内容不会被重新解释。若尚无已发布版本，`verify` 仅做原档审计，返回 `built=false, ok=false`，不代替 build 发布，也不把下载完成称为版本验收。

共享机器上可把验收验证拆成两条限时命令，复用同一已构建版本，再严格合并覆盖范围：

```sh
python -m bench.crypto_metrics_validation --root '<metrics 数据集>' --minute-root '<分钟数据集>' --version '<版本>' --verify-symbol BTCUSDT
python -m bench.crypto_metrics_validation --root '<metrics 数据集>' --minute-root '<分钟数据集>' --version '<版本>' --verify-symbol ETHUSDT
python -m bench.crypto_metrics_validation --root '<metrics 数据集>' --minute-root '<分钟数据集>' --version '<版本>' --collect
```

分币报告使用独立文件名，记录版本、分钟版本和 `complete=false`（仅覆盖部分币时）；不覆盖全库报告。合并要求内容版本、分钟版本、所有币种覆盖和差异计数相符。Python 校验入口也可传 `symbols=["BTCUSDT"]`，默认仍核对整个版本。

### C8 独立审查后的对齐保护

当前登记常量、规则版本、阈值 0.01/0.1/0.05、最少样本 120、K=2、窗口 24 小时均进入内容版本。独立 verify 不调用生产对齐函数或读取生产常量，另行固定登记表与规则。真实分布、版本逐行对照和验收数字见 `C8_NOTES.md`。
