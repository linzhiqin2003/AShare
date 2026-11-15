# 币安规则（加密货币回测的事实依据）

设计见 DESIGN 4.17、6.10。下文由 2026-11-15 的调研整理，每条带出处；标「未核实」的不能当历史事实用，档案里按 4.17 的规则处理。

调研日期：2026-11-15。范围：币安，普通用户 VIP0，不用 BNB 抵扣。

## 0. 速览（回测可直接用的参数）

| 项目 | 取值 | 状态 |
|---|---|---|
| 现货 VIP0 | maker 0.1% / taker 0.1%，2017 至今未变；BTCUSDT 在 2022-07-08 14:00 ~ 2023-03-22 00:00 UTC 为 0/0 | 已核实（中间年份为推断） |
| U 本位 VIP0 | maker 0.02%；taker 0.04%（官方公告 2019-12-20 08:00 UTC 起，至少到 2023-05 收费页仍是）→ 0.05%（现行） | 两端已核实；切换在 2023-05-31 与 2024-02-20 之间，确切日未核实，也没有公告（见第 2 节） |
| 交割费 | = taker 费率 | 已核实（FAQ 2021-02-03） |
| 交割价 | 交割前 30 分钟（07:30:00–07:59:59 UTC）每秒价格指数算术平均 | 已核实（FAQ 2026-09 更新版） |
| 资金费 | 持仓数量 × 结算时标记价 × 费率；00/08/16 UTC；BTCUSDT/ETHUSDT 一直 8h；其他品种看 CSV 的 funding_interval_hours | 已核实 |
| 结算时刻开仓 | 官方只说以结算时刻是否持仓为准、有约 15 秒偏差，无确定规则 | 部分核实 |
| 强平触发 | 标记价 | 已核实 |
| BTC/ETH 第一档 MMR | BTC 0.40%（2019-11-14 起一直如此）；ETH 2020-05~09 0.65%、2021-01-24 起 0.50%、2023-12-24 09:35 起 0.40% | 已核实（ETH 2020-09-27~2021-01-24 之间的切换日未核实） |
| 强平清算费 | 2021-06 以前按公告逐段（例：2021-02-07 至 2021-05-20 BTC 2.00%、ETH 3.00%）；2021-07 起只有月度存档点，2021–2022 变动频繁；2023-11 起 1.25% | 公告段已核实；存档点之间未核实 |
| 最小名义 | 2021-01-25 起有门槛；2023-11-02 起 BTC 100 / ETH 20；现在 BTC 50、ETH 20、多数 5 | 部分核实 |
| 时间戳 | 现货 2025-01-01 起微秒；U 本位至 2026-10 仍毫秒 | 已核实（实测） |

## 1. 现货 VIP0 手续费（不用 BNB 抵扣）

| 时期 | maker | taker | 证据 |
|---|---|---|---|
| 2017 上线 ~ 2018-01 | 0.1% | 0.1%（不分 maker/taker，统一 0.1%） | 旧收费页 Wayback 存档 2018-01-09：「0.1% trading fee … A 50% discount will be applied, if you use BNB … which is 0.05%」https://web.archive.org/web/20180109144035/https://www.binance.com/fees.html |
| 2018-07 | 0.1% | 0.1% | 收费页内嵌 JSON（Wayback 2018-07-25）：`"level":0,"makerCommission":0.001,"takerCommission":0.001` https://web.archive.org/web/20180725204653/https://www.binance.com/en/fee/schedule |
| 2020-01 | 0.1% | 0.1% | 同上（Wayback 2020-01-04）：level 0 maker 0.001 / taker 0.001 https://web.archive.org/web/20200104084532/https://www.binance.com/en/fee/schedule |
| 当前（2026-11-15） | 0.100% | 0.100% | 官方收费页 https://www.binance.com/en/fee/schedule ：Regular User（30 天交易量 < 1,000,000 USD）maker 0.100% / taker 0.100% |

- 结论：**普通用户现货 maker/taker 自 2017 年上线至今一直 0.1%/0.1%**，没找到任何调整 VIP0 基础费率的公告。2021–2025 年的收费页存档改为前端渲染，无法从存档逐年核对（2021-01、2023-01、2024-05、2025-01 四个存档均取不到费率表），中间年份是「前后一致 + 无调整公告」推断。
- 2017-07 上线时的具体费率：**未核实**（最早存档为 2018-01；0.1% 是该存档值）。
- **重要例外：BTC 零手续费活动**（回测 BTCUSDT 现货时必须处理）：
  - 2022-07-08 14:00 UTC 起，13 个 BTC 现货对 maker 与 taker 均 0 手续费，含 **BTC/USDT**、BTC/BUSD、BTC/USDC、BTC/TUSD、BTC/EUR 等；这些对不再参与 BNB 抵扣与其他返佣。来源：公告《Binance Launches Zero-Fee Bitcoin Trading》https://www.binance.com/en/support/announcement/binance-launches-zero-fee-bitcoin-trading-10435147c55d4a40b64fcbf43cb46329 （2022-07 发布；摘要来自搜索结果，未打开全文）
  - 2023-03-22 00:00 UTC 起只保留 BTC/TUSD 零费，**BTC/USDT 恢复 0.1%**。来源：官方公告摘要（搜索结果引用的 2023-03 更新说明，链接同上的公告系列）；结束日期另由 CoinDesk / Kaiko 报道佐证（https://hotfix.coindesk.com/markets/2023/03/22/binance-curb-on-zero-fee-trading-may-cost-market-share-boost-trueusd-stablecoin ，2023-03-22；公告日 2023-03-15 来自 coingape，**未核实**）。
  - 其后还有 TUSD、FDUSD 交易对的零费活动（例如 FDUSD 零费后来结束，theblock 报道 https://theblock.co/post/300748/fdusd-trading-dominance-persists-despite-binance-ending-zero-fee-promotion ），只影响这些稳定币计价对，不影响 USDT 对；具体起止日期**未核实**。
- 回测建议：BTCUSDT 现货 0.1%/0.1%，但 2022-07-08 14:00 UTC ~ 2023-03-22 00:00 UTC 期间为 0/0。其他 USDT 现货对一直 0.1%。

## 2. U 本位合约 VIP0 手续费（不用 BNB 抵扣）

| 时期 | maker | taker | 证据 |
|---|---|---|---|
| 2019-09-09 上线 ~ 2019-12-20 | 未取得原文 | 未取得原文 | 上线初期有折扣活动（二手：theblock.co https://www.theblock.co/linked/39490// ，**未核实**） |
| 2019-12-20 08:00 UTC 起 | 0.02% | 0.04% | 官方公告（Codex 审查时核对）https://www.binance.com/en/support/announcement/detail/360037673592 |
| 2020-10 ~ 2023-06 | 0.02% | 0.04% | 合约收费页内嵌 JSON，level 0：`"makerCommission":0.0002,"takerCommission":0.0004`，Wayback 存档 2020-10-03、2021-04-16、2022-04-23、2023-05-31 均一致：https://web.archive.org/web/20201003083922/https://www.binance.com/en/fee/futureFee ；https://web.archive.org/web/20230531224744/https://www.binance.com/en/fee/futureFee 。官方 FAQ《Binance Futures Fee Structure & Fee Calculations》存档 2021-12-29 ~ 2023-06-21 的示例均写「regular maker commission: 0.02%; taker commission: 0.040%」（https://web.archive.org/web/20230621192025/https://www.binance.com/en/support/faq/binance-futures-fee-structure-fee-calculations-360033544231 ） |
| 现行（切换日未核实） | 0.02% | 0.05% | 同一 FAQ 存档 2024-02-20 示例改为「maker commission: 0.02%; taker commission: 0.05%」（https://web.archive.org/web/20240220064115/https://www.binance.com/en/support/faq/binance-futures-fee-structure-fee-calculations-360033544231 ）；当前官方 FAQ https://www.binance.com/en/support/faq/360033544231 （更新 2026-05-01）原文 "a Regular User's maker fee is 0.02% and a Regular User's taker fee is 0.05%" |

- **2026-11-15 再查**：Wayback 上 2023-05-31 之后到 2024-03 没有收费页存档（2024 年以后的存档是前端渲染、HTML 里没有费率）；繁中 FAQ 存档 2023-09-29（https://web.archive.org/web/20230929144654/https://www.binance.com/zh-CN/support/faq/%E5%B9%A3%E5%AE%89%E5%90%88%E7%B4%84%E6%89%8B%E7%BA%8C%E8%B2%BB%E7%B5%90%E6%A7%8B%E5%92%8C%E6%89%8B%E7%BA%8C%E8%B2%BB%E8%A8%88%E7%AE%97-360033544231 ）里币本位示例已按 2023-09-26 调费改过、U 本位示例仍是 0.040%，暗示 2023-09 底仍是 0.04%（示例，弱证据）；「Latest Binance News」等 4 个公告栏目 2023-06~2024-03 没有 U 本位标准费率调整的标题。回测口径不变（假设 2023-06-01，偏悲观），可用 `fee_switch` 在 2023-06-01~2024-02-20 之间做敏感性。
- **taker 0.04% → 0.05% 的确切生效日：未核实**。硬证据只有两端：2023-05-31 收费页存档仍是 0.04%，现行收费页 / FAQ 是 0.05%。FAQ 示例写明是假设费率，它在 2023-06-21 与 2024-02-20 存档之间从 0.04% 变成 0.05% 不能当作切换区间的证据。没有找到官方公告；若需精确日期，可再查 Wayback 2023-06~2024-01 期间 `binance.com/en/fee/futureFee` 的存档。
- 注：FAQ 里币本位例子 2022–2023 年写「regular user's maker 0.015%; taker 0.040%」，那是**币本位**示例，不是 U 本位，别混用。
- 注意：FAQ《Binance Futures Fee Structure & Fee Calculations》现行版写明示例采用假设费率，所以示例从 0.04% 变成 0.05% 不能证明真实费率在两个存档日期之间切换；收费页内嵌 JSON（2023-05-31 存档）仍是 0.04% 是硬证据，现行收费页 0.05% 是硬证据，切换日仍未核实。
- **回测口径（唯一的一份费率时间表，DESIGN 4.17 引用这里）**：U 本位 VIP0 maker 0.02%（2019-12-20 起）；taker 2019-12-20 08:00 UTC 至 2023-05-31 为 0.04%（证据：官方公告 + 2023-05-31 收费页存档），**假设** 2023-06-01 00:00 UTC 起为 0.05%（证据只到「现行 0.05%」，切换日是为回测选的假设，偏悲观），可选按区间做敏感性；2019-12-20 以前的回测缺费率，报错。交割费 = taker。现货 VIP0 maker/taker 0.1%（BTCUSDT 零费率窗口不体现，内核费率不能按交易对分，偏悲观）。交割合约交割费 = taker 费率（第 3 节）。

## 3. U 本位季度交割合约

### 3.1 交割时间与交割价
- 到期日：季度最后一个周五 08:00 UTC（例：BTCUSDT 0326 到期 2021-03-26 08:00 UTC）。来源：FAQ《USDT-Margined Quarterly Contract Specification and Trading Rules》https://www.binance.com/en-AE/support/faq/detail/3ae441db4ae740e19af3fe9228eb6619 （发布 2021-02-03）。实测 exchangeInfo：BTCUSDT_261225 `deliveryDate` 1798185600000 = 2026-12-25 08:00 UTC，一致。
- 交割价计算：FAQ《What Are Mark Price and Price Index in USDⓈ-Margined Futures》https://www.binance.com/en/support/faq/360033525071 （首发 2019-09-09，更新 2026-09-25）以 BTCUSDT 0925 为例：
  - 交割日距交割 >30 分钟：Mark Price = Price Index + Moving Average（30 秒基差均值）
  - 距交割 ≤30 分钟：原文 "Average of the Price Index, calculated every second between 07:30:00 and 07:59:59 (UTC) on the delivery day."
  - 即**交割价 = 交割前最后 30 分钟（07:30:00–07:59:59 UTC）每秒价格指数的算术平均，共 1800 个样本**。FAQ 把它写成「交割前最后 30 分钟的标记价」，交割按该价结算（「交割价 = 这一标记价」是该 FAQ 的语境，未另找到单独写明「Delivery Price =」的 U 本位页面）。
  - 二手信息称另一份 U 本位季度 FAQ 用「最后 1 小时每秒指数均价」（07:00:00–07:59:59，以 2020-09-25 为例）——**未核实**，来自搜索摘要，与上面 2026-09 更新的官方 FAQ 冲突，以官方 FAQ 的 30 分钟为准。
- 回测近似：交割价 ≈ indexPriceKlines 1m 中 07:30–07:59 共 30 根 K 线收盘价（或 OHLC 均值）的平均。data.binance.vision 的 indexPriceKlines 按 pair（如 BTCUSDT）存放，与永续共用同一指数（推断，未核实）。

### 3.2 交割费
- 官方 FAQ（同 3.1 第一个链接，发布 2021-02-03）原文："The Settlement Fee is charged the same as the taker fee for all positions settled on the delivery date." 即**交割费率 = 当时的 taker 费率**（VIP0 无 BNB 抵扣按 taker 计，见第 2 节）。
- 交割合约没有资金费（该 FAQ 未提资金费；交割合约本身无资金费机制，属常识，未另找官方原文）。
- 交割前是否禁止开新仓：二手资料称交割前 10 分钟只能平仓（**未核实**，来源：搜索摘要中的 Binance Square 旧帖，针对币本位）。
- 极端行情下币安可推迟交割（搜索摘要，**未核实**）。

## 4. U 本位永续资金费

主来源：币安 FAQ《Introduction to Binance Futures Funding Rates》https://www.binance.com/en/support/faq/introduction-to-binance-futures-funding-rates-360033525031 （首发 2019-09-09，页面显示更新于 2026-03-06，2026-11-15 读取）

### 4.1 公式
- 原文："Funding Amount = Nominal Value of Positions * Funding Rate"，"Nominal Value of Positions = Mark Price * Size of a Contract"。即 **资金费 = 持仓数量 × 标记价 × 资金费率**，费率为正时多付空，为负时空付多；交易者之间互付，币安不抽成。
- 用的标记价：FAQ 只写 Mark Price，并说明结算用的是实时计算的标记价，可能与交易页显示的略有差异（同页）。
- 实测佐证（2026-11-15）：API `GET https://fapi.binance.com/fapi/v1/fundingRate?symbol=BTCUSDT` 每条记录带 `markPrice` 字段；2026-11-10 三次结算的 markPrice 与 `markPriceKlines` 1m 在结算整分钟那根 K 线的 open 相同或差 <0.003%（75331.44223188 vs 75331.44223188；74812.7 vs 74812.7；73187.136 vs 73188.955）。回测可用「结算时刻 1m 标记价 K 线开盘价」近似。
- API 的 markPrice 字段**从 2023-10-31 08:00 UTC 起才有值**，之前为空字符串（实测 BTCUSDT，最早一条 2019-09-10 00:00 markPrice 为 ""）。更早的结算要用 markPriceKlines（BTCUSDT 从 2020-01 起）补；2019-09~2019-12 期间连标记价 K 线都没有（data.binance.vision 上 BTCUSDT 的 markPriceKlines 最早 2020-01，见第 6 节）。
- 费率公式（同 FAQ）："Funding Rate (F) = [Average Premium Index (P) + clamp (interest rate - Premium Index (P), 0.05%, -0.05%)] / (8 / N)"，N 为结算间隔小时数；利率默认每 8 小时 0.01%。上下限：部分品种 ±0.75×维持保证金率，其余 U 本位永续 ±2%（回测直接用历史实际费率即可，不需要重算）。

### 4.2 结算时间
- 默认每 8 小时：00:00、08:00、16:00 UTC（同 FAQ）。
- 4 小时品种：00/04/08/12/16/20 UTC。来源：公告《Important Updates on Funding Rate Settlement Frequency of USDⓈ-M Perpetual Contracts (2025-05-02)》https://www.binance.com/en/support/announcement/3243c81a35bd4f0c86a37315c3dc96cc （搜索摘要，正文未打开）
- 规则（FAQ 8.1/8.2）：前一次结算触及费率上/下限时，可把 8h 或 4h 品种改为 1h，从下一次结算起生效；1h 品种在连续 16 个周期 |费率| ≤ 0.025% 后可恢复为 4h。
- 实际时间有偏差：FAQ 写结算有约 15 秒偏差；API/CSV 的时间戳常带几毫秒尾数（如 `1794326400002`）。

### 4.3 哪些品种改了间隔、从何时起
- **BTCUSDT、ETHUSDT 从上线到 2026-10 一直是 8 小时**：实测 data.binance.vision fundingRate 月文件 2020-01~2026-10 共 82 个月，`funding_interval_hours` 全部为 8（各 7488 条）。
- 当前全量快照（实测 2026-11-15 `GET https://fapi.binance.com/fapi/v1/fundingInfo`，只列出被调整过上下限或间隔的品种）：822 个品种中 4h 466 个、8h 354 个、1h 2 个。每条带 `updateTime`（例如 TRBUSDT 1697703073725 = 2023-10-19，LPTUSDT 1752854309429 = 2025-07-18）。fundingInfo 未列出的品种为默认 8h。
- 已知的批量改 4h 公告（均为搜索摘要，未逐一打开）：
  - LPTUSDT 8h→4h，2023-08-13 公告 https://www.binance.com/en/support/announcement/detail/ed3c7b7e35b546aeab635cb4a5b3ee80
  - 一批（IMX、LPT、API3、YGG 等）2023-10-09 08:30 UTC 起、另一批（NMR、TRB、BLZ 等）2023-10-12 08:00 UTC 起 4h（来源：搜索结果摘要，对应公告 https://www.binance.com/en/support/announcement/98d6b24d3e5c4f84a8ed04087997d8d0 ，**未核实**具体名单）
  - 2025-07-22 12:00 UTC 起 XTZUSDT、MYXUSDT 等 5 个恢复 4h（**未核实**，二手来源 invezz.com https://invezz.com/news/2025/07/22/binance-resumes-4-hour-funding-rate-settlement-for-five-usdt-contracts/ ）
- **回测建议**：不必追公告，直接用 fundingRate CSV 每行的 `funding_interval_hours` 与 `calc_time`，它逐条记录了当时的实际间隔（2020-01 的文件里就已有该列）。

### 4.4 恰好在结算时刻开仓是否收费
- FAQ 原文："You are only liable for funding payments in either direction if you have open positions at the pre-specified funding times."；"If you close your position prior to the funding time, you will not pay or receive any funding."；并提示实际结算时间有约 15 秒偏差，结算时刻附近开/平的仓位可能被收或不被收。
- 结论：官方没有给出「恰好在 00:00:00 开仓」的确定规则，只说以结算时刻是否持仓为准、且有 ±15 秒不确定。回测约定建议：在结算时刻 t 成交的开仓**不收** t 的资金费、在 t 成交的平仓**收** t 的资金费（等价于「t 之前已持有的仓位收费」）；这是建模约定，不是官方规则。

## 5. 强平

### 5.1 触发价格
- 用**标记价**触发。FAQ《Binance Futures Liquidation Protocols》原文："Liquidation occurs when the Mark Price hits the liquidation price of a position."；条件："Collateral = Initial Collateral + Realized PnL + Unrealized PnL < Maintenance Margin"。https://www.binance.com/en/support/faq/360033525271 （首发 2019-09-09，更新 2026-01-04，2026-11-15 读取）
- 同页：Margin Ratio = Maintenance Margin / Margin Balance；有「Smart Liquidation」尽量避免一次全部强平（部分减仓），页面无细则。回测一般简化为「标记价触及强平价即全部平仓」，会略微高估强平损失。

### 5.2 BTCUSDT / ETHUSDT 第一档维持保证金率
| 时间 | BTCUSDT 第一档 | ETHUSDT 第一档 | 来源 |
|---|---|---|---|
| 2021-06-22 03:30 UTC 调整时（第一档未改） | 0 < 名义 ≤ 50,000 USDT，101–125x，MMR 0.4%，IMR 0.8% | 该公告未含 ETH | 《Updates on Leverage and Margin Tiers of Binance Futures Contracts (2021-06-22)》https://www.binance.com/en/support/announcement/f55d25695ca8453ab4c132c1fd4dc36a （2021-06-22） |
| 2023-12-24 09:35 UTC 调整前 | 0–50,000，101–125x，MMR 0.40% | 0–200,000（原文如此），76–100x，MMR 0.50% | 公告 https://www.binance.com/es/support/announcement/detail/d75c5ca94f704e96a6a4e55ffddfd65d （发布 2024-01-02，生效 2023-12-24 09:35 UTC） |
| 2023-12-24 09:35 UTC 调整后 | 不变：0–50,000，101–125x，0.40% | 0–50,000，101–125x，0.40% | 同上 |
| 2025-08-19 06:30 UTC 调整前 | 0–300,000，101–125x，0.40% | 0–300,000，101–125x，0.40% | 公告 https://www.binance.com/en/support/announcement/detail/8e428625ebeb4cc7ae8678026846095c （发布 2025-08-17） |
| 2025-08-19 06:30 UTC 起 | 0–300,000，101–150x，0.40% | 0–300,000，101–150x，0.40% | 同上 |

- 结论：**BTCUSDT 第一档 MMR 至少自 2021-06 到 2025-08 一直是 0.40%**，变化的只是第一档名义上限（50,000 → 某时改为 300,000，具体日期**未核实**）和最高杠杆（125x → 150x）。ETHUSDT 第一档 2023-12-24 前为 0.50%，之后 0.40%。
- **2026-11-15 补齐（BTCUSDT）**：2019-09 上线时最高 20x，第一档 0–500,000、维持 2.5%（FAQ 存档 https://web.archive.org/web/20190918184207/https://binance.zendesk.com/hc/en-us/articles/360033162192-Leverage-Overview ）；2019-10-18 升到 125x（公告存档 https://web.archive.org/web/20200513174225/https://binance.zendesk.com/hc/en-us/articles/360035327011-Binance-Futures-Increases-Maximum-Leverage-to-125x-and-Launches-Leverage-Adjustment-Function ）；最早见到 0–50,000、0.4% 的是 2019-11-14 FAQ 存档（https://web.archive.org/web/20191114222312/https://binance.zendesk.com/hc/en-us/articles/360033162192-Leverage-Overview ，原文 "0-50,000, 0.4% Maintenance Margin"），之后 2020-05、2020-09、2021-01、2021-03 FAQ、2021-06-22 与 2023-12-24 公告、2023 年档位接口存档都一致。**第一档上限 50,000 → 300,000 在 2025-06-17 06:30 UTC**（公告 https://www.binance.com/en/support/announcement/detail/e778988edec446038ad536bcb0c8d460 ，2025-06-14 发布，原文 "update the leverage and margin tiers of USDⓈ-M BTCUSDT Perpetual Contract at 2025-06-17 06:30 (UTC)"，并写明已有仓位也受影响）。2026-11-15 实测档位接口 `https://www.binance.com/bapi/futures/v1/friendly/future/common/brackets`：BTC、ETH 第一档都是 0–300,000、0.40%、101–150x，`updateTime` 都是 2025-08-19。
- **ETHUSDT**：2019-11-29 上线 50x；2019-12-19 升 75x；2020-05-19 与 2020-09-27 FAQ 表（图片存档 https://web.archive.org/web/20200927195452im_/https://public.bnbstatic.com/image/cms/content/body/202009/a480eab7aa90d625b1dbd94a7301b789.png ）首 10,000、MMR 0.65%；2021-01-24 FAQ（https://web.archive.org/web/20210124091911/https://www.binance.com/en/support/faq/360033162192 ）100x、首 10,000、0.5%（切换日在两次存档之间，未核实）；档位接口存档显示 2022-11-17 起上限 100,000、2023-12-12 起 200,000（按接口的 `updateTime`，官方没有定义这个字段，作为存档证据）；2023-12-24 09:35 改为 0–50,000、0.40%、125x；2025-08-19 公告时已是 0–300,000（50,000 → 300,000 的日期未核实）。
- 档案（`panelbt.markets.get("binance_usdm").rules`）的证据分两级：**证据**（公告生效的那一小时、下一份公告的调整前值能接上的公告区间、存档或接口快照所在的那一小时、FAQ 写明的上限）默认直接用；**推断区间**（两端和中间几次观测都是同一个值的「观测连续」，以及按档位接口 `updateTime` 回填的区间——官方没有定义这个字段）默认不用，给了 `assume` 才用，并在诊断里记为假设。剩下没有任何记录的时段是缺口：`assume="conservative"` 取前后两个记录中更不利的一个（维持保证金率、清算费取高，上限、杠杆取低），`assume="previous"` 沿用前一个记录；诊断写明候选值、选中的值和在缺口的哪一侧。显式规则映射优先于推断区间和这两种策略。
- 2025-08 那次的完整 BTCUSDT 档位（第二档起）：0.3–0.8M 0.50%；0.8–3M 0.65%；3–12M 1.00%；12–70M 2.00%；70–100M 2.50%；100–230M 5%；230–480M 10%；480–600M 12.5%；600–800M 15%；800M–1.2B 25%；1.2–1.8B 50%（同上公告）。
- 当前档位查询接口 `GET /fapi/v1/leverageBracket` 需要签名（未调用）。exchangeInfo 里的 `maintMarginPercent: 2.5` / `requiredMarginPercent: 5.0` 是旧字段，不代表档位 MMR，别用。
- 维持保证金计算：币安档位制下 维持保证金 = 名义 × 本档 MMR − 速算扣除数（maintenance amount）；第一档速算扣除数为 0（第一档以外的扣除数本次未取得原文，**未核实**）。

### 5.3 强平清算费（Liquidation Clearance Fee / Insurance Clearance Fee）
- 公式（FAQ 原文）：按「适用的 Liquidation Clearance Fee 费率 × 仓位名义价值」计算，费率见交易规则页（https://www.binance.com/en/support/faq/360033525271 ）。搜索摘要中另一官方页写作 "Insurance Clearance Fee = Notional Value * Fee Rate"，从为仓位保留的资产中扣除，强平后已穿仓的仓位不收，费用可能划入保险基金（https://www.binance.com/en/support/faq/detail/98488a516eb84e3eb34605683dffd554 ，搜索摘要，未打开原文）。
- **当前费率（实测 2026-11-15 exchangeInfo 的 `liquidationFee` 字段）**：BTCUSDT 0.0125（1.25%），ETHUSDT 0.0125（1.25%）；全部在交易合约中 1.5% 617 个、2.0% 154 个、1.25% 28 个、2.5% 7 个。交割合约 BTCUSDT_261225 也是 1.25%。
- **历史费率（2026-11-15 补齐）**。公告（链接 `https://www.binance.com/en/support/announcement/detail/<编号>`，均用币安公告原文接口核对）：

| 生效（UTC） | BTCUSDT | ETHUSDT | 证据 |
|---|---|---|---|
| 2019-09 | 至多 0.5%（"A fee of up to 0.5% of the NOMINAL position value"） | 同 | FAQ 存档 https://web.archive.org/web/20190918185837/https://binance.zendesk.com/hc/en-us/articles/360033525271-Liquidation |
| 2020-05-20 ~ 2020-11-23 07:00 | 0.3% | 0.5% | FAQ 存档 https://web.archive.org/web/20200520125129/https://binance.zendesk.com/hc/en-us/articles/360033525271-Liquidation 、2020-09-27 存档，2020-11-23 公告的调整前值 |
| 2020-11-23 07:00 | 0.50% | 0.75% | 41d8a353103c4fe18bf25c403bf7af3b |
| （2021-02-07 之前某时） | 1.00% | 1.50% | 2021-02-07 公告的调整前值；与上一行接不上，中间的调整没找到公告 |
| 2021-02-07 04:00 ~ 2021-05-20 09:00 | 2.00% | 3.00% | aeeccb6d332a4e74a51c2ac05f9078e5（下一份公告的调整前值一致） |
| 2021-05-20 09:00 | 0.30% | 0.50% | 839a865d251f435f95fba499d2ad95c9 |
| 2021-06-08 07:00 | 0.60% | 未列 | 03cac798a8454f42848f4467a434ba5d |
| 2021-06-10 04:00 | 1.20% | 未列 | 6bd7a166980648469ab771b5835847c3 |

  2021-07 起只有 `exchangeInfo` 的 `liquidationFee` 存档点（`https://web.archive.org/web/<时间戳>id_/https://fapi.binance.com/fapi/v1/exchangeInfo`）：2021-07-16、08-15 BTC 1.2% / ETH 0.5%；09-14 0.6% / 1.0%；10-14 2.0% / 2.0%；11-13 3.0% / 3.0%；12-13 1.2% / 1.2%；2022-01-02 1.5% / 1.5%；03-18 0.5% / 0.5%；09-08 1.5% / 1.5%；09-16、11-16、11-27 1.75% / 1.5%；2023-11-02、2023-12-06、2024-04-08、2025-04-14 1.25% / 1.25%（后三份是 `www.binance.com/fapi/v1/exchangeInfo?showall=…`）。这段时间没有公告，存档点之间的变化日未核实；2021–2022 年变动频繁，存档点之间按缺口处理（默认报错，`assume="conservative"` 取较高者）。

## 6. data.binance.vision 数据集与路径（本节大部分为 2026-11-15 实测）

来源一：官方仓库 README https://github.com/binance/binance-public-data （无日期，2026-11-15 读取）
来源二：S3 目录列举 `https://s3-ap-northeast-1.amazonaws.com/data.binance.vision?delimiter=/&prefix=...`（data.binance.vision 前端页面就是读这个桶），2026-11-15 实测
来源三：下载样本文件解压查看（2026-11-15）

### 6.1 数据集目录（实测 S3 列举）
- U 本位合约 daily：`data/futures/um/daily/{aggTrades,bookDepth,bookTicker,indexPriceKlines,klines,markPriceKlines,metrics,premiumIndexKlines,trades}/`
- U 本位合约 monthly：`data/futures/um/monthly/{aggTrades,bookTicker,fundingRate,indexPriceKlines,klines,markPriceKlines,premiumIndexKlines,trades}/`
- **fundingRate 只有 monthly，没有 daily**；metrics、bookDepth 只有 daily。
- 现货 daily / monthly：只有 `aggTrades, klines, trades` 三类（现货没有标记价、指数价）。

### 6.2 路径模式（实测确认存在的文件）
- K 线：`https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2026-10.zip`
  - 模式 `data/futures/um/{daily|monthly}/klines/{SYMBOL}/{interval}/{SYMBOL}-{interval}-{YYYY-MM | YYYY-MM-DD}.zip`
- 标记价 K 线：`data/futures/um/monthly/markPriceKlines/BTCUSDT/1h/BTCUSDT-1h-2020-01.zip`（文件名里没有 markPrice 字样，与普通 K 线同名，只靠目录区分；下载后放同一目录会互相覆盖，注意）
- 指数价 / 溢价指数 K 线：同上结构，目录名 `indexPriceKlines` / `premiumIndexKlines`（目录实测存在，文件名模式按同一规则推断，未逐个下载）
- 资金费率：`data/futures/um/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-2020-01.zip`，BTCUSDT 从 2020-01 起到 2026-10 都有
- 交割合约：与永续同在 `klines` 目录下，符号带到期日，如 `data/futures/um/monthly/klines/BTCUSDT_241227/...`。BTCUSDT 交割合约目录实测从 `BTCUSDT_210326` 开始（2021-03 季度），此后每季度一个：210625、210924、211231、220325 …… 250926 等
- metrics：`data/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-2020-09-01.zip`（BTCUSDT 最早 2020-09-01）
- 每个 zip 旁边有同名 `.CHECKSUM` 文件（README）
- README 说：daily 文件次日可用，monthly 文件在每月第一个周一生成；历史文件可能被修补（README 列了 2022-08-08 K 线修复、2022-04-21 aggTrades 对齐）

### 6.3 fundingRate CSV 列（实测）
```
calc_time,funding_interval_hours,last_funding_rate
1577836800000,8,-0.00012359
```
- **不含标记价**，只有结算时间（毫秒）、结算间隔小时数、费率三列。要算资金费金额，需另取 markPriceKlines（或用 API `GET /fapi/v1/fundingRate`，它返回 `markPrice` 字段，见第 4 节）。
- 2026-10 的 calc_time 带几毫秒偏移（如 `1790812800001`、`1790841600005`），对齐时要向下取整到秒/整点，不能用等号匹配。

### 6.4 CSV 表头（实测）
- U 本位 monthly K 线：2021-12 及以前**无表头**，2022-01 起**有表头** `open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore`（BTCUSDT 1h 抽查 2021-06/2021-12/2022-01/2022-03/2022-06/2022-12/2023-06/2026-10；daily 文件与其他品种未逐一核对）。读取时要按首行是否为数字判断。
- markPriceKlines 列名同普通 K 线，但 volume/quote_volume/taker 列全为 0，count 为 3600（1h）。
- 现货 K 线：2024-12、2025-01、2026-10 均**无表头**。

### 6.5 时间戳单位
- 现货：README 原文 "The timestamp for SPOT Data from January 1st 2025 onwards will be in microseconds."（https://github.com/binance/binance-public-data ）。实测：现货 `BTCUSDT-1h-2024-12` 为毫秒（`1733011200000`），`BTCUSDT-1h-2025-01` 为微秒（`1735689600000000`），2026-10 仍为微秒。**切换点 = 2025-01-01（按文件所属日期，不是按发布日期）**。
- U 本位合约：README 未提单位变更；实测 2026-10 的 klines、markPriceKlines、fundingRate 仍是**毫秒**。即截至 2026-11 合约数据没有切换到微秒（实测结论，非公告）。
- README 的 COIN-M K 线列名与现货略有不同（第 8 列 Base asset volume），本研究不涉及。

## 7. U 本位永续最小名义价值（MIN_NOTIONAL）

- **2021-01-25 07:00 UTC 起**引入最小下单名义价值规则，当时示例门槛为 1 USD；Reduce-Only 单不受限；币安可不经公告调整门槛。来源：公告《New Minimum Order Notional Limit Rule for USDⓈ-M Futures》https://www.binance.com/en/support/announcement/76719bbaeeb847bbac4daa2906fcdcc0 （发布 2021-01-21）。2021-01-25 之前没有该规则（以公告措辞 "New ... Rule" 推断）。
- 1 USDT → 5 USDT 的调整时间：**未核实**。只找到 FAQ 旧版写「threshold of 5 USDT」（搜索摘要，https://www.binance.com/de/amp/support/faq/360033161972/ ，无日期），没找到对应公告。
- **2023-11-02 10:00 UTC 起** BTCUSDT 永续 5.0 → 100.0 USDT，ETHUSDT 永续 5.0 → 20.0 USDT；不影响已挂订单。来源：公告《Updates on Minimum Notional Value for BTCUSDT and ETHUSDT Perpetual Contracts (2023-11-02)》https://www.binance.com/en/support/announcement/updates-on-minimum-notional-value-for-btcusdt-and-ethusdt-perpetual-contracts-2023-11-02-e4384cba297a4bd2a154be644d5d76f9 （发布 2023-10-27，更新 2023-10-30）
- **当前值（实测 2026-11-15 `GET https://fapi.binance.com/fapi/v1/exchangeInfo`）**：BTCUSDT **50**（不是 100）；ETHUSDT、BCHUSDT、LTCUSDT、ETCUSDT、LINKUSDT 为 20；其余 516 个在交易的 USDT 永续为 5。交割合约 BTCUSDT_261225 为 5。BTCUSDT 由 100 改为 50 的时间：**未核实**（没找到公告，FAQ 称可不经公告调整）。
- 其他实测规格（同一接口，2026-11-15）：BTCUSDT 数量步长 0.001、最小数量 0.001、tick 0.10；ETHUSDT 数量步长 0.001、tick 0.01；BTCUSDT `onboardDate` 1567965300000 = 2019-09-08 17:55 UTC。
- 回测含义：现在的 50 USDT 只对很小的账户有约束；历史上不同时期门槛不同，若要严格复现，2021-01-25 前无门槛、之后 1→5（时间未核实）、2023-11-02 起 BTC 100 / ETH 20、某时起 BTC 50。

