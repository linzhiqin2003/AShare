# 美国股票市场（NYSE / Nasdaq 等交易所上市普通股与 ADR）交易规则事实表

> 调研日期：2026-11-13。视角：中国内地个人通过海外券商开户（非美国税务居民，下称「非居民」），并注明美国本土个人的差别。
> 不含期权、场外粉单（OTC Pink）。标「旁证」的是券商帮助页、媒体或二手汇编；标「未核实」的没有找到官方原文。
## 现行规则速览表（截至 2026-11-13）

| 项目 | 现行值 | 生效日 | 出处 |
|---|---|---|---|
| SEC Section 31 规费 | 卖出成交额 × $20.60/百万美元，无上下限，买入不收 | 2026-04-04（此前 2025-05-14 至 2026-04-03 为 $0） | SEC Release 34-104909；FINRA Information Notice 3/17/26 |
| FINRA TAF | 卖出 $0.000195/股，每笔上限 $9.79；**2026-10-01 至 12-31 暂停为 $0**，2027-01-01 恢复 | 2026-01-01；暂停见 SR-FINRA-2026-021 | SR-FINRA-2024-019 费用表；SR-FINRA-2026-021 |
| 印花税/交易税 | 无 | — | — |
| 券商佣金（参考） | IBKR Pro Fixed $0.005/股，最低 $1，最高 1% | 查询日 2026-11-13 | IBKR 官网（搜索摘要） |
| 交易单位 | 1 股（无每手限制） | — | Reg NMS（round lot 仅影响报价） |
| 涨跌停 | 无；个股 LULD 5 分钟暂停；全市场熔断 7%/13%/20%（标普 500） | LULD 2013-04-08；MWCB 现行版 2013-04-08 | LULD Plan；NYSE Rule 7.12 |
| 当日买入当日卖出 | 允许 | — | — |
| 交收周期 | T+1 | 2024-05-28（T+2 自 2017-09-05） | SEC Rule 15c6-1；SEC 新闻稿 2017-68 |
| PDT（$25,000） | **已取消**，改为 intraday margin 标准 | 2026-06-04（券商最迟 2027-10-20 完成实施） | FINRA Regulatory Notice 26-10 |
| 卖空提价规则 | Rule 201：较前收跌 ≥10% 后当日余下及次日，卖空价须高于全国最优买价 | 合规日 2011-02-28 | SEC Rule 201 FAQ |
| 初始保证金 | 50%（Reg T），卖空 150% | 1974 年起未变（日期未核实） | 12 CFR 220.12 |
| 维持保证金 | 多头 25%；空头 max(30%, $5/股)（≥$5），max(100%, $2.50/股)（<$5）；最低权益 $2,000 | 2010 年以来未变（未逐年核实） | FINRA Rule 4210(b)(c) |
| 股息预扣（非居民） | 一般 30%；中国内地居民 W-8BEN 适用协定 10% | 协定 1984 年签署；IRS 表 Rev. May 2023 | IRS Tax Treaty Table 1 |
| ADR 存托费 | 逐只约定，常见 ≤ $0.05/ADS/年，从股息中扣 | — | 各 ADR 存托协议（F-6/20-F 附件） |
| 除息日 | 现金股息除息日 = 登记日（T+1 下）；拆股在支付日后一日除权 | 2024-05-28 | FINRA Rule 11140（未核实） |
| 最小报价单位 | ≥$1：$0.01；<$1：$0.0001；半分钱 tick 修订已通过但推迟（据称至 2027-11） | 现行规则 2005 年 Reg NMS | Reg NMS Rule 612；SEC 34-101070 |
| 交易时段 | 9:30–16:00 ET；半日市 13:00 收 | — | NYSE 官网 |

---

## 1. 交易费用

### 1.1 SEC Section 31 交易规费（只对卖出收取）

- **性质与计费基础**：《1934 年证券交易法》第 31 条（15 U.S.C. 78ee）向交易所和 FINRA 按「covered sales」（受覆盖的**卖出**成交金额）征收，交易所/FINRA 再向会员券商收取，券商通常按笔转嫁给卖出的客户。买入不收。按成交金额 × 费率。
- **费率形式**：每年由 SEC 发「Fee Rate Advisory」并发布命令设定，单位为「每百万美元成交额 $X」。无每笔上限、无下限。
- **取整**：法律本身不规定客户层面的取整；券商普遍按笔计算后**向上取整到 0.01 美元**（旁证：各券商费用说明，未找到统一官方规定）。
- **生效日口径**：费率按「charge date」生效；FINRA 说明对 OTC 交易，charge date 即交易日（trade date），不是交收日（FINRA Information Notice 2/16/21 尾注）。

**历次费率（2010 年至今）**：

| 起始日（含） | 费率（$/百万美元） | 来源 | 备注 |
|---|---|---|---|
| 2009-04-10 | 25.70 | 旁证 traderstatus.com 汇编 | 2010 年初时生效值 |
| 2010-01-15 | 12.70 | 旁证 traderstatus.com；FINRA 当期 Information Notice 未逐条打开 | 未核实（官方原文） |
| 2010-04-01 | 16.90 | 同上 | 未核实（官方原文） |
| 2011-01-21 | 19.20 | 同上；FINRA Information Notice 1/6/11「New Rate」（标题对应） | 数字未核对原文 |
| 2012-02-21 | 18.00 | 同上；FINRA Information Notice 2/3/12 | 数字未核对原文 |
| 2012-04-01 | 22.40 | 同上；FINRA Information Notice 3/8/12 | 数字未核对原文 |
| 2013-05-25 | 17.40 | 同上；FINRA Information Notice 5/1/13 | 数字未核对原文 |
| 2014-03-18 | 22.10 | 同上；FINRA Information Notice 3/3/14 | 数字未核对原文 |
| 2015-02-14 | 18.40 | 同上；FINRA Information Notice 1/20/15；SEC 2016-2 advisory 写明 2016-02-15 前为 $18.40 | 18.40 已有两源 |
| 2016-02-16 | 21.80 | SEC 新闻稿 2016-35、SEC Release 34-76848、FINRA Information Notice 012016 | 已核实 |
| 2017-07-04 | 23.10 | SEC 新闻稿 2017-111；FINRA Information Notice 6/9/17 | 已核实 |
| 2018-05-22 | 13.00 | FINRA Information Notice 5/7/18（由 23.10 降至 13.00） | 已核实 |
| 2019-04-16 | 20.70 | 旁证 traderstatus.com；FINRA Information Notice 2/3/20 写明 2020 年调整前现行费率为 $20.70 | 日期单源 |
| 2020-02-18 | 22.10 | FINRA Information Notice 2/3/20；SEC 新闻稿 2020-7 | 已核实 |
| 2021-02-25 | 5.10 | SEC 新闻稿 2021-8（Fee Rate Advisory #2 FY2021）；FINRA Information Notice 2/16/21 | 已核实 |
| 2022-05-14 | 22.90 | SEC 新闻稿 2022-60；FINRA Information Notice 4/21/22 | 已核实 |
| 2023-02-27 | 8.00 | SEC 新闻稿 2023-15；FINRA Information Notice 2/14/23 | 已核实 |
| 2024-05-22 | 27.80 | SEC Release 34-99973（2024-04-17）；FINRA Information Notice 05/01/24 | 已核实 |
| 2025-05-14 | **0.00** | SEC Fee Rate Advisory FY2025（2025-04-08）；FINRA Information Notice 4/24/25 | 已核实：FY2025 剩余期间费率为零 |
| 2026-04-04 | **20.60** | SEC Release 34-104909（2026-02-27）；SEC Fee Rate Advisory 2026-2；FINRA Information Notice 3/17/26 | **现行**。有效至 FY2027 拨款法生效后第 60 日 |

- 注意 traderstatus 汇编在 2020-10-01 一行写 20.70，与 SEC/FINRA「22.10 延续至 2021-02-24」冲突，以 SEC 为准（**冲突**，已按官方处理）。
- 截至 2026-11-13，SEC 网站「Fee Rate Advisories」列表中最新的 Section 31 advisory 仍是 2026-02-27 的 FY2026 版（之后只有 2026-08-21 的 Section 6(b) 注册费 advisory），故 $20.60 仍有效。FY2027 拨款落地后会再调，**需定期复查**。
- 出处：
  - SEC Fee Rate Advisories 列表：https://www.sec.gov/rules-regulations/fee-rate-advisories
  - FY2026：https://www.sec.gov/rules-regulations/fee-rate-advisories/2026-2 ；https://www.sec.gov/files/rules/other/2026/34-104909.pdf ；https://www.finra.org/rules-guidance/notices/information-notice-20260317
  - FY2025 零费率：https://www.finra.org/rules-guidance/notices/information-notice-20250424
  - FY2024：https://www.sec.gov/files/rules/other/2024/34-99973.pdf ；https://www.finra.org/rules-guidance/notices/information-notice-050124
  - FY2023：https://www.sec.gov/news/press-release/2023-15 ；https://www.finra.org/rules-guidance/notices/information-notice-021423
  - FY2022：https://www.sec.gov/news/press-release/2022-60 ；https://www.finra.org/rules-guidance/notices/information-notice-042122
  - FY2021：https://www.sec.gov/news/press-release/2021-8 ；https://www.finra.org/rules-guidance/notices/information-notice-021621
  - FY2020：https://www.sec.gov/news/press-release/2020-7 ；https://www.finra.org/rules-guidance/notices/information-notice-020320
  - 2018：https://www.finra.org/industry/information-notice-050718
  - 2017：https://www.sec.gov/newsroom/press-releases/2017-111
  - 2016：https://www.sec.gov/news/press-release/2016-35 ；https://www.sec.gov/file/34-76848
  - FINRA Section 31 通知目录（2011 起各期标题）：https://www.finra.org/taxonomy/term/1758
  - 二手汇编（2001–2022）：https://traderstatus.com/traders/trader-info/sec-fee-rates/

### 1.2 FINRA 交易活动费 TAF（只对卖出收取）

- **性质**：FINRA By-Laws Schedule A Section 1，向会员券商按**卖出**的「covered equity security」**股数**收取，券商通常转嫁给卖出客户。按股数 × 费率，**每笔有上限**；若成交价低于 TAF 费率本身，则按成交金额收取（Schedule A 原文「if the execution price for a covered security is less than the Trading Activity Fee rate ... the fee assessed will be the execution price」）。
- 客户层面取整：券商普遍向上取整到 0.01 美元（旁证，未核实）。

| 起始日（含） | 费率（$/股，卖出） | 每笔上限 | 来源 |
|---|---|---|---|
| 2010 年初（实施于 2010-11-01 前已有） | 0.000075 | 3.75 | SEC 2011 年公告（引述现行费率），FINRA Regulatory Notice 11-27 背景 |
| 2011-07-01 | 0.000090 | 4.50 | FINRA Regulatory Notice 11-27（由 12-06 引述） |
| 2012-03-01 | 0.000095 | 4.75 | FINRA Regulatory Notice 12-06 |
| 2012-07-01 | 0.000119 | 5.95 | FINRA Regulatory Notice 12-31 |
| 2022-01-01 | 0.000130 | 6.49 | SR-FINRA-2020-032（Exhibit 5，「to take effect on January 1, 2022」） |
| 2023-01-01 | 0.000145 | 7.27 | SR-FINRA-2020-032 |
| 2024-01-01 | 0.000166 | 8.30 | SR-FINRA-2020-032 |
| 2025-01-01 | 0.000166（不变） | 8.30 | SR-FINRA-2024-019 费用调整表（2025 列「no change」） |
| 2026-01-01 | 0.000195 | 9.79 | SR-FINRA-2024-019 费用调整表（「fee increases take effect on January 1 of the year stated」） |
| **2026-10-01 至 2026-12-31** | **0.00** | 0.00 | SR-FINRA-2026-021（立即生效的规则修改，「temporarily pause assessment of the TAF for three months」） |
| 2027-01-01 | 0.000195（恢复） | 9.79 | SR-FINRA-2026-021 Exhibit 5；SR-FINRA-2026-020 把 2024 年文件里 2027–2029 的继续上调**推迟两年**，「maintains TAF rates at their current levels through December 31, 2028」 |

- SR-FINRA-2024-019 原计划 2027 年 0.000232/11.61、2028 年 0.000240/12.05、2029 年 0.000249/12.50，已被 SR-FINRA-2026-020 推迟，不应写进档案（SR-FINRA-2026-020 正文本次未打开，以 2026-021 中的引述为据：**单源**）。
- 注意：费率暂停期间券商是否同步停止向客户收取，取决于券商；部分券商可能照收后返还或继续收取（未核实）。
- 出处：
  - FINRA TAF 页面（历次通知目录）：https://www.finra.org/rules-guidance/guidance/trading-activity-fee
  - Regulatory Notice 12-06：https://www.finra.org/rules-guidance/notices/12-06 ；12-31：https://www.finra.org/rules-guidance/notices/12-31
  - SR-FINRA-2020-032：https://www.finra.org/sites/default/files/2020-10/SR-FINRA-2020-032.pdf
  - SR-FINRA-2024-019 费用调整表：https://www.finra.org/rules-guidance/rule-filings/sr-finra-2024-019/fee-adjustment-schedule
  - SR-FINRA-2026-021（TAF 暂停）：https://www.finra.org/sites/default/files/2026-09/SR-FINRA-2026-021.pdf
  - SR-FINRA-2026-020（推迟后续上调）：https://www.finra.org/sites/default/files/2026-09/SR-FINRA-2026-020.pdf （未打开）


### 1.3 其他法定/自律组织费用

- **印花税、交易税**：美国联邦层面对股票买卖**没有**印花税或交易税。（一般常识，未找到需要引用的单独官方页面；纽约州股票过户税 1981 年起全额退税，与回测无关。）
- **CAT 费用（Consolidated Audit Trail）**：由各交易所和 FINRA 向会员券商收取，按「executed equivalent share」计，买卖双方都可能被收取，券商**可以但不必**转嫁给客户。据交易所 2026 年费用文件摘要：2026-05-01 至 2026-12-31 的 CAT 费率为每股 $0.000001（NMS 股票 1 股 = 1 个 executed equivalent share）。量级比 TAF 低两个数量级，日线回测可忽略。出处为 policyrisk.com 对 Federal Register 文件的摘要（如 https://policyrisk.com/federal-register/2026-09260 ），**未核实**原始 Federal Register 文本；2024–2025 年各期 CAT 费率未查。
- **交易所费、清算费**：交易所 taker 费（上限按 Reg NMS Rule 610(c)：现行每股 $0.003，股价 ≥ $1）和 DTC/NSCC 清算费由券商承担或按「分层/阶梯」套餐转嫁，非法定向客户收取的费用，按佣金处理。Rule 610(c) 降至 $0.001（10 mils）的修订见第 8 节，尚未生效。
- **ADR 存托费**：见第 6 节。
- **非居民资本利得税**：非居民个人卖出美股的资本利得一般不在美国征税（不在美国停留 183 天以上的前提下），不预扣。**例外**：卖出公开交易合伙企业（PTP，多为能源 MLP，如部分在 NYSE 上市的 LP 单位）时，券商须按 IRC §1446(f) 对**卖出总价款**预扣 10%，自 2023-01-01 起执行（出处：IRS T.D. 9926 最终规则，本次**未打开原文**，未核实）。回测若会选中 PTP，必须排除或单独建模。

### 1.4 券商佣金（非法定，仅作参考；查询日 2026-11-13）

| 券商 / 套餐 | 美股佣金 | 每单下限 / 上限 | 监管费处理 | 出处与可信度 |
|---|---|---|---|---|
| Interactive Brokers，IBKR Pro Fixed | $0.005/股 | 下限 $1.00；上限为成交额 1% | 官方页面称固定价「包含所有交易成本」，但 IBKR 历来把 SEC 费与 FINRA TAF 作为「Transaction Fees」另列转嫁（**未核实**） | IBKR Commissions 页面（https://www.interactivebrokers.com/en/pricing/commissions-stocks.php 直连 403，内容来自搜索摘要） |
| Interactive Brokers，IBKR Pro Tiered | 按月成交股数分档 $0.0035 → $0.0005/股 | 搜索摘要称下限 $1.00、上限 1%；我印象中 Tiered 下限为 $0.35（**冲突，未核实**） | 交易所费、清算费、监管费全部转嫁（含返佣） | 同上 |
| moomoo Financial Inc.（富途旗下美国券商） | 佣金 $0（页面佣金栏为空，未确认） | 平台费固定式 $0.005/股，每单最低 $1.00；另有阶梯式可选 | 交收费 $0.003/股；SEC 费只在卖出收 | moomoo US 收费页（2026-05-20 更新）https://www.moomoo.com/us/hans/support/topic4_66 （旁证：券商页面） |
| 富途证券国际（香港），面向内地个人常用 | 第三方称佣金低至 $0.0049/股 | 第三方称最低 $0.99；平台费另计 | 监管费转嫁 | MoneySmart HK 比较页 https://www.moneysmart.hk/zh-hk/investments/futu-securities/us-stock （第三方，**未核实**） |

- 建议档案里把「佣金」做成可配置项，默认取 IBKR Fixed：$0.005/股，下限 $1，上限 1% 成交额，双边收取。

### 1.5 第 1 节未核实或有冲突的条目

1. Section 31 在 2010-01-15 至 2014-03-18 之间的 6 档费率数字只来自 traderstatus.com 汇编，FINRA 当期通知仅核对了标题与日期吻合，未逐份打开核对数字。
2. Section 31 2019-04-16 的生效日为单源（数值 $20.70 已由 FINRA 2020 通知旁证）。
3. TAF 2010-01-01 至 2011-06-30 的 $0.000075/$3.75 只见于 2011 年 SEC 公告摘要，未开原文。
4. SR-FINRA-2026-020「2027–2028 年维持现行 TAF」只从 SR-FINRA-2026-021 的引述得知，其 SEC 公告与是否被暂停（suspend）未查。
5. 客户层面的取整规则（向上取整到分）、是否最低收 $0.01：未找到官方规定，各券商不同。
6. CAT 费率：只有二手摘要。
7. IBKR Tiered 每单下限：冲突。
8. §1446(f) PTP 10% 预扣：未开原文。

---

## 2. 交易单位

- **没有法定「每手」**：美股可以按 1 股买卖。交易所报价体系中「round lot」传统上为 100 股，不足 100 股为 odd lot，可正常成交，只是历史上 odd lot 不进入 NBBO 报价（SEC Regulation NMS Rule 600 定义）。对日线回测：**每手 = 1 股**，所有证券统一，不随时间变化。
- **round lot 定义修订**：SEC 2024-09-18 通过的 Reg NMS 修订（Release 34-101070）按股价分档重新定义 round lot（股价 ≤$250 为 100 股，$250.01–$1,000 为 40 股，$1,000.01–$10,000 为 10 股，>$10,000 为 1 股）——这部分数字来自我对该规则的记忆，搜索摘要只确认了修订存在，**未核实**数字与实施日。它只影响报价发布，不影响可交易的最小数量。出处：https://www.sec.gov/files/34-101070-fact-sheet.pdf
- **碎股（fractional shares）**：部分券商（IBKR、moomoo、Robinhood 等）提供小于 1 股的碎股交易，属券商内部撮合/记账服务，不是交易所规则。回测建议按整股。
- **历史数据**：不需要每手股数的历史表。

---

## 3. 价格限制与停牌

### 3.1 没有涨跌停
美股没有按前收盘价计算的日涨跌幅限制。一只股票一天内可以涨跌几十个百分点甚至更多。

### 3.2 个股层面：Limit Up-Limit Down（LULD）

- **适用**：所有 NMS 股票（交易所上市股票与 ETP），常规交易时段 9:30–16:00 ET。
- **实施**：Phase I 于 **2013-04-08** 开始（Tier 1 证券），Phase II 于 2014-02-24 覆盖全部 NMS 股票（日期来自 FINRA Regulatory Notice 13-12 的背景与 LULD 计划，**Phase II 日期未逐字核对**）。之前（2010-06 起）是「单只股票熔断」试点（5 分钟内涨跌 10% 即暂停 5 分钟），**未核实**具体日期。
- **价格带**：以过去 5 分钟成交均价为参考价，
  - Tier 1（S&P 500、Russell 1000 成分及部分 ETP）：参考价 > $3 为 ±5%；
  - Tier 2（其他 NMS 股票）：参考价 > $3 为 ±10%；
  - 参考价 $0.75–$3：±20%；< $0.75：取 $0.15 与 75% 中较小者。
  - 收盘前 15:35–16:00 Tier 1 与 Tier 2 中参考价 ≤ $3 的证券价格带**加倍**；Amendment 18（SEC 2019-04-11 永久批准，Cboe 称 2020-02-24 实施）取消了开盘 9:30–9:45 的加倍和 Tier 2（> $3）收盘段的加倍。
  - 出处：LULD Plan https://www.nyse.com/publicdocs/LULD_Plan.pdf ；Cboe LULD Amendment 18 说明 https://www.cboe.com/document/tech-spec/content/technical-specifications/cboe-limit-updown-faq/luld-amendment-18-update ；LULD 2025 年报 https://cdn.luldplan.com/reports/LULD-2025-Annual-Report.pdf
- **机制**：价格触及带边沿进入「Limit State」，15 秒内不能恢复即暂停交易 5 分钟（Trading Pause），然后由上市交易所重新集合竞价开盘。15 秒/5 分钟这两个数字本次搜索**未从原文核实**（来自我对 LULD 计划的了解）。
- **对日线回测的影响**：LULD 只是盘中暂停 5 分钟后再开，**不会**像涨跌停那样让股票整天无法成交，也不会封板。日线的 OHLC 已经包含暂停后的价格。对「能否按收盘价成交」基本无影响；但若当日最后 25 分钟进入 LULD 暂停，收盘价可能由重开竞价或延迟收盘产生（收盘竞价可能被推迟），按收盘价成交的假设在极端日不够可靠。

### 3.3 全市场熔断（Market-Wide Circuit Breakers）

- **现行（2013-04-08 起，替代旧的道琼斯 10%/20%/30% 规则）**：以标普 500 指数相对**前一交易日收盘**的跌幅为准：
  - Level 1：跌 7%；Level 2：跌 13%——在 9:30 至 15:25 之前触发，全市场暂停 15 分钟；15:25 及之后触发不暂停；每日各最多一次。
  - Level 3：跌 20%——任何时间触发，全市场当日余下时间停止交易。
  - 出处：NYSE Rule 7.12（永久化申请见 SEC 文件 https://sec.gov/files/rules/sro/nyse/2021/34-92428.pdf ）；Nasdaq 说明 https://www.nasdaq.com/articles/market-wide-circuit-breakers%3A-what-you-need-to-know-2020-03-09 。永久化生效日**未核实**。
- **历史触发**：2020-03-09、03-12、03-16、03-18 四次触发 Level 1（各暂停 15 分钟），2010 年以来从未触发 Level 3（媒体报道，旁证）。
- **对日线回测的影响**：Level 1/2 只暂停 15 分钟，当日仍有收盘价；若 Level 3 触发，当日的「收盘价」是停止交易时的价格，此后无法成交。2010 年以来未发生，可在档案中声明不模拟。

### 3.4 停牌（trading halt）的常见情形

- 交易所对重大消息待发布的停牌（news pending，代码 T1/T2 等）、监管停牌（SEC 可依 Exchange Act §12(k) 暂停最长 10 个交易日）、未满足上市标准、LULD 暂停、IPO/SPAC 并购前后等。多数消息停牌持续数十分钟到数小时，少数持续多日（如被 SEC 暂停交易或面临退市的公司）。
- 停牌数据：Nasdaq Trader Trade Halts（https://www.nasdaqtrader.com/trader.aspx?id=TradeHalts ）当前与近期停牌列表，历史需自行积累；SEC Trading Suspensions 列表（https://www.sec.gov/litigation/suspensions ）。本次未打开核对。
- 日线回测：停牌整天的证券当天没有日线记录（或数据商以前收价填充、成交量 0），应视为不可成交。

### 3.5 第 3 节未核实或有冲突的条目

1. LULD Phase II 日期 2014-02-24、15 秒 Limit State、5 分钟暂停：未从原文核实。
2. Amendment 18 取消加倍的范围（开盘段全部取消；收盘段只对 Tier 2 且参考价 > $3 取消）：已从 Cboe 说明页原文确认，但未用 LULD Plan 正文二次核对；实施日 Cboe 写作「expected to be effective February 24, 2020」。
3. 2010–2013 单只股票熔断试点的起止日：未核实。
4. MWCB 永久化日期：未核实（规则内容本身多源一致）。

---

## 4. 交收与卖出限制

### 4.1 当日买入能否当日卖出
**可以**。美国没有 T+1 卖出锁定。无论现金账户还是保证金账户，当日买入的股票当日即可卖出。限制只来自账户类型（见 4.3、4.4）。

### 4.2 资金与证券交收周期（SEC Rule 15c6-1）

| 期间 | 周期 | 依据 |
|---|---|---|
| 1995-06-07 起至 2017-09-04 | T+3 | Rule 15c6-1（1993 年通过，T+5 → T+3）；起始日来自我对规则的了解，**未核实** |
| 2017-09-05 起 | T+2 | SEC 新闻稿 2017-68（2017-03-22）：「compliance date ... Sept. 5, 2017」 https://www.sec.gov/news/press-release/2017-68-0 |
| **2024-05-28 起（现行）** | **T+1** | SEC 2023-02-15 通过的 Rule 15c6-1(a) 修订，合规日 2024-05-28（周二，紧随阵亡将士纪念日）。旁证：A&O Shearman https://www.aoshearman.com/en/insights/t1-settlement-coming-may-28-2024 ；Davis Polk https://www.davispolk.com/insights/client-update/sec-adopts-t-1-settlement-effective-may-2024 （SEC 原 Release 34-96930 未打开） |

- 对日线回测的影响：交收周期决定**卖出所得资金何时可再用于现金账户买入**，以及除息日与登记日的关系（T+1 下除息日 = 登记日，见第 6 节）。

### 4.3 现金账户：Reg T 的 free-riding 规则

- 现金账户可以当日买当日卖，但如果用**尚未交收的卖出款**买入、又在付款前卖出该股票，属于「free-riding」，券商须对账户冻结 90 天（Regulation T, 12 CFR 220.8(c)）。**未逐字核对**原文。
- 回测含义：现金账户策略若每天全仓换股，在 T+1 下卖出款次日才交收，严格来说当天的卖出款不能立即用于再卖出——多数日线引擎忽略这一点。建议档案参数：`cash_account_settlement_days`，默认不模拟，在文档中声明。

### 4.4 日内交易者规则（Pattern Day Trader，PDT）——**已废除并替换**

- **旧规则（2001 年起至 2026-06-03）**：FINRA Rule 4210(f)(8)(B)：5 个交易日内在保证金账户中做 4 次及以上日内回转（且占该账户交易 6% 以上）即被认定为 PDT，账户权益须不少于 **$25,000**，否则不得日内交易；PDT 享有「日内买入力」4 倍维持保证金超额。
- **现行**：SEC 于 **2026-04-14** 加速批准 SR-FINRA-2025-017（Release 34-105226），修订 Rule 4210，以新的「intraday margin」标准（新 (d)(2) 款）**整体取代**日内交易保证金规定，取消 PDT 认定、$25,000 最低权益与「day trading buying power」。**生效日 2026-06-04**；券商可分阶段实施，最迟 **2027-10-20**。
  - 出处：FINRA Regulatory Notice 26-10 https://www.finra.org/rules-guidance/notices/26-10 ；FINRA Rule 4210 现行文本（已无 PDT 字样，含 (d)(2) Intraday Margin）https://www.finra.org/rules-guidance/rulebooks/finra-rules/4210 ；WilmerHale 客户通讯 https://www.wilmerhale.com/en/insights/client-alerts/20260423-sec-approves-amendments-to-finra-rule-4210-replacing-day-trading-margin-requirements-with-a-modernized-intraday-margin-standard （旁证）。
  - 券商实施时间不一：E*TRADE 称 2026-06-09 实施（旁证）。
- **回测含义**：以日期为界：2026-06-03 及以前，账户权益 < $25,000 的保证金账户在滚动 5 日内最多 3 次日内回转；2026-06-04 起法规层面无此限制，但个别券商可能到 2027-10-20 前仍沿用旧做法。现金账户从来不受 PDT 约束（只受 free-riding 约束）。日线引擎一般不产生日内回转（同一天买又卖），所以影响主要是「当日买入当日卖出」类策略。
- **非 FINRA 会员的境外券商**（如富途香港）：PDT 是 FINRA 规则，只约束美国 FINRA 会员券商；通过香港券商经美国清算的账户，券商通常沿用同样规则（**未核实**，以券商为准）。

### 4.5 第 4 节未核实或有冲突的条目

1. T+3 起始日 1995-06-07：未核实（2010 年以前，影响小）。
2. T+1 的 SEC 原始 Release 未打开，日期由两家律所多源一致。
3. Reg T 220.8 free-riding 90 天：未逐字核对。
4. 境外券商是否执行 PDT：未核实。

---

## 5. 做空与融资

### 5.1 谁可以做空、标的从哪里来
- 任何开通保证金账户的投资者（含非居民）均可做空，前提是券商能借到股票（Reg SHO Rule 203(b) 的「locate」要求：卖空前券商须有合理理由相信可借到并按期交付）。
- 没有官方的「可做空名单」。可借标的由各券商的借券池决定，常用公开来源：IBKR 公开的可借券数量与费率文件（旁证，未打开）。官方相关名单：
  - Reg SHO Threshold Securities List（交付失败持续的证券，强制平仓规则更严）：各上市交易所每日公布，如 Nasdaq https://www.nasdaqtrader.com/Trader.aspx?id=RegSHO
  - FINRA 每月两次的空头持仓（short interest）数据；SEC 交付失败（fails-to-deliver）数据。
- **Reg SHO Rule 204**：卖空交付失败须在 T+1（交收日后一个交易日）开盘前平仓（**未核实**）。

### 5.2 提价规则：Reg SHO Rule 201（替代性 uptick 规则）
- 生效 2010-05-10，**合规日 2011-02-28**（SEC Rule 201 FAQ https://www.sec.gov/files/divisions/marketreg/rule201faq.htm ；Nasdaq Reg SHO FAQ）。2007-07-06 至 2011-02-27 美国没有任何卖空价格测试（旧 uptick rule 10a-1 于 2007 年取消，**日期未核实**）。
- **触发**：个股在常规时段较**前一交易日收盘价**下跌 ≥ 10%。
- **效果**：当日余下时间及**下一交易日全天**，卖空申报价必须**高于**当前全国最优买价（NBB），即不能打买盘；可再次触发、次数不限。
- **回测含义**：日线引擎无法精确表达。建议：若某日相对前收的最低价跌幅 ≥ 10%，则当日（触发后）和次日的卖空开仓不允许按「打到买价」的价格成交——实际上日线引擎可以简化为「不允许在这两天以开盘价/收盘价开空」，或在文档中声明不模拟。

### 5.3 保证金
- **初始保证金（Reg T，12 CFR 220.12）**：买入保证金证券和卖空均为 **50%**，自 1974-01-03 起未变（日期来自美联储公开资料的记忆，**未核实**；50% 这一数值是公认事实）。卖空时还须存入 100% 卖出款作为担保，即 150% 规定（Reg T 220.12(c)）。
- **最低权益**：保证金账户至少 $2,000（FINRA Rule 4210(b)(4)，已核对现行文本）。
- **维持保证金（FINRA Rule 4210(c)，已核对现行文本）**：
  - 多头：市值的 25%；
  - 空头：股价 ≥ $5 时为 max($5/股, 30% 市值)；股价 < $5 时为 max($2.50/股, 100% 市值)。
  - 券商「house requirement」通常更严（30%–100%，按个股波动调整）。
- 历史：2010 年至今上述法定数值未变（**未逐年核实**）；变化的只有日内保证金（见 4.4）。

### 5.4 借券费用
- 无法定费率。一般流动性好的「易借券」（general collateral）年化借券费约 0.25%–0.5%，难借券可达数十个百分点甚至 100%+（行业常识，**未核实**）。券商按日计提，余额为正时还有空头回扣利息（short credit interest，IBKR 等对大额账户提供）。
- 数据来源：IBKR 公开的 shortable 文件（旁证）；付费：S&P Global Securities Finance（原 Markit）、FIS Astec。Tushare 无美股借券费数据（本次未见相关接口）。

### 5.5 第 5 节未核实或有冲突的条目
1. Reg T 50% 的起始日 1974-01-03。
2. 旧 uptick rule 取消日 2007-07-06。
3. Rule 204 平仓时限。
4. 借券费一般水平。

---

## 6. 分红与公司行为

### 6.1 现金分红预扣税

| 投资者身份 | 美国来源股息预扣率 | 依据 |
|---|---|---|
| 非居民个人，**无**协定或未提交 W-8BEN | 30% | IRC §871(a)、§1441；IRS 协定表 1 首段 |
| **中国内地税务居民个人**，提交 W-8BEN 并声明中美协定 | **10%** | 《中美关于对所得避免双重征税的协定》（1984）第 9 条第 2 款；IRS Tax Treaty Table 1（Rev. May 2023）China 行：股息 10%，Article 9(2)，已从 PDF 原文核对 https://www.irs.gov/pub/irs-lbi/tax-treaty-table-1.pdf |
| 香港税务居民（无美港税收协定） | 30% | 美国与香港之间无所得税协定（常识，IRS 表中无香港行） |
| 美国税务居民个人 | 不预扣（提交 W-9）；股息计入应税所得，「qualified dividends」按 0/15/20% 资本利得税率，普通股息按普通税率；未提供 TIN 时可能被按 24% backup withholding | IRC §1(h)(11)、§3406（常识，**未逐条打开**） |

- 关键点：协定税率看的是**税务居民身份**，不是券商所在地。内地居民通过香港券商开户，只要 W-8BEN 填中国居民并主张协定，股息预扣 10%。实际执行取决于券商是否支持协定税率（多数支持，**未逐家核实**）。
- **REIT、RIC（基金）分红**：协定表中 China 的 RIC 一栏同为 10%；REIT 的普通分红在部分条件下适用 10%（脚注 w），资本利得分配按 21%/30% 预扣（FIRPTA，**未核实**）。
- **ADR 的股息**：ADR 的标的是外国公司股票，股息是**外国来源**，美国不预扣；由**发行人所在国**预扣（按该国对存托银行或实际受益人的税率），再由存托银行发放。例：中国内地公司（含 VIE 结构下的开曼公司若被认定为居民企业）向非居民企业支付股息适用 10% 企业所得税预扣；不同 ADR 差别大，需逐只查 20-F「Taxation」一节（**未核实**，旁证：PwC 税务摘要 https://taxsummaries.pwc.com/peoples-republic-of-china/corporate/withholding-taxes ）。
- **中国内地个人在国内的补税**：境外股息按 20% 计入个人所得税「利息、股息、红利所得」，可抵免已在境外缴纳的税款（个人所得税法及财政部 税务总局公告 2020 年第 3 号，**未打开原文，未核实**）。这是投资者层面的事，回测一般只模拟预扣。
- **资本利得**：非居民在美国一般不征（见 1.3）。

### 6.2 ADR 存托费
- 存托银行按**每份 ADS**收取存托服务费（custody / depositary service fee），常见约定为每年不超过 $0.05/ADS，可从现金分红中扣除，或经 DTC 由券商向持有人另行收取；具体以每只 ADR 的存托协议（Form F-6 附件、20-F 附件 2）为准。
  - 例：知乎 20-F 附件存托协议「aggregate fee of US$0.05 or less per ADS per calendar year」https://www.sec.gov/Archives/edgar/data/1835724/000110465923051969/zh-20221231xex2d5.htm
  - ING 20-F：分发现金股息每 ADS 不超过 $0.05。
  - Fidelity 说明：每次分红通常 $0.01–$0.05/ADR（旁证）https://fidelity.com/learning-center/investment-products/stocks/understanding-american-depositary-receipts
- **回测含义**：按「每股（份）年费 X 美元，在登记日扣除」建模；没有统一费率，建议默认 $0.03/ADS/年，可按证券覆盖。ADR 与普通股的比例（ADS ratio）变化会表现为拆股/合股。

### 6.3 除息日约定
- 现金股息：**T+1 时代（2024-05-28 起）除息日 = 登记日**（同一个交易日）；T+2 时代（2017-09-05 至 2024-05-27）除息日 = 登记日前 1 个交易日；T+3 时代为登记日前 2 个交易日。依据 FINRA Rule 11140 / 交易所规则（**未打开核实**，但与交收周期一一对应，属标准约定）。
- **大额分红例外**：股息 ≥ 股价 25% 时，除息日改为**支付日后第一个交易日**（FINRA Rule 11140(b)(2)，**未核实**）。
- 股票股利/拆股：除权日一般为**分配日（payable date）后一个交易日**（「due bill」机制），而不是登记日前后——这是美股与 A 股最大的差别之一（FINRA Rule 11140(b)(1)，**未核实**）。

### 6.4 公司行为的常见做法及在日线数据里的体现

| 公司行为 | 美股常见做法 | 日线未复权数据的样子 | 处理建议 |
|---|---|---|---|
| 拆股（forward split，如 4:1） | 常见，用于降低股价；无送股/转增之分 | 除权日价格骤降为 1/4，成交量放大 | 必须用拆股因子调整价格和持仓股数 |
| 合股（reverse split，如 1:10） | 低价股为维持 $1 最低价上市要求常用；零碎股多以现金结算 | 除权日价格骤升 10 倍 | 同上；零碎股现金化可忽略 |
| 股票股利（如 5% stock dividend） | 较少 | 价格小幅下降 | 按拆股处理 |
| 分拆（spin-off） | 常见（如 GE → GE HealthCare 2023、GE Vernova 2024） | 母公司在除权日价格下跌（减去被分拆部分价值），子公司作为新代码开始交易；常有「when-issued」交易期 | 需要分拆比例和价值分配；多数数据源把分拆调整做进复权因子（价值比例法），但持仓侧要给持有者发放子公司股票，否则收益失真 |
| 配股/供股（rights offering） | 极少（美国公司以增发、ATM 增发为主） | — | 可不模拟 |
| 特别股息 | 有 | 除息日价格下跌 | 和现金分红一样处理；注意 25% 规则 |
| 收购（现金/换股） | 常见 | 交易最后一天后无数据 | 需按对价把持仓变为现金或收购方股票，否则会被当作「停牌到永远」 |
| 退市转场外 | 被交易所摘牌后通常转到 OTC Markets（Pink/Expert Market）继续交易，股东仍持有股票 | 交易所日线在最后交易日后中断；场外价格通常大幅低于最后交易所价格 | 用最后交易所收盘价平仓会**高估**收益；CRSP 提供 delisting return 修正 |
| 破产 | 普通股常最终价值归零 | 同上 | 同上 |
| 代码变更、重组 | 常见 | 同一公司换代码；同一代码可能被另一家公司复用 | 必须用永久证券标识（CRSP PERMNO、FIGI 等），不能只用代码 |

- 退市流程依据：发行人或交易所向 SEC 提交 Form 25，10 日后生效（Exchange Act Rule 12d2-2，**未核实**）。Nasdaq 最低出价规则：连续 30 个交易日收盘价 < $1 即收到不合规通知（Nasdaq Rule 5550(a)(2)，**未核实**）。

### 6.5 复权因子：哪些包含现金分红、哪些只含拆股

| 数据源 | 价格/因子口径 | 依据 |
|---|---|---|
| Yahoo Finance | 「Adj Close」= 按拆股**和**分红调整；分红因子 = 1 − 股息/前一日收盘价。「Close」列在 Yahoo 实际是**已按拆股调整**的收盘价（我对 Yahoo 的使用经验；帮助页示例把 Close 当作未调整，**冲突/未核实**） | Yahoo 帮助页 https://help.yahoo.com/kb/SLN28256.html |
| CRSP | `CFACPR`（累计价格调整因子）**只含**拆股、股票股利、分拆等改变每股价值的分配，**普通现金分红的 FACPR = 0，不进入因子**；分红在 `RET`（含息收益率）中体现，`RETX` 为不含息收益率；另有 delisting return | 密歇根大学镜像的 CRSP 文档 https://leiq.bus.umich.edu/docs/crsp_factor_adjustment.pdf ；CRSP 计算方法指南 https://crsp.org/wp-content/uploads/guides/CRSP_Calculations_and_Index_Methodologies.pdf （**未从 CRSP 官方 CFACPR 定义原文核实**） |
| Tushare `us_adjfactor` / `us_daily_adj` | 前复权（close × adj_factor = 前复权收盘价）；文档**没有说明**是否含现金分红 | https://tushare.pro/document/2?doc_id=402 ；https://tushare.pro/document/2?doc_id=338 |
| Polygon.io（现更名 Massive）、Nasdaq Data Link Sharadar SEP | Polygon 的 `adjusted=true` 只按拆股调整，分红另有 dividends 接口；Sharadar SEP 的 `closeadj` 含分红与拆股、`close` 只含拆股——均为我对文档的记忆，**未核实** | — |

- **建议**：引擎自己用「未复权价格 + 拆股/合股事件 + 现金分红事件（含预扣税）」重建总收益，不要直接用含分红的复权因子（会把税前分红无损再投资，系统性高估非居民收益约 股息率×预扣率）。

### 6.6 第 6 节未核实或有冲突的条目
1. 美国居民的税率细节、backup withholding 24%。
2. ADR 外国预扣：逐只不同，未核实。
3. 中国内地个人境外股息补税规则：未打开原文。
4. FINRA Rule 11140 除息日约定与 25% 例外：未核实。
5. Yahoo「Close」是否拆股调整：冲突。
6. Tushare 美股复权因子是否含现金分红：文档未说明。
7. Polygon、Sharadar 口径：未核实。
8. Form 25、Nasdaq $1 规则：未核实。

---

## 7. 交易日历与时段

- **常规时段**：9:30–16:00 ET（NYSE、Nasdaq 相同）。盘前/盘后：NYSE Arca 4:00–9:30、16:00–20:00；Nasdaq 类似（4:00–9:30，16:00–20:00）。日线 OHLC 一般只取常规时段（以数据源为准）；盘前盘后成交是否计入成交量各数据源不同。出处：NYSE Hours & Calendars https://www.nyse.com/markets/hours-calendars
- **半日市**：13:00 ET 收盘。通常为感恩节后的星期五、圣诞前夕（12-24，逢工作日）、独立日前一天（7-03，逢工作日且非假期观察日）。2026 年：11-27、12-24；2027 年：11-26（NYSE 页面）。
- **2026 年休市日**（NYSE）：1-01、1-19、2-16、4-03（耶稣受难日）、5-25、6-19、7-03（独立日观察日）、9-07、11-26、12-25。
- **2027 年休市日**：1-01、1-18、2-15、3-26、5-31、6-18、7-05、9-06、11-25、12-24。
- **规则变化**：Juneteenth（6-19）自 2022 年起成为休市日。
- **临时休市（2010 年以来）**：2012-10-29、10-30（飓风桑迪）；2018-12-05（老布什国葬日）；2025-01-09（卡特国葬日）。来自记忆与媒体报道，**未从交易所原文核实**。美国没有类似香港「台风/黑雨」的成文天气休市规则，临时休市由交易所和 SEC 临时决定。
- **日历数据**：NYSE 官网假期页（仅当年和后两年）；Tushare `us_tradecal`（doc_id=253，本次未打开）；开源 `exchange_calendars`/`pandas_market_calendars`（XNYS，含历史临时休市，未核实许可证）。
- **24 小时交易**：多家交易所（如 Nasdaq、NYSE Arca、24X）推进 23/5 或 24/5 交易，LULD 已有夜盘修订（https://cdn.luldplan.com/plan-amendments/LULD_Overnight_Amendment_Final.pdf ）；**截至 2026-11-13 的实施状态未核实**。对日线回测：若数据源把夜盘成交并入日线，开盘价/最高最低价含义会变，需确认。

---

## 8. 最小报价单位

- **现行**：Reg NMS Rule 612：股价 ≥ $1.00 的 NMS 股票，报价最小单位 $0.01；< $1.00 为 $0.0001。成交价可以出现更小单位（如中间价撮合、暗池），这是报价规则不是成交规则。
- **最近一次修订（已通过、未生效）**：SEC 2024-09-18 通过（Release 34-101070）：对「tick-constrained」股票（时间加权平均报价价差 ≤ $0.015）改为 $0.005；同时把 Rule 610(c) 访问费上限从 $0.003 降至 $0.001（股价 ≥ $1）。原合规日 2025-11-03；SEC 于 2025-10-31 发布临时豁免令推迟到 2026-11-02；据 Virtu 2026 年三季度 10-Q，合规日**再次推迟到 2027 年 11 月**（单源，SEC 原始命令未找到，**未核实**）。
  - 出处：SEC 新闻稿 2024-137 https://sec.gov/newsroom/press-releases/2024-137 ；SEC 情况说明 https://www.sec.gov/files/34-101070-fact-sheet.pdf ；SIFMA 延期请求信 https://www.sifma.org/advocacy/letters/supplemental-request-for-immediate-extension-of-tick-size-and-access-fee-compliance-dates ；Virtu 10-Q https://www.sec.gov/Archives/edgar/data/0001592386/000159238626000034/virt-20260930.htm
- 2016-10 至 2018-10 曾有 Tick Size Pilot（部分小盘股 $0.05 报价单位），**未核实**日期。
- 日线回测：tick 只影响盘口，可不模拟；成交价按数据源价格即可。

---

## 9. 数据从哪里来

### 9.1 Tushare（只看公开文档，未登录）

| 接口 | 内容 | 权限 | 覆盖起始 | 已退市 | 文档 |
|---|---|---|---|---|---|
| `us_basic` | 证券列表：代码、中英文名、类别（ADR/GDR/EQ/PF）、`list_status`（L/D/G/P/UN）、上市日、退市日、交易所 | 120 积分试用，5000 积分正式 | — | 有 D（退市）状态与 `delist_date` 字段，应包含（**未实测**）；文档警告代码复用会产生重复行 | https://tushare.pro/document/2?doc_id=252 |
| `us_tradecal` | 美股交易日历 | 未查 | 未查 | — | doc_id=253 |
| `us_daily` | 未复权日线：OHLC、前收、涨跌幅、成交量、成交额、VWAP、换手率、总市值、PE、PB | 单独开通「美股历史日线」权限，积分要求未写 | 文档称「全部股票的全历史数据」，示例为 2019 年，起始年**未说明** | 未说明 | https://tushare.pro/document/2?doc_id=254 |
| `us_daily_adj` | 前复权日线，含 adj_factor、流通/总股本、市值 | 同上 | 2006 年起 | 未说明 | https://tushare.pro/document/2?doc_id=338 |
| `us_adjfactor` | 累计复权因子 `cum_adjfactor`，每日美股收盘后滚动刷新 | 同上 | 2006 年起 | 未说明 | https://tushare.pro/document/2?doc_id=402 |
| 其他 | ETF 列表与日线、三大报表、财务指标 | — | — | — | doc_id 504、506、393–396 |

- 缺口：Tushare 没有美股的分红明细、拆股事件、借券费、指数成分接口（侧栏未见）。复权因子是否含现金分红文档未说明，**接入前必须用已知拆股与分红样本实测**。

### 9.2 官方免费来源
- SEC EDGAR：8-K（公司行为公告）、Form 25（退市）、20-F/F-6（ADR 存托协议与费用）、10-K 股本数据——全部免费，可再分发（美国政府作品）。
- Nasdaq Trader：证券代码目录（nasdaqlisted.txt / otherlisted.txt，仅当前）、每日停牌、Reg SHO 阈值名单、公司行为日报（Daily List，部分付费）。
- NYSE：假期日历、公司行为通知（部分）。
- FINRA：空头持仓（半月）、OTC 交易数据；SEC：交付失败数据。
- 美国交易所不免费提供完整历史日线和复权因子。

### 9.3 第三方来源

| 来源 | 内容 | 许可（未逐一核实） |
|---|---|---|
| CRSP（经 WRDS） | 1925 年至今，含已退市、PERMNO、分红/拆股/分拆事件、delisting return；学术金标准 | 机构订阅，仅限订阅机构学术使用，不可再分发 |
| Yahoo Finance（经 yfinance 等） | 日线、Adj Close、分红与拆股列表；不含已退市股票 | Yahoo 条款只允许个人非商业使用，不可再分发；非官方 API，可随时变更 |
| Polygon.io / Massive、Nasdaq Data Link Sharadar、Norgate Data | 含已退市股票、拆股与分红事件 | 付费；个人研究可用，再分发受限 |
| Stooq | 免费日线（含部分已退市） | 条款未查 |

- 指数成分：标普 500 等历史成分官方不免费（S&P DJI 付费）；Wikipedia 有变更表，质量不保证（旁证）。

---

## 10. 让日线回测明显失真的坑（8 条）

1. **幸存者偏差**：Yahoo 等免费源不含已退市股票；美股每年有数百只退市（并购、破产、低价被摘牌）。只用现存股票回测会大幅高估收益，小盘、低价策略尤甚。
2. **退市后价值处理**：最后一个交易所收盘价之后，股票可能在 OTC 以低得多的价格交易或归零；用最后收盘价自动平仓会高估收益。需要 delisting return（CRSP）或保守假设。
3. **用含分红的复权价算非居民收益**：Adj Close 默认把税前股息全额再投资；中国内地居民实际只拿到 90%（ADR 还要扣存托费和外国预扣）。高股息组合每年偏差约 股息率 × 10%（或 × 30%，若未提交 W-8BEN）。
4. **除权日约定不同于 A 股**：拆股、股票股利在**支付日后**除权；T+1 之前现金股息除息日在登记日前。若引擎按 A 股「登记日后一日除权」处理会错一两天，并在除权日前后错配价格与持股数。
5. **代码复用与变更**：同一代码在不同年份可能是不同公司（例如退市后代码被新公司使用），按代码拼接会把两家公司的价格接成一条序列，产生假的暴涨暴跌。必须用永久 ID。
6. **低价股与反向拆股**：大量低价股频繁 1:10、1:50 合股；若复权因子缺失一次，价格序列出现 10 倍跳变，动量/反转策略会捕捉到虚假收益。Tushare 美股因子质量需实测。
7. **监管费随时间剧烈变化**：Section 31 在 2025-05-14 至 2026-04-03 为零、2026-04-04 起 $20.60/百万；TAF 在 2026 年 10–12 月为零。用常数费率不会造成大偏差（单边万分之 0.2 左右），但高频换手的小额交易受 TAF 每笔上限与券商每单最低佣金 $1 影响大：小单的成本率可以是大单的几十倍。
8. **收盘价成交假设与收盘竞价**：美股收盘价来自交易所收盘竞价（NYSE closing auction / Nasdaq Closing Cross），流动性好时按收盘价成交合理；但小盘股收盘价可能是一笔很小的成交、或来自盘后成交被个别数据源计入，开盘价同理（盘前成交）。对小盘股按 OHLC 成交会低估冲击成本。另外 Rule 201 在暴跌日次日限制卖空，空头策略的开仓假设会失真。

---

## 11. 给实现者的建议

**档案需要的参数（按日期分段）：**

1. 费用表（单位：美元）：
   - `sec_fee`：卖出方向、按成交金额比例、无上下限；分段值见 1.1 表（2010-01-15 起逐段）。
   - `finra_taf`：卖出方向、按股数、每笔上限；分段值见 1.2 表，**2026-10-01 至 2026-12-31 为 0**。若成交价低于费率本身则按成交额收取——日线引擎可忽略。
   - `commission`：双边、按股数 $0.005，每笔下限 $1、上限成交额 1%（可配置）。
   - `cat_fee`：可忽略或双边 $0.000001/股（2026-05 起），默认不建模。
   - 取整：建议按笔向上取整到 $0.01（声明为券商惯例）。
2. 交易单位：所有证券每手 1 股，恒定。
3. 价格限制：无涨跌停。LULD 与全市场熔断声明为**不模拟**。
4. 卖出锁定：无（当日买入可当日卖出）。
5. 交收：`settlement_days` = 3（≤ 2017-09-04）、2（2017-09-05 至 2024-05-27）、1（≥ 2024-05-28）；用于现金账户资金可用性和除息日推算。
6. 日内交易限制：`pdt_min_equity` = $25,000，滚动 5 日 4 次（≤ 2026-06-03）；≥ 2026-06-04 取消（券商最迟 2027-10-20 完成）。日线引擎若不做日内回转可不实现，但应在档案里写明。
7. 做空与保证金：初始 50%（Reg T，全期不变）；维持多头 25%，空头 max(30%, $5/股)（股价 ≥ $5）或 max(100%, $2.50/股)（< $5）；最低权益 $2,000。借券费作为可配置年化费率（默认 0.3%/年，声明为假设）。
8. 分红扣税：非居民无协定 30%；中国内地居民（W-8BEN）10%；美国居民 0%（预扣层面）。ADR 单独参数：外国预扣率（逐只）与存托费（默认 $0.03/ADS/年）。
9. 日历：XNYS 交易日历（含 2012-10-29/30、2018-12-05、2025-01-09 临时休市）；半日市不影响日线。

**日线引擎表达不了、应在文档中声明不模拟的：**
- LULD 盘中暂停、全市场熔断（Level 3 2010 年以来未发生）。
- Rule 201 卖空提价规则（只能粗略近似为「触发后当日及次日禁止开空」）。
- 现金账户 free-riding 与未交收资金的使用限制。
- 盘前盘后与夜盘交易、收盘竞价的参与度、odd lot 报价。
- 借券可得性（locate）与召回（recall）、Reg SHO Rule 204 强制平仓。
- 公司行为中需要发放新证券的情形（分拆、换股收购）若数据源只给复权因子而无事件明细，只能近似；建议引擎支持「分拆/换股事件」后再上线美股档案。
- 多币种：一次回测只用美元；内地投资者的人民币换汇成本与外汇额度不模拟。
- PTP 卖出总价款 10% 预扣（§1446(f)）：建议直接从股票池中排除 PTP。

