> 说明：这是 2026-11-12 选型时联网调研的原始笔记（数据源部分），由调研子代理撰写，每条标了【读到】【推断】【未核实】并附来源。
> 结论和本机实测结果见 `docs/SELECTION.md`；两者不一致时以 SELECTION 为准。笔记里提到的临时目录没有保留。

# A 股量化数据源选型调研（核实时间：2026-11-12）

需求编号：① 含退市股全历史日线 + 复权因子；② 每日 ST / 停牌 / 涨跌停价；③ 财务数据带公告日（最好区分首次与更正）；④ 业绩预告 / 快报及公告日；⑤ 指数历史成分与权重（沪深300 / 中证500 / 中证1000 / 中证2000）；⑥ 每日总市值 / 流通市值、行业分类历史（申万或中信）；⑦ 分析师一致预期 / 研报盈利预测；⑧ 分红送转。

标注约定：【读到】= 我从该页面原文读到；【推断】= 由读到的事实推出，页面没有直说；【未核实】= 没查到或来源矛盾。

---

## 1. Tushare Pro

### 维护状态
- 【读到】官方「上新及变更」页最新条目日期 20261111（新上线若干宏观接口）、20261110（新上线「中信指数日度成分权重」）、20261105（「修改日线行情 ts_code 不支持多个取数」「指数成分和权重 index_code 改为必填」）。页面自述「SDK 目前已经稳定，不再经常做升级发布，我们每日维护数据内容」。来源：https://tushare.pro/document/1?doc_id=9
- 【推断】接口整体稳定，但参数规则仍在收紧（11-05 两条变更都是「以前能这么调、现在不行」），拉数脚本要容错。

### 费用（来源页无更新日期，页脚 © 2026；读取于 2026-11-12）
来源：https://tushare.pro/document/1?doc_id=290

| 积分 | 价格（元/年） | 每分钟频次 | 每天总量 | 能调什么 |
|---|---|---|---|---|
| 120 | 0 | 50 | 8000 次 | 「非复权日线数据，其他接口无法调取」 |
| 2000 以上 | 200 | 200 | 100000 次 / 每个 API | 按各接口文档的积分要求 |
| 5000 以上 | 500 | 500 | 常规数据无上限 | 同上，另可用 `*_vip` 按季度取全市场财务 |
| 10000 以上 | 1000 | 500 | 常规无上限，特色数据 300 次/分钟 | 特色数据：盈利预测、筹码、券商金股等 |
| 15000 以上 | 1500 | 500 | 特色数据无总量限制 | 特色数据专属权限 |

- 【读到】「日常购买积分比例…为 1:10，例如 200 元可获得 2000 积分」；「以上为个人价格，公司机构费用为个人的 10 倍」；不支持退款。分钟数据、港美股等是独立权限，另外收费（历史分钟单独 2000 元）。同页。
- 【读到】「正常获取和购买的积分及单独权限都是一年有效期，一年内不消耗不扣减」；注册 100 分 + 完善资料 20 分 = 120 分。来源：https://tushare.pro/document/1?doc_id=13
- 【读到】学生 / 高校老师可申请积分，但页面没写数额。同页。

### ①–⑧ 逐项
| 项 | 结论 | 接口 / 字段 / 积分门槛 | 来源 |
|---|---|---|---|
| ① 日线 | 有 | `daily`：未复权，字段含 `vol`、`amount`（千元）；「停牌期间不提供数据」；120 积分可试用，5000 积分 500 次/分钟、每次 6000 条 | https://tushare.pro/document/2?doc_id=27 |
| ① 复权因子 | 有 | `adj_factor`：2000 积分起；可按 `trade_date` 取全市场；「本接口由 Tushare 自行生产」，盘前 9:15–9:20 入库 | https://tushare.pro/document/2?doc_id=28 |
| ① 退市股 | 部分核实 | `stock_basic` 的 `list_status` 支持 `D 退市`、`P 暂停上市`，输出有 `delist_date`；2000 积分起。【推断】退市股的 `daily` / `adj_factor` 可按代码取到全历史——文档页没有明说，需要拿几只退市股实测 | https://tushare.pro/document/2?doc_id=25 |
| ② ST | 有 | `stock_st`：按交易日返回当日 ST 名单，3000 积分起，「本接口数据从 20000101 开始，太早历史无法补齐」，每天 9:20 更新，单次 1000 行。另有 `namechange`（曾用名，含 `start_date` / `end_date` / `ann_date` / `change_reason`，示例值有「ST」「撤销ST」）可自行重建；`st`（风险警示板变更明细，含 `pub_date` / `imp_date` / `st_type`）要 6000 积分 | https://tushare.pro/document/2?doc_id=397 ；https://tushare.pro/document/2?doc_id=100 ；https://tushare.pro/document/2?doc_id=423 |
| ② 停牌 | 有 | `suspend_d`：2000 积分起，历史从 2000-01-03，字段 `suspend_type`（S 停牌 / R 复牌）、`suspend_timing`（日内停牌时段），单次 5000 行，「更新时间不定期」 | https://tushare.pro/document/2?doc_id=214 |
| ② 涨跌停价 | 有（2007 年起） | `stk_limit`：2000 积分，历史起点 2007-01-04，字段 `up_limit` / `down_limit`，单次 5800 行，交易日 9 点左右更新 | https://tushare.pro/document/2?doc_id=183 |
| ③ 财务 + 公告日 | 有 | `income` / `balancesheet` / `cashflow`：2000 积分只能按单只股票取，5000 积分用 `income_vip` 按季度取全市场。公告日字段 `ann_date`（公告日期）与 `f_ann_date`（实际公告日期）；`update_flag`（更新标识）；`report_type` 区分 1 合并报表（最新）、4 调整合并报表、5 调整前合并报表（「数据发生变更，将原数据进行保留」）等 12 种 | https://tushare.pro/document/2?doc_id=33 |
| ④ 业绩预告 | 有 | `forecast`：2000 积分（单只），`forecast_vip` 5000 积分；字段 `ann_date`、`first_ann_date`（首次公告日）、`type`、净利润上下限 | https://tushare.pro/document/2?doc_id=45 |
| ④ 业绩快报 | 有 | `express`：2000 积分（单只），`express_vip` 5000 积分；字段 `ann_date` | https://tushare.pro/document/2?doc_id=46 |
| ⑤ 指数成分权重 | 部分 | `index_weight`：2000 积分；**只有月度数据**（「月度数据，交易日期取月末最后一个交易日」），来源「指数公司网站公开数据」。中证1000：【推断】有——开源项目 chenditc/investment_data 每天用 `index_weight` 取 `000852.SH`（见第 9 节）。中证2000 是否有、各指数从哪年开始：【未核实】（文档页没有列指数清单） | https://tushare.pro/document/2?doc_id=96 |
| ⑥ 市值 | 有 | `daily_basic`：2000 积分起，`total_mv` / `circ_mv`（万元）、`total_share` / `float_share` / `free_share`，可按日取全市场 | https://tushare.pro/document/2?doc_id=32 |
| ⑥ 行业历史 | 有 | 申万 `index_member_all`：2000 积分，字段 `in_date` / `out_date` / `is_new`，示例里含已退市股票；中信 `ci_index_member`：5000 积分，同样有 `in_date` / `out_date`；`ci_index_weight_d`（中信行业日度成分快照，2026-11-10 新上线）要 10000 积分 | https://tushare.pro/document/2?doc_id=335 ；https://tushare.pro/document/2?doc_id=373 ；https://tushare.pro/document/2?doc_id=518 |
| ⑦ 盈利预测 | 部分 | `report_rc`：逐篇研报的预测明细（机构、作者、预测报告期、`np`、`eps`、评级、目标价），2010 年起；「2000 积分可以试用，每天 10 次请求，正式权限需 8000 积分」。**不是现成的一致预期**，要自己聚合 | https://tushare.pro/document/2?doc_id=292 |
| ⑧ 分红送转 | 有 | `dividend`：2000 积分，2000-01-01 起，字段 `ann_date`、`div_proc`（预案 / 实施）、`record_date`、`ex_date`、`imp_ann_date`、`stk_div`、`cash_div_tax` | https://tushare.pro/document/2?doc_id=103 |

【推断】拿齐 ①–⑥、⑧ 的最低档：按文档门槛 3000 积分（按 1:10 约 300 元/年）即可全部调通（行业用申万），但 2000 档财务 / 预告 / 快报只能逐只股票取、每个 API 每天 10 万次。实际够用的是 **5000 积分（500 元/年）**：`*_vip` 按季度取全市场 + 中信行业。⑦ 要 8000 积分以上（按 1:10 约 800 元/年，页面只列了 10000 档 1000 元）。

### 已知问题
- 【读到】`index_weight` 只有月末快照，不是逐日权重。同上。
- 【读到】`daily` 停牌日没有行，需要自己用交易日历补齐。同上。
- 【读到】`stk_limit` 从 2007 年才有；`stock_st` 从 2000 年才有。
- 【读到】`amount` 单位是千元、市值单位是万元，和其他源混用时要换算。
- 其余社区反馈见文末「社区反馈补充」。

### 许可
- 【读到】服务协议：许可是「个人的、不可转让的、非商业用途的、可撤销的、有期限及非排他性的」，「仅可为非商业目的使用，并仅可用作个人查看使用」；不得「以营利、经营等非个人使用的目的」开通；账号不得出租、分享。来源：https://tushare.pro/document/1?doc_id=405
- 【读到】用户协议与服务协议都没有出现禁止本地存储的条款；官方文档自带「如何存入 MySQL 数据库」「数据存入 MongoDB」教程（https://tushare.pro/document/1?doc_id=231 标题）。
- 【推断】个人本地落盘、个人账户实盘参考属于「个人、非商业」范围；对外分发数据、代客理财或机构使用不在许可内（机构价是个人 10 倍）。协议没有逐字写「可用于个人实盘」。

- 【读到】Tushare SDK：PyPI 最新 1.4.29（上传 2026-03-25，https://pypi.org/project/tushare/ ）；GitHub waditu/tushare 最后提交 2020-03-04（https://github.com/waditu/tushare/commits ）。【推断】开源仓库已停更，Pro 是闭源数据服务，靠 SDK + HTTP 接口，不依赖 GitHub 仓库。

---

## 2. AKShare（akfamily/akshare）

### 维护状态
- 【读到】PyPI 最新 1.20.37（2026-11-09），前两版 1.20.36（11-08）、1.20.35（11-06）；GitHub 最后提交 2026-11-11。来源：https://pypi.org/project/akshare/ ；https://github.com/akfamily/akshare/commits
- 【读到】接口**不稳定是常态**：几乎每 1–3 天发一版修上游改版；1.20.36 标了「不兼容的改动」（`stock_zh_index_daily_tx` 的 amount 列含义改了）。来源：https://github.com/akfamily/akshare/blob/main/docs/changelog.md
- 许可：MIT（代码）。https://github.com/akfamily/akshare/blob/main/LICENSE

### 费用
- 【读到】免费，无账号、无积分。数据是从东方财富、新浪、交易所官网等公开网页抓取。来源：https://github.com/akfamily/akshare/blob/main/docs/introduction.md

### ①–⑧ 逐项（文档源文件：https://github.com/akfamily/akshare/tree/main/docs/data/stock ，下同）
| 项 | 结论 | 接口 / 字段 | 说明 |
|---|---|---|---|
| ① 日线 | 有 | `stock_zh_a_hist`（东财，含成交额，`adjust=''/qfq/hfq`）；`stock_zh_a_daily`（新浪，可取 `hfq-factor` / `qfq-factor` 复权因子） | 逐只股票取，没有「按日取全市场」的历史接口 |
| ① 退市股 | 部分 | 退市名单：`stock_info_sh_delist`、`stock_info_sz_delist`（含终止上市日期）。行情：changelog 1.20.14 写明「新浪仍提供其（已退市股票，如 sh600001）历史行情，但没有股本数据；现在照常返回行情，outstanding_share 与 turnover 为空」 | 【推断】退市股日线靠新浪这条路能取，但没有官方保证覆盖全部退市股；东财 `stock_zh_a_hist` 对退市股是否可用【未核实】 |
| ② ST | 没有历史 | `stock_zh_a_st_em` 只返回「当前交易日风险警示板」 | 历史 ST 只能用 `stock_info_change_name`（曾用名）自己推，无日期字段的可靠性【未核实】 |
| ② 停牌 | 部分 | `stock_tfp_em(date=…)`：东财停复牌信息，字段 停牌时间 / 停牌截止时间 / 停牌原因 | 历史能回溯多远【未核实】 |
| ② 涨跌停价 | 没有 | `stock_zt_pool_em` 是涨停股池，「该接口只能获取近期的数据」 | 没有历史每日涨跌停价，只能按规则自己算（要处理 ST、新股、板块差异） |
| ③ 财务 + 公告日 | 部分 | `stock_balance_sheet_by_report_em` 等三表（东财 F10，单只股票；另有 `*_delisted_em` 退市股版本）；`stock_yjbb_em` 业绩报表只有「最新公告日期」 | 三表输出 300+ 字段文档未逐一列出，公告日字段名【未核实】（东财原始字段通常含 NOTICE_DATE / UPDATE_DATE，文档页没写）；**没有「首次公告 vs 更正」的版本区分** |
| ④ 预告 / 快报 | 有（单一公告日） | `stock_yjyg_em`（2008-12-31 起）、`stock_yjkb_em`（2010-03-31 起），都有「公告日期」 | 按报告期取全市场 |
| ⑤ 指数成分权重 | 只有最新 | `index_stock_cons_csindex`、`index_stock_cons_weight_csindex`：中证官网当前成分与权重（带「日期」列，是最新一期） | **没有历史成分 / 历史权重**；文档里没有历史成分接口 |
| ⑥ 市值 | 有（逐只） | `stock_value_em`：单只股票全历史的 总市值 / 流通市值 / 总股本 / 流通股本 | 起始日期【未核实】；全市场要循环 5000+ 次 |
| ⑥ 行业历史 | 有 | `stock_industry_clf_hist_sw`：申万个股行业分类变动历史（symbol / start_date 计入日期 / industry_code），一次返回全部；`stock_industry_change_cninfo`：巨潮的行业归属变动（含「变更日期」「分类标准」） | 申万这份是官网下载中心文件，是难得的免费时点数据 |
| ⑦ 盈利预测 | 只有当前快照 | `stock_profit_forecast_em`（研报数、评级分布、各年预测 EPS）、`stock_profit_forecast_ths`、`stock_research_report_em` | 没有历史时点的一致预期序列 |
| ⑧ 分红送转 | 有 | `stock_fhps_em`（1990-12-31 起，按报告期取全市场，含 预案公告日 / 股权登记日 / 除权除息日 / 方案进度）、`stock_history_dividend_detail`、`stock_dividend_cninfo` | |

### 已知问题
- 【读到】东财行情接口限流：「东方财富 push2 行情服务器在短时间内收到较多请求时，会临时拒绝来自该 IP 的连接」「首次限流一般约 5 分钟后恢复…实测超过 15 分钟仍未恢复」；`stock_zh_a_hist` 属于受限接口（「没有 _em 后缀却请求 push2his」）。来源：https://github.com/akfamily/akshare/blob/main/docs/faq.md ；changelog 1.20.37、1.20.7
- 【读到】新浪接口 `stock_zh_a_daily`：「多次获取容易封禁 IP」。来源：docs/data/stock/stock_quote.md
- 【读到】`index_stock_cons`（新浪最新成分）「返回的数据有部分是重复会导致数据缺失」。来源：docs/data/index/index.md
- 【推断】全市场 5000+ 只 × 逐只拉取，在限流下首次建库要按小时到天计；上游网页一改版接口就坏，回测数据的可重复性没保证。

### 许可
- 【读到】「本项目提供的数据接口及相关数据仅用于学术研究，任何个人、机构及团体使用本项目的数据接口及相关数据请注意商业风险」。来源：docs/introduction.md
- 【推断】库本身 MIT，但数据版权属于各上游网站（东财、新浪、巨潮等），AKShare 不给任何数据授权；本地保存和个人使用的合规性取决于各网站条款，本次未逐一核实。

---

## 3. BaoStock（证券宝，www.baostock.com）

说明：官网 2025–2026 年改版成单页应用，文档正文由页面自己的公开接口 `helpdocs/api/markdown/<文件>.md` 提供；下面引用的页面地址是 `https://www.baostock.com/mainContent?file=<文件>.md`。

### 维护状态
- 【读到】PyPI 最新 0.9.4（2026-09-21），此前 0.9.3（2026-07-10）、0.9.2（2026-06-06）、0.9.1（2026-04-14），再往前是 0.8.9（2024-05-31）。来源：https://pypi.org/project/baostock/
- 【读到】官网版本信息：2026/10/10「新增支持 ETF 历史日、周、月 K 线」；2026/09/21「新增对 API_KEY 的支持，可以访问 vip-api.baostock.com 节点」；2026/07/10 新增「每日更新」类接口（`query_daily_history_k_AStock()`、`query_daily_adjust_factor()`）。来源：https://www.baostock.com/mainContent?file=home.md
- 【推断】2019–2024 年几乎停更，2026 年重新活跃并开始商业化（站内出现「流量套餐」「API Key 管理」「交易技术商城」）。

### 费用
- 【读到】「免费、开源的证券数据平台（无需注册）」。来源：home.md
- 【读到】访问规则：「每日 API 请求不能超过 5 万次，超过后进入黑名单控制。并且不能并发连接访问。」「本自然年度初次被黑名单限制访问，冻结 6 小时后自动解除…限制时长 = 本年累计限制次数 × 6 小时」。来源：https://www.baostock.com/blacklist （页面文字）
- 【未核实】VIP 节点 / 流量套餐的价格（在登录后的个人中心里，没有访问）。

### ①–⑧ 逐项
| 项 | 结论 | 接口 / 字段 | 说明 |
|---|---|---|---|
| ① 日线 | 有 | `query_history_k_data_plus`：1990-12-19 至今，字段含 `volume`、`amount`、`turn`、`preclose`；`adjustflag` 1 后复权 / 2 前复权 / 3 不复权 | 代码格式只有 `sh.` / `sz.`，【推断】没有北交所 |
| ① 复权因子 | 有 | `query_adjust_factor`；「BaoStock 提供的是涨跌幅复权算法复权因子」「与同花顺、通达信等存在不同」；因子只在除权除息日有记录；`start_date` 为空默认 2015-01-01（会丢早期因子） | 来源：pythonAPI.md、default.md |
| ① 退市股 | 部分核实 | `query_stock_basic` 有 `status`（1 上市 / 0 退市）与 `outDate`（退市日期）；行业分类示例里出现 sh.600001 邯郸钢铁（已退市） | 【推断】退市股 K 线可取；文档没有明说覆盖是否完整 |
| ② ST / 停牌 | 有（内嵌在日线里） | 日线字段 `isST`（1 是 / 0 否）、`tradestatus`（1 正常交易 / 0 停牌）；停牌日有行：「开、高、低、收价都相同，且都为前一交易日的收盘价，成交量、成交额为 0」 | 免费源里唯一直接给每日 ST 标记的；`isST` 是 2018-12 后补的（数据调整记录「2018-12-16：补充日线 isST 数据」），早年准确性【未核实】 |
| ② 涨跌停价 | 没有 | — | 要自己按规则算 |
| ③ 财务 + 公告日 | 部分 | 6 个季频接口（`query_profit_data` 等），2007 年起，字段 `pubDate`（公司发布财报的日期）、`statDate`；**只有比率和少量科目，不是完整三表**（官网：「部分上市公司资产负债信息…」） | 【读到】「同一证券同一报告期在数据库中仅保留一条记录，如财报数据被修正，返回的是最新披露口径的数据」→ **不保留更正前版本**；逐只股票 × 逐季度取 |
| ④ 业绩预告 | 有 | `query_forecast_report`：2003 年起，`profitForcastExpPubDate`、`profitForcastType`、增长上下限 | 逐只股票取 |
| ④ 业绩快报 | 有 | `query_performance_express_report`：2006 年起，`performanceExpPubDate`、`performanceExpUpdateDate`（更正后会晚于首次发布日） | 逐只股票取 |
| ⑤ 指数成分权重 | 部分 | `query_sz50_stocks` / `query_hs300_stocks` / `query_zz500_stocks(date=…)`：按周一更新的历史快照，2006-01 起 | **没有权重；没有中证1000、中证2000** |
| ⑥ 市值 | 没有（可近似） | 日线只有 `peTTM` / `pbMRQ` / `psTTM` 估值；股本 `totalShare` / `liqaShare` 只在季频 `query_profit_data` 里 | 【推断】只能用季频股本 × 收盘价近似，股本变动日会错 |
| ⑥ 行业历史 | 部分 | `query_stock_industry(date=…)`：每周一更新的快照，分类标准「申万一级行业」或「证监会行业分类」 | 只有一级；历史起点【未核实】 |
| ⑦ 盈利预测 | 没有 | — | |
| ⑧ 分红送转 | 有 | `query_dividend_data`：`dividPlanAnnounceDate`（预案公告日）、`dividRegistDate`、`dividOperateDate`（除权除息日）等 | 逐只 × 逐年取 |

### 已知问题
- 【读到】官方「数据调整记录」列了 20 条修正，最近两条是 2026-01-05（加 ETF）和 2020-11-26；2019–2020 年修过非交易日有 K 线、涨跌幅异常、财务比率放大 100 倍等。来源：https://www.baostock.com/mainContent?file=modifyRecord.md
- 【读到】每日 5 万次上限 + 禁止并发：全市场 5000+ 只 × 多接口逐只拉取，一天只够拉一两类数据；黑名单统计页当天就有 IP 因 5 万多次被封。
- 【读到】复权算法（涨跌幅复权）与其他源不一致，不能把 BaoStock 因子和别家价格混用。

### 许可
- 【读到】官网：「同时支持通过 BaoStock 的数据存储功能，将数据全部保存到本地后进行分析」；SDK 许可证 BSD（PyPI）。
- 【未核实】没有找到单独的用户协议或对实盘 / 商用的限制条款。

---

## 4. 聚宽 JQData（jqdatasdk）

说明：聚宽文档页是壳页面，正文由公开接口 `help/api/getContentById` 提供；下面给的是对应的文档页地址 `https://www.joinquant.com/help/api/doc?name=<模块>&id=<编号>`。

### 维护状态
- 【读到】PyPI jqdatasdk 最新 1.9.8（2026-01-29），上一版 1.9.7（2024-12-26）；GitHub 最后提交 2026-01-29。来源：https://pypi.org/project/jqdatasdk/ ；https://github.com/JoinQuant/jqdatasdk/commits
- 【推断】SDK 更新慢但服务在运营（文档里有 2024-02 证监会行业切换的说明、多个标「新」的接口）。接口多年未破坏性变更。

### 费用（文档页无日期，读取于 2026-11-12）
- 【读到】试用：提交申请即得「有效时长为 3 个月的试用账号」；历史数据范围「前 15 个月~前 3 个月」（「不包含最近 3 个月，且可调取的历史范围最长为 1 年」）；连接数 1 个；每日流量 100 万条。正式账号：12 个月、历史不限、可多连接、每日 20000 万条。「中国大陆以外地区无法使用」。来源：https://www.joinquant.com/help/api/doc?name=logon&id=9830
- 【读到】购买：「只能一次购买一年」「不提供行情数据实时获取；不提供 level2 数据」，价格不在页面上，写的是「付费方案可点击此处参考」（指向腾讯文档表格 https://docs.qq.com/sheet/DTmZwbXdwY3RHZ1dU ，表内工作表名为「股票基础付费方案」「股票进阶付费方案」「期货期权付费方案」）并要求加企业微信咨询。来源：https://www.joinquant.com/help/api/doc?name=logon&id=9831
- 【未核实】个人正式版具体价格：表格内容没能加载出来，搜索也没找到带日期的官方报价。
- 【推断】试用账号**不能做回测建库**：只给 12 个月窗口且不含最近 3 个月，只够验证字段和数据质量。

### ①–⑧ 逐项（正式账号口径；文档模块 name=JQDatadoc）
| 项 | 结论 | 接口 / 字段 | 来源（id） |
|---|---|---|---|
| ① 日线 + 复权 | 有（2005 年起） | `get_price`，字段 `open/close/high/low/volume/money/factor/high_limit/low_limit/avg/pre_close/paused`；`fq='pre'/'post'/None`，`factor` 返回对应复权因子 | 9874 |
| ① 退市股 | 有 | `get_all_securities(date=None)` 返回「所有日期的股票信息」，`end_date` 为退市日期（未退市为 2200-01-01）；`get_price` 对「上市前或者退市后数据都为 nan」。**行情只从 2005 年开始**，2005 年前退市的股票没有行情 | 10243、9874、9849 |
| ② ST | 有 | `get_extras('is_st', …)`：2005 年至今，盘前 9:15 更新；「股改 s、st、*st 和退市整理期标的都会被判定为 True」。另有 `finance.STK_STATUS_CHANGE`（实行 ST、*ST、暂停上市、终止上市的变动记录） | 9851、10032 |
| ② 停牌 | 有 | `get_price` 的 `paused` 字段；停牌日默认用停牌前数据填充（`skip_paused` / `fill_paused` 可控） | 9874 |
| ② 涨跌停价 | 有 | `get_price` 的 `high_limit` / `low_limit`（会随复权选项一起复权，取真实价要 `fq=None`） | 9874 |
| ③ 财务 + 公告日 | 有，时点处理较好 | 单季度：`get_fundamentals`（`pubDate` / `statDate`），`get_fundamentals(date=…)` 返回「某天所能看到的最新一期的财报」。报告期：`finance.STK_INCOME_STATEMENT` 等，字段 `pub_date`、`end_date`、`report_type`（0 本期 / 1 非本期，一般为上年同期）、`report_date`、`source_id` | 10285、9894、10286 |
| ③ 更正版本 | 部分 | 官方说明：「理论上要保留同一报告期的所有版本，目前受数据源的限制，这块处理规则是按最新更新的（部分比较久远的数据可能会用第一个版本）」；报告期数据「覆盖维护后会保留本期和上期数据；pub_date 披露时间按第一次披露时间」 | 10285、10286 |
| ④ 业绩预告 | 有 | `finance.STK_FIN_FORCAST`：2005 年起，`pub_date`（公布日期）、`type`、`profit_min/max`、`profit_ratio_min/max` | 9893 |
| ④ 业绩快报 | 有 | `finance.STK_PERFORMANCE_LETTERS`：2005 年起，`pub_date`、`report_type`（0 本期 / 1 上期） | 10547 |
| ⑤ 指数成分 | 有 | `get_index_stocks(index, date)`：2005 年至今，按日期返回成分列表 | 9958 |
| ⑤ 指数权重 | 有（月度） | `get_index_weights(index_id, date)`：「中证指数公司一般只在月末 / 月初披露」，查不到当日时返回最近一期。官网指数列表页（https://www.joinquant.com/indexData ，约 730 个指数）里有中证1000（000852.XSHG，日期 2014-10-17，即发布日）、沪深300（2005-04-08）、中证500（2007-01-15），**没有中证2000（932000），也没有任何 93 开头的代码** | 9959、10291 |
| ⑥ 市值 | 有 | `get_valuation`：2005 年至今，`market_cap`、`circulating_market_cap`（亿元）、`capitalization`、`circulating_cap`，1.9.5 起有 `free_market_cap`、`a_market_cap` | 10325 |
| ⑥ 行业历史 | 有 | `get_history_industry(name, securities)`：支持申万一 / 二 / 三级（sw_l1、sw_l2、sw_l3）、聚宽行业、证监会行业（zjw），返回 `start_date` / `end_date` 纳入剔除记录；`get_industry(security, date)` 按日期查。备注了申万 2014-02-20、2021-12-11 两次分类标准切换。**没有中信行业** | 10769、10282 |
| ⑦ 盈利预测 | 没有（基础包里） | 文档目录里没有一致预期 / 研报预测类接口 | 目录 https://www.joinquant.com/help/api/doc?name=JQDatadoc |
| ⑧ 分红送转 | 有 | `finance.STK_XR_XD`：2005 年起，`board_plan_pub_date`、`shareholders_plan_pub_date`、`implementation_pub_date`、`a_registration_date`、`a_xr_date`、派息 / 送股 / 转增比例 | 10022 |

### 已知问题（都是官方「常见数据问题」里写的）
- 【读到】行情自 2005 年开始；长期停牌股在 2005 年附近可能整段 NaN（举例 000638）。id 9849
- 【读到】代码变更（含退市后重新上市）：多数前后独立，「个别是直接覆盖的」（000931 / 601607）；指数 / 行业成分里「许多变更代码的标的都是用新代码直接覆盖原代码」。id 9849、10291
- 【读到】`get_price` 默认前复权（`fq='pre'`），价格和成交量都被复权，回测要显式取 `fq=None` + `factor`。id 9849
- 【读到】`is_st` 是广义口径（含股改 S、退市整理期）。id 9851
- 【读到】财务更正：单季度数据对半年内的修正会直接覆盖，不保留旧版本。id 10285

### 许可
- 【读到】「本地数据指接口在本地运行，数据可以保存在本地」。来源：https://www.joinquant.com/help/api/doc?name=logon&id=9828
- 【读到】用户协议 5.1：「用户承诺使用聚宽 JQData 的数据或利用 JQData 数据加工之后的数据，只做研究目的使用，不用作商业目的和商业用途」；3.2：未经书面许可不得提供给第三方；账号仅限本人，「一台设备仅支持一个账号使用」。来源：https://www.joinquant.com/help/api/doc?name=JQDatadoc&id=11188
- 【推断】本地落盘允许；个人实盘是否算「研究目的」协议没有明说，属灰色地带，介意的话购买前问客服。

---

## 5. 米筐 RQData（rqdatac）

### 维护状态
- 【读到】PyPI rqdatac 最新 3.8.2（2026-11-06），此前 3.8.1（2026-10-16）、3.7.1（2026-09-14）；官方更新履历同步记录到 3.8.2，约每 2–4 周一版。来源：https://pypi.org/project/rqdatac/ ；https://www.ricequant.com/doc/rqdata/python/changelogs.html
- 【读到】3.8.2 更新含「index_weights_ex 增加创业板指和中证2000 发布日前回填权重」。同上。

### 费用（官网购买页由脚本渲染，以下文字读自页面脚本资源，读取于 2026-11-12）
- 【读到】试用：「试用期 15 天」「1GB / 日流量」；文档写「RQData 对于试用账户进行了每天 1G 的配额限制」。来源：https://www.ricequant.com/welcome/purchase ；https://www.ricequant.com/doc/rqdata/python/manual.html
- 【读到】公开标价的只有期货 / 期权数据包：「期货 - 标准版 原价 5000 元/年，折扣价 4000 元/年」「多 IP 版本 原价 7500 元/年，折扣价 6000 元/年」；限制「1GB 流量/天…最多同时 3 线程连接（只能同时在一台电脑上登录）」。**股票数据写的是「更细粒度历史，实时数据，与更多数据种类，欢迎致电咨询购买」**。来源：https://www.ricequant.com/welcome/purchase
- 【未核实】个人买 A 股数据（含财务 PIT、指数权重、一致预期）的价格：官网没有标价，搜索也没找到。【推断】比期货包（4000 元/年）贵，一致预期等另类数据大概率单独计价。

### ①–⑧ 逐项
来源页：通用 https://www.ricequant.com/doc/rqdata/python/generic-api.html ；A 股 https://www.ricequant.com/doc/rqdata/python/stock-mod.html ；指数 https://www.ricequant.com/doc/rqdata/python/indices-mod.html ；另类 https://www.ricequant.com/doc/rqdata/python/alternative-data.html

| 项 | 结论 | 接口 / 字段 |
|---|---|---|
| ① 日线 + 复权 | 有 | `get_price`：`open/close/high/low/volume/total_turnover/prev_close/limit_up/limit_down`，`adjust_type` none / pre / post；`get_ex_factor` 返回 `ex_date`、`ex_factor`、`ex_cum_factor` |
| ① 退市股 | 有 | `all_instruments(type='CS', date=None)`（传 date 才按当日可交易筛选）；`instruments` 字段 `de_listed_date`、`status`（Active / Delisted / TemporarySuspended…）。行情起始：文档首页写「过去 10 多年以来」，rqalpha 文档写「A 股 2005 年至今」 |
| ② ST | 有 | `is_st_stock(ids, start, end)`：「从公告日的下一个交易日开始标记」；`get_special_treatment_info` 给出每次 ST / 撤销的 `change_date`（实施日）与 `info_date`（信息发布日），示例回溯到 1999 年 |
| ② 停牌 | 有 | `is_suspended(ids, start, end)`（全天停牌） |
| ② 涨跌停价 | 有 | `get_price` 的 `limit_up` / `limit_down` |
| ③ 财务 + 公告日 | 有，**真正的时点数据** | `get_pit_financials_ex(ids, fields, start_quarter, end_quarter, date, statements='latest'/'all')`：返回 `info_date`（公告发布日）、`if_adjusted`（0 当期 / 1 非当期）；`statements='all'` 返回某报告期的所有记录，按 `date` 回溯当时可见版本 |
| ④ 预告 / 快报 | 有 | `performance_forecast`（按 `info_date` 公告日取，字段含预告类型、净利润上下限）；`current_performance`（按 `info_date` 或 `quarter`） |
| ⑤ 指数成分 | 有 | `index_components(index, date / start_date+end_date)`，可返回入库时间 |
| ⑤ 指数权重 | 有（月度 + 日度） | `index_weights`：月度；`index_weights_ex`：**日度权重**，「目前仅支持上证 50、沪深 300、中证 A500、中证 500、中证 800、中证 1000、中证 2000、科创 50 和中证红利」，是「基于上月月末成分股自算权重」并结合 ETF 申赎清单调整；中证 2000「正式发布日（2023 年 8 月 31 日）之前的历史权重，依据当时适用的编制规则自行计算」，2026-03-18 起中证 2000 日度权重不再结合 ETF 清单 |
| ⑥ 市值 | 有 | `get_shares`（`total`、`circulation_a`、`total_a`、`free_circulation` 自由流通 2005 年起）；因子 `market_cap` 等经 `get_factor` 取 |
| ⑥ 行业历史 | 有（中信 + 申万未核实） | `get_industry_change(industry, source='citics_2019'/'citics'/'gildata', level)` 返回 `start_date` / `cancel_date`；`get_instrument_industry(ids, source, level, date)` 按日期查。文档列的 source 只有中信 2019、中信旧版、聚源；申万是否支持【未核实】（`shenwan_instrument_industry` 这类旧接口在本次读的页面里没看到） |
| ⑦ 一致预期 | 有（另类数据） | `consensus.get_comp_indicators`：每个交易日一条，一致预期营收 / 净利润 / EPS（T+1~T+3、未来 12 个月）、目标价、评级；`report_range` 参数区分「考虑 / 不考虑补录入」（传 3 可得不随历史修复变动的值）；另有 `consensus.get_security_change`、`get_expect_appr_exceed` 等。是否包含在基础授权内【未核实】 |
| ⑧ 分红送转 | 有 | `get_dividend`（分红宣布日 `declaration_announcement_date`、股权登记日、除权除息日、到账日）、`get_dividend_info`（3.8.2 新增 `proposal_announcement_date` 等）、`get_split` |

### 已知问题
- 【读到】`get_price` 默认 `adjust_type='pre'`（前复权），`skip_suspended=False` 时停牌日用停牌前数据补齐。回测需显式取不复权价。generic-api
- 【读到】`index_weights_ex` 的日度权重是米筐自算，不是指数公司原始权重；中证 2000 发布前是按规则回填的模拟成分。indices-mod
- 【读到】一致预期默认 `report_range=0`「历史值修复会存在数值变动」，回测要用 3 或 4。alternative-data
- 【读到】流量按字节计：试用 / 普通账户 1GB/天。【推断】全市场日线建库够用，分钟线不够。

### 许可
- 【读到】试用协议：「仅限授权许可在用户自身内部研究的范围内使用，不得用于商业用途」；「每个 License 仅限于在 1 台用户拥有的载体设备上使用」；「试用期结束后…应在该等载体设备上彻底删除已安装的本终端以及其中包含的全部数据和其他信息」。来源：RQData 金融数据终端试用协议（购买页链接的 PDF，https://assets.ricequant.com/welcome/RQData金融数据终端试用协议.63683e60.pdf ）
- 【读到】正式使用协议（数据终端使用协议）：未经同意不得将内容「储存于检索系统、传送、复制、分发或以任何其他方式作商业或公共用途」。来源：https://assets.ricequant.com/welcome/米筐科技RQData数据终端使用协议.c128c1b5.pdf
- 【推断】付费期内本地缓存供自己研究可以（文档自带 to_csv 教程）；到期后能否继续保留历史数据、个人实盘是否允许，要以购买时签的《RQData 数据 API 服务合同》为准，本次没读到该合同正文。

---

## 5b. RQAlpha 免费数据 bundle

### 维护状态
- 【读到】rqalpha PyPI 最新 6.5.0（2026-10-28），GitHub 最后提交 2026-10-27。来源：https://pypi.org/project/rqalpha/ ；https://github.com/ricequant/rqalpha/commits

### 费用
- 【读到】「RiceQuant 免费提供日级别的股票、常用指数、场内基金和期货数据供回测使用。数据每个月月初更新」，命令 `rqalpha download-bundle`；每日更新（`rqalpha update-bundle`）需要 RQDatac license。来源：https://github.com/ricequant/rqalpha/blob/master/docs/source/intro/install.rst

### 内容（读自 bundle 生成代码 https://github.com/ricequant/rqalpha/tree/master/rqalpha/data/bundle ）
- 【读到】起始日 `START_DATE = 20050104`。
- 【读到】`stocks.h5` 字段：`open, close, high, low, prev_close, limit_up, limit_down, volume, total_turnover`（不复权）；另有 `indexes.h5`、`funds.h5`、`futures.h5`。
- 【读到】`ex_cum_factor.h5`（复权因子）、`split_factor.h5`、`dividends.h5`、`st_stock_days.h5`、`suspended_days.h5`、`instruments.pk`（由 `rqdatac.all_instruments()` 生成）、`trading_dates.npy`、`share_transformation.json`、`yield_curve.h5`。
- 【推断】`instruments.pk` 含已退市股票（`all_instruments()` 不传 date 时不做筛选），所以 bundle 能覆盖 ① 和 ② 的大部分：2005 年起含退市股的日线、成交额、涨跌停价、复权因子、ST 日、停牌日、现金分红。**需要下载后实测确认退市股数量。**

| 项 | 结论 |
|---|---|
| ① | 有（2005 年起；月更） |
| ② | 有（ST 日、停牌日、涨跌停价都在） |
| ③ ④ | 没有 |
| ⑤ | 没有（只有指数行情，无成分权重） |
| ⑥ | 没有（无股本 / 市值 / 行业） |
| ⑦ | 没有 |
| ⑧ | 部分（现金分红 + 拆分因子，无公告日口径的预案信息） |

### 许可
- 【读到】rqalpha README：「仅限非商业使用。如需商业使用，请联系我们」。来源：https://github.com/ricequant/rqalpha/blob/master/README.rst
- 【推断】bundle 就是本地文件，长期保存没有技术限制；个人研究属于非商业使用。

---

## 6. 掘金量化 gm SDK（myquant.cn）

### 维护状态
- 【读到】PyPI gm 最新 3.0.193（2026-11-05），此前 3.0.192（10-27）、3.0.191（10-23），发版很勤。来源：https://pypi.org/project/gm/
- 【读到】**平台限制**：「SDK 需要和掘金终端通信，但是掘金终端只有 Windows 版本」；Linux 下可用 Python SDK 但要指向一台 Windows 终端。来源：https://www.myquant.cn/docs2/faq/
- 【推断】在 macOS 上不能单机使用，需要另备 Windows 机器或虚拟机常开终端。

### 费用
- 【读到】文档把函数分成「通用数据函数（免费）」「股票财务数据及基础数据函数（免费）」「股票增值数据函数（付费）」三组。来源：https://www.myquant.cn/docs2/docs/
- 【读到】调用上限：通用数据接口「每日封顶 60 万次」，股票数据接口「每日封顶 159.4 万次」，另有每 5 分钟的频度限制（部分基本面接口每 5 分钟 10 次）；「免费版目前只支持订阅 50 个标的」。来源：https://www.myquant.cn/docs2/faq/数据问题.html
- 【未核实】增值数据（行业、分红、复权因子、业绩预告等）的价格，以及体验版 / 专业版的差价：产品页是脚本渲染的，没读到；文档只说「可联系商务开通更多权限」。

### ①–⑧ 逐项
来源：通用 https://www.myquant.cn/docs2/sdk/python/API介绍/通用数据函数（免费）.html ；财务基础 …/股票财务数据及基础数据函数（免费）.html ；增值 …/股票增值数据函数（付费）.html ；行情 …/行情数据查询函数（免费）.html

| 项 | 结论 | 接口 / 字段 |
|---|---|---|
| ① 日线 | 有（免费） | `history` / `history_n`，`adjust` 不复权 / 前复权 / 后复权，`adjust_end_time` 定点复权；日线 2005-01-01 起（数据文档首页），FAQ 又写「日频行情从标的上市以来均可获取」——**两处说法不一致，【未核实】** |
| ① 复权因子 | 有（付费） | `stk_get_adj_factor`（前 / 后复权因子与累计因子）；另外 `get_history_symbol` 返回字段里有 `adj_factor`（免费） |
| ① 退市股 | 部分核实 | `get_symbols(trade_date=None)`「取最新截面（包含退市标的）」，返回 `delisted_date`。退市股日线能否取：文档没说，【未核实】 |
| ② ST / 停牌 / 涨跌停价 | 有（免费） | `get_history_symbol`（单标的多日）与 `get_symbols`（单日多标的）返回 `is_st`、`is_suspended`、`upper_limit`、`lower_limit`。注意 `get_symbols` 默认 `skip_suspended=True, skip_st=True` 会把它们过滤掉 |
| ③ 财务 + 公告日 | 有（免费），带版本 | `stk_get_fundamentals_balance/income/cashflow`（单标的时序）与 `_pt`（截面，返回「发布日早于查询日的最新报告期」）；字段 `pub_date`（发布日期）、`rpt_date`（报告截止日）；`data_type`：101 合并原始 / 102 合并调整 / **100 合并最初（未修正的最初版本）**，200 系列为母公司。数据文档首页写资产负债表、利润表「维护至 1989-12-31」、现金流量表「维护至 1996-12-31」（应指最早年份） |
| ④ 业绩预告 | 有（付费） | `stk_get_finance_forecast`，`pub_date` 释义「最新公告日期」 |
| ④ 业绩快报 | 没有 | 增值函数页里没有快报函数 |
| ⑤ 指数成分权重 | 成分有，**中证权重没有** | `stk_get_index_constituents(index, trade_date)` 可查历史；「中证系列指数因版权原因 weight 固定为 0」 |
| ⑥ 市值 | 有（免费） | `stk_get_daily_mktvalue(_pt)`（市值）、`stk_get_daily_basic(_pt)`（股本）、`stk_get_daily_valuation(_pt)` |
| ⑥ 行业历史 | 有（付费） | `stk_get_symbol_industry` / `stk_get_industry_constituents` 有 `date` 参数可查历史；`source` 只有 `zjh2012`（证监会）和 `sw2021`（申万 2021）。没有中信；2021 年前的旧申万口径【未核实】 |
| ⑦ 盈利预测 | 没有 | — |
| ⑧ 分红送转 | 有（付费） | `stk_get_dividend`：`pub_date`、`equity_reg_date`、`ex_date` |

### 已知问题
- 【读到】`pub_date`「这个日期会根据财报更新的时间实时更改」——同一报告期的公告日会被更正公告覆盖，做时点回测要配合 `data_type=100`（最初版本）使用，并自己验证。来源：FAQ 数据问题
- 【读到】「停牌股票当天不会生成日线和 bar 数据」；盘前才发的停牌公告当天才处理到。同上
- 【读到】`history` 的 `skip_suspended` 和 `fill_missing` 参数「暂不支持」；`history_n` 单次最多 33000 条。行情函数页
- 【读到】中证指数权重恒为 0。财务基础函数页

### 许可
- 【读到】FAQ 只说取到的数据可用 pandas 存成 csv / excel。服务协议正文没有读到，【未核实】本地长期保存和实盘的条款。

---

## 7. 通达信本地数据（mootdx / pytdx）

### 维护状态
- 【读到】pytdx：GitHub 最后一次提交 2020-04-15，提交信息就是「Archive pytdx」；PyPI 最新 1.72（2019-08-26）。**已停止维护。** 来源：https://github.com/rainx/pytdx/commits ；https://pypi.org/project/pytdx/
- 【读到】mootdx：PyPI 最新 0.11.7（2024-05-04），GitHub 最后提交 2024-07-16。**两年多没更新。** 来源：https://pypi.org/project/mootdx/ ；https://github.com/mootdx/mootdx/commits
- 【推断】两者都是对通达信私有协议 / 本地文件格式的逆向实现，通达信一改服务器或文件格式就失效，且没人修。

### 费用
- 【读到】免费；需要自己装通达信客户端下载盘后数据（`Reader.factory(market='std', tdxdir=…)` 读本地 vipdoc），或连通达信行情服务器（`Quotes.factory`）。来源：https://github.com/mootdx/mootdx/blob/master/README.md

### ①–⑧ 逐项（依据 mootdx 源码 https://github.com/mootdx/mootdx/tree/master/mootdx ）
| 项 | 结论 | 说明 |
|---|---|---|
| ① 日线 | 有 | `reader.daily(symbol)` 读本地 .day 文件；`client.bars / client.k` 在线取 K 线 |
| ① 复权因子 | 部分 | 没有现成因子；`client.xdxr(symbol)` 取除权除息 / 股本变迁记录，要自己算复权 |
| ① 退市股 | 【未核实】 | 没找到官方说明；通达信盘后数据包是否保留已退市代码需要实测 |
| ② ST / 停牌 / 涨跌停价 | 没有 | 源码里没有对应接口；只能从名称变更或零成交日自己推 |
| ③ 财务 + 公告日 | 部分 | `Affair.fetch / Affair.parse` 下载通达信专业财务包 `gpcwYYYYMMDD.zip`（README 示例从 `gpcw19960630.zip` 起），每个报告期一个文件、580 列左右，列名里有「**财报公告日期**」「业绩预告公告日期」「业绩快报公告日期」。每个报告期只有一份文件，【推断】是最新口径、不保留更正前版本 |
| ④ 预告 / 快报 | 部分 | 同一财务包里有「业绩预告公告日期」「业绩预告-本期净利润下限 / 上限（万元）」「业绩快报公告日期」；快报的具体数值列【未核实】 |
| ⑤ 指数成分权重 | 没有历史 | `client.block()` 读板块文件，只有当前成分，无权重 |
| ⑥ 市值 / 行业 | 部分 | 财务包里有「总股本」「已上市流通 A 股」「自由流通股」（按报告期）；`xdxr` 里有股本变动。每日市值要自己拼。行业只有通达信自有分类的当前值 |
| ⑦ 盈利预测 | 没有 | — |
| ⑧ 分红送转 | 有 | `client.xdxr(symbol)` |

### 已知问题
- 【读到】mootdx README：「M1 mac 系统 PyMiniRacer 不能使用」（Apple Silicon 上有依赖问题）。
- 【推断】通达信复权、财务口径与其他源不一致；没有任何时点保证；数据完整性完全取决于本机通达信客户端下载了什么。

### 许可
- 【读到】mootdx README：「郑重声明：本项目只作学习交流，不得用于任何商业目的」；代码 MIT。
- 【未核实】通达信数据本身的使用条款（没有读通达信软件许可协议）。本地文件天然可长期保存。

---

## 8. adata（1nchaos/adata）

### 维护状态
- 【读到】PyPI 最新 2.9.5（2025-12-26），此前 2.9.4（2025-09-09）；GitHub 最后提交 2026-10-18（合并外部 PR），最后一个 GitHub Release 是 v2.9.0（2025-04-02）。来源：https://pypi.org/project/adata/ ；https://github.com/1nchaos/adata/commits
- 【推断】维护变慢，2026 年以来没有发过 PyPI 版本。

### 费用
- 【读到】免费开源（PyPI 标注 Apache License）；「专注股票行情数据，为了保证数据的高可用性，采用多数据源融合切换」；数据来自同花顺、东方财富、百度、新浪、腾讯等公开接口，「项目是基于公开接口，可能存在限制等，因此增加代理设置功能」。来源：https://github.com/1nchaos/adata/blob/main/README.md

### ①–⑧ 逐项（依据 README 接口表）
| 项 | 结论 | 说明 |
|---|---|---|
| ① 日线 | 部分 | `stock.market.get_market()` 单只股票日 / 周 / 月 K 线；README 接口表里没有复权因子接口，退市股覆盖【未核实】 |
| ② ST / 停牌 / 涨跌停价 | 没有 | 接口表里没有 |
| ③ 财务 | 基本没有 | 只有 `stock.finance.get_core_index()` 核心财务指标，「三大报表详细数据，暂时不提供」 |
| ④ 预告 / 快报 | 没有 | |
| ⑤ 指数成分权重 | 只有当前成分 | `stock.info.index_constituent()`，无历史、无权重 |
| ⑥ 市值 / 行业 | 部分 | `get_stock_shares()` 单只股本信息；`get_industry_sw()` 单只股票当前申万一二级行业（来源：百度），无历史 |
| ⑦ 盈利预测 | 没有 | |
| ⑧ 分红 | 有 | `stock.market.get_dividend()` 单只股票分红信息 |

### 结论性判断
- 【推断】定位是「实时 / 近期行情 + 概念热度」，不是回测数据源；①–⑧ 里只有日线和分红算可用，不建议作为选型主力。
- 许可：【未核实】README 里没找到数据使用声明；数据版权归各上游网站。

---

## 9. Qlib 社区数据（chenditc/investment_data）

### 维护状态
- 【读到】每日发布：GitHub Releases 最新标签 2026-11-11（发布时间 2026-11-11T16:37Z），11-10、11-09、11-08 连续都有；最后提交 2026-11-11。来源：https://github.com/chenditc/investment_data/releases
- 【读到】microsoft/qlib 本体：最新 v0.9.8（2026-10-08）。来源：https://github.com/microsoft/qlib/releases

### 费用
- 【读到】免费。下载 release 里的 `qlib_bin.tar.gz` 解压到 `~/.qlib/qlib_data/cn_data`；原始表托管在 DoltHub（`dolt clone chenditc/investment_data`）。自己跑每日更新需要 Tushare token。来源：https://github.com/chenditc/investment_data/blob/main/README.md

### 内容
- 【读到】数据源前缀：w（Wind，「high quality static data source. Only available till 2019」）、c（财汇，同样只到 2019）、ts（Tushare）、ak、yahoo、baostock，final 为合并校验后的结果；「Currently the daily update is only using tushare data source」。README、docs/final_a_stock_eod_price.md
- 【读到】项目初衷第一条：「fill in missing data by combining data from multiple data source. For example, delist company's data」→ 目标是包含退市股。
- 【读到】价格表 `final_a_stock_eod_price`（开高低收、成交量、成交额、adjclose，导出时算 vwap）；涨跌停表 `final_a_stock_limit`（Tushare 涨跌停价为主，早期用 BaoStock 的 ST 标记推算，1996 年 12 月涨跌停制度实施前的数据删除（文档原文日期写作「1996-12-196」，有笔误））。docs/final_a_stock_limit.md
- 【读到】指数成分：`csi300`（399300.SZ）、`csi500`（000905.SH）、`csi800`、`csi1000`（000852.SH）、`csiall`（000985），来自 Tushare `index_weight`，导出为「symbol + 起止日期」区间文件，「Historical members remain in the file」。qlib/dump_index_weight.py、tushare/dump_index_weight.py
- 【推断】由此可知 Tushare `index_weight` 能取到中证1000（000852.SH）；**没有中证2000**。

| 项 | 结论 |
|---|---|
| ① | 有（含退市股是项目目标；复权用 adjclose / factor） |
| ② | 部分：有涨跌停价；ST、停牌状态不在导出的 qlib 数据里（【推断】，README 未列） |
| ③ ④ | 没有 |
| ⑤ | 部分：沪深300 / 中证500 / 800 / 1000 / 中证全指的历史成分区间；qlib 格式不带权重；无中证2000 |
| ⑥ ⑦ ⑧ | 没有（无市值、行业、预期、分红明细） |

### 已知问题
- 【读到】Issue #163（2026 年处理）：「BJ (Beijing Stock Exchange) adjclose on 2025-10-31 is corrupted and still present on master」，244 只北交所股票复权价错了一年才修（README 里有专门的 `repair-2026-11-08` 修复流程）。来源：https://github.com/chenditc/investment_data/issues/163
- 【读到】2019 年以后只有 Tushare 单一来源，没有交叉校验。
- 【推断】本质是 Tushare 行情的免费再分发 + 2019 年前的 Wind / 财汇底稿；适合当行情的第二来源做交叉核对，不适合当唯一来源。

### 许可
- 【读到】仓库 LICENSE 为 Apache-2.0（针对代码）。
- 【未核实】数据本身的再分发授权（底稿含 Wind / 财汇 / Tushare 数据，仓库没有说明授权链）。个人下载使用没有技术限制，合规风险自担。

---

## 10. 中证指数官网成分 / 权重文件（csindex.com.cn）

### 现状（2026-11-12 实测文件头，未下载文件内容）
- 【读到】官网每个指数的详情接口只列出一份「样本列表」（`<代码>cons`）和一份「样本权重」（`<代码>closeweight`），另有编制方案、指数单张、指数估值。以中证2000 为例：https://www.csindex.com.cn/csindex-home/indexInfo/index-details-data?fileLang=1&indexCode=932000
- 【读到】四个目标指数的文件都在，且在更新：
  - 成分列表 `https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/file/autofile/cons/{000300,000905,000852,932000}cons.xls`，Last-Modified 都是 2026-11-11。
  - 权重 `…/autofile/closeweight/{代码}closeweight.xls`，Last-Modified 都是 2026-11-10。
- 【读到】指数基本信息（官网接口 `csindex-home/indexInfo/index-basic-info/<代码>`）：中证1000 基日 2004-12-31、发布日 **2014-10-17**；中证2000 基日 2013-12-31、发布日 **2023-08-11**。（米筐文档写中证2000「正式发布日 2023 年 8 月 31 日」，与官网不一致，以官网为准。）

### 对 ⑤ 的含义
| 需求 | 结论 |
|---|---|
| 当前成分 | 有（四个指数都有，免费，无需登录） |
| 当前权重 | 有（最新一期；文件内的具体日期没有打开核对，【推断】是最近一个月末或最近收盘） |
| 历史成分 / 历史权重 | **没有**。官网只放最新一份，不提供历史文件 |
| 中证1000 发布前（2014-10 以前）、中证2000 发布前（2023-08 以前） | 不存在官方成分，任何数据商给的都是按规则回填的模拟值 |

- 【推断】可行用法：从今天起每月（或每日）自己定时存档这两个文件，慢慢攒出一份自有的官方权重历史；历史部分仍要靠数据商。定期调样名单在官网公告里，可用来核对数据商的成分变更日期（本次未读公告页）。

### 许可
- 【未核实】官网的版权 / 使用声明页是脚本渲染的，没读到正文。【推断】指数成分与权重是中证指数公司的版权数据：掘金文档明写「中证系列指数因版权原因 weight 固定为 0」，说明再分发受限；个人下载自用的边界本次没有核实到条款原文。

---

## 11. 补充候选：迅投 miniQMT / xtquant（xtdata）

只读了官方文档一页，结论较粗，列出来供参考。来源：https://dict.thinktrader.net/nativeApi/xtdata.html ；https://pypi.org/project/xtquant/
- 【读到】「xtdata 提供和 MiniQmt 的交互接口，本质是和 MiniQmt 建立连接」——必须运行券商提供的 MiniQMT 客户端。PyPI 最新 xtquant 250807.1.2（2026-07-21）。
- 【读到】有的：`get_market_data_ex`（`dividend_type` none / front / back / front_ratio / back_ratio）、`get_divid_factors`（除权数据）、`get_financial_data`（Balance / Income / CashFlow / Capital / Pershareindex 等，`m_anntime` 披露日期、`m_timetag` 截止日期，`report_type='announce_time'` 可按披露日取）、`get_instrument_detail`（`ExpireDate` 退市日、`InstrumentStatus` 停牌状态、当日 `UpStopPrice` / `DownStopPrice`）、`download_history_contracts`（过期 / 退市合约）、`get_index_weight`（需先 `download_index_weight`，是否含历史文档没说）。
- 【读到】「ST 变更历史（specialtreatment）」「涨跌停数据（stoppricedata）」「过期合约列表（historycontract）」被列为**投研版特色数据**。
- 【未核实】费用：MiniQMT 一般由券商给达到门槛的客户免费开通，投研版另收费，具体门槛和价格因券商而异，本次没有查。
- 【推断】MiniQMT 客户端是 Windows 程序（文档没写平台要求），对 macOS 用户和 gm 一样要另备 Windows 环境；没有业绩预告 / 快报和一致预期。

---

## 12. 社区反馈补充（Tushare）

Tushare 的 GitHub 仓库代码已停更，但 Issues 区仍是用户提数据问题的地方（https://github.com/waditu/tushare/issues ，最近更新 2026-11-10）。近一个月的议题标题能看出社区在核对什么（都是提问或询证，**本次没看到官方回复**，不能当成已确认的缺陷）：
- 【读到】#1720（2023-10-09）：「adj_factor 文档上说更新时间是早上 9 点 30 前，结果只是部分更新…'689009.SH' 在 16:45 仍未更新」「'836504.BJ' 在 2022-08-22 有行情，而复权因子直到 2023-08-17 才有数据」。https://github.com/waditu/tushare/issues/1720
- 【读到】#1961（2026-11-07）：询问 adj_factor 对「现金分红加转增」的计算口径、精度、以及更正后历史因子的影响范围，指出**没有公开的算法说明和版本记录**。https://github.com/waditu/tushare/issues/1961
- 【读到】#1951、另一条 2026-10-30 的议题：要求核对 600733.SH（2018-09）、600031 / 600519（2024-06）的 adj_factor 变化与 `daily.pre_close` 是否对得上。
- 【读到】#1954（2026-10-30）：「daily_basic.circ_mv 历史口径与版本事实询证」；#1957（2026-11-01）：「dividend 接口…相同冻结键下出现非主键字段冲突行」；#1943（2026-10-20）：单季度 EPS 空值与「数值版本历史可得时间」。
- 【读到】#1950（2026-10-27，未回复）：一位个人用户问的正是许可问题——5000 积分下本地保存快照是否允许、**年度权限到期后能否继续保留已下载的数据离线使用**；提问者写明登录页显示「5000 积分按年 500 元」。https://github.com/waditu/tushare/issues/1950
- 【读到】官方常见问题：Tushare 的前复权「以用户设定的 end_date 开始往前复权，跟所有行情软件…看到的前复权可能存在差异」；「达到 5000 积分以上基本没有太多频次限制」。https://tushare.pro/document/1?doc_id=122
- 【读到】官方：`update_flag`「更新标志（1 最新）」（现金流量表文档 https://tushare.pro/document/2?doc_id=44 ）；`disclosure_date` 接口有 `pre_date`（预计披露日）、`actual_date`（实际披露日）、`modify_date`（披露日期修正记录）（https://tushare.pro/document/2?doc_id=162 ）。
- 【推断】Tushare 的财务表能拿到「调整前（report_type=5）」和「最新」两种口径，加上 `f_ann_date`，可以拼出近似时点数据；但复权因子、市值、行业成分都没有版本记录，历史值被悄悄修订时无从察觉，建库时要自己留快照。

- 【读到】旁证（退市股）：开源项目 chenditc/investment_data 的取数脚本用 `pro.stock_basic(list_status="D", …)` 取退市名单，用 `pro.daily(trade_date=…)` + `pro.adj_factor(trade_date=…)` 按交易日取全市场（https://github.com/chenditc/investment_data/blob/main/tushare/dump_tushare_stock_list.py 、update_a_stock_eod_price_to_latest.py ）。【推断】按交易日取数会带上当日仍在交易、后来退市的股票，所以 Tushare 能覆盖退市股历史；但该项目同时说要靠多数据源「fill in missing data… For example, delist company's data」，说明单一来源的退市股数据并不完整，仍需实测。

---

## 13. 汇总对比（无标记 = 页面读到；△ = 推断；? = 未核实）

| 候选 | 维护（最近发版 / 提交） | 个人费用 | ① 日线 / 复权 / 退市 | ② ST / 停牌 / 涨跌停价 | ③ 财务公告日 | ④ 预告 / 快报 | ⑤ 指数成分权重 | ⑥ 市值 / 行业历史 | ⑦ 预期 | ⑧ 分红 | 许可 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Tushare Pro | 数据日更（changelog 2026-11-11）；SDK 1.4.29（2026-03） | 2000 分 200 元/年；5000 分 500 元/年；⑦ 要 8000+ 分 | 有；退市名单有，退市股行情 △ | 有（ST 3000 分，2000 年起；涨跌停 2007 年起） | `ann_date` / `f_ann_date` + `update_flag`、`report_type` 5 为调整前 | 有（`ann_date`、`first_ann_date`） | 月度权重；1000 △有；2000 ? | 有；申万 2000 分、中信 5000 分，带纳入剔除日 | 研报明细，非一致预期 | 有 | 个人非商业；到期后保留 ? |
| AKShare | 1.20.37（2026-11-09），几乎日更 | 免费 | 逐只取；退市股靠新浪 △ | 无历史 ST；无涨跌停价；停牌部分 | 字段名 ?，无版本 | 有（公告日期） | 只有最新 | 市值逐只；申万变动历史有 | 只有当前快照 | 有 | MIT 代码；数据「仅用于学术研究」 |
| BaoStock | 0.9.4（2026-09-21） | 免费；每日 5 万次、禁并发 | 1990 年起；退市 △；无北交所 △ | `isST`、`tradestatus` 在日线里；无涨跌停价 | `pubDate`；只留最新口径；非完整三表 | 有（预告 2003 起、快报 2006 起） | 50 / 300 / 500 周快照，无权重，无 1000 / 2000 | 无市值；行业周快照（一级） | 无 | 有 | 可存本地；条款 ? |
| 聚宽 JQData | SDK 1.9.8（2026-01-29） | 试用 3 个月但只给「前 15 个月~前 3 个月」；正式价 ? | 2005 年起；含退市 | `is_st`、`paused`、`high_limit` / `low_limit` 全有 | `pub_date` / `pubDate`；更正多为覆盖 | 有 | 成分按日、权重月度；1000 有（2014-10 起）；**无 2000** | 有；申万 / 证监会带起止日；无中信 | 无 | 有 | 可存本地；「只做研究目的」 |
| 米筐 RQData | 3.8.2（2026-11-06） | 试用 15 天、1GB/天；股票包价格 ?（期货包 4000 元/年） | 有；含退市 | `is_st_stock`、`is_suspended`、`limit_up` / `limit_down` 全有 | **PIT**：`info_date`，可取某日可见的全部版本 | 有 | 成分按日；权重月度 + **日度（含 1000、2000，自算）** | 有；中信带起止日；申万 ? | 有（另类数据，授权 ?） | 有 | 内部研究、非商业；试用到期须删数据 |
| RQAlpha bundle | 6.5.0（2026-10-28） | 免费，月更 | 2005 年起；含退市 △ | ST 日、停牌日、涨跌停价都有 | 无 | 无 | 无 | 无 | 无 | 现金分红 | 仅限非商业 |
| 掘金 gm | 3.0.193（2026-11-05） | 基础免费；增值数据价 ?；**无 macOS 版** | 有；退市名单有，行情 ? | `is_st`、`is_suspended`、`upper_limit` / `lower_limit` 有 | `pub_date`（会被更正覆盖）+ `data_type` 100 最初版 | 预告有（付费）；快报无 | 成分可查历史；**中证权重恒为 0** | 市值有；行业历史付费（申万 2021、证监会） | 无 | 有（付费） | ? |
| 通达信（mootdx / pytdx） | mootdx 0.11.7（2024-05）；pytdx 2020 年归档 | 免费 | 有；无现成因子；退市 ? | 无 | 有「财报公告日期」列；最新口径 △ | 仅公告日和预告上下限 | 无历史 | 股本按报告期；无行业历史 | 无 | 有 | 「不得用于商业目的」 |
| adata | 2.9.5（2025-12） | 免费 | 逐只 K 线；无因子 | 无 | 无三表 | 无 | 只有当前成分 | 当前值 | 无 | 有 | ? |
| investment_data | 每日发布（2026-11-11） | 免费 | 有；以含退市为目标 | 只有涨跌停价 | 无 | 无 | 300 / 500 / 800 / 1000 成分区间，无权重，无 2000 | 无 | 无 | 无 | 代码 Apache-2.0；数据授权 ? |
| 中证官网文件 | 文件 2026-11-10 / 11 更新 | 免费 | — | — | — | — | 四个指数**只有最新一期**成分与权重 | — | — | — | ? |

## 14. 结论

**起步组合 A（约 500 元/年，macOS 可用）**：Tushare Pro 5000 积分做主库（①–⑥、⑧），BaoStock 的 `isST` / `tradestatus` 和 RQAlpha 月度 bundle 的 ST 日、停牌日、涨跌停价做免费交叉核对，AKShare 取申万行业变动历史，再每月自己存档中证官网的成分与权重文件。缺口：指数权重只有月度，中证2000 历史未核实；财务更正版本要自己用 `report_type` 拼；一致预期只有研报明细且要 8000 积分以上；复权因子和市值没有版本记录。

**组合 B（要严格时点）**：米筐 RQData 单独就能覆盖 ①–⑧，有时点财务和含中证1000 / 2000 的日度权重。缺口：个人价格未公开，需询价；申万行业未见；一致预期是否另收费、到期后数据能否保留都没核实。先用 15 天试用验证。

聚宽没有中证2000 和一致预期，试用窗口不够建库；掘金没有 macOS 版且中证权重为 0；通达信、adata 不建议作主力。

## 15. 没核实到的事项（需要你自己确认）
1. 聚宽、米筐（股票包）、掘金增值数据的个人价格。
2. Tushare `index_weight` 是否有中证2000，以及各指数的起始年份。
3. Tushare、BaoStock、gm、通达信对退市股行情的实际覆盖率（都只读到「有退市名单」）。
4. Tushare 年度权限到期后，已下载数据能否继续离线使用（协议没写，社区提问未获答复）。
5. 米筐一致预期是否含在基础授权里；米筐是否支持申万行业。
6. AKShare 东财三表的公告日字段名；`stock_tfp_em`、`stock_value_em` 的历史起点。
7. 中证官网权重文件内的日期口径，以及官网数据使用条款。
