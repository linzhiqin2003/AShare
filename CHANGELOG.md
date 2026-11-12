# Changelog

## 2026-11-12
- 立项：选型与方案（`docs/DESIGN.md` 0.1）

- 步骤 1：建立 Python 3.13/uv 与 Cargo 工作区；基于三方摸底选用 Rust panelbt 内核，提供 numpy 输入/规则/稀疏目标、独立参考实现、随机对拍与性质测试、10 项源码变异检查和可复现性能基准；panelops 与 ashare 保持空壳。
- 方案 0.2：市场参数化、做空与保证金、期货对冲腿（4.11）、模拟盘（第 8 节）、数据层四层封装（6.1）、风险模型与组合优化（6.4）、券商接口与 miniQMT 适配（第 9 节，只做系统侧）；整体结构图 `docs/architecture.svg`。
- 数据：Tushare 全历史原始库下载完成（2007-01 至 2026-11-11，46393 个分区，2.6 GB，`data/cn`，不入库）；实测限额按接口计、每接口每分钟 500 次。实时行情原始录制任务 `com.ashare.rt-recorder`（launchd，交易日 9:05，`data/live`）。
- 模拟盘合入主干：`paperbroker`（Rust 内核的订单、撮合、账户、结算状态机，日志与快照恢复）和 `ashare.paper`（行情源、调度、目标调仓、原始录制导入、停机后补做公司行为、保留策略）。三轮独立审查与返工；180 项 Python 测试、7 项 Rust 测试；真实行情夹具放在 `data/fixtures`（不入库）。
- 根 `.gitignore` 的 `data/` 改为 `/data/`（原写法把 `src/ashare/data` 等代码目录一起忽略了）。
- 数据层合入主干（步骤 2）：Tushare 源适配、原始库（Parquet 分区 + SQLite 清单）、稠密面板（4824 × 5923，46 个字段，约 6.7 GB，只留一代）、读取接口 `ashare.data.open()`、校验报告。三轮独立审查。公司行为以源复权因子为准做对账（容差 0.5%）：按经济条款去重、候选裁决、对不上的因子变化合成等效送转、因子没变的事件否决，来源记在 `action_source`；全历史偏差超过 1% 的格数从 515 降到 0。三只换码股票的历史接到新代码；行业缺口回填并带来源标记；新增 `tradable`。全量构建 43 秒，追加一天 4 秒且与全量逐位一致。
- `panelbt` 完整版合入主干（步骤 1b）：做空与保证金、追保与强平、融资融券费用、公司行为（停牌期间估值价随除权调整）、按证券的申报单位与卖出下限、退市、资金流、诊断计数；回测与模拟盘共用的逐笔算术 `crates/tradecore`；独立参考实现、2 万组摘要夹具、与 numpy / polars / vectorbt / bt / backtrader 的对比基线。两轮审查：独立账本 37000 组 0 不符；把四条规则修正逐条关掉后 2 万组旧样本与 1a 逐位相同。遗留：完整规则 4000×5500 为 3.8 ms（目标 2 ms）、基准需在空闲机器上重测、小数股路径缩减后的零头单、4 处测试缺口（见 `docs/step1b/`）。
- 风险模型与组合优化合入主干（6.4）：结构化 (B, F, D) 风险模型（基本面型：行业加风格因子、Rust 截面回归、EWMA / Newey-West 协方差、波动状态调整、特质风险收缩；统计型：PCA 加 Ledoit-Wolf），按时点构造、追加一天与全量逐位一致，缓存无损压缩且只留一代（4000 日 0.75 GB）；组合优化 `portopt`（Clarabel，Rust 与 Python 两条路径）：约束写在 K 维因子暴露上（N=5000 单次 QP 368→186 ms）、换手、跟踪误差锥、冻结股票自动放宽个股界并逐只记录、只有明确不可行才放松约束、数值失败不动约束。两轮审查：扰动未来输入 5064 个日模型 0 变化；独立 KKT 证书通过；75 处变异全部抓到。遗留：尚未接真实行情（停牌缺价和复牌收益归属、引擎回传真实持仓两项对接事项记在 `docs/RISK_NOTES.md`）。
- 数据层第四轮：差异化分红保留公告值（`action_source = 5`），`verify` 改为独立实现。
- `panelops` 与因子层合入主干：68 个面板算子、表达式求值与缓存、首批因子与因子检验。
- `panelbt` 步骤 1c 合入主干：股指期货对冲腿并入现货主循环，输入检查挪进 Rust 内核，新增 `validate` 开关。
- 端到端剖析工具 `bench/e2e`（`make perf-profile`、`make perf-compare`）和第一份基线。

## 2026-11-13
- 性能优化一轮合入主干（13 项，逐项记录见 `docs/PERF_LOG.md`）：一次完整研究 189.6 → 75.9 s，改用新增的批量和并行入口 66.5 s。新增 `Evaluator.evaluate_many`、`panelops.rank_ic_many`、`optimize_sequence(parallel=True)`、`FundamentalConfig.specific_incremental`；风险模型缓存格式升到 5、因子缓存格式升到 2（旧缓存自动重算）。
- 面板新增 `can_buy_open`、`can_sell_open` 两个字段（共 48 个）。
- 开源：许可证 Apache-2.0，新增根目录 README（中英文，附对比数字和图）。
- 回测报告 `ashare.report`：`build_tearsheet` 从一次 `panelbt` 回测算出报告的全部序列和指标（口径见 `docs/REPORT_NOTES.md`），`render_html` 生成自包含、离线可打开的 HTML（共用时间轴的轨道图、十字线联动、区间切换与回撤定位、净值图上直接标注最深几次回撤、月度热力图、分年度表；打印为 A4 纵向单栏）。`tearsheet()` 一行出报告，从数据层补齐名称、行业、行业换版日和基准。在真实面板上重跑设计稿的低波策略，`summary` 与旧导出逐项一致。
- 易用性（写教程时暴露的四处绕路）：
  - 信号日 → 成交日的错位有了唯一入口 `ashare.backtest.execution_targets`（稠密或稀疏目标；末尾信号默认丢掉，`tail="error"` 报错；成交日停牌交给引擎顺延）和 `execution_days`。
  - `ashare.backtest.market_from_panel(data, start, end, exec="open"|"close"|"vwap")` 一步建好行情、板块整手和按面板日期展开的费用；板块取证券表的板块列，认不出就报错。`panelbt.markets.ashare.infer_board` 认 `302` 开头的创业板代码和 `.SH/.SZ/.BJ` 后缀。
  - `ASHARE.costs` 的费率表补到 2005-01-24（印花税和过户费每次调整，依据见 `docs/ASHARE_FEES.md`，2015 年以前的过户费标「未核实」并取保守值），全历史回测不用再手填；新增 `FeeSchedule.components`，覆盖一种过户费写法时同方向另一种清零。改动前能用的写法结果逐位不变。
  - `ashare.risk.from_panel(data, start=...)` 把面板接到风险模型（字段改名、行业重编号），行业因子用申万一级名称（`industry_银行`，新增 `FundamentalConfig.industry_names`，不给名字时缓存键不变）；`PortfolioProblem` 的 `industry_factors`、`style_factors` 可按因子名给；导出 `ashare.risk.STYLE_NAMES`。
  - `bench/e2e` 改用这些入口，真实面板上与旧手写做法逐位相同（`bench/e2e/usability_equivalence.py`）。`DESIGN.md` 3、4.3、4.4、6.1、6.3、6.4 同步，组合优化的选型改成实际的「直接调 Clarabel，cvxpy 只作后备和参考」。
