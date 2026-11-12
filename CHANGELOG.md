# Changelog

## 2026-11-13（事件驱动）
- 新增 `ashare.events`（`docs/DESIGN.md` 6.6、`docs/EVENTS_NOTES.md`）：事件表与可知时点、四类内置事件提取（业绩预告、业绩快报、分红、指数成分调整）、事件转 `panelbt.Targets`、事件研究（三种异常收益口径；首选日历时间组合 Newey–West 检验，另并排朴素、按月聚类与自助法；窗口不完整与上市不足的事件剔除并计数）；两个真实例子 `python -m ashare.events.examples`，性能基线 `python -m ashare.events.bench`。错位、回测行情与费用用主干公共的 `execution_days`、`market_from_panel`。

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
- 境内期货数据（第 1 批）：六个交易所（中金所、上期所、大商所、郑商所、上期能源、广期所）的合约表、日线、主力映射、逐日结算参数，1995-04-17 起全部历史（`sync-futures`，按交易日分区、断点续跑）；与来源无关的统一字段（`futures_schema`，以后境外来源只需写适配器）；「交易日 × 合约」面板（按品种分块、每个合约只存存续窗口，`build-futures`）；读取入口 `ashare.data.open(root).futures()`，主力 / 次主力连续合约（来源映射、持仓量、成交量三种换月规则，不前视；不调整、差值、比例三种后复权）与换月日程表；`futures_panelbt.panelbt_inputs` 直接给出 `panelbt` 输入，中金所股指期货新旧两条路径回测逐位一致；独立校验 `verify-futures`。口径与已知问题见 `docs/FUTURES_DATA.md`。
- 境内期货数据复核返工（7 条 P1 与全部 P2）：合约乘数改为按日期取值（`multipliers` 表、面板 `multiplier` 字段；成交额反查 → 人工修正表 → 合约表，PB、FB 存续期中途改单位的合约分段，RU 2008 年前用修正表）；郑商所结算参数年份补错的 2 万多行按「当天上市且唯一」找回，对不上的计数进校验；`panelbt_inputs` 不再用以后的参数回填（缺值由 `missing` 显式给），区间内改过单位的合约默认报错、可选拆列；负手续费、整交易所整天为 0 的手续费、为 0 的价格置空；`verify-futures` 改为只认已知例外清单（逐条）、清单外不符一律 fail 并以非零状态退出，新增取值范围、引用完整性、单双边复发等检查；期货日分区加刷新保护；期货数据集不再并入股票的 `DATASETS`，股票的校验和内容版本不受期货条目影响；SCTAS（原油 TAS 指令）不进统一字段。
- 境内期货数据增量复核收尾：已知例外清单去掉由行情算出的比值，键集合按白名单测试；前结算例外改归正确原因（长假、上一日零成交、原因未查明），「原因未查明」的条目在校验报告和命令行里单独计数；`panelbt_inputs` 拆列时返回 `splits`（同一合约的前后两列与分界日）；补四个变异漏网的测试；`build-futures` 原始日线只读一次（中位 14.2 → 12.7 秒，峰值内存不变，输出逐位相同）。
- `panelbt` 声明式退出规则和限价目标（DESIGN 4.13、SPEC「Declarative exit rules and limit targets」）：`Market` 可选 `open`/`high`/`low`；`Config(rules=Rules(...))` 支持止损、止盈、移动止损（标量、每只证券或 T×N）、最长持有期、限价目标，在内核里执行，`run`、`run_batch`、`Simulator` 通用；日内先后不明的情形一律按不利一侧处理；受限顺延、卖出锁定、重新买入的冷却都照常生效；限价单默认要求当天价格穿过限价才成交（`limit_fill="cross"`，「碰到即成交」是显式选项），依据是复核在沪深 300 成分上的实测；规则计数进 `Result.diagnostics`，成交明细带 `TradeReason`。不配规则时由编译期开关去掉全部规则代码，结果与改动前逐位相同。纯 Python 参考实现对拍、手算例子三方核对、性质测试、变异检查；两个真实数据例子 `bench/rules_examples.py`（接 `ashare.backtest`，带冒烟测试）。期货列暂不支持规则（输入检查拒绝）。
- `panelbt` 逐日步进接口 `bt.Simulator`（DESIGN 4.12）：内核逐日循环拆成开盘前和成交收盘两段，`bt.run` 就是逐行调用这两段（结果逐位不变、不变慢）；`state` 只含成交前已知的信息（持仓、成本、建仓日、现金、权益、顺延订单），数组为零拷贝视图、底层缓冲区只读；`snapshot`/`restore` 存档恢复（带 CRC-32 校验，指纹含全部规则参数）；`step` 释放 GIL。每步约 4.5 µs，与证券数无关（backtrader 约 29 µs/根）。
- 归因与容量（`ashare.analysis`，`docs/DESIGN.md` 6.5、`docs/ANALYSIS_NOTES.md`）：基于风险模型的收益归因（市场 / 行业 / 风格 / 特质 / 交易 / 成本，GRAP 或 Carino 多期链接）、事前风险拆分与偏差统计、Brinson 行业归因（BF / BHB）、按冲击模型的容量与成本（解析模拟与 `panelbt` 实跑对账）、按年的成本拆解；结果都能转成列式 dict。
- 归因与容量返工：停牌中除权的收益改为后复权价整体沿用（原先在除权日记假收益、复牌日反向，沪深 300 基准 2015 年起累计被高估约 1.3 个百分点）；持仓估值改用 panelbt 的除权参考价；成本率加上红利税和融资利息。容量改为取成本曲线上升段的交点（原先报搜索下限 1 万元，真实例子现为约 7.15 千万元），另报最小可行规模，边界情况给 NaN/inf 加状态。未覆盖基准成分超过阈值时告警；诊断加停牌持仓在特质项里的部分；偏差区间改用块自助法。行情、账户规则、费用表、信号错位改用主干入口。
- 归因与容量收尾：解析模拟一次算多个规模（`simulate_many`），交点改用成批多分法，真实例子一次容量求解 15.7 s → 3.1 s，结果相对差 ≤ 2e-7；容量表里被最低佣金打穿的规模记 NaN 并加 `wiped_out` 列；诊断加分红口径差 `dividend_convention`。
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
- 新增 `ashare.ml`（DESIGN 6.9，说明与实测在 `docs/ML_NOTES.md`）：因子当特征 → 滚动训练 → 预测值当新因子。按回测成交约定的标签（与 `execution_days`、`market_from_panel` 同一口径）、`panelops` 截面预处理、只存股票池内样本的 float32 列存磁盘映射特征库、由标签跨度推出且只能加大的切分间隔、模型协议与注册表（岭回归、Lasso、ElasticNet、LightGBM 回归与排序、集成；第三方和以后 `nn` 组的模型登记名字即可用）、按逐日 Rank IC 提前停止、JSONL 试验账本与运行缓存、网格/随机搜索与 Bonferroni 折算、特征与置换重要性。可选依赖组 `ml`（LightGBM、scikit-learn），没装时 `test/ml` 跳过。
- 回测报告的文本版与补充内容：`print(ts)` 打印一屏摘要，`ts.text(sections, fmt="text"|"markdown"|"json", lang, start, end)` 按小节取，`save_json` / `load` 与命令行 `python -m ashare.report show|html`；自选区间按页面「切换区间」的公式重算（与页面 JS 对拍）；`warnings` 小节写出看图才看得出的问题。数据层新增对比指数（`benchmarks=`）、胜率、个股贡献、收益分布与尾部、集中度，以及多策略对比 `ashare.report.compare`；原有键和数值逐位不变，页面不嵌入新内容（HTML 逐字节不变），新内容的页面版式先出设计稿。复核后补充：提醒按自选区间重算（只能按全区间给的标明「全区间」），新增「换手或成本很高」一条（年化成本占平均权益超过 3%）；价格指数基准只在区间内收到过分红时才说超额偏高；`--from/--to` 校验日期、坏文件和形状不对的报告数据给中文报错（退出码 2）；摘要行数按折行后的实际行数守住 40 行，命令片段不折断；JSON 摘要加日超额算术年化，与信息比率反号时文本说明口径；对比表的星号不再挤歪数字列；各提醒阈值补「恰好等于 / 差一个最小量」的测试。另修 `test/risk` 一个依赖线程时序的测试（改用闸门等待）。
