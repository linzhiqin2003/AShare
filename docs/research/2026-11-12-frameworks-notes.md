> 说明：这是 2026-11-12 选型时联网调研的原始笔记（框架部分），由调研子代理撰写，每条标了【读到】【推断】【未核实】并附来源。
> 结论和本机实测结果见 `docs/SELECTION.md`；两者不一致时以 SELECTION 为准。笔记里提到的临时目录没有保留。

# A 股量化研究框架选型调研（核实日期 2026-11-12）

目标环境：Python 3.13、pandas ≥ 3.1、numpy ≥ 2.5、uv。
环节编号：(i) 单因子检验 (ii) 多因子合成/ML (iii) 带 A 股规则的组合回测 (iv) 组合优化 (v) Barra 风格风险模型 (vi) 绩效报告 (vii) 实盘 QMT/PTrade。

标注约定：
- 【读到】= 直接从 PyPI JSON / GitHub 页面 / 官方文档读到的内容，附链接。
- 【推断】= 根据读到的内容做的推理，没有页面直接这么写。
- 【未核实】= 没查到或来源矛盾。

---

## 0. PyPI 元数据快照（2026-11-12 从 `https://pypi.org/pypi/<包名>/json` 读取）

全部为【读到】。来源链接格式：`https://pypi.org/project/<包名>/`。

| 包 | 最新版 | 发布日 | requires_python | 分类器里的 Python | 许可字段 | 关键依赖声明 |
|---|---|---|---|---|---|---|
| pyqlib | 0.9.7 | 2025-08-15 | >=3.8.0 | 3.8–3.12 | MIT（分类器） | numpy、pandas>=0.24、lightgbm、cvxpy；`rl` extra 钉 numpy<2.0.0 |
| rdagent | 1.0.0 | 2026-09-23 | >=3.10 | 3.10、3.11 | MIT（分类器） | numpy、pandas、scikit-learn |
| vnpy | 4.5.0 | 2026-10-06 | >=3.10 | 3.10–3.13 | MIT | numpy>=2.2.3、pandas>=2.2.3、ta-lib>=0.6.4；`alpha` extra：lightgbm>=4.6、polars>=1.26、scikit-learn、scipy、torch |
| hikyuu | 3.0.2 | 2026-11-06 | 未声明 | 3.10–3.13 | MIT | numpy>=2.0、pandas>=2.3.0、h5py>=3.14 |
| quantaxis | 1.10.19（稳定）/ 2.0.0.dev33（预发布） | 2021-05-12 / 2021-12-27 | 未声明 | 无 | MIT | pandas>=1.1.5、numba |
| rqalpha | 6.5.0 | 2026-10-28 | >=3.8 | 3.8–3.14 | 元数据写 Apache-2.0（实际条款见 rqalpha 一节） | numpy>=2.0（py>=3.12）、pandas>=1.0.5,<4.0.0、h5py |
| backtrader | 1.9.78.123 | 2023-04-19 | 未声明 | 3.2–3.7 | GPLv3+ | 无 |
| vectorbt | 1.2.1 | 2026-11-11 | >=3.11,<3.15 | 3.11–3.14 | 元数据未填（见 vectorbt 一节） | numpy>=2.4.6、pandas>=3.0.3,<4.0、numba>=0.66 |
| zipline-reloaded | 3.2.1 | 2026-10-25 | >=3.11,<3.15 | 3.11–3.14 | Apache-2.0 | **pandas<3.0**、numpy>=2.1（py>=3.13）、bcolz-zipline、h5py |
| lib-pybroker | 2.0.2 | 2026-10-20 | >=3.11 | 无 | "Apache License 2.0 with Commons Clause"，分类器 "Free for non-commercial use" | numba>=0.64、numpy>=1.26.4,<3、pandas>=2.2.0,<4 |
| bt | 1.4.1 | 2026-10-10 | >=3.9 | 3.9–3.13 | MIT | （运行依赖经 ffn）ffn 1.4.1：numpy、pandas>=0.19 |
| alphalens-reloaded | 0.5.1 | 2026-10-26 | >=3.11,<3.16 | 3.11–3.15 | Apache-2.0 | pandas>=1.5.0,<4.0、numpy>=1.26（py>=3.12）、empyrical-reloaded>=0.5.7 |
| alphainspect | 0.5.10 | 2026-10-14 | >=3.8 | 无 | MIT | pandas、polars-ta、scipy |
| polars-ta | 0.5.18 | 2026-10-20 | >=3.8 | 无 | MIT | polars>=1.28、polars-ols、numba、numpy、pandas |
| expr-codegen | 0.17.0 | 2026-10-22 | >=3.9 | 无 | BSD-3 | — |
| quantstats | 0.0.86 | 2026-09-27 | >=3.10 | 3.10–3.13 | Apache-2.0 | numpy>=1.24、pandas>=2.0、scipy>=1.11 |
| pyfolio-reloaded | 0.10.2 | 2026-11-02 | >=3.11,<3.16 | 3.11–3.15 | Apache-2.0 | **pandas<3.2**、numpy<3、empyrical-reloaded>=0.5.11 |
| empyrical-reloaded | 0.7.2 | 2026-11-02 | >=3.11 | 3.11–3.15 | Apache-2.0 | pandas>=2.2.2（py>=3.12）、numpy>=1.26 |
| cvxpy | 1.9.3 | 2026-09-19 | >=3.11 | 无 | Apache-2.0 | numpy>=2.0.0、scipy>=1.13、clarabel |
| riskfolio-lib | 7.4.1 | 2026-11-04 | >=3.10 | 3.10–3.14 | BSD-3 | numpy>=1.26.4、pandas>=2.2.2、cvxpy>=1.7.2、arch>=7.2 |
| pyportfolioopt | 1.6.0 | 2026-02-26 | 未声明 | 3.10–3.14 | MIT | numpy>=1.26,<3、pandas>=1.0,<4、cvxpy>=1.1.19 |
| skfolio（新增候选） | 1.12.9 | 2026-11-10 | >=3.10 | 3.10–3.14 | BSD-3 | numpy>=1.24、pandas>=2.1、cvxpy-base>=1.5、scikit-learn>=1.6 |
| xtquant | 250807.1.2 | 2026-07-21 | >=3.6 | 3.6–3.13 | MIT（分类器） | 无 |
| vnpy-xt | 1.4.6 | 2025-10-18 | >=3.10 | 3.10–3.13 | MIT | — |
| wtpy（顺带查） | 0.9.9.3 | 2023-12-31 | >=3.8 | 无 | MIT | **pandas==1.3.5 钉死**，不适用，后面不再展开 |

对目标环境的直接含义（【推断】，依据上表依赖上限）：
- `zipline-reloaded 3.2.1` 声明 `pandas<3.0`，与 pandas ≥ 3.1 **不能共存**。
- `pyfolio-reloaded 0.10.2` 声明 `pandas<3.2`，pandas 3.1.x 可以装，3.2 起会被解析器拒绝。
- 其余有 pandas 上限的包上限都是 `<4.0`，声明层面不冲突；没写上限的（pyqlib、rdagent、quantstats、backtrader 等）声明层面不拦，但实际是否兼容要看代码，逐个在下文核实。


---

## 1. 一体化框架

### 1.1 Qlib（pyqlib，microsoft/qlib）

1. 维护状态
   - 【读到】PyPI 最新 0.9.7（2025-08-15），wheel 只有 cp38–cp312，**没有 cp313 wheel**（Qlib 带 Cython 扩展，3.13 只能源码编译）。https://pypi.org/project/pyqlib/#files
   - 【读到】GitHub 有更新的 v0.9.8（2026-10-08），发布说明写的是安全加固（限制加载可执行的 recorder 产物、表达式改为受限解析器等），并写明 "This GitHub release provides source archives. Check PyPI separately for package availability"；截至 2026-11-12 PyPI 上没有 0.9.8。https://github.com/microsoft/qlib/releases
   - 【读到】main 分支近半年有提交但不密：2026-10-08、09-16、07-23、04 月若干。https://github.com/microsoft/qlib/commits/main
   - 【推断】处于「低速维护」：安全修复和小 bug 修复为主，大量社区 PR（包括 pandas 3 修复）挂着没合并；README 首屏在推 RD-Agent。没有「停止维护」的声明。
2. 许可：【读到】MIT。https://github.com/microsoft/qlib/blob/main/pyproject.toml
3. Python / pandas / numpy
   - 【读到】pyproject 分类器与 README 支持表都只到 Python 3.12；`requires-python >=3.8.0`；依赖 `pandas>=1.1`、`numpy` 无上限；`rl` extra 钉 `numpy<2.0.0`；另有 `mlflow<3.13`、`gym`（已停更的旧包）等。https://github.com/microsoft/qlib/blob/main/pyproject.toml
   - 【读到】PR #2360「fix: make datetime_groupby_apply work with string functions on pandas 3」（2026-09-28）仍是 open；issue #2018「Backtest Module Crash Due to Gym Dependency and NumPy 2.0+ Incompatibility」（2025-09-23）仍是 open。https://github.com/microsoft/qlib/pulls?q=is%3Apr+pandas+3 、https://github.com/microsoft/qlib/issues/2018
   - 【推断】Python 3.13 + pandas ≥ 3.1 + numpy ≥ 2.5 不在官方支持范围内，大概率要自己编译并打补丁。需本机实测。
4. 覆盖环节：(ii) 强（Alpha158/360 特征、LightGBM 等模型、滚动训练 workflow）；(i) 有（IC、分组收益图表在 `qlib.contrib.report`，需 `analysis` extra，且钉 `plotly<7`）；(iii) 有（日频撮合）；(iv) 有 `EnhancedIndexingOptimizer`（cvxpy，目标 = 超额收益 − λ·风险，约束含换手上限 delta、个股偏离 b_dev、因子暴露偏离 f_dev）；(v) 只有统计风险模型 `StructuredCovEstimator`（PCA / 因子分析），源码注释明说「we use latent factor models」，**不是 Barra 基本面因子模型**；(vi) 有 `risk_analysis`；(vii) 无。
   - 来源：https://github.com/microsoft/qlib/blob/main/qlib/contrib/strategy/optimizer/enhanced_indexing.py 、https://github.com/microsoft/qlib/blob/main/qlib/model/riskmodel/structured.py 、https://github.com/microsoft/qlib/blob/main/qlib/contrib/evaluate.py
5. A 股规则
   - 【读到】`REG_CN` 默认 `trade_unit=100`、`limit_threshold=0.095`、`deal_price=close`；Exchange 默认 `open_cost=0.0015`、`close_cost=0.0025`、`min_cost=5.0`；`$close` 为空视为停牌；`limit_threshold` 可以是浮点数，也可以是一对表达式（自定义买入/卖出受限条件）。https://github.com/microsoft/qlib/blob/main/qlib/config.py 、https://github.com/microsoft/qlib/blob/main/qlib/backtest/exchange.py
   - 【推断】单一 0.095 阈值不区分主板 10% / 创业板科创板 20% / ST 5% / 北交所 30%，要自己用表达式对提供每只股票的涨跌停标记；分红送转靠复权价 + `$factor` 处理，没有现金分红入账和送股到账的账户级模拟；没有显式的 T+1 可卖数量（日频调仓下影响小）。印花税只能折进 `close_cost`，不单列。
6. 数据格式：【读到】自有 `.bin` 格式；官方脚本 `scripts/dump_bin.py` 可把 CSV 或 Parquet 转成 `.bin`；`StaticDataLoader` 可以直接吃 DataFrame/文件做特征（绕过表达式引擎）。官方数据集因「data security policy」停供，README 指向社区数据 chenditc/investment_data。https://github.com/microsoft/qlib/blob/main/docs/component/data.rst 、https://github.com/microsoft/qlib/blob/main/README.md
   - 【推断】回测 Exchange 仍从 Qlib 数据提供者取行情（`$close/$factor/$change/$volume`），所以要用它的回测就绕不开 `.bin`。
7. 学习成本和坑：配置驱动（YAML + `qrun`）、抽象层多（DataHandler/Dataset/Recorder/Executor）；依赖重（mlflow、redis、pymongo、gym、jupyter 全是必装）；v0.9.8 起加载旧 recorder 产物、本地 .py 模块要显式 `trusted=True`（【读到】发布说明）。

### 1.2 RD-Agent(Q)（rdagent，microsoft/RD-Agent）

1. 维护状态：【读到】PyPI 1.0.0（2026-09-23），此前 0.8.0（2025-11-03）；master 近半年有提交（2026-05 到 09-23 之间多次，09-23 之后 Atom 提要里没有新提交）。https://pypi.org/project/rdagent/ 、https://github.com/microsoft/RD-Agent/commits
2. 许可：【读到】MIT（pyproject 分类器）。https://github.com/microsoft/RD-Agent/blob/main/pyproject.toml
3. Python：【读到】`requires-python >=3.10`，分类器只列 3.10、3.11；README 写「RD-Agent currently only supports Linux」，安装示例 `conda create -n rdagent python=3.10`，并要求先装好 Docker（多数场景在容器里跑）。https://github.com/microsoft/RD-Agent/blob/main/README.md
   - 【推断】它不是装进研究环境的库，而是一个在 Linux + Docker 里跑、调用 LLM API 的自动化代理；量化场景（`rdagent fin_quant / fin_factor / fin_model`）在容器内用 Qlib 做回测，所以与本项目的 Python 3.13/pandas 3 环境基本无交集，macOS 本机不在支持范围。
4. 覆盖环节：(ii) 的「自动提因子/提模型」这一步（LLM 生成因子代码和模型代码 → Qlib 回测 → 反馈迭代）。(i)(iii)(vi) 借 Qlib；(iv)(v)(vii) 无。【读到】README 的场景列表。
5. A 股规则：同 Qlib（它用 Qlib 回测）。【推断】
6. 数据格式：同 Qlib 的 `.bin`。【推断】
7. 学习成本和坑：需要 LLM API key（默认 LiteLLM 后端）和 embedding 模型；README 称论文实验成本「under $10」，实际花费取决于循环次数和模型，【未核实】；生成并执行代码，README 自己提醒要在隔离、最小权限环境里跑。

### 1.3 vnpy 4.x 的 `vnpy.alpha`（vnpy/vnpy）

1. 维护状态：【读到】PyPI 4.5.0（2026-10-06），此前 4.4.0（2026-05-14）、4.3.0（2025-12-24）；master 2026-10-04～10-06 有 20 多次提交。https://pypi.org/project/vnpy/ 、https://github.com/vnpy/vnpy/commits
2. 许可：【读到】MIT。https://github.com/vnpy/vnpy/blob/master/pyproject.toml
3. Python / pandas / numpy：【读到】`requires-python >=3.10`，分类器到 3.13，README 写「推荐使用 Python 3.13」；核心依赖 `numpy>=2.2.3`、`pandas>=2.2.3`（无上限）；`alpha` extra：`polars>=1.26`、`lightgbm>=4.6`、`scikit-learn`、`scipy`、`torch>=2.6`、`pyarrow`。https://github.com/vnpy/vnpy/blob/master/README.md
   - 【读到】核心包的必装依赖里有 `pyside6==6.8.2.1`（钉死）、`pyqtgraph`、`ta-lib>=0.6.4`、`pyzmq`：只想用 `vnpy.alpha` 也得装整套 GUI 依赖。https://pypi.org/pypi/vnpy/json
   - 【推断】`vnpy.alpha` 内部用 polars，pandas 3 的影响面小；是否有 pandas 3 的问题【未核实】。
4. 覆盖环节：(ii) 是主体：`dataset`（表达式引擎 + Alpha 158）、`model`（Lasso / LightGBM / MLP）、`lab`（数据、模型、信号的存取管理）；(iii) 有一个截面组合回测引擎 `vnpy/alpha/strategy/backtesting.py`；(vi) 有内置统计和 plotly 图（收益、超额、换手、回撤）；(vii) vnpy 本体是实盘交易平台，但 `vnpy_xt` 只是迅投研的行情/数据接口，不是 miniQMT 下单通道（见 7.3 节）；A 股下单走中泰 XTP 等柜台接口。(i) 有：`AlphaDataset.show_feature_performance(name)` / `show_signal_performance(signal)` 调用自带的 `vnpy/alpha/dataset/factor_performance.py`（alphalens 式：10 分位分层收益、Spearman IC、分位换手、因子秩自相关、alpha/beta，plotly 出图；内部转成 pandas 计算），但只针对已加进 dataset 的单个特征/信号，没有因子间相关性矩阵；(iv)(v) 无。【读到】README 模块清单和源码 https://github.com/vnpy/vnpy/blob/master/vnpy/alpha/dataset/factor_performance.py
   - 【读到】`factor_performance.py` 里 `import statsmodels`，但 pyproject 的 `alpha` extra（polars、scipy、scikit-learn、lightgbm、torch、pyarrow）里没有声明 statsmodels。【推断】要自己补装。
5. A 股规则（读源码 `vnpy/alpha/strategy/backtesting.py`，https://github.com/vnpy/vnpy/blob/master/vnpy/alpha/strategy/backtesting.py）
   - 【读到】涨跌停：写死 `pre_close * 1.1` / `pre_close * 0.9`，只在「全天一字涨停不能买、全天一字跌停不能卖」时拦截。
   - 【读到】费用：逐合约的 `long_rate` / `short_rate` 比例费率，没有最低佣金参数。
   - 【读到】成交价：`fill_price` 只能是 `open` 或 `vwap`。
   - 【推断】缺 K 线（停牌）时用上一根收盘价补一根 bar 再撮合，代码里没看到按成交量为 0 拒单的判断，停牌股可能被撮合成交；20%/5%/30% 涨跌幅板块不区分；没有现金分红、送转的账户处理（要自己喂复权价）；没有最低 5 元佣金；印花税只能并进 `short_rate`。T+1：日频下单在下一根 bar 撮合，等效满足，但没有「当日买入不可卖」的持仓冻结逻辑。
6. 数据格式：【读到】`AlphaLab` 把 K 线按「每个合约一个 Parquet」存在本地目录（polars 读写），成分股用 `shelve`，合约参数用 `contract.json`，信号也是 Parquet。https://github.com/vnpy/vnpy/blob/master/vnpy/alpha/lab.py
   - 【推断】无专有二进制锁定，自己按它的列结构写 Parquet 即可接入，这点比 Qlib 轻。
7. 学习成本和坑：vnpy 是事件驱动交易平台，`alpha` 是 2025 年 4.0 才加的模块，README 自述设计「受 Qlib 启发」；示例数据下载依赖 RQData 或迅投研（都是付费数据服务）；回测引擎对 A 股规则的处理比 rqalpha 粗。

### 1.4 hikyuu（fasiondog/hikyuu）

1. 维护状态：【读到】非常活跃。PyPI 3.0.2（2026-11-06），3.0.0 于 2026-10-27 发布，此前 2.8.3（2026-09-30）；master 在 2026-11-04～11-11 每天都有提交。https://pypi.org/project/hikyuu/ 、https://github.com/fasiondog/hikyuu/commits
   - 【推断】3.0 是两周前才出的大版本，API 和文档（README 已整体改写为英文版 + `readme.zh.md`）刚变过，短期内还会有修补版本。
2. 许可：【读到】PyPI 元数据和分类器写 MIT；仓库默认分支的 `LICENSE` 文件是 Apache License 2.0。两处不一致，都是宽松许可。https://pypi.org/project/hikyuu/ 、https://github.com/fasiondog/hikyuu/blob/master/LICENSE
   - 【读到】有捐赠/订阅档位（¥30～¥300/年），附赠历史数据和「捐赠权益」；ClickHouse 存储「经插件扩展」。https://github.com/fasiondog/hikyuu/blob/master/readme.zh.md
   - 【读到】文档「捐赠用户功能」页的目录列出：扩展指标、**事件驱动式回测（`backtest()` 接口）**、dataserver 数据缓存服务、shm 共享内存数据服务、动态周期 K 线。即这几项是捐赠用户专属；页面只有目录，具体授权方式和设备数【未核实】。https://hikyuu.readthedocs.io/zh-cn/latest/vip/index.html
3. Python / pandas / numpy：【读到】README 写「official packages are built for Python 3.10 – 3.13 only」；wheel 为 `py3-none-<平台>`（含 macOS arm64、manylinux x86_64/aarch64、win_amd64）；依赖 `numpy>=2.0`、`pandas>=2.3.0`、`h5py>=3.14.0`，无上限。pandas 3 的实际兼容【未核实】。https://pypi.org/project/hikyuu/#files
4. 覆盖环节：(iii) 是主体（C++ 内核的事件/系统化回测，组合层 `PF_*`、选股 `SE`、资金分配 `AF`、多因子打分 `MF`）；(i) 有 `MultiFactor`（因子评分排序，IC 等）；(vi) 有绩效统计；(vii) 有 `OrderBroker` 下单接口槽位，README 声明「不提供、不内置证券交易服务」，文档列出的内建订单代理只有 `TestOrderBroker` 和 `MailOrderBroker`（https://hikyuu.readthedocs.io/zh-cn/latest/trade_manage/trade_manage.html ），没有 QMT；对接 QMT 要自己写 `OrderBrokerBase` 子类。(ii) 机器学习滚动训练、(iv) 带约束的组合优化、(v) 风险模型：README 模块表里没有。
5. A 股规则：【读到】README「Trading rules」一段：默认下一根 bar 开盘成交（`buy_delay`/`sell_delay`）；一字板（high == low）或停牌时顺延，超过 `max_delay_count` 放弃；成本按 `CostRecord` 分项（佣金 / 印花税 / 过户费），买卖分开算；`updateWithWeight` 在除权除息日调整持仓和现金。内置数据导入只覆盖 A 股。https://github.com/fasiondog/hikyuu/blob/master/readme.md
   - 【推断】在候选里它和 rqalpha 是对 A 股规则（含分红送转入账）建模最完整的两个。【读到】文档目录有「沪深 A 股交易成本算法」`TC_FixedA()`、`TC_FixedA2015()`、`TC_FixedA2017()`、`TC_FixedAPre2015()`（按年份区分规则）和 `TradeManager.update_with_weight()`；最低 5 元佣金、整手的具体参数页面（`trade_cost.html`）没有打开，【未核实】。
6. 数据格式：【读到】存储后端 HDF5（默认）/ MySQL / ClickHouse（插件）/ SQLite；用自带的 `HikyuuTDX` / `importdata` 从通达信导入 A 股数据。https://github.com/fasiondog/hikyuu/blob/master/readme.md
   - 【推断】行情要先进它自己的库（HDF5 等）才能被 C++ 内核读，不能直接把 Parquet/DataFrame 当行情源；因子值可以从外部算好再喂进去的接口细节【未核实】。属于中等程度的格式锁定。
7. 学习成本和坑：自成体系的部件命名（EV/CN/SG/MM/ST/TP/PG/SP/SE/AF/MF/TM），概念多；核心在 C++，出问题难以在 Python 侧调试；3.0 刚发布。

### 1.5 QUANTAXIS（yutiansut/QUANTAXIS）

1. 维护状态
   - 【读到】PyPI 稳定版停在 1.10.19（2021-05-12），最后的预发布 2.0.0.dev33（2021-12-27）。https://pypi.org/project/quantaxis/#history
   - 【读到】GitHub README 标题是「QUANTAXIS 2.1.0-alpha2」，写「更新日期 2025-10-25」「Python 3.9-3.12」，安装方式是 `pip install -e .`（源码安装）；GitHub Releases 最后一条是 2022-05。master 提交集中在 2025-10-25/26（一次性大改），之后只有 2026-02、2026-09 各几次。https://github.com/yutiansut/QUANTAXIS
   - 【推断】主线已转向作者的 Rust 项目（QARS 等），Python 包长期没有正式发版，属于「没宣布停更但实际接近停滞」。不建议作为新项目的基础。
2. 许可：【读到】MIT。
3. Python / pandas / numpy：【读到】README 声明 Python 3.9–3.12、pandas 2.0+；没有声明 3.13 和 pandas 3。https://github.com/yutiansut/QUANTAXIS/blob/master/README.md
4. 覆盖环节：【读到】README 列了 QAFactor（单因子研究入库、因子管理/测试/合并）、QAStrategy（CTA/套利回测）、QIFI 账户体系。(iv)(v) 无。各模块在 2.1 alpha 里的可用程度【未核实】。
5. A 股规则：【未核实】（没读到 2.1 的回测规则文档）。
6. 数据格式：【读到】「支持 MongoDB / ClickHouse 存储」「MongoDB 友好」——数据层绑定数据库服务，不是直接吃 Parquet。
7. 坑：依赖 60+，要起 MongoDB；PyPI 版本是 2021 年的旧代码。

---

## 2. 回测引擎

### 2.1 rqalpha（ricequant/rqalpha）

1. 维护状态：【读到】活跃。PyPI 6.5.0（2026-10-28），2026-07 以来发了 6.3.0、6.4.0、6.4.1、6.4.2、6.5.0；master 在 2026-10 几乎每个工作日都有提交。https://pypi.org/project/rqalpha/#history 、https://github.com/ricequant/rqalpha/commits
2. 许可（重点）：【读到】`LICENSE` 原文——「非商业用途（个人出于非商业目的，或高校、研究所等非营利机构出于教育、科研目的）：遵守 Apache License 2.0」；「商业用途（个人出于任何商业目的，或者**法人或其他组织出于任何目的**）：未经米筐科技授权…不得使用」。README 也写「仅限非商业使用」。PyPI 元数据里的 `Apache-2.0` 不完整。https://github.com/ricequant/rqalpha/blob/master/LICENSE
   - 【推断】个人自己研究、自己账户交易属于「个人非商业用途」可以用；一旦以公司名义使用、对外提供策略/信号服务或做成产品，就需要米筐授权。「用自己的钱实盘赚钱」算不算商业目的，许可文本没有明说，【未核实】，介意的话要问米筐。
3. Python / pandas / numpy：【读到】pyproject 分类器 3.8–3.14；`numpy>=2.0.0`（py>=3.12）；`pandas>=1.0.5,<4.0.0`；CHANGELOG 6.4.2 明写「支持 pandas 3.0: 依赖范围放宽至 <4.0.0，适配索引访问、字符串类型及已移除的配置项」。https://github.com/ricequant/rqalpha/blob/master/pyproject.toml 、https://github.com/ricequant/rqalpha/blob/master/CHANGELOG.rst
   - 【推断】是候选回测引擎里唯一一个「A 股规则齐全 + 官方声明支持 Python 3.13 和 pandas 3」的。
4. 覆盖环节：(iii) 强；(vi) 有（`sys_analyser` mod 输出逐日下单/成交/持仓、风险指标、csv 和图；指标计算用米筐的 `rqrisk`）；(vii) 有 `AbstractBroker` 接口，但开源部分没有 QMT/PTrade 实现（【未核实】有无可用的社区 mod）。(i)(ii)(iv)(v) 不在开源 rqalpha 里——米筐的因子检验、组合优化器（rqoptimizer）、风险因子数据（README：RQData 提供「风格因子暴露度、收益率、协方差和特异风险」）是付费产品。https://github.com/ricequant/rqalpha/blob/master/README.rst
5. A 股规则（读各系统 mod 的默认配置）
   - 【读到】`sys_accounts`：`stock_t1: True`、`dividend_reinvestment`、`dividend_tax_enabled`/`dividend_tax_rate`、`cash_return_by_stock_delisted: True`、`validate_stock_position`（禁止卖空）。https://github.com/ricequant/rqalpha/blob/master/rqalpha/mod/rqalpha_mod_sys_accounts/__init__.py
   - 【读到】`sys_transaction_cost`：`stock_min_commission: 5`、`cn_stock_min_commission`、`stock_commission_multiplier`、`tax_multiplier`、`pit_tax`（按历史时点的印花税率）。https://github.com/ricequant/rqalpha/blob/master/rqalpha/mod/rqalpha_mod_sys_transaction_cost/__init__.py
   - 【读到】`sys_simulation`：`price_limit: True`（涨跌停限制）、`volume_limit: True` + `volume_percent: 0.25`、`inactive_limit: True`（成交量为 0 不撮合）、撮合方式 `current_bar / next_bar / vwap` 等、滑点模型。`sys_risk`：`validate_is_trading`（停牌/未上市校验）、`validate_price`、`validate_cash`。https://github.com/ricequant/rqalpha/blob/master/rqalpha/mod/rqalpha_mod_sys_simulation/__init__.py 、https://github.com/ricequant/rqalpha/blob/master/rqalpha/mod/rqalpha_mod_sys_risk/__init__.py
   - 【读到】下单 API 有 `order_lots`（按一手取整）、`order_target_portfolio`（一次给出目标权重组合）、`is_suspended`、`is_st_stock`、`get_dividend`；数据源接口含 `get_dividend`、`get_split`。https://github.com/ricequant/rqalpha/blob/master/rqalpha/mod/rqalpha_mod_sys_accounts/api/api_stock.py
   - 结论【推断】：T+1、涨跌停、停牌、印花税、最低 5 元佣金、分红送转、整手全部原生覆盖。
6. 数据格式：【读到】默认数据是 bundle（HDF5，依赖 `h5py`、`hdf5plugin`），`rqalpha download-bundle` 免费下载日线数据（股票、常用指数、场内基金、期货），「每个月月初更新」；每日更新要 `rqalpha update-bundle`，需要 RQDatac 许可（付费，有试用）。https://github.com/ricequant/rqalpha/blob/master/docs/source/intro/install.rst
   - 【读到】可替换数据源：实现 `AbstractDataSource`（`get_bar`、`history_bars`、`get_dividend`、`get_split`、`is_suspended`、`get_trading_calendars`、`available_data_range` 等）并通过 Mod 注入；策略里也可以直接读自己的文件。https://github.com/ricequant/rqalpha/blob/master/rqalpha/interface.py 、https://github.com/ricequant/rqalpha/blob/master/docs/source/development/data_source.rst
   - 【推断】想完全用自己的 Parquet 做行情源，要写一个自定义 DataSource mod（工作量中等：合约信息、交易日历、行情、分红拆股、停牌、ST 都得供给）；更省事的是行情用 bundle、因子/信号从自己的 Parquet 读。免费 bundle 月更，日更需付费或自己写数据源。
7. 学习成本和坑：事件驱动（`init / before_trading / handle_bar / after_trading`），全市场日频回测速度比向量化慢；6.4.1 改了 `DataProxy.history_bars` 的参数名、6.5.0 把 Event 机制扩成 Hook 机制（【读到】CHANGELOG）——近期核心在重构，自己写的 mod 可能要跟着改。

### 2.2 backtrader（mementum/backtrader）

1. 维护状态：【读到】停滞。PyPI 最后一版 1.9.78.123（2023-04-19）；master 最后提交 2023-04-19。https://pypi.org/project/backtrader/#history 、https://github.com/mementum/backtrader/commits
2. 许可：【读到】GPLv3+（PyPI 分类器）。自己用没问题；把它嵌进要分发的闭源软件有传染性。
3. Python / pandas / numpy：【读到】分类器只列 Python 3.2–3.7，README 写「Python >= 3.2」；纯 Python、无必装依赖。https://github.com/mementum/backtrader/blob/master/README.rst
   - Python 3.13、pandas 3、numpy 2 下能否正常跑（尤其 pandas 数据源和 matplotlib 绘图）【未核实】；上游已不修。
4. 覆盖环节：(iii) 通用事件驱动回测；自带若干 analyzer（(vi) 部分）。其余无。
5. A 股规则：不原生支持。【推断】T+1、涨跌停、停牌、最低佣金、整手都要自己写 `CommInfoBase` 子类、sizer 和下单过滤；分红送转只能喂复权价。社区有大量 A 股改造教程，但没有维护中的官方实现。
6. 数据格式：【读到】README：数据源支持 csv、在线源、pandas（`PandasData`）。无格式锁定。
7. 坑：全市场几千只股票的截面策略在它的逐 bar、逐 data 循环下很慢（【推断】，社区普遍反馈）；项目无人维护。

### 2.3 vectorbt 开源版（polakowo/vectorbt）

1. 维护状态：【读到】活跃，而且今年进入 1.x：PyPI 1.2.1（2026-11-11），1.2.0（10-19）、1.1.2（10-09）、1.1.0（07-05）；master 2026-11 多次提交。README 自述「VectorBT is the open-source community edition of VectorBT PRO」，1.x 增加了可选 Rust 引擎（`vectorbt[rust]`）。https://pypi.org/project/vectorbt/#history 、https://github.com/polakowo/vectorbt/blob/master/README.md
2. 许可（重点）：【读到】`LICENSE.md` 开头是 Commons Clause v1.0 条件 + Apache 2.0，「License: Apache 2.0 with Commons Clause」。Commons Clause 禁止「Sell」——即以收费方式向第三方提供「价值全部或实质上来自本软件功能」的产品或服务（含托管、咨询/支持服务）。README 称其为 fair-code。https://github.com/polakowo/vectorbt/blob/master/LICENSE.md
   - 【推断】个人研究、自己交易不受影响；不能把基于它的回测能力包装成收费服务/产品卖。
3. Python / pandas / numpy：【读到】`requires-python >=3.11,<3.15`；`numpy>=2.4.6`、`pandas>=3.0.3,<4.0`、`numba>=0.66`。https://github.com/polakowo/vectorbt/blob/master/pyproject.toml
   - 【推断】声明层面与目标环境（3.13 / pandas ≥ 3.1 / numpy ≥ 2.5）完全吻合，是候选里对新版本跟得最紧的。numba 对 numpy 2.5+ 的支持取决于 numba 版本，需本机实测。
4. 覆盖环节：(iii) 向量化组合回测（`Portfolio.from_orders / from_signals / from_order_func / from_holding`）；(vi) 有较完整的收益、回撤、交易统计和 plotly 图。(i) 无专门的 IC/分层工具；(ii)(iv)(v)(vii) 无。
5. A 股规则：不原生支持。【读到】`from_orders` 有 `size_granularity`（可设 100 实现整手）、`fees`、`fixed_fees`（每笔固定费用，不是「最低 5 元」）、`min_size`、`max_size`、`cash_sharing`、`call_seq`；源码 `portfolio/base.py` 里搜不到 `T+1`、`limit_up`、`cash_dividends`、`min_fees`。https://github.com/polakowo/vectorbt/blob/master/vectorbt/portfolio/base.py
   - 【推断】涨跌停/停牌可以在生成订单矩阵前自己把不可成交的格子置空来近似；最低佣金 `max(5, 费率×金额)` 和分红现金入账要用 `from_order_func`（numba 回调）自己写，或接受近似；更细的功能在付费的 VectorBT PRO 里。
6. 数据格式：直接吃 pandas DataFrame / numpy 宽表，无格式锁定。【读到】README 示例。
7. 学习成本和坑：宽表 + 广播的思维方式和事件驱动差别大；`from_order_func` 要写 numba 函数，报错难读；1.x 刚做过大升级（要求 pandas ≥ 3.0.3），老教程多基于 0.2x。

### 2.4 zipline-reloaded（stefan-jansen/zipline-reloaded）

1. 维护状态：【读到】2026-10 恢复发版：3.2.0（10-18）、3.2.1（10-25），此前 3.1.1（2025-07）；master 2026-10-22～11-02 有提交。https://github.com/stefan-jansen/zipline-reloaded/releases
2. 许可：【读到】Apache-2.0。
3. Python / pandas / numpy：【读到】`requires-python >=3.11,<3.15`；`numpy>=2.1`（py>=3.13）；**`pandas>=1.5.3,<3.0`**。3.2.0/3.2.1 发布说明原文：「Pandas 3 and Python 3.15 are not supported; pandas remains constrained below 3.0」「Pandas 3 and Python 3.15 remain excluded」。https://github.com/stefan-jansen/zipline-reloaded/blob/main/pyproject.toml
   - 结论：与 pandas ≥ 3.1 的目标环境**不兼容**，直接排除。
4. 覆盖环节：(iii) 事件驱动回测 + Pipeline 因子计算；配 alphalens/pyfolio 可覆盖 (i)(vi)。
5. A 股规则：不原生支持。【推断】面向美股（交易日历可换 XSHG，但 T+1、涨跌停、最低佣金都要自己写 slippage/commission/trading control）。
6. 数据格式：【读到】依赖 `bcolz-zipline`、`h5py`，数据要先 ingest 成 bundle（bcolz）。有格式锁定。
7. 坑：C 扩展多，安装和 ingest 是出名的门槛。

### 2.5 PyBroker（lib-pybroker，edtechre/pybroker）

1. 维护状态：【读到】活跃。PyPI 2.0.2（2026-10-20），2.0.0（08-17）；master 2026-08、10 月有提交。https://pypi.org/project/lib-pybroker/#history
2. 许可（重点）：【读到】`LICENSE` 为 Commons Clause v1.0 + Apache 2.0（「License: Apache 2.0 with Commons Clause」），PyPI 分类器写「Free for non-commercial use」。限制同 vectorbt：不能把它作为收费产品/服务的实质内容出售。https://github.com/edtechre/pybroker/blob/master/LICENSE
3. Python / pandas / numpy：【读到】`requires-python >=3.11`；PyPI 依赖 `numba>=0.64,<1`、`numpy>=1.26.4,<3`、`pandas>=2.2.0,<4`；仓库 `requirements.txt`（开发环境）写的是 `numpy>=2.5.2,<2.6`、`pandas>=3.0.5`、`numba>=0.67.0`，说明作者在 pandas 3 / numpy 2.5 上开发测试。https://github.com/edtechre/pybroker/blob/master/requirements.txt
4. 覆盖环节：(ii) 部分（自带 walkforward 训练/回测框架，模型自己提供）；(iii) 有（numba 加速的 bar 回测，支持按分数排序选股、最大持仓数）；(vi) 有（含 bootstrap 指标）。(i)(iv)(v)(vii) 无。【读到】README。
5. A 股规则：不原生支持。【读到】`StrategyConfig` 有 `fee_mode`（可传自定义函数 `Callable[[FeeInfo], Decimal]`，能实现最低 5 元和印花税）、`buy_delay`/`sell_delay`（默认 1 根 bar）、`enable_fractional_shares`、`max_long_positions`。README 列了 AKShare 数据源，有中文文档。https://github.com/edtechre/pybroker/blob/master/src/pybroker/config.py
   - 【推断】涨跌停、停牌、整手、分红要在策略函数里自己判断；它的模型是「信号 → 逐股下单」，不是「目标权重组合再平衡」，做指数增强的权重型组合不顺手。
6. 数据格式：【读到】可自定义数据源，直接给 DataFrame。无格式锁定。
7. 坑：面向规则/信号型策略；截面权重组合支持弱。

### 2.6 bt（pmorissette/bt）

1. 维护状态：【读到】活跃。PyPI 1.4.1（2026-10-10），2026-09～10 连发 1.2.x、1.3.0、1.4.0；master 2026-10-31～11-08 每天有提交。配套的 ffn 同步发到 1.4.1。https://pypi.org/project/bt/#history
2. 许可：【读到】MIT。
3. Python / pandas / numpy：【读到】`requires-python >=3.9`，分类器 3.9–3.13；依赖经 ffn（`numpy`、`pandas>=0.19`，无上限）；带 Cython 扩展。https://github.com/pmorissette/bt/blob/master/pyproject.toml
   - 【读到】Releases 页 v1.1.3 的条目「Support pandas 3, add regression testing for pandas 1 and 2」；v1.1.5 的条目「Drop Python 3.9, add Python 3.14」。后一条与 PyPI 上 1.4.1 的元数据（`>=3.9`、分类器 3.9–3.13）对不上，Python 支持范围的上下界【未核实】，但 3.13 在两处都在范围内。https://github.com/pmorissette/bt/releases
   - 【读到】近期行为变更较多：v1.2.3「Migrate core.py to numpy arrays for performance」「Phase out omitted rebalance targets」，v1.3.0「Normalize Herfindahl index」，v1.4.0「Reject nonfinite capital flows」。要钉版本。
4. 覆盖环节：(iii) 权重型组合回测（algo 积木：调仓频率、选股、定权重、再平衡，可嵌套策略树）；(vi) 经 ffn 出统计和图。其余无。
5. A 股规则：不原生支持。【读到】有 `commission_fn(quantity, price)`（可自己写最低 5 元 + 印花税）、`integer_positions`（整数股，不是整手）。https://github.com/pmorissette/bt/blob/master/bt/core.py
   - 【推断】T+1、涨跌停、停牌、整手、分红都没有；适合「给定每期目标权重 → 快速算净值」的粗回测，不适合要求成交约束的 A 股回测。
6. 数据格式：直接吃价格宽表 DataFrame。无锁定。
7. 坑：逐日 Python/Cython 树更新，全市场几千只票时速度一般（【推断】）。

---

## 3. 因子检验

### 3.1 alphalens-reloaded（stefan-jansen/alphalens-reloaded）

1. 维护状态：【读到】沉寂一年多后在 2026-10 恢复维护：0.5.0（10-17）、0.5.1（10-26），此前 0.4.6（2025-06）；master 2026-10-14～10-26 有 20 多次提交。https://github.com/stefan-jansen/alphalens-reloaded/releases
2. 许可：【读到】Apache-2.0。
3. Python / pandas / numpy：【读到】`requires-python >=3.11,<3.16`；`pandas>=1.5.0,<4.0`；0.5.0 发布说明「Supports Python 3.11-3.15…Correct all-cash holdings under pandas 3」，0.5.1「Preserve inferred trading-calendar metadata during factor cleaning on pandas 3.1」。即官方已在 pandas 3.1 上测过。https://github.com/stefan-jansen/alphalens-reloaded/blob/main/pyproject.toml
4. 覆盖环节：(i) 的 IC（含按组、按月）、分位数分层收益、多空价差、换手率和因子秩自相关、事件研究。**不含因子之间的相关性矩阵**（它一次只分析一个因子）【推断，基于其 API 设计】。
5. A 股规则：不涉及交易规则（不计成本）。【推断】涨跌停/停牌股票要在传入前自己从样本里剔除，否则分层收益虚高。
6. 数据格式：pandas 长表（MultiIndex：date × asset）因子 + 价格宽表，直接吃 DataFrame。
7. 坑：【读到】0.5.0 把分位数多空价差改成「50% 多 / 50% 空、100% 总敞口」口径，「Historical raw spreads and error bands were twice these values」——和旧版、旧教程的数值差一倍；全 A 股 × 多年 × 多因子时 pandas 实现偏慢（【推断】）。

### 3.2 alphainspect（wukan1986/alphainspect）+ polars_ta + expr_codegen

1. 维护状态：【读到】个人项目，低频但持续：alphainspect 0.5.10（2026-10-14），上一版 0.5.9（2025-10-23），中间约一年没发版；polars-ta 0.5.18（2026-10-20），上一版 2026-01；expr-codegen 0.17.0（2026-10-22）。三者同一作者（wukan）。https://pypi.org/project/alphainspect/#history 、https://pypi.org/project/polars-ta/#history
2. 许可：【读到】alphainspect MIT、polars-ta MIT、expr-codegen BSD-3。
3. Python / pandas / numpy：【读到】`requires-python >=3.8`，没有声明上限或分类器；依赖 `polars>=1.28`、`polars-ols`、`numba`、`numpy`、`pandas`（都无版本上限）。Python 3.13 / pandas 3 的兼容【未核实】（计算主体在 polars，pandas 只是辅助，【推断】风险较小）。
4. 覆盖环节：(i)。【读到】README 自述「仿 alphalens 的单因子分析工具」，示例覆盖因子分层、IC 统计、分层收益和累计收益、多空价差、换手率、事件分析，可多进程批量出 HTML/Notebook 报表。polars_ta 是算子库（仿 WorldQuant 的 `ts_*`/`cs_*`、TA-Lib、通达信公式），expr_codegen 把因子表达式编译成 polars 代码——属于 (ii) 的特征计算层。因子相关性矩阵【未核实】有没有现成函数。https://github.com/wukan1986/alphainspect/blob/main/README.md
5. A 股规则：【读到】README：不考虑滑点和手续费；远期收益率由用户自己算好传入（可以自己定义 T+1 开盘买、T+N 开盘卖等口径）；不做去极值、标准化、中性化；只做等权。
6. 数据格式：【读到】示例直接 `pl.read_parquet(...)`，输入是 polars 长表（date、asset、因子列、远期收益列）。和 Parquet 数据湖天然契合。
7. 坑：文档少（主要靠 examples）；单人维护；API 以函数 + 约定列名为主（`_DATE_` 等），要读源码。

---

## 4. 绩效报告

### 4.1 quantstats（ranaroussi/quantstats）

1. 维护状态：【读到】活跃但近期数值口径变动频繁：0.0.82～0.0.86 在 2026-09-25～09-27 三天内连发，发布说明多处写「Statistical corrections — these change reported numbers」，并有 0.0.83 引入回归、0.0.84 修复的记录（Kelly、RAR、greeks 等）。master 2026-11-02 仍有提交。https://github.com/ranaroussi/quantstats/releases
2. 许可：【读到】Apache-2.0。
3. Python / pandas / numpy：【读到】`requires-python >=3.10`，分类器到 3.13；`pandas>=2.0.0`、`numpy>=1.24.0` 无上限。issue #582「Matplotlib tick-label warning with pandas 3.1.0」（2026-10-23，open）和 PR #583「Fix yearly returns tick labels with pandas 3.1」（open）。https://github.com/ranaroussi/quantstats/issues?q=pandas+3
   - 【推断】pandas 3.1 下能跑，已知问题是绘图刻度标签警告，尚未合并修复。
4. 覆盖环节：(vi)。收益序列 → 指标表 + HTML tearsheet（可带基准）。只需要日收益率序列，和回测引擎解耦。
5. A 股规则：不涉及。【推断】默认年化 252 天、基准示例是 SPY，用 A 股时要自己传基准收益序列和 `periods_per_year`。
6. 数据格式：pandas Series/DataFrame。
7. 坑：见第 1 条——不同小版本算出的指标可能不同，要钉版本；自己核对关键指标（夏普、最大回撤、信息比率）的口径。

### 4.2 pyfolio-reloaded / empyrical-reloaded（stefan-jansen）

1. 维护状态：【读到】同 alphalens-reloaded，2026-10 恢复维护。pyfolio-reloaded 0.10.2（2026-11-02）；empyrical-reloaded 0.7.2（2026-11-02 上传 PyPI）。https://github.com/stefan-jansen/pyfolio-reloaded/releases 、https://github.com/stefan-jansen/empyrical-reloaded/releases
2. 许可：【读到】均为 Apache-2.0。
3. Python / pandas / numpy
   - pyfolio-reloaded：【读到】`requires-python >=3.11,<3.16`；`pandas>=1.5.0,<3.2`；0.10.1 发布说明「Permit pandas 3.0 and 3.1 on Python 3.11-3.15…Pandas 3.2 is not yet qualified. Pandas 3.1 requires NumPy 2.0.2 or newer on Python 3.11-3.13」。https://github.com/stefan-jansen/pyfolio-reloaded/blob/main/pyproject.toml
   - empyrical-reloaded：【读到】`requires-python >=3.11`；`pandas>=2.2.2`（py>=3.12）无上限；0.7.0「Includes pandas 3.0 compatibility and current NumPy compatibility」。
   - 【推断】pandas 3.1.x 可用；项目将来升到 pandas 3.2 时 pyfolio-reloaded 会成为解析器上的阻塞点（empyrical 不会）。
4. 覆盖环节：(vi)。empyrical 是纯指标函数库（夏普、回撤、alpha/beta、捕获率等）；pyfolio 出 tear sheet（收益、持仓、交易、归因）。
5. A 股规则：不涉及。
6. 数据格式：pandas Series/DataFrame。【读到】0.10.2「Keep missing optional Zipline quiet so standalone imports also work」——可以脱离 zipline 独立使用。
7. 坑：【读到】近期多次「numerical corrections」（Sortino、回撤表、Omega、常数收益的夏普等），同样要钉版本；pyfolio 的持仓/交易级 tear sheet 需要按它的列格式喂数据。

---

## 5. 组合优化

### 5.1 cvxpy（cvxpy/cvxpy）

1. 维护状态：【读到】非常活跃。1.9.3（2026-09-19），2026 年已发 1.8.2、1.9.0～1.9.3；master 2026-11 每周多次提交。https://pypi.org/project/cvxpy/#history
2. 许可：【读到】Apache-2.0。
3. Python / pandas / numpy：【读到】`requires-python >=3.11`；`numpy>=2.0.0`、`scipy>=1.13.0`；默认带 clarabel、osqp、scs 求解器，可选 highspy。不依赖 pandas。https://github.com/cvxpy/cvxpy/blob/master/pyproject.toml
4. 覆盖环节：(iv) 的底层建模语言。跟踪误差约束（二次/二阶锥）、行业和风格暴露约束（线性）、换手约束（L1 范数）、个股上下限都能直接写。Qlib 的 `EnhancedIndexingOptimizer` 和 riskfolio、skfolio、PyPortfolioOpt 全部建在它上面。
5. A 股规则：不涉及。【推断】整手取整、最小交易金额属于整数规划，一般是优化出连续权重后在下单环节再取整。
6. 数据格式：numpy 数组。
7. 学习成本和坑：要自己写问题（一个指数增强优化器约 50–100 行）；全 A 5000 只股票时要用因子结构（F、Σ_f、对角特异方差）而不是稠密协方差，否则慢（【推断】，Qlib 的实现就是这么写的，见 1.1）。

### 5.2 riskfolio-lib（dcajasn/Riskfolio-Lib）

1. 维护状态：【读到】活跃。7.4.1（2026-11-04），7.4.0（10-04）、7.3.0（05-31）；master 2026-10 下旬到 11-04 密集提交。https://pypi.org/project/Riskfolio-Lib/#history
2. 许可：【读到】BSD-3。
3. Python / pandas / numpy：【读到】`requires-python >=3.10`，分类器 3.10–3.14；`numpy>=1.26.4`、`pandas>=2.2.2`、`cvxpy>=1.7.2`、`scipy>=1.16.1`，无上限；含 C++ 扩展（wheel 分平台）。issue 列表里没搜到 pandas 3 相关条目，pandas 3 兼容【未核实】。https://github.com/dcajasn/Riskfolio-Lib/issues?q=pandas+3
4. 覆盖环节：(iv)。【读到】README：「Portfolio optimization with constraints on tracking error and turnover」「Tools to build linear constraints on assets, asset classes and risk factors」「Portfolio optimization with Risk Factors model」，以及 26 种风险度量、风险平价、HRP/HERC、Black-Litterman、基数约束。https://github.com/dcajasn/Riskfolio-Lib/blob/master/README.md
5. A 股规则：不涉及。
6. 数据格式：pandas DataFrame（收益率宽表）。
7. 坑：【推断】接口以「历史收益率矩阵」为中心（内部估计 μ/Σ），面向几十到几百个资产的配置问题；把自己的 alpha 预测 + Barra 式因子协方差灌进去做几千只股票的指数增强不是它的主场，不如直接写 cvxpy。

### 5.3 PyPortfolioOpt（PyPortfolio/PyPortfolioOpt）

1. 维护状态：【读到】低频。1.6.0（2026-02-26），上一版 1.5.6（2024-12）；仓库已迁到 `PyPortfolio` 组织（README 标「GC.OS Sponsored Project」），master 最后提交 2026-07-07。https://pypi.org/project/pyportfolioopt/#history
2. 许可：【读到】MIT。
3. Python / pandas / numpy：【读到】分类器 3.10–3.14；`numpy>=1.26,<3`、`pandas>=1.0,<4`、`cvxpy>=1.1.19`。pandas 3 实测情况【未核实】。
4. 覆盖环节：(iv) 的入门级封装。【读到】`objective_functions.py` 有 `transaction_cost`、`ex_ante_tracking_error`、`ex_post_tracking_error`、`L2_reg`（是目标函数项，不是硬约束）；可用 `add_constraint` 加自定义凸约束。https://github.com/PyPortfolio/PyPortfolioOpt/blob/main/pypfopt/objective_functions.py
5–7. 不涉及 A 股规则；吃 DataFrame；【推断】适合教学和小规模均值-方差，做带因子风险模型的指数增强不够用。

### 5.4 skfolio（skfolio/skfolio）——新增候选

1. 维护状态：【读到】非常活跃。1.12.9（2026-11-10），2026-10-25 以来发了 10 个版本；master 2026-11-10 有提交。https://pypi.org/project/skfolio/#history
2. 许可：【读到】BSD-3。
3. Python / pandas / numpy：【读到】`requires-python >=3.10`，README 徽章和分类器 3.10–3.14；`numpy>=1.24`、`pandas>=2.1`、`cvxpy-base>=1.5`、`scikit-learn>=1.6`；v1.12.2 发布说明「Avoid pandas inplace keyword deprecated in pandas 3.1」，说明在跟 pandas 3.1。https://github.com/skfolio/skfolio/releases
4. 覆盖环节：(iv) + 部分 (v)。【读到】README 的 Optimization Features：Transaction Costs、Weight / Group / Budget Constraints、**Tracking Error Constraints**、**Turnover Constraints**、Cardinality；有「Benchmark Tracker」优化器；Prior Estimator 下有「**Characteristics-Based Cross-Sectional Factor Model**：46 descriptors across 17 families (e.g. value, size, momentum, profitability)、Factor Exposures、Cross-Sectional Regression、Alpha Estimators、Forecast Evaluation、Ex-post and Ex-ante Attribution」；另有 Walk Forward / Combinatorial Purged CV 等模型选择工具。https://github.com/skfolio/skfolio/blob/main/README.rst
   - 【读到】README 示例和 Factor Models 用户指南（https://skfolio.org/user_guide/factor_models.html ，页面很长，读了全文两段）里的 `CharacteristicsFactorModel`：
     - 因子结构：`GlobalFactor`（市场/国家因子）、`OneHotCategoricalFactors`（行业）、`FixedWeightedFactor`（风格因子 = 若干描述符按固定权重合成，如 `EWMarketBeta`、`EWMomentum(half_life, skip)`、`LogMarketCap`、`BookToPrice`、`EWResidualVolatility`）、`DerivedFactor`（如非线性市值 = size³）；`neutralize_against`（因子间正交化）、`constrained_families`（行业因子收益按基准权重加总为零的约束）、`exposure_lag`。
     - 截面回归：默认权重 ∝ 市值^0.5（`regression_mcap_power=0.5`），可用两步可行 GLS 混入特异方差倒数权重。
     - 输出：因子暴露、因子收益、因子协方差、特异方差、资产协方差 Σ = B F Bᵀ + D。
     - 协方差估计：默认 `RegimeAdjustedEWCovariance`（方差和相关系数分开设半衰期、可选 Newey-West HAC 修正 `hac_lags`、短期波动率的标量 regime 乘数）；特异风险用 `RegimeAdjustedEWVariance`。指南里**没有**读到特征值调整（eigenfactor adjustment）和特异风险贝叶斯收缩。
     - 评估与归因：`CovarianceForecastEvaluation`（Mahalanobis 校准比、组合 bias statistic、QLIKE）、`predicted_attribution` / `realized_attribution`（事前/事后归因）、`alpha_forecast_evaluation`（Spearman/Pearson IC、ICIR、t 值、分位数汇总、衰减、`factor_correlation_summary`）。
     - 接优化：作为 `MeanRisk` 的 `prior_estimator`，可写因子暴露线性约束（如 `"momentum >= 1.0"`）、交易成本。
     - 规模：指南给的基准是 5000 只资产 × 10 年日频，16 因子模型 fit 约 25 秒、58 因子约 87 秒（笔记本），增量 `partial_fit` 不到 1 秒；5000 资产 × 80 字段 × 2500 天的面板在 float64 下约 8 GB。
     - 输入：自己的 `AssetPanel`（日期 × 资产的对齐数组，含 `returns`、`market_cap`、行业类别字段和各描述符所需的财务字段，外加 `active_mask` / `estimation_mask`）。
   - 【推断】这是本次查到的、唯一在维护且带宽松许可的「Barra 式截面因子风险模型」通用实现，结构上已经覆盖 CNE5 的主干（国家 + 行业 + 风格、加权约束回归、EWMA + Newey-West + 波动率 regime 调整、bias 统计检验）。但它不是 CNE5/CNE6 的现成复刻：A 股的描述符口径（如 CNE5 的 BETA/RSTR/DASTD/CMRA/BTOP 等具体定义和权重）、申万/中信行业、停牌与新股处理、财务数据的时点对齐都要自己配；指南也说示例参数「intentionally simple and has not been fine-tuned」。该模块首次出现的版本【未核实】（skfolio 1.0.0 发布于 2026-08-23，之后两个半月发了 12 个小版本），属于很新的功能，需要自己用 bias 统计验证。
   - 顺带：`alpha_forecast_evaluation` 也覆盖了环节 (i) 的 IC / 分位数 / 因子相关性，是 alphalens 之外的另一个选择（基于同一个 `AssetPanel`）。
5. A 股规则：不涉及。停牌资产如何处理，指南没有明确写（只说不可投资的资产权重为 0、矩为 NaN）。
6. 数据格式：numpy 数组组成的 `AssetPanel` + scikit-learn 风格的 `fit / predict`；从 Parquet 长表要自己 pivot 成 日期 × 资产 的宽数组。
7. 坑：scikit-learn 式的 estimator 组合抽象需要适应；发版极频繁（两周 10 版），要钉版本；因子模型功能新，社区实践少。

---

## 6. 风险模型：开源的 A 股 Barra 风格实现

检索方式：两次网页搜索 + relatedrepos 聚合页（https://relatedrepos.com/gh/xinyue6688/Barra-CNE5 ）列出的同类仓库，再逐个读 GitHub 提交记录和 README。星数来自聚合页，未在 GitHub 上复核。

| 仓库 / 包 | 内容（README 自述） | 最后提交 | 许可 | 结论 |
|---|---|---|---|---|
| ShiliangZhang-nku/Barra_CNE6（约 389 星） | 【读到】只做 CNE6 **因子构建**（dask 并行算日频因子），一致预期类因子没做；数据「如有需求请联系本人」 | 【读到】2020-01-20 | 【读到】仓库无 LICENSE 文件（默认保留所有权利） | 只能当公式参考，不含截面回归、协方差、特异风险 |
| rosie068/BARRA_risk（约 84 星） | 【读到】「follows BARRA's CNE6 and USE4 risk model」的实习项目 | 【读到】2020-12-11 | 无 LICENSE 文件 | 实习作品，停更 5 年 |
| YTZzzzz/Barra_CNE5（约 184 星） | 【读到】从 Wind 取数 → 10 个风格因子 → 行业因子 → 加权回归求因子收益 → 风险预测 | 【读到】2018-08-21 | 无 LICENSE 文件 | 流程最完整的一个，但 8 年没动、绑 Wind 数据库 |
| xinyue6688/Barra-CNE5（约 23 星） | 【读到】CNE5 风格因子复刻 + 组合风险分析，README 给了各因子公式（BETA 252 日窗口半衰期 63、RSTR 504 日滞后 21 等），取数用 Wind 落地库表名 | 【读到】2025-06-17 | 无 LICENSE 文件 | 最新的一个，仍是个人项目、绑 Wind |
| UePG-21/Barra-risk-model（约 44 星） | 【读到】（聚合页描述）Barra 风险模型里的数学处理 | 【读到】2023-04-11 | 【未核实】 | 片段 |
| mangoquant/Barra（约 13 星） | 【读到】（聚合页描述）计算 Barra 因子及其收益率 | 【读到】2022-04-08 | 【未核实】 | 片段 |
| PyPI `barra-risk-model` 0.1.5 | 【读到】搜索摘要：自学项目、禁止商用、依赖 Wind API；PyPI 最后发布 2018-12-26 | 2018-12 | 禁止商用（搜索摘要，未打开原页复核） | 不可用 |
| toraniko（0xfdf/toraniko） | 【读到】通用的 Barra/Axioma 式特征因子模型（市场 + 行业 + 风格因子收益估计、因子协方差），polars 实现，自带 value/size/momentum 三个风格；示例是美股 GICS | 【读到】2024-08-05（当天 21 次提交后再无更新） | 【读到】MIT | 代码小而干净，可借鉴；但 pyproject 钉 `numpy ~1.26.2`、`polars ~1.0.0`，与 numpy ≥ 2.5 **冲突**，且已停更 |
| skfolio 的 `CharacteristicsFactorModel` | 见 5.4：国家 + 行业 + 风格因子、加权约束截面回归、EWMA + Newey-West + regime 调整的因子协方差、特异方差、bias/QLIKE 检验、归因 | 2026-11-10 | BSD-3 | 唯一在维护的通用实现；不是 CNE5 复刻，A 股描述符和数据要自己配；功能很新 |
| Qlib `StructuredCovEstimator` | 见 1.1；PCA/因子分析的统计因子模型 | 2026-10 | MIT | 不是 Barra 基本面因子模型 |
| riskprism 0.9.1（PyPI） | 【读到】PyPI 摘要「Open-source US equity factor risk model」 | 2026-08-24 | MIT | 只做美股，不适用 |

结论
- 【读到 + 推断】**没有**一个「在维护、带许可证、能直接装、不绑 Wind」的开源 A 股 CNE5/CNE6 实现。GitHub 上的 CNE5/CNE6 仓库都是 2018–2025 年的个人/实习项目，全部没有 LICENSE 文件，多数只到「因子构建」或「截面回归」，缺 Newey-West 调整、特征值调整、波动率偏误调整（VRA）、特异风险贝叶斯收缩这些让模型可用的步骤（后半句是【推断】：从 README 列出的功能里没看到这些步骤，没有逐个读源码确认）。
- 商业数据路线：【读到】米筐 RQData 提供「风格因子暴露度、收益率、协方差和特异风险」（rqalpha README），是付费数据。
- 这一环节的现实做法【推断】：
  - A 股风格描述符（10 个 CNE5 风格的具体公式、申万/中信行业映射、财务数据时点对齐）**必须自己写**，没有可直接用的轮子；公式可参考 xinyue6688/Barra-CNE5 的 README 和 MSCI 公开的 CNE5 文档。
  - 「截面回归 → 因子协方差 → 特异风险 → 检验/归因」这一半，可以先试 skfolio 的 `CharacteristicsFactorModel`（见 5.4），不行再自己用 numpy 写（toraniko 的代码结构可借鉴）。skfolio 指南里没读到的特征值调整、特异风险贝叶斯收缩，如果需要得自己补。
  - 用 skfolio 还是自己写，取决于本机实测：它能否吃下全 A 约 5000 只 × 多年的面板、停牌和新股的掩码怎么给、bias 统计是否接近 1。

---

## 7. 实盘接入：QMT / miniQMT（xtquant）、PTrade

### 7.1 xtquant（迅投 miniQMT 的 Python 库）

- 【读到】PyPI 有 `xtquant` 包，最新 250807.1.2（2026-07-21），包描述写「系统要求：Windows x64；Python 3.6 - 3.13」「`pip install xtquant`」，wheel 是 `py3-none-any`。PyPI 的上传者是否为迅投官方【未核实】（描述链接到迅投官方文档）。https://pypi.org/project/xtquant/
- 【读到】迅投官方文档：「在运行使用 XtQuant 的程序前需要先启动 MiniQMT 客户端」；`xtdata` 提供行情、财务、合约信息、板块行业分类，`xttrader` 负责下单撤单和资产/委托/成交/持仓查询与推送；快速开始页写支持 64 位 Python 3.6–3.12，下载页的更新记录写 20250516 版「支持 python3.13 版本」，最新包 xtquant_250807（更新日期 20251219）。官方页面没有提 Linux/macOS。https://dict.thinktrader.net/nativeApi/start_now.html 、https://dict.thinktrader.net/nativeApi/download_xtquant.html
- 结论【推断】：xtquant 支持 Python 3.13，但**只能在 Windows 上、且 miniQMT 客户端保持运行时使用**；本项目的 macOS 研究环境不能直接 import 它下单。实盘需要一台 Windows 机器（或虚拟机/云主机）跑 miniQMT，研究端和交易端要分离。券商开通 miniQMT 权限的门槛（资金量等）各家不同，【未核实】。
- 许可：PyPI 分类器写 MIT【读到】；xtquant 含编译好的二进制，实际使用受券商/迅投的服务条款约束【推断】。

### 7.2 研究端（Mac/Linux）到 Windows miniQMT 的桥接

- qmt-bridge（atompilot/qmt-bridge）：【读到】PyPI 2.1.3（2026-02-11），MIT，Python >=3.10；README：「miniQMT / xtquant 只能在 Windows 上运行，且必须依赖 QMT 客户端保持登录」，本项目让 Windows 作为中转站，把行情和（可选的）交易能力经 HTTP/WebSocket 暴露给局域网里的 Mac/Linux，客户端只用标准库。2026-02 之后是否还在维护【未核实】。https://pypi.org/project/qmt-bridge/ 、https://github.com/atompilot/qmt-bridge
- xtquant-rpc-client：【读到】PyPI 0.1.4（2026-04-17），MIT，Python >=3.11,<3.14，「Cross-platform gRPC client for remotely hosted xtquant domains」。https://pypi.org/project/xtquant-rpc-client/
- 【推断】两者都是小型个人项目，涉及真实下单，用之前要读完源码；最稳妥的做法是自己在 Windows 端写一个很薄的执行脚本：读研究端产出的目标持仓文件 → 用 xttrader 下单。

### 7.3 各框架对 QMT 的支持

- vnpy：【读到】`vnpy_xt`（1.4.6，2025-10-18；仓库 README 徽章已是 1.4.7，2026-10 有提交）自述是「VeighNa 框架的迅投研**数据服务接口**」「实时行情和数据服务接口」，平台徽章 windows，Python 3.10–3.13；连接方式是迅投研 Token 或本机迅投客户端。它是行情/数据接口，README 没有写交易下单功能。vnpy 的 A 股交易接口在 README 里列的是中泰 XTP 等柜台。https://github.com/vnpy/vnpy_xt
  - 更正 1.3 节：`vnpy.alpha` → miniQMT 下单没有现成的官方通路，【未核实】有无社区 gateway。
- easytrader（shidenggui/easytrader）：【读到】PyPI 0.23.7（2025-04-16），BSD，有 `miniqmt` extra（依赖 xtquant）；仓库 2026-10-16 有提交。只是下单封装，不含回测。https://pypi.org/project/easytrader/
- qteasy（shepherdpp/qteasy）：【读到】PyPI 2.6.5（2026-10-03），BSD-3，A 股本地化框架（Tushare 数据、向量化 + numba 回测、「支持 T+1 交割、交易费用、MOQ、交割队列」），README 写「未来将通过 QMT 接口接入券商提供的实盘交易接口」（文档里已有 XtQuant/MiniQMT broker 适配契约页）。但 `requires-python >=3.9,<3.13`、`numpy>=1.26.4,<2.0.0`——与 Python 3.13 / numpy ≥ 2.5 **不兼容**，排除。https://pypi.org/project/qteasy/ 、https://github.com/shepherdpp/qteasy
- hikyuu、rqalpha：有 broker 抽象接口（`OrderBroker`、`AbstractBroker`），本次没有读到官方的 QMT 实现，【未核实】。
- Qlib、vectorbt、bt、PyBroker：无实盘（PyBroker/vectorbt 的券商集成是 Alpaca 等海外通道）。

### 7.4 PTrade（恒生）

来源是社区整理的 PTrade API 文档站 https://ptradeapi.com/ （不是恒生官方域名；各券商版本有差异，以券商提供的文档为准）。
- 【读到】PTrade「运行在券商机房，属于托管模式」；「无法通过 python 的 pip 安装第三方的库」，只能用内置库；因在券商内网「无法连接到互联网」，只有「极少数的券商」开放外网；内置 Python 部分券商「更新到 3.11」，「大部分券商停留在 3.5」；回测支持日线和分钟，「券商实盘版无法盘中回测，可盘后回测」；可用 `get_research_path()` 目录做文件持久化，有 `convert_position_from_csv`。
- 结论【推断】：PTrade 不是「把本地框架接上去」的模式，而是把策略代码改写成 PTrade 的事件函数（`initialize / handle_data / before_trading_start` 等）上传到券商机房运行。本地研究产出的模型（如 LightGBM 文件）能否带进去取决于券商内置库版本；更现实的做法是本地每天算好目标持仓，上传文件到研究目录，PTrade 端只写一个「读文件 → 调仓」的薄策略。能否每天自动上传文件，各券商不同，【未核实】。
- SimTradeLab（kay-ou/SimTradeLab）：【读到】PyPI 2.13.8（2026-11-02），本地模拟 PTrade 回测 API 的社区项目（「tested subset」，非恒生官方），**AGPL-3.0 + 商业许可双授权**，Python >=3.9。可用于本地调试 PTrade 策略代码。https://pypi.org/project/simtradelab/ 、https://github.com/kay-ou/SimTradeLab

### 7.5 小结

【推断】(vii) 没有「研究框架直连实盘」的成熟开源轮子。QMT 路线要自己写 Windows 端执行层（xtquant），PTrade 路线要写券商机房内的薄策略；两条路线下，研究框架都只需要产出「目标持仓/目标权重文件」。

---

## 8. 调研中新发现的候选

### 8.1 AKQuant（akquant，akfamily/akquant）——值得本机试

1. 维护状态：【读到】极活跃但很年轻。PyPI 首版 2026-01-29，至 2026-11-11 已发 233 个版本，最新 0.3.85（2026-11-11）；master 2026-11-08～11-11 有 21 次提交。作者组织 akfamily 即 AKShare 的作者。https://pypi.org/project/akquant/#history 、https://github.com/akfamily/akquant
2. 许可：【读到】MIT。
3. Python / pandas / numpy：【读到】`requires-python >=3.10`，分类器 3.10–3.15；`pandas>=2.2.0`、`numpy>=2.2.2`、`polars>=0.20.0`，无上限；Rust 内核，PyPI 提供预编译包（README：「无需安装 Rust 环境」）。pandas 3 实测【未核实】。
4. 覆盖环节：(ii) 部分（README：内置 Walk-forward Validation 滚动训练框架，适配 PyTorch / scikit-learn；Polars 驱动的因子表达式引擎，支持 `Rank(Ts_Mean(Close, 5))` 这类 Alpha101 风格公式）；(iii) 有（事件驱动回测，`rebalance_weights(target_weights=…)`、`order_target_percent` 等组合调仓 API）；(vi) 有（plotly HTML 报告，带基准时出超额、跟踪误差、信息比率、Beta、Alpha）。(i) 的 IC/分层、(iv)、(v) 没读到；实盘文档讲的是 CTP（期货），miniQMT 没提。https://github.com/akfamily/akquant/blob/main/README.md 、https://akquant.akfamily.xyz/guide/strategy/
5. A 股规则：【读到】`broker_profile` 内置 `cn_stock_miniqmt`、`cn_stock_t1_low_fee`、`cn_stock_sim_high_slippage`（README 注释：「A 股佣金/印花税/过户费、T+1、一手 100 股」）；参数有 `t_plus_one`、`stamp_tax_rate`、`min_commission`、`transfer_fee_rate`、`lot_size`；README 测试一节说有针对「T+1、涨跌停、保证金」的黄金测试。README 明确警告：「`run_backtest` 默认不计交易成本、不开 T+1」，必须显式传 `broker_profile`。
   - 涨跌停的具体撮合规则、停牌、分红送转/复权：策略指南里没读到，【未核实】。
6. 数据格式：【读到】直接吃 pandas DataFrame（`run_backtest(data=df, …)`），也可 `add_arrays` 零拷贝导入 numpy。无格式锁定。
7. 坑：10 个月发 233 版，API 稳定性存疑，必须钉版本；默认参数是「理想化」设置，漏传 profile 结果会偏乐观；社区案例少。

### 8.2 其他（都不建议作为基础）

- qteasy：见 7.3，Python <3.13、numpy <2.0，与目标环境不兼容。
- deltafq 1.1.1（2026-07-19，MIT，Delta-F/deltafq）：【读到】PyPI 摘要「A-share low-frequency quantitative trading framework covering research, backtesting, and execution」，2026-07 之后无提交。成熟度【未核实】。
- atrs 2026.9.10.1（2026-09-10，MIT）：【读到】PyPI 摘要「Ultra-fast backtest engine for A-shares」，Rust + polars，仅 4 个版本；PyPI 指向的 GitHub 仓库 link-yundi/atrs 的提交提要和 README 都取不到（404/空）。【未核实】，不采用。
- polars-backtest 0.1.7（2026-07-05）：【读到】许可是 PolyForm Noncommercial 1.0.0（非商业）。不采用。
- alpha101-pipeline 0.6.2（2026-09-28，MIT）：【读到】DuckDB 驱动的因子研究流水线（公式引擎、因子库、分层回测），2026-06 才出现。【未核实】。
- backtrader 的社区分支：搜索结果提到 cloudQuant 的分支和 PyPI 的 backtrader-next（2.3.7，2026-03-26），都没打开核实。【未核实】。
- wtpy：钉 pandas==1.3.5，不适用。

---

## 9. 目标环境层面的几条约束（影响多个候选）

- 【读到】numba 0.68.1（2026-10-27）声明 `numpy<2.6,>=1.22`、Python 3.10–3.15。https://pypi.org/project/numba/ 。【推断】凡是依赖 numba 的候选（vectorbt、PyBroker、polars_ta、alphainspect 经 polars_ta）都会把 numpy 限制在 2.5.x，不能跟着升 2.6；这与「numpy ≥ 2.5」不冲突，但要知道上限在哪。
- 【读到】polars 已到 2.x：2.0.0（2026-10-06）、2.1.0（2026-10-25），`requires-python >=3.11`。https://pypi.org/project/polars/#history 。polars_ta（`polars>=1.28`）、vnpy.alpha（`polars>=1.26`）、akquant（`polars>=0.20`）都没有上限，uv 会解析到 polars 2.x；它们在 polars 2.x 下是否正常【未核实】（polars-ta 0.5.18 发布于 2026-10-20，在 2.0 之后、2.1 之前）。本机测试时要特别看这一点，必要时钉 `polars<2`。
- 【读到】LightGBM 4.7.0（2026-07-18），分类器 3.10–3.14，`numpy>=1.21.3`。https://pypi.org/project/lightgbm/ 。(ii) 的模型本身没有环境障碍。
- 【读到】statsmodels 0.15.0（2026-08-27），Python 3.10–3.14，`pandas>=1.4`（alphalens-reloaded、vnpy.alpha 的因子分析依赖它）。
- 【推断】与目标环境**声明层面直接冲突**的候选：zipline-reloaded（pandas<3.0）、qteasy（Python<3.13、numpy<2）、toraniko（numpy~1.26）、wtpy（pandas==1.3.5）。**官方未声明支持、需自己编译/打补丁**的：pyqlib（无 cp313 wheel，pandas 3 修复 PR 未合并）、rdagent（3.10/3.11 + Linux + Docker）、QUANTAXIS（3.9–3.12）、backtrader（停更）。

### 补充（组合优化）：cvxportfolio（cvxgrp/cvxportfolio）

【读到】PyPI 1.5.1（2025-07-06），**GPLv3**；仓库最后提交 2026-04-27。README：多期优化（`MultiPeriodOptimization`）+ 自带市场模拟器回测，含交易成本和持有成本模型。https://pypi.org/project/cvxportfolio/ 、https://github.com/cvxgrp/cvxportfolio 。【推断】面向美股数据（示例用美股代码）、一年多没发版、GPL，不如直接 cvxpy 或 skfolio；没有进一步核实。

---

## 10. 汇总

### 10.1 对比表

「3.13 / pandas 3」一列：「支持」= 官方声明或发布说明明确支持；「未声明」= 依赖声明不拦、但没读到明确声明；「不支持」= 声明层面冲突或官方未支持。「A 股规则」指 T+1、涨跌停、停牌、印花税/最低佣金、分红送转、整手是否原生支持。

| 候选 | 最新版（日期）/ 维护 | 许可 | 3.13 / pandas 3 | 覆盖环节 | A 股规则 |
|---|---|---|---|---|---|
| Qlib | PyPI 0.9.7（2025-08）；GitHub 源码版 0.9.8（2026-10）；低速维护 | MIT | 不支持：wheel 只到 cp312；pandas 3 修复 PR 未合并 | i ii iii iv(增强指数优化器) vi；v 只有 PCA 统计模型 | 部分：涨跌停阈值、停牌、整手、最低 5 元；无分红入账、印花税不单列；数据须转 `.bin` |
| RD-Agent(Q) | 1.0.0（2026-09）；活跃 | MIT | 不支持：3.10/3.11，仅 Linux + Docker | ii 的自动挖因子/模型（底层用 Qlib） | 同 Qlib |
| vnpy.alpha | 4.5.0（2026-10）；活跃 | MIT | 支持：推荐 3.13；pandas 3 未声明 | i(单特征 tear sheet) ii iii vi | 弱：写死 ±10% 一字板、比例费率；无最低佣金/分红/停牌拦截 |
| hikyuu | 3.0.2（2026-11）；很活跃，3.0 刚发 | PyPI 写 MIT / 仓库 LICENSE 为 Apache-2.0 | 支持：3.10–3.13；pandas 3 未声明 | iii（主）i vi | 强：下一根开盘成交、一字板/停牌顺延、佣金/印花税/过户费分项、除权除息调整；数据须导入自有库；事件驱动 `backtest()` 属捐赠用户功能 |
| QUANTAXIS | PyPI 停在 2021；GitHub 2.1.0-alpha2（2025-10）；近乎停滞 | MIT | 不支持：声明 3.9–3.12 | i iii（成熟度未核实） | 未核实；绑 MongoDB/ClickHouse |
| rqalpha | 6.5.0（2026-10）；活跃 | **仅限个人/非营利的非商业使用**，其余需米筐授权 | 支持：3.8–3.14；6.4.2 起支持 pandas 3 | iii vi | 最全：T+1、涨跌停、停牌、最低 5 元、时点印花税、分红送转、整手、成交量限制；bundle 月更免费、日更需 RQData |
| backtrader | 1.9.78.123（2023-04）；停更 | GPLv3+ | 不支持：分类器只到 3.7，未核实 | iii | 无，全部自己写 |
| vectorbt（开源版） | 1.2.1（2026-11-11）；活跃 | Apache 2.0 + **Commons Clause** | 支持：3.11–3.14，要求 pandas ≥ 3.0.3、numpy ≥ 2.4.6 | iii(向量化) vi | 无；有整手粒度和固定费用，涨跌停/停牌/最低佣金/分红要自己处理 |
| zipline-reloaded | 3.2.1（2026-10）；恢复维护 | Apache-2.0 | 不支持：`pandas<3.0` | iii（+Pipeline） | 无；bcolz bundle |
| PyBroker | 2.0.2（2026-10）；活跃 | Apache 2.0 + **Commons Clause** | 支持：≥3.11；开发环境用 pandas ≥ 3.0.5 | ii(walkforward) iii vi | 无；可自定义费用函数 |
| bt | 1.4.1（2026-10）；活跃 | MIT | 支持：3.9–3.13；1.1.3 起支持 pandas 3 | iii(权重型) vi | 无；可自定义佣金函数 |
| alphalens-reloaded | 0.5.1（2026-10）；恢复维护 | Apache-2.0 | 支持：3.11–3.15；在 pandas 3.1 上测过 | i | 不涉及 |
| alphainspect + polars_ta | 0.5.10 / 0.5.18（2026-10）；单人低频维护 | MIT | 未声明：无声明；polars 2.x 兼容未核实 | i（+ ii 的算子层） | 不涉及；远期收益口径自定 |
| quantstats | 0.0.86（2026-09）；活跃，指标口径近期多次修正 | Apache-2.0 | 未声明：3.10–3.13；pandas 3.1 有未合并的绘图告警修复 | vi | 不涉及 |
| pyfolio-reloaded / empyrical-reloaded | 0.10.2 / 0.7.2（2026-11）；恢复维护 | Apache-2.0 | 支持：pandas 3.0–3.1（pyfolio 上限 `<3.2`） | vi | 不涉及 |
| cvxpy | 1.9.3（2026-09）；很活跃 | Apache-2.0 | 支持：≥3.11，numpy ≥ 2.0 | iv（底层） | 不涉及 |
| riskfolio-lib | 7.4.1（2026-11）；活跃 | BSD-3 | 未声明：3.10–3.14；pandas 3 未核实 | iv | 不涉及 |
| PyPortfolioOpt | 1.6.0（2026-02）；低频 | MIT | 未声明：3.10–3.14，pandas<4 | iv（入门级） | 不涉及 |
| skfolio（新增） | 1.12.9（2026-11-10）；很活跃 | BSD-3 | 支持：3.10–3.14；在跟 pandas 3.1 | iv、v（通用 Barra 式因子模型）、部分 i | 不涉及 |
| AKQuant（新增） | 0.3.85（2026-11-11）；10 个月 233 版 | MIT | 未声明：3.10–3.15，pandas ≥ 2.2 | ii(walk-forward) iii vi | 有：T+1、印花税/过户费/最低佣金、整手的 A 股 profile；涨跌停细节、停牌、分红未核实 |
| A 股 Barra 开源仓库 | 2018–2025 的个人项目，均无 LICENSE | 无 | — | v 的片段 | — |
| xtquant | 250807.1.2（2026-07） | PyPI 标 MIT | 支持：3.6–3.13，**仅 Windows** | vii | 实盘通道 |

### 10.2 结论（按环节）

读到的事实见各节链接；下面的取舍判断都是【推断】。

- (i) 单因子检验：有成熟轮子。alphalens-reloaded 0.5.1（官方在 pandas 3.1 上测过）可直接用；数据量大时用 alphainspect（polars，吃 Parquet 长表）。因子间相关性矩阵两者都没有现成函数，自己写几行。
- (ii) 多因子合成：模型本身（LightGBM 4.7）没有障碍；「特征 → 滚动训练 → 预测」的编排层没有适配目标环境的成熟轮子——Qlib 最完整但官方不支持 3.13/pandas 3，vnpy.alpha 支持 3.13 但要装整套 GUI 依赖。建议自己写一层薄的滚动训练。
- (iii) A 股回测：rqalpha 规则最全且官方支持 3.13/pandas 3，但许可仅限个人非商业使用、日更数据要付费或自写数据源；hikyuu 规则也全但数据要进它自己的库；AKQuant 新而快变。其余引擎都要自己补 A 股规则。
- (iv) 组合优化：cvxpy 直接写（可照 Qlib 的 `EnhancedIndexingOptimizer` 的问题形式），或用 skfolio；有成熟轮子。
- (v) 风险模型：没有可直接用的开源 CNE5/CNE6。A 股风格描述符和行业映射必须自己写；回归、协方差、检验可以试 skfolio 的 `CharacteristicsFactorModel`。
- (vi) 绩效：quantstats 或 pyfolio-reloaded + empyrical-reloaded，成熟，钉版本。
- (vii) 实盘：没有成熟的开源一体化轮子。xtquant 只能在 Windows + miniQMT 客户端下运行，PTrade 是券商机房托管；都需要自己写执行层，研究端只输出目标持仓。

### 10.3 方法与限制

- 版本、日期、依赖声明来自 PyPI JSON 接口（2026-11-12 读取）；提交活跃度来自 GitHub 的公开 Atom 提要（`/commits.atom`、`/releases.atom`）；许可和源码细节来自 raw.githubusercontent.com 的文件。GitHub REST API 因出口 IP 匿名限额用尽没有使用，所以 star 数、open issue 数没有直接核实（表里的 star 数来自第三方聚合页）。
- 没有安装或运行任何候选；所有「能否在 Python 3.13 + pandas ≥ 3.1 + numpy ≥ 2.5 下工作」的判断都只基于声明和 issue/发布说明。
- 网页搜索共 12 次。没有访问需要登录的页面。
- 未核实的重点：hikyuu / AKQuant / riskfolio / quantstats / alphainspect 在 pandas 3.1 和 polars 2.x 下的实际表现；skfolio 因子模型的首发版本与在全 A 上的效果；rqalpha 许可对「个人用自有资金实盘」的界定；各券商 miniQMT / PTrade 的开通条件和 PTrade 内置库版本。
