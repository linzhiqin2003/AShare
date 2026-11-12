# 风险模型与组合优化实现笔记

本实现按公开方法独立编写，不声称与 MSCI 商业模型的暴露、风险预测或数值一致。所有收益、alpha、F、D、成本和跟踪误差输入采用日频小数单位；方差不是波动率，年化由调用方显式换算。

## 工作范围与阶段记录

- 只修改本 worktree 的风险/组合模块、对应测试/基准，以及新增 `portopt` 包和 crate。未导入 panelops，未调用行情接口，未读取密钥，未提交或推送。
- Aline onecontext 历史检索返回 `unable to open database file`，未修改项目外配置。
- 全局 worklog 的要求与本任务“不得改项目外文件”及沙箱冲突；阶段记录保存在本文件，不写 `~/worklog`。
- 第一轮先实现 numpy 参考并实测完整 T=4000、N=5500、K=45，再写 Rust 候选。原始数字 `bench/risk/results/numpy-initial.json`；Rust 对拍及三次测量 `kernels.json`。
- 初始测试中的 ESTU 计数预期 73 错误：该样本的新股已有 20 天历史，大于配置的 15 天门槛，应为 74。改为严格整数 74，没有放宽任何数值断言。

## 接口和时点

- `RiskModel(symbols, B, factors, F, D, date, metadata)`：复制并冻结数组，校验有限数、维度、符号唯一性、F 对称半正定、D 非负。`variance/volatility` 可传基准，`decomposition/marginal_risk/contributions` 保留因子结构。`dense()` 硬限制 N≤256，只给测试。
- `PanelSource` 只有 dates、symbols、version、`__getitem__(field)`。`data.FIELDS` 集中记录单位。换手率是小数，市值是元，行业未知是 -1。行业编码字典由 `FundamentalConfig.industry_codes` 固定配置，默认 0…30，另设未知行业，不能由未来出现过的行业推导。
- `model_series` 按日生成，B_t 解释 r_(t+1)。模型 t 包含截至 t 收盘的数据；不表示能在 t 收盘价形成前获得并成交。复权因子、行业、估值和外部暴露的历史版本必须已经满足 point-in-time 契约。
- 停牌和缺价收益保留 NaN，不把缺失时间删掉后计算 Newey–West。上市年限不足或停牌者不进入估计池；覆盖证券仍有暴露和结构化特质风险。估计池为空时，暴露标准化临时用当日覆盖池，因子风险采用显式冷启动先验。
- 缓存键包括 schema、参数 JSON、数据版本、证券顺序、外部因子名和逐日前缀内容链摘要；追加日期不改变此前的键。缓存 NPZ 不用 pickle，用同目录临时文件和原子替换。源版本定义为历史修订版本，追加可以不改版本；改版本会保守地重算。历史输入被改却忘记改 version，也会使此前缀之后的键失效。
- 增量是只计算新快照，但恢复时会顺序读既有快照以重建有限历史窗口；尚未实现单独的轻量检查点索引。因此追加避免数值重算，但不是 O(1) 磁盘读取。

## 公式来源与参数台账

公开原始作者文档（镜像仍是原文）：

- U：Menchero、Orr、Wang，2011-08，[USE4 Methodology Notes](https://dmmn26wgpgtie.cloudfront.net/wp-content/uploads/2011/09/23120404/USE4_Methodology_Notes_August_2011.pdf)。已读取原文，定位以下节号和表号。
- C：Orr、Mashtaler、Nagy，2012-07，[CNE5 Empirical Notes 原文转录](https://studylib.net/doc/28502218/msci)。描述子在附录 A，检验在第 5 节、附录 C。没有拿研究机构的二手实现冒充 MSCI 参数。
- L：[Ledoit–Wolf, Honey, I Shrunk the Sample Covariance Matrix](https://www.ledoit.net/honey.pdf)，第 2 节及附录；[scikit-learn LedoitWolf 文档与源码](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.LedoitWolf.html)用作 scaled-identity 数值对拍。
- S：[Clarabel 问题形式](https://clarabel.org/stable/)、[OSQP 问题形式](https://osqp.org/docs/solver/)。优化目标及约束来自本次任务，均不是 MSCI 公式。

| 计算 | 来源 | 本实现及默认值 | 是否自行决定 |
|---|---|---|---|
| BFBᵀ+D、主动风险和 Euler 拆分 | U §1.3 | 用因子乘积、D 点乘；边际波动=(Σw)/σ | 风险接口和 N≤256 测试限制自定 |
| EW 权重 | U §4.1、§5.1 | p∝2^(-age/h)，有限窗口归一化，按有效列重新归一化 | 缺失推广自定，未实现 U 的 EM |
| LW scaled identity | L；sklearn 对应估计量 | μI，等权完整样本强度与 sklearn 对拍 | EW 权重影响函数推广、缺失策略自定 |
| 常相关目标 | L §2/附录 | 对角保留样本方差，非对角为平均相关×两者波动 | 平均相关裁剪到 [0,1]；有限 N 的平均相关导数修正自定 |
| PCA | 样本协方差特征分解 | 矩阵自由 eigsh，默认 45 个，余下对角进 D；固定起点和符号 | 截断数、容差 1e-11、缺失先验自定 |
| 市值 | C 附录 A，Size | ln(total_mv) | 无 |
| 贝塔 | C 附录 A，A1 | 252 日、半衰期 63，截距回归；市场收益用前日市值和前日 ESTU | 市场权重明确取前日，缺省 rf=0 自定 |
| 动量 | C 附录 A，A2 | 504 日，跳过最近 21 日，半衰期 126 | 对缺失有效权重归一化自定 |
| 残差波动 | C 附录 A，A3–A5；§3.3 | .74 DASTD + .16 CMRA + .10 HSIGMA；252 日；DASTD h=42，HSIGMA h=63；对 Size/Beta 正交 | CMRA 改为累积 log 收益最大值减最小值，见设计疑问 |
| 非线性市值 | C 附录 A，NLSIZE | 标准化 Size³ 对截距和 Size 做 sqrt(cap) 权重正交化，再去极值标准化 | 无 |
| 流动性 | C 附录 A，A9–A11 | 21/63/252 日换手和除以 1/3/12 后取 log，组合 .35/.35/.30，再对 Size 正交 | 零换手视为缺失自定 |
| BP、EP | C 附录 A，BTOP、Earnings Yield | 1/pb、1/pe_ttm；允许负值，0 视为缺失 | EP 只用价格比率，未实现原 C 的预测收益/现金流复合项，按任务范围简化 |
| 标准化 | U §2.1–2.3 | ESTU cap 加权均值=0、等权 std=1；先 MAD×1.4826×5 截断，行业+Size 回归填缺失 | MAD、5 倍阈值、填补形式自定；全缺失描述子置 0 |
| 截面 WLS | U §3.1，3.1–3.3 | 损失 Σsqrt(cap)·u²；行业收益约束 Σcap_industry·f_industry=0 | 正规方程 Rust 候选与独立 numpy 白化最小二乘对拍 |
| 因子协方差 | U §4.1，4.1；表 4.1 USE4S | 波动 h=84/NW lag=5；相关 h=504/NW lag=2 | 1008 日窗口自定；NW Bartlett 权重与缺失 Gram 扩展自定 |
| 特征因子调整 | U §4.2、附录 B | 固定 seed 高斯 Monte Carlo，模拟特征方向预测/真值比后旋回 | 默认关闭，seed=871，scale=1，模拟次数由调用方指定；未复制商业平滑标定 |
| 因子波动状态调整 | U §4.3，4.2–4.4；表 4.1 | 当日因子 r²/前日未含 NW 的方差截面均值，再 h=42 EW 平滑，乘入 F | 冷启动和无有效因子处理自定 |
| 特质 EW/NW | U §5.1，5.2；表 5.1 USE4S | 波动 h=84；自相关 h=252、lag=5；NW 比率乘原方差 | 逐列缺失归一化自定 |
| 结构化特质风险 | U §5.1，5.3–5.5 | log 波动对 B 回归，指数残差均值校正；按可靠历史比例混合波动 | γ=min(valid_days/252,1)、可靠阈值、冷先验均自定 |
| 贝叶斯收缩 | U §5.2，5.6–5.9；表 5.1 | 10 个市值组，组内 cap 加权先验；q=.1；对波动做收缩 | 小 N 分组缩减至 min(10,N) 自定 |
| 特质状态调整 | U §5.3，5.10–5.12 | 前日基准方差标准化残差，cap 加权，h=42 EW | 基准方差用未含 NW 的时间序列项，短史回退规则自定 |
| 偏差统计 | U 附录 A；C §5/附录 C | std(realized/predicted, ddof=1)，滚动 252 日，chi-square 名义 95% 区间 | 日频窗口及正态区间自定；不把重叠窗口视为独立样本 |
| Woodbury 闭式解 | 矩阵求逆引理，标准恒等式 | D⁻¹−D⁻¹U(I+UᵀD⁻¹U)⁻¹UᵀD⁻¹，附预算拉格朗日项 | 作为独立验证公式，不作受约束求解器 |

其余工程默认值均为自行决定：上市至少 60 天、描述子至少 20 天、结构化历史 252 天、历史上限 1008、方差地板 1e-10、冷特质方差 .02²、冷因子方差 .005²；缓存 schema=1；风险厌恶 10、个股上限 .02、满仓 1、默认无成本、无额外可选约束；验证容差 1e-7、求解器容差 1e-9、最大迭代 Clarabel 200/OSQP 100000；目标等价缩放为 1000N 只改善数值尺度，报告用原目标独立重算。风险范围、股票池、估值字段的经济有效性须由真实数据接入后的检验决定。

## 优化器结构及失败语义

变量为 a=w−w_b、y=Bᵀa、买入 u、卖出 v；a−u+v=w_current−w_b。二次项只含对角 D 和 K×K 的 F。线性双边费率分别乘 u/v；另有非负对角二次 impact 入口。跟踪误差采用 SOC ||[F½y, √D a]||≤limit。

`Constraints` 每项可关闭；满仓或现金通过 invested 配置；行业与风格矩阵由调用方传入，与风险模型因子名解耦。单边换手=max(买入和,卖出和)，满仓且总权重不变时恰为 L1/2；现金流入或流出时仍限制实际交易的一侧。不能买只限制增持，不能卖只限制减持；同时冻结进入等式。

`optimize` 只接受 optimal/Solved/solved，独立检查所有经济约束、有限数、原始目标及原生残差/间隙。`optimal_inaccurate` 不成功。显式放松规则指定目标上界，不能借“放松”收紧、打开已关闭约束，也不能放松预算、只做多和方向冻结。放松后报告约束名、前值、后值、增加量和每次状态。最终失败返回 previous，并明确 success=False、status=fallback_previous，附原约束违反量。

`optimize_sequence` 按前期结果递推，要求证券顺序不变；目标四数组不导入 panelbt。`prune_and_resolve` 将可卖出的极小权重资产上限设为零重解，失败保留原解并单独报告后处理失败，不硬删冻结资产。

## 设计疑问

| 原文/要求 | 问题 | 处理和理由 |
|---|---|---|
| 五种调用方式在“约束齐全”时比较 | OSQP 仅支持线性约束 QP，无法表达 TE 二阶锥 | QP 共同域比较五种；SOCP 比较三种 Clarabel 路径，OSQP 标 unsupported，绝不偷去掉 TE 或自己写非线性求解算法 |
| Rust 对多个调仓日并行；上一期解是下一期持仓 | 存在真依赖，不能独立并行而保持语义 | 200 期正式基准顺序；Rust 四线程另测共享风险矩阵的独立问题吞吐，不能作为依赖序列的耗时 |
| t 日收盘前已知字段 | close 和当日估值仅在收盘时可知 | asof 明确为收盘完成后，预测下一日；不保证可按当日收盘价成交 |
| USE4/CNE5 与 A 股申万行业 | 公开模型用 GICS，且 CNE5 有更多财务和预测数据 | 采用用户指定申万哑变量及八类风格；未知行业显式列，不声称 MSCI 数值复刻 |
| 公开 CMRA A3/A4 | A3 将 Z 定义为 log 累计收益，A4 再 log(1+Z)，极端亏损时可能无定义 | 使用累计 log 收益范围；标明这是自定修正，避免隐式截断到任意大负数 |
| 缺失样本和 PSD | 成对删失的样本协方差可能不是 PSD | 使用逐列有效权重标准化的缺失零中心 Gram，不采用成对删失，也未声称实现 U 的 EM |
| NW 日频风险 | USE4 的 NW 是较长预测周期的序列相关修正；严格次日方差不应自动含未来跨日协方差 | 默认按任务采用 USE4S 的日单位长期风险；要严格次日 IID 方差可设所有 lag=0；TE 单位必须与所选模型一致 |
| 缺省无风险收益 | 任务输入未要求 rf | 可选 risk_free 字段；缺省 0 并明确记录，不暗接外部利率 |
| 缓存“增量追加一天” | 全量重算顺序和缓存恢复的浮点归约容易不同 | 采用相同有限窗口及固定归约顺序；逐位一致测试覆盖 B/F/D；磁盘状态恢复仍读旧快照，性能限制如上 |
| 模型和验证应无 N×N 股票协方差 | LW 和 PCA 常见实现显式生成 N×N | 用 N×T 的加权样本因子、时间 Gram 分块和 LinearOperator，股票协方差不展开；完整收缩的 K=T+1 可能很大，优化一般用截断 PCA |

## 验证与性能结果

结果由 `bench/risk/results/` 的 JSON 和日志保存，最终数字见 `docs/RISK_BENCHMARKS.md`。合成报告使用 T=3000、N=500、K=13，前 2000 日拟合、独立后 1000 日检验；固定真值暴露是回归/协方差估计的隔离测试，行情描述子的金融含义需真实数据验证，不能从这项合成检验推出。

## 根目录共享文件改动

行号和增量已由 diff 核对（详见文末）。仅添加 scipy/cvxpy/clarabel/osqp/portopt 运行依赖、单独 risk-reference 依赖组、portopt workspace source、Rust 工作区两个成员，以及风险相关 make 目标；锁文件仅为这些必要依赖解析。

- [05:45] AShare — numpy/Rust 完整规模内核对拍、独立留出样本还原和 240 个双求解器问题通过；首轮 make check 90 项全绿，补充无效解/时序/热更新边界后增至 94 项。
- 发现连续调仓的数值边界传递问题：合法容差内的上一期权重若略超上限，下一期不能卖时会出现真正冲突的窄区间。修复为先独立核验原解，再精确修回个股、主动和方向边界，预算用有余量的股票补齐，最后重新验全部约束。不会用裁剪掩盖超过容差的错误解。
- 另将锥变量改为 N 倍权重，目标等价缩放为 1000N，返回时除回 N；Clarabel 静态正则化为 1e-10。数值预处理不改变经济问题或验收阈值。旧试验 JSON 保留，不能拿含回退的耗时当有效解性能。

## 最终实现补充

- 单日回归采用带条件数检测、零列裁剪和 SVD 回退的 numpy BLAS Gram/KKT；批量回归采用 Rust Rayon。不能拿单线程 Rust 与只用 SVD 的 numpy 比较后直接推广到单日追加。基线同时保存 numpy 1/4/15 线程及真实逐日变化暴露的小块测量。
- 基础收益统一为简单超额收益；动量与 CMRA 单独还原 `log(1+r)-log(1+rf)`，不能使用 `log(1+r-rf)`。DASTD/CMRA 的 252 日窗口不随可配置的 beta 窗口改变。简单收益接近或低于 -1 时 log 下界为 -0.999999999，这是自行决定的数值保护，不是 MSCI 参数。
- 最终数值缩放为：锥变量=N×原变量，目标=1000N×原目标。静态正则化默认 1e-10；只在非最优数值状态时，依次尝试关闭均衡、正则化 1e-12、关闭均衡且 1e-12。容差不变。失败不是最优，状态序列进入 diagnostics。
- `auto` 先走实测最快的原生候选；遇到未证明无解的数值失败，使用独立 CVXPY 参数化建模重新表达同一个问题。CVXPY 抛出的 SolverError 也显式标记，不会返回上一轮缓存解当作成功。OSQP 的 inaccurate 状态和成功状态但残差超限，都不计成功。
- 一个最优解先检查原始约束，再修正容差内的边界浮点误差，再次检查；这防止下一期方向冻结把微小边界误差变成真实冲突。没有放宽任何断言或验收阈值。
- 实际面板检验的组合只使用当日已上市且有合法市值的覆盖证券；因子组合使用完整 B 的伪逆。实现收益按下一日合法收盘价估值，停牌仍可有有效估值收益；缺价不填零，组合持有缺价证券时该日检验样本缺失。样本窗口数量明确输出，避免报告看似有统计意义的空样本结果。
- 增量恢复只读取此前快照的状态数组，直到前一日才读取完整 B/F/D；仍需扫描前缀哈希和历史状态，并非 O(1) 检查点。
- 稀疏目标转换可传 `columns` 映射到全市场下标，各行自动排序；days/indptr/cols/weights 的布局与 panelbt.Targets 一致，未导入 panelbt。

## 验证记录

- 最终未变异基线和源码恢复后都通过 `make dev && make check`，恢复后 112 项测试通过。日志：`bench/risk/results/mutation-baseline-check.txt`、`mutation-restored-check.txt`。
- 15 个变异均以断言失败捕获；编译失败、导入失败不算杀死变异。逐个失败日志、原始及恢复 SHA-256 见 `bench/risk/results/mutations.json` 和 `mutation-00.txt`…`mutation-14.txt`。没有通过跳过或放宽测试完成。

| 变异 | 捕获测试 | 结果 |
|---|---|---|
| 漏特质方差 D | 闭式解 | killed |
| 用绝对权重代替主动权重 | 主动风险/零 alpha | killed |
| 把单边换手算成双边 | 换手和方向 | killed |
| 冻结买卖方向反转 | 换手和方向 | killed |
| 少约束最后一个行业 | 行业和 TE | killed |
| 无解回退不打失败标记 | 无解和显式放松 | killed |
| WLS 用 cap 而非 sqrt(cap) | Rust 与独立 numpy 参考 | killed |
| 行业约束不用 cap 加权 | Rust 与独立 numpy 参考 | killed |
| 暴露读取下一日市值 | 所有未来字段扰动、前缀逐位一致 | killed |
| NW 滞后项改为负号 | Rust 与独立 numpy 参考 | killed |
| 用 B_t 解释 r_t | 独立检查缓存中的因子收益 | killed |
| 单日 numpy WLS 用 cap 而非 sqrt(cap) | Gram/KKT 与独立 SVD 对拍 | killed |
| 单日 numpy 行业约束不用 cap 加权 | Gram/KKT 与独立 SVD 对拍 | killed |
| numpy NW 滞后项改为负号 | 独立解析滞后协方差 | killed |
| 遗漏 TE 二阶锥 | 跟踪误差约束实际活跃的解 | killed |

双求解器随机问题 240 个，覆盖计数：无解 12、显式放松 12、冻结 228、换手活跃 212、相切边界 16；代码断言每种分支计数大于零。成功解的目标差 <2e-8，权重 atol=3e-5/rtol=1e-5。五种调用路径均与 Woodbury 闭式解对拍，权重 atol/rtol=2e-7。Ledoit–Wolf scaled-identity 在三种 T/N（100/8、40/80、500/20）下对拍 sklearn：收缩强度 atol=1e-12，协方差 rtol=1e-11/atol=1e-13。完整样本常相关目标的有限 N 修正是本项目定义，未声称与 sklearn 不存在的对应类一致。

## 调用示意

```python
from ashare.risk import FundamentalConfig, model_series
from ashare.portfolio import Constraints, PortfolioProblem, optimize, to_targets

# source 满足 PanelSource；历史追加时保持 version，修订历史则更新 version。
config = FundamentalConfig(industry_codes=tuple(range(31)))
model = next(model_series(source, config=config, cache=".cache/risk-models",
                          start=len(source.dates)-1))
problem = PortfolioProblem(alpha, model, benchmark, current,
    constraints=Constraints(max_weight=0.01, active_bound=0.005,
                            turnover=0.1, tracking_error=0.005),
    buy_cost=0.0003, sell_cost=0.0008, can_buy=can_buy, can_sell=can_sell)
result = optimize(problem)  # 默认 auto；必须检查 success/status/relaxations。
targets = to_targets([rebalance_day], [result.weights], columns=universe_columns)
```

日期下标由调用方对齐数据层和回测的交易日；当前持仓、基准、alpha、买卖掩码和风险模型必须使用同一证券顺序。行业/风格偏离约束启用时，另传 `industries`/`styles`；没有隐含依赖因子名称解析。风险接口可单独接入外部 B/F/D，成长/杠杆等外部日频暴露通过 `model_series(external=..., external_names=...)` 传入。

根目录改动的最终位置：`Cargo.toml:2` 增加两个 workspace 成员；`pyproject.toml:5` 添加运行依赖、`:17` 添加 portopt source、`:20` 将对照依赖组纳入开发、`:21` 定义 risk-reference；`Makefile:4` 增加 PHONY，`:27`/`:30`/`:33` 增加三个目标。锁文件为必要解析增量：Cargo.lock +624/-2 行，uv.lock +615 行。没有改动其他会话的源码、既有 test/*.py 或 DESIGN/SELECTION。

## 尚待真实数据验证的部分

合成还原测试证明的是数值估计和预测尺度，不能证明行情描述子的经济解释或真实 A 股效果。真实数据协议尚待合并后的数据层接入；没有调用 Tushare。公开文档无法完整确定缺失、ESTU、结构化可靠性和商业特征调整标定，这些自定选择已逐项列出。完整统计模型 components=None 的因子数接近 T，适合还原/研究；正式优化通常选择 PCA 截断。缓存磁盘占用和长历史恢复扫描、数据层的 point-in-time 正确性，以及共享负载下较小的性能差距，都需要真实全历史和空闲机器复测。

基准元数据校正：numpy 的 `show_config` 明确为 Apple Accelerate，`threadpool_info()` 是空列表。原脚本的“请求 BLAS=1”不能作为实际线程数证据，已将 JSON 的实际值改为 null，并保留请求值和修正说明；所有耗时原样保留。Rust 1/4/15 和 Python 线程池 1/4/15 是明确配置值，BLAS 为系统管理、未核实。参见 [threadpoolctl 的 Accelerate 控制问题](https://github.com/joblib/threadpoolctl/issues/136)。共享负载和这一线程控制限制均应在空闲复测时记录，不能声称是严格单线程 BLAS 对比。

- 全量逐日生成已将上市天数的前缀累计移到循环外，每个日期复用当时及以前的整数累计值，避免逐日扫描全历史导致 O(T²N) 重复工作。独立描述子入口仍可自行计算；两条路径的暴露与 ESTU 逐位相同。`pipeline.json` 另存前缀预处理耗时和复用后的描述子耗时。

CVXPY 调用方式的补测：默认参考形式直接保留 D/impact 的对角 Hessian 和 F 的 K×K 二次型，F 或 λ 变化时重建图，避免人为增加平方项辅助等式。参数化 Hessian 的 DPP 首次编译在 N=3000 的实测约 40–46 秒，且会影响会话热启动；禁用这部分 DPP 编译（`ignore_dpp=True`）后，同规模冷启动降到约 1 秒，保持问题对象和 OSQP 原生会话复用。原来的平方项展开形式保留为 auto 数值失败时的独立重表达。两个形式、变动 F/λ 和非零 impact 另有严格对拍；没有修改问题、收敛容差或成功定义。旧形式与新形式的原始试验均留档，最终 CVXPY 表使用 `optimizers-cvxpy-final.json`。

短窗口分派依据：默认历史上限是 1008 天，补测发现 K=45 时完整因子协方差 numpy 约 0.00091 秒、Rust 约 0.00155 秒；T=4000 时 Rust 较快。因此 T≤1008、K≤64 的因子方差/协方差走 numpy，大样本和大截面特质风险走 Rust。K≤64 是自行选择的分派边界；完整数字和重跑脚本为 `windows.json` / `window_benchmark.py`，没有把 4000 天的结果机械推广到短历史。

EW 协方差使用归一化权重的加权总体矩（ML 口径），不另外乘 `1/(1−Σp²)` 的无偏校正；这与等权完整样本的 sklearn LedoitWolf 对照口径一致。该有限样本选择、回归残差不作额外自由度放大，均属于本实现的明确取舍；还原容差说明中计入了约 K/N 的残差自由度偏差。

第三轮扩充至 14 个源码变异：在原来的 11 个基础上，增加正式单日 numpy Gram 路径的错误回归权重、错误行业权重，以及短窗口 numpy Newey–West 负号变异。当轮 14 个全部 killed，原始/恢复源码 SHA-256 相同，恢复后的完整检查为 106 passed。最后一轮再加入遗漏 TE 锥的变异并重跑全部 15 个，最终记录以本页「验证记录」和 `mutation-restored-check.txt` 为准。外部暴露未来值改动的逐位不变性，同时覆盖基本面型和统计型。

合成检验增加了对因子旋转不敏感的公共协方差误差：用 K 维 Gram 矩阵计算 `||B F Bᵀ−B_true F_true B_trueᵀ||_F`，不构造股票协方差。基本面型为 7.886%，统计型为 10.062%，均低于依据协方差抽样误差量级设定的 16% 容差。这样 PCA 无需与真值强行逐列对齐，也能检验其公共风险的还原程度。行业收益约束中的行业市值在当日有效估计池上求和。

- [07:20] AShare — 统一接口、缓存时点、两类风险模型、五种调用路径及回退/放松全部实现；14 个变异捕获且源码恢复，106 项完整检查通过。完整大规模 200 期求解基准仍在执行，所有失败均显式记录，未外推耗时。

流动性短史修正：IPO 前的 turnover 即使误填了数值也不参与描述子；有效日不足完整季度/年度时，以有效日数折算月换手（21×有效日均值），而非把未上市日期当零换手。完整窗口仍精确还原 21/63/252 日及 1/3/12 月分母；这项缺失/短史推广是自定规则，少于 min_descriptor_days 仍走行业＋Size 填补。专门测试验证了 IPO 前数据扰动不影响暴露，恒定日换手不会因上市时间短而被误判为低流动性。

市值缺失时，Size 的原始值保持缺失，进入行业填补，而不是把 1 元的计算保护值当成真实市值暴露。若缺失样本所属行业在估计池没有任何有效观测，行业＋Size 回归的未识别零系数也不当成零原始值，而回退到全估计池的截距/Size 回归。这些是明确的自定填补策略；已有单独断言覆盖没有观察样本的行业。

上市首日不使用未上市前的占位价格计算收益：`adjusted_returns` 要求当日和前一日均已上市。专门测试扰动 IPO 前价格，确认首日收益保持 NaN、后续合法收益不受影响。

最终选型采用完整 200 期序列，而不是不改变 alpha 的热启动单次：N=1000 的 QP 选 Python Clarabel、SOCP 选 Rust；N=3000 两者选 Python Clarabel；N=5000 两者选 Rust。其他规模取最近的已测 N，等距离取较小值；这一分派边界是工程选择。单次 OSQP 很快，但连续 alpha 变化后迭代和失败重试明显拖慢，且 N=3000 有解被自检拒绝，不能用这些失败作为速度优势。自动路径另跑六组各 200 期，全部成功；N=1000/SOCP 有一期间原生数值失败，经同容差 CVXPY 等价重表达恢复。失败前后的诊断保留在 `optimizers-auto-final.json`。

正式可比较的五路径基线已冻结为 `optimizers-baseline.json`，36 行含 30 个路径/问题组合（6 行 OSQP/SOCP 不支持）与 6 行独立 Rust 并行吞吐；每行记录原始 JSON 来源。CVXPY 使用最终优化后的建模形式，单次、冷启动和实际 200 期全部留档。较早的建模形式试验只作诊断，不覆盖正式表。`make risk-bench` 写入新目录并与这一冻结基线比较；不会混入旧试验。

- [08:02] AShare — 五路径、三个规模、完整 200 期优化基线及独立并行吞吐全部完成；最终 auto 的 1200 期全部通过自检（一次等价重表达恢复）。15 个真实源码变异全部命中，源码哈希逐字节恢复，最终 `make dev && make check` 为 112 passed in 6.56s；正式 JSON、中文基准报告和公式/自定参数台账已落盘。

最终数值对拍误差另存 `comparisons.json`：五路径 Woodbury 权重最大绝对误差不超过 6.241e-11；三组 sklearn 对拍的收缩强度误差不超过 3.331e-16，协方差元素误差不超过 6.939e-18。240 个双求解器问题的覆盖日志为 `comparisons.txt`。

早期平方项展开/DPP 大规模诊断进程在 N=5000/SOCP 结束前以退出码 137 终止，现有信息无法判定原因；其未完成项没有外推或列为正式测量。最终紧凑二次型/`ignore_dpp=True` 的 CVXPY 补测已完整跑完三个规模、各 200 期，冻结基线的 CVXPY 行全部来自这轮补测。早期 OSQP 展开形式对照也已结束，仅保留作诊断。


## 第二轮审查修复（2026-11-12）

- [09:14] AShare — 逐项复现审查的内存、冻结边界、数值失败放松、库名碰撞、回归病态、分组污染和 y 约束性能问题；重跑原测试上的缓存版本键变异，44 项仍通过，确认测试缺口。复现输出仅保存在被忽略的 `.cache/risk-round2/reproduction/`。
- 原回退 N=1000/SOCP 实测峰值 8.11 GB；冻结冲突两个例子均失败；MaxIterations 后原代码关闭换手，实际单边换手 0.649（原限 0.02）；近似共线 Rust 残差差 7.98e-4、SSE 相对偏差 2.20e-4；市值分组污染差 5.6%。约束改写复测 N=1000/3000/5000 为 67→31、200→98、368→199 ms。
- 修复后默认单次/递推调用 Python Clarabel，独立批次使用 Rust `solve_many`；不再按 N 在同一求解器的两个封装间切换。P/A 和锥布局不变时缓存结构，仅更新 q/b；风险、矩阵或边界行布局变化会重建。CVXPY 两种形式都禁用 DPP 展开缓存。
- 冻结自动放宽只处理方向冲突：不能卖时放宽必要的个股上侧，不能买时放宽必要的主动下侧。可以卖出的超限持仓仍须减仓。逐只记录在 `bound_relaxations`，与用户配置的全局 `relaxations` 分开；预算、只做多、行业、风格、换手和 TE 不自动放宽。只有明确的 primal infeasible 才进入配置放松。
- 全量回归按 64 日分块，追加一天使用同一个 Rust 函数。正规方程先去空列、检测条件；病态时对约束的正交零空间做 SVD，求最小范数解。Rayon 池按线程数复用，`portopt.set_num_threads/get_num_threads` 控制默认值。
- 缓存 schema=3：历史只存压缩、float64 的 F/D 和递推状态；只留一份最新 B，历史 B 按需从输入重建。无损方案不改变全量/增量精度。数据版本或参数变化清理旧代，历史前缀变化清理旧尾部；提供占用统计和清理入口。正在测完整 4000 日占用与读取。
- 描述子的 beta/HSIGMA、DASTD、momentum、三种 liquidity 窗口改用固定窗口矩递推；全量和追加从同一前缀起点播放，浮点顺序相同。截面标准化、正交化保留直接算法；随后已补上 CMRA 累计端点递推。新递推和原窗口实现另做数值对拍。
- 新增分组不变性测试最初在 N=60/120 比较，跨过了 NumPy/Rust 分派边界，差约 8.7e-19；现用 N=80/160 固定同一个内核，以隔离分组成员影响，仍保持逐位相等断言。数值内核间的独立对拍另有覆盖，没有放宽断言。

### 第二轮审查逐条处理

以下记录覆盖本轮最终实现；上文第一轮的按规模选型、每天保存 B、逐日 NumPy 回归和旧结果文件名是历史记录，已经被这里的方案取代。第一轮一次性日志不再入库；保留必要旧优化器数字作为新基线中的前后对照。本轮 OneContext 检索正常执行但未找到历史会话，处理依据为用户提供的审查和本地源码。

| 项目 | 结论 | 实现及守护证据 |
|---|---|---|
| P1-1 | 已修；复现 8.11 GB | `reference.py` 两种形式统一 `ignore_dpp=True`；`test_auto_fallback_peak_memory` 子进程 N=1000 <1 GB，X02 变异命中。实测新回退 155/185/216 MB（N=1000/3000/5000）。 |
| P1-2 | 已修；两个冻结冲突原例失败 | `problem.box_limits/weight_bounds` 按方向自动扩充个股界，`bound_relaxations` 逐只报告；五求解路径的 `test_directional_box_relaxation` 含多只同时冲突；X04–X06 命中。 |
| P1-3 | 已修；原 MaxIterations 导致放宽 | `optimize.py` 只有明确 primal infeasible 放松。数值失败先替换等价求解形式或重试容差，再明确回退 current；四种错误状态和保持原约束重试测试，X03 命中。 |
| P1-4 | 已修；原四条 filename collision 已复现 | `[lib] name=portopt_ext`，PyO3 显式 `_core`。同一 target 构建 panelbt→portopt→panelbt，再构建实验 spike，三者分别导入自己的库。这个 worktree 没有 paperbroker，不能替不存在的包声称完成构建；主干已有唯一库名与本名不冲突。 |
| P1-5 | 已修；原 C1 变异下 44 项仍绿 | 新增 `test_risk_review.py`、`test_portfolio_review.py`，手算矩、手算两资产非对称成本、独立 KKT/对角重建、状态和递推断言。审查 46 项全部捕获，加本轮 29 项共 75 项；下表列每项。 |
| P1-6 | 已修；原污染约 5.6% | `specific_variance` 结构化拟合与市值分组限定当天 ESTU；新增非估计股票不改变已有 ESTU 的方差，严格逐位断言；X07 命中。 |
| P1-7 | 已修；原流水线未调用 Rust | `series.py` 分块回归，`engine.cross_section` 也调用同一 Rust 函数一天；空列裁剪、条件检测、正交零空间 SVD。近共线/重复列、空行业、单股行业、线程数/分块/追加逐位一致均有测试；X16/X21 命中。 |
| P1-8 | 已修 | 删除 results 中 77 个旧文件，只保存 `baseline.json`；所有重跑、变异、检验默认输出到 `.cache/`。冻结器去掉无关本机路径、PID、逐组合目标数组；脚本不依赖审查的临时目录。最终文件大小清单见下文。 |
| P1-9 | 已修；原配对提速约 2 倍 | `canonical.py` 行业/风格匹配 B 列后约束 y；也可显式指定 `industry_factors/style_factors`。任意不在 B 中的外部暴露保留合法 N 行。P/A/锥布局不变时只更新 q/b；新五路径全量基准、N 行/y 行配对、矩阵非零元和 X24/X25 变异守护。 |
| P2-1 | 已修 | `kernels.psd` 与 `RiskModel` 输出前精确对称化；QP/TE 使用同一份 F；外部近对称 F 的手算平均断言，X15/X22 命中。 |
| P2-2 | 已修 | 默认按调用形态选择：单次及递推为 Python Clarabel，独立并行问题为 Rust；删除按 N 的选型表。两者都是 Clarabel，不以微小计时差断言某种封装普遍更快。 |
| P2-3 | 已修 | 无损压缩历史状态、按需 B、最新递推检查点、一代清理/统计入口。4000 日实际 754,353,363 字节，低于 2 GB；当前与重建 B/F/D 逐位一致，缓存键及清理测试覆盖。 |
| P2-4 | 已修 | `set_num_threads/get_num_threads` 全局默认，显式 threads 可覆盖；Rust 按线程数复用池；线程设置、1/4 线程逐位一致测试，X17 命中。 |
| P2-5 | 已修本分支可做部分 | 新 `rolling.py` 将 beta/HSIGMA、DASTD、momentum、流动性做固定窗口递推。末日 44.5→3.61 ms，完整流水线与阶段剖析留档。CMRA 随后也已增量化；截面标准化、填补和正交化需读取当日截面，后续 panelops 融合需合并后另测。 |
| P2-6 | 已补对照 | `review_comparisons.py` 独立写 LW2004 Appendix B，四种配置各 60 次蒙特卡洛；含自定有限 N 修正的本实现和论文公式损失接近，不宣称两者是同一估计量。 |
| P2-7 | 已修短史；明确完整模型用途 | `statistical_model(min_history=60)` 向成熟股票截面方差先验收缩，保留 PSD；仅 3 个小收益的例子原方差 6.67e-9，新方差 3.73e-4（成熟中位数 3.92e-4）。`components=None` 明确是 T+1 因子的研究接口，正式默认 45 PCA；X11 命中。 |
| P2-8 | 已修 | 单股行业的当日精确零残差不作为方差样本，置 NaN；零样本方差也不取得完全历史可靠性。结构化先验提供风险；单股行业与零残差手算测试，X08/X09 命中。 |
| P2-9 | 已修 | 特质状态偏差权重限定上一日 ESTU 且当日收益有效，与该次回归一致；未上市/停牌者权重为零；VRA 手算乘数 4/9 与 X10 命中。 |
| P2-10 | 不在本分支定行情填补政策；已列对接事项 | 见下面“数据层对接”。保留缺失收益，不跨缺口人为制造日收益。 |
| P2-11 | 保留任务书序列定义；已列对接事项 | 见下面“回测引擎对接”。实际持仓漂移与中断决策归调用方，数值自检仍严格抛错。 |
| P2-12 | 已补 | `validate.bias_summary` 同时报重叠/不重叠；合成检验加真模型，另跑六种子及 IID 20,000 个不重叠窗口对照，覆盖率 94.84%。 |
| P2-13 | 已补 | 1e-9/1e-6/1e-5 的 OSQP 15 期配对表见基准文档；放宽求解器容差的结果单列原始违反量，没有降低正式 1e-7 自检标准。 |
| P2-14 | 已修 | T1/T5/T7/W4/S1/O10 的独立期望值与实际触发断言齐备，六项变异全部命中。 |

没有以“复现不出”关闭任何 P1。P2-10、P2-11 是用户指定保留到对接阶段的事项。

### 新逻辑的公式与自定参数

- 原公开公式及出处沿用前表。本轮新增的工程数值阈值和短史保护是本项目自定，不归于 MSCI：Rust 正规系统最小/最大奇异值比阈值 `1e-12`，回退 SVD `rcond=1e-12`；零空间使用正交 Householder，避免非正交约束消元改变最小范数解。空行业列归零；空有效设计返回零系数、原残差。近共线残差对独立 NumPy 的最大差从约 7.98e-4 降为 1.82e-8（病态系数本身不宜逐项比）；线程数不影响同一日计算顺序。
- 递推有限窗口矩为 `S_t = decay*S_(t-1) + v_t - decay^window*v_(t-window)`，`decay=2^(-1/half_life)`；缺失值贡献为零，另递推有效计数和权重分母。beta 的市场方差低于 `1e-20` 时置零，方差抵消产生的微小负数夹为零。momentum 在 skip 后的终点应用同一递推。全量与追加都按日期顺序更新同一个实现；它与原窗口归约在舍入位不同，独立窗口对照用原先严格误差量级，不能声称与第一轮输出逐位相同。
- 统计短史 `gamma=min(count/min(min_history,T),1)`，count<2 时 gamma=0；U 乘 sqrt(gamma)，D 为 gamma*D+(1-gamma)*截面先验。完整短样本以可用 T 为阈值保留标准估计量/sklearn 的含义；新上市个股相对同窗成熟股票受到保护。默认 min_history=60、无成熟样本时 0.02² 先验均自定。
- 特质风险 γ 仍按结构化历史天数混合，但方差≤floor 的残差不被视为完整信息；单股行业当日观测缺失处理是本轮自定保护，不是商业 Barra 的逐值复刻。
- 冻结个股界扩大到 current 是用户本轮决定；只解除方向冲突，不默认放松行业、风格、TE 或现金预算。数值重试把求解容差收紧到 `min(original*0.001,1e-10)`；最优状态和独立违反量验收标准保持不变。

### 缓存方案与增量开销

严格只为“下一日追加”需要的是最近 history（默认 1008）日的因子收益、特质收益、偏差统计，上一日 B/base_F/base_D/ESTU/cap，以及最新滚动描述子矩。历史 F/D 是已生成预测的查询缓存，可从原数据重算；历史 B 体积最大且可重建，已移出逐日缓存。当前保留各日的小型回归状态以支持从任意历史截点续算和局部历史修订，不再保留多代。进一步只保留最后一个历史窗口可减少状态量，但会增加历史截点重算成本；目前实存 0.754 GB 已满足目标。

浮点选择实测：5500×45 的 B，float64 原始 1,980,000 字节、压缩 539,568；float32 原始 990,000、压缩 269,276。读回分别约 1.36/0.70 ms；float32 暴露最大误差约 1.88e-7，单个等权组合方差相对差约 5.80e-12。这不是所有组合的误差保证，正式方案因此仍为 float64 无损。4000 日 cache 总量含每日 F/D/状态、generation 标记和单份最新 B/描述子检查点；不包含调用方本来就持有的行情。

最新整模型读回中位数约 29.64 ms，只读状态约 0.55 ms；历史日 B 按需重放前缀（本次查询耗时 3.86 s，其中暴露约 1.00 s）。追加一天约 3.31 s，其中新暴露 4.17 ms、单日回归 1.47 ms、协方差 63.7 ms，主要剩余成本是验证完整输入前缀和读取历史状态。没有把该入口称为 O(1) 追加。缓存 `stats/clear` 只处理自身命名文件，测试确认不删除调用方的其他 NPZ。

### 数据层、回测引擎对接与设计疑问

| 原文/问题 | 本轮处理和理由 |
|---|---|
| “复牌当日的收益被丢掉” | 若停牌缺 close，当前相邻两日收益为 NaN。数据层需决定是否提供前收填充的价格序列、复牌跨日收益的归属及累计天数；不能由风险层把跨多日收益暗当一日观测。接入时增加对应缺口/复牌测试，并显式区分价格占位与可交易状态。 |
| “上一期的解是下一期的当前持仓”与实际持仓漂移 | `optimize_sequence` 按任务书递推目标，用作研究和基准。回测/模拟盘接入应逐期调用 optimize，并传引擎结算后的实际权重、现金和交易掩码；成交不足、费用、价格漂移后不可继续用旧目标代替实际权重。 |
| “VerificationError 会中断整个序列” | 保留异常，不将自检失败包装成可接受目标。调用方对接时明确选择中止、保存中间结果、报警并保持真实持仓；若选择保持持仓，也必须记录失败和违反量。 |
| 库名验证要求包含其他分支包 | 本 worktree 只有 panelbt、portopt 和 bench/spike 三个原生扩展；都在同一 target 验证。没有改 panelbt 的旧 `_core` 库名，也没有读取或构建别的 worktree。paperbroker 的联合构建留到合并后。 |
| “连续调仓只更新向量” | 当 B/F/D、锥布局、固定上下界行模式不变时复用。15 期实测 N=1000 复用 7 次、0.535→0.461 s；N=3000 的固定行模式每期变化，无复用且无明确收益；N=5000 复用 1 次、4.088→3.994 s，差异接近共享负载噪声。风险矩阵每天变化时必须更新矩阵/重建，不能复用旧风险。Clarabel 没有迭代点热启动接口，OSQP 有；没有声称 Clarabel 做了热启动。 |
| 全局 worklog 与“只在本目录”冲突 | 遵守更具体的任务范围，阶段记录仍写本笔记，不越界写全局日志。 |
| 新鲜副本放哪里 | 放在本 worktree 的忽略目录 `.cache/risk-round2/fresh`，不借用主目录或其他 worktree；独立 venv、target、CARGO_HOME。Makefile 明确关闭 gitignore 发现规则并排除 .cache，避免副本在被忽略的父目录下导致 Ruff 检查零文件。 |

### 第二轮验收与共享文件

当前目录和全新副本均执行 `make dev && make check`：Ruff 62 个 Python 文件，cargo fmt/clippy/test 通过，pytest 各 158 项通过。最终输出：当前目录 `158 passed in 9.48s`，重新创建的全新副本 `158 passed in 41.37s`。新副本的 Hypothesis 数据库为空且构建冷启动；不把两次测试墙钟差解释成模型性能差。

本轮根目录只改 **Makefile 第 10、11 行**：两条 Ruff 命令添加 `--no-respect-gitignore --extend-exclude .cache`。Cargo.toml、Cargo.lock、pyproject.toml、uv.lock 与本轮开始的审查哈希相同；它们相对主干仍有第一轮的必要改动，不能误记成本轮新增。没有修改范围外源码或既有 test/*.py；没有 git commit/push/配置操作。

- [09:46] AShare — 第二轮核心修复及两套完整环境检查通过，157 项测试；当阶段 71 处变异全部捕获且源码/扩展恢复，缓存 0.744 GB；后来补入 CMRA 增量，最终数据见后文。


CMRA 的增量恒等式为 `range_j(q_t − q_(t−21j)) = range_j(q_(t−21j))`，j=1..12，q 为逐日 log 超额收益累计量；保持 253 个前缀的环形缓冲及有效计数。窗口中任何缺失仍使原始 CMRA 缺失。这是对原 CNE5 描述子公式的代数变换，252/21 参数沿用原台账，不引入新经济参数。检查点因此增加约 10.7 MB，仍只存末日一份；schema 升至 3，避免旧检查点缺少新状态。

最终全量 4000 日×5500 只×45 因子为 79.43 s（15 线程），其中暴露 14.37 s、回归 0.64 s、协方差/特质风险 53.17 s。缓存前 3999 日再追加一天共覆盖 4000 个模型，所有 B/F/D 哈希逐位相同；分别用 4/1 线程、31/1 日分块，与全量 15 线程/64 日分块比较。完整缓存 0.754 GB；单次模型末日读回约 29.6 ms。

基准工具记下占用/读取数字之后默认清理自身的 4000 日模型缓存，避免时间戳目录重复存放约 0.754 GB；调查缓存内容可显式使用 `rebuild_benchmark.py --keep-cache`。业务 ModelCache 只按 generation/前缀失效或显式 clear 清理。

### 最终变异明细

审查原有 46 项 + 新增 29 项 = 75 项全部捕获。最终源码上使用此前已证实会失败的测试节点逐项核验，前后均运行完整 90 项专项测试；默认 `make risk-mutate` 仍对每个变异运行全套测试。最后两套完整工程各 158 项通过。编译错误、收集错误不计为捕获。每项恢复源码 bytes，7 个 Rust 条目同时核对扩展哈希、导入路径和恢复哈希。

CMRA 的 X26 最初在常数月内收益夹具下存活；加入逐日递增收益，独立手算 253..483 日的 log 收益之和后捕获，没有改变容差。W1 是数值等价但变慢的变异，用良态问题不应调用 SVD 的执行断言捕获。

| ID | 故障 | 结果 | 捕获测试 |
|---|---|---|---|
| T1 | 回归估计池用当日而非前一日 | caught；已恢复 | `test_previous_estimation_unknown_industry_and_history` |
| T2 | DASTD 窗口含 t+1 | caught；已恢复 | `test_rolling_descriptors_independent_windows_and_pipeline_chunks` |
| T3 | 动量窗口后移一天 | caught；已恢复 | `test_rolling_descriptors_independent_windows_and_pipeline_chunks` |
| T4 | 统计模型窗口含 t+1 | caught；已恢复 | `test_statistical_series_and_external_prefix[statistical]` |
| T5 | 因子收益历史窗口多含一天 | caught；已恢复 | `test_previous_estimation_unknown_industry_and_history` |
| T6 | 估计池偷看次日停牌 | caught；已恢复 | `test_previous_estimation_unknown_industry_and_history` |
| T7 | 贝塔的市场收益用当日市值加权 | caught；已恢复 | `test_rolling_descriptors_independent_windows_and_pipeline_chunks` |
| C1 | 缓存键漏 version | caught；已恢复 | `test_cache_keys_and_one_generation[version]` |
| C2 | 缓存键前缀摘要漏字段取值 | caught；已恢复 | `test_cache_keys_and_one_generation[values]` |
| C3 | 缓存键漏 config | caught；已恢复 | `test_cache_keys_and_one_generation[config]` |
| W1 | Gram 路径丢掉行业约束 | caught；已恢复 | `test_well_conditioned_gram_avoids_svd_and_psd_exact` |
| W2 | 标准化用等权均值 | caught；已恢复 | `test_standardization_and_missing` |
| W3 | 标准化的标准差用全体而非估计池 | caught；已恢复 | `test_standardization_and_missing` |
| W4 | 行业约束漏掉 unknown 行业 | caught；已恢复 | `test_previous_estimation_unknown_industry_and_history` |
| N1 | NW 协方差少一阶滞后 | caught；已恢复 | `test_nw_sign_and_missing_lag` |
| N2 | EWMA 不去均值 | caught；已恢复 | `test_nw_sign_and_missing_lag` |
| N3 | 半衰期用 e 而非 2（仅 numpy） | caught；已恢复 | `test_rust_kernels` |
| N4 | 因子波动率用相关系数的半衰期 | caught；已恢复 | `test_half_life_and_covariance_hand_moments` |
| N5 | 波动状态乘子开方后再乘协方差 | caught；已恢复 | `test_regime_multipliers_and_estimation_bias` |
| L1 | LW 收缩强度减半 | caught；已恢复 | `test_sklearn_ledoit_wolf` |
| L2 | PCA 特质方差漏掉收缩目标的对角项 | caught；已恢复 | `test_statistical_short_history_and_pca_diagonal` |
| S1 | 特质方差去掉下限 | caught；已恢复 | `test_structural_blend_zero_residual_and_floor` |
| S2 | 短历史不混合结构化模型 | caught；已恢复 | `test_structural_blend_zero_residual_and_floor` |
| S3 | 贝叶斯收缩权重写反 | caught；已恢复 | `test_synthetic_recovery` |
| R1 | 组合方差漏 F | caught；已恢复 | `test_risk_structure_and_derivative` |
| R2 | matvec 漏 D | caught；已恢复 | `test_risk_structure_and_derivative` |
| O1 | 预算约束写在主动权重上 | caught；已恢复 | `test_closed_form[clarabel]` |
| O2 | 买卖费率对调（仅实现） | caught；已恢复 | `test_asymmetric_cost_hand_solution[clarabel]` |
| O3 | 不能卖的冻结被丢掉 | caught；已恢复 | `test_turnover_and_direction` |
| O4 | 行业/风格偏离只约束上侧 | caught；已恢复 | `test_lower_exposure_bound[industries-True]` |
| O5 | TE 锥漏掉特质部分 | caught；已恢复 | `test_tracking_error_is_active[clarabel]` |
| O6 | 放松顺序反转 | caught；已恢复 | `test_relaxation_order_and_previous_holdings` |
| O7 | 求解后自检被关掉 | caught；已恢复 | `test_independent_postsolve_violation_rejection` |
| O8 | 只做多写在主动权重上 | caught；已恢复 | `test_turnover_and_direction` |
| O9 | 批量求解不递推持仓 | caught；已恢复 | `test_sequence_has_actual_recursion_and_prune_preserves_frozen` |
| O10 | 裁剪重解删掉不能卖的持仓 | caught；已恢复 | `test_sequence_has_actual_recursion_and_prune_preserves_frozen` |
| O11 | 无解回退不返回上一期权重 | caught；已恢复 | `test_relaxation_order_and_previous_holdings` |
| D3 | 双改：买卖费率对调（实现+cvxpy 参考） | caught；已恢复 | `test_asymmetric_cost_hand_solution[clarabel]` |
| D4 | 双改：冻结方向反转（实现+参考+自检+清理） | caught；已恢复 | `test_turnover_and_direction` |
| D6 | 双改：换手按双边（实现+参考+自检） | caught；已恢复 | `test_turnover_and_direction` |
| RU1 | Rust NW 方差滞后项漏系数 2 | caught；已恢复 | `test_rust_kernels` |
| RU2 | Rust 偏差统计用总体标准差(ddof=0) | caught；已恢复 | `test_rust_kernels` |
| D1 | 双改：NW 滞后项取负（numpy 两处 + Rust 两处） | caught；已恢复 | `test_nw_sign_and_missing_lag` |
| D2 | 双改：WLS 权重用市值（numpy 两处 + Rust） | caught；已恢复 | `test_regression_weight_and_industry_constraint` |
| D5 | 双改：行业约束用根号市值加权（numpy 两处 + Rust） | caught；已恢复 | `test_regression_weight_and_industry_constraint` |
| D7 | 双改：半衰期用 e 而非 2（numpy + Rust 两处） | caught；已恢复 | `test_half_life_and_covariance_hand_moments` |
| X01 | 日期轴不参与缓存键 | caught；已恢复 | `test_cache_keys_and_one_generation[dates]` |
| X02 | 回退重新启用 DPP | caught；已恢复 | `test_auto_fallback_peak_memory` |
| X03 | 数值失败错误进入放松 | caught；已恢复 | `test_numerical_failure_never_relaxes[MaxIterations]` |
| X04 | 个股冻结上限不自动放宽 | caught；已恢复 | `test_directional_box_relaxation[clarabel]` |
| X05 | 禁买主动下界不自动放宽 | caught；已恢复 | `test_directional_box_relaxation[clarabel]` |
| X06 | 缺失逐只放宽记录 | caught；已恢复 | `test_directional_box_relaxation[clarabel]` |
| X07 | 市值分组混入非估计池 | caught；已恢复 | `test_specific_groups_ignore_non_estimation_stocks` |
| X08 | 零残差不使用结构化先验 | caught；已恢复 | `test_structural_blend_zero_residual_and_floor` |
| X09 | 单股行业不屏蔽残差 | caught；已恢复 | `test_singleton_industry_residual_is_uninformative` |
| X10 | 偏差统计混入非估计池 | caught；已恢复 | `test_regime_multipliers_and_estimation_bias` |
| X11 | 短历史统计风险不收缩 | caught；已恢复 | `test_statistical_short_history_and_pca_diagonal` |
| X12 | 递推遗漏移出窗口项 | caught；已恢复 | `test_rolling_descriptors_independent_windows_and_pipeline_chunks` |
| X13 | 递推市场收益偷用当日市值 | caught；已恢复 | `test_rolling_descriptors_independent_windows_and_pipeline_chunks` |
| X14 | 不恢复递推检查点 | caught；已恢复 | `test_checkpoint_restores_only_one_day_and_clear_preserves_unrelated` |
| X15 | PSD 投影不精确对称化 | caught；已恢复 | `test_well_conditioned_gram_avoids_svd_and_psd_exact` |
| X16 | Rust 病态设计不走确定回退 | caught；已恢复 | `test_rust_minimum_norm_and_thread_determinism` |
| X17 | 忽略线程数设置 | caught；已恢复 | `test_rust_minimum_norm_and_thread_determinism` |
| X18 | 主动风险漏掉特质项 | caught；已恢复 | `test_closed_form[clarabel]` |
| X19 | 无解回退错误标成功 | caught；已恢复 | `test_infeasible_relaxation_and_fallback` |
| X20 | 主动风险写成绝对风险 | caught；已恢复 | `test_closed_form[clarabel]` |
| X21 | 正式回归绕过 Rust | caught；已恢复 | `test_pipeline_calls_native_and_degenerate_append[0.0]` |
| X22 | 外部 F 不对称化 | caught；已恢复 | `test_external_model_covariance_is_exactly_symmetric` |
| X23 | 复用时不更新约束向量 | caught；已恢复 | `test_infeasible_relaxation_and_fallback` |
| X24 | 回归到 N 维行业风格行 | caught；已恢复 | `test_factor_form_and_vector_only_reuse` |
| X25 | 漏掉最后一个行业约束 | caught；已恢复 | `test_all_industries_and_tracking_error` |
| X26 | CMRA 回看端点错一天 | caught；已恢复 | `test_cmra_endpoints_missing_and_checkpoint` |
| X27 | CMRA 检查点遗漏历史前缀 | caught；已恢复 | `test_cmra_endpoints_missing_and_checkpoint` |
| X28 | 清理缓存未删除旧文件 | caught；已恢复 | `test_cache_keys_and_one_generation[version]` |
| X29 | 历史快照重新保存大矩阵 B | caught；已恢复 | `test_cache_lossless_history_exposure_rebuild` |

### 最终原始检查尾部

当前目录：

```text
test/test_random.py .                                                    [ 93%]
test/test_rules.py ..........                                            [100%]

============================= 158 passed in 9.48s ==============================
```

全新目录（空 venv/target 重新开始）：

```text
test/test_random.py .                                                    [ 93%]
test/test_rules.py ..........                                            [100%]

============================= 158 passed in 41.37s =============================
```

- [10:19] AShare — 第二轮全部必修项及 P2 的本分支项完成；75 个变异捕获，两个独立环境各 158 项通过；全量 4000 日 79.43 s、追加 3.31 s、缓存 0.754 GB，五路径完整 200 期基线冻结。P2-10/11 及跨分支 paperbroker 联合构建留作合并对接。

### 最终未忽略新增文件及大小

下面是相对 git HEAD 的全部未跟踪、未忽略文件，包含第一轮已有文件，不把它们全算成本轮新增；本轮主要新增 cache.py、rolling.py、两个 review 测试及增量/结构/对照基准脚本。根共享文件本轮只有 Makefile 两行变化。

| 文件 | 字节 |
|---|---:|
| `bench/risk/baseline.py` | 2910 |
| `bench/risk/compare_kernels.py` | 3056 |
| `bench/risk/consolidate.py` | 2648 |
| `bench/risk/factor_kernels.py` | 2770 |
| `bench/risk/gram_benchmark.py` | 2244 |
| `bench/risk/mutation_cases.py` | 26464 |
| `bench/risk/mutations.py` | 5716 |
| `bench/risk/optimizer_benchmark.py` | 8910 |
| `bench/risk/pipeline_benchmark.py` | 3891 |
| `bench/risk/rebuild_benchmark.py` | 9044 |
| `bench/risk/report.py` | 17266 |
| `bench/risk/results/baseline.json` | 152053 |
| `bench/risk/review_comparisons.py` | 6420 |
| `bench/risk/run.py` | 3208 |
| `bench/risk/structure_benchmark.py` | 5214 |
| `bench/risk/varying_regression.py` | 3328 |
| `bench/risk/window_benchmark.py` | 2352 |
| `crates/portopt-core/Cargo.toml` | 142 |
| `crates/portopt-core/src/lib.rs` | 15998 |
| `docs/RISK_BENCHMARKS.md` | 13042 |
| `docs/RISK_NOTES.md` | 56503 |
| `packages/portopt/Cargo.toml` | 304 |
| `packages/portopt/README.md` | 1811 |
| `packages/portopt/README.zh-CN.md` | 860 |
| `packages/portopt/pyproject.toml` | 679 |
| `packages/portopt/python/portopt/__init__.py` | 5238 |
| `packages/portopt/src/lib.rs` | 4602 |
| `src/ashare/portfolio/__init__.py` | 876 |
| `src/ashare/portfolio/canonical.py` | 7267 |
| `src/ashare/portfolio/construction.py` | 4524 |
| `src/ashare/portfolio/optimize.py` | 7108 |
| `src/ashare/portfolio/problem.py` | 10133 |
| `src/ashare/portfolio/reference.py` | 10501 |
| `src/ashare/portfolio/solvers.py` | 7834 |
| `src/ashare/risk/__init__.py` | 463 |
| `src/ashare/risk/cache.py` | 7196 |
| `src/ashare/risk/data.py` | 2620 |
| `src/ashare/risk/engine.py` | 1837 |
| `src/ashare/risk/fundamental.py` | 16783 |
| `src/ashare/risk/kernels.py` | 6790 |
| `src/ashare/risk/model.py` | 4005 |
| `src/ashare/risk/rolling.py` | 8165 |
| `src/ashare/risk/series.py` | 11429 |
| `src/ashare/risk/statistical.py` | 6512 |
| `src/ashare/risk/validate.py` | 13354 |
| `test/portfolio/test_optimizer.py` | 10988 |
| `test/portfolio/test_portfolio_review.py` | 10223 |
| `test/risk/test_core.py` | 7919 |
| `test/risk/test_risk_review.py` | 19462 |
| `test/risk/test_series.py` | 10252 |
| `test/risk/test_validation.py` | 968 |

共 51 个文件，543,882 字节。results/ 仅含一份 baseline.json，无本机绝对路径。

## 合并后的独立复核（2026-11-12，主干 `4acb51c`）

结论：没有必须在继续之前修的问题。第一轮 9 条必修项逐条独立复现，确认已修；时点探测、增量追加与全量逐位一致、描述子递推与逐窗口重算一致（最大差 1e-12）都通过。真实面板试接（最近 300 日、随机 800 只）能跑通，量级合理。下面 6 条小问题和对接清单并入「接真实数据」那条线的任务书，其中标「验收项」的必须有测试守住。

### 遗留的小问题

| 编号 | 位置 | 现象 | 处理 |
|---|---|---|---|
| a | `risk/statistical.py` 的 `eigsh` | 最大的几个特征值完全相同时（收缩到纯对角、早期样本很少），同一输入连续两次调用给出不同的 B；注释里的「可重复」不成立。主干上重跑复现：310 个窗口里 65 个不同。真实数据抽样未触发 | 验收项：换成确定性的求解，或在特征值簇前截断并记入 metadata；加「同进程连跑两次逐位相同」的测试和谱平坦的夹具 |
| b | `risk/fundamental.py` 结构化特质回归 | 国家列与行业哑变量共线，最小范数解把国家效应分摊给各行业；所在行业没有可靠样本的股票（单股行业、未知行业）特质波动偏高，31 个行业时约 1.13 倍 | 验收项：回归里去掉国家列或加约束，空行业的股票走只含国家和风格的回退；加测试 |
| c | `portopt-core` 回归 | 两个风格列近乎重复（差 1e-9）时保留极小奇异值，得到一正一负的大额因子收益 | 因子构造端保证不共线；截断阈值是否放宽到 1e-8 在接数据时按实测定 |
| d | `risk/cache.py` | 快照被截断时直接抛错不自愈；内容被换成另一份有效快照会被静默使用；`stats` 会创建不存在的目录 | 已在 P-004 修复（`perf/risk`）：快照存键并核对，截断、换文件、非 zip、缺字段都当未命中；`stats` 不再建目录 |
| e | 测试缺口 | 复核人另做 25 处变异，存活 5 处（代码是对的，缺测试）：检查点被删后的倒回重放、换代清空、冻结放宽的幅度、冲击成本线性项、DASTD 半衰期 | 逐条补测试 |
| f | `portfolio/optimize.py`、基准表 | 指定后端时数值失败的重试只是收紧容差；基准表的「单次」是同一会话重解同一题，直接 OSQP 冷解约 77 ms 而不是 3.11 ms | 重试改为放宽到原容差或换等价形式；基准表改标签并补冷解一列 |

### 接真实数据时的验收项

1. **行业编码**：面板的 `industry` 是申万代码（801010…，0 为缺失），`FundamentalConfig.industry_codes` 默认 `range(31)`。直接把面板交给模型不会报错，所有股票都落进 `industry_unknown`。对接层做编码映射，并在「已知行业占比为 0」时报错。
2. **上市天数**：现在按传入窗口里的 `is_listed` 累加，只喂一段历史时前 60 日估计池为空。改读面板的 `listing_age`。
3. **亏损公司的 E/P**：`pe_ttm` 对亏损公司是空值（末日缺失约三成），现在被行业加市值回归填成平均值而不是负值。由数据层提供带符号的 E/P 或 TTM 净利润。
4. 字段改名（`total_market_cap`、`float_market_cap`、`turnover`），`pb`、`pe_ttm` 转 float64。
5. 收益是否有效看 `has_quote`，能否交易看 `tradable`，不单看 `is_suspended`。
6. `industry_source=3`（回填）在 2014-02-21 之前普遍存在：接受或按来源过滤，写进 metadata。
7. 无风险利率：面板没有，缺省为 0，需要数据层提供。
8. CMRA 要求 252 日内无缺失，末日覆盖约 88%，与复牌收益口径一起定。
9. size 暴露在 A 股上分布很偏（约 27% 的股票绝对值大于 3），优化器里的风格界按实际分布设。
10. 换码的公司按数据层的别名表接续，否则上市天数清零。

### 这次没有覆盖的

实现者自做的 75 处变异没有逐项复跑；4000 × 5500 全量重建和追加一天的耗时没有复测；`solve_many` 和默认关闭的 `eigen_adjustment` 没有独立对拍；真实数据只跑了 300 日 × 800 只。

## 性能优化轮（2026-11-12，分支 `perf/risk`）

逐项记录在 `docs/perf_log.d/P-004` 至 `P-008` 和 `risk-suggestions.md`。对使用者有影响的变化：

- 缓存格式 4（`cache.FORMAT`，与参与 `parameter_hash` 的 `CACHE_SCHEMA` 分开）：后台线程写、deflate 1、证券表每代只存一次、快照带键。旧格式目录第一次使用时自动清空重算。另一个进程要读刚写的目录前先 `ModelCache.flush()`。
- 新增配置 `specific_incremental`（默认 True，特质风险分块累加，相对误差 ≤ 1e-12，超出风险的列自动改用从头算）和 `specific_reanchor_days`（默认 252）。设为 False 回到从头计算，结果与旧实现逐位相同（`parameter_hash` 不同）。
- `model_series` 新增 `threads`（默认取性能核数，或环境变量 `ASHARE_RISK_THREADS`）；任意线程数结果逐位相同。
- 缓存目录新增 `.prefix-index`（逐日输入摘要与前缀链摘要）；`ModelCache.load` 新增 `fields`，新增 `load_index` / `save_index`。历史输入改写仍会被发现。
