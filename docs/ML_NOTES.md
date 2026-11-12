# 机器学习（`ashare.ml`）说明

设计要点见 `DESIGN.md` 6.9。本文是用法、时间轴、实测数字、真实数据例子和验证记录。

## 安装

`ashare.ml` 需要可选依赖组 `ml`（`lightgbm`、`scikit-learn`）；框架其余部分不需要。

```bash
uv sync --all-packages --group dev --group ml --inexact
```

没装时 `import ashare.ml` 报：`ashare.ml 需要可选依赖组 ml（缺 lightgbm）。在仓库根目录运行 uv sync … --group ml …`。`make dev` 目前只装 `dev` 组，要跑 `test/ml` 需要先执行上面这条（`test/ml` 在没装时整体跳过）。

神经网络这一轮不做。以后的 PyTorch 模型走单独的可选组 `nn`（名字已预留，这一轮没有写进 `pyproject.toml`），默认不装。`mlp`、`lstm`、`gru`、`transformer` 这几个模型名预留给它：没装时用到会提示「需要可选依赖组 `nn`，运行 `uv sync … --group nn`」；装了以后，提供模型的包只要实现协议并登记名字（`ml.register_model("mlp", "某包:MLP", group="nn")`，或在入口点组 `ashare.ml.models` 里写 `mlp = "某包:MLP"`），就能 `model="mlp"`，不用改 `ashare.ml`。

## 十行从面板到回测净值

```python
import ashare.data as ad, ashare.ml as ml, panelbt
from ashare.backtest import market_from_panel

data = ad.open()
features = ml.basic_features()                       # 或 {名字: Expr / T×N 数组}
bm = market_from_panel(data, fields=ml.required_fields(features, "ret5"))
res = ml.walk_forward(features, "ret5", data=bm.panel, model="lgbm",
                      retrain="quarter", train_years=5, start="2015-01-01")
print(res.summary())                                 # Rank IC、ICIR、十分组、换手
out = panelbt.run(bm.market, ml.to_targets(res, top=50), bm.costs(),
                  bm.config(initial_cash=1e7))       # out.nav 是净值
```

`res.pred` 是 T×N 预测面板（没有预测的格子为 NaN），可以直接交给 `panelops.rank_ic`、`group_returns`、因子合成或 `ashare.portfolio` 的组合优化。

## 时间轴

行号就是交易日，与 `ashare.backtest.timing` 相同。以 `ret5`（lag = 1，h = 5）为例：

```
行号      s        s+1                     s+6
          │ 收盘算特征 │ 开盘价买入（can_buy_open） │ 开盘价卖出
训练样本：信号日 s，标签 = P[s+6]/P[s+1] − 1，标签结束日 = s + 6

切分（gap 默认 = lag + h = 6）：
  训练 [a, b) │ 跳过 6 天 │ 验证 [b+6, c) │ 跳过 6 天 │ 预测 [c+6, d)
  训练最后一天 b−1 的标签结束于 b+5 < b+6 = 验证第一天，验证与预测之间同理。
```

- P = 成交价 × 复权因子。复权因子把分红再投资和送转都算进去；`panelbt` 按面板的公司行为记账（扣红利税、有零股），两者略有差别，标签只用来训练和评估，回测仍以 `panelbt` 为准。
- 买入日不可买（停牌、涨停、一字板；口径同 `market_from_panel`，单测核对）：默认样本不进训练集，但照样给预测值。卖出日不可卖默认仍按当天成交价算（偏乐观：真实回测会顺延到能卖的那天），`sell_blocked="drop"` 改为置缺失。成交价缺失（停牌、退市）时标签缺失。
- 评估（Rank IC、分组收益）总是对原始未来收益；`relative`、`rank` 只改训练目标。
- `to_targets(res, top=50)` 每 h 行（默认等于标签持有期）在信号日选前 50 名，成交日由 `execution_targets` 错位 lag 行。

## 防前视

1. 特征只用信号日及以前：表达式求值器只有向过去看的算子；截面预处理每步只用当日截面（测试：改动后半段数据，前半段特征逐位不变）。
2. 训练样本的最晚数据日期是标签结束日，`check_lookahead(res)`（`res.check()`）逐折核对「用到的训练和验证样本里最晚的标签结束日 < 预测期首日」，`walk_forward` 结束时自动调用一次。
3. LightGBM 跨折共享分箱时，分箱边界只用第一折训练集定（`bins="shared"`），以后的数据不影响分箱。
4. 端到端测试：改动第 D 日及以后的全部数据（价格、可买性、市值、特征），D 之前的预测逐位不变；岭回归、Lasso、LightGBM（共享 / 每折分箱）、LightGBM 排序都测了。

### 间隔为什么必要（测试 `test_gap_removes_overlap_leak`）

任务书设想的是「把未来收益混进特征」。如果特征本身直接含有测试期的未来收益，那它在预测期同样含有，样本外 IC 会一直虚高，留不留间隔都一样——这类前视只能靠第 1 条（特征只看过去）挡住，间隔挡不住。间隔挡的是另一条通道：**未来收益经由重叠的标签进入训练集**。测试这样造：

- 收益是独立噪声；标签是 10 日收益，相邻信号日的标签重叠 9 天；
- 特征是一个随时间单调变化、各股票错开的「指纹」，和收益无关，只让记忆型模型（最近邻）找到同一只股票最近的那天；
- 每天重训、预测下一天（200 只股票、150 个预测日）。不留间隔时，训练集最后一天的标签覆盖了测试日的大部分未来收益，样本外 Rank IC = 0.891（虚高）；随着间隔加大线性下降，间隔 ≥ 9 天（标签收益区间不再重叠）后回到 0 附近，按 lag + h = 11 天的默认间隔为 0.008：

| 跳过天数 | 0 | 1 | 2 | 4 | 6 | 8 | 9 | 10 | 11（默认） |
|---|---|---|---|---|---|---|---|---|---|
| 样本外 Rank IC | 0.891 | 0.789 | 0.687 | 0.486 | 0.291 | 0.102 | 0.010 | 0.009 | 0.008 |

收益区间在跳过 9 天时已不重叠，默认还要多跳 2 天，是因为规则按「信息何时可知」定：信号日 s 的标签要到第 s + lag + h 行开盘后才知道，必须早于预测日。

正式入口 `schedule()` 不允许把间隔设得小于 lag + h；这个实验直接调用底层的 `fit_folds` 和手写的日程。

## 模型

| 名字 | 说明 |
|---|---|
| `ridge` | 岭回归，带截距；按 100 万行一块累积正规方程的充分统计量（float64）后求解。`solver="sklearn"` 是参考实现，单测 1e-9 内一致 |
| `lasso`、`elasticnet` | scikit-learn 坐标下降 |
| `lgbm` | LightGBM 回归，提前停止看验证集逐日 Rank IC 均值 |
| `lgbm_rank` | LightGBM lambdarank，按日分组，训练目标按日分成 20 个等级，等级增益线性 |
| `ensemble` | 多模型 × 多种子，`combine="mean"` 或 `"rank"`（先按日截面排名再平均） |

默认的 LightGBM 参数（写代码时定下，真实数据例子沿用，不为结果调）：`n_estimators=500, learning_rate=0.05, num_leaves=63, min_data_in_leaf=1000, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10, max_bin=63, early_stopping=50, deterministic=True, force_row_wise=True`。

自定义模型的例子在 `ashare.ml.models` 的模块文档里（二十行以内，`test_docstring_custom_model_runs_through_walk_forward` 原样执行它）：实现 `fit(train, valid)`、`predict(batch)`、`save(path)`、`load(path)`，`ml.register_model("corr", CorrWeights)` 后 `model="corr"`。`train.X` 是 float32 的样本 × 特征矩阵（用到才读盘），`train.y` 是训练目标，`train.day`、`train.col` 是每个样本的信号日和证券列。

### 可复现

- 同样的定义跑两次逐位一致；`fold_workers=2` 与 1 逐位一致（各折独立，结果按折序拼回）。
- LightGBM 逐位可复现的条件：固定 `seed`、固定线程数、`deterministic=True`、`force_row_wise=True`（默认都设好了）。会破坏可复现性的：改线程数（直方图的并行累加顺序不同）、`force_col_wise` 与 `force_row_wise` 自动选择、换 LightGBM 版本、`bins="shared"` 与 `"per_fold"` 之间切换（分箱边界不同）。因此运行缓存的键里含线程数；但账本的「试验组」不含线程数，只换线程数重跑不算新的一组。
- `bins="per_fold"` 时与最直白的写法（每折新建 `lgb.Dataset`）逐位一致，单测和基线脚本都核对了。

## 留痕与缓存

默认目录：环境变量 `ASHARE_ML_DIR`，否则数据根目录下的 `ml/`（不入库）。

- `stores/<键>/`：特征库（每个特征一个 `.f32` 文件、`index.npz`、`meta.json`）；
- `runs/<run_id>/`：样本外预测（float64，只存有预测的样本）、各折信息、各折模型；
- `ledger.jsonl`：每次训练一行 `trial`，记数据版本、轴与股票池哈希、特征名与表达式哈希、标签、预处理、切分日程表、模型参数、种子、代码版本（git 提交、工作区是否有改动、`ashare.ml` 源码哈希）、各折与汇总指标（Rank IC、ICIR、t、十分组、多空、头组换手）、耗时；命中缓存记一行 `cache_hit`。

`Ledger().trials()` 返回到现在为止不同定义的组数（可按时间和标签过滤），超参搜索的每一组都会入账。「定义」不含框架代码版本和线程数：框架修 bug 或提速后同样的研究重跑不算新的一组（代码版本仍记在每一行里，运行缓存也按代码版本区分，代码一改就重算）。

## 超参搜索与显著性折算

```python
r = ml.search(features, "ret5", data=panel, model="lgbm",
              grid={"num_leaves": [15, 63], "min_data_in_leaf": [500, 2000]},
              end="2022-12-31", retrain="halfyear", train_years=5, start="2016-01-01")
print(r.describe())
```

每组都走同一套时间切分、都入账。结果表给出每组的 Rank IC、ICIR、t、p，以及 Bonferroni 折算：门槛 = α / N，N 取**账本里到现在为止的全部试验组数**（不只是这次搜索）；最好那组的 p × N 仍小于 α 才算显著。t 值把日 IC 按不重叠样本数 n/h 折算（简单保守）。更精细的方法（Deflated Sharpe、White Reality Check、Romano–Wolf、按相关性折算有效试验数）留到因子挖掘那条线。搜索请把 `end` 设在最终检验期之前。

## 解释

- `res.importance()`：各折特征重要性（LightGBM 为增益，线性模型为系数绝对值，每折归一化），`folds` 是折 × 特征，按折的预测期排开就是随时间的变化；
- `ml.permutation_importance(res)`：每折在验证集上，把一个特征在每天的截面内随机打乱，看逐日 Rank IC 均值下降多少。

## 性能

机器：15 核、24 GB 内存的 Mac，同时有十来个别的任务在跑（每条记录旁边是开始时的 1 分钟负载）。脚本都在 `bench/ml/`，原始数字在 `bench/ml/results/*.json`。内存记两种：驻留（RSS，会把读过的内存映射特征库文件页也算进去）和 phys_footprint（不含可回收的文件页，更接近真正占用）。测量时设 `MallocLargeCache=0`：macOS 的 libmalloc 默认把释放的大块内存留在缓存里继续算进 footprint，逐折分配大矩阵时会只涨不跌，关掉后 footprint 才是存活内存。训练线程 6（默认是核数的一半，7）。

### 全市场全历史规模（`bench/ml/scale.py`）

4825 个交易日 × 5923 只 × 147 个特征（`basic_features` 加换窗口的同族变体）。稠密 float64 要 31.3 GB；股票池内样本 14,521,680 个（占 51%）。

| 项目 | 耗时 | footprint 峰值（增量） | 说明 |
|---|---:|---:|---|
| 建特征库（冷，147 列） | 28–51 s，4 次中位数 33 s | 2.6 GB（+2.2） | 落盘 7.95 GB（float32、只存池内样本），相当于稠密 float64 的 25% |
| 再次调用（命中缓存） | 0.16 s | +0.25 GB | 只读 meta 和索引，列按需映射 |
| 取一折训练矩阵（470 万行 × 147） | 7.0 s | +2.6 GB | 即矩阵本身；不整体拷贝特征库 |
| 岭回归单折：NumPy 正规方程（分块） | 1.8 s | +2.3 GB | 同一矩阵上 sklearn `Ridge` 6.1 s、+10.3 GB；两者系数相差 < 1e-9（单测） |
| LightGBM 单折，每折分箱，200 轮 | 54 s | +4.3 GB | |
| LightGBM 单折，共享分箱，200 轮 | 107 s | +4.7 GB | 含一次性把 1452 万样本全部分箱（约 50 s），之后每折只取子集 |
| 整个滚动：岭回归按季重训，2012 年起 60 折 | 53 s | 3.1 GB（+2.2） | 按日充分统计量：每个样本只读一次（改之前同一件事 526 s） |
| 整个滚动：LightGBM 按年重训，15 折，200 轮 | 576 s | 6.4 GB（+5.1） | 共享分箱 |

整个滚动都只跑了一次（单次十分钟级，机器同时很忙），不是三次中位数。

### 与最直白写法对比（`bench/ml/baseline.py`、`baseline_ridge.py`）

同一件事：2019 年起的面板（1906 天 × 5923 只，852 万样本，42 个特征），按 63 个交易日重训、训练 750 天、2023 年起 14 折、不要验证集。直白写法是 pandas 长表 + sklearn `TimeSeriesSplit`（`gap` = 标签跨度、`max_train_size` = 750）+ 每折按日期过滤拼 X、每折新建模型。两边喂同样的特征值，只比训练和预测这一段。

| | 直白写法 | `ashare.ml` | 预测核对 |
|---|---:|---:|---|
| 截面预处理（6 个特征） | pandas 按日 groupby 5.7 s | `panelops` 0.32 s | 最大差 4.4e-15 |
| 岭回归，14 折（3 次中位数） | 12.3 s，footprint +4.7 GB | 4.1 s，+0.6 GB | 最大绝对差 2.3e-12，有预测的格子完全一致 |
| LightGBM 200 轮，14 折，每折分箱 | 210 s，+3.1 GB | 207 s，+1.5 GB | **逐位一致** |
| LightGBM，共享分箱 | — | 192 s，+1.6 GB | 与每折分箱最大差 0.15，Rank IC 0.0547 对 0.0549 |

结论照实写：LightGBM 的时间几乎全在训练本身，框架省下的是内存（一半）和拼表；岭回归这类训练很快的模型，取数和拼表占大头，按日充分统计量让它快 3 倍、内存少 7 倍。共享分箱在单折里多花一次全量分箱的时间，在按季、按年重训、很多折的时候摊薄（上表 14 折时比每折分箱快 7%），代价是分箱边界只用第一折训练数据定。

性能上没做到、留作后续的：共享分箱的全量分箱走 LightGBM 的 `Sequence` 接口，逐批从 Python 推数据，单线程，约 50 s；岭回归以外的模型每折仍要把训练矩阵读进内存（470 万行 × 147 列 2.6 GB）。


## 真实数据例子

代码 `bench/ml/real_example.py`，结果 `bench/ml/results/real_*.json`。参数在跑之前定下，不为结果调：

- 特征：`ml.basic_features()`，即 `default_registry()` 的 15 个示例因子加自己写的 27 个量价特征（动量 60/120/250 日和跳过 5 日的 60 日动量、1 日反转、60 日波动、波动比、下行波动、偏度、峰度、5 日换手与换手比、换手波动、价量相关、收益与对数成交量相关、量比、额比、20 日振幅、20 日区间位置、均线偏离 20/60、60 日回撤、日内与隔夜收益、收盘相对均价、价格时序排名、对数股价、自由流通比例），共 42 个；
- 股票池：`default_universe`（已上市、非 ST、上市满 120 个交易日、当天有行情）；预处理默认（1%/99% 截断、截面标准化、缺失补 0）；
- 标签：`ret5`（t+1 开盘买、t+6 开盘卖，复权）；买入日不可买的样本不进训练集；
- 切分：2012-01-01 起预测，每半年重训一次（30 折），滚动 5 年训练窗口，末尾 120 天做验证集，间隔按标签自动为 6 天；
- 模型：岭回归 alpha = 1；LightGBM 用上文的默认参数；排序版同参数加 20 个等级；集成是岭回归加三个种子的 LightGBM、按日截面排名平均；朴素基准是「各特征等权合成，方向取训练期里该特征的 Rank IC 符号」（用协议写的自定义模型）；
- 组合：每 5 个交易日在信号日选预测最高的 50 只等权，下一交易日开盘成交（`to_targets` → `panelbt.run`），初始 1000 万，佣金万 2.5（最低 5 元）、滑点万 5、历史印花税与过户费（`market_from_panel(...).costs()` 的默认值）。参照是同样每 5 天调仓、等权持有整个股票池。

预测期 2012-01-04 至 2026-11 共 3601 个交易日；训练线程 7，各模型依次跑（机器同时很忙）。

### 预测能力（对原始 5 日收益）

| 模型 | Rank IC 均值 | ICIR（日） | t（按 n/5 折算） | 十分组多空（万分之一，5 日） | 头组换手 | 滚动训练耗时 |
|---|---:|---:|---:|---:|---:|---:|
| 等权合成（朴素基准） | 0.0907 | 0.586 | 15.7 | +91 | 0.43 | 141 s |
| 岭回归 | 0.0845 | 0.674 | 18.1 | +128 | 0.66 | 8 s |
| LightGBM 回归 | 0.0784 | 0.781 | 21.0 | +127 | 0.66 | 312 s |
| LightGBM 排序 | 0.1014 | 0.824 | 22.1 | +131 | 0.40 | 940 s |
| 集成（岭回归 + 3 种子 LightGBM，按日排名平均） | 0.0873 | 0.760 | 20.4 | +142 | 0.65 | 926 s |

十分组 5 日收益（万分之一，D1 最低 → D10 最高）：

| 模型 | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 等权合成（朴素基准） | -42 | +10 | +25 | +35 | +41 | +46 | +47 | +49 | +50 | +48 |
| 岭回归 | -51 | +0 | +14 | +24 | +34 | +42 | +49 | +56 | +65 | +77 |
| LightGBM 回归 | -53 | +6 | +21 | +29 | +35 | +40 | +45 | +52 | +60 | +74 |
| LightGBM 排序 | -55 | +1 | +17 | +27 | +32 | +40 | +48 | +56 | +66 | +77 |
| 集成（岭回归 + 3 种子 LightGBM，按日排名平均） | -60 | +5 | +19 | +27 | +34 | +40 | +46 | +53 | +63 | +82 |

分年 Rank IC：

| 模型 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 等权合成（朴素基准） | 0.092 | 0.071 | 0.088 | 0.126 | 0.119 | 0.069 | 0.082 | 0.096 | 0.053 | 0.087 | 0.097 | 0.087 | 0.105 | 0.095 | 0.094 |
| 岭回归 | 0.114 | 0.114 | 0.107 | 0.143 | 0.125 | 0.040 | 0.088 | 0.060 | 0.059 | 0.063 | 0.076 | 0.074 | 0.077 | 0.071 | 0.053 |
| LightGBM 回归 | 0.088 | 0.093 | 0.090 | 0.148 | 0.116 | 0.045 | 0.085 | 0.062 | 0.063 | 0.065 | 0.082 | 0.051 | 0.083 | 0.055 | 0.042 |
| LightGBM 排序 | 0.123 | 0.115 | 0.111 | 0.144 | 0.155 | 0.067 | 0.110 | 0.086 | 0.080 | 0.079 | 0.095 | 0.089 | 0.104 | 0.080 | 0.080 |
| 集成（岭回归 + 3 种子 LightGBM，按日排名平均） | 0.108 | 0.108 | 0.103 | 0.154 | 0.130 | 0.045 | 0.093 | 0.066 | 0.068 | 0.071 | 0.090 | 0.064 | 0.088 | 0.064 | 0.051 |

### top 50 等权组合（扣成本，2012-01 至 2026-11）

| 模型 | 年化收益 | 年化波动 | 夏普 | 最大回撤 | 年换手（倍） | 相对股票池等权的年化超额 | 超额最大回撤 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 等权合成（朴素基准） | 10.5% | 24.5% | 0.53 | -53.6% | 28 | +0.9% | -52.0% |
| 岭回归 | 25.0% | 29.7% | 0.90 | -49.6% | 38 | +14.0% | -52.5% |
| LightGBM 回归 | 23.1% | 31.3% | 0.82 | -48.7% | 38 | +12.3% | -49.1% |
| LightGBM 排序 | 34.6% | 24.9% | 1.32 | -45.9% | 27 | +22.8% | -29.0% |
| 集成（岭回归 + 3 种子 LightGBM，按日排名平均） | 32.7% | 30.1% | 1.09 | -44.4% | 38 | +21.1% | -38.9% |
| 参照：股票池全部等权 | 9.6% | 23.2% | 0.51 | -61.9% | 1 | — | — |

### 怎么读

- 朴素的等权合成 Rank IC 均值最高（0.091），但它的预测力集中在空头端：十分组里只有最低一组明显差，D4–D10 几乎一样，选前 50 只做多只比股票池等权多 0.9% 年化。机器学习模型的 IC 均值不比它高（岭回归 0.085、LightGBM 回归 0.078），但单调性好得多，头部组更强，ICIR 更高，top 50 多头扣成本后年化超额 12–23%。
- LightGBM 回归在原始收益上提前停止得很早（30 折里一半在 30 轮以内停，最早 1 轮）：验证集 Rank IC 在前几十轮就到顶，之后学到的是训练期的厚尾收益。排序目标（lambdarank）不受收益幅度影响，是五个里最好的：Rank IC 0.101、年化 34.6%、夏普 1.32、头组换手也最低（0.40）。这是一个结果，不是调出来的；回归版改训练目标为截面排名（`label("ret5", rank=True)`）大概率也会改善，但这一轮没试，免得事后挑参数。
- 收益主要来自 2012–2016 年；2017 年以后各模型 Rank IC 都降到 0.05–0.09。小盘、反转、流动性这一类特征贡献大（LightGBM 排序的增益重要性前几位：log_market_cap, amount_20, amihud_20, turnover_std_20, bp），top 50 组合偏小盘，容量和冲击成本没有在这里评估（成交量占比限制用的是引擎默认值）。
- 账本：这份例子在 `data/dev/ml/ledger.jsonl` 里记了 5 组（另有 3 组是代码定稿前的同定义预跑，代码当时把版本算进了试验组，见下面「已知局限」）。按 Bonferroni，5 组的门槛是 p < 0.01，表里的 t 值都远超；但这些特征本身是公开文献里反复用过的，真正的多重检验成本远不止这 5 组。


## 变异检查

`bench/ml/mutations.py` 每次只改一处、跑 `test/ml`、恢复原文；先确认未改动时全过，并确认测试导入的是本工作区的源码。14 处全部被抓到：

| 变异 | 改了什么 | 抓到它的测试（第一个失败） |
|---|---|---|
| `label_shift_one_day` | 标签少移一天（卖出日提前一天） | `test/ml/test_labels.py::test_label_alignment_matches_backtest_timing[1-1]` |
| `label_buy_one_day_early` | 买入价早取一天（信号日当天成交） | `test/ml/test_labels.py::test_label_alignment_matches_backtest_timing[1-1]` |
| `buyable_misaligned` | 可买性取信号日而不是买入日 | `test/ml/test_labels.py::test_buyable_aligned_to_buy_day_and_blocked_excluded` |
| `blocked_not_excluded` | 买不进的样本也进训练集 | `test/ml/test_labels.py::test_buyable_aligned_to_buy_day_and_blocked_excluded` |
| `gap_minus_one` | 验证段与预测段的间隔少一天 | `test/ml/test_search_explain.py::test_grid_search_ledgers_every_config` |
| `valid_overlaps_train` | 训练集与验证集重叠 | `test/ml/test_search_explain.py::test_grid_search_ledgers_every_config` |
| `check_off_by_one` | 防前视检查放过等号 | `test/ml/test_walk.py::test_check_lookahead_rejects_tampered_fold` |
| `shared_bins_use_all_data` | 共享分箱边界用了全部样本（含未来） | `test/ml/test_walk.py::test_changing_later_data_never_changes_earlier_predictions[lgbm-params1]` |
| `gram_window_one_day_late` | 岭回归的按日统计量多加了训练期之后的一天 | `test/ml/test_walk.py::test_ridge_matches_naive_sklearn` |
| `ridge_no_intercept` | 岭回归截距算错 | `test/ml/test_models.py::test_ridge_numpy_matches_sklearn` |
| `early_stop_lower_better` | 提前停止把 Rank IC 当成越小越好 | `test/ml/test_models.py::test_lgbm_early_stopping_uses_rank_ic` |
| `ledger_counts_runs_not_definitions` | 账本按运行（含线程数）而不是按定义数试验组数 | `test/ml/test_walk.py::test_cache_and_ledger` |
| `winsor_upper_ignored` | 去极值上界没生效 | `test/ml/test_preprocess_store.py::test_preprocess_matches_daily_loop[pre0]` |
| `ic_misaligned` | 逐日 Rank IC 的预测与标签错位 | `test/ml/test_search_explain.py::test_permutation_importance_finds_real_feature` |

说明：`gap_minus_one` 和 `valid_overlaps_train` 由 `schedule()` 末尾的 `check_schedule` 当场拒绝，所以任何用到切分的测试都会先失败；`test_split.py` 的性质测试另用逐日的直白检查核对间隔。最初写的「账本把缓存命中也算成新试验」是等价变异（按定义哈希去重，命中缓存的记录哈希相同），换成了「按运行而不是按定义计数」。

## 已知局限

- 卖出日跌停或停牌时标签仍按当日成交价算，偏乐观；回测按 `panelbt` 顺延，两者口径不同。
- 标签用复权因子计算收益（等效分红再投资、不扣红利税），与 `panelbt` 的现金分红记账有小差别。
- 共享分箱用第一折训练集定边界，若特征分布随时间漂移较大（没做截面标准化时），后面的折分箱会变粗；默认预处理做了截面标准化，影响小。需要时用 `bins="per_fold"`。
- `incremental=True` 的特征库只核对已存的最后 `verify_days` 天；更早的历史数据被修订不会被发现。默认关闭（面板版本进缓存键，数据一变就重建）。
- t 值和 Bonferroni 都是简单保守的做法，见上。
- 账本在代码定稿前把框架代码版本算进了试验组，所以 `data/dev/ml/ledger.jsonl` 里真实数据例子的前 3 行（同样的定义、预跑）是另外 3 个试验组 id；定稿后的代码不再这样算。
- 特征库和运行缓存的键含 `ashare.ml` 的源码哈希，改一行代码就整库重建（147 列约 30 s）；旧库不自动清理，`data/dev/ml/stores/` 目前约 21 GB，可整目录删掉。
- `make dev` 只装 `dev` 组，没装 `ml` 组，`make check` 里的 `test/ml` 会被跳过；要不要把 `--group ml` 加进 `make dev` 由负责人定。
