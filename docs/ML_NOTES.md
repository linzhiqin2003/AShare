# 机器学习（`ashare.ml`）说明

设计要点见 `DESIGN.md` 6.9。本文是用法、时间轴、实测数字、真实数据例子和验证记录。

## 安装

`ashare.ml` 需要可选依赖组 `ml`（`lightgbm`、`scikit-learn`）；框架其余部分不需要。

```bash
uv sync --all-packages --group dev --group ml --inexact
```

`make dev` 默认装上这个组，`make check` 会跑 `test/ml`。没装时 `import ashare.ml` 报：`ashare.ml 需要可选依赖组 ml（缺 lightgbm）。在仓库根目录运行 uv sync … --group ml …`；`import ashare` 和框架其余部分不加载 `ashare.ml`，照常可用，`test/ml` 整体跳过。

验证「不装这个组时其余部分照常」：scikit-learn 本来就在 `dev` 组里，所以只卸 LightGBM 就等于没装 `ml` 组。

```bash
uv pip uninstall lightgbm          # 模拟没装 ml 组
uv run --no-sync pytest            # 其余测试全过，test/ml 整体跳过
make dev                           # 装回
```

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
- 买入日不可买（停牌、涨停、一字板；口径同 `market_from_panel`，单测核对）：默认样本不进训练集，但照样给预测值。
- 卖出日不可卖（跌停开盘等），三种口径：
  - `sell_blocked="price"`（默认）：仍按当天成交价算，偏乐观；
  - `"drop"`：标签置缺失；
  - `"carry"`：顺延到之后第一个可卖且有价的日子卖出，最多顺延 `max_carry` 天（默认 10），仍卖不出则缺失。这时标签的最晚结束日是 s + lag + h + max_carry，`span` 也加上 max_carry，切分的最小间隔随之变大（`ret5` 加 10 天顺延时间隔是 16 天，给 6 天会被拒绝）；每个样本的实际卖出日记在 `LabelValues.end`，`check_lookahead` 按它再核对一遍。
  - 成交价缺失（停牌、退市）时：price、drop 标签缺失；carry 同样顺延。
  - 真实数据上差多少见下文「卖出日卖不出」。默认暂不改；以后要改默认，必须作为新的一组入账，并在文档里注明是看过结果之后改的。
- 训练目标：默认是原始收益。`relative="market"|"industry"` 先扣市场或行业均值，再可选截面排名 `rank=True`（居中到 ±0.5）或截面标准分 `zscore=True`（二选一）。评估（Rank IC、分组收益）总是对原始未来收益。默认目标以后要改成排名或标准分，同样作为新的一组入账并注明是事后改的；真实例子里已有的那张表不替换。
- 后加的选项（`zscore`、`max_carry`）取默认值时不进定义，已有试验的编号不变。
- `to_targets(res, top=50)` 每 h 行（默认等于标签持有期）在信号日选前 50 名，成交日由 `execution_targets` 错位 lag 行。

## 防前视

1. 特征只用信号日及以前：表达式求值器只有向过去看的算子；截面预处理每步只用当日截面（测试：改动后半段数据，前半段特征逐位不变）。
2. 训练样本的最晚数据日期是标签结束日，`check_lookahead(res)`（`res.check()`）逐折核对两道：用到的训练和验证样本里「最晚信号日 + span」< 预测期首日；这些样本的实际卖出日（`last_end_day`，顺延口径下逐样本不同）< 预测期首日。`walk_forward` 结束时自动调用一次。
3. LightGBM 跨折共享分箱时（`bins="shared"`），分箱边界只用**第一折训练集里可训练的样本**定。第一折训练集早于所有折的预测期，以后的数据不影响分箱；代价是后面的折沿用较早年份的边界（特征做了截面标准化，影响小，见「与最直白写法对比」）。
4. 端到端测试：改动第 D 日及以后的全部数据（价格、可买性、市值、特征），D 之前的预测逐位不变；岭回归、Lasso、LightGBM（共享 / 每折分箱）、LightGBM 排序都测了。D 取三处：第一折中间、第二折中间（不在任何折边界上）、折边界。只放在后面的折边界上测不出「分箱边界用了较晚一折的训练集」：最后一折的训练集截止在 D 之前，改动碰不到它（复核时发现的漏洞，变异 `shared_bins_use_last_fold`）。

### 间隔为什么必要（测试 `test_gap_removes_overlap_leak`）

任务书设想的是「把未来收益混进特征」。如果特征本身直接含有测试期的未来收益，那它在预测期同样含有，样本外 IC 会一直虚高，留不留间隔都一样——这类前视只能靠第 1 条（特征只看过去）挡住，间隔挡不住。间隔挡的是另一条通道：**未来收益经由重叠的标签进入训练集**。测试这样造：

- 收益是独立噪声；标签是 10 日收益，相邻信号日的标签重叠 9 天；
- 特征是一个随时间单调变化、各股票错开的「指纹」，和收益无关，只让记忆型模型（最近邻）找到同一只股票最近的那天；
- 每天重训、预测下一天（200 只股票、150 个预测日）。不留间隔时，训练集最后一天的标签覆盖了测试日的大部分未来收益，样本外 Rank IC = 0.891（虚高）；随着间隔加大线性下降，与理论值「重叠天数 / 10 = (9 − 跳过天数) / 10」逐档吻合，间隔 ≥ 9 天（标签收益区间不再重叠）后回到 0 附近，按 lag + h = 11 天的默认间隔为 0.008：

| 跳过天数 | 0 | 1 | 2 | 4 | 5 | 6 | 8 | 9 | 10 | 11（默认） |
|---|---|---|---|---|---|---|---|---|---|---|
| 样本外 Rank IC | 0.891 | 0.789 | 0.687 | 0.486 | 0.388 | 0.291 | 0.102 | 0.010 | 0.009 | 0.008 |

测试断言三档：不留间隔 > 0.6、跳过 5 天落在 0.4 ± 0.05、默认间隔 |IC| < 0.04——是剂量关系，不只是开关。换随机种子 1 时跳过 5 天为 0.392。

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

`Ledger().trials()` 返回到现在为止不同定义的组数（可按时间和标签过滤），超参搜索的每一组都会入账。规则：

- **失败的也算**：训练中途抛错时同样追加一行 `trial`，`status: "failed"`、`error` 只记异常类型（不记消息，消息里可能有本机路径或数据），计入组数；成功的是 `status: "ok"`。
- **「定义」不含框架代码版本、库版本和线程数**：框架修 bug、提速或升级库后同样的研究重跑不算新的一组。这三样都记在每一行里（`code`、`versions`、`threads`），运行缓存也按它们区分，任何一项变了就重算。`versions` 是 numpy、panelops 构建号、因子求值器代码哈希、LightGBM 和 scikit-learn 的版本；前三项同时进特征库的缓存键。
- **种子只认实际传给模型的那个**：`walk_forward(seed=7)` 与 `params={"seed": 7}` 是同一组；模型不接受 seed（如岭回归）时 walk_forward 的 seed 不影响结果，定义里记 None。
- **自定义模型按类的 qualname + 类源码哈希识别**，不看模块路径：同一个类从 `__main__` 还是别的脚本导入都是同一组，改了类的代码就是新的一组。取不到源码（交互式定义、`exec` 生成的类）时只用 qualname，并在 `model.source` 里写 `"unavailable"`。只哈希类本身：类里调用的外部函数改了，身份不变，这种改动请自己换类名或参数。内置模型只用类名。
- 这些规范化是复核后加的，`data/dev/ml/ledger.jsonl` 里之前的 8 行仍按旧规则算了编号（见「已知局限」）。

## 超参搜索与显著性折算

```python
r = ml.search(features, "ret5", data=panel, model="lgbm",
              grid={"num_leaves": [15, 63], "min_data_in_leaf": [500, 2000]},
              end="2022-12-31", retrain="halfyear", train_years=5, start="2016-01-01")
print(r.describe())
```

`end` 必填：搜索用到的最后一个预测日，后面那段留作最终检验，选定参数后只在它上面跑一次。不给 `end` 直接报错，避免默认用到全部日期、没有留出段。

每组都走同一套时间切分、都入账。结果表给出每组的 Rank IC、ICIR、t、p，以及 Bonferroni 折算：门槛 = α / N，N 取**账本里到现在为止的全部试验组数**（不只是这次搜索）；最好那组的 p × N 仍小于 α 才算显著。t 值把日 IC 按不重叠样本数 n/h 折算（简单保守）。更精细的方法（Deflated Sharpe、White Reality Check、Romano–Wolf、按相关性折算有效试验数）留到因子挖掘那条线。

## 解释

- `res.importance()`：各折特征重要性（LightGBM 为增益，线性模型为系数绝对值，每折归一化），`folds` 是折 × 特征，按折的预测期排开就是随时间的变化；
- `ml.permutation_importance(res)`：每折在验证集上，把一个特征在每天的截面内随机打乱，看逐日 Rank IC 均值下降多少。

## 性能

机器：15 核、24 GB 内存的 Mac，同时有十来个别的任务在跑（每条记录旁边是开始时的 1 分钟负载）。脚本都在 `bench/ml/`，原始数字在 `bench/ml/results/*.json`。内存记两种：驻留（RSS，会把读过的内存映射特征库文件页也算进去）和 phys_footprint（不含可回收的文件页，更接近真正占用）。测量时设 `MallocLargeCache=0`：macOS 的 libmalloc 默认把释放的大块内存留在缓存里继续算进 footprint，逐折分配大矩阵时会只涨不跌，关掉后 footprint 才是存活内存。训练线程 6（默认是核数的一半，7）。

### 全市场全历史规模（`bench/ml/scale.py`）

4825 个交易日 × 5923 只 × 147 个特征（`basic_features` 加换窗口的同族变体）。稠密 float64 要 31.3 GB；股票池内样本 14,521,680 个（占 51%）。

| 项目 | 耗时 | footprint 峰值（增量） | 说明 |
|---|---:|---:|---|
| 建特征库（冷，147 列） | 36.7 s、36.0 s（两次） | 2.6 GB（+2.2） | 落盘 7.95 GB（float32、只存池内样本），相当于稠密 float64 的 25% |
| 再次调用（命中缓存） | 0.16 s | +0.25 GB | 只读 meta 和索引，列按需映射 |
| 取一折训练矩阵（470 万行 × 147） | 7.0 s | +2.6 GB | 即矩阵本身；不整体拷贝特征库 |
| 岭回归单折：NumPy 正规方程（分块） | 1.8 s | +2.3 GB | 同一矩阵上 sklearn `Ridge` 6.1 s、+10.3 GB；两者系数相差 < 1e-9（单测） |
| LightGBM 单折，每折分箱，200 轮 | 54 s | +4.3 GB | |
| LightGBM 单折，共享分箱，200 轮 | 107 s | +4.7 GB | 比每折分箱多 53 s，多出的是一次性把 1452 万样本全部分箱（没有单独计时），之后每折只取子集 |
| 整个滚动：岭回归按季重训，2012 年起 60 折 | 53 s | 3.1 GB（+2.2） | 按日充分统计量：每个样本只读一次 |
| 整个滚动：LightGBM 按年重训，15 折，200 轮 | 576 s | 6.4 GB（+5.1） | 共享分箱 |

整个滚动都只跑了一次（单次十分钟级，机器同时很忙），不是三次中位数。

上表是返工前的代码测的。返工给建库加了「特征在股票池内全部缺失就报错」的检查（从最近的日期往前按 256 天一块找有限值，正常的特征只看最后一块），缓存键里加了依赖版本，都不在热路径上；没有重测这张表（再写一份 8 GB 的特征库，机器同时很忙）。文中的每个数字都核对过 `bench/ml/results/` 里的原始记录：原先写的「冷建 4 次中位数 33 s」和「改之前 526 s」在原始记录里找不到，已改成记录里的两次 36.7 s、36.0 s 并删掉后者。

### 与最直白写法对比（`bench/ml/baseline.py`、`baseline_ridge.py`）

同一件事：2019 年起的面板（1906 天 × 5923 只，852 万样本，42 个特征），按 63 个交易日重训、训练 750 天、2023 年起 14 折、不要验证集。直白写法是 pandas 长表 + sklearn `TimeSeriesSplit`（`gap` = 标签跨度、`max_train_size` = 750）+ 每折按日期过滤拼 X、每折新建模型。两边喂同样的特征值，只比训练和预测这一段。

| | 直白写法 | `ashare.ml` | 预测核对 |
|---|---:|---:|---|
| 截面预处理（6 个特征） | pandas 按日 groupby 5.7 s | `panelops` 0.32 s | 最大差 4.4e-15 |
| 岭回归，14 折（3 次中位数） | 12.3 s，footprint +4.7 GB | 4.1 s，+0.6 GB | 最大绝对差 2.3e-12，有预测的格子完全一致 |
| LightGBM 200 轮，14 折，每折分箱 | 210 s，+3.1 GB | 207 s，+1.5 GB | **逐位一致** |
| LightGBM，共享分箱 | — | 192 s，+1.6 GB | 与每折分箱最大差 0.15，Rank IC 0.0547 对 0.0549 |

结论照实写：LightGBM 的时间几乎全在训练本身，框架省下的是内存（一半）和拼表；岭回归这类训练很快的模型，取数和拼表占大头，按日充分统计量让它快 3 倍、footprint 增量约为八分之一（0.60 GB 对 4.73 GB）。共享分箱在单折里多花一次全量分箱的时间，在按季、按年重训、很多折的时候摊薄（上表 14 折时比每折分箱快 7%），代价是分箱边界只用第一折训练数据定。

性能上没做到、留作后续的：共享分箱的全量分箱走 LightGBM 的 `Sequence` 接口，逐批从 Python 推数据，单线程（上表单折里多出的 53 s 主要是它）；岭回归以外的模型每折仍要把训练矩阵读进内存（470 万行 × 147 列 2.6 GB）。


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

「相对股票池等权的年化超额」一列只是相对一个市值中性的参照，不能读成模型的超额：持仓偏小盘，换成市值最小 50 只等权作参照就没有超额，见下一节。年换手是单边的。

### 分段与基准（`bench/ml/real_segments.py`，结果 `results/real_segments.json`）

复核指出上表的读法偏乐观，补了下面这些。**没有重新训练、没有改参数、特征或标签**：脚本直接读当时存下的预测（运行目录 `runs/<run_id>/pred.npy`，run_id 取自 `results/real_<模型>.json`），先核对 Rank IC 均值与当时的记录逐位一致，再回测。所以账本没有新增行。分段是同一条净值曲线截取的，回测引擎、成本、调仓频率（每 5 日）都与上表相同。

两个新基准：
- **市值最小 50 只**：每 5 日在股票池里取总市值（`total_market_cap`）最小的 50 只等权，同一引擎同一成本；
- 股票池全部等权（上表的参照）。

年化收益 / 夏普，1000 万、不限成交量、不算冲击（与上表同设定）：

| | 全期 | 2012–2016 | 2017 起 | 2020 起 |
|---|---:|---:|---:|---:|
| 等权合成（朴素基准） | 10.5% / 0.53 | 34.3% / 1.19 | 0.1% / 0.12 | 2.3% / 0.21 |
| 岭回归 | 25.0% / 0.90 | 74.0% / 1.96 | 5.7% / 0.34 | 11.9% / 0.52 |
| LightGBM 回归 | 23.1% / 0.82 | 62.2% / 1.68 | 7.0% / 0.37 | 13.7% / 0.55 |
| LightGBM 排序 | 34.6% / 1.32 | 82.0% / 2.12 | 15.5% / 0.78 | 19.1% / 0.97 |
| 集成 | 32.7% / 1.09 | 83.6% / 2.08 | 12.6% / 0.55 | 17.8% / 0.69 |
| 股票池全部等权 | 9.6% / 0.51 | 23.5% / 0.90 | 3.1% / 0.25 | 9.1% / 0.52 |
| **市值最小 50 只等权** | **38.4% / 1.12** | 88.1% / 2.11 | **18.5% / 0.66** | **29.0% / 0.85** |

LightGBM 排序逐年收益（组合 / 市值最小 50 只）：2012 +58/+38，2013 +47/+72，2014 +75/+86，**2015 +267/+297**，2016 +33/+34，2017 −18/−29，2018 +4/−12，2019 +47/+48，2020 +27/−6，2021 +36/+32，2022 +13/−33，2023 +9/+116，2024 +54/+107，2025 +12/+103，2026（到 11 月）−11/−24。

**持仓市值分布**（每个调仓日的 top 50 在当日股票池总市值截面里的分位，0 = 最小）：

| | 落在最小 10% 的比例 | 落在最小 20% 的比例 | 分位均值 |
|---|---:|---:|---:|
| 等权合成（朴素基准） | 14% | 22% | 0.51 |
| 岭回归 | 32% | 46% | 0.31 |
| LightGBM 回归 | 31% | 47% | 0.31 |
| LightGBM 排序 | 55% | 64% | 0.32 |
| 集成 | 40% | 57% | 0.24 |
| 市值最小 50 只 | 100% | 100% | 0.01 |

**容量**：用主干 `ashare.analysis` 的冲击模型（`impact_coefficients`：k = 此前 20 日日收益波动率，冲击 = k × √(成交量占比)，系数 1，与 `an.capacity` 同口径）和引擎的单日成交量占比上限重跑：

| 年化 / 夏普 | 全期 | 2012–2016 | 2017 起 | 2020 起 |
|---|---:|---:|---:|---:|
| LightGBM 排序，1000 万，不限量、不算冲击（上表） | 34.6% / 1.32 | 82.0% / 2.12 | 15.5% / 0.78 | 19.1% / 0.97 |
| LightGBM 排序，1000 万，限 5%、**算冲击** | 15.2% / 0.70 | 54.3% / 1.58 | −0.7% / 0.08 | 3.3% / 0.26 |
| LightGBM 排序，1 亿，限 5%、不算冲击 | 26.3% / 1.26 | 73.7% / 2.05 | 7.4% / 0.60 | 9.3% / 0.82 |
| LightGBM 排序，1 亿，限 5%、**算冲击** | 10.2% / 0.54 | 37.1% / 1.22 | −1.3% / 0.03 | 1.5% / 0.17 |
| 岭回归，1000 万，限 5%、算冲击 | 1.0% / 0.18 | 38.5% / 1.22 | −13.9% / −0.37 | −7.1% / −0.09 |
| 岭回归，1 亿，限 5%、不算冲击 | 21.5% / 0.92 | 63.2% / 1.79 | 4.7% / 0.32 | 11.0% / 0.59 |
| 岭回归，1 亿，限 5%、算冲击 | −10.2% / −0.22 | 13.5% / 0.57 | −20.3% / −0.64 | −13.0% / −0.31 |
| 集成，1 亿，限 5%、算冲击 | −8.0% / −0.13 | 20.1% / 0.75 | −19.6% / −0.61 | −14.1% / −0.35 |
| 市值最小 50 只，1000 万，限 5%、算冲击 | 36.7% / 1.10 | 85.8% / 2.07 | 17.0% / 0.63 | 27.3% / 0.83 |
| 市值最小 50 只，1 亿，限 5%、不算冲击 | 36.4% / 1.21 | 92.9% / 2.20 | 14.4% / 0.63 | 22.9% / 0.87 |
| 市值最小 50 只，1 亿，限 5%、算冲击 | 32.0% / 1.06 | 79.7% / 1.99 | 12.9% / 0.56 | 22.4% / 0.80 |

其余模型各设定的数字都在 `real_segments.json` 里。差别主要来自换手：`an.capacity` 的解析表给出双边年换手岭回归、LightGBM、集成约 78 倍，LightGBM 排序 55 倍，市值最小 50 只 11 倍。算冲击的情景都同时限成交量：不限量时会有单笔成交超过当日全天成交量的订单（复核人查到市值最小 50 只在 1000 万、不限量时有 326 笔，最大是全天成交量的 3011 倍，单笔冲击最高是成交额的 72%），按平方根律算出的冲击没有意义，所以不报「不限量、算冲击」。两点保留：引擎按成交日全天成交量限量，而例子按开盘价成交，开盘集合竞价只占全天成交的一小部分，所以「限 5%」仍偏宽松；冲击系数 1 是常用假设，不是对 A 股实测校准的。

### 怎么读

- **全期的好数字主要来自 2012–2016 年的小盘行情**。五个模型 2012–2016 年年化 34–84%，2017 年起降到 0–16%；LightGBM 排序仅 2015 一年 +267%。
- **相对「市值最小 50 只等权」这个傻瓜组合没有超额**：同样的引擎和成本下它全期年化 38.4%、2017 年起 18.5%、2020 年起 29.0%，都高于五个模型。LightGBM 排序 2017 年起夏普 0.78 略高于它的 0.66，收益更低。原表「相对股票池等权的年化超额 12–23%」把小盘暴露算成了模型的超额，这个读法撤回。
- **持仓明显偏小盘**：LightGBM 排序平均 55% 的持仓在股票池市值最小的 10%，集成 40%，岭回归和 LightGBM 回归约 31–32%；朴素基准只有 14%，分位均值 0.51（中性）。
- **加上容量限制后，2017 年起的收益**：1000 万、限 5%、算冲击时 LightGBM 排序 −0.7%、岭回归 −13.9%，市值最小 50 只 17.0%。1 亿资金、单日成交量占比限 5%、不算冲击时，LightGBM 排序 7.4%、岭回归 4.7%、集成 4.6%；再按平方根律算冲击，LightGBM 排序 −1.3%，其余模型 −12% 到 −20%。市值最小 50 只同样设定下是 14.4% 和 12.9%（换手低得多）。按这个例子的设定（每 5 日全换、top 50 等权），这些模型在 1 亿规模上没有可用的收益。
- **朴素基准 IC 最高、收益最低，原因不只是「尾部」**。把每天的截面按预测分成上下两半分别算 Rank IC：

  | | 上半截面 IC | 下半截面 IC | top 50 的 5 日平均超额（相对股票池均值，万分之一） |
  |---|---:|---:|---:|
  | 等权合成（朴素基准） | 0.028 | 0.086 | +12.6 |
  | 岭回归 | 0.026 | 0.080 | +55.1 |
  | LightGBM 回归 | 0.026 | 0.079 | +49.3 |
  | LightGBM 排序 | 0.045 | 0.084 | +55.2 |
  | 集成 | 0.035 | 0.084 | +63.9 |
  | 市值最小 50 只 | — | — | +67.6 |

  预测力主要在下半截面（空头端），这一点成立；但朴素基准和岭回归的上半截面 IC 几乎一样，差别在最极端的那一小撮（top 50 只占池子 1–2%）和市值：机器学习模型的 top 50 偏小盘，而市值最小 50 只本身的 5 日平均超额就有 +67.6。所以「单调性更好」只解释了一部分，多出来的收益有相当一部分是小盘暴露。
- LightGBM 回归在原始收益上提前停止得很早（30 折里一半在 30 轮以内停，最早 1 轮）：验证集 Rank IC 在前几十轮就到顶。排序目标（lambdarank）不受收益幅度影响，Rank IC 0.101、头组换手 0.40，是五个里最好的。回归目标改成截面排名或标准分（`rank=True`、`zscore=True`）现在可以选，但这一轮没有拿它重跑真实例子：看过结果再换目标就是事后调参，要跑也得作为新的一组入账并注明。
- 小盘、反转、流动性这一类特征贡献大（LightGBM 排序的增益重要性前几位：log_market_cap, amount_20, amihud_20, turnover_std_20, bp），与上面的市值分布一致。
- 账本：这份例子在 `data/dev/ml/ledger.jsonl` 里记了 5 组（另有 3 组是代码定稿前的同定义预跑，见「已知局限」）。按 Bonferroni，5 组的门槛是 p < 0.01，Rank IC 的 t 值都远超；但这只说明预测与未来收益相关，不说明 top 50 组合有超额。特征本身是公开文献里反复用过的，真正的多重检验成本远不止这 5 组。

### 卖出日卖不出（`bench/ml/sell_blocked.py`，结果 `results/sell_blocked*.json`）

2012 年起、默认股票池、`ret5`，默认口径（卖出日不可卖仍按当天价）对比 `carry`（顺延到第一个可卖日）：

| | 顺延上限 10 天 | 顺延上限 60 天 |
|---|---:|---:|
| 卖出日不可卖的样本占比 | 0.147% | 0.149% |
| 这些样本标签的平均变化（顺延 − 默认） | −2.7 个百分点 | −3.2 个百分点 |
| 中位变化 | −2.1 个百分点 | −2.2 个百分点 |
| 顺延天数中位 / 90 分位 | 1 / 3 天 | 1 / 3 天 |
| 摊到全部样本的平均变化（每 5 日） | −0.39 bp | −0.47 bp |
| 顺延到上限仍卖不出、标签缺失 | 483 个 | 179 个 |
| 默认口径缺失（卖出日停牌）、顺延后有标签 | 37,390 个 | 49,586 个 |

默认口径偏乐观，但量级很小：只影响训练目标和 IC、十分组，不影响 `panelbt` 回测（引擎自己顺延）。另一个偏差是默认口径把卖出日停牌的样本整个丢掉（多是有事件的股票），顺延口径会把它们找回来，方向不定。默认值暂不改，交给负责人定。


## 变异检查

`bench/ml/mutations.py` 每次只改一处、跑 `test/ml`、恢复原文；先确认未改动时全过，并确认测试导入的是本工作区的源码。30 处全部被抓到（复核后新增 16 处，其中 `shared_bins_use_last_fold` 是复核人找到的、原测试漏掉的那一处，`store_key_missing_preprocess`、`ledger_only_positive_ic` 是复核人的另两处）：

| 变异 | 改了什么 | 抓到它的测试（第一个失败） |
|---|---|---|
| `label_shift_one_day` | 标签少移一天（卖出日提前一天） | `test/ml/test_labels.py::test_label_alignment_matches_backtest_timing[1-1]` |
| `label_buy_one_day_early` | 买入价早取一天（信号日当天成交） | `test/ml/test_labels.py::test_label_alignment_matches_backtest_timing[1-1]` |
| `buyable_misaligned` | 可买性取信号日而不是买入日 | `test/ml/test_labels.py::test_buyable_aligned_to_buy_day_and_blocked_excluded` |
| `blocked_not_excluded` | 买不进的样本也进训练集 | `test/ml/test_labels.py::test_buyable_aligned_to_buy_day_and_blocked_excluded` |
| `gap_minus_one` | 验证段与预测段的间隔少一天 | `test/ml/test_labels.py::test_carry_widens_minimum_gap` |
| `valid_overlaps_train` | 训练集与验证集重叠 | `test/ml/test_labels.py::test_carry_widens_minimum_gap` |
| `check_off_by_one` | 防前视检查放过等号 | `test/ml/test_walk.py::test_check_lookahead_rejects_tampered_fold` |
| `shared_bins_use_all_data` | 共享分箱边界用了全部样本（含未来） | `test/ml/test_walk.py::test_changing_later_data_never_changes_earlier_predictions[lgbm-params1-330]` |
| `shared_bins_use_last_fold` | 共享分箱边界用了最后一折的训练集（晚于前面各折的预测期） | `test/ml/test_walk.py::test_changing_later_data_never_changes_earlier_predictions[lgbm-params1-330]` |
| `carry_ignores_can_sell` | 顺延口径不看卖出日能否卖出 | `test/ml/test_labels.py::test_sell_blocked_carry_against_daily_loop[1-1-3]` |
| `carry_span_not_widened` | 顺延时最小间隔没有加上顺延天数 | `test/ml/test_labels.py::test_sell_blocked_carry_against_daily_loop[1-1-3]` |
| `end_day_ignores_carry` | 实际卖出日记成了不顺延的那天 | `test/ml/test_labels.py::test_sell_blocked_carry_against_daily_loop[1-1-3]` |
| `lookahead_end_off_by_one` | 按实际卖出日的防前视检查放过等号 | `test/ml/test_ledger_identity.py::test_carry_label_end_to_end_lookahead` |
| `zscore_not_applied` | 截面标准分目标没有生效 | `test/ml/test_labels.py::test_zscore_target_against_daily_loop` |
| `store_key_missing_preprocess` | 特征库缓存键漏掉预处理定义 | `test/ml/test_ledger_identity.py::test_store_key_includes_preprocess` |
| `ledger_only_positive_ic` | 账本只记 Rank IC 为正的试验 | `test/ml/test_ledger_identity.py::test_negative_ic_trial_is_ledgered` |
| `seed_not_normalized` | 试验定义记 walk_forward 的 seed 而不是模型实际用的 | `test/ml/test_ledger_identity.py::test_seed_written_either_way_is_one_trial` |
| `custom_identity_module_path` | 自定义模型的身份带模块路径 | `test/ml/test_ledger_identity.py::test_custom_model_identity_is_source_not_module` |
| `versions_not_in_run_id` | 运行缓存键不含库版本 | `test/ml/test_ledger_identity.py::test_library_versions_in_run_key_and_ledger` |
| `failure_not_ledgered` | 训练失败不入账 | `test/ml/test_ledger_identity.py::test_failed_training_is_ledgered_without_paths` |
| `no_constant_prediction_check` | 常数预测不报错 | `test/ml/test_ledger_identity.py::test_constant_prediction_raises` |
| `no_empty_fold_check` | 训练集为空不报错 | `test/ml/test_ledger_identity.py::test_fold_without_training_samples_raises` |
| `all_missing_feature_allowed` | 特征全缺失不报错 | `test/ml/test_ledger_identity.py::test_all_missing_feature_raises` |
| `search_end_optional` | 超参搜索不要求留出段 | `test/ml/test_search_explain.py::test_search_requires_explicit_end` |
| `gram_window_one_day_late` | 岭回归的按日统计量多加了训练期之后的一天 | `test/ml/test_walk.py::test_ridge_matches_naive_sklearn` |
| `ridge_no_intercept` | 岭回归截距算错 | `test/ml/test_models.py::test_ridge_numpy_matches_sklearn` |
| `early_stop_lower_better` | 提前停止把 Rank IC 当成越小越好 | `test/ml/test_models.py::test_lgbm_early_stopping_uses_rank_ic` |
| `ledger_counts_runs_not_definitions` | 账本按运行（含线程数）而不是按定义数试验组数 | `test/ml/test_ledger_identity.py::test_library_versions_in_run_key_and_ledger` |
| `winsor_upper_ignored` | 去极值上界没生效 | `test/ml/test_preprocess_store.py::test_preprocess_matches_daily_loop[pre0]` |
| `ic_misaligned` | 逐日 Rank IC 的预测与标签错位 | `test/ml/test_search_explain.py::test_permutation_importance_finds_real_feature` |

说明：`gap_minus_one` 和 `valid_overlaps_train` 由 `schedule()` 末尾的 `check_schedule` 当场拒绝，所以任何用到切分的测试都会先失败（表里列的是按文件顺序第一个失败的）；`test_split.py` 的性质测试另用逐日的直白检查核对间隔。最初写的「账本把缓存命中也算成新试验」是等价变异（按定义哈希去重，命中缓存的记录哈希相同），换成了「按运行而不是按定义计数」。

## 已知局限

- 默认口径下卖出日跌停（不可卖但有价）时标签仍按当日成交价算，偏乐观；卖出日停牌（无价）时标签缺失、样本被丢掉。回测按 `panelbt` 顺延，两者口径不同。量级见「卖出日卖不出」；`sell_blocked="carry"` 与回测一致，但默认没改。
- 顺延口径下，顺延到 `max_carry` 天仍卖不出（含退市）的样本标签缺失、被丢掉，这是幸存者偏差，方向偏乐观。
- 标签用复权因子计算收益（等效分红再投资、不扣红利税），与 `panelbt` 的现金分红记账有小差别。
- 共享分箱用第一折训练集定边界，若特征分布随时间漂移较大（没做截面标准化时），后面的折分箱会变粗；默认预处理做了截面标准化，影响小。需要时用 `bins="per_fold"`。
- `incremental=True` 的特征库只核对已存的最后 `verify_days` 天；更早的历史数据被修订不会被发现。默认关闭（面板版本进缓存键，数据一变就重建）。
- t 值和 Bonferroni 都是简单保守的做法，见上。
- 账本在代码定稿前把框架代码版本算进了试验组，所以 `data/dev/ml/ledger.jsonl` 里真实数据例子的前 3 行（同样的定义、预跑）是另外 3 个试验组 id。复核后又改了编号规则（种子规范化、自定义模型按源码哈希、新增 `status`、`versions` 字段）：账本里已有的 8 行保留原样、不改写，它们和以后按新规则算的编号不可比，同样的定义以后重跑会多算一组（偏保守）。返工没有重跑任何模型，账本没有新增行。
- 特征库和运行缓存的键含 `ashare.ml` 的源码哈希，改一行代码就整库重建（147 列约 30 s）；旧库不自动清理，`data/dev/ml/stores/` 目前约 21 GB，可整目录删掉。
- 自定义模型的身份只哈希类本身的源码，类里调用的外部函数改了不会变成新的一组。
- `max_participation` 按成交日全天成交量限制，而例子按开盘价成交，开盘集合竞价的成交量只是全天的一小部分，所以「限 5%」仍偏宽松；冲击成本的系数（σ × √占比，系数 1）也只是常用假设。
