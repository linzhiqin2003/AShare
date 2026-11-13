# 算子清单

本文按代码清点 `panelops` 对外提供的全部算子，并给出与 WorldQuant BRAIN 的对照。
文档对照 `main` @ `b02e98c`（2026-11-13）清点；加算子时同步更新本文。

统计口径：`packages/panelops/python/panelops/api.py` 的 `TS_OPS`、`CS_OPS`、
`ELEMENTWISE_ARITY`，以及 `src/ashare/factor/expression.py` 的 `ELEMENTWISE`。
两处逐元素表必须一致（当前都是 27 项）。

## 1. 数量

| 类别 | 数量 | 入口 |
|---|---:|---|
| 时序算子 `ts_*` | 32 | 独立函数 |
| 截面算子 `cs_*` | 8 | 独立函数 |
| 逐元素算子 | 27 | `elementwise(op, *args, mask=None, lower=-1.0, upper=1.0)` |
| **数值算子合计** | **67** | 默认 Rust 内核 |
| 评估核 | 7 | `ic` / `rank_ic` / `rank_ic_many` / `factor_autocorr` / `group_returns` / `group_turnover` / `targets` |

`panelops.__all__` 共 50 个公开名字：32 + 8 个算子名、`elementwise`、
7 个评估核、`SparseTargets`、`set_num_threads` / `get_num_threads`。

2026-11-13 对 `main` 复核：32 / 8 / 27 三张表未变；远端新增的是算子参数
（`ts_sum` / `ts_mean` / `ts_sumif` 的 `exact`，`ts_argmin` / `ts_argmax` /
`ts_idxmin` / `ts_idxmax` 的 `ties`）和 `factor/formula/` 机制，不是新算子。

根 `README.md` 写的是"68 个时序、截面和逐元素算子"。按上面三张表清点是
32 + 8 + 27 = **67**；差的 1 个应是文档把 `elementwise` 这个统一入口本身也算作
一个算子，仓库里没有写明该口径。以本文的代码清点为准，改 README 前先确认口径。

核对命令（在 `make dev` 之后）：

```python
from panelops.api import TS_OPS, CS_OPS, ELEMENTWISE_ARITY
print(len(TS_OPS), len(CS_OPS), len(ELEMENTWISE_ARITY))   # 32 8 27
import panelops as po
print(len(po.__all__))                                    # 50
```

## 2. 统一约定

- 输入必须是 C 连续 float64 二维面板（T×N）；掩码是同形状 bool，行业编号是 int64。
  类型或布局不符直接报错，不做隐式转换。
- 时序窗口**含当日**：取 `max(0, t-window+1) .. t` 行；`min_periods=None` 等于 `window`。
- NaN 和正负无穷都视为缺失；掩码为 False 的格子既是历史缺失也是当日缺失。
- 不读未来行；追加或修改未来行不改变更早的输出位。
- 同一机器同一构建下，不同线程数的结果逐位一致（reduction 不跨序列/截面拆分）。
- 返回新数组，不修改输入；零方差相关、零方差标准化、非法定义域返回 NaN。
- 默认参数在表达式层构建时归一化后进哈希（`min_periods` 缺省补 `window`，
  `alpha` 缺省补 `2/(window+1)`），避免"省略默认值"和"显式写默认值"产生两份缓存。

## 3. 时序算子（32）

### 位移与差分

| 算子 | 签名默认值 | 语义 |
|---|---|---|
| `ts_delay` | `periods=1` | `x[t-periods]`；历史不足 NaN |
| `ts_delta` | `periods=1` | `x[t]-x[t-periods]`；两端观测都必须有效 |

### 基础矩与分布

| 算子 | 语义 |
|---|---|
| `ts_sum` / `ts_mean` | 窗口内有效观测的求和 / 算术平均；`exact=True` 给正确舍入的窗口和（均值 = 该和 / 计数） |
| `ts_std` / `ts_var` | 标准差 / 方差，`ddof=1`（可改 0）；`count <= ddof` 为 NaN |
| `ts_zscore` | 当日中心化值除以滚动标准差；当日缺失或零方差 NaN |
| `ts_median` | 窗口中位数（等于 `ts_quantile(q=0.5)`） |
| `ts_mad` | 窗口内绝对中位差；**不乘 1.4826** |
| `ts_count` | 有限**非零**观测个数；`min_periods` 统计全部有限观测（含零） |
| `ts_product` | 有效观测连乘；中间溢出按 IEEE inf，窗口内有 0 则结果 0 |
| `ts_skew` / `ts_kurt` | 有偏总体三阶 / 四阶标准化中心矩（kurt 减 3）；分别要求 ≥3 / ≥4 个观测 |
| `ts_rank` | 当日值在窗口内的平均名次；`normalize=True` 映射 `(rank-1)/(count-1)`，单样本 0.5 |

### 极值与位置

| 算子 | 语义 |
|---|---|
| `ts_min` / `ts_max` | 有效观测的最小 / 最大值 |
| `ts_argmin` / `ts_argmax` | 距并列极值的交易日数（当日 = 0），按交易行位置计；`ties="latest"`（默认）取最近，`"earliest"` 取最早 |
| `ts_idxmin` / `ts_idxmax` | 名义窗口内的 1 基位置（含缺失占位）；并列按 `ties`（默认 `latest`） |
| `ts_quantile` | 排序后按 `(count-1)*q` 线性插值，`0 <= q <= 1` |

### 回归与关系

| 算子 | 语义 |
|---|---|
| `ts_corr` / `ts_cov` | 配对完整窗口的 Pearson 相关 / 协方差（`ddof=1`）；相关要求 ≥2 对且两边方差 > 0 |
| `ts_slope` / `ts_residual` | x 对**交易行号**带截距回归的斜率 / 当日残差；残差要求当日值有效 |
| `ts_regression_slope` / `ts_regression_residual` | x 对 y 带截距回归的斜率 / 当日残差；常数预测子给 NaN |
| `ts_rsquare` | x 对时间回归的 R²（与时间的 Pearson 相关平方） |

### 加权与递推

| 算子 | 语义 |
|---|---|
| `ts_decay_linear` | 1..window 线性权重（越近越大）；缺失值剔除后幸存权重重新归一 |
| `ts_ema` | `adjust=False` 递推，`alpha=2/(window+1)` 或显式给；首个有效值播种、缺失行保持状态；`min_periods` 统计从开头起的全部有效观测，`lookback` 无界 |

### 条件与前填

| 算子 | 语义 |
|---|---|
| `ts_sumif` | 条件为真处求和；配对有效；全假窗口返回 0；`exact=True` 同 `ts_sum` |
| `ts_ffill` | 最近一个有限、未掩码的值；`limit` 为最大年龄（交易行数），`None` 无限 |

## 4. 截面算子（8）

每行独立计算，缺失输入产生缺失输出。

| 算子 | 关键参数 | 语义 |
|---|---|---|
| `cs_rank` | `normalize=False` | 行内平均名次；可选映射到 [0,1] |
| `cs_zscore` | `ddof=0` | 行内减均值除标准差；零方差 NaN |
| `cs_winsor_quantile` | `lower=0.01, upper=0.99` | 按线性样本分位截断 |
| `cs_winsor_mad` | `scale=3.0` | `median ± scale*MAD` 截断；不乘 1.4826，MAD=0 时全压到中位数 |
| `cs_group_demean` | `groups`（int64 面板） | 组内去均值；负组号 NaN，单样本组给 0 |
| `cs_neutralize` | `exposures` / `groups` / `weights` / `rcond=1e-12` | 带截距、每行业一个指示变量、K 个连续暴露的加权最小二乘取残差；秩亏用零空间最小范数解，不做最小行业样本量或 ridge |
| `cs_scale` | — | 除以绝对值之和；和为零或溢出为 inf 时整行 NaN |
| `cs_demean` | — | 减行均值；常数行精确为 0 |

## 5. 逐元素算子（27）

统一入口 `elementwise(op, *args, mask=None, lower=-1.0, upper=1.0)`，支持标量广播；
`gt/ge/lt/le/eq/ne` 输出 float64 的 1/0；`and_/or_/not_` 把任意有限非零当 True；
`where(cond,a,b)` 只取选中分支；`maximum/minimum` 任一缺失即缺失；
`signed_power` 是 `sign(x)*abs(x)**p`；非法定义域与非有限结果转 NaN；
`is_nan` 对缺失返回 1；`fill_nan` 用给定值替换缺失（股票池掩码仍然优先）。

| 分组 | 算子 | 说明 |
|---|---|---|
| 算术（7） | `add` `subtract` `multiply` `divide` `log` `abs` `negate` | `divide` 分母为 0 按 IEEE 出 inf/NaN，需自己加 epsilon |
| 比较（6） | `gt` `ge` `lt` `le` `eq` `ne` | 输出 1.0 / 0.0 |
| 逻辑（3） | `and_` `or_` `not_` | 无三值短路 |
| 选择（1） | `where` | 三元 |
| 极值/符号/幂（5） | `maximum` `minimum` `sign` `power` `signed_power` | |
| 数学（3） | `sqrt` `exp` `clip` | `clip` 的两个边界是有限标量，`lower <= upper` |
| 缺失（2） | `is_nan` `fill_nan` | |

## 6. 评估核（7）

收益必须由调用方先对齐；这些函数**不做任何日期移位**。

| 函数 | 语义 |
|---|---|
| `ic(x, returns, rank=False)` | 逐日 Pearson 相关，返回长度 T；要求 ≥2 对且两边方差 > 0 |
| `rank_ic(x, returns)` | 先在配对交集上双向排名再 Pearson（Spearman） |
| `rank_ic_many(scores, returns)` | 每天只排一次收益、多个因子复用；每个输出与逐个 `rank_ic` 逐位相同 |
| `factor_autocorr(x, periods=1, rank=True)` | 今日与滞后截面的相关；排名在共同交集上重算 |
| `group_returns(scores, returns, q=5, weights=None)` | 分数升序分组（并列按列号），返回 T×q；只用分数决定分组，缺失收益在组内剔除后权重重归一 |
| `group_turnover(scores, q=5, periods=1)` | 每组 `1 - |今∩昨| / |今|`；历史不足或当组为空 NaN |
| `targets(scores, k=None, fraction=0.2, days=None)` | 取分数最高的目标，返回 CSR 结构的 `SparseTargets(days, indptr, cols, weights)`，权重 = 1/选中数；空行表示清仓 |

`targets` 的 CSR 字段与 `panelbt.Targets` 一一对应，但 `panelops` 不 import
回测包——这是两个包之间唯一的接口点。

## 7. 因子表达式层的额外语法

`src/ashare/factor/expression.py` 在 67 个算子之上提供：

- 中缀 `+ - * /`，分别映射 `add/subtract/multiply/divide`；一元 `-` 映射 `negate`；
- `neutralize(expr, *exposures, groups=None, weights=None, rcond=1e-12)` → `cs_neutralize`；
- `clip(x, lower, upper)`；
- `field(name, dtype="float64", broadcast=False)` 与 `constant(value)` 是叶子；
  `broadcast=True` 声明市场级序列（如指数收益），求值时广播到 N 列并各自应用股票池掩码；
- `mask` / `backend` / `layout` / `groups` / `exposures` / `weights` 不能作为表达式
  节点参数（面板和执行配置不属于表达式），构建时即报错；
- `ts_*` / `cs_*` 从 `panelops.__all__` 自动发现，加算子不用改表达式层；
  逐元素算子需要同时改 `expression.py` 的 `ELEMENTWISE`（当前与 `ELEMENTWISE_ARITY` 一致）；
- 公式型因子的机制在 `src/ashare/factor/formula/`：`dsl.py` 是"一个原文函数一个构造函数"的
  翻译用语言，`qlib.py` 把 Qlib 表达式字符串确定性转成 `Expr`，`plugins.py` 经入口点组
  `ashare.factors` 发现外部因子包，`verify.py` 把候选表达式与独立参考实现逐格核对。
  成套因子不放本仓库（见 `docs/DESIGN.md` 6.2.1），本仓库只放机制和一个内置示例。

## 8. 实现位置与后端

| 层 | 文件 |
|---|---|
| 公共 API、算子表、参数校验 | `packages/panelops/python/panelops/api.py` |
| 导出（`ts_`/`cs_` 按算子表生成） | `packages/panelops/python/panelops/__init__.py` |
| Rust 分派 | `crates/panelops-core/src/lib.rs`（`ts` / `cs` / `elementwise` / `ic` / `grouped` / `targets`） |
| 时序内核 | `crates/panelops-core/src/time.rs` |
| 逐元素内核 | `crates/panelops-core/src/element.rs` |
| 评估内核 | `crates/panelops-core/src/evaluation.rs` |
| NumPy 实现（截面、逐元素、评估核） | `packages/panelops/python/panelops/_numpy.py` |
| 公式型因子机制（DSL、Qlib 转换、插件、核对） | `src/ashare/factor/formula/` |
| 语义规格 | `packages/panelops/SPEC.md` |
| 基准与冻结结果 | `bench/panelops/*.py`、`bench/panelops/results/baseline.json` |
| 测试 | `test/panelops/`（对拍、对抗语料、精确有理数参考、变异） |

后端选择：`BACKENDS` 目前为空，即全部默认走 Rust。公开 API 的
`backend="numpy"` 只在部分算子上可用——`elementwise` 与全部 `cs_*` / 评估核有
NumPy 实现；`ts_*` 的公开 NumPy 分支只配置了 `delay` / `delta`，其余算子的
NumPy 参考用于基准与对拍（`bench/panelops/`），不作为公开后端。没配到就走 Rust，
传非法值报错，不会静默回退。

## 9. 加一个算子要动的地方

1. `api.py`：加进 `TS_OPS` / `CS_OPS`，或 `ELEMENTWISE_ARITY`；写包装函数
   （参数类型与取值范围、默认值归一、`min_periods <= window` 之类的约束）。
2. Rust 内核：`crates/panelops-core` 对应文件加实现，`lib.rs` 加匹配分支；
   保持"不跨序列拆分 reduction"，否则线程数会改变结果。
3. `_numpy.py`：需要 NumPy 后端或对拍时补一份直白实现。
4. `SPEC.md`：写清默认值、缺失/掩码、并列、退化（零方差、常数窗口）的行为。
5. 测试：随机对拍、对抗语料、退化行为、以及"改坏必须有测试失败"的变异检查。
6. 基准：`bench/panelops/` 加入该算子并重跑冻结结果，把选择写成理由。
7. 表达式层：`ts_`/`cs_` 自动可用；逐元素要同步 `expression.py` 的 `ELEMENTWISE`。

## 10. 与 WorldQuant BRAIN 的对照

BRAIN 侧一共 186 个算子（2026-11-13 通过 MCP `get_operators` 取得，含
COMBO / SELECTION / VECTOR 专用）。下面是**语义近似**对照，不是同名映射。

### 有对应或可组合

`ts_corr`、`ts_covariance`≈`ts_cov`、`ts_sum`、`ts_mean`、`ts_std_dev`≈`ts_std`、
`ts_median`、`ts_rank`、`ts_percentage`≈`ts_quantile(q)`、`ts_product`、`ts_delay`、
`ts_delta`、`ts_zscore`、`ts_skewness`、`ts_kurtosis`、`ts_arg_max`、`ts_arg_min`、
`ts_decay_linear`、`ts_backfill`≈`ts_ffill(limit)`、`rank`≈`cs_rank`、`zscore`≈`cs_zscore`、
`winsorize`≈`cs_winsor_*`、`scale`≈`cs_scale`、`if_else`≈`where`、`clamp`≈`clip`、
`greater/less/equal/...`、`and/or/not`、`add/subtract/multiply/divide/log/abs/sign/exp/
sqrt/power/signed_power/is_nan`。

`group_neutralize`≈`cs_group_demean` 或 `cs_neutralize(groups=行业)`；
`regression_neut` / `vector_neut` 的"对一组暴露回归取残差"由 `cs_neutralize` 覆盖。

### 缺的族

| 族 | 代表算子 | 处理建议 |
|---|---|---|
| `group_*`（约 20 个） | `group_rank`、`group_zscore`、`group_scale`、`group_sum/mean/std/max/min/count/median`、`group_backfill`、`group_normalize` | 只有组内去均值/中性化可直接表达；组内排序、组内标准化需要按行业掩码逐组循环，或加 `cs_group_rank` 一类算子 |
| `transform` 族 | `trade_when`、`hump`、`hump_decay`、`jump_decay`、`keep`、`filter`、`tail/left_tail/right_tail`、`bucket` | 降换手与条件交易是主要缺口；`bucket` 可用 `cs_quantile` + 整数映射近似 |
| `vector` 族（13 个） | `vec_*` | A 股面板没有向量字段，属结构性缺失 |
| `reduce` / `combo` / `selection` | `reduce_*`、`combo_a`、`self_corr`、`universe_size` | 对应本项目的组合优化/回测层，不放在算子库 |
| 部分 `ts` 变体 | `ts_ir`、`ts_moment`、`ts_av_diff`、`ts_returns`、`ts_scale`、`ts_step`、`ts_entropy`、`ts_partial_corr`、`ts_triple_corr`、`ts_co_skewness`、`ts_co_kurtosis`、`ts_theilsen`、`ts_poly_regression`、`ts_vector_proj/neut`、`ts_decay_exp_window`、`ts_weighted_decay`、`ts_delta_limit`、`kth_element`、`days_from_last_change`、`last_diff_value` | 一部分可由现有算子组合：`ts_ir`=mean/std、`ts_returns`=delta/close、`ts_step`=行号、`ts_av_diff`=x−mean；其余需要新核 |
| 算术细节 | `inverse`、`log_diff`、`round/floor/ceiling/round_down/fraction`、`sigmoid/tanh/arc_*`、`nan_out/nan_mask/replace/purify/pasteurize/to_nan` | 多为低频，`sign`+`power` 可拼一部分 |

结论：覆盖 BRAIN REGULAR 里"量价时序 + 截面标准化/中性化"的主干（高频算子
约 25～30 个），缺分组运算、降换手变换、向量字段三族。

### BRAIN 算子名快照（186，便于以后 diff）

```
add exp multiply sign subtract pasteurize log purify arc_tan max arc_sin to_nan abs
round_down sigmoid divide min nan_mask arc_cos tanh nan_out signed_power inverse round
sqrt s_log_1p reverse replace log_diff power densify fraction floor ceiling or and not
is_not_finite negate is_nan is_not_nan less equal greater is_finite if_else not_equal
less_equal greater_equal ts_corr ts_zscore ts_returns ts_product ts_std_dev
ts_triple_corr ts_backfill days_from_last_change last_diff_value ts_scale ts_entropy
ts_vector_neut ts_percentage ts_step ts_sum ts_co_kurtosis ts_partial_corr inst_tvr
ts_decay_exp_window ts_av_diff ts_rank_gmean_amean_diff ts_kurtosis ts_mean
ts_min_max_diff ts_arg_max ts_max ts_min_max_cps ts_rank ts_ir ts_delay ts_theilsen
hump_decay ts_weighted_decay ts_quantile ts_min ts_count_nans ts_covariance
ts_co_skewness ts_min_diff ts_decay_linear jump_decay ts_moment ts_arg_min
ts_regression ts_skewness ts_max_diff kth_element hump ts_median ts_delta
ts_poly_regression ts_vector_proj ts_target_tvr_decay ts_target_tvr_delta_limit
ts_target_tvr_hump ts_delta_limit winsorize rank regression_proj vector_neut
regression_neut zscore scale_down truncate scale rank_by_side normalize
generalized_rank one_side rank_gmean_amean_diff quantile vector_proj multi_regression
vec_norm vec_percentage vec_kurtosis vec_ir vec_min vec_count vec_sum vec_skewness
vec_max vec_powersum vec_avg vec_stddev vec_range vec_choose vec_filter right_tail
bucket tail filter clamp keep left_tail trade_when generate_stats group_min group_mean
group_median group_max group_percentage group_vector_neut group_rank group_vector_proj
group_normalize group_coalesce group_extra group_backfill group_scale group_count
group_zscore group_std_dev group_sum group_neutralize group_multi_regression
group_cartesian_product combo_a convert inst_pnl self_corr in universe_size reduce_ir
reduce_avg reduce_powersum reduce_max reduce_min reduce_norm reduce_sum reduce_range
reduce_percentage reduce_skewness reduce_count reduce_kurtosis reduce_choose
reduce_stddev
```
