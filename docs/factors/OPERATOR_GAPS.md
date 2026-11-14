# 算子缺口与补法

对照三套公式用到的全部函数，逐个列出本仓库原来能否表达、怎么补。补法分三种：**已有**（现成算子直接对应）、**组合**（在 `src/ashare/factor/formula/ops.py` 用现有算子精确拼出，不写新内核）、**新内核**（在 `panelops` 里加）。另有 **不补**：对应的原文条目不收（理由见 `CONVENTIONS.md` 第 4 节）。

组合的精确性靠两条事实，都有测试：比较、逻辑运算的结果恰好是 1.0 / 0.0（或 NaN）；IEEE 除以 1.0 不改变任何位，除以 0.0 按 SPEC 得 NaN。所以「条件为真保留、否则 NaN」写成 `x / keep`，逐位不改变被保留的值。

## 1. 缺口一览

| 原文函数（出处） | 原来能否表达 | 补法 | 实现 | 测试 |
|---|---|---|---|---|
| 截面百分位 `rank`（A101）、`RANK`（国君），口径 秩/个数 | 不能：`cs_rank(normalize=True)` 是 (秩−1)/(个数−1)，又没有截面计数 | 组合：`cs_rank(x) / n`，`n = 2·cs_rank(x·0+1) − 1`（全部相等的值平均秩为 (n+1)/2，整数与半整数在双精度里精确） | `ops.cs_pct_rank`、`ops.cs_valid_count` | `test_cs_pct_rank_matches_pandas`（对 pandas `rank(pct=True)`，含全空行、并列） |
| 时序百分位 `ts_rank`/`Ts_Rank`（A101）、`TSRANK`（国君）、Qlib `Rank` | 不能：同上 | 组合：`ts_rank(x) / 窗口有效个数`，有效个数 = `ts_sum(not_(is_nan(x)), N, min_periods=1)` | `ops.ts_pct_rank`、`ops.ts_valid_count`、`ops.qlib_rank` | `test_ts_pct_rank_and_valid_count_match_pandas`（对 pandas `rolling.rank(pct=True)`，窗口 1/3/10/30，含长缺口、常数列）、手算小例 |
| 并列取最早的位置：Qlib `IdxMax/IdxMin`（`np.argmax`）、A101 `ts_argmax` 的读法 B、国君 `HIGHDAY/LOWDAY` 的备选 | 不能：`ts_argmax/ts_idxmax` 一族固定取最近一次 | **新内核选项**：`ties="earliest"`（见第 2 节）。组合的写法要对窗口内每一天各做一次延迟和比较，W=60 时是 60 张中间面板，明显慢 | `panelops` 四个函数加 `ties` 参数 | `test/panelops/test_ties.py` |
| Qlib `IdxMax/IdxMin` 的其余行为：窗口有 NaN 时返回第一个 NaN 的位置；上市不满 N 行时位置从上市第一行数起 | 不能 | 组合：在上一行的基础上，用 `is_nan(x)·上市区间` 的「并列取最早的最大值位置」找第一个 NaN；用 `ts_sum(上市区间)` 把名义窗口位置换成 Qlib 的实际窗口位置；没有有效值时置 NaN | `ops.qlib_idx` | `test_qlib_idx_matches_rolling_apply`（逐只股票对 pandas `rolling.apply(lambda x: x.argmax()+1, raw=True)`，含上市前后、停牌、并列、常数列）；关掉 NaN 规则的版本另测 |
| Qlib 比较 `>`、`<`（NaN 比较为 False） | 不能：`gt/lt` 遇 NaN 给 NaN | 组合：`fill_nan(gt(a, b), 0) × 上市区间` | `ops.qlib_compare` | `test_qlib_compare_counts_missing_as_false` |
| Qlib「上市区间」：股票序列只有上市期间的行 | 不能直接表达 | 组合：`is_listed / is_listed`（1 或 0/0=NaN）；转换程序在每条表达式最外层乘它 | `ops.listed_domain`、`qlib.to_expr` | 核对工具全量结果（掩码逐格一致） |
| Qlib `Corr`、`Rsquare` 的「标准差 ≤ 2e-5 置 NaN」 | 不能 | 组合：`ts_std(·, N, min_periods=1) ≤ 2e-5` 得 0/1，再 `x / eq(判定, 0)` | `ops.qlib_corr`、`ops.qlib_rsquare` | `test_qlib_corr_and_rsquare_guards` |
| 国君 `SMA(A, n, m)` | 部分：`ts_ema(alpha=m/n)` 是同一递推，但停牌日输出保持值 | 组合：`ts_ema` 加「当天输入 NaN 时输出 NaN」（约定 D-10） | `ops.ewm`、`dsl.sma` | `test_ewm_matches_pandas_recursion`（对 pandas `ewm(adjust=False, ignore_na=True)`）、手算小例 |
| 国君 `WMA(A, n)`（权重 0.9^i） | 不能 | 组合：`Σ 0.9^i · delay(A, i) / Σ 0.9^i`，n 个延迟（只在 027、030 出现，n = 12、20） | `ops.decayed_mean`、`dsl.wma` | `test_decayed_mean_matches_direct_weights` |
| 国君 `FILTER(A, cond)` | 不能 | 组合：`A / ne(cond, 0)`：条件真保留、假或 NaN 给 NaN | `ops.filter_out`、`dsl.filter_` | `test_filter_out` |
| 截面计数 | 不能 | 组合（见第一行） | `ops.cs_valid_count` | 同第一行 |
| 基准指数 `BANCHMARKINDEXOPEN/CLOSE` | 面板没有 | 取数：`PublicSource` 从数据层 `index_daily` 取，作为市场序列字段（`field(..., broadcast=True)`）广播 | `inputs.PublicSource`、`dsl.benchmark_*` | `test_fields_follow_conventions` |
| 行业组号 `IndClass.*` | 面板 `industry` 是 uint32、0 表示缺失；直接当组号会把缺失的股票并成一组 | 取数：`PublicSource` 派生 `industry_l1`（int64，缺失 −1，`cs_group_demean` 对负组号给 NaN） | `inputs.PublicSource`、`dsl.industry` | `test_cross_section_scale_and_neutralize` |
| `SUMAC`（国君 165、183） | 不能（累加和，窗口与区间无法确定） | 不补 | — | — |
| `SELF`（国君 143） | 不能（递归，起点未定义） | 不补 | — | — |
| 多元回归残差 `REGRESI(y, x1, x2, x3, n)`（国君 030） | 不能 | 不补（也没有 MKT/SMB/HML 数据） | — | — |

## 2. 已有算子的对应（无缺口）

| 原文 | 本仓库 |
|---|---|
| `abs log sign`，`+ - * /`，比较，`||`/`&`/`&&`，`?:` | 逐元素 `abs log sign add subtract multiply divide gt ge lt le eq ne or_ and_ where` |
| `x ^ y`、`SignedPower` | `power`、`signed_power`（指数可以是序列） |
| `max/min(x, y)` 逐元素、`MAX/MIN(A, B)` | `maximum`、`minimum` |
| `delay/DELAY`、`delta/DELTA`、Qlib `Ref` | `ts_delay`、`ts_delta` |
| `sum/SUM`、`MEAN`、`adv{d}`、Qlib `Sum/Mean` | `ts_sum`、`ts_mean` |
| `stddev/STD`、Qlib `Std` | `ts_std(ddof=1)` |
| `ts_min/ts_max`、`TSMIN/TSMAX`、Qlib `Min/Max` | `ts_min`、`ts_max` |
| `ts_argmax/ts_argmin`（读法 A）、`HIGHDAY/LOWDAY` | `ts_argmax`、`ts_argmin`（距今天数，并列取最近） |
| `correlation/CORR`、`covariance/COVIANCE` | `ts_corr`、`ts_cov(ddof=1)` |
| `decay_linear/DECAYLINEAR` | `ts_decay_linear` |
| `product/PROD` | `ts_product` |
| `COUNT`、`SUMIF` | `ts_count`、`ts_sumif` |
| `REGBETA/REGRESI(A, B, n)` | `ts_regression_slope`、`ts_regression_residual` |
| `REGBETA/REGRESI(A, SEQUENCE(n), n)`、Qlib `Slope/Resi/Rsquare` | `ts_slope`、`ts_residual`、`ts_rsquare` |
| Qlib `Quantile` | `ts_quantile` |
| `scale(x, a)` | `a · cs_scale(x)` |
| `indneutralize(x, g)` | `cs_group_demean(x, groups)` |

## 3. 新内核：`ties="earliest"`

`ts_argmin`、`ts_argmax`、`ts_idxmin`、`ts_idxmax` 新增关键字参数 `ties`，取 `"latest"`（默认，原行为）或 `"earliest"`（并列时取窗口里最早的一次）。极值本身与选哪一次无关，`ts_min/ts_max` 不受影响。

- 实现：`crates/panelops-core/src/time.rs` 的两趟分块极值算法里有三处比较（块内后缀从新到旧扫、块内前缀从旧到新扫、后缀与前缀合并），原来的并列规则分别是「保留较新的」；`earliest` 用位运算把相等的情况改判给较旧的一方，仍然是无分支的选择，不改动其他逻辑。`Options` 加 `earliest` 字段，Python 绑定 `_core.ts(..., earliest=False)` 默认值不变，旧的位置参数调用照常可用。
- 约定：NaN 语义、`min_periods`、掩码、名义窗口位置、单线程与多线程逐位一致，全部与原算子相同（`SPEC.md` 已补写）。
- 测试（`test/panelops/test_ties.py`，46 项）：四个算子 × 窗口 1/2/5/20/60 × 两种并列规则 × 三档 `min_periods`，对照一份按 SPEC 逐格扫描的独立实现和 `reference.py`（大量并列的整数、常数列、长缺口、±0、inf）；2100 列宽面板加掩码，1 线程与 4 线程逐位相同、跨 1024 列分块边界；完整窗口时等于 `np.argmax/np.argmin + 1`。变异检查：三处比较各删掉 `earliest` 那一项，结果见下。
- 基准：见 `docs/PANELOPS_BENCHMARKS.md` 末节「并列取最早」。

## 4. 组合的代价

组合算子只在 Alpha158 和翻译里用；核对工具实测 158 条 Alpha158 候选在 1010 × 5923 的面板上求值加比较每 20 条约 1 秒（负载 7–9，见 `ALPHA158_VERIFY.md`），不构成瓶颈，所以没有再为它们写内核。`qlib_idx` 是最重的组合（每次 5 个滚动算子），IMAX/IMIN/IMXD 共 15 条。
