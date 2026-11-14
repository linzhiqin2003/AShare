# 公开因子翻译规范（Alpha101、国泰君安 191）

给逐条翻译公式的人看。只用本文件和公式原文（`sources/alpha101.jsonl`、`sources/gtja191.jsonl` 的 `formula` 字段；函数定义的原文不随公开仓库发布，查论文附录 A 与研报附录 2，口径以 `CONVENTIONS.md` 为准）就能干活，不需要看代码库。

翻译的目标：把每条原文公式**一比一**改写成本仓库的表达式。不化简、不重排、不按自己对因子含义的理解改任何东西。本文没覆盖的写法一律停下来标记（第 7 节），不要猜。

## 1. 交付格式

每套一个 Python 文件（`src/ashare/factor/alphas/alpha101.py`、`src/ashare/factor/alphas/gtja191.py`），内容只有一行导入和一串函数：

```python
from ashare.factor.formula import dsl as d


def alpha101_001():
    """(rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)"""
    return (
        d.rank(
            d.ts_argmax(
                d.signed_power(d.where(d.lt(d.returns, 0), d.stddev(d.returns, 20), d.close), 2.0),
                5,
            )
        )
        - 0.5
    )
```

- 函数名：`alpha101_001` … `alpha101_101`，`gtja191_001` … `gtja191_191`，三位数编号，与 JSONL 的 `n` 一致。没有参数。
- 文档字符串：**原样**抄 JSONL 的 `formula` 字段（国君以研报原文为准，不用 `formula_alt`）。
- 返回值：一个表达式（下面的 `d.` 函数和字段拼出来的对象）。
- 文件末尾写一个字典 `NOTES = {"gtja191_xxx": "标记：…", ...}`，**只登记第 7 节要标记、因而没写函数的条目**。按第 6 节处理过的条目不用登记：它们的元数据标记（「两读」「补参数」等）由注册表按条目编号自动挂上。
- 第 6 节列为不收的 5 条**不写函数**（写了注册表会报错），也不用进 `NOTES`。

## 2. 写法

### 2.1 字段（都是 `d.` 下的名字，直接用，不加括号）

| 原文 | 写法 | 含义（实现方负责口径，翻译方不用管） |
|---|---|---|
| `open` `OPEN` | `d.open` | 后复权开盘价 |
| `high` `HIGH` | `d.high` | 后复权最高价 |
| `low` `LOW` | `d.low` | 后复权最低价 |
| `close` `CLOSE` | `d.close` | 后复权收盘价 |
| `vwap` `VWAP` | `d.vwap` | 后复权均价 |
| `volume` `VOLUME` | `d.volume` | 成交量 |
| `AMOUNT` | `d.amount` | 成交额（元） |
| `returns`（A101）、`RET`（国君） | `d.returns` | 日收益率；上市首日缺失 |
| `cap`（A101） | `d.cap` | 市值 |
| `adv20`、`adv180` 等（A101） | `d.adv(20)`、`d.adv(180)` | d 日平均成交额；数字照写 |
| `BANCHMARKINDEXOPEN` | `d.benchmark_open` | 基准指数开盘点位（拼写原文如此，照此对应） |
| `BANCHMARKINDEXCLOSE` | `d.benchmark_close` | 基准指数收盘点位 |
| `DTM` `DBM` `TR` `HD` `LD`（国君） | `d.dtm` `d.dbm` `d.tr` `d.hd` `d.ld` | 研报定义表里的派生变量，已按定义实现，**不要展开** |
| `IndClass.sector` / `.industry` / `.subindustry` | 作为 `d.indneutralize` 的第二个参数，写成字符串 `"sector"`、`"industry"`、`"subindustry"` | 行业层级 |

### 2.2 数字

原文的数字原样写成 Python 数字：`5`、`0.5`、`2.`（写成 `2.0`）、`1e-12`。不要自己算（`(1 - 0.728317)` 就写 `(1 - 0.728317)`，不写 `0.271683`）。天数参数也原样写，包括非整数（`8.22237`），实现方统一向下取整。

### 2.3 运算

| 原文 | 写法 | 说明 |
|---|---|---|
| `a + b`、`a - b`、`a * b`、`a / b`、`-a` | 同样写 | Python 的优先级与结合方向与原文相同：先乘除后加减，同级从左到右 |
| `a ^ b`（幂） | `d.power(a, b)` | **不要写 `a ^ b` 或 `a ** b`**：Python 的 `^` 是按位异或，`**` 对表达式没有定义 |
| `a < b` `a > b` `a <= b` `a >= b` | `d.lt(a, b)` `d.gt(a, b)` `d.le(a, b)` `d.ge(a, b)` | **不要写 Python 的 `<` `>`**（会报错） |
| `a == b`、国君的 `a = b` | `d.eq(a, b)` | **绝不能写 Python 的 `==`**：它比较的是两个表达式结构是否相同，返回 True/False，不会报错，是最危险的错误 |
| `a != b` | `d.ne(a, b)` | |
| `a || b`、`a OR b` | `d.or_(a, b)` | |
| `a & b`、`a && b` | `d.and_(a, b)` | |
| `cond ? a : b`（含全角 `？：`） | `d.where(cond, a, b)` | 嵌套的 `?:` 从右边结合：`c1 ? a : c2 ? b : c` = `d.where(c1, a, d.where(c2, b, c))` |
| `(-1 * 1)` | `(-1 * 1)` | 照写 |

**优先级**（原文没有括号时）从高到低：函数调用与括号；`^`；一元负号；`*` `/`；`+` `-`；比较；`&` `&&`；`||` `OR`；`?:`。`-x^2` 是 `-(x^2)`，写成 `-d.power(x, 2)`。拿不准的写法按第 7 节标记。

### 2.4 函数

原文函数名大小写不敏感（`Ts_Rank` 就是 `ts_rank`）。对应关系见第 5 节的两张表，语义见第 4 节。嵌套照原文一层层写进去，括号层次保持不变。`IndNeutralize(x, IndClass.industry)` 写成 `d.indneutralize(x, "industry")`。

## 3. 五个例子（由易到难）

1. 国君 014 `CLOSE-DELAY(CLOSE,5)`

   ```python
   return d.close - d.delay(d.close, 5)
   ```

2. Alpha101 012 `(sign(delta(volume, 1)) * (-1 * delta(close, 1)))`

   ```python
   return d.sign(d.delta(d.volume, 1)) * (-1 * d.delta(d.close, 1))
   ```

3. 国君 053 `COUNT(CLOSE>DELAY(CLOSE,1),12)/12*100`

   ```python
   return d.ts_count(d.gt(d.close, d.delay(d.close, 1)), 12) / 12 * 100
   ```

4. Alpha101 001 `(rank(Ts_ArgMax(SignedPower(((returns < 0) ? stddev(returns, 20) : close), 2.), 5)) - 0.5)`

   ```python
   return (
       d.rank(
           d.ts_argmax(
               d.signed_power(d.where(d.lt(d.returns, 0), d.stddev(d.returns, 20), d.close), 2.0),
               5,
           )
       )
       - 0.5
   )
   ```

5. Alpha101 072 `(rank(decay_linear(correlation(((high + low) / 2), adv40, 8.93345), 10.1519)) / rank(decay_linear(correlation(Ts_Rank(vwap, 3.72469), Ts_Rank(volume, 18.5188), 6.86671), 2.95011)))`

   ```python
   return d.rank(d.decay_linear(d.correlation((d.high + d.low) / 2, d.adv(40), 8.93345), 10.1519)) / d.rank(
       d.decay_linear(
           d.correlation(d.ts_rank(d.vwap, 3.72469), d.ts_rank(d.volume, 18.5188), 6.86671),
           2.95011,
       )
   )
   ```

这五个例子在仓库里有测试，能求值，前三个与 NumPy 直算逐格一致。

## 4. 每个函数的精确语义

下面所有说法都已对着实现和测试核实。术语：**第 t 行**是第 t 个交易日；**缺失**指 NaN（停牌、上市前、退市后、算出非有限数）。

**通则**

- **窗口含当日**：d 日窗口是第 t−d+1 到第 t 行，按交易日行数计；停牌的日子占一行（值是缺失）。
- **窗口必须满**：窗口内 d 行全部有值才出数，否则结果缺失；所以每个滚动函数输出的前 d−1 行一定缺失，停牌一天会让之后 d−1 天的这个窗口统计缺失。
- **天数取整**：所有天数参数先向下取整（`8.22237` → 8，`1.06157` → 1）。窗口类取整后至少要 1；`delay/delta` 可以是 0（`delay(x, 0)` 就是 x）。
- **逐元素运算**：任一操作数缺失，结果缺失；结果不是有限数（除以 0、log 非正数、负数开非整数次方、溢出）一律缺失。
- **截面运算**：缺失的股票不参与，自身结果缺失。

**时间序列**

| 写法 | 语义 |
|---|---|
| `d.delay(x, k)` | 第 t−k 行的值（向过去取）；前 k 行缺失 |
| `d.delta(x, k)` | `x − delay(x, k)`，两端都要有值 |
| `d.ts_sum(x, k)`、`d.ts_mean(x, k)` | 窗口和、均值 |
| `d.stddev(x, k)` | 样本标准差，分母 k−1；窗口内值完全相同时恰为 0 |
| `d.ts_min(x, k)`、`d.ts_max(x, k)` | 窗口最小、最大值 |
| `d.ts_rank(x, k)` | 当天值在窗口内的平均秩除以 k，取值 (0, 1]；并列取平均秩（窗口 1,3,3 里今天的 3 得 2.5/3） |
| `d.ts_argmax(x, k)`、`d.ts_argmin(x, k)` | 默认：最大（小）值出现在几天前，今天为 0，取值 0..k−1；并列取最近的一次。注册表可以整体切到另一种读法（窗口内位置，最早一天为 1，并列取最早）。**翻译时一律写 `d.ts_argmax`/`d.ts_argmin`，不要自己选读法，也不要用 `d.highday` 代替** |
| `d.highday(x, k)`、`d.lowday(x, k)` | 国君 `HIGHDAY`、`LOWDAY` 专用：距今天数，今天为 0，并列取最近一次；不随 `ts_argmax` 的读法切换 |
| `d.ts_product(x, k)` | 窗口连乘 |
| `d.correlation(x, y, k)` | 窗口内 Pearson 相关；任一序列窗口内值全部相同（方差 0）时缺失 |
| `d.covariance(x, y, k)` | 样本协方差，分母 k−1 |
| `d.decay_linear(x, k)` | 加权平均，今天权重 k、昨天 k−1 …… 最早一天 1，除以权重和 |
| `d.ts_count(cond, k)` | 窗口内条件为真（非零）的天数；条件缺失算窗口不满 |
| `d.ts_sumif(x, cond, k)` | 窗口内条件为真的那些天的 x 之和；全假为 0。**参数顺序与原文 `SUMIF(A, n, cond)` 不同**：写成 `d.ts_sumif(A, cond, n)` |
| `d.sma(x, n, m)` | 国君 SMA：`Y_t = (m·x_t + (n−m)·Y_{t−1}) / n`，第一个有效值作初值；至少累计 n 个有效值才出数；当天 x 缺失时输出缺失（但递推状态保留到下一个有效日）；要求 0 < m ≤ n |
| `d.wma(x, n)` | 国君 WMA：权重 0.9^i，i 为距今天数（今天 i=0），除以权重和；窗口必须满 |
| `d.regbeta(y, x, n)` | y 对 x 加截距回归的斜率（`REGBETA(A, B, n)` 写成 `d.regbeta(A, B, n)`，A 是因变量）；x 在窗口内为常数时缺失。`d.regbeta(y, x, n, min_pairs=m)`：窗口内有效配对不少于 m 对就出数，**只在国君 149 用**（第 6 节） |
| `d.regresi(y, x, n)` | 同一回归在今天的残差 |
| `d.ts_slope(x, n)` | `REGBETA(x, SEQUENCE(n), n)`：x 对时间 1..n 回归的斜率 |
| `d.ts_residual(x, n)` | `REGRESI(x, SEQUENCE(n), n)`：x 对时间回归在今天的残差 |
| `d.filter_(x, cond)` | 国君 FILTER：条件不成立（或缺失）的日子为缺失，成立的日子为 x |
| `d.mask_missing(x, ref)` | ref 缺失的日子 x 也缺失，其余原样；**只在第 6 节指明的地方用**（国君 149） |

**截面**

| 写法 | 语义 |
|---|---|
| `d.rank(x)` | 当天截面百分位：平均秩除以当天有效个数，取值 (0, 1]；并列取平均（[3, 1, 缺, 1, 2] 得 [1, 0.375, 缺, 0.375, 0.75]） |
| `d.scale(x)`、`d.scale(x, a)` | 除以当天有效值绝对值之和，再乘 a（默认 1） |
| `d.indneutralize(x, level)` | 当天同一行业内减去行业均值；行业缺失的股票结果缺失；一个行业只有一只股票时为 0 |

**逐元素**

| 写法 | 语义 |
|---|---|
| `d.abs(x)`、`d.log(x)`、`d.sign(x)`、`d.sqrt(x)` | 标准定义；`sign(0) = 0` |
| `d.power(x, y)` | x 的 y 次方（y 可以是序列）；负数的非整数次方缺失 |
| `d.signed_power(x, a)` | sign(x)·\|x\|^a；a 可以是序列 |
| `d.maximum(x, y)`、`d.minimum(x, y)` | 逐元素取大、取小；任一侧缺失则缺失（不是忽略缺失取另一侧） |
| `d.lt` `d.gt` `d.le` `d.ge` `d.eq` `d.ne` | 结果 1 或 0；任一侧缺失则缺失；`d.eq`、`d.ne` 是精确的浮点比较，不加容差（原文 `== 1`、`== 0.05` 照写） |
| `d.and_(a, b)`、`d.or_(a, b)`、`d.not_(a)` | 非零为真，结果 1 或 0；任一侧缺失则缺失（不短路） |
| `d.where(c, a, b)` | c 非零取 a、为零取 b；c 缺失则缺失；只看被选中那一侧是否缺失 |

## 5. 映射表

### 5.1 Alpha101 函数

| 原文 | 写法 |
|---|---|
| `abs(x)` `log(x)` `Log(x)` `sign(x)` `Sign(x)` | `d.abs(x)` `d.log(x)` `d.log(x)` `d.sign(x)` `d.sign(x)` |
| `rank(x)` | `d.rank(x)` |
| `delay(x, d)` | `d.delay(x, d)` |
| `delta(x, d)` | `d.delta(x, d)` |
| `correlation(x, y, d)` | `d.correlation(x, y, d)` |
| `covariance(x, y, d)` | `d.covariance(x, y, d)` |
| `scale(x)`、`scale(x, a)` | `d.scale(x)`、`d.scale(x, a)` |
| `SignedPower(x, a)`、`signedpower(x, a)` | `d.signed_power(x, a)` |
| `x ^ a` | `d.power(x, a)` |
| `decay_linear(x, d)` | `d.decay_linear(x, d)` |
| `IndNeutralize(x, IndClass.level)`、`indneutralize(...)` | `d.indneutralize(x, "level")` |
| `ts_min(x, d)` `ts_max(x, d)` | `d.ts_min(x, d)` `d.ts_max(x, d)` |
| `ts_argmax(x, d)` `Ts_ArgMax(x, d)` | `d.ts_argmax(x, d)` |
| `ts_argmin(x, d)` `Ts_ArgMin(x, d)` | `d.ts_argmin(x, d)` |
| `ts_rank(x, d)` `Ts_Rank(x, d)` | `d.ts_rank(x, d)` |
| `min(x, d)`（d 是数字） | `d.ts_min(x, d)` |
| `max(x, d)`（d 是数字） | `d.ts_max(x, d)` |
| `min(x, y)`（两个都是表达式） | `d.minimum(x, y)`（Q-A-01） |
| `max(x, y)`（两个都是表达式） | `d.maximum(x, y)`（Q-A-01） |
| `sum(x, d)` | `d.ts_sum(x, d)` |
| `product(x, d)` | `d.ts_product(x, d)` |
| `stddev(x, d)` | `d.stddev(x, d)` |
| `x ? y : z`，`||`，`==`，`<`，`>` | 见 2.3 |

### 5.2 国泰君安 191 函数

| 原文 | 写法 |
|---|---|
| `RANK(A)` | `d.rank(A)` |
| `MAX(A, B)`、`MIN(A, B)`（两个都是表达式，或一个是 0） | `d.maximum(A, B)`、`d.minimum(A, B)` |
| `MAX(A, n)`、`MIN(A, n)`（n 是大于 0 的整数） | 按 Q-G-02 写成 `d.ts_max(A, n)`、`d.ts_min(A, n)` |
| `STD(A, n)` | `d.stddev(A, n)` |
| `CORR(A, B, n)` | `d.correlation(A, B, n)` |
| `DELTA(A, n)` | `d.delta(A, n)` |
| `LOG(A)` `ABS(A)` `SIGN(A)` | `d.log(A)` `d.abs(A)` `d.sign(A)` |
| `SUM(A, n)` | `d.ts_sum(A, n)` |
| `MEAN(A, n)` | `d.ts_mean(A, n)` |
| `TSRANK(A, n)` | `d.ts_rank(A, n)` |
| `COVIANCE(A, B, n)` | `d.covariance(A, B, n)` |
| `DELAY(A, n)` | `d.delay(A, n)` |
| `TSMIN(A, n)` `TSMAX(A, n)` | `d.ts_min(A, n)` `d.ts_max(A, n)` |
| `PROD(A, n)` | `d.ts_product(A, n)` |
| `COUNT(cond, n)` | `d.ts_count(cond, n)` |
| `REGBETA(A, B, n)` | `d.regbeta(A, B, n)` |
| `REGRESI(A, B, n)` | `d.regresi(A, B, n)` |
| `REGBETA(A, SEQUENCE(n), n)` | `d.ts_slope(A, n)` |
| `REGRESI(A, SEQUENCE(n), n)` | `d.ts_residual(A, n)` |
| `SMA(A, n, m)` | `d.sma(A, n, m)` |
| `SUMIF(A, n, cond)` | `d.ts_sumif(A, cond, n)`（注意顺序） |
| `WMA(A, n)` | `d.wma(A, n)` |
| `DECAYLINEAR(A, d)` | `d.decay_linear(A, d)` |
| `FILTER(A, cond)` | `d.filter_(A, cond)` |
| `HIGHDAY(A, n)` | `d.highday(A, n)` |
| `LOWDAY(A, n)` | `d.lowday(A, n)` |
| `SEQUENCE(n)` | 只能出现在 `REGBETA/REGRESI` 的第二个参数，见上两行；出现在别处就标记 |
| `SUMAC(A, n)` | 没有对应，标记（这类条目不收） |
| `A ^ B` | `d.power(A, B)` |
| `&` `||` `OR` `?:` `=` `==` `<` `>` `<=` `>=` | 见 2.3 |
| `./` `.*`（MATLAB 写法） | 按 `/` `*` |

### 5.3 输入数据

见 2.1。另外：`SELF`、`MKT`、`SMB`、`HML`、`MKE` 没有对应，含这些名字的条目标记不收。

## 6. 已知疑点怎么翻（负责人 2026-11-13 已裁决，照下表翻；元数据标记自动挂上，不用登记）

Alpha101：

| 编号 | 条目 | 怎么翻 |
|---|---|---|
| Q-A-01 | 071、073、076、077、082、087、088、092、096 | `max/min` 两个参数都是表达式：`d.maximum/d.minimum` |
| Q-A-02 | 059、066、082、089 | `(x * w) + (x * (1 - w))` 照写，不化简成 x |
| Q-A-04 | 100 | 两层 `indneutralize` 照写 |

国泰君安 191（以研报 `formula` 为准）：

| 编号 | 条目 | 怎么翻 |
|---|---|---|
| Q-G-02 | 007、017、028（第一处 `MAX(HIGH,9)`）、041、064、091、108、119、121、127、154、157、162、164 | `MAX/MIN(x, 正整数)` 写成 `d.ts_max/d.ts_min(x, 正整数)`；`MAX(x, 0)`、`MAX(0, x)` 仍是逐元素 |
| Q-G-010 | 010 | 按 `RANK(TSMAX(((RET < 0) ? STD(RET, 20) : CLOSE)^2, 5))` |
| Q-G-021、147 | 021、147 | `REGBETA(MEAN(CLOSE,n),SEQUENCE(n))` 写成 `d.ts_slope(d.ts_mean(d.close, n), n)` |
| Q-G-116 | 116 | `d.ts_slope(d.close, 20)` |
| Q-G-022 | 022 | `SMEAN(…,12,1)` 按 `d.sma(…, 12, 1)` |
| Q-G-023 | 023 | `STD(CLOSE:20)` 按 `d.stddev(d.close, 20)` |
| Q-G-028 | 028 | 第二个 SMA 的分母按 `(TSMAX(HIGH,9)-TSMIN(LOW,9))` |
| Q-G-036 | 036 | 按 `RANK(SUM(CORR(RANK(VOLUME), RANK(VWAP), 6), 2))` |
| Q-G-052 | 052 | `-L` 按 `-LOW` |
| Q-G-054 | 054 | `STD(ABS(CLOSE - OPEN))` 按窗口 5 |
| Q-G-069 | 069 | 全角 `？：` 按 `?:` |
| Q-G-078 | 078 | `MA` 按 `MEAN` |
| Q-G-111 | 111 | `VOL` 按 `VOLUME` |
| Q-G-127 | 127 | 外层 `MEAN(…)` 按窗口 12；`MAX(CLOSE,12)` 按 `TSMAX` |
| Q-G-131 | 131 | `DELAT` 按 `DELTA` |
| Q-G-146 | 146 | 去掉末尾分号；末尾 `SMA(…,60)` 按 `SMA(…,60,2)` |
| Q-G-173 | 173 | 去掉末尾分号 |
| Q-G-159 | 159 | `HGIH` 按 `HIGH` |
| Q-G-181 | 181 | 第二个 `SUM` 按窗口 20 |
| Q-G-182 | 182 | `OR` 按 `d.or_` |
| Q-G-188 | 188 | en dash 按减号 |
| Q-G-190 | 190 | `DELAY(CLOSE)` 按 `DELAY(CLOSE,1)` |
| Q-G-003、069 | 003、019、059、069 | 单个 `=` 按 `d.eq` |
| Q-G-149 | 149 | `REGBETA(FILTER(A, cond), FILTER(B, cond), 252)` 写成 `d.mask_missing(d.regbeta(d.filter_(A, cond), d.filter_(B, cond), 252, min_pairs=20), d.close / d.delay(d.close, 1) - 1)`（参照序列照原文字面写成股票自己的 `CLOSE/DELAY(CLOSE,1)-1`，不用 `d.returns`：股票当天停牌、以及复牌首日（前一天收盘缺失）都不出数）；`BANCHMARKINDEXCLOSE` 按 `d.benchmark_close` |
| R-G-01 | 180 | 按 `cond ? ((-1 * TSRANK(ABS(DELTA(CLOSE, 7)), 60)) * SIGN(DELTA(CLOSE, 7))) : (-1 * VOLUME)` 翻：`d.where(cond, (-1 * d.ts_rank(d.abs(d.delta(d.close, 7)), 60)) * d.sign(d.delta(d.close, 7)), (-1 * d.volume))` |
| R-G-03 | 035、087、190、078、159、173、146 | 原文看着奇怪（同一字段加权两次、优先级不对称、分母和权重和异样、三重 SMA、`r-(r-SMA)`），一律照字面翻，不改不化简 |
| 不收 | 030、143、165、166、183 | 不写函数（写了注册表会报错），不用进 `NOTES` |

## 7. 不许做的事，以及遇到问题怎么办

不许：

- 化简或改写公式结构：不合并同类项、不约分、不把 `x * w + x * (1 - w)` 写成 x、不把 `-1 * x` 写成 `-x`、不把 `a / b / c` 改成 `a / (b * c)`。恒等写法也照写：`delta(close - 1, k)` 里的 `- 1`、`sum(x, 1)`、`product(x, 1)`、`x ^ 1`（`d.power(x, 1)`）、只对分子做的中性化，一律保留。
- 给分母加保护（`+ 1e-12`、`where(分母 == 0, …)`）：分母为 0 的结果就是缺失，按约定向后传播。
- 调换运算顺序或参数顺序（`d.ts_sumif` 的参数顺序是规定好的例外）。
- 自己算常数、改参数、改窗口（非整数天数原样写）。
- 凭对因子含义的理解「修正」公式（例如觉得符号反了、窗口不合理）。
- 用 `d.` 以外的任何东西：不 import 别的模块，不写 Python 的 `==` `<` `>` `^` `**` `if` `and` `or` `abs()` `max()` `min()` `sum()`。
- 用聚宽版本（`formula_alt`）代替研报原文，除非第 6 节这么写了。

遇到本文没覆盖的写法（未定义的名字、参数个数不对、括号对不上、优先级不确定、`SEQUENCE` 出现在别处等）：**不写这个函数**，在 `NOTES` 里写 `"gtja191_xxx": "标记：<哪里不清楚，原文片段>"`，交给负责人。宁可多标记，不要猜。
