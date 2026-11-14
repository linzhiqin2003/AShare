# 参考实现的输入约定

第二步要为 Alpha101 和国君 191 各写一份**独立的** pandas 参考实现，用来和翻译出来的表达式逐格核对。本文说明参考实现的函数签名、输入数据、怎么取数、怎么接到核对工具上。口径（复权、缺失、排名、窗口）和原文疑点的处理以 `CONVENTIONS.md`（第 4 节是疑点与裁决）为准；参考实现要**照着它自己写**，不读翻译规范，不调用 `ashare.factor`（表达式、`dsl`、`ops`）和 `panelops`，否则就不独立了。Alpha158 的参考实现已经写好（`ashare.factor.alphas.reference.qlib_alpha158`），可以照它的结构。

**路径变更（2026-11-13）**：机制模块从 `ashare.factor.public` 改名为 `ashare.factor.formula`（旧路径已删除）；三套因子的公式、参考实现和文档当时搬进本地私有包；2026-11-14 起随仓库公开，在 `ashare.factor.alphas`（参考实现在 `ashare.factor.alphas.reference`），文档在 `docs/factors/`。

## 1. 函数签名

一个模块（例如 `ashare.factor.alphas.reference.alpha101`），提供：

```python
REFERENCES: dict[str, Callable[[dict[str, pandas.DataFrame | pandas.Series]], pandas.DataFrame]]
# 例：REFERENCES = {"alpha101_001": alpha101_001, ...}

def alpha101_001(data: dict[str, pandas.DataFrame], **params) -> pandas.DataFrame:
    ...
```

- 键是因子名，与翻译的函数名相同（`alpha101_001`、`gtja191_017`）。
- 每个函数接收下面第 2 节的 `data`，返回与 `data["close"]` **同索引、同列**的 DataFrame（行为交易日，列为证券代码），值为 float；缺失用 NaN。±inf 会被当成 NaN。
- `**params` 接收因子集选项：核对时用了 `--option 键=值`，核对工具会把同样的选项（值是字符串）传给每个参考函数；没给选项时不传任何参数。现在只有 Alpha101 的 `argmax`（`"days_ago"` 或 `"position"`，见下）。
- 可选：提供 `compute_many(data, names, **params) -> dict[str, numpy.ndarray]` 一次算一批（共享中间结果），核对工具优先用它。

## 2. `data` 里的键

全部由 `ashare.factor.formula.inputs.reference_frames(source)` 生成，形状都是 T × N（`listed`、`industry` 也是），基准两项是长度 T 的 Series。

| 键 | 内容 | 与 CONVENTIONS 的对应 |
|---|---|---|
| `open` `high` `low` `close` | 后复权价格：面板价格 × `adj_factor`，元 | D-01 |
| `vwap` | 面板 `vwap`（= 成交额 / 成交量，量为 0 时 NaN）× `adj_factor` | 1.1 |
| `volume` | 面板 `volume`（股）/ `adj_factor` | D-02 |
| `amount` | 成交额，元，不复权 | 1.1 |
| `returns` | `close / previous_close − 1`（不复权，`previous_close` 是交易所除权参考价）；**上市首日（上市区间第一行）为 NaN**，那天的前收盘是发行价 | D-03 |
| `cap` | 总市值，元 | D-04 |
| `listed` | 是否上市（bool）；上市前、退市后为 False | 1.2 |
| `industry` | 申万 2021 一级行业编码（float），缺失为 NaN；三个 IndClass 层级都用它 | D-06 |
| `benchmark_open` `benchmark_close` | 基准指数（默认沪深 300）的开盘、收盘点位，按日期对齐，缺失为 NaN | D-07 |

停牌日、上市前、退市后的价格、量、额都是 NaN（不前填）。`adv{d}`、`DTM/DBM/TR/HD/LD` 不在 `data` 里，参考实现按定义自己算（`adv{d}` = `amount` 的 d 日均值，窗口必须满）。

参考实现必须遵守的口径（摘要，细节见 CONVENTIONS 第 2、3、4 节）：

- 滚动窗口必须满：pandas 写 `rolling(d)`（默认 `min_periods=d`）；天数先 `math.floor`。
- `rank`：`df.rank(axis=1, pct=True)`；`ts_rank`：当天值在窗口内的平均秩 / d。
- `HIGHDAY/LOWDAY`：距今天数，今天 0，并列取最近一次。
- Alpha101 `ts_argmax/ts_argmin` 有两种读法，**两种都要实现**，由 `argmax` 参数选：`"days_ago"`（默认）距今天数，今天 0，并列取最近一次；`"position"` 窗口内位置，最早一天为 1，并列取最早一次（D-13）。
- 相关系数在任一侧窗口方差为 0 时为 NaN（pandas 可能给 ±inf 或舍入值，要显式处理）；标准差、协方差分母 n−1。
- 除以 0、log 非正数等非有限结果都是 NaN；比较、逻辑运算遇 NaN 结果为 NaN（pandas 的 `>` 会给 False，要显式处理）。
- `SMA(A, n, m)`：α = m/n、`adjust=False`、缺失日保持状态且当天输出 NaN、累计至少 n 个有效值；`WMA`：权重 0.9^i（今天 i=0）归一。
- 原文疑点按 `CONVENTIONS.md` 第 4 节（含 4.4、4.5 两节补充裁决）的处理实现；不收的 5 条（030、143、165、166、183）不写。
- 国君 149：`FILTER` 掉的日子为 NaN，252 行窗口内有效配对不少于 20 对才出数，否则 NaN；面板开头不足 252 行时用已有的行；股票当天停牌（当天收益为 NaN）时为 NaN（Q-G-149、R-G-02）。

## 3. 怎么取数

```python
from ashare.factor.formula.inputs import load_inputs, reference_frames

source = load_inputs(root="/path/to/AShare/data", days=750, warmup=260, benchmark="000300.SH")
data = reference_frames(source)        # 第 2 节的字典
```

- `load_inputs` 从数据层只读取最近 `days + warmup` 个交易日（默认 1010 行）× 全部证券，返回 `PublicSource`（求值器也用它）；`symbols=200` 只取前 200 只，`symbols=[...]` 取指定代码，用于试跑。`root` 不给时用 `ASHARE_DATA_DIR` 或仓库下的 `data/`。
- 全截面 1010 × 5923 的 `data` 约 9 张 DataFrame、每张 48 MB。
- 递推类（SMA）依赖数据起点：参考与候选在同一段数据上算，所以可比；正式计算从面板第一天开始。

## 4. 接到核对工具

```bash
uv run --no-sync python -m ashare.factor.formula.verify alpha101 \
    --reference ashare.factor.alphas.reference.alpha101 [--option argmax=position] \
    --root /path/to/AShare/data --days 750 --warmup 260 --out alpha101_verify.json
```

工具对 `alpha101` 因子集里每个已翻译的因子，在同一段数据上求候选和参考，只比较最后 `days` 行，按 CONVENTIONS 第 6 节给出「一致 / 仅浮点差 / 不一致」，结果存 JSON。含截面运算的因子集按整截面计算（不按证券分块）。`--factors alpha101_001,alpha101_002` 只核对指定的几条。

公开仓库的 `test/factor/formula/test_formula_verify.py` 里有一个最小的例子（`REFERENCES` 字典 + 截面排名，及一个故意用错口径必须报「不一致」的版本）；内置示例集 `example` 的参考实现 `ashare.factor.formula.example_reference` 也可以照着写。
