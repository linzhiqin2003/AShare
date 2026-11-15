# 加密货币资金费与基差时序因子

实现 DESIGN 4.18 v5.1。只用于 BTC/ETH 时序研究，不注册进只接受 Expr 的表达式因子库。定义、参数与确认族预先冻结；没有搜索策略参数。

## 使用

```python
from ashare.data import crypto
from ashare.data.crypto.signals import SignalDataset
from ashare.factor.crypto import crypto_factors, daily_factor
from ashare.backtest.crypto import daily_targets

ds = crypto.open("tmp/crypto")  # latest.json 严格版本
signals = SignalDataset("tmp/crypto")
f = crypto_factors(ds, signals, ["BTCUSDT", "ETHUSDT"],
                   "2024-01-01", "2025-01-01")
at, values = f["funding_mean_7d"]
days, daily_values = daily_factor(f["funding_z_90d"], trade_hour=0)
# f.selected_contract、f.roll 同为 (available_at, T×N)。
# daily_targets 接收 days，不能再手动移位。
```

每个因子是 `(available_at, T×N numpy 数组)`。`names=` 可筛因子；`start/end` 按 available_at 半开裁剪，计算先读取全部覆盖历史预热。`f.identity` 为严格面板版本、季度信号版本、因子定义版本三元组；任何缓存还须包含 symbols、names 和查询区间。缺季度数据可显式传 `signals=None`，三个季度相关因子为 NaN，其余仍可用。

## 因子定义

资金费强度、季度基差和 carry_spread 使用年化口径，年取8760小时；premium 系列是未年化价差比例，z 是标准化分数。它们都不是每张合约的 USDT 现金。

| 名称 | 定义 |
|---|---|
| funding_last | 最近已结算 r × 8760 / 这一笔的间隔小时 |
| funding_mean_3d | (τ−3日,τ] 内结算费率之和 × 8760 / 72 |
| funding_mean_7d | (τ−7日,τ] 内结算费率之和 × 8760 / 168 |
| funding_mean_30d | (τ−30日,τ] 内结算费率之和 × 8760 / 720 |
| funding_z_90d | 当前 7 日强度相对过去90个同钟点日值的 z，ddof=1；至少60个有效参考、排除当天；常数窗口缺失 |
| premium_1d / premium_7d | 同小时永续成交收盘 / 现货收盘 − 1，24 / 168 小时有效均值 |
| mark_premium_1d / mark_premium_7d | 同小时标记收盘 / 指数收盘 − 1，24 / 168 小时有效均值 |
| basis_q | (季度收盘 / 现货收盘 − 1) × 8760 / 从 available_at 到交割的小时数 |
| basis_q_chg_7d | 所选同一合约当前与7天前的年化基差之差；没有历史则 NaN |
| carry_spread | funding_mean_7d − basis_q |

价格两边都要求正常小时：排除缺口持平占位、半截线、时间异常、停机、生命周期外。季度额外排除零成交和连续同收盘（从第二小时起）。现有面板是多种价格合并质量位，因此整行保守排除。窗口至少 2/3 小时有效；资金费均值必须具有完整日历覆盖，不把暂时1小时结算重新稀释成8小时。

季度信号按 v5.1 核对最后成交小时，收盘晚于交割的原始行自动登记 `post_delivery` 并剔除、逐期计数进版本；4期未到期合约的最后成交下界核验标为待到期。季度选约仅看元数据：上市满7日、余期严格大于336小时，取最近交割。选中近月缺价格时保持代码、因子缺失，不回退远月。换期由元数据决定，变化因子仍用同一合约历史。

## 时点

小时 `time=s` 覆盖 `[s,s+1h)`，收盘在 `available_at=s+1h` 可知。资金费在结算边界可知，结算先于同刻成交；这是零延迟模型，未重放交易所边界约数秒的不确定性。

日频因子在 τ=trade_hour 的值被 `daily_factor` 标为 τ−24h 的日线左端，交由 `daily_targets` 唯一一次移至 τ 成交。0/1/8/23 时、跨月跨年均有合成测试。截断检验按原始结算时刻和 K 线可得时刻截断。

## 标签与检验怎么读

`crypto_eval.crypto_labels` 同时返回描述收益、逐日对数收益与 h 天推断和。现货描述收益是出/入场 open−1；永续为固定一币价格盈亏减 `(入场,退出]` 的资金费现金、除以入场价，不含费用和强平。入/出不可成交则缺失、不顺延。

推断标签从面板建立连续日历收益轴，读取到最后一个请求入场日之后的最大持有期，先将单日净收益取 log1p 并聚合，再映射回请求日期。裁剪信号或缺少一个信号日不会删掉该日收益；持有期内任一天收益缺失则整体缺失。覆盖外的入场或退出返回 NaN，面板自身缺少小时占位则报错。这与固定一币的 h 天描述收益不同。Spearman 和过去365天滚动分位表仅作描述，阈值不含当天、至少180个有效日期，边界并列归下组，另列重叠日期对数。

正向 Hodrick 1B 使用零假设下去均值单日收益；S 是逐项求和，不除以 T；过去 h 个因子任一缺失就不形成该项 q。OLS 保留截距，在有效 OLS 样本上中心化、标准化因子并用 QR 求解；q 和协方差采用同一变换，最后还原原单位的 β 与 se。合法的近常数输入不因数值消减被删样本。

模拟从样本内日收益独立重抽，因子为 AR(1)，同期创新相关 δ。φ 扫 [0,.999] 的37点；δ 用样本内99.9% Fisher区间11点；每格4000次公共随机数，内点最大再加密，p=(1+超越次数)/4001，最后加 .001、截至1。报告 φ 与 |δ| 两方向的非单调性。

确认族为 BTC/ETH × funding_mean_7d、funding_z_90d、premium_7d、basis_q × 7日永续推断收益。固定八名额，包括无法计算的 NaN，调用 `stats.adjust(method="holm")`。只有样本外这八项可以使用“显著”。其余因子、期限、现货与样本内均标“探索，不作推断”，没有额外校正。

2024-01-01 切分：样本内入出都严格早于边界，样本外入场不早于边界，跨界不收。样本外因子允许历史预热；δ、创新尺度及收益重抽池不使用样本外。

功效按逐日备择 `g[i]=μ+b*x[i]+ε[i+1]` 生成后聚合，正负 b 分开。零假设分布预先排序，备择查表，以 p≤.00625 的完整保守规则取各格最低功效；目标80%。确认项预先计算所有可能的“一次内点加密”格，观察与功效共用这一更保守的固定全集，避免备择选择不同内点产生嵌套模拟。MDE 主值 `h*b*sd_x` 表示因子持续 h 日增加一个样本内标准差的收益效应；单次初始因子冲击对应 `b*sd_x*Σφ^j`，JSON 同时给出允许 φ 范围的效应区间，不能把两者混读。

MDE 主表明确标为“持续七日增加 1 sd 的等价效应（对数收益）”，列出正负方向首次达到80%的离散搜索点、前一格及各自最低功效。搜索上限为10；它是粗网格阈值，不是精确连续阈值，也不是本日信号冲击的七日预测效应。JSON 的 `search_bracket`、`mde_is_discrete` 和 `search_upper_bound` 保留同样口径，未改变功效算法。

## 可复现命令

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 POLARS_MAX_THREADS=1
nice -n 15 timeout 840 .venv/bin/python bench/crypto_factors_report.py --data tmp/crypto --workers 2
nice -n 15 timeout 300 .venv/bin/python bench/crypto_factor_example.py --data tmp/crypto
nice -n 15 timeout 180 .venv/bin/python bench/crypto_factors_hand_check.py --data tmp/crypto
nice -n 15 timeout 300 .venv/bin/python bench/mutations_crypto_factors.py
nice -n 15 timeout 120 .venv/bin/python bench/crypto_signals_audit.py --data tmp/crypto
```

报告见 `crypto_factors_report.md` 和 `bench/results/crypto_factors_report.json`。策略例子是在已经公开的趋势规则上加冻结的资金费 z 阈值 ±2、对应方向减半，标“回溯探索”；确认须使用2026-11-15之后新数据。实现选择、问题、回归来源和实测性能见 `C6_NOTES.md`。
