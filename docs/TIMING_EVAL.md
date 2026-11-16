# 单标的择时评估

`ashare.factor.timing` 实现 DESIGN 4.19。数据、因子代码和研究规则都带身份，台账追加写入，验证期每候选一次、确认期整份名单一次。正式研究文件存私有目录；本次演示仅使用本 worktree 的 `tmp/`，未运行真实验证或确认。

## 研究设定

JSON 包含规范台账位置 `ledger`、`symbol`、UTC `trade_hour`、主目标 `horizon`（天）、三段 `stages`（`mining` / `validation` / `confirmation` 的 `[开始,结束)`）、`validation_hash` 和 `confirmation_hash`。日期必须为 UTC 整点、有序、不重叠。另可冻结 `repetitions`（默认 4000）、`seed`（默认 6182026）、回测 `initial_cash`（默认 10000）、风险规则缺档时的 `assume`（默认严格拒绝）。

`ledger` 纳入完整设定哈希。`Settings.load` 将相对位置按设定文件所在目录解析，`Settings.from_dict` 要求绝对路径；两者都规范化真实路径。`Study` 默认使用该位置，显式传入不同的 `ledger` 会以 `LEDGER_PATH` 拒绝；改设定里的位置会产生新设定哈希。

日期字符串和毫秒整数规范化为同一内容哈希；每行台账保存完整设定哈希、日期签名、评估设定签名和数据版本。已有验证或确认记录后，修改任意阶段日期都会被拒绝。设定内容冻结，`study.settings` 引用也不可重新赋值。补登记名单哈希改变完整设定身份，须重新取得该设定下的挖掘缓存，尝试数不因此增加。改变持有期、标的、调仓钟点、标签口径或任一阶段日期会形成新的检验身份，各计一次尝试。

两个名单文件都是**检验 SHA-256** 的 JSON 数组，由 `study.test_hash(factor)` 或 `settings.test_hash(factor.definition_hash)` 取得。公式哈希本身不能充当名单项。验证最多 10 项、确认最多 5 项，不重复。设定里登记的是**文件原始字节的 SHA-256**，空白改动也改变名单身份。验证候选需要同评估设定、同数据版本的挖掘记录；确认候选还必须有对应验证完成记录。

## 因子接口与数据防火墙

```python
import numpy as np
from ashare.factor.timing import Factor, Settings, Study
from ashare.data import crypto

# 普通、可读取源码的函数；数据来自 view，数值参数单独登记。
def momentum(view, hours):
    frame = view.hourly()
    close = frame["close"].to_numpy()
    result = np.full(len(close), np.nan)
    result[hours:] = close[hours:] / close[:-hours] - 1
    return frame["available_at"].to_numpy(), result

factor = Factor(momentum, params={"hours": 168}, family="momentum")
study = Study(
    Settings.load("<私有研究设定.json>"),
    crypto.open("<小时数据集>"),
    # ledger 默认取研究设定；显式传入时必须指向相同文件。
    minute_root="<分钟数据集>",
    metrics_root="<metrics 数据集>",
)
result = study.evaluate(factor)  # 仅挖掘
summary = study.summary()       # 不同检验的尝试数和 BH q 值
```

`view.hourly()` 只返回该标的永续小时面板；`view.minutes(start, end=None, kind='klines')` 和 `view.metrics(start, end=None)` 使用冻结版本读取。全部观测的 `available_at`（区间结束）必须**严格早于** view 的截止时刻。指定超出截止时刻的请求报错；历史预热允许早于阶段开始。小时 view 添加 `available_at = time + 1h`，不交出原始 dataset、路径或未截断的输入快照。

因子输出 `(available_at, 一维 float64 数值)`，时间须唯一递增 UTC 整点、不得越过 view。日频只取 `trade_hour`；缺信号日保留 NaN，不压缩日历。阶段标签独立读取退出时刻的开盘价；该读取器不提供给因子。

`evaluate` 和 `confirm` 在查缓存、读名单之前只接受精确类型为 `Factor` 的对象；普通函数、伪造哈希的 callable、覆盖执行方法的子类均以 `FACTOR_TYPE` 拒绝。因子 AST 去掉注释、文档、函数名、局部变量名差异，编译后执行的就是这份规范代码。公式 `definition_hash` 从实际执行的字节码、常量、名称/参数结构和冻结参数计算，不信任调用方挂载的身份。执行代码、参数、family 均不可重新赋值，参数通过副本读取。解释器版本固定在研究 venv 中；字节码格式变更可能改变公式身份。不同公式的数学等价性不自动求证。函数不允许闭包数据、外部全局变量、任意导入或文件 I/O；允许 `np`、`pl`、`math` 和基本内建函数，在新名字空间执行。NumPy 内部的懒加载可使用导入机制，但因子源码不能直接导入。此限制用于防研究误用，**不是运行恶意 Python 的系统安全沙箱**。

每次新评估先做截断自检：由定义哈希确定若干历史截断点，加上前、中、后固定锚点；完整 view 和截断 view 的重叠日期、缺失位置、数值字节必须一致。下一日收盘、全样本均值标准化等会被拒绝。有限截断点无法数学证明任意程序从不前视，因子仍需遵守接口合同。

公开的 `study.view()` 只开放挖掘期。验证和确认 view 仅在名单预约成功后由评估器传给因子，避免先取封存数据再补记一次运行。

## 标签、推断和台账

复用 C6 的永续开盘价与资金费现金标签。描述收益与可加对数推断收益分别报告。样本只收入场 `>= start` 且退出 `<= end` 的完整持有期；跨界不顺延，阶段末端的退出开盘价仅供标签使用。C8 只读取永续标签所需的列，C6 接口中的现货列设为缺失，不要求另有现货数据。

推断使用正向 Hodrick 1B 和 Berger–Boos 保守模拟 p 值。δ 的 Fisher 区间、收益创新池仅从挖掘期校准，阶段前预热不参与校准。验证/确认不更新校准参数。确认使用 C6 更保守的完整加密网格；样本不足 100 或退化时结果不可估计，仍保留 Holm 名额。

公式身份 `definition_hash` 仅表示规范执行代码与参数。检验身份 `test_hash` 再加入标的、UTC 调仓时点、持有期、标签版本 `c6-usdm-open-additive-log-return-with-funding-v1`，以及挖掘、验证、确认三段完整的规范日期。标签公式改变必须更新此版本。尝试数、BH、验证名单和验证一次性闸门均使用检验身份；同一公式的 1/2/7 日检验计为三次；同一公式与持有期改变挖掘起点三次，同样计三次，BH 使用全部三项。

挖掘期同检验、同数据版本、同完整设定直接返回通过校验的缓存，不追加记录，不是新尝试也不是本次复算（返回结果中的标志仍描述该缓存首次产生时的计算）。只有同检验最近一次有完整身份的评估记录（started/failed，或校验通过的 completed）与当前数据版本不同时，started 和新结果才标记 `recomputation=true`，尝试数不增加。同数据版本的失败重试、修复坏缓存或修改模拟 seed 均不称为数据复算；不能因为更早历史中存在不同版本就一直标记复算；该检验最新结果参与 BH，失败检验仍保留名额。旧版仅含公式身份的台账以 `LEDGER_IDENTITY` 拒绝，不能静默忽略历史记录或重置闸门；本次临时演示使用新的设定和台账。BH 直接调用 `ashare.factor.stats.adjust(..., 'bh')`，NaN 不缩小分母。BH 的独立或正相关假设以及 C6 的工作模型、模拟上确界网格近似仍适用。

缓存命中前必须校验完整的外层事件、统计量、归一化参数及类型。外层和 `result` 内的 `definition_hash`、`test_hash`、`data_version`、`settings_hash` 必须逐项一致并等于当前请求；内层 `data_versions` 还必须重算出相同的数据版本摘要。日期签名和评估设定签名也必须吻合。候选缓存不一致时按未命中重新计算，追加 `cache_warning / CACHE_INCONSISTENT` 告警，不返回更老的缓存替代它；坏完成项不进入 BH，也不能制造额外检验名额。同检验的旧数据或旧完整设定结果也不能直接复用，遇到时按同一规则留下缓存身份不符告警。旧版不含检验身份的台账和受损闸门仍失败关闭。

**信任边界**：台账是研究者控制的私有文件。以上校验防止结构损坏、身份错配与误用；不做加密认证，不抵抗主动修改全部相互一致字段的文件篡改。不能把同一可编辑文件里的哈希当作可信签名。

`study.evaluate(factor, stage='validation', list_path=...)` 检验验证名单。`study.confirm(factors, list_path=...)` 一次运行最终名单、固定名额 Holm 和组合回测。读记录、核对、预约由同一文件锁保护；**预约后异常也消耗名额**，不能借失败重试查看更多结果。损坏或截断的台账拒绝继续。

## 确认组合

各因子按挖掘期日频均值、样本标准差标准化，缺失值按零分，固定名单等权平均，仓位截在 `[-1, 1]`，杠杆为 1。标准差退化则拒绝回测。通过 `ashare.backtest.crypto` 的 `daily_targets` 唯一一次移位，交由 `panelbt` 处理成交、费用、资金费和风险档案。

同一确认区间和成交钟点同时回测组合、永续持有，以及 C3 的 20/60/120 日动量投票、30 日波动率、20% 年化波动预算趋势规则。为比较一倍账户，趋势总名义上限也固定为 1（C3 原例为 2）。不在确认期调参数。

## 本次演示

`python -m bench.timing_demo --root '<小时数据集>' --minute-root '<分钟数据集>' --folder tmp/c8-recheck-timing-demo` 创建 `tmp/c8-recheck-timing-demo/` 中的临时设定和台账。挖掘期为 2020-01-01 至 2025-05-01（含 2025-04-30），运行 7 日动量、30 日动量、7 日资金费均值。脚本没有验证/确认开关，公开 JSON 只输出统计量。这三个因子无需 metrics，原演示未传入 metrics 根。使用已通过 build/verify 的登记常量版本时，可显式添加 `--metrics-root`；本次 metrics 规则修订没有重新调用真实验证期或确认期。
