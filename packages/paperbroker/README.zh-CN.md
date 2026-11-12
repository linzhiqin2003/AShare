# paperbroker

确定性的快照模拟交易引擎。Rust 内核不读时间、不联网、不操作文件；Python 运行依赖只有 numpy。许可证待定。

完整接口、数组列定义、费用口径和恢复格式见 [README.md](README.md)。主要入口：

```python
from paperbroker import Broker, MarketConfig
broker = Broker(["EXAMPLE"], initial_cash=100_000, config=MarketConfig.ashare())
broker.session(0, 20261111, "continuous")
# broker.feed(quotes)  # C 连续 int64[N,18] 数组，首次快照只建立累计量基线
# broker.order(time, "EXAMPLE", "buy", 100, price=10.5)
```

- 价格、金额均为万分之一元的整数，时间为毫秒。买一/卖一价格 -1 表示缺失，0 表示已知无报价。
- 同一快照的参与率额度由所有订单共享；每个方向还共享该方向盘口量。集合竞价必须显式提供独立竞价量，不能直接用全天累计成交量。
- 限价单和带保护价的市价单均当日有效。下单后等待下一次更新；滑点后的价格仍须满足保护价与涨跌停范围。
- `cash` 为可用现金，`frozen` 为冻结现金；持仓 `sellable` 不含冻结卖单与 T+1 尚未解锁的买入。`cost` 为总成本，除以股数才是成本价。
- 临时费用按整笔订单累计计算最低佣金，部分成交只补扣差额；临时申报单位为 100 股整数倍，卖单可包含全部零头及任意整手：持有 250 股时可卖 50、150、250 股，不可卖 30、130 股。它们并非完整 A 股规则，后续由共用模型替换。
- `target_weights` 保留方向和数量兼容的挂单，先卖后买，按各证券缺口比例分配可用现金；成交后重复调用使目标逐步收敛。保护价默认取盘口外两档，档数可配置。
- 日终要求先进入 closed，随后过期订单、处理显式公司行为、解锁、产出结算事件。除权日盘前也可用 `corporate` 处理隔夜持仓，按日按证券防重复。暂不模拟分红到账日、送转上市日及零碎股补偿。
- `Broker(..., journal=目录)` 启用先落盘后处理的日志。行情在独立不可变文件，日志只存引用和校验和。独立包默认使用 压缩的 `.npz`；通过 `QuoteStore` 可替换为 Parquet。`checkpoint()` 原子存快照，构造时自动修复不完整尾行并恢复。
- 状态机可跨平台；持久化实现目前依赖 POSIX 文件锁和目录 fsync，Windows 持久化尚未验证。
- `paperbroker.replay` 合并录制行情与订单脚本。`reference.py` 是不调用 Rust 的直白参考实现；包内测试全部为合成数据。

从源码安装用 `pip install .`，或 `maturin develop --release`。maturin 源码分发包包含纯 Rust 内核，不依赖 ashare、panelbt 或 panelops。

当前状态、WAL 和快照协议为 version 3（行情列结构仍为 version 2）：生产行情用 `feed(quotes, received_at=接收毫秒)`，行情时间不代替本机事件时间；行情日期必须匹配交易日。下单前后跨越的成交量区间不用于该订单，先建立下单后的基线。集合竞价同样检查封板锁定。收盘后日线可修订最终估值，绝不撮合。

`Instrument` 为每只证券指定报价单位、T+1、费用及最小申报量/步长（科创板 200/1、北交所 100/1、ETF tick=10 即 0.001）。trait 同时接收证券 ID 和档案；模型标识随快照保存。`append_symbols` 将新证券追加到日志，已有 ID 永不重排。`order` 返回含 request_id、accepted、order_id、events 的结果。

macOS 文件落盘使用 `F_FULLFSYNC`，不支持时明确失败。内存和快照只保留当日输出与日报摘要，往日日报单独归档。每个输出带单调 seq，events(after=seq) 只返回该序号之后的事件，历史按需读取；恢复后的默认 events() 仍是当日至少一次投递。消费者持久化已确认的 seq，不能由此假定外部副作用恰好一次。`read_only=True` 不截断日志、不写文件。旧版日志明确拒绝，暂不自动迁移。

默认只留最近 3 个已校验快照、最近 20 个有行情的交易日录制。snapshot_retention 与 quote_retention_days 可配置，后者为 0 时不清行情。checkpoint() 清旧快照，broker.journal.gc() 还清位于所有保留恢复点之前的旧行情；WAL 和日报不删。清行情后重启需要有效的保留快照，从头全量重放需要另存的录制或备份；所有保留快照损坏时明确报错。日报作为审计历史仍会逐日增长。

盘前 corporate 可显式传历史除权日 day，依次补做停机期间公司行为；按证券和除权日防重复，actions_through 持久化进度。仍未实现完整交易所订单类型、禁撤单窗口和午间订单队列，也没有旧账户迁移工具；tradecore 接入属于 P3。

工作区 release 对实际生成绑定代码的包也启用溢出检查。扩展捕获算术陷阱、恢复原状态并抛 OverflowError；日志层先试跑，坏输入不进入日志或行情录制，账户可正常重启。Rust 库名为 paperbroker_native，Python 模块名仍为 paperbroker._core。
