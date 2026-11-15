# BTC、ETH 分钟线（C7）

`ashare.data.crypto.minute` 实现 DESIGN 6.10.1，是独立的 `crypto_minute` 数据集，不改变小时面板版本，也不接入回测引擎。只有 BTCUSDT、ETHUSDT：现货成交 K 线，U 本位永续成交 K 线、标记价、指数价，共八条序列。

## 使用

时间使用 UTC，区间左闭右开。`sync` 的窗口须为完整日；省略终点时取今天 UTC 00:00，只取已完成的日度归档。所有入口接受显式 `root`；当前默认 `data/crypto_minute`，D1 合并后改走统一数据路径。

```bash
nice -n 15 timeout 840 .venv/bin/python -m ashare.data.crypto.minute sync \
  --root tmp/crypto_minute --start 2025-03-01 --end 2025-04-01
nice -n 15 timeout 840 .venv/bin/python -m ashare.data.crypto.minute build --root tmp/crypto_minute
nice -n 15 timeout 840 .venv/bin/python -m ashare.data.crypto.minute verify --root tmp/crypto_minute
.venv/bin/python -m ashare.data.crypto.minute status --root tmp/crypto_minute
```

`--root` 也可以放在子命令之前。完整历史同步默认从 2020-01-01 开始，默认每次最多下载 2 GB（`--max-bytes` 可调整）；被 `timeout` 停止后原命令续跑。只有显式 `sync --refresh` 才重新下载已核对的归档，用于发现上游修订。

```python
from ashare.data.crypto.minute import bars, aggregate_hourly

frame = bars(
    "BTCUSDT", "usdm", "klines", "2025-03-31", "2025-04-02",
    root="tmp/crypto_minute",
)
hours = aggregate_hourly(frame)
```

`bars` 返回 Polars DataFrame，按需校验和读取涉及的月文件；可以额外传 `version=` 固定旧版本。不存在的月分区报 `KeyError`，已建但不完整的月返回实际存在的行。相同起止返回空表。`market` 为 `spot/usdm`，`kind` 为 `klines/mark/index`，现货只允许 `klines`。

## 数据格式与版本

- 成交线字段：`open_time:Int64`（UTC 毫秒），`open/high/low/close/volume/quote_volume/taker_buy_volume/taker_buy_quote_volume:Float64`，`count:Int64`。
- 标记价、指数价仅含 `open_time` 和 OHLC。原档的 `close_time`、`ignore` 等未选入整理表，原始 ZIP 原样保留。
- `objects/<sha256>`、`manifest.sqlite3` 复用现有 `RawStore`；下载、代理、超时、退避和 CHECKSUM 核对复用 `PublicClient`。每次只处理一个月，不建宽面板。
- `bars/<market>/<symbol>/<kind>/<YYYY-MM>/<sha256>.parquet` 保存不可变分块。每个内容版本每个月只有一个被引用分块；相同分块跨版本复用。
- `versions/<version>.json` 包含归档哈希与 CHECKSUM 原文、Parquet 哈希、完整质量例外、请求日、定义版本和可选 `quality_notes`。规范 JSON 的 SHA-256 就是 `version`；`latest.json` 在构建全部完成后原子发布。
- 读取先验证元数据内容版本，再验证所读月份的原始 ZIP 哈希/CHECKSUM 和 Parquet 哈希、schema。未读取的月由全库 `verify` 审计。读取不打开可写清单，不创建数据文件。

## 同步与缺失

月档优先。只要月档在某日有任意一分钟，该日就不使用日档；只有整日没有记录才取日档。月内部分缺失不修补、不填值、不用 REST。新月档发布后，新版本用月档替代相应旧日档，旧版本仍指向旧内容。

13 位数量级的开盘时间视为毫秒，16 位视为微秒；微秒必须能无损转成毫秒。同一文件混用单位、其他量级、非分钟网格、越界、重复或倒序都报错。表头依据 `open_time` 内容识别，不依据市场或年份。ZIP 只接受一份非空 12 列 CSV，检查 CRC，数值字段拒绝空值、负数和非有限数。

每月 `quality.expected_rows` 为完整日历月分钟数；`quality.rows` 是实际行数；`quality.gaps` 用 `{start,end,minutes}` 逐段记录所有空洞。局部采样、尚未请求、尚未发布以及未来时段也属于该月未覆盖区间，可结合 `requested_days` 区分。缺口只陈述“该版本没有数据”，不推断停牌或交易所故障。

HTTP 404 记入 `missing`，整理时表现为缺口；其他下载异常记入 `failures` 并抛 `SyncError`，命令返回非零，构建拒绝带失败的计划。失败记录只有重试对应月份才会解除，同月的原请求范围也保留。已验证的缓存不重下；网络中断的响应不写入内容对象。公用客户端对超时和网络错误至多尝试四次，退避为 1、2、4 秒；服务端限流遵循其既有限额和等待上限。

## 独立校验与小时对账

`verify_minute.py` 不调用整理器的解析、缺口、哈希和聚合函数。它用标准库 CSV、独立的时间单位判据、整月期望网格差集和逐字段对比，核验原始归档与整理表、重复/顺序、行数、缺口、来源选择和版本。`ok=true` 说明本地数据与原档、质量记录一致，允许已记录的缺口。

```bash
nice -n 15 timeout 840 .venv/bin/python -m ashare.data.crypto.minute verify \
  --root tmp/crypto_minute --hourly-root data/crypto \
  --hourly-version <严格小时版本> --report tmp/crypto_minute/reconciliation.json
```

未指定 `--hourly-version` 时固定读取小时库 `latest.json`，拒绝显式传入推断版本。小时库通过 `ashare.data.crypto.open(..., version=...)` 只读访问。对账结果单列 `hourly_matches`，不以 `ok` 混同两种结论；CLI 的退出码用于本地完整性，自动验收小时一致性还须检查 `hourly_matches`。

小时聚合取第一根开盘、最后一根收盘、高低极值，成交量/额/笔数/主动买入量额求和。不足 60 根的小时保留并标 `complete=false`，重复或非网格输入拒绝。独立对账另写聚合实现：价格精确比较，量用 `math.fsum`；所有非零差异逐条输出。量差不超过 `1e-8 + 1e-12 × abs(小时值)` 分类为二进制浮点舍入，其余为来源差异，不放宽价格比较。

成交线可与小时面板比 OHLC、成交量、成交额、主动买入额；标记价比 OHLC；小时面板只有指数收盘，故指数仅比收盘。小时表未保存成交笔数、主动买入量，这两项只核对原始分钟档和聚合规则。缺完整小时、小时来源未覆盖和所有差异分别列出。逐条报告含真实行情，只能放被忽略的数据目录，不能提交到公开仓库。

## 验收与复现

```bash
nice -n 15 timeout 180 .venv/bin/python -m pytest -q test/crypto/test_minute.py -n 4
nice -n 15 timeout 240 .venv/bin/python bench/mutations_crypto_minute.py
nice -n 15 timeout 840 .venv/bin/python bench/crypto_minute_sample.py --hourly-root data/crypto
nice -n 15 timeout 120 .venv/bin/python bench/crypto_minute_source_audit.py --hourly-root data/crypto
```

小样本脚本只允许向本 worktree 的 `tmp/` 写真实数据。默认下载 2021-05、2025-03 的八条序列及当前月前三日；完整逐小时差异放 `tmp/crypto_minute/reconciliation.json`，公开产物只写汇总。验收数字、官方源差异、性能外推和合并事项见 [C7_NOTES.md](C7_NOTES.md)。
