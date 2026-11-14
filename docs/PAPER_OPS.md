# 模拟盘每日运行

2026-11-14。把模拟盘运行器（`python -m ashare.paper start`，DESIGN 8.6）做成每个交易日自动运行的
流水线，本机和服务器用同一套命令，调度器只负责按时启动。代码在 `src/ashare/paper/ops/`，调度模板在
`deploy/paper/`。只做纸面记账，不连接任何券商。

## 一天的时间线（北京时间）

| 时间 | 任务 | 做什么 | 依据 |
|---|---|---|---|
| 08:55（周一至周五） | `daily run` | 开盘前启动运行器，跑到 17:00 退出；非交易日立即退出 0 | 运行器 9:00 进入盘前取涨跌停价（Tushare `stk_limit`「每个交易日 9 点左右更新」），9:15 起参加开盘集合竞价；录制任务 9:05 启动 |
| 15:00–17:00 | （运行器内） | 收盘后每分钟试一次日线，取到就结算、出日报 | 日线「交易日每天 15 点～16 点之间入库」；17:00 前没取到时，次日开盘前运行器自动补结算 |
| 18:30 | `daily nightly` | 盘后阶段 + 信号阶段 | 每日指标「交易日每日 15 点～17 点之间」；指数日线「每日 18 点左右」 |
| 20:30 | `daily nightly` | 18:30 数据没齐时重试；已完成则立即退出 | 同上，给晚发布的表留余量 |
| 07:30（次日） | `daily nightly` | 最后一次补：前一晚两次都没成时在开盘前补齐；仍没成就记 ERROR 并留「今天没有目标」标记 | 复权因子「盘前 9 点 15～20 分」才入当日值，但前一交易日的早已在库 |

时间来自 Tushare 各接口文档的「更新时间」（2026-11-14 查阅：日线 doc 27、复权因子 doc 28、每日指标
doc 32、涨跌停价 doc 183、停复牌 doc 214「不定期」、ST doc 397「每天上午 9:20」、指数日线 doc 95）。
数据层代码本身不记录发布时间；以后某张表发布变晚，只要调盘后任务的钟点。

## 三个阶段

所有阶段：用文件锁防重复启动（第二个实例退出码 3）；盘后和信号共用一把锁（`locks/nightly.lock`），信号读的
面板不会正被盘后阶段重建；运行阶段单独一把。每次结束把结果写进
`<账户根>/.ops/state/<阶段>.json`；日志写 `<账户根>/.ops/logs/<日期>.log`（同时输出到标准错误）。

退出码：`0` 完成或今天无事可做，`1` 出错（不在半成品状态下进入下一步），`2` 参数错误，`3` 已有实例在跑，
`4` 数据还没发布或上游阶段没完成（稍后重跑）。

**没有启用的账户时**（例如还没开账户、全部停用），`signal` 和 `run` 记一行日志后直接退出 0，不读日历、
不连行情；`after-close` 照常每天同步数据。

### `daily after-close`

1. 同步交易日历（含未来 40 天所在年份，年底也知道下一年的开市日），定出 E = 已经收盘的最近一个交易日；
2. E 已处理过、面板版本没变：立即退出 0（幂等；`--force` 重做）；
3. 增量同步：与 `python -m ashare.data sync` 同一个函数、同样的默认窗口（61 天）和刷新规则，终点是 E；
4. E 的日线、复权因子、每日指标、涨跌停价、ST 五张表都必须完整入库，否则退出 4。数据层会把空表或
   行数骤降的分区记为可疑，下次自动重抓；
5. 增量重建面板（`build-panel --incremental`），面板最后一天必须是 E；
6. `verify`：与上次通过时比，硬性检查（清单、分区覆盖、主键、开盘可买可卖掩码、公司行为对账）出现下面任一
   情况就退出 1：不一致数变多；基线里有、这次没有检查（`verify` 在前提不满足时不写这一项）；基线里没有、
   第一次出现就带不一致。其余检查变多或新出现带不一致只记警告。第一次运行以当前计数为基线；人工核对后用
   `--accept-verify` 接受新计数。通过后写 `.ops/state/after-close-done.json`（交易日、面板版本）。

### `daily signal`

前提是 after-close 已对 E 通过且面板版本一致，否则退出 4。对每个启用的账户：

- 信号日 = 账户第一次出信号（建仓），或 E 是调仓周期的最后一个交易日（下一个开市日落在另一个周期；
  周期划分用回测页的 `period_keys`，与回测的 `signal_days` 逐日对照有测试），或上次检查以来漏过了某个
  周期末（机器停了几天：用 E 的数据补一次调仓，理由里写明漏的是哪天）；
- 信号日写 `<账户>/ops/targets/<成交日>.json`，成交日 = 交易日历里 E 的下一个开市日；
- 非信号日不写目标：运行器那天只盯市和结算，持股数不动，与回测「非调仓日保持股数」一致；
- 同一成交日的目标已用同一面板版本算过就跳过；面板被刷新过且成交日还没开始运行就重算；
- 成交日已经开始运行（`ops/runs/<成交日>.json` 存在）就不给这一天写目标：运行器开跑时读一次目标，之后
  写的它看不到。这次调仓记为待执行（`plan.json` 的 `pending_execution_day`），按下一条补发；
- 上一次调仓没有执行——待执行的成交日早于本次成交日，而那天没有运行记录、运行时没带目标、或运行出错——
  本次出信号时补发，理由里写「补发：<日> 的调仓没有执行」（机器那天关着、隔夜信号没出成等）；
- 选股结果不对劲不写目标：0 只一律不写（空目标等于让运行器次日清仓）；股票池内最新一行的有效信号数
  少于上一行的 80%（`COVERAGE_FLOOR`：A 股每天上市、退市、停复牌带来的变动通常在 1%～3%，少两成几乎只会
  是某个字段当天整列缺失），或入选只数少于 min(top_n, 上一行有效数) 的一半（`FILL_FLOOR`），这个账户
  退出 4，等下一次盘后重试；最近两天都选不出股票算策略本身的问题，退出 1；
- 一个账户出错不影响其他账户，阶段整体退出 1；只有「未就绪」的账户时退出 4。

`daily nightly` = after-close，成功后接着 signal（调度用这一条）。开盘前（9:00 以前）跑的那次如果仍没
成功、今天开市、又有启用的账户，日志记 ERROR（`no_targets_today`）并写 `.ops/state/no-targets.json`；
`daily status` 的 `alerts` 里会出现「今天没有目标：…」。之后任何一次 nightly 成功就删掉这个标记。
运行器照常启动，没有新目标的账户那天只盯市和结算。

### `daily run`

交易日（按原始库的交易日历，含节假日）开盘前启动，`--until`（默认 17:00）后自动退出：

- 每个启用的账户一个 `LiveRunner`（与 `start` 同一个类、同样的账户恢复方式）；当天有目标文件就带上，没有
  就只盯市、结算；新账户在拿到第一份目标之前不启动；当天已结算的账户不再启动；
- 所有账户共用一个 Tushare 出口（`TushareClient`：本机独占锁，与 `start` 同一个锁文件；rt_k 令牌桶
  所有账户合计每分钟 15 次，见「已知限制」），账户按关注列表轮询，默认 6 秒一轮，按账户轮流推进；
- 截止按墙上时间判断：机器睡眠醒来已过 `--until`（或已是另一天）就立即收尾，不会拿当天的目标在
  下一个交易日交易；只用 `execution_day` 等于今天的目标文件；
- 运行器每一步先写日志再推进，同一天重跑从日志恢复，不会重复下单；
- 每个账户写 `ops/runs/<日期>.json`（开始、结束、用的目标文件、是否结算），结束时更新 `ops/plan.json`；
- `--dry-run` 把已有账本复制到临时目录再恢复（持仓、现金与正式运行一致），在副本上推进，结果里给出每个
  账户的持仓（`holdings`）和当天的持仓变化（`changes`，股数）；账户目录一个字节都不写，也不写状态文件。

目标文件的成交时点：回测「开盘成交」对应 9:15 起参加开盘集合竞价（运行器用昨收参考价下保护价单），
没成交的部分在连续竞价里继续对齐到 14:56:50；「收盘成交」对应 14:50 起在连续竞价末段对齐（运行器不在
收盘集合竞价下单）。

## 命令

```sh
# 账户（只写策略定义；账本由第一个运行日的运行器建立）
python -m ashare.paper account create <名称> --expression '<表达式>' --settings '{"universe":"csi300","rebalance":"month","top_n":30}' --capital 1000
python -m ashare.paper account create <名称> --backtest <回测 id> [--capital 万元] [--title 展示名]
python -m ashare.paper account create <名称> --backtest-file <回测记录.json>
python -m ashare.paper account list | show <名称> | enable <名称> | disable <名称>

# 每日流水线（调度器调用的就是这几条）
python -m ashare.paper daily nightly          # = after-close + signal
python -m ashare.paper daily after-close [--force] [--accept-verify]
python -m ashare.paper daily signal [--account <名称>]
python -m ashare.paper daily run [--until 17:00] [--account <名称>] [--dry-run]
python -m ashare.paper daily status           # alerts（今天没有目标等）+ 各阶段最近一次的结果

# 调度文件（默认只打印；--apply 只写文件，launchd 写到暂存目录，都不加载任务）
python -m ashare.paper install-schedule --launchd
python -m ashare.paper install-schedule --systemd --python <解释器> --env-file <放密钥的文件>
```

模拟盘页左栏「从回测新建账户」与 `account create --backtest` 等价（`POST /api/paper/accounts`）。

## 账户目录

```
<账户根>/<名称>/
  strategy.json          策略定义（下面）
  ops/plan.json          上次信号日、上次成交日、下次信号日、待执行的成交日
  ops/targets/<日>.json  成交日的目标（运行器 --targets 的格式，另记信号日、面板版本、理由）
  ops/signals.jsonl      每次出信号一行
  ops/runs/<日>.json     每个运行日的开始、结束、结算结果
  manifest.json …        运行器的账本、快照、行情录制、日报（DESIGN 8.5）
<账户根>/.ops/
  state/<阶段>.json       每个阶段最近一次的结果；after-close-done.json；verify-baseline.json
  logs/<日期>.log         文本日志；launchd 的标准输出另在同目录 com.ashare.paper.*.log
  state/no-targets.json   开盘前最后一次 nightly 仍没出成目标时的标记
  locks/nightly.lock      盘后与信号共用的锁；locks/run.lock 运行阶段的锁（里面是持有者的进程号）
  tmp/                    建账户的暂存区：先在这里建好整个目录再原子改名过去，不留半成品
```

账户根默认 `$ASHARE_DATA_DIR/paper`，`ASHARE_PAPER_DIR` 可改；目录名以点开头的 `.ops` 不会被当成账户。

### 策略定义 `strategy.json`（版本 1）

```json
{
  "version": 1,
  "name": "展示名",
  "source": "<平台表达式，只能一个候选>",
  "settings": {"universe": "csi300", "rebalance": "month", "top_n": 30, "trade_at": "open",
               "capital": 1000, "commission": 0.00025, "commission_min": 5.0, "slippage_bps": 5.0,
               "dividend_tax": 0.2, "carry_blocked": true, "lots": "board"},
  "initial_cash": 10000000.0,
  "enabled": true,
  "created_at": "2026-11-14T10:00:00+08:00",
  "origin": {"kind": "expression"}
}
```

`settings` 就是回测页的设置，经回测页的 `settings.normalize` 校验（同一个白名单，不收窗口和起止日期）。
模拟盘实际用到 `universe`、`rebalance`、`top_n`、`trade_at`、`capital`；费用、整手、红利税由模拟盘内核
自己的规则决定（DESIGN 8.3/8.4，P3 接入 `tradecore` 后与回测统一），这里原样记录便于对账。
`origin` 从回测建的是 `{"kind": "backtest", "id": "<回测 id>", "data_version": "<回测时的数据版本>"}`。

### 选股口径

与回测页逐位同一套函数：表达式用 `ashare.factor.Evaluator` 在整张最新面板上求值（与回测页的因果算子
相同，只是不截在验证期末——模拟盘本来就是向前跑的），股票池用回测页的 `universe_mask`，
`ashare.portfolio.top_equal` 取前 N 只等权，再经 `to_targets` 规整（回测页 `build_targets` 的同一条
路径；在面板中间的日子上与 `build_targets` 逐位对照有测试）。

运行器按十进制（`Decimal(str(w))`）检查权重和不超过 1；1/N 的最短十进制表示偶尔合计多出末位（例如
N = 11），这时把最大的权重往下挪一个浮点末位（约 1e-17），挪动次数记在目标文件的 `nudged`。

### 回测记录约定

从回测建账户只依赖下面这条 JSON（「回测记录」），不依赖回测页的代码：

```json
{"id": "<24 位十六进制>", "name": "…", "source": "<表达式>",
 "settings": {<回测页设置>}, "status": "complete", "data_version": "…"}
```

`account create --backtest <id>` 和页面接口默认只读打开回测页结果库 `<平台目录>/backtests/index.sqlite3`
的 `backtests` 表取这条记录（不启动回测工作进程）；也可以把它存成文件交给 `--backtest-file`。

## 与回测的差别

- 回测默认 `carry_blocked=True`：成交日因停牌、涨跌停没做完的部分，之后每天重试到下一个调仓日。模拟盘
  只在成交日当天对齐目标（9:15–14:56:50 每 30 秒一轮），之后几天不补；没做完的差额留到下一次调仓。
- 回测按开盘价或收盘价一次成交；模拟盘按实时快照撮合（DESIGN 8.3、8.8）。
- 漏过的周期末、没执行的调仓用补跑当天的数据调仓，回测里没有这种情况。
- 指数成分：股票池用面板里当时的成分数据。指数公司事后修订成分（或 Tushare 补录）时，面板重建后历史
  上的股票池会变，回测按修订后的成分算，而模拟盘当天用的是修订前的；两者在这种日子上会有出入，模拟盘
  不回头改已经下过的单。

## 已知限制

- **rt_k 配额**：Tushare 官方 rt_k 每分钟 50 次、按账号计，不分进程。录制任务（`com.ashare.rt-recorder`）
  全市场 6 秒一轮、每轮 3 次，约 30 次/分钟。模拟盘所有账户共用一个令牌桶，合计每分钟 15 次（`run.py` 的
  `RT_K_PER_MINUTE`），两者合计 45 次，给录制任务的重试留 5 次余量。一个账户 6 秒一轮正好 10 次/分钟；
  两个账户需要 20 次，被令牌桶压到 15 次，每个账户约 8 秒一轮；账户越多每个账户越慢，总量不变。两边
  没有共享配额，录制任务若改快，要同时把这里调低。
- 运行器的出口锁是本机独占的，同一时间只能有一个 `run`（或手工 `start`）在取数。
- 关注列表只增不减（账本里登记过的证券都会轮询）；持仓换手多年后列表会变长。
- 停电或睡眠：launchd 模板用 `caffeinate -i` 防止空闲睡眠，但合上笔记本盖子、断电仍会睡；醒来时还在
  当天的运行时段内就从日志继续，已过截止就收尾退出；错过的行情不补。

## 本机安装（审查后由人执行）

```sh
python -m ashare.paper install-schedule --launchd            # 先看渲染结果
python -m ashare.paper install-schedule --launchd --apply    # 写到暂存目录 ~/.local/share/ashare/paper-schedule/launchd
cp ~/.local/share/ashare/paper-schedule/launchd/com.ashare.paper.*.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.ashare.paper.nightly.plist
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.ashare.paper.run.plist
```

`--apply` 默认不写 `~/Library/LaunchAgents`：放进那里的 plist 下次登录时会被系统自动加载，等于装上了。
拷贝和加载由人执行（命令会一并打印）；`--target ~/Library/LaunchAgents` 也能写，但会提示这一点。

launchd 按本机时区触发；本机不是北京时间时安装命令会提示。模板用 `/bin/zsh -c` 启动（与录制任务相同），
`TUSHARE_TOKEN` 从 zsh 的环境文件里来，不写进 plist。卸载：`launchctl bootout gui/$(id -u)/com.ashare.paper.run`
（nightly 同理）。

## 迁到服务器

同一个 Tushare 账号只在一台机器上取数（项目约定），所以要先停本机、再开服务器：

1. 本机停任务：`launchctl bootout gui/$(id -u)/com.ashare.paper.run`、`…/com.ashare.paper.nightly`，
   以及录制任务 `…/com.ashare.rt-recorder`。确认没有残留进程：`pgrep -fl 'ashare.paper|rt_recorder'` 为空。
2. 选一个收盘后、运行器已退出（17:00 以后）且 nightly 已完成的时间点，确认 `daily status` 里 run 和
   signal 都是 ok。
3. 拷数据：`data/cn`（原始库和面板）、账户根（含 `.ops`）、平台目录（要从回测建账户时），以及
   `data/live`（录制，可选）。用 `rsync -a`，拷完对比文件数和大小。
4. 服务器上：克隆仓库、`make dev`，确认 `python -c "import ashare; print(ashare.__file__)"` 指向克隆；
   密钥放 `<家目录>/.config/ashare/paper.env`（只有一行 `TUSHARE_TOKEN=…`，权限 600），不进仓库。
5. 先不装定时器，手工跑一遍：`daily status`、`daily nightly`（应为「当天已完成」或正常完成）、
   `daily run --dry-run --until <几分钟后>`（交易时段内）。
6. 装定时器：`python -m ashare.paper install-schedule --systemd --python <venv 解释器> --env-file <上面的文件> --apply`，
   然后 `systemctl --user daemon-reload`、`systemctl --user enable --now ashare-paper-nightly.timer ashare-paper-run.timer`、
   `loginctl enable-linger "$USER"`（用户不登录时定时器也运行）。计时器的 `OnCalendar` 写明 `Asia/Shanghai`，
   与服务器时区无关；`Persistent=true` 让关机错过的触发在开机后补一次（阶段本身幂等，已过 17:00 的运行
   阶段会立即退出）。开机补跑的运行阶段最早可能从零点跑到 17:00，所以运行服务的 `TimeoutStartSec` 是 18
   小时。写进 `~/.config/systemd/user` 的单元在 `enable` 之前不会生效。
7. 录制任务若也要迁，同样在服务器上另装，并确认本机的那份已停。
8. 第一个交易日看 `journalctl --user -u ashare-paper-run`、`<账户根>/.ops/logs/<日期>.log` 和
   `daily status`。本机的数据副本从此不再更新，不要再在本机跑任何取数命令。
