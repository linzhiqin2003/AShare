# AShare — 项目记忆

A 股量化研究框架。方案见 `docs/DESIGN.md`，改设计先改它，再改代码。

## 结构
- `packages/` 三个独立的 Python 原生扩展包：`panelbt`（组合回测）、`paperbroker`（实时模拟盘）、`panelops`（因子算子），都不依赖 `ashare`，注释和 README 用英文（另附中文 README）
- `crates/` 这些包的 Rust 内核，以及回测和模拟盘共用的 `tradecore`；先测 numpy / polars 基线，只要 Rust 实测更快就用 Rust，对比数字留档作基线
- `src/ashare/` 研究项目本体：数据、因子、检验、合成、组合
- `test/` 测试；`docs/` 文档；`research/` 每个研究一个子目录；`data/` 不入库

## 约定
- Python 3.13 + uv，Rust stable；构建 `make dev`，全部检查 `make check`
- Tushare 密钥在环境变量 `TUSHARE_TOKEN`（`~/.config/secrets/env.zsh`），不写进仓库
- 信号与成交的错位只在 `ashare` 里统一处理；`panelbt` 不做移位
- 各条线（回测、模拟盘、数据层、因子算子、风险与组合）都由 Codex 按 `docs/DESIGN.md` 在 `.worktrees/` 里各自的分支上并行做；Claude 做独立审查（副本试验、变异检查），问题退回原会话返工，通过后由 Claude 提交并合入主干；不推送远程
- 数据在 `data/`（不入库）：`data/cn` 是 Tushare 原始库和面板，`data/live` 是实时行情的原始录制（launchd 任务 `com.ashare.rt-recorder`，交易日 9:05 启动）。Tushare 限额按接口计，每接口每分钟 500 次；同一账号只在这一台机器上用
- 根 `.gitignore` 里忽略根目录的东西写成 `/name/`（`data/` 曾把 `src/ashare/data` 等代码目录一起忽略）
- 2026-11-13 起整个仓库开源：GitHub 个人公开仓库 `linzhiqin2003/AShare`，Apache-2.0，只推送 `main`（研究分支不推）。推送前先确认没有密钥、本机绝对路径和真实行情数据进入提交；每次推送仍要用户点头。发布到 PyPI 由用户提醒时再做

## 进度
- 2026-11-12 立项；步骤 1a（脚手架 + panelbt 基础版）完成并合入主干
- 2026-11-12 Tushare 全历史原始库下载完成（2007-01 至 2026-11-11，46393 个分区，2.6 GB）；实时行情录制任务装好
- 2026-11-12 模拟盘 P1/P2（`paperbroker` + `ashare.paper`）三轮审查后合入主干
- 2026-11-12 数据层（步骤 2）三轮审查后合入主干：面板 46 个字段，公司行为以复权因子为准对账（`action_source`），读取入口 `ashare.data.open()`
- 2026-11-12 数据层第四轮合入主干：差异化分红保留公告值（`action_source = 5`，72 格），`verify` 改为独立实现
- 2026-11-12 `panelbt` 完整版（步骤 1b）两轮审查后合入主干；遗留的小问题和测试缺口并入 1c 的任务书
- 2026-11-12 风险模型与组合优化（6.4，`ashare.risk`、`ashare.portfolio`、`portopt`）两轮审查后合入主干；尚未接真实行情
- 2026-11-12 `panelops` 与因子层（`ashare.factor`）两轮审查加一次小返工后合入主干：66+ 个算子、表达式求值与缓存、首批因子与检验；滑动求和一族单线程与 bottleneck 持平或更快，精度由精确有理数参考的测试守住
- 2026-11-12 `panelbt` 步骤 1c 合入主干（两次独立复核、两次返工）：期货对冲腿并入现货主循环、结构独立的参考实现、输入检查挪进 Rust 内核、`validate` 开关；主干 1072 个测试。遗留两条小事见 `docs/STEP1C_NOTES.md` 末尾
- 进行中：研究 001（红利低波）预注册 v0.2 待用户确认后冻结
- 2026-11-13 端到端性能优化一轮合入主干（13 项，三组并行，各过独立复核）：一次完整研究 189.6 → 75.9 s（默认用法），全用新增的批量和并行入口 66.5 s；主干 1395 个测试；正式面板重建为 `dbe96104…`（48 个字段，多了 `can_buy_open`、`can_sell_open`）。总表和逐项记录在 `docs/PERF_LOG.md`、`docs/perf_log.d/`
  - 三条不许退仍然有效：易用性、逻辑准确（逐位一致，做不到的事先声明容差并有测试守住）、输出质量；每项优化单独一个提交并记进 `docs/PERF_LOG.md`
  - 唯一带容差的是特质风险增量算法（P-006，相对从头算 ≤ 1e-12，`specific_incremental=False` 切回从头算）；并行优化序列（`parallel=True`）不逐位，是显式选项
  - 测量工具：`make perf-profile` 跑一遍端到端各阶段，`make perf-compare` 与 `bench/e2e/results/baseline.json`（优化前）逐阶段对比；优化后的一轮冻结在 `bench/e2e/results/round1-after.json`。**用户重启电脑后会让我再跑一轮**，届时跑这两条命令并把对比报给用户
  - 需要再放宽容差的候选见 `docs/perf_log.d/risk-suggestions.md`，做之前先问用户
- 下一步：P3 模拟盘接入 `tradecore`、公开因子库（Alpha158 / Alpha101 / 国泰君安 191）批量复现、步骤 4 rqalpha 对账
