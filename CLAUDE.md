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
- 引擎以后可能开源并发布到 PyPI，由用户提醒时再做；在那之前不建远程仓库、不上传，许可证待定

## 进度
- 2026-11-12 立项；步骤 1a（脚手架 + panelbt 基础版）完成并合入主干
- 2026-11-12 Tushare 全历史原始库下载完成（2007-01 至 2026-11-11，46393 个分区，2.6 GB）；实时行情录制任务装好
- 2026-11-12 模拟盘 P1/P2（`paperbroker` + `ashare.paper`）三轮审查后合入主干
- 进行中（各自分支，均已过一轮以上独立审查并在返工）：1b `panelbt` 完整版、数据层（公司行为与复权因子对账）、`panelops` 与因子层（内核改逐行流式）、风险模型与组合优化；研究 001（红利低波）预注册 v0.2 待用户确认后冻结
