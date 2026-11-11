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
- 研究框架和回测引擎由 Codex 按 `docs/DESIGN.md` 第 7 节分步做，模拟盘由 Claude 子代理在 `.worktrees/` 里并行做；Claude 审查通过后提交；不推送远程
- 引擎以后可能开源并发布到 PyPI，由用户提醒时再做；在那之前不建远程仓库、不上传，许可证待定

## 进度
- 2026-11-12 立项；步骤 1a（脚手架 + panelbt 基础版）完成并合入主干；1b（panelbt 完整版）和模拟盘 P1/P2 并行进行中
