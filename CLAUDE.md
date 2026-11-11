# AShare — 项目记忆

A 股量化研究框架。方案见 `docs/DESIGN.md`，改设计先改它，再改代码。

## 结构
- `crates/` 纯 Rust 内核：`panelbt-core`（组合回测）、`panelops-core`（因子算子）
- `packages/` 对应的 Python 包（maturin）：`panelbt`、`panelops`，都不依赖 `ashare`
- `src/ashare/` 研究项目本体：数据、因子、检验、合成、组合
- `test/` 测试；`docs/` 文档；`research/` 每个研究一个子目录；`data/` 不入库

## 约定
- Python 3.13 + uv，Rust stable；构建 `make dev`，全部检查 `make check`
- Tushare 密钥在环境变量 `TUSHARE_TOKEN`（`~/.config/secrets/env.zsh`），不写进仓库
- 信号与成交的错位只在 `ashare` 里统一处理；`panelbt` 不做移位
- 实现由 Codex 按 `docs/DESIGN.md` 第 7 节分步做，Claude 审查通过后提交；不推送远程

## 进度
- 2026-11-12 立项，方案 0.1；步骤 1（脚手架 + panelbt）进行中
