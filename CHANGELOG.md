# Changelog

## 2026-11-12
- 立项：选型与方案（`docs/DESIGN.md` 0.1）

- 步骤 1：建立 Python 3.13/uv 与 Cargo 工作区；基于三方摸底选用 Rust panelbt 内核，提供 numpy 输入/规则/稀疏目标、独立参考实现、随机对拍与性质测试、10 项源码变异检查和可复现性能基准；panelops 与 ashare 保持空壳。
- 方案 0.2：市场参数化、做空与保证金、期货对冲腿（4.11）、模拟盘（第 8 节）、数据层四层封装（6.1）、风险模型与组合优化（6.4）、券商接口与 miniQMT 适配（第 9 节，只做系统侧）；整体结构图 `docs/architecture.svg`。
- 数据：Tushare 全历史原始库下载完成（2007-01 至 2026-11-11，46393 个分区，2.6 GB，`data/cn`，不入库）；实测限额按接口计、每接口每分钟 500 次。实时行情原始录制任务 `com.ashare.rt-recorder`（launchd，交易日 9:05，`data/live`）。
- 模拟盘合入主干：`paperbroker`（Rust 内核的订单、撮合、账户、结算状态机，日志与快照恢复）和 `ashare.paper`（行情源、调度、目标调仓、原始录制导入、停机后补做公司行为、保留策略）。三轮独立审查与返工；180 项 Python 测试、7 项 Rust 测试；真实行情夹具放在 `data/fixtures`（不入库）。
- 根 `.gitignore` 的 `data/` 改为 `/data/`（原写法把 `src/ashare/data` 等代码目录一起忽略了）。
