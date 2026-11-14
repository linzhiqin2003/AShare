# 测试体系补全计划

2026-11-14 凌晨记下，作为下一个工作日的目标：先补全本地测试，再把它接进 GitHub Actions。

## 现状

- 本地 `make check`：ruff 检查和格式、Rust fmt / clippy / `cargo test`、pytest（4556 通过、7 跳过，约 8 分钟，单进程串行）。
- 已有：
  - 参考实现对拍：`test/comparison.py`、分组回测 441 组合成逐位对照、真实数据对照工具 `bench/groupbt/check.py`。
  - 属性测试：`hypothesis`，`test/test_properties.py`、`test/panelops/test_operators.py` 等。
  - 变异检查：`bench/mutations*.py`，各条线一份脚本，手动运行，没有统一的分数。
  - 性能：只有零星的相对耗时守卫（如 `test/panelops/test_constant_cost.py`）；`make perf-profile` / `make perf-compare`、`bench/platform/speed.py` 要手动跑，没有阈值。
- 没有：覆盖率统计、并发测试分类、冒烟测试入口、flaky 记录、测试耗时预算。

## 目标：每类测试都有、都能一条命令跑

| 类别 | 要做到的 |
|---|---|
| 单元测试 | 每个公开函数和每条错误分支至少一条；按 branch coverage 补缺口 |
| 集成测试 | 数据层 → 因子 → 回测 → 报告，平台 HTTP → 工作进程 → 落盘，各一条主链路 |
| 端到端测试 | 合成数据目录上从命令行和平台 API 各跑一次完整研究，检查产物 |
| 差分测试 | 每个 Rust 内核都有结构独立的 Python/NumPy 参考实现，随机输入逐位或按声明容差对拍；优化前后结果库逐字节对比（`bench/platform/` 工具） |
| 属性测试 | 不变量：现金守恒、权重和、换手非负、时间平移、列置换不变等，用 hypothesis 生成 |
| 变异测试 | 统一入口 `make mutate-all`，Python 用 mutmut（或 cosmic-ray），Rust 用 cargo-mutants；现有手写变异脚本并入 |
| 并发测试 | 平台多 tab 同时跑、同一 tab 取消重跑、后台写持仓与读取、共享缓存加锁；GIL 释放期间的只读保护 |
| 性能回归测试 | `make perf-check`：合成数据上的固定基准（分组回测内核、表达式求值、平台 4 组 / 8 组），与存档基线比，超过 20% 失败；再加相对比例和复杂度守卫进 pytest |
| 冒烟测试 | `make smoke`：1 分钟内跑完，导入全部包、每个命令行入口 `--help`、平台启动后打一遍主要接口 |
| 回归测试 | 开发中发现的每个错误、每个审查发现都写成用例，注明来源（见下） |
| 其他 | 模糊测试（表达式解析器、数据读取的坏输入）、跨平台构建检查（Linux 依赖锁）、文档里的示例代码可执行 |

## 要看的指标（每次 `make test-report` 输出，并记进 `docs/TESTING_METRICS.md`）

- **Branch coverage**：Python 用 coverage.py（`branch = true`）；Rust 用 cargo-llvm-cov（分支覆盖要 nightly，没有时先报行覆盖）。按模块列出，标出低于目标的文件。
- **Test runtime**：总耗时、最慢 20 个测试、各类别耗时；单元测试并行（pytest-xdist）后的耗时。给慢测试定预算。
- **Mutation score**：被杀死的变异 / 全部有效变异，按模块报；存活的变异逐个看，要么补测试，要么写明是等价变异。
- **Flaky rate**：全量连跑 N 次（随机顺序，pytest-randomly），统计时好时坏的测试；flaky 的一律从根上修，不靠重跑。

## GitHub Actions CI

仓库目前没有任何 workflow，`main` 只靠本地检查。本地各类测试补全后全部接进 CI，四个指标每次都出、有门槛。

**每个 PR 和推送 `main`**（合并前必须通过，规则集里加必需检查）：

| job | 内容 |
|---|---|
| 检查 | ruff 检查和格式、Rust fmt / clippy、`make web-check` |
| 单元 + 属性 + 差分 | pytest（xdist 并行、随机顺序）、`cargo test`；hypothesis 用 CI 档（更多样例、固定种子可复现） |
| 集成 + 端到端 + 并发 | 合成数据目录上的完整研究链路、平台 API 主链路、并发与取消用例 |
| 冒烟 | `make smoke` |
| 性能守卫 | pytest 里的相对比例和复杂度守卫（不比绝对耗时） |
| 覆盖率 | Python branch coverage、Rust 行覆盖；低于基线即失败，变化写进任务摘要 |
| 变异（增量） | 只对本次改动的文件跑 mutmut / cargo-mutants；新存活的变异要么补测试，要么在 PR 里标为等价变异 |
| Linux 安装 | 在 Linux 上从锁文件装一遍，确认 macOS 生成的锁在 Linux 上能用 |

**每晚定时**（不挡 PR，失败开 issue）：

- 全量变异测试，按模块出 mutation score，低于门槛报警；
- flaky 检测：全量测试随机顺序连跑 N 次，统计时好时坏的用例，flaky 一律从根上修；
- 性能基准：`make perf-check` 的固定基准与存档基线比，超过 20% 报警（共享机器有噪声，以本机复测为准）；
- 耗时：总时长、最慢 20 个、各类别耗时，超出预算的列出来。

**指标**：每次运行把 branch coverage、test runtime、mutation score、flaky rate 写进任务摘要；每晚的结果汇总进 `docs/TESTING_METRICS.md` 看趋势（由本机 `make test-report` 拉取后提交，CI 不往仓库推送）。

**约束**：只用合成数据和夹具，不需要任何密钥，Tushare 不进 CI；action 钉 SHA；artifact 只在失败时上传、保留 1–3 天；推送前用 `ci-local` 在本机把 PR 上的 job 完整跑一遍。

## 长期习惯

开发中出现的每个错误、审查中发现的每个问题、排查时找到的每个可测试点，都写成测试用例留作回归测试：

- 用例名或注释写明来源（日期、现象）；
- 先确认它在旧代码上失败，再修；
- 能做变异检查的，同时确认测试能抓到这类错误。
