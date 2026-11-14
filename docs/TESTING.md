# 测试怎么跑

总目标见 `TESTING_PLAN.md`，每天的指标记在 `TESTING_METRICS.md`。工具在 `bench/testing/`，阈值都在 `bench/testing/budgets.py`，产物在 `.cache/testing/`（不入库）。

## 命令

| 命令 | 做什么 |
|---|---|
| `make test` | 全量并行跑（pytest-xdist，默认 6 个进程），记录耗时 |
| `make test-serial` | 全量串行跑一遍，记录串行耗时 |
| `make test-cat CAT=property` | 只跑一类；`CAT` 是 pytest `-m` 表达式，如 `CAT="unit or property"` |
| `make coverage` | Python 分支覆盖 + Rust 行覆盖；`coverage-python`、`coverage-rust` 只跑一半 |
| `make flaky N=5` | 全量随机顺序、不同种子、并行连跑 N 次，找时好时坏的测试 |
| `make test-report` | 汇总四个指标，更新 `TESTING_METRICS.md` 当天那一节 |
| `make test-metrics` | 依次跑 test、coverage、flaky，再出汇总 |

通用参数：`WORKERS=0` 串行，`WORKERS=4` 改进程数；`make flaky SEED=…` 换起始种子。直接用 pytest 也行：`uv run --no-sync pytest -n 6 -m differential`。

随机顺序默认关闭（`pyproject.toml` 的 `-p no:randomly`），要用时加 `-p randomly --randomly-seed=N`。复现某轮 flaky 用 `flaky.json` 里记的种子。

## 类别

`test/conftest.py` 在收集时给每个测试打一个主类别（耗时按它统计，写进 junit 的 `category` 属性），同时打上它属于的全部类别标记：

1. 测试上显式写了类别标记的，以它为准；
2. 目录：`test/smoke`、`test/integration`、`test/e2e`、`test/concurrency`、`test/perf`；
3. 用了 hypothesis 的是 `property`；
4. 差分对拍是 `differential`：文件名带 bitwise / reference / crosscheck / exact / independent / parity / comparison，导入了参考实现模块（`test/comparison.py`、`legacy.py`、`digest.py`、`events/reference.py`、`analysis/analysis_reference.py`），或测试名带 bitwise / legacy / reference / parity；
5. 其余是 `unit`。

`packages/paperbroker/tests` 也按这套规则分类，但只在和 `test/` 一起跑时才打标记。

## 指标口径

- **Branch coverage**：coverage.py，`branch = true`，配置在 `.coveragerc`，范围 `src/ashare` 和 `packages/*/python`；xdist 的工作进程靠 `patch = subprocess` 计入。HTML 报告在 `.cache/testing/htmlcov/`。
- **Rust 覆盖**：`cargo llvm-cov --workspace`，只算 `cargo test` 跑到的代码。PyO3 绑定和只被 Python 测试调到的内核不在这个口径里，所以绑定 crate 的数字偏低，不代表没被测。分支覆盖要 nightly 工具链，没有时只报行覆盖。
- **Test runtime**：墙钟时间、各测试耗时合计、各类别耗时、最慢 20 个；单个测试超过类别预算（`CATEGORY_BUDGETS`，如 unit 2 秒）的列出来。
- **Mutation score**：读 `.cache/testing/mutation.json`（`make mutate-all` 产出），被杀死 / (杀死 + 存活)，超时算杀死，等价变异不进分母。
- **Flaky rate**：N 轮里既通过过又失败过的测试数 / 测试总数；一直失败的单列。flaky 一律从根上修，不靠重跑。
- **性能**：读 `.cache/testing/perf.json`（`make perf-check` 产出），列出退步项。

## 写测试时守住隔离

全量测试必须在并行和任意顺序下都通过：

- 文件写到 `tmp_path`，不写仓库里的固定路径或共享临时目录；
- 环境变量、模块属性、`sys.modules`、`sys.path` 用 `monkeypatch` 改；
- 不占固定端口；随机数用本地 `np.random.default_rng(seed)`，不动全局种子；
- 不依赖别的测试先跑（模块级缓存、注册表要在测试里自己建好或清掉）。
