# 变异测试

一条命令给出全仓库按模块的 mutation score，PR 上只变异改动过的代码。Python 用 mutmut 3，Rust 用 cargo-mutants；代码在 `bench/mutate/`，make 目标在 `mk/mutate.mk`，分片在 `bench/mutate/shards.toml`。先 `make dev`。

## 命令

| 命令 | 做什么 |
|---|---|
| `make mutate-module MODULE=src/ashare/platform/config.py` | 一个分片或路径（文件、目录、分片名，逗号分隔可多个） |
| `make mutate-diff BASE=origin/main` | 只变异相对 BASE 改动过的 Python 函数和 Rust 行 |
| `make mutate-all` | 全部分片依次跑；`MUTATE_LANG=python` 或 `rust` 只跑一种 |
| `make mutate-report` | 改了等价变异登记后，用已存的结果重出分数和存活清单 |
| `make mutate-estimate` | 数每个分片有多少变异体、估全量耗时 |
| `make mutate-controls` | 对照实验（下面「怎么知道测的是变异后的代码」） |
| `make mutate-legacy [LEGACY="rules groupbt"]` | 旧的手写变异脚本：列状态，或跑能跑的几个 |

`JOBS`（默认 2，同时跑的变异体数，上限 4）、`MUTATE_TIMEOUT`（每个分片的墙钟上限，默认 3600 秒）可调。`MUTATE_ARGS` 传给驱动，常用：`--part K/N`（Rust：只跑第 K 份，K 从 0 起，把一次运行压在 20 分钟内）、`--resume`（Python：超时停下的分片接着跑，已出结论的不重跑）、`--re 正则`（Rust：只跑名字匹配的变异体）。

变异测试和日常工作共用一台机器，不能把它拖慢：驱动启动后自己降到 `nice 15`，mutmut、cargo、pytest 都继承；make 目标外面再包 `nice -n 15`；子进程里 rayon、polars 默认单线程；cargo-mutants 同时最多 `JOBS` 个变异体、每个只用 1 个编译线程。实测 4 路并发会让这台 15 核的机器明显变卡（2026-11-14），共享时保持 2 路。Makefile 里原有的 `make mutate` 是旧脚本 `bench/mutations.py`，单模块目标因此叫 `mutate-module`。

## 输出

- `.cache/testing/mutation.json`：`make test-report` 读的固定格式，按分片一行：`killed`、`survived`、`timeout`、`equivalent`、`score`，外加 `total`。
- `.cache/testing/mutation_survivors.md`：存活的变异逐条列出（文件:行、函数、变异内容、id），下一轮补测试用。
- `.cache/testing/mutation_details.json`：每个分片的运行参数、耗时、各状态计数（含不进分数的 unviable / skipped / error），以及登记了却没对上的等价变异。
- `.cache/mutate/results/<scope>-<分片>.json`：每次运行的逐个变异体结果。`full` 的 `mutation.json` 由这些文件合并而成，所以分片可以分几次跑、最后汇总；同一个变异体以较新的结果为准。`diff` 每次运行前清空。

## 口径

- **score = (killed + timeout) / (killed + timeout + survived)**。超时算被杀；`survived` 含「所选测试没有一个经过这段代码」的变异（mutmut 的 no tests）。
- 不进分母：登记过的等价变异；Rust 编译不过的变异（unviable）；没跑到的（skipped）；工具自身出错的（error，见 details）。
- 分母为 0 的分片记 1.0。
- 测试在变异体上崩溃（段错误、pytest 收集出错）算被杀：前提是同一套测试在未变异的代码上通过，这一点每次运行开头都会检查。

## 等价变异

存活的变异逐个看：能补测试的补测试；确实在任何输入下行为都不变的，登记进 `bench/mutate/equivalent.toml`，写明理由，然后 `make mutate-report`。登记按（文件，函数，变异内容）匹配，变异编号、行号变了也对得上；函数改动后对不上的条目会列在存活清单开头。「测试没覆盖」「影响很小」不算等价。

## 工作方式

**Python（mutmut）**。每个分片在 `.cache/mutate/py/<分片>/` 下有自己的工作目录：`src` 指向分片的源码根（`src` 或 `packages/<包>/python`），驱动按 `setup.cfg` 的 `[mutmut]` 段生成本分片的配置（只变异本分片的文件，统计只用本分片的测试）。mutmut 把源码复制到 `mutants/` 并插入变异，测试在 `mutants/` 里跑。仓库的构建守卫要求 ashare、panelbt、panelops、paperbroker 都从 `mutants/` 下导入，所以驱动把扩展包（含编译好的 `.so`）、crates、bench、docs 也复制进去，并用 `PYTHONPATH` 让这些副本排在 editable 安装之前，被变异的 `mutants/src` 排第一。因此不要在仓库根目录直接跑 `mutmut run`。

- 进程隔离用 forkserver，且分叉前不跑任何测试代码（`forkserver_warmup = none`）：rayon 和 polars 的全局线程池如果在父进程里初始化过，分叉出的子进程一用就死锁，会被误记成超时（=被杀）。代价是每个变异体都要重新导入一遍，约 1–2 秒。
- mutmut 先跑一遍分片的测试，记下每个函数被哪些测试经过，之后每个变异体只跑这些测试。**只在子进程里执行的代码**（平台的工作进程）记不到，会显示为 no tests；变异本身会传到子进程（`MUTANT_UNDER_TEST` 环境变量），只是没有测试被选去跑它。

**Rust（cargo-mutants）**。cargo-mutants 只会跑 `cargo test`，而内核主要靠 Python 测试守。驱动直接启动 `cargo-mutants` 二进制（经 `cargo mutants` 启动时 cargo 会覆盖 CARGO 变量），并把 `CARGO` 指向 `bench/mutate/cargo-shim`：

1. 构建阶段（`cargo test --no-run`）：照常构建，再从变异后的临时副本编出四个 PyO3 扩展（`cargo rustc --release`，关掉 LTO 以缩短链接；浮点语义不受 LTO 影响），放到副本里各包的 Python 目录下。编不过即 unviable。
2. 测试阶段（`cargo test`）：先跑本包的 Rust 测试；通过后在副本里证明导入的就是刚编的扩展（`test/build_guard.py` 比对扩展里嵌入的源码摘要和副本源码的摘要，并检查四个 `_core` 和 ashare 的路径都在副本里）；证明失败打印 `MUTATE-SHIM INFRA-ERROR`，驱动把这个变异体记为 error 而不是被杀；证明通过才跑分片的 pytest 选择。
3. cargo-mutants 先在未变异的副本上跑一遍完整的 1–2（baseline），不过就整片中止。

扩展绑定层 `packages/*/src/*.rs` 不变异：cargo-mutants 27 解析 `packages/panelbt/src/lib.rs` 里的宏时内存暴涨被系统杀掉（所以每次都带 `-p <包>`），而且这一层只做参数转换。

## 怎么知道测的是变异后的代码

变异测试最常见的假象是测试导入的仍是原来的代码，所有变异「存活」或所有变异「被杀」都不说明问题。`make mutate-controls` 跑三组对照，结果写进 `.cache/testing/mutation_controls.json`：

- Python（在跑过的分片工作目录上）：目标模块从 `mutants/src` 导入且带 mutmut 的跳板代码；不激活变异时分片测试全过（变异 0）；往副本末尾加一行 `raise` 后测试失败且报的就是这句（必死变异）。mutmut 每次运行自己也做「干净测试」和「强制失败」两项检查，结果记在 details 的 `clean_run_passed`、`forced_fail_caught`。
- Rust（仓库的一份临时副本，走同一个 shim）：未变异时构建、证明、pytest 都过；把权重模式内核（分组回测用的那个）的换手减半后，扩展里的源码摘要随之改变、测试失败；把原来编好的扩展放回变异后的副本，证明步骤报 INFRA-ERROR，而不是算作被杀。

## 分片与耗时

30 个分片：Python 17 个（ashare 按子包，加 4 个扩展包的 Python 层），Rust 13 个（按内核文件）。`make mutate-estimate` 数变异体并按已有运行的实测速度重估。

2026-11-14 实测（15 核共享机器）：

| 运行 | 变异体 | 并发 | 墙钟 | 每个变异体（墙钟） |
|---|---:|---:|---:|---:|
| `src/ashare/platform/config.py` | 258 | 4 | 544 s | 2.1 s |
| `src/ashare/factor/groupbt/`（跑完 948 / 2673） | 948 | 4 | 约 42 min | 2.7 s；`api.py`、`plot.py` 的函数被约 500 个测试经过，单个变异体存活时要跑约 2 分钟 |
| diff：`config.py` 相对 HEAD~4（只变异改过的函数） | 114 | 2 | 311 s | 2.7 s，含统计 |
| `group_weights.rs` 第 0/8 份 | 40 | 2 | 660 s | 16.5 s（每个约 5–10 s 增量编译 + 50–60 s 测试） |

全量估计（`make mutate-estimate` 的变异体数 × 上表速度，2 路并发）：Python 约 13.4 万个变异体，按 4 秒一个约 150 小时；Rust 约 7200 个，按 16 秒一个约 32 小时。全量一次跑不完，建议：

- PR 上只跑 `make mutate-diff`（几十到一两百个变异体，十分钟上下）；
- 每晚按分片轮转跑全量（每晚挑几个分片，`--part` / `--resume` 把单次运行压在 20 分钟内，跑完一轮约两到三周），结果按分片累积，`make mutate-report` 汇总；
- 先轮转核心：Rust 内核、`panelbt-py`、`ashare-factor*`、`ashare-risk`、`ashare-portfolio`；`ashare-data`（4.3 万个，多是下载与解析代码）和 `ashare-report` 放后面；
- 变异体存活时代价最高：给慢测试多的分片（如 groupbt 的逐位对照）补快的针对性测试，比放宽超时更有效。

## 旧的手写变异脚本

`bench/mutations*.py`、`bench/*/mutations.py` 等是之前各条线审查时写的定向变异（如「印花税漏算」「挂单延续碰到所有持仓」「参考实现和内核一起改」），每条对准一个测试。它们和工具生成的算子级变异互补，不并进分数，单独记在 `.cache/testing/mutation_legacy.json`。`python -m bench.mutate.legacy list` 列出每个脚本能不能无人值守地跑及原因：多数直接改工作区源码并 `make dev` 重建，只能在干净的工作区上一个一个跑，不能和 mutmut / cargo-mutants 同时跑；几个已经过时（用例的原文对不上现在的代码）或需要联网下载依赖，先修脚本再用。会覆盖已入库结果文件的脚本，跑完后驱动把入库版本恢复，新结果留在报告和 `.cache/mutate/legacy/` 的日志里。
