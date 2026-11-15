# D1 实现与验收记录

## 范围

实现 DESIGN 6.11，未修改 DESIGN。未提交、未推送；未搬迁、读取或改写真实数据，未安装或重启定时服务。`make dev` 创建并使用本 worktree 的 `.venv`，没有使用主检出的虚拟环境。没有修改 C4 的 `crates/`、`packages/panelbt`、`src/ashare/backtest/crypto*`，也没有修改 C6 的因子或新增季度数据集文件。

历史上下文按 onecontext 技能尝试检索，但 Aline 返回无法打开数据库；以 DESIGN 与已有代码/测试为准。本任务禁止改 worktree 外的文件，因此全局 `~/worklog/codex/` 未写入，阶段记录保留在本文。

## 实现选择与设计问题

1. **整根替换与指针的关系**：`ASHARE_DATA_DIR` 先选择根，再查该根下的指针；不会再查原根。否则设置替代根后无法使用其中的指针，和“整根替换”含义不一致。显式数据集路径、专用兼容变量直接选位置，不再解释该目录里的指针。
2. **两种显式路径**：股票 API 的 `root` 一直是整根，不能改成 `cn` 本身。新增 `dataset_dir(name, path=None, *, root=None, create=False)`：`path` 直指数据集，`root` 选整根并继续查其中的指针；显式 `root` 优先于环境配置。模拟盘显式 `data` 参数沿用整根语义；显式 `paper` 参数是完整账户根。
3. **登记及热保护**：设计热列表有六项，而 move 举例只列 `cn/live/paper`。按更保守的一边，保护全部热数据集；未明确温度的现有 `ml/hk/us/foreign/fixtures` 也先登记为热。冷数据登记 `dev/crypto/crypto_minute`；`crypto_minute` 按本次审查要求补入，无专用变量，分钟读取器不在本分支，本轮未改。ML 公共默认保持 `ml`，ML 基准保持 `dev/ml`，没有为了冷热分类改变业务默认目录。
4. **指针验证**：四字段必填，`path` 必须绝对，时间可解析，文件数和字节数为非负整数。目标必须为已有目录；缺失和“目标是普通文件”同样报中文 `DataLocationError`。`create=True` 不会为指针目标建目录。计数是搬迁快照，不要求之后每次读取重新扫描验证全部内容。
5. **保守搬迁边界**：不支持重复搬迁已有指针的数据集；默认拒绝被专用变量遮住的搬迁；目标父目录须已存在，避免拔盘时自动造挂载路径。拒绝源/目标重叠、目标在整个数据根内部、符号链接和特殊文件。源目录应停止写入；工具的排他锁只协调搬迁进程，不取代各业务的写锁。
6. **核对及失败处理**：复制前后核对源和目标的目录清单、文件大小、SHA-256，空目录也保留。`copy2` 保留文件修改时间。核对完成后，临时指针写入并 fsync，再以 `os.replace` 发布。发布前失败删除本次创建的目标；原有空目标恢复为空。清理依据实际指针是否发布，不使用发布后的内存标志；发布后若中断或源删除失败，保留完整目标和指针，不能回滚成悬空指针。断电或 SIGKILL 不能运行 Python 清理；可能留下目标/锁，须人工检查，不自动续拷或覆盖。
7. **status 约定**：普通未初始化目录不使整条命令失败；失效/损坏的指针使退出码为 1。显示有效位置的文件逻辑字节数，不是磁盘分配块数。JSON 损坏时仍显示原指针所在位置并标记重定向失败。
8. **保留原默认位置**（独立审查修正）：最初取消主检出回退是错误的兼容性选择，已恢复。`bench_data_root(checkout=...)` 统一处理端到端、规则及 ML 基准的原回退；显式根或环境根不回退，本地有指针也不回退。平台及 crypto 示例原有 cwd 默认由 `cwd_default=True` 保留，包含未重定向时的相对 Path 返回值；其优先级不变。
9. **后台服务必须保留动态指针**：原模板/生成器会把解析后的目录固化成专用变量，导致以后启动跳过指针检查。默认平台和模拟盘调度现在只保留整根配置，每次运行重新解析。平台新增 `--service-results` 保持常驻服务的 `platform/results` 布局；用户明确给的专用变量仍保留，且不重复追加子目录。launchd 标准输出路径只能在生成定义时确定，改变热数据位置后需要重新生成定义。
10. **测试保护**：股票库原有的 pytest 真实目录保护集中到 paths，并检查重定向目标。只读模拟盘路由不应在初始化时解析无关的平台回测结果库，改为真正查询回测时才解析；缺失指针的错误不会被当成空回测库。
11. **静态豁免**：只允许 `crypto/archive.py` 的 `data/spot`、`data/futures/um` 两个币安远端对象键，以及三个平台临时合成脚本里的精确 `root / "data"` 表达式。没有整文件豁免。AST 忽略文档字符串，识别单/双引号字面量、`Path("data")`、分段的 `/ "data"` 拼接、环境变量的 `"data"` 默认值和 `os.path.join("data", ...)`。临时结果、缓存和源码路径不属于真实数据根；显式调用方路径保持现有含义。

## 回归过程

每个下列产品错误都先用失败测试确认，再修复；没有放宽原有测试断言或新增 skip；本次将原测试里 Store 的根目录反推改为读取保留的整根属性。

| 来源 / 问题 | 失败证据 | 修复与测试 |
|---|---|---|
| 原业务入口忽略指针，crypto 没有默认入口 | 迁移调用点前 `test_entries.py` 的 12 项全部失败 | 六类入口重定向、目标缺失且不建目录 |
| `status` 坏 JSON 被当普通缺目录，错误地成功退出 | 新增测试失败 | `test_status_bad_pointer_is_failed_redirect` |
| 第一次检查和取得搬迁锁之间可能已有其他搬迁完成 | 注入新指针后旧流程仍发布，测试失败 | `test_move_rechecks_pointer_after_lock`，锁内重新检查 |
| 平台基准漏传 PanelCache 的必填 root 参数 | 替身 HTTP 驱动 `main()` 时 TypeError | `test_platform_verify_default_root`，传 None 走默认入口 |
| 合成面板脚本拼接 `cn/panel` 绕过市场指针 | 指针根下读不到面板，测试失败 | `test_synthetic_panel_writer_respects_market_pointer` |
| 只读模拟账户提前解析无关的平台库 | 首轮全量的 8 项原测试失败；新增断盘案例也失败 | `test_paper_read_does_not_resolve_unused_platform`，原 8 项恢复通过 |
| 平台生成器将重定向结果固化为专用变量 | 新增“不得固化”测试失败 | `test_service_does_not_freeze_pointer_as_legacy_override`；另测显式覆盖不追加子目录 |
| 模拟盘 launchd/systemd 同样固化默认账户位置 | 两种模板回归都失败 | `test_paper_schedule_keeps_pointer_dynamic`，生成时保留是否显式覆盖的信息 |

开发过程另有两次编辑错误：替换函数时误覆盖相邻 `Context` 和 `_Rows` 类。前者被入口回归捕获，后者由现有 `test_store_incremental_append_equals_full` 确认 NameError；已恢复原类，入口与整个 ML 存储测试通过。新写的 systemd 模板测试最初错误地要求出现 launchd 的日志路径，依据原 systemd 模板修正为检查运行时 `.ops` 路径和 nightly 命令；没有放宽原测试。

集成测试使用临时“外置盘”，实际读写股票 NumPy 面板、crypto Parquet 面板、ML 特征缓存及 JSONL、挖掘 JSONL、平台 SQLite、模拟盘状态 JSON，核对内置位置只含指针。搬迁测试包含同大小内容破坏、大小变化、复制异常、源变化、原子发布失败、默认保留、删除源、非空目标、全部热数据保护、force、重叠/链接和状态输出。

## 文件清单与调用点

| 模块 | 文件及改动 |
|---|---|
| 统一入口 | 新增 `src/ashare/paths.py`：登记表、根/数据集解析、错误、status/move、基准兼容入口 |
| 股票和市场 | `src/ashare/data/store.py`、`src/ashare/data/__init__.py`：Store、market_root、read_manifest；所有依赖 Store 的股票/港美股/期货读取继承重定向 |
| crypto（小范围改动） | `src/ashare/data/crypto/__main__.py`、`dataset.py`、`storage.py`、`sync.py`：只改路径导入、解析和默认参数；不改构建、资金费、季度合约逻辑。`__main__.py` 可能与 C6 产生合并冲突，由审查人处理 |
| ML / 挖掘 | `src/ashare/ml/store.py`、`src/ashare/factor/mining/ledger.py`：默认目录；`src/ashare/factor/mining/llm/strategy.py`：引用统一入口的仓库数据根常量，保留私有转写目录保护 |
| 平台 | `src/ashare/platform/config.py`、`paper_api.py`、`__main__.py`：平台目录、模拟盘根、按需解析回测库、常驻结果子目录 |
| 模拟盘 | `src/ashare/paper/ops/context.py`、`cli.py`、`install.py`：账户位置及调度时保留显式覆盖语义 |
| 基准 | `bench/e2e/common.py`、`rules_examples.py`、`ml/mlbench.py`、`gpu/common.py`、`crypto_examples.py`、`allocation/etf_example.py`、`data/round3/inspect_real.py`、`platform/verify_real.py`、`platform/synthetic.py` |
| 服务定义 | 新增 `bench/platform/service.py`，修改 `mk/platform.mk`、`deploy/platform/launchd/com.ashare.platform.plist.in`；根 Makefile 经 include 使用该目标，本身无硬编码数据路径 |
| 测试 | 新增 `test/paths/`：`conftest.py`、`test_paths.py`、`test_entries.py`、`test_contents.py`、`test_bench.py`、`test_static.py` 及包初始化文件 |
| 变异 | 新增 `bench/mutations_data_paths.py`、`bench/results/mutations_data_paths.json` |
| 文档 / 验收 | 新增 `docs/DATA_LOCATIONS.md`、本文、`bench/results/data_paths_validation.json` |

`bench/crypto_prepare.py` 的 `tmp/crypto` 是要求调用方准备的隔离副本，其他 `tmp/`/`.cache/` 合成产物也保留原语义；它们不是仓库数据根的后备位置。没有运行真实数据基准或下载脚本。

## 验收

以下是独立审查前的验收历史，独立审查后的结果见文末。该轮在当时源码停止修改后执行。没有改变现有的 GPU、可选依赖或真实夹具缺失跳过条件，也没有改变既有 xfail。完整摘要见 `bench/results/data_paths_validation.json`。

共享机器约束：所有重命令均请求 `nice -n 15`，沙箱拒绝 setpriority；当轮 Python 测试至多 `-n 4`（本次返工已改为最多 `-n 2`），数值库线程限制为 1–2，Cargo 使用 `CARGO_BUILD_JOBS=2` / `-j 2`。长命令加 `timeout`，没有超过 15 分钟的单命令或无限等待。

- `make dev`：本 worktree 独立环境和四个扩展构建成功。
- Ruff、Rust fmt / Clippy / Rust tests：通过；Rust 共 33 项测试通过。
- 定向检查：95 项通过（其中新增路径测试 71 项，含业务读写、静态门禁；其余为服务与模拟盘既有测试）。
- 变异：五项均被抓住；运行前检查原源码导入位置，每个变体也检查导入临时副本。变体包括去掉缺失目标检查、专用变量优先于显式路径、跳过 SHA、先发布指针后核对、放开热数据。
- 全量首轮：5882 passed、12 failed、127 skipped、1 xfailed。8 项是提前解析平台库的回归，已修复并通过；另外 4 项是当前沙箱禁止 `ps`，属于进程清理/监听检查的执行限制。

沙箱受限的四个测试为：`test/smoke/test_smoke.py::test_platform_starts_serves_and_stops`、`test/e2e/test_platform_api.py::{test_full_research_over_http,test_listens_on_loopback_only,test_leftover_check_catches_orphaned_children}`。保留原进程残留断言和 `ps` 检查，不改成跳过或忽略权限错误；审查人需要在可执行进程检查的本机终端完成最终全量验收。

最终 `make check` 的 Ruff、Rust 部分均通过；其中全量 `pytest -n 4`：**4 failed, 5904 passed, 127 skipped, 1 xfailed, 25 warnings in 866.10s (0:14:26)**。失败仅为上列四个沙箱 `ps` 权限用例，因此 `make check` 返回非零，**尚未达到全量全绿的交付条件**。没有忽略或删除失败用例。另已通过子进程方式的 `python -m ashare.paths move/status` 临时数据冒烟；产物扫描未发现本机绝对路径或真实数据。

在允许进程检查的终端，进入此 worktree 后重跑：

```sh
UV_PROJECT_ENVIRONMENT="$PWD/.venv" CARGO_BUILD_JOBS=2 PYTEST_ADDOPTS="-n 2" \
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 POLARS_MAX_THREADS=2 RAYON_NUM_THREADS=2 \
  timeout 900 nice -n 15 make check
```

沙箱也使测试夹具在 `snapshot_children()` 中断，无法完成其进程清理验证。尝试仅依据本任务测试日志识别服务 PID，但相应 pytest 临时目录已被清理；没有可确认的 PID，因此没有执行宽泛的 kill/pkill，也不声称已验证无残留进程。

## 阶段日志

- [19:34] AShare D1 — 统一入口、六类业务迁移和隔离搬迁测试落地；五项变异全部抓住，Rust 检查通过，全量定位到只读模拟盘提前解析平台库的问题及沙箱进程权限限制。

- [19:59] AShare D1 — 完成动态调度指针修复、71 项新增测试和 5/5 变异检查；最终全量 5904 passed、4 项受沙箱 ps 权限阻塞，完整清单与重跑命令已记录。

## 独立审查返工（D1）

本轮按审查意见先写回归再改产品代码。原始失败断言脱敏保存于 `bench/results/data_paths_review_red.txt`：首批 **19 failed / 4 passed**；补查 Reference 基准读取 **1 failed**；补查 crypto/ML 基准默认位置 **2 failed**。首批通过的四项是发布前和临时指针阶段，本来就安全；发布后的四项确实复现了悬空指针。

| 项目 | 修复位置与选择 | 首红测试（均在 `test/paths/test_review.py`） |
|---|---|---|
| P1-1 发布边界中断 | `paths.move` 的 finally 检查磁盘上的 `LOCATION.json` 是否实际发布；不依赖改名之后才赋值的布尔量。发布前、中、改名后、helper 返回后四个点，分别带/不带 `--delete-source` 注入 KeyboardInterrupt；源内容始终保留，发布后目标始终完整 | `test_move_interrupt_publication` 的 renamed / returned 参数 |
| P1-2 任意目录名 | `platform/datastatus.py` 传保留的原始根；`data/store.py` 保存 `data_root`；`data/cb_panel.py` 使用该整根读取正股。根参数省略和显式两种都测试。六类业务读写、入口和合成写入测试的目标均改为 `renamed-*` | `test_datastatus_renamed_dataset`、`test_cb_panel_renamed_dataset` |
| P1-3 旧默认兼容 | `paths` 增加 `cwd_default`，供 `platform/config.py`、`platform/paper_api.py`、`bench/crypto_examples.py` 保留 cwd 语义。`bench_data_root(checkout=...)` 保留 `bench/e2e/common.py`、`bench/rules_examples.py` 的 git 主检出回退；`bench/ml/mlbench.py` 将同一次选定的整根传给 `bench_ml_dir(root=...)` | `test_cwd_defaults_outside_checkout`、`test_cwd_pointer_and_overrides`、`test_bench_main_checkout_fallback`、`test_crypto_bench_cwd_default`、`test_ml_bench_preserves_fallback_root` |
| P2-4 位置错误展示 | `Reference._load` 和 `benchmark` 清空失败状态、传播 DataLocationError；`app.py` 单独处理为 HTTP 503“数据位置不可用”，完整原因只进日志；meta 的失败不更新时间戳，挂盘后下一请求成功 | `test_reference_location_error_not_cached`、`test_meta_location_error_not_cached`；追加 `test_reference_benchmark_location_error_not_cached` |
| P2-5 静态门禁 | `test/paths/test_static.py` 增加 `os.environ.get` / `os.getenv` 的默认值（位置/关键字参数）及 `os.path.join` 检测；不扩大运行路径允许清单 | `test_static_environment_and_join_defaults` 三个参数 |
| crypto_minute | `paths.DATASETS` 登记冷数据，无专用变量；变量不误用 CRYPTO_DATA_ROOT，覆盖创建、搬迁、读取和 status。同步 DATA_LOCATIONS 登记表；未改 DESIGN 和本分支不存在的 minute.py | `test_crypto_minute_registered_cold` |

全仓同类审计还修改了 `bench/data/round3/compare_real.py` 的临时构建位置，使用保存的整根；`test/data/` 中 11 个旧测试文件由 `.root.parent` 改为 `.data_root`，原内容/数值断言保持。剩余 `.parent` 是文件父目录、源码/模板目录、测试日志/诱饵账户目录，或故意放回错误的变异文本；没有从数据集路径倒推数据根的运行代码。通过 Python AST 扫描全仓非测试 Python，新增两种默认路径写法没有其他业务命中；文档和纯合成测试路径仍是说明/夹具。变异源码文件名也使用 `ashare/data/...`，避免与本地数据目录字面量混淆，不扩大豁免。

本轮保守边界：磁盘上已有任何指针时都保留完整目标，宁可留下可人工检查的副本，也不冒险删掉读者可能使用的目录。强杀/断电仍不承诺执行 finally；正在删除源阶段的中断可能留下部分源副本，但完整目标不回滚。cwd 相对入口与仓库默认入口的差异属于既有行为，统一的是解析规则，不擅自改默认位置。主检出回退只用于原有基准入口，有指针时即使目标缺失也绝不回退。

新增变异（保留原五项）：`publication_memory_flag`、`datastatus_reconstruct_root`、`cb_reconstruct_root`、`cwd_uses_checkout_root`、`bench_loses_main_checkout`。运行前确认导入本 worktree 源码；每个变体同时确认统一入口及被修改业务模块来自临时副本，逐变体恢复原文件。变异只跑相关断言，避免无关失败被算作捕获。

本轮测试最多 `-n 2`，数值线程至多 2；命令使用 timeout 和 nice -n 15（当前沙箱拒绝 setpriority，已在日志保留事实）。全量 pytest / make check 按用户指示留给审查人在沙箱外执行，本轮不将历史全量结果记为返工后已通过。未提交、未推送，未碰真实数据、全局 worklog 或其他 worktree。

验证执行调整：一次性选择 `test/paths test/data test/platform test/ml/test_preprocess_store.py test/paper/ops` 的大批次于 900 秒自动超时（退出 124），停止前未报告断言失败，不能计为通过。随后按实际改动拆分：路径全套及 11 个修改的数据测试文件一批，平台及 ML/模拟盘运维一批；每批限制 840 秒、最多 2 个 worker。未将超时算成通过，也未放宽断言或跳过用例。

返工后定向结果：

- 第一批 `test/paths` 全套 + 本轮修改的 11 个 `test/data` 文件：**200 passed, 2 warnings in 416.51s**。
- 第二批 `test/platform` 全套 + `test/ml/test_preprocess_store.py` + `test/paper/ops`：**674 passed, 2 warnings in 193.09s**，paper 跳过数为 0。
- 两批共 **874 项通过**，没有新增跳过或 xfail；警告是既有 Starlette TestClient 的 httpx 弃用提示。
- Ruff 全仓检查通过；format 全仓 793 个文件通过；`git diff --check` 通过；新产物扫描无本机绝对路径。

变异最终结果：**10/10 killed**。修改前本 worktree 导入检查通过，路径基线 **97 passed, 1 warning in 52.13s**；每个临时变体的统一入口和被改业务模块导入位置均通过检查。新增 P1 对应捕获：发布标志 4 个中断参数失败；数据概览 1 项失败；转债两种根参数各 1 项失败；cwd 默认两种入口各 1 项失败；主检出回退两个基准各 1 项失败。详见 `bench/results/mutations_data_paths.json`；验证总表保留审查前历史，并单列本轮 `review`，避免把旧全量结果当作本轮结果。

- [21:07] AShare D1 — 完成独立审查 3 项 P1、2 项 P2 和 crypto_minute 登记；22 个首红实例及 4 个安全边界对照保留日志，874 项定向测试通过，10/10 变异捕获，Ruff/format 通过；全量按用户要求交审查人执行。

## 审查修正与已知局限（2026-11-15，审查人）

- 审查第二轮 P1：位置类环境变量「设了但为空」时，旧代码各入口处理不一（有的退回 cwd，有的视同未设），新代码又按真假值视同未设，两种都会让数据静默落到另一个目录。统一改为 `paths._env` 读取，空值抛 `DataLocationError` 并写明变量名；回归测试 `test_review2_empty_location_variable_is_an_error` 覆盖 8 个入口，在改动前的代码上 8 项全部失败。这是对旧行为的有意改变：从静默换目录改为明确报错。
- 已知 P2，暂不修：`move` 在拿锁之后、进入清理范围之前，以及目标目录刚建好、尚未标记归属之前被 Ctrl-C，会留下陈旧锁或空目录（源数据不受影响，需手工删除后重试）；静态门禁漏掉 `os.path.join(ROOT, "data", ...)` 与 `pathlib.Path("data") / ...`，并把 `os.getenv("X", "data")` 这类非路径默认值误报；ML、挖掘、模拟盘运维的相对环境路径被规范成绝对路径（落点相同）。
