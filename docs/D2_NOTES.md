# D2 实现与验收笔记

## 规范解读与审查决定

- 校验器独立读取 JSON、Arrow 元数据和内容，不调用封存、构建、读取方的转换/检查函数，只共用规范常量。
- Data Package v2 的 resource `bytes`、`hash` 是标量，但本规范同时要求逐文件清单。采用 `path` 列表、汇总 `bytes`/`rows`、按 path 顺序连接文件字节的 `hash`，并加 `files` 数组保存逐文件四项。数组顺序参与版本，不暗中排序语义列表；JSON UTF-8、ensure_ascii=False、sort_keys=True、紧凑分隔符、不允许 NaN，顶层删除 version/created。
- 不补行、不前填、不改历史代码、缺口必须披露及单位语义，不能仅凭一次数据包快照证明。校验能检查可观测约束；来源真实性由适配器负责。合法零成交量、零计数、负值扩展因子不能被当成缺失哨兵一概拒绝。
- 2026-11-16 审查决定：available_at 下界只约束 kind=bars。公司行为和其他事件允许提前公告，仍须有非空的 UTC available_at；第四轮明确其自身必须 start/end，其余事件字段按 column:available_at 声明。A 股导出继续使用来源日终裁决时刻，不改导出逻辑、不伪称原公告时间。
- 审查接受 A 股原始全矩阵差异；验收口径为：在「有报价格子 + 规范化后的共享字段」上逐位一致（含 NaN 位置），时间轴与 has_quote 独立核对。无报价格子的状态值变 NaN、涨跌停价 0 变 null，均为规范要求；计数与原因见下表，不再作为未决问题。
- knownAt 新增 unknown，表示来源不能保证时点；普通值字段语法接受，校验用独立 known_at_unknown 警告，不影响返回码；第四轮禁止将 unknown 列用作时间依赖，另以 known_at_dependency 失败。拼写错误仍以 known_at 失败，警告不能掩盖其他字段的语法失败。info 和面板字段元数据保留 unknown，使用方不得当成时点数据。
- 3.1“每张时间序列表恰好一种”不计 available_at 等可知时点；公司行为采用 5.5 明定的 ex_date。
- 日频查询起止含两端，兼容原 Panel；日内查询左闭右开，兼容 crypto。数据时间轴按日历/规则网格生成，不前填。
- 合并前复核明确：行情引用列及全表 available_at 的有效可知时刻为 max(时间值, 该列自身 start/end 边界)，不能把 end 列的盘中时间值当成盘中已知。静态表与事件表不构造行情区间，start/end 随行可知，依赖按时间值判断；仍拒绝延迟、unknown、多级时间依赖。

## 环境与过程

- 在本 worktree 执行 make dev，CARGO_HOME、UV_PYTHON_INSTALL_DIR、MPLCONFIGDIR 均指向本地 .cache；cargo jobs=2。
- onecontext 已读取并尝试搜索，Aline 返回 unable to open database file；以当前规范与代码为依据。
- 所有重任务通过 nice -n 15 启动；当前沙箱 setpriority 返回 Operation not permitted，额外限制 Polars、BLAS 等线程数为 2。
- 全局 worklog 在本次允许写入范围之外且用户明确要求不改 worktree 外文件，故阶段工作日志保存在本笔记，不修改全局目录。

## 合并前复核修复（2026-11-16）

- 先只读通读 `.review/recheck2_d2.md`，确认 P0 收盘前泄漏、P2 静态表投影不一致和 P2 导出摘要矛盾。新文件 `test/spec/test_r5_availability.py` 在改产品代码前确认 **16 failed、22 passed，4.25s**；失败覆盖日频/小时的 end 时间依赖、静态自定义时间列默认读取，以及检查静态语义时发现的依赖空值漏检。随后补上取 max 时 null 不可被边界替代、校验器不调用读取方时钟的回归。
- `timing.dependency_time` 对合法时间依赖取 max；公共 read(as_of)、lazy 时点扫描的字段屏蔽和全表门槛都使用它。`panel_timing.Placement.fill` 同样按该口径落格，按批次缓存依赖时间数组，不增加整包载入；面板决策点仍为区间结束。保留复杂依赖链硬拒绝；没有重新引入 unknown 链传播。
- `timing.knowable` 将静态表和事件表的 start/end 统一解释为随行可知，不再仅为名叫 available_at 的列开例外。默认全字段与投影 currency / 时间列在公告之前、恰好公告时和公告之后都一致，且不访问不存在的 date。
- 校验器仍独立执行声明规则、依赖非空和行情原始时间下界：02:00 的原始时间值合法，即使其列声明 end，也不能错误要求原始值晚于 07:00。`check_content._availability` 移除“无时间轴就跳过整个检查”的分支，使静态依赖 null 精确报 available_at。新独立性回归把读取方三个时钟函数替换为抛错，校验器照常接受合法包、拒绝仅破坏下界的包；无读取侧函数复用。
- 规范主要补充限于 §3.3、§9.3。另依据本次任务第 3 条“另一项 P2 一并处理”的具体授权，仅修正 §10.1 的一处摘要：available_at 自身 end，其余字段 column:available_at，与现有导出器一致。未改导出逻辑或实际值。下面第四轮保留旧句的说明属于历史记录，本轮已消除该矛盾。
- 首轮修复后定向 **96 passed，9.17s**，包含上一轮规则、事件提前公告和 **80 个普通数组逐字节一致**，冻结基准未修改。此次新增 41 个测试实例；最终完整测试、变异和静态验收数字见下方收尾记录。
- OneContext 本轮查询仍返回 unable to open database file；未写工作区外日志。沿用自己的 venv、worktree 内缓存与 tmp、nice 和最多 2 计算线程，不并行运行完整测试组。本轮不重新导出真实数据，历史真实验证结果不作为本轮新口径的实跑结果。

合并前复核最终验收：

- 完整“变异 0”先于所有变异执行：`test/spec` **292 passed，59.12s**，打印并断言 `ashare.__file__` 为本 worktree 的源码。随后 `test/paths` 以 2 个 worker 顺序执行，**106 passed，72.12s**；共 **398 passed、0 skipped**。唯一警告为既有 Starlette/httpx 弃用提示。
- **45/45 变异被抓到**，结果写入 `bench/results/mutations_data_spec.json`。新增 7 处：max 缺少列自身时刻、max 缺少时间值、column 引用绕过 max、全表 available_at 绕过 max、静态自定义时间列恢复旧例外、max 吞掉 null、静态依赖漏查非空。全部变异进程返回 1，输出包含目标测试失败，未用导入/收集错误充数；45 个变异子进程累计 467.58s。
- `make check` 的静态项全部通过：ruff check；ruff format（865 个文件）；静态路径测试（包含在 test/paths）；cargo fmt；cargo clippy（offline、workspace/all-targets、`-j 2 -- -D warnings`，1.40s）。`git diff --check` 通过。
- 关键回归：`test_s3_3_s9_r5_review_dependency_effective_max`（02:00/07:00、start/end、column/global、公共 read/lazy/直接 helper、字段投影）；`test_s9_r5_intraday_dependency_boundary`；`test_s9_r5_panel_matches_effective_close_view`；`test_s3_3_s9_r5_static_default_equals_projection`；`test_s3_3_r5_static_dependency_null_rejected`；`test_s3_3_s9_r5_null_time_not_replaced_by_interval_end`；`test_s3_3_r5_checker_independent_of_reader_clocks`。现有 `test_s10_cn_bitwise_roundtrip` 继续核对导出声明与 §10.1 摘要一致。
- `test_s9_r3_80_ordinary_arrays_bitwise_unchanged` 的四组合成场景均通过，80 个数组逐字节相同，未重新生成冻结基准。通过与本轮开始时规范快照比较，确认修改仅限 §3.3、§9.3 及已认可 P2 指定的 §10.1 摘要句。
- 本轮没有新增未决问题。旧包迁移与历史真实验证的范围仍按第四轮记录执行。未提交、未推送、未改 worktree 外文件。

## 第四轮复核修复（2026-11-16）

- 已先只读通读 `.review/recheck_d2.md`。采用审查人指定的“拒绝不支持组合”，不推导依赖链的有效可知时刻。规范本轮仅改 **3.3（knownAt 定义）和 9.3（读取）**；时间依赖的值不能证明依赖列自身已经可知，因此 `column:<列>` 的目标及全表 available_at 必须显式 start/end，延迟、unknown、多级/自引用/循环、缺省声明均拒绝。
- 改产品代码前新增完整包回归：**28 failed、4 passed，14.37s**。三个复核变体分别只改一个时间列的 knownAt，校验精确断言 `known_at_dependency`；分别测试 check、read(as_of)、lazy `_scan`、panel，并追加 raw read/raw lazy 和投影主键/不相关字段的入口。`allow_unknown=True` 同样不能放过这些非法声明。
- `check_schema.time_dependencies` 独立遍历资源声明，`report.py` 新增稳定错误代码 known_at_dependency；读取侧在 `dependencies.validate_time_dependencies` 另行实现，reader._scan、panel.make_panel、timing.as_of_view 在读取数据前调用。两份实现互不调用，双向禁用对方检查的回归证明各自仍可拒绝错误声明。
- 删除 unknown 引用链遍历、全表 unknown 传播与已经无用途的 declaredKnownAt；普通值字段的 unknown 仍是警告，时点读取默认拒绝、显式允许时保留元数据；成交掩码对它实际使用的未知值继续传播标记。R3 中允许 unknown 时间依赖显式放行的断言按本次规范决定改为强制拒绝；未删除对应场景。
- 事件表 available_at 自引用现为非法。独立校验对该列使用随行可知的例外，其他事件字段仍要求 column:available_at；时点读取仍按公告时间过滤，available_at 自身不按事件日收盘重新屏蔽。公司行为导出器只把该列的声明改成 end，未改实际公告时间、事件值、行筛选或任何 A 股规范化口径；合成导出回归检查新声明并保留逐位对拍。
- 第 10.1 节旧的“所有字段 column:available_at”摘要应按新第 3.3 节对 available_at 的明确例外理解；按授权范围不改 10.1。开发中的 specVersion 保持 1；此前含 available_at 自引用或其他非法依赖的包需更新声明、重新封存并校验，不会在读取时偷偷修正。此前真实验证 JSON 是当时规范下的历史记录，本轮未将其作为新规范验收结果，也未重跑真实来源导出/对拍。
- 修复后首轮定向 **75 passed，9.53s**，包括事件提前公告 start/end 两种声明、校验/读取侧独立性和 **80 个普通数组逐字节一致**；冻结基准未改。完整测试、变异和静态结果见本节收尾记录。
- 变异脚本把四个过时的 unknown 引用链目标换成当前规则的列引用发现、全表 available_at 发现、读取规则拒绝和普通 unknown 值元数据目标，并新增独立校验规则删除变异；没有保留不可达代码来凑数。每次仍先运行完整变异 0，断言源码来自本 worktree。
- 本轮沿用自有 venv；CARGO_HOME、UV_PYTHON_INSTALL_DIR、MPLCONFIGDIR、TMPDIR 全在 worktree，重任务 nice 启动、计算线程最多 2，完整测试组顺序执行。OneContext 搜索仍报 unable to open database file。工作日志按用户禁止工作区外写入的要求留在本笔记。

第四轮收尾结果：

- 首轮完整变异 0：**249 passed，50.63s**，无跳过；打印并断言导入本 worktree 的 ashare。此时比第三轮新增 35 个测试实例，原有时间依赖用例按本次规范改为强制拒绝，普通 80 数组冻结对拍保持不变。
- 首轮 **37/37 变异被抓到**，最终结果见 `bench/results/mutations_data_spec.json`。读取侧 `reader_time_dependency_rule` 与独立校验侧 `checker_time_dependency_rule` 分别删除新规则；另分别删除 column 引用发现和全表 available_at 发现，均由指定回归抓到。所有变异返回 1、有失败断言，没有用收集/导入错误充数。
- 核心回归为 `test_s3_3_s9_r4_review_variants_rejected`（三个变体 × 六个入口）、`test_s3_3_s9_r4_row_known_reference_works`、`test_s3_3_s9_r4_other_dependency_declarations_rejected`、`test_s9_r4_invalid_dependency_cannot_hide_behind_projection`、`test_s3_3_r4_checker_does_not_use_reader_guard`、`test_s9_r4_reader_does_not_rely_on_checker` 与 `test_s3_3_s9_r4_event_time_column_is_row_known`。

- 收尾追加静态 instruments.available_at 的 start/end 两个正例，先确认 **2 failed，0.77s**：旧判断只照顾事件表，静态表会错误读取不存在的 date。改成所有非 bars 表的 available_at 均只执行行级公告时间过滤；新增 `test_s3_3_s9_r4_static_available_at_is_row_known` 和对应变异，再从完整变异 0 重跑。没有增加依赖链推导。

第四轮最终验收：

- `test/spec`：**251 passed，39.36s**；`test/paths`：**106 passed，14.79s**（2 个 worker）。合计 **357 passed、0 skipped**，路径测试仅 1 条既有 Starlette/httpx 弃用警告。完整 spec 基线仍先于所有最终变异执行，导入路径断言通过。
- 最终 **38/38 变异被抓到**，包括校验侧、读取侧新规则及非 bars 公告列的边界；全部为断言失败退出，未用导入/收集错误充数。`bench/results/mutations_data_spec.json` 保存最终这次运行。
- `make check` 的静态项全部通过：ruff check；ruff format（864 个文件）；静态路径检查（包含在 test/paths）；cargo fmt；cargo clippy（offline、workspace/all-targets、`-j 2 -- -D warnings`，2.65s）。`git diff --check` 通过。
- 新增测试共 37 个实例，80 数组冻结基准及其文件未修改，逐字节对拍通过。通过与本轮开始时的规范快照比对，确认只有第 3.3、9.3 节发生改动。没有新增待审查人决定的问题；旧包迁移和真实数据未重跑的范围如上所述。未提交、未推送、未改 worktree 以外文件。

## 第三轮复核修复（2026-11-16）

- 先只读通读 `.review/review_d2_r3.md`。新增回归在改产品代码前确认 **14 failed、2 passed，3.04s**；两个正例分别是允许延迟历史回读，以及不相关的 unknown 列不阻止读取。旧实现的四组合成场景（普通、缺行/null、小时、涨跌停/停牌）× 两种窗口 × 十个字段，共 **80 个数组**，预先冻结字节摘要、NaN 位、日期/代码轴和元数据在 `test/spec/r3_panel_baseline.json`，修复后逐项一致，没有重生成基准。
- P1-1：`panel_timing.Placement.fill` 对每条来源行情验证日历成员或 UTC 规则网格。仅当查询确实扩大了历史窗口、来源行也在合法日历/网格上且早于原请求起点时，才允许不落入原位置；这些行仍参与延迟值计算。原复现的 00:30/01:30、公开 panel 入口、有/无延迟、仅历史行错位、日历缺日均有回归。
- P1-2：新增读取侧 `dependencies.py`，递归检查 `column:<列>` 引用与全表 available_at，访问集合避免自引用死循环。不涉及的未知列不阻止安全字段；主键请求和 has_quote 也受全表 available_at 约束。panel/read(as_of) 默认拒绝任何未知时点依赖，显式 allow_unknown=True 时在只读 field_metadata 中传播有效 knownAt=unknown、unknownDependencies，原声明另存 declaredKnownAt。Polars 返回长表的附加元数据不承诺经过后续 Polars 转换仍保留。校验器未引用这套读取侧实现。
- P1-3：reader._verify 在缓存未命中时读取单个分区的不可变 bytes，同一份 bytes 用于 SHA-256 和 Parquet 解析。scan.verified_source 在缓存命中时使用同一个文件实例，并在读取完成、交付前检查 dev/ino/size/mtime_ns/ctime_ns，变化则拒绝。测试覆盖复核的内存流竞态、真实 inode 原地改写且恢复 mtime、损坏冷读后下一次读取拒绝，以及 Parquet 解析已结束后的改写。
- `scan.py` 使用 Polars register_io_source 包装逐分区 lazy scan，仍下推列与行谓词。只预取首分区获取物理 schema（保留毫秒/微秒精度），后续逐分区核对和解析，再按批次交付；不会在 lazy concat 中积存整包字节。热读不重复哈希；单次输入额外内存由单个压缩分区与该分区投影结果决定，非整包长表。Polars 该扩展 API 属于不稳定接口，当前锁定依赖版本下有投影、渐进打开分区及完整套件覆盖，升级时需复验。
- 产品修复后的首轮定向 **56 passed，7.44s**（包含 80 数组对拍）；追加解析后竞态与分区按需打开测试后，完整性文件 **7 passed，1.15s**。首轮完整基线 213 passed，30.68s；新增八个变异全部被抓。旧的 verified_file_instance 变异在新架构中只改变 schema 来源，实际数据仍用快照，属于等价变异；改成同时切换实际数据来源，保留原测试断言，再从变异 0 完整重跑。另补 read 直接 unknown 值显式放行的正例。最终测试、变异和性能数字见本节收尾结果。
- 本轮规范只修改 **9.3**，其他规范章节、导出逻辑及 A 股已认可的规范化口径不改。OneContext 本轮搜索仍返回数据库无法打开；nice 的系统限制及线程限制与前述环境记录相同。本轮工作日志留在本文件，不写工作区外目录。

第三轮收尾结果：

- `test/paths`：**106 passed，49.91s**，2 个 worker，无跳过；仅 1 条现有 Starlette/httpx 弃用警告。连同 spec 合计 **320 passed、0 skipped**。
- `make check` 的静态项全部通过：ruff check；ruff format（863 个文件）；cargo fmt；cargo clippy（offline、workspace/all-targets、`-j 2 -- -D warnings`，0.45s）。静态路径测试包含在 paths 中。
- 变异 0：**214 passed，48.95s**，无跳过；打印并断言导入本 worktree 的源码。最终 **36/36 变异均被抓到**（28 个既有目标、8 个新增目标），返回码全部为 1 且有失败断言，无导入/收集错误充数。结果见 `bench/results/mutations_data_spec.json`。
- 本轮新增 24 个测试实例；网格与时点依赖在 `test_r3_timing.py`，竞态和分区读取在 `test_r3_integrity.py`，80 数组冻结对拍在 `test_r3_parity.py`。三项核心回归分别是 `test_s9_r3_off_grid_public_panel_rejected` / `test_s9_r3_off_grid_history_is_not_discardable`，`test_s9_r3_unknown_dependency_public_entries` / `test_s9_r3_review_unknown_available_at_dependency`，以及 `test_s7_r3_review_mutable_stream_race` / `test_s7_r3_real_inode_overwrite_rejected` / `test_s7_r3_cached_identity_checked_after_parquet_read`。

- 再次只读检查 worktree/tmp 中此前导出的三份真实数据包，全部 check 返回 0；本轮未重新导出、未与原始来源重新对拍。只投影 close、volume，首读清空进程哈希缓存，不清系统页缓存；缓存后哈希调用均为 0。仅汇总结果保存于 `bench/results/data_spec_r3_read.json`，不含逐行行情。

| 既有隔离包 / 范围 | panel 形状 | check 秒 | panel 首读秒 | panel 缓存秒 | read 秒 | 首次/缓存哈希次数 |
|---|---|---:|---:|---:|---:|---:|
| A 股 2024 全市场 | 242 × 5923 | 11.373 | 0.543 | 0.454 | 0.093 | 14/0 |
| 加密小时 2025 全市场 | 8760 × 4 | 5.686 | 0.399 | 0.023 | 0.013 | 13/0 |
| BTCUSDT 现货分钟 2025 | 525600 × 1 | 5.656 | 0.203 | 0.187 | 0.036 | 13/0 |

本轮三项 P1 均已修复，没有新增待审查人决定的问题。已知接口限制为上述 Polars 扩展 API 的升级复验和长表附加元数据的传播范围。`git diff --check` 通过；未提交、未推送、未改工作区外文件。

## 独立审查六项修复（2026-11-16）

- 已只读通读 `.review/review_d2.md`，先将两项 P1 和四项 P2 的复现写成完整包回归。首次测试中发现 Polars 默认整数字面量是 Int32，改成显式 Int64 后重新运行，确认旧产品代码 **25 failed，2.72s**，没有依靠无关的物理类型错误获得通过。
- 时点实现分为 `timing.py`（长表 as_of / 可知时刻）和 `panel_timing.py`（决策网格、历史回读、流式落格）。日频决策时刻是当地 sessionClose，日内是右端；延迟按实际时长加到 UTC 区间结束，再放入首次到期的决策行。跨休市期多个值同时到期时选来源区间最晚的值，包含 null，不前填；有 start 的查询也读入需要的历史分区。未知时点字段及掩码依赖默认拒绝，显式 allow_unknown=True 保留元数据后放行。
- `read(as_of=...)` 过滤晚公告行及尚未开始的 bars 区间；每个请求字段按自己的可知时刻保留或置 null，主键不屏蔽。事件日期允许在 as_of 之后，只需已经公告。unknown 不可作时点读取；无 as_of 的原始长表照常可读。行内存在 available_at 时，面板所有值也受它约束。只有清单中的完整日历/资源网格范围能承接延迟值，不臆造范围外的未来交易日；日频时点入口需要显式 sessionOpen/sessionClose。
- 读取缓存键加入设备、inode、ctime；使用 fstat 对同一个打开的文件做缓存身份和哈希检查，Polars lazy scan 接收该文件句柄并持有其文件实例。路径替换不会重开到新文件；回归同时覆盖原地等长改写后恢复 mtime、替换 inode 后恢复 mtime，以及核对后立即替换路径。缓存最多 4096 项，列裁剪和分区裁剪仍在。
- 价格掩码按 open.allowNegative 使用正值域或非零域；单位词汇新增纯声明常量 PRICE_UNITS。独立校验器按 unit 限制 number/Float64 与 count/Int64，检查所有显式价格字段及其 allowNegative；仍不调用读取或写入方的转换函数。数量、金额非负约束也覆盖扩展字段。
- 单位检查在逐批扫描时只积累当前 resource 的 distinct symbol，最后过滤 instruments 再推导。info 同样按本表引用代码推导省略单位。公共质量位在非 bars 的提前返回之前检查，qualityFlags 声明另在包级检查一次，空表或没有 quality 列也不能藏非法声明。
- 本轮规范文本仅改第 9.3 节；导出器、A 股规范化验收口径和初轮真实行情汇总均保留。本轮新增测试覆盖了延迟碰撞、截取一致、无 instruments 时的历史代码、提前公告事件读取和只读主键的 as_of 过滤。
- 首批修复后 34 个定向测试通过。完整套件首轮 184 passed / 1 failed；失败的旧 count 正例声明为浮点，按审查要求改成 integer/Int64，原有断言保留，新增单独的 number/count 反例。随后第一轮基线 186 passed、27/27 变异被抓。
- 收尾自审追加四个回归，均先在待修代码确认失败：qualityFlags 的 Unicode 数字/超长位值不得令检查崩溃；合法 ISO 开盘和收盘时间中的秒不能被丢弃或拒绝。读取方和独立校验器分别实现 ISO 时间解析，未引入共享转换函数；四项修复后均通过。最终完整基线 **190 passed，49.63s**（无跳过），导入路径断言指向本 worktree；最终 **28/28 变异被抓到**，全部以断言失败退出，没有导入/收集错误充数。静态门禁和路径测试见本轮收尾结果。
- 延迟字段的额外状态为每个字段一份 T×N int64 最近来源时间数组，用于跨批次碰撞选择；其余输入仍按 65536 行批次流式处理，不保留完整长表。日历只读取较小的日期轴；扩大的历史窗口仍按分区和列裁剪。


本轮收尾结果：

- `test/spec` **190 passed，49.63s**；`test/paths` **106 passed，15.65s**，使用 2 个 worker。合计 **296 passed、0 skipped**；路径测试仅 1 条现有 Starlette/httpx 弃用警告。
- `make check` 的静态项全部通过：ruff check；ruff format（858 个文件）；cargo fmt；cargo clippy（offline、workspace/all-targets、`-j 2 -- -D warnings`，21.00s）。静态路径测试包含在 paths 中；`git diff --check` 通过。
- `bench/results/mutations_data_spec.json`：先跑完整变异 0 并打印/断言本 worktree 的导入路径，然后 **28/28** 定向变异均被抓到。新增覆盖延迟落格、panel 公告屏蔽、read 公告过滤及字段时点、unknown 拒绝、缓存身份、核对后换文件、负价掩码、扩展类型/价格域、按表引用推导单位、事件质量位及包级 qualityFlags 声明。
- 本轮只读复查此前留在 worktree/tmp 的三份真实数据导出包，全部 check 返回 0；**未重新导出、未重新与原始来源逐位对拍**。本轮性能仅投影 close、volume 两列，各包先清空进程哈希缓存测 panel，再测缓存后 panel 和 read；不清系统页缓存。汇总保存于 `bench/results/data_spec_review_read.json`，没有逐行行情。

| 既有隔离包 / 读取范围 | panel 形状 | check 秒 | panel 首次秒 | panel 缓存秒 | read 秒 | 首次/缓存后哈希次数 |
|---|---|---:|---:|---:|---:|---:|
| A 股 2024 全市场 | 242 × 5923 | 19.416 | 0.605 | 0.402 | 0.080 | 14/0 |
| 加密小时 2025 全市场 | 8760 × 4 | 4.911 | 0.214 | 0.018 | 0.009 | 13/0 |
| BTCUSDT 现货分钟 2025 | 525600 × 1 | 3.555 | 0.444 | 0.098 | 0.024 | 13/0 |

审查六项均已修复，没有新增需要审查人批准的未决事项。上述到期碰撞规则、时点 read 对 unknown 的拒绝，以及不超出已声明日历/网格补未来日期，是本轮采用并记录的具体行为；未来如扩展这些行为，应同步规范和测试。真实来源的真实性及未提供的历史修订时点仍不由标准包自行证明。导出逻辑未改，A 股已接受的规范化差异计数继续适用。

## 审查返工验收（2026-11-16）

- 先读取审查人修订的第 3.3 节；本轮只调整时间校验、字段元数据和回归验证，三个导出器未改。第 8.4 节的旧摘要同步为仅约束 bars，10.1 节不再把提前公告列为规范冲突；设计摘要同步新增 unknown。
- 新增 `test/spec/test_timing.py`，将旧的事件下界反例改为日频/日内 bars 下界反例。修改产品代码前，定向测试为 **9 failed、2 passed**，覆盖提前公告被拒、unknown 被拒、面板缺元数据；修改后 **11 passed**。
- 完整 `test/spec` 作为变异 0 重跑，**156 passed，36.51s**，无跳过；断言并打印导入的是本 worktree 的 `ashare.__file__`。
- `test/paths` 用 2 个 worker 顺序复跑，**106 passed，33.22s**，无跳过；仅两条现有 Starlette/httpx 弃用警告。本轮两个要求的目录共 **262 passed**。
- `make check` 中的静态门禁全部通过：ruff check、ruff format（853 个 Python 文件）、cargo fmt、cargo clippy（离线，workspace/all-targets，`-j 2 -- -D warnings`，1m11s）。静态路径测试已包含在 paths 中；`git diff --check` 通过。
- 变异检查 **15/15 被抓到**。原有 available_at 变异改由 bars 正反例守住；新增恢复事件下界、拒绝 unknown 语法、遗漏 unknown 警告、把警告变成失败、丢失面板 unknown 标记五处。结果见 `bench/results/mutations_data_spec.json`；所有变异目标均以测试失败退出，未用收集/导入错误充数。
- `panel()` 仍满足原 Panel/PanelSource 接口，新增只读 `field_metadata`，例如 `panel.field_metadata["close"]["knownAt"]`。源字段保留声明并展开词典默认时点；派生成交掩码按读取的掩码输入保守传播 unknown，并以 sourceFields 列出输入字段。元数据只读清单，不调用 info 的全表范围扫描；info 原有字段元数据同样保留 unknown。
- 校验警告代码是 `known_at_unknown`，值中包含 table、field、knownAt；unknown 的字典字段和扩展字段均可通过 build/check。非行情时间序列仍遵循第 3.3 节的 column:available_at 要求，公告时刻不可缺失。
- 本轮 OneContext 重试成功但无匹配。`taskpolicy -b` 因沙箱 setpriority 限制在启动测试前退出，改用 nice 启动并限制 Polars/BLAS/Rayon 各 2 线程；测试与变异顺序执行，临时副本放 worktree/tmp。
- 本轮不重新导出真实数据，沿用下方初轮实测及差异计数；A 股验收已按审查决定收敛为「有报价格子 + 规范化后的共享字段」逐位一致。

## 初轮验收记录（2026-11-15 至 16 日）

最终验收使用本 worktree 的 `.venv` 和源码，未提交或推送。测试按目录顺序执行，每组最多 2 个 worker；未同时启动多组全量测试。

- `test/spec`：最终变异 0 的完整基线 146 passed，36.37s；进程断言并打印 `ashare.__file__` 指向本 worktree。涵盖第 2–8 节独立校验正反例、第 9 节读写/原子发布/缓存/剪枝，以及第 10 节三个导出器的合成对拍。
- `test/paths`：最终单独复跑 106 passed，40.94s；包含静态路径门禁。仅两条现有 Starlette/httpx 弃用警告。
- `test/crypto`：360 passed，259.74s。
- `test/data`：944 passed、7 skipped，555.13s。7 项均为原有境外期货 Databento 适配器测试，缺少可选 `databento` 包；本次未添加跳过或放宽断言。
- ruff check、ruff format、cargo fmt、cargo clippy（`-j 2 -- -D warnings`）均通过。静态路径检查包含在 `test/paths` 中。
- 合成例子为 3 只股票 × 40 个交易日，120 条 bars，约 8 KiB；独立 check、因子表达式求值和 CSV 适配器均有运行测试。

四个要求的测试目录合计 **1556 passed、7 skipped**；跳过均如上所述来自现有可选依赖。

初轮变异检查 10/10 被杀死：时区、整数 epoch、NaN、主键唯一、价格自洽、生命期、available_at 下界、文件哈希、清单版本、日历成员。变异 0 返回 0，十个临时源码副本的目标测试均返回 1；不接受导入/收集错误冒充杀死。当前 `bench/results/mutations_data_spec.json` 记录最近一次审查修复的变异结果，见上方对应验收节。

真实数据三项均实际执行，外置盘已挂载；独立 check 均返回 0。输出只在 `tmp/data-spec-*`，入库 JSON 仅含汇总，不含逐行行情。以下为 2026-11-16 最终复跑测量，单位秒；导出合计包含适配器准备与写文件/封存，check 单独计时。

| 数据 | 导出行数 | 包大小 MiB | 准备 | 写入/封存 | 导出合计 | check | panel 首读 | panel 缓存后 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A 股 2024 全市场 | 1,293,651 | 85.442 | 1.293 | 14.045 | 15.338 | 11.272 | 0.874 | 0.551 |
| 加密小时完整冻结版 | 239,550 | 9.964 | 0.279 | 6.683 | 6.962 | 5.466 | 0.034 | 0.027 |
| BTCUSDT 现货分钟 2025 | 525,600 | 21.912 | 0.024 | 0.475 | 0.498 | 3.536 | 0.160 | 0.146 |

- A 股 panel 为 242 × 5923，对拍 25 字段；标准缺失映射后的报价格子逐位差异 0（含 NaN 位置）。**原始矩阵不是全量无差异**：23 个 lower_limit=0 转 null；无报价格子共 148,991 个字段值按规范变为 NaN，其中 adj_factor、limit_up、limit_down 各 3092，suspended 为 139,715。这个偏差保留在 JSON 中，没有放宽对报价格子的断言。
- 小时包覆盖原冻结版 2020-01-01 至 2026-10-31T23:00Z（终点不含）；panel 计时/对拍范围为 2025 全年，8760 × 4，对拍 8 字段、35,040 报价格子，逐位差异和 NaN 位置差异均为 0。
- 分钟 panel 为 525600 × 1，对拍 9 字段，逐位差异和 NaN 位置差异均为 0。
- A 股报告 12 项公司行为/复权偏差、13 项价格跳变警告；按第 8.5 节不判失败、不改源数据。两个加密包无警告。
- 三类数据都独立核对源时间轴、品种轴与 has_quote，轴一致且报价掩码差异为 0；对拍不以导出结果自己的掩码作为唯一基准。收尾自审加入了“故意漏一行”的回归：旧验收脚本误报 pass，补强后准确报告 1 处掩码差异并判失败。
- 未清空操作系统页缓存；panel 首读指本进程尚未缓存这些文件的哈希，不能视为磁盘冷读。这不是稳定微基准，受共享机器负载和数组分配影响；哈希缓存是否免重复读取由定向测试断言，不能从一次端到端耗时推断。

完整数字见 `bench/results/data_spec_validation.json`。这些测试与实测没有覆盖一份十倍规模的真实数据集；十倍规模的设计保障来自分区/列裁剪、流式填充最终面板、有限哈希缓存和落盘去重键，最终 T×N 数组本身仍随输出规模增长。

## 阶段日志

- [02:00，2026-11-16] AShare D2 — 完成独立审查的 2 项 P1、4 项 P2；先确认 25 项审查回归失败，再修时点读取、同文件哈希与缓存身份、价格/单位/事件质量约束。最终 spec+paths 296 项通过、28/28 变异被抓、静态门禁全过；三份既有真实包只读复检通过，性能及缓存摘要已记录，未重新导出或源对拍。

- [22:30] AShare D2 — make dev 完成（9m55s），季度信号 3 项回归在旧 Path(root) 实现失败后修复；新测试随后全部通过。
- [22:45] AShare D2 — 标准包、独立校验、原子构建、三个导出器与例子形成完整闭环；test/spec 129 passed，205.09s。发现并以回归固定空目标目录构建、固定单位推导两个开发错误；示例包约 8 KiB。
- [00:18，2026-11-16] AShare D2 — 落实审查决定：事件可提前公告、unknown 独立警告、面板保留字段时点元数据；旧实现 9 个回归失败确认后修复，完整 spec 156 passed，15/15 变异被抓。导出逻辑未改，A 股规范化对拍口径与差异计数写入笔记。
- [00:20，2026-11-16] AShare D2 — 本轮路径测试 106 passed，两个目录合计 262 passed；ruff/format、cargo fmt/clippy 和静态路径门禁通过。清除废弃时点写法，规范摘要与笔记同步审查决定；不提交、不推送。

## 真实源发现与规范边界

- 2024 A 股包含带前缀的交易所代码，cn 的“代码.交易所”没有六位纯数字限制；正则接受一位字母前缀加六位数字。原实现误报的 243 处已有合成回归。
- 2024 A 股有 23 个 lower_limit=0，OHLC 无不自洽记录。正价格域中的零下限没有约束作用，导出为 null，明确记录原值→标准缺失的转换；原始面板逐位比较会有这 23 格差异，不能宣称原始矩阵毫无变化。相同映射适用于 upper_limit=0。
- 季度信号显式 root 仍按 dataset_dir 的 path 参数语义直接访问；默认 root=None 才走环境根/LOCATION 指针，与 crypto.open 一致。未改变 D1 的显式路径绕过指针约定。
- 进一步自审回归：复用分区必须同时比较 schema；同版本旧清单损坏需要重新发布；显式 null/错误类型的单位与可知时点返回对应错误项，不能崩溃；没有停牌/涨跌停字段的可交易掩码只取 has_quote。

## 使用、报告与实现边界

- 公开入口：`from ashare.data.spec import seal, check, build, open, Adapter`；CLI 支持 check/seal/info/export。报告 `exit_code` 为 0/1/2，`checks` 按第 8 节顺序排列，每项含稳定 code、节号、status、count、最多 20 个 file/row/value 样例，行号从 1 开始。warning 不改变返回码。
- 描述性顶层要求 name/title/source/licenses/specVersion/symbology/calendar/resources；这是对规范示例中元数据完整性的保守解读。资源名按 Data Package 常见的小写数字、点、下划线、连字符格式；扩展字段仍严格 snake_case。
- 默认 knownAt 在读取执行及元数据展示时展开；版本按照落盘清单本身规范 JSON 计算，不增删用户未写的默认声明。end+ 时长接受正整数 us/ms/s/m/h/d/w；月份 1M 仅用于 frequency。源修改历史代码、前填、零哨兵和未披露缺口的事实，需要来源证据，不能靠一份快照证明。
- 字符串包名直接走 ext；明确路径请用 Path 对象、绝对路径或带 `./` 的相对路径。已打开句柄不自动回退到其他版本；发布后旧句柄读到变更文件会报完整性错误，应重新 open。
- build 在目标同级临时目录逐分区写入；原子目录交换使用 macOS renamex_np(RENAME_SWAP) / Linux renameat2(RENAME_EXCHANGE)，其他平台明确拒绝覆盖发布，不做“先删后改名”的降级。不变分区硬链接复用并保留 inode/mtime；schema 也是内容比较的一部分。跨进程构建锁在包目录外，单纯 check 不创建源目录文件。
- 一致性报告采用相邻 close 比值 >2 或 <0.5 的跳变阈值；公司行为对账用理论复权比值，偏差 >0.5% 给 warning。没有来源证据时不据此改行情。
- 缺行填 NaN，数值面板统一 float64，has_quote 和派生可交易掩码为 bool；不把单位/可知时点转换隐藏在因子引擎里。日频 dates 与原 Panel 一致用 YYYYMMDD，日内为表示 UTC 的 numpy datetime64[us]。
- read 先按分区剪枝再 lazy scan 投影；哈希按 1 MiB 分块，进程内 (绝对路径, 大小, mtime_ns, st_dev, st_ino, st_ctime_ns) 缓存摘要，命中后仍与当前清单的期望摘要比较，缓存最多 4096 项。check 行情每批 65536 行，跨分区键使用系统临时 SQLite；辅助小表一次加载。
- panel 直接从 lazy scan 的批次迭代器填充最终 T×N 数组（每批 65536 行），不经 read 收集整段长表；只为坐标轴做键列聚合。当前 uv.lock 为 Polars 2.1.0，批次接口属上游标注的 unstable API，依赖升级时需运行这里的流式回归。
- 第一轮合并运行四个相关测试目录，在持续推进、尚未出现失败时达到自设 840 秒上限；已停止并改按目录组顺序执行，不能把这次未完成运行算作通过。没有放宽断言或新增跳过。
- 真实性能验证按 9.1 直写封存路线分别测 export 与 check，隔离输出限定在本 worktree 的 tmp；check 失败的包保留为诊断产物但不会经 build 发布。小时线导出整个冻结版本，panel 对拍/计时用 2025 一年全市场；分钟线限定所要求的 2025 BTCUSDT 现货。
- [23:27] AShare D2 — 改按目录组后，test/spec + test/paths 共 250 passed，283.43s；包括流式面板、月频网格、v2 资源名、事件缺口和静态路径门禁。仅一条现有 Starlette/httpx 弃用警告。
- [23:35] AShare D2 — test/crypto 用 2 个 worker、按文件分配，360 passed，259.74s。cargo fmt 通过；clippy 首轮依赖构建达到 840 秒后停止，增量重跑完成（4m13s），零警告；ruff check/format 全仓通过（852 个 Python 文件）。
- [23:44] AShare D2 — test/data 完成，944 passed、7 项因可选 Databento 依赖缺失跳过；未新增跳过。CSV 例子与小时线停牌质量位回归通过，后者先确认旧导出漏掉 suspended，再补字段和掩码。
- [23:56] AShare D2 — 变异基线 145 项通过，10/10 定向变异被抓；路径测试复跑 106 项通过。三类真实数据导出、独立校验和标准字段对拍全部完成，原始 A 股差异及一致性警告单列保留；最终 ruff 与 diff 空白检查通过。
- [00:05，2026-11-16] AShare D2 — 补强验收脚本独立时间轴与报价掩码对比，故意漏行回归先失败后通过；最终基线 146 passed、变异 10/10，三类真实数据复跑均通过。汇总保留原始零限价与无报价格子的差异；未提交代码，未改写来源数据。
