# 数据存放位置

`ashare.paths` 是本地数据位置的统一入口。没有配置重定向时，保留各入口原有的默认位置：股票、ML、挖掘和模拟盘运维以仓库为根；平台 `platform_dir()`、平台模拟盘 `paper_root()` 和 crypto 示例基准以 cwd 为根。端到端、规则和 ML 基准保留无本地面板时的主检出回退。外置盘上的真实路径只写入被 git 忽略的 `LOCATION.json`，不要把该文件或行情数据加入版本控制。

## 查看

在已执行 `make dev` 的检出里运行：

```sh
.venv/bin/python -m ashare.paths status
```

CLI 默认检查仓库数据根。需要查看原本相对 cwd 的目录时，可运行 `python -m ashare.paths --root data status`；设置 `ASHARE_DATA_DIR` 则能让各入口使用同一根。

输出每个登记数据集的有效位置、文件总字节数、是否可读、是否经指针重定向。普通未初始化目录显示不可读；失效或损坏的指针同时显示原因，退出码为 1。大小按有效位置统计，不重复计入默认保留的内置盘副本；这是读取所有目录项的检查，大数据集可能需要一些时间。

## 登记表

| 名称 | 属性 | 内容 / 选择依据 |
|---|---|---|
| `cn` | 热 | A 股、境内期货、基金及可转债 |
| `live` | 热 | 实时录制 |
| `paper` | 热 | 模拟账户和 `.ops` |
| `platform` | 热 | 平台结果与服务日志 |
| `mining` | 热 | 私有因子台账、搜索记录 |
| `research` | 热 | 研究产物 |
| `dev` | 冷 | 开发与基准缓存，包括 `ml/`、`futures/` 等子目录 |
| `crypto` | 冷 | 加密货币原始库与面板 |
| `crypto_minute` | 冷 | 加密货币分钟数据，无专用环境变量 |
| `crypto_metrics` | 冷 | U 本位 5 分钟合约指标、日度原始归档和月分区，无专用环境变量 |
| `ml` | 热（保守默认） | 原有 ML 公共入口的缓存和账本 |
| `hk`、`us` | 热（保守默认） | 港股、美股独立市场库 |
| `foreign` | 热（保守默认） | 境外期货导入数据 |
| `fixtures` | 热（保守默认） | 不入库的数据夹具 |

表的代码来源是 `ashare.paths.DATASETS`。未登记的名字报 `DataLocationError`，不能传 `dev/ml` 作为数据集名。子目录写成 `dataset_dir("dev") / "ml"`。所有热数据集都受搬迁保护；设计未明确温度的现有名称也先按热数据处理。

## 解析顺序和兼容变量

| 优先级 | 输入 | 含义 |
|---|---|---|
| 1 | `dataset_dir(name, path)` 的显式路径 | 完整数据集目录，直接使用 |
| 2 | 下表的专用兼容变量 | 完整数据集目录，直接使用 |
| 3 | `ASHARE_DATA_DIR` | 替换整根 `data/`，不是 `cn/` 或 `crypto/` 本身 |
| 4 | 所选根下 `<name>/LOCATION.json` | 读取 `path`；目标不存在即报错 |
| 5 | 所选根下 `<name>/` | 没有指针时使用原目录 |

第 3 级的“整根替换”意味着只查询新根下的指针，旧根下的指针不再参与；新根仍支持自己的 `LOCATION.json`。显式数据集路径和专用变量跳过指针，适合隔离副本或测试。空环境变量视为未设置，路径支持 `~`，通常规范化为绝对路径；兼容 cwd 的入口使用 `cwd_default=True`，未重定向时保留原有相对路径表示。指针目标始终为绝对路径。

| 变量 | 对应位置 | 兼容行为 |
|---|---|---|
| `ASHARE_DATA_DIR` | 整根目录 | 所有登记数据集共享的根 |
| `ASHARE_ML_DIR` | `ml`；ML 基准中覆盖完整 `dev/ml` 路径 | 保留原有两种使用场景，不额外追加 `ml` |
| `ASHARE_MINING_DIR` | `mining` | 完整台账/搜索根 |
| `ASHARE_PAPER_DIR` | `paper` | 完整账户根 |
| `ASHARE_PLATFORM_DIR` | `platform` | 完整平台结果目录，不额外追加 `results` |
| `CRYPTO_DATA_ROOT` | `crypto` | 完整 crypto 根；现在命令行和 Python 读取入口均适用 |
| `ASHARE_BENCH_DATA` | 基准脚本的整根目录 | 优先于 `ASHARE_DATA_DIR`；只作用于基准入口 |

股票 `ashare.data.open(root)` 的显式 `root` 保持“整根目录”语义：通过 `dataset_dir("cn", root=root)` 访问该根下的市场及指针。`root=` 优先于环境根和专用变量，但不跳过所选根下的指针；要直接指定一个数据集目录，用 `path`。模拟盘的显式 `data` 根同理；显式 `paper` 参数优先。

基准通过 `bench_data_root(checkout=...)` 保留主检出回退：显式根、`ASHARE_BENCH_DATA`、`ASHARE_DATA_DIR` 先于自动选择；本检出有 `cn/LOCATION.json` 时始终尊重指针（失效即报错），没有指针且没有本地股票面板时才查 git 公共目录。ML 基准的 `dev/ml` 跟随同一次选出的根，再解析 `dev` 指针。

```python
from ashare.paths import dataset_dir

cold_cache = dataset_dir("dev") / "futures"
crypto_root = dataset_dir("crypto")
new_local = dataset_dir("research", create=True)
```

`create=True` 只在有效位置没有指针时创建目录。发现指针后绝不创建目标目录；已有目标里的业务子目录可由业务代码正常创建。

## 搬迁

先停止该数据集的写入任务。外置盘和目标父目录须已存在；目标本身可以不存在，也可以是空目录。目标目录可自由命名，无需与数据集同名。以下 `<外置盘目录>` 是占位符，使用时替换为已挂载磁盘的实际目录：

```sh
.venv/bin/python -m ashare.paths move dev '<外置盘目录>/dev'
.venv/bin/python -m ashare.paths move crypto '<外置盘目录>/crypto'
.venv/bin/python -m ashare.paths status
```

流程为复制（保留文件修改时间）、逐文件核对大小和 SHA-256、确认源未发生内容变化、原子发布指针。目录清单和空目录也核对。默认保留源文件，新读取立即改走目标；如果希望核对成功后立即回收源空间，在同一次 `move` 中加 `--delete-source`，源目录最后只剩 `LOCATION.json`。

```sh
.venv/bin/python -m ashare.paths move dev '<外置盘目录>/dev' --delete-source
```

没有“先移动再重跑加删除”的隐式行为：已有指针的再次搬迁会拒绝。默认保留后想删除源副本，应先人工确认指针和目标，再只清理源中的非指针文件。热数据必须显式加 `--force`；定时服务依赖的热数据仍建议留内置盘。

工具拒绝非空目标、源树里的符号链接/特殊文件、与数据根互相包含的目标、缺失的目标父目录和已存在的搬迁锁。兼容变量正在覆盖该数据集时，默认拒绝搬迁，避免写了指针却仍被旧变量遮住。搬迁失败且指针尚未发布时，清掉不完整目标；原本存在的空目标恢复为空。清理以磁盘上指针是否已经发布为准：改名后即使马上收到 Ctrl-C 也保留完整目标。发布边界中断不开始删除源文件；已经进入 `--delete-source` 清理阶段后的中断可能留下部分源副本，完整目标和指针仍保留。

指针结构包含 `path`（绝对目录）、`moved_at`（ISO 时间）、`files`（文件数）、`bytes`（总字节数）。四个字段均必填，格式和类型会校验。数量是搬迁当时的记录，不是每次读取时重新计算的校验和。

## 外置盘未挂载

例如指针指向一个未挂载的位置，业务入口和 CLI 会报类似：

```text
DataLocationError: 数据集 crypto 的目标目录不可用，期望路径：<外置盘目录>/crypto；请挂载外置盘，禁止回退或自动创建
```

JSON 损坏或字段缺失则报“数据集 crypto 的指针无效”，附指针位置和原因。平台 HTTP API 将位置错误呈现为 503 和“数据位置不可用”，详细路径记录于服务日志；模拟盘参考数据读取向上传播 `DataLocationError`。失败不缓存，挂盘后下一次请求即可重试。挂载正确磁盘后重试；不要删除指针来“修复”报错，这会改变数据来源。

## 调度服务与开发验证

平台 `make platform-service` 用 `bench/platform/service.py` 生成路径；常驻启动带 `--service-results`，保留原来的 `platform/results` 布局。模拟盘 `install-schedule` 同样通过统一入口解析路径。默认位置不固化为专用环境变量，每次启动重新解析指针；显式配置的专用变量照常保留。launchd 的标准输出日志路径是安装时生成的，调整热数据位置后需重新生成服务定义。本任务没有安装、加载或重启任何服务。

测试只使用临时目录，不依赖真实行情或真实外置盘。定向检查：

```sh
.venv/bin/python -m pytest test/paths -q
.venv/bin/python bench/mutations_data_paths.py
```

变异脚本先确认导入当前检出的代码，再在临时源码副本里做十项变异（原五项加三个 P1 的五个故障点）；报告写入 `bench/results/mutations_data_paths.json`。静态测试扫描 `src/`、`bench/`，只豁免币安远端对象键和三个临时合成夹具里的精确表达式；文档字符串不作为运行路径扫描；门禁也识别环境变量默认值 `"data"` 和 `os.path.join("data", ...)`。验收详情和限制见 [D1_NOTES.md](D1_NOTES.md)。
