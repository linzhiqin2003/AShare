# 三套因子库原文抄录报告

日期：2026-11-13。本目录只做抄录，不翻译、不改写、不纠错。

## 一、取数方法（为什么可信）

- **所有原文都用 curl 直接下载原始文件后用自己写的脚本解析**，没有经过网页读取工具的模型转述：论文和研报是 PDF，用 `pdftotext -layout` / `mutool` 抽取文本层；GitHub 上的文件取 raw 原文。
- 比对规则：先做 NFKC（全角半角统一）、把 en dash/em dash/减号统一成 `-`、删除全部空白，再逐字符比较；大小写不归一。
- PDF 中公式跨行的地方：Alpha101 按行拼接时用一个空格连接（论文换行都落在空格处）；国君 191 的表格单元格会在词中间断行（如 `DELAY(CL` + `OSE`），拼接时不加空格，原有空格保留，连续空格压成一个。**因此 gtja191 里 `formula` 的空格位置不可靠，只有去空白后的字符序列可靠**（研报是两端对齐的表格，PDF 本身也没有保留原始空格）。
- Alpha158 的展开：没有运行 qlib 代码。用 Python `ast.parse` 只解析源码文本，取出每个 `if use("…")` 块里的格式字符串和名称模板，再由我自己的脚本做 `%` 代入（窗口 5/10/20/30/60，price 窗口 [0]、字段 OPEN/HIGH/LOW/VWAP）。脚本（一次性，未入库）：`expand158.py`。
- 抽查时另外用网页读取工具（WebFetch）取了几条：它对超过 125 字符的内容会自动截断，所以只能做前缀比对，不能作为主来源。

## 二、条数与一致率

| 套 | 条数 | n 连续 | 来源一 | 来源二 | 一致 | 不一致 | 单一来源 |
|---|---|---|---|---|---|---|---|
| Alpha101 | 101 | 1–101 无缺无重 | arXiv PDF 附录 A.1 | yli188/WorldQuant_alpha101_code 代码注释 | 101/101 | 0 | 0 |
| GTJA191 | 191 | 1–191 无缺无重 | 研报 PDF 表 6 | JoinQuant 文档（GitHub 镜像） | 170/191 | 21（其中 12 条只是乘号写成 ×） | 0（每条都有两个来源；见下文说明） |
| Alpha158 | 158 | 1–158 无缺无重 | qlib main `loader.py` | qlib v0.8.6 `handler.py` | 158/158 | 0 | 0 |

所有 450 条 `formula` 非空；三套原文公式的括号全部配对（半角与全角括号合并计数；没有出现右括号先于左括号）。

各字段含义：`formula` = 来源一原文；`formula_alt` = 来源二原文；`agree` = 两者去空白后逐字符相同。Alpha101 另有 `formula_alt2`/`agree_alt2`（JoinQuant Alpha101 文档镜像），GTJA191 另有 `formula_alt2`/`agree_alt2`（kangchihlun 的 GTJA_Alpha191.py gist 代码注释，只有 90 条有公式注释）。Alpha158 另有 `group` 字段。

### Alpha158 分组计数（展开后核对）

kbar 9；price 4（OPEN0、HIGH0、LOW0、VWAP0）；rolling 29 个算子 × 5 个窗口 = 145：ROC、MA、STD、BETA、RSQR、RESI、MAX、MIN（源码开关名是 `use("LOW")`）、QTLU、QTLD、RANK、RSV、IMAX、IMIN、IMXD、CORR、CORD、CNTP、CNTN、CNTD、SUMP、SUMN、SUMD、VMA、VSTD、WVMA、VSUMP、VSUMN、VSUMD 各 5 条。合计 9 + 4 + 145 = 158。默认配置取自 `qlib/contrib/data/handler.py` 的 `Alpha158.get_feature_config`（main 与 v0.8.6 相同：kbar、price windows [0] 字段 OPEN/HIGH/LOW/VWAP、rolling 默认，没有 volume）。

## 三、不一致清单

### Alpha101

论文与 yli188 注释 101 条全部一致。第三来源 JoinQuant 镜像有 2 条不同（已写进对应行的 note）：

- 021、024：论文 `||`，JoinQuant 镜像写作 `or`。

### GTJA191（研报 vs JoinQuant）

**只差乘号写法（JoinQuant 用 `×`，研报用 `*`），12 条**：078、079、080、082、083、085、086、091、092、096、098、099。其中 078、079、080 的 gist 注释与研报逐字符一致。

**内容不同，9 条**（两个版本都保存在行内，未判断对错）：

| n | 差在哪 | gist 注释 |
|---|---|---|
| 002 | JoinQuant 为小写，外层少一对括号，分母写作 `((high-low))` | 与 JoinQuant 相同，与研报不同 |
| 005 | JoinQuant 把第二个 `TSRANK` 写成 `YSRANK` | 无 |
| 031 | JoinQuant 开头少一个 `(` | 无 |
| 039 | JoinQuant 结尾少一个 `)` | 无 |
| 042 | JoinQuant 开头少一个 `(` | 无 |
| 075 | JoinQuant 镜像缺开头 `COUNT(CLOSE>OPEN &`（另有 `BANCHMARKIN DEXOPEN` 中间空格） | 与研报一致 |
| 123 | JoinQuant 镜像在 `<` 处断掉，只剩前半段加一个反斜杠 | 与研报一致 |
| 135 | JoinQuant 开头多一个负号 `-` | 与研报一致 |
| 167 | JoinQuant 结尾少一个 `)` | 小写且括号写法不同，与两者都不一致 |

另外 190：JoinQuant 标注「公式有部分缺失 有调整」，给出的「原公式」与研报一致（`agree: true`），「修改后公式」把 `DELAY(CLOSE)` 改成 `DELAY(CLOSE,1)`，已记在 note。030、165、183 在 JoinQuant 标「尚未实现」，但文档里仍给出了公式且与研报一致。

**关于独立性**：gist 的 002 与 JoinQuant 逐字相同、与研报不同，说明 gist 很可能抄自聚宽版本，不能当作完全独立的第三来源。JoinQuant 文档本身声明「有对缺失部分和不合理部分的因子公式进行调整」。只有研报 PDF 是一手原文；JoinQuant 与研报不一致、又没有 gist 佐证的 13 条（005、031、039、042、082、083、085、086、091、092、096、098、099）可以视为「研报单一来源」。其中除 005、031、039、042 外都只是乘号写法不同。

### Alpha158

main（与 v0.9.8、v0.9.7 逐字节相同）和 v0.8.6 的展开结果 158 条名称和表达式完全一致。**注意两个来源同出 microsoft/qlib 仓库**，没有找到仓库以外列出全部 158 条的可读页面（BigQuant 的 Alpha158 页面需要脚本渲染，取不到正文）。

## 四、原文中注意到的疑点（只描述，不修正）

### Alpha101（也写在对应行的 note 中）

- 附录小节编号：引言说公式在 A.1、函数在 A.2、输入数据在 A.3；正文标题实际是「A.1. Formulaic Expressions」「A.1. Functions and Operators」「A.2. Input Data」（两个 A.1）。
- 附录定义 `min(x, d) = ts_min(x, d)`、`max(x, d) = ts_max(x, d)`，但 071、073、076、087、096 的 `max` 和 077、088、092 的 `min` 两个参数都是序列。
- 059（vwap 两次）、066（low 两次）、082（open 两次）、089（low 两次）：加权组合 `(x * w) + (x * (1 - w))` 两项是同一个字段。
- 084：`SignedPower` 的指数是序列 `delta(close, 4.96796)`。
- 100：`indneutralize` 嵌套两次，都是 `IndClass.subindustry`。
- 007：`(-1 * 1)` 原文如此。056 使用 `cap`。
- 大量窗口为非整数（如 `16.1219`）；附录写明 `non-integer number of days d is converted to floor(d)`。
- 函数名大小写混用（`Ts_Rank`/`ts_rank`、`IndNeutralize`/`indneutralize`、`Sign`、`Log`）；附录写明 `All expressions are case insensitive`。
- 用到的 adv 窗口：5、10、15、20、30、40、50、60、81、120、150、180。

### GTJA191（也写在对应行的 note 中）

| n | 疑点 |
|---|---|
| 010 | 按括号位置，`,5` 是 RANK 的第二个参数；MAX 只有一个参数 |
| 017、091、108、121、127、154 | `MAX/MIN(x, 数字)`，而附录 `MAX(A, B)`/`MIN(A, B)` 定义为两者取大/取小（TSMAX/TSMIN 才是时序） |
| 021、147 | `REGBETA` 只有两个参数（附录为三个） |
| 116 | `REGBETA(CLOSE,SEQUENCE,20)` 中 `SEQUENCE` 无参数 |
| 022 | `SMEAN` 未定义 |
| 023 | 第一处写作 `STD(CLOSE:20)`（冒号） |
| 028 | 第二个 SMA 内分母为 `(MAX(HIGH,9)-TSMAX(LOW,9))`，与前项 `(TSMAX(HIGH,9)-TSMIN(LOW,9))` 写法不同 |
| 030 | `DELAY(CLOSE)` 无 n；REGRESI 有 4 个序列参数；`MKT` 未定义（附录写 `HML SMB MKE`）；`，60` 是全角逗号 |
| 036 | `CORR` 无 n；`RANK` 带第二个参数 |
| 052 | `-L)` 中 `L` 未定义 |
| 054 | `STD(ABS(CLOSE - OPEN))` 无 n |
| 011、060 | MATLAB 式 `./`、`.*` |
| 069 | 全角 `？` `：` |
| 078 | `MA` 未定义 |
| 111 | `VOL` 未定义 |
| 124 | `DECAYLINEAR(RANK(TSMAX(CLOSE, 30)),2)` 原文如此 |
| 127 | `MEAN(...)` 无 n |
| 131 | `DELAT` |
| 143 | 使用 `SELF`（附录定义为上一期本因子值） |
| 146、173 | 末尾带分号 |
| 159 | `HGIH`（三处） |
| 165、183 | `SUMAC` 只有一个参数（附录为两个）；`MAX`/`MIN` 只有一个参数 |
| 166 | 全角括号 `（20-1）`；`(20-2)(SUM(` 之间无运算符；`(CLOSE/DELAY(CLOSE,1),20)^2` 原文如此。这一行在 PDF 里 `-20*` 与 `（20-1）` 排成上下两行，抽取顺序按 mutool 给出的阅读顺序，与 JoinQuant 一致 |
| 181 | 第二个 `SUM` 无 n |
| 182 | 用 `OR`（附录为 `||`） |
| 188 | `LOW–SMA` 中是 en dash（U+2013）。比对时已统一，`formula` 保留原字符 |
| 190 | `DELAY(CLOSE)` 无 n（多处） |
| 075、149、181、182 | 字段拼写 `BANCHMARKINDEX…`（原文如此） |

函数定义表（`gtja191_defs.md`）里的问题：`BANCHMARKINDEXCLOSE` 被写成「基准指数的开盘价」、`BANCHMARKINDEXOPEN` 被写成「收盘价」（与名称相反，JoinQuant 版本与名称一致）；`LOWDAY` 的说明与 `HIGHDAY` 相同，都是「最大值」；`COVIANCE` 拼写；DELTA、DELAY、SIGN、SMA 四行数学式在 PDF 里用私有区字形，文本层取不到，原样保留了残缺的抽取结果，并附上 JoinQuant 版本。

### Alpha158

- `MIN5…MIN60` 由 `if use("LOW"):` 开关生成（名称 MIN，开关名 LOW）。
- ROC 的注释写「price change … divided by latest close」，表达式是 `Ref($close, d)/$close`（不是差值）。
- `Rank` 在 v0.8.6 中是 `percentileofscore(...) / len(x1)`，在 main 中是 pandas `rank(pct=True)`（或 `/ 100`）；两个版本的 RANK5…RANK60 数值口径不同。原文都抄在 `alpha158_defs.md`。

## 五、定义文件的来源情况

- `alpha101_defs.md`：只有论文一个来源（yli188 只抄了公式；JoinQuant 的中文说明是转述，未收入）。
- `gtja191_defs.md`：研报附录 2 原文 + JoinQuant「公用函数说明」原文，两份都完整抄录，差异见上。
- `alpha158_defs.md`：ops.py（main）各相关算子 docstring 与实现原文、rolling.pyx 的 Slope/Resi/Rsquare、base.py 运算符重载、handler.py 默认 conf、v0.8.6 的 Rank。开头的「要点」一节回答了 Ref 的方向、窗口含当日、Rank 是时序分位、Slope/Rsquare/Resi 的自变量，每条都引用了下方原文。
- `alpha158_loader_source.py.txt`：main 版 `qlib/contrib/data/loader.py` 原文，与 v0.9.8 tag 逐字节相同。

## 六、自查与抽查

自查（脚本）：三份 JSONL 行数 101 / 191 / 158；`n` 连续、不重不漏；`formula` 全部非空；括号全部配对。

抽查（随机种子 20261113，每套 10 条，重新下载来源后用另一种方式抽取再比对）：

- Alpha101 [2, 3, 9, 13, 17, 23, 47, 59, 66, 82]：重新从 https://arxiv.org/pdf/1601.00991 下载（sha1 与首次 e-print 相同），改用 mutool 抽取：10/10 一致。另用 WebFetch 读 yli188 原文件取 2、59、82：2 全文一致；59、82 被工具截断在约 120 字符，前缀一致。
- GTJA191 [8, 13, 14, 29, 77, 79, 101, 128, 142, 154]：重新下载研报（sha1 相同），改用 `pdftotext -layout` 抽取（首次用 mutool），去掉标签和空白后检查子串：10/10 一致。另用 WebFetch 读 JoinQuant 镜像取 77、128、154：77、154 全文一致；128 被截断，前缀一致。
- Alpha158 [4, 13, 67, 73, 78, 86, 101, 112, 125, 130]：重新取 v0.9.7 tag 的 loader.py 展开：10/10 一致（全部 158 条也与 main 相同）。另用 WebFetch 读 main loader.py 的 RSV、CORD、WVMA 模板，代入后与文件中的 5 和 60 窗口逐字一致。

## 七、没取到的来源

- arXiv 1601.00991 只提供 PDF，没有 TeX 源；ar5iv 没有 HTML 版（重定向回摘要页）。
- JoinQuant 官方帮助页（https://www.joinquant.com/help/api/help#name:Alpha191）是脚本渲染的，取不到正文，改用 GitHub 上的文档镜像 lzwme/finance-quant-skills。该镜像有截断和转义问题（075、123 截断，`\<` 转义；`\<` 已在 `formula_alt` 中还原为 `<`）。
- BigQuant 的 Alpha158 页面（https://bigquant.com/wiki/doc/nODcNAKYPJ）需要脚本渲染，未取到。
- DolphinDB gtja191 文档：搜索没有找到可读的公式页面，未继续尝试。
- shrektan/techfactor 的 C++ 注释改写过公式（如 Alpha166 写成 `let dr = …`），popbo/alphas 的注释只覆盖少数因子，两者都没有作为比对来源。

共用了 4 次网页搜索，没有访问任何需要登录的网站，没有运行任何下载来的代码。
