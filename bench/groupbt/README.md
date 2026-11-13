# 分组回测对照与测量

所有命令在 worktree 根目录运行。`run.py` 给子进程组设置超时，并将构建缓存放到
本 worktree 的 `.cache/`，git 禁止可选写锁。不读写其他 worktree，不提交或推送。
真实面板通过调用方设置的 `ASHARE_DATA_DIR` 只读打开，不复制或修改原数据目录。

```sh
python3 bench/groupbt/run.py 300 .venv/bin/python -m bench.groupbt.check baseline
python3 bench/groupbt/run.py 300 .venv/bin/python -m bench.groupbt.check compare
python3 bench/groupbt/run.py 14400 .venv/bin/python -m bench.groupbt.check baseline --data real
python3 bench/groupbt/run.py 3600 .venv/bin/python -m bench.groupbt.check compare --data real
python3 bench/groupbt/run.py 2400 .venv/bin/python -m bench.groupbt.mutate
python3 bench/groupbt/run.py 2400 .venv/bin/python -m bench.groupbt.review_mutations
python3 bench/groupbt/run.py 7200 .venv/bin/python -m bench.groupbt.suite
python3 bench/groupbt/run.py 3600 make check
```

`oracle/manifest.json` 记录 main 修订和逐源文件 SHA-256。冻结参考不导入生产版
`groupbt` 或新权重引擎。对照直接比较 dtype、shape、bytes，包含 NaN 位和有符号零；
不使用容差。覆盖三个池、四种中性化、两种衰减、两种成交时点、是否算权重组合、
是否带子池、IS/VALIDATION，共 384 种，全部收益、换手、暴露和派生表/图/文本均比较。
合成数据另加 32 组 193 行（3×64+1）及 300 行的长窗口，共 416 种；pytest 与存盘
对照使用同一矩阵。已有 384 组存盘须换一个新的 `--cache .cache/groupbt-review` 目录。
测试另覆盖向量中性化、5 组、长缺价、宽截面、空窗口和失败行号。

`review_mutations` 分别把工作块末端改为 `begin + block - 1`、把八路成对求和改为
顺序累加；要求全输出逐位对照或导入自检明确失败，Rust 单测也必须失败，最后恢复
源码与两个扩展并重新通过合成矩阵。详细日志默认保存在 `.cache/groupbt-review/`。

基准按参数编号原子写入 `.cache/groupbt/{synthetic,real}`，中断后可接着存盘；
`identity.json` 守住源文件、参数、面板版本、因子字节和 NumPy 版本。早期中断生成的
存盘在加入 identity 前已完成全量对照；首次复跑会补写身份信息。旧有存盘不冒充本次
重新计算，摘要分别记录 `cached` / `computed`。环境或输入变化时须另指定 `.cache/`
内的新 `--cache` 目录。比较失败只显示字段和最大误差，不打印行情值。

`simulation.py` 以冻结源代码隔离重建“只替换模拟”的阶段，不修改参考模块；
`--implementation simulation` 可检验它，`measure --backend simulation` 可测量它。
`suite` 逐进程交替测 main、模拟替换版和最终版，再做剖析、线程扩展性和四组场景。
每个计时进程做三次调用，冷调用和后续调用均保留；线程表每格同时运行 8 组，含
每组的样本内及验证。内部 Rayon 也使用线程，外部线程数不等于 CPU 工作线程总数。

时间不含面板读取和表达式求值；峰值 RSS 包含整个进程的输入和表达式准备高水位。
tracemalloc 不跟踪 Rust 的全部分配，不能当作总内存。四组场景选定 20/60 日波动率
与 industry/cap+industry 两种中性化；分别测 1、2、4 个外部线程。
原始剖析和真实输出只留 `.cache/`。公开报告只写性能和逐位一致数量，不写行情值。

扫描共享这一轮另有以下工具（同样先由 `run.py` 设置超时和本地缓存）：

- `python -m bench.platform.bitwise baseline|compare -o .cache/groupbt-fusion/platform.md`：
  经 HTTP API 跑 4/8 变体和调仓扫描，等待全部持仓写完，比较每个数组 dtype、shape、bytes
  和原始摘要 JSON。UUID 与压缩容器时间戳不作为数值输出；按 ordinal 对齐独立空库。
- `python -m bench.groupbt.ablation -o .cache/groupbt-fusion/ablation.json`：分别开关行情准备
  共享、均值缓存与目标合批，同进程交替计时，包含原始内核输出的逐位断言。
- `python -m bench.platform.evaluation -o .cache/groupbt-fusion/evaluation.json`：独立求值
  与公共子式共享的交替计时、分配量及信号逐位比较。
- `python -m bench.groupbt.shared_mutations -o .cache/groupbt-fusion/mutations.json`：8 种
  共享键、输入别名、表达式窗口、暴露归约和调仓日历变异，恢复源码与扩展后再次通过测试。
- `python -m bench.platform.grid --output .cache/groupbt-fusion/grid`：依次运行
  `bench/platform/speed.py` 的并发 × 算子线程组合；可用 `--reference` 指向 `.cache`
  内保存的旧版 `src/ashare` 和 `panelbt` 包，交替测前后版本。不要与其他测试同时计时。

`shared.py` 的 `--reference` 是修改前复制到 `.cache` 的 Python 包与编译扩展（分别名为
`groupbt_a5`、`panelbt_a5`）；它用于隔离旧版与当前版的单组、批量配对测量，不能在修改
之后用当前包覆盖参考。参考文件和行情结果均不加入 git。
