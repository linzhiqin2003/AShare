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
python3 bench/groupbt/run.py 7200 .venv/bin/python -m bench.groupbt.suite
python3 bench/groupbt/run.py 3600 make check
```

`oracle/manifest.json` 记录 main 修订和逐源文件 SHA-256。冻结参考不导入生产版
`groupbt` 或新权重引擎。对照直接比较 dtype、shape、bytes，包含 NaN 位和有符号零；
不使用容差。覆盖三个池、四种中性化、两种衰减、两种成交时点、是否算权重组合、
是否带子池、IS/VALIDATION，共 384 种，全部收益、换手、暴露和派生表/图/文本均比较。
测试另覆盖向量中性化、5 组、长缺价、宽截面、空窗口和失败行号。

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
