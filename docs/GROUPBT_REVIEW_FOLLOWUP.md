# 分组回测复核收尾

基于 `4129258`，本轮改动留在工作区，未提交、未推送。冻结的 main 参考源及其摘要未改。

## 1. 工作块边界

合成矩阵由 384 组扩为 416 组。新增 32 组覆盖四种中性化、open/close、是否计算权重
组合，使用 ALL 并带两个子池，decay=3。窗口分别为 193 行（3×64+1）与 300 行，后者
还跨过 240 日风险历史。pytest 和存盘对照共用 `cases(long_windows=True)`；另一个测试
直接核对窗口实际行数，防止以后改合成日期又缩回单块。真实数据矩阵仍是原来的 384 组。

重新生成冻结 main 基准：416/416，52.848 秒；逐位对照：416/416，9.467 秒。
耗时包含快照、文件读写与派生表/图/文本，不代表单次 `groupbt.run` 的耗时。
比较 dtype、shape 和原始字节，包含 NaN 位与有符号零，没有放宽断言。

变异验证将 `end = (begin + block).min(t)` 改成 `(begin + block - 1).min(t)`。
原 384 组仍通过，第一组新增长窗口在 `group_pnl` 的逐位断言失败；新增 Rust 边界单测
也失败。源码和扩展恢复后重新通过完整 416 组矩阵（pytest 35.660 秒）。

## 2. GIL 释放期间的输入

`groupbt.run` 复制有效窗口的因子。`panelbt.run_group_weights` 与 `run_weights_batch`
将实际传给 Rust 的数组临时设为只读，在正常或异常退出时恢复原标志。同一数组重叠借用
采用引用计数，最后一个调用结束才恢复；数组和其父数组都被借用时按依赖顺序恢复。
已只读数组保持只读；若旧的可写视图指向已只读父数组，则先复制该视图，避免无法恢复
可写标志，同时保持父数组原状态。

NumPy 的只读标志属于数组对象，不是内存锁，不能阻止预先存在的其他别名写入。
因此 docstring、设计文档和 README 明确要求调用期间不要修改输入或其别名。

`test_weight_inputs.py` 覆盖重叠线程、重复参数、父子视图、原本只读、正常及异常返回。
另在原生入口处检查全部传入视图不可写，并从另一个线程改写原 factor，验证窗口副本
与事先结果仍逐位一致。输入保护及导入自检的专项测试共 12 项通过。

## 3. NumPy 求和次序自检

导入 `panelbt` 时，通过私有原生入口调用生产使用的同一个 `numpy_exact::sum`，
与 `np.add.reduce` 逐位核对 32 条探针。长度为 1、7、8、9、127、128、129、1000，
每种长度含抵消序列、混合数量级、全负零及 NaN。任意字节不一致即抛 `ImportError`，
明确说明 NumPy 求和次序改变、必须更新 Rust 内核。英文和中文 README 均说明依赖的是
NumPy 内部实现，并非 NumPy 对归约次序的稳定承诺。

单独测自检函数：7 轮，每轮 1000 次，中位 0.064 毫秒/次，范围 0.061–0.067 毫秒。
这不包含 Python、NumPy 和扩展模块本身的导入时间。

Python 测试分别改坏有限结果的一位、零符号和 NaN payload，确认重新导入失败。
源码变异另把 Rust 八路成对合并改为顺序累加，新进程 `import panelbt` 按预期明确失败，
Rust 固定常量测试也抓到，排除仅因源码摘要过期而失败的情况。两项源码变异检查共耗时
181.842 秒，包含变异前后测试与构建，最终源码、两个扩展均已恢复。

## 4. NumPy 版本

依赖声明、Python 导入检查及 `numpy_dot.rs` 报错统一为 NumPy >= 2.1。
原生 C API 检查同步使用 `NPY_2_1_API_VERSION` 对应的 `0x00000013`。
测试核对包依赖、原生错误文本，并模拟 2.0.2，确认导入检查拒绝。

## 5. 中间阶段代码

- 删除无调用方的 `risk_exposure_from_prices` 和 `api.PRICE_TOL`。
- 删除 `risk_exposure`、`group_demean` 的 32 行递归分块。函数本身仍被指标/权重手算及
  随机测试调用，作为 NumPy 参考保留，注释明确不在 `run` 热路径。
- `build_alpha` 只有 `build_weights` 调用，已内联并删除该中间函数。
- `build_weights` 仍被 `test_groupbt_weights.py` 的顺序、随机、错误及输入不变性测试
  调用，按复核要求保留并说明用途。全仓搜索包含 bench、test；本工作树没有 research
  目录。冻结 oracle 的同名实现未改。

新增 97 行参考回归，将保留的三个 NumPy 函数与冻结 main 直接逐位对照，覆盖原 32 行
分块边界。原有测试没有删除或放宽。

## 6. 提交记录

P-017、P-018、P-019、P-020 的“提交”均改为 `4129258`；`PERF_LOG.md` 总表新增提交列，
每项都标明“四项一起”。这些是已提交优化的归属，本轮收尾改动仍未提交。

## 7. Rust 单测与完整检查

新增 9 个 Rust 单测，由普通 `cargo test` 运行：

- `weights.rs`：买卖方向与仓位符号的区别；受限方向冻结、相反方向允许；多空分别占用
  预算；冻结仓位超过新预算时自由仓位不反向。
- `group_weights.rs`：63/64/65/128/129/193/300 行逐日校验收益、换手及暴露，两个池
  输出一致；无成交行的末信号仍校验；空窗口。
- `numpy_exact.rs`：固定手算常量守住八路合并、128/129 递归分界、负零和 NaN；
  稳定并列分组与缺失；组内缓冲复用及暴露。

最终 `make check` 退出码为 0：Ruff 检查及格式、Rust fmt、Clippy（warnings 视为错误）
全部通过；全 workspace 的 31 个 Rust 测试通过，其中两个相关内核 crate 共 21 个。
Python 为 **3942 passed、7 skipped、21 warnings，571.76 秒**。7 个跳过均来自原有
Databento 可选依赖未安装，本轮没有新增跳过或放宽断言。两项源码变异均由 Python 与
Rust 检查捕获，恢复后 416/416 合成矩阵通过。

## 复跑与记录

```sh
python3 bench/groupbt/run.py 300 .venv/bin/python -m bench.groupbt.check baseline --cache .cache/groupbt-review
python3 bench/groupbt/run.py 300 .venv/bin/python -m bench.groupbt.check compare --cache .cache/groupbt-review -o .cache/groupbt-review/bitwise-summary.txt
python3 bench/groupbt/run.py 2400 .venv/bin/python -m bench.groupbt.review_mutations -o .cache/groupbt-review/mutations.json
python3 bench/groupbt/run.py 3600 make check
```

原始基准、日志和计时保存在忽略的 `.cache/groupbt-review/`。本轮未重跑真实面板对照，
也未重测整组耗时、RSS、线程扩展性；原性能报告的数字保留其原测量口径，内存未达 1.5 GB 的
遗留项仍见 [GROUPBT_PERF_REPORT.md](GROUPBT_PERF_REPORT.md)。
