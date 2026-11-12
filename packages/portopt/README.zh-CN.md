# portopt

仅依赖 numpy 的数组内核与 Clarabel 绑定，不导入 ashare、panelbt 或 panelops。
接口与数组单位详见英文 README；回归损失权重为市值平方根，行业约束用市值。

`ew_moments_segment` 给出窗口内一段连续行的逐列指数加权矩（只加不减），`ew_moments_variance`
由矩还原出与 `ew_variance` 相同定义的 Newey-West 方差，并给出各项绝对值之和，用于判断抵消程度。

`solve_many` 并行处理独立问题；有前期持仓依赖的调仓序列由上层顺序求解。
输入数组在 Rust 调用期间不得被其他线程修改。`threads=0` 使用 `set_num_threads(n)` 设置的进程默认值；设置 0 恢复硬件默认。线程池按线程数复用。
构建及检查：在工作区执行 `make dev && make check`。

`get_num_threads()` 返回当前默认线程数；`available_threads()` 返回硬件可用线程数。
病态回归统一走约束零空间的 SVD 最小范数解，全量与单日使用同一个内核。
Rust 库产物名为 `portopt_ext`，Python 导入路径仍为 `portopt._core`。
