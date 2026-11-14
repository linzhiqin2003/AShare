"""独立参考实现（pandas 写法，不经过 ``Expr`` 和 ``panelops``），供 ``verify`` 核对候选表达式。

每个参考模块提供 ``compute_many(frames, names) -> {名字: ndarray[T, N]}``；输入约定见
``docs/REFERENCE_INPUTS.md``。
"""
