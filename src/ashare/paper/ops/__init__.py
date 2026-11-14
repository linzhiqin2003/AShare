"""模拟盘每日自动运行：盘后取数、生成次日目标、开盘运行器，以及账户与策略绑定。

入口 ``python -m ashare.paper daily <阶段>``、``python -m ashare.paper account …``、
``python -m ashare.paper install-schedule …``；说明见 ``docs/PAPER_OPS.md``。

各阶段只编排已有模块：数据层的 ``sync`` / ``build`` / ``verify``，组合层的 ``top_equal`` /
``to_targets``，模拟盘的 ``LiveRunner``。不连接任何券商。
"""
