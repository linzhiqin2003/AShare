"""Alpha158 参考实现的入口：名字 ``alpha158_<原名小写>`` → Qlib 原文 → ``qlib_alpha158``。"""

from ..alpha158_formulas import FORMULAS
from .qlib_alpha158 import compute, compute_columns

BY_NAME = {f"alpha158_{name.lower()}": formula for _, name, formula in FORMULAS}


def compute_many(frames, names):
    """逐只股票算一组因子（同一只股票内共享子表达式）。"""
    formulas = {name: BY_NAME[name] for name in names}
    return compute_columns(frames, formulas, range(frames["close"].shape[1]))


__all__ = ["BY_NAME", "compute", "compute_many"]
