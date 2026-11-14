"""三套因子的注册表（名字 ``alpha158_kmid``、``alpha101_001``、``gtja191_001``）。

元数据标记按 ``notes.py`` 的「条目 → 标记」表自动挂上；口径见 ``docs/CONVENTIONS.md``。
"""

from ashare.factor.formula import dsl, qlib, registry_from_module
from ashare.factor.library import Registry

from . import alpha101, gtja191, notes
from .alpha158_formulas import FORMULAS as ALPHA158_FORMULAS


def alpha158_registry():
    """158 条全部由转换程序从 Qlib 原文生成；名字为 ``alpha158_<原名小写>``。"""
    registry = Registry()
    for _, name, formula in ALPHA158_FORMULAS:
        registry.register(
            f"alpha158_{name.lower()}",
            qlib.to_expr(formula),
            formula,
            notes.alpha158_notes(name),
        )
    return registry


def alpha101_registry(*, argmax="days_ago"):
    """已翻译的 Alpha101 条目（``alpha101.py`` 里的 ``alpha101_NNN`` 函数）。

    ``argmax`` 选 ``ts_argmax/ts_argmin`` 的读法（D-13）：``"days_ago"``（默认，距今天数）
    或 ``"position"``（窗口内位置，最早一天为 1，并列取最早）。
    """
    with dsl.argmax_reading(argmax):
        return registry_from_module(
            alpha101, "alpha101", 101, notes=notes.ALPHA101_NOTES
        )


def gtja191_registry():
    """已翻译的国君 191 条目；裁决不收的 5 条不许出现。"""
    return registry_from_module(
        gtja191,
        "gtja191",
        191,
        notes=notes.GTJA191_NOTES,
        excluded=notes.GTJA191_EXCLUDED,
    )
