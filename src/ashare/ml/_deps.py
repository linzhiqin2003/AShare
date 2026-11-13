"""可选依赖组的检查：缺了就给出缺哪个组、怎么装的中文提示。

`ashare.ml` 只在导入时需要 `ml` 组（LightGBM、scikit-learn）；以后的神经网络模型
走单独的 `nn` 组（PyTorch），默认不装，只有用到那个模型名时才检查。
"""

from __future__ import annotations

import importlib
import importlib.util

GROUPS = {
    "ml": ("lightgbm", "sklearn"),
    "nn": ("torch",),
}
_PIP = {"sklearn": "scikit-learn"}


class MissingDependency(ImportError):
    """缺少可选依赖组。消息里写明缺哪个组、怎么装。"""


def importable(module):
    """真的导入一次（只检查 find_spec 会把装坏的包当成已装）。"""
    try:
        if importlib.util.find_spec(module) is None:
            return False
        importlib.import_module(module)
    except (ImportError, ValueError):
        return False
    return True


def missing(group):
    return [m for m in GROUPS[group] if not importable(m)]


def require(group, purpose=None):
    """确认可选依赖组已装；没装时抛出带安装方法的 MissingDependency。"""
    absent = missing(group)
    if absent:
        names = "、".join(_PIP.get(m, m) for m in absent)
        what = f"{purpose}需要" if purpose else "需要"
        raise MissingDependency(
            f"{what}可选依赖组 `{group}`（缺 {names}）。在仓库根目录运行 "
            f"`uv sync --all-packages --group dev --group {group} --inexact` 安装；"
            f"用 pip 时运行 `pip install {' '.join(_PIP.get(m, m) for m in absent)}`。"
        )
