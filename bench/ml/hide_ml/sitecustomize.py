"""模拟「没装 ml 依赖组」的环境：让 lightgbm 导入失败。

用法（在仓库根目录）：
    PYTHONPATH=bench/ml/hide_ml .venv/bin/python -m pytest
预期：test/ml 下的模块全部 skipped（「需要可选依赖组 ml」），其余测试照常通过；
`import ashare.ml` 抛出写明 `uv sync ... --group ml` 的 MissingDependency。
scikit-learn 不隐藏：它同时在 dev 组里（风险模型的参考实现要用）。
"""

import sys
from importlib.abc import MetaPathFinder

HIDDEN = {"lightgbm"}


class _Hide(MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in HIDDEN:
            raise ModuleNotFoundError(f"No module named {name!r}（已被 hide_ml 隐藏）")
        return None


sys.meta_path.insert(0, _Hide())
