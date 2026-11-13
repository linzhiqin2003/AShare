"""试验账本：每次训练追加一行 JSON，记下复现和多重检验校正需要的一切。

一行一个事件：`"trial"`（真的训练了一次）或 `"cache_hit"`（同样的定义重跑、直接返回
缓存）。`trial_id` 是定义的哈希（数据版本、特征、标签、预处理、股票池、切分、模型参数、
种子、代码版本），不含线程数；`trials()` 数的是不同的 trial_id，这是「到今天为止一共
试了多少组」——超参搜索的每一组都会入账，不只最好的那组。

写法在 `ashare.ledger`（与因子挖掘共用，只用标准库）；这里补上默认路径和代码版本。
"""

from __future__ import annotations

import functools
from pathlib import Path

from .. import ledger as _base
from ..ledger import _default  # noqa: F401
from .store import default_root


@functools.cache
def code_version():
    """仓库提交号和工作区是否有改动（拿不到时为 None）。"""
    return _base.code_version(Path(__file__).resolve().parent)


class Ledger(_base.Ledger):
    """追加写的 JSONL 账本。path 默认 `<ASHARE_ML_DIR 或 data/ml>/ledger.jsonl`。"""

    def __init__(self, path=None):
        super().__init__(path if path is not None else default_root() / "ledger.jsonl")
