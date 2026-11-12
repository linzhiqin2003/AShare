"""分析结果的通用容器：二维数组加行名、列名，可转 dict 与 pandas 表。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Table:
    """行 × 列的数值表。

    `values[i, j]` 是第 i 行（通常是日期或资金规模）、第 j 列的值；`index` 与 `columns`
    都是字符串或数值。报告模块可以直接读 `values`，或转成 dict / pandas。
    """

    index: np.ndarray
    columns: tuple[str, ...]
    values: np.ndarray
    name: str = ""
    index_name: str = "index"
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        v = np.asarray(self.values, dtype=float)
        if v.ndim == 1:
            v = v[:, None]
        idx = np.asarray(self.index)
        if v.shape != (len(idx), len(self.columns)):
            raise ValueError(
                f"{self.name or 'Table'}: values 形状 {v.shape} 与行列数不符"
            )
        object.__setattr__(self, "values", v)
        object.__setattr__(self, "index", idx)
        object.__setattr__(self, "columns", tuple(map(str, self.columns)))

    def __getitem__(self, column: str) -> np.ndarray:
        try:
            return self.values[:, self.columns.index(column)]
        except ValueError:
            raise KeyError(column) from None

    @property
    def shape(self):
        return self.values.shape

    def to_dict(self) -> dict:
        """列式 dict：{行名列: [...], 列名: [...]}，只含 JSON 基本类型（NaN、±inf 记 None）。

        与 `ashare.report` 的输出同一风格：键名英文小写下划线，比例用小数。"""
        out = {self.index_name: plain(self.index)}
        for j, c in enumerate(self.columns):
            out[c] = plain(self.values[:, j])
        return out

    def to_frame(self):
        """转成 pandas.DataFrame（调用时才导入 pandas）。"""
        import pandas as pd

        return pd.DataFrame(self.values, index=self.index, columns=list(self.columns))

    def last(self) -> dict:
        """最后一行（累计表的最终值）转成 {列名: 值}。"""
        return {c: float(self.values[-1, j]) for j, c in enumerate(self.columns)}

    def __repr__(self):
        head = f"Table({self.name!r}, {len(self.index)} 行 × {len(self.columns)} 列)"
        return head


def format_table(table: Table, digits: int = 4, scale: float = 1.0) -> str:
    """简单的 Markdown 表格文本，用于文档和命令行查看。"""
    head = "| | " + " | ".join(table.columns) + " |"
    sep = "|---" * (len(table.columns) + 1) + "|"
    rows = [head, sep]
    for i, label in enumerate(table.index):
        cells = " | ".join(f"{v * scale:.{digits}f}" for v in table.values[i])
        rows.append(f"| {label} | {cells} |")
    return "\n".join(rows)


def plain(value):
    """递归转成 JSON 基本类型：数组与元组变 list，numpy 标量变 Python 数，NaN/inf 变 None。"""
    if isinstance(value, Table):
        return value.to_dict()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return [plain(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value
