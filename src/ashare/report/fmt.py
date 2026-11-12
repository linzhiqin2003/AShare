"""文本版用的数字格式与等宽排版。

数字格式照页面（`templates/tearsheet.html` 里的 `pct`、`num`）：先乘 100 再按 JS
`Number.prototype.toFixed` 的规则取位（对二进制精确值四舍五入，恰好一半时取绝对值大的一侧），
取整后为 0 的不带正负号。与页面的差别只有两处：负号用 ASCII 的 `-`（页面用 U+2212），
缺失写 `-`（页面写 `—`）。
"""

from __future__ import annotations

import math
import re
import unicodedata
from decimal import ROUND_HALF_UP, Decimal

MISSING = "-"


def to_fixed(x: float, d: int) -> str:
    """JS 的 `x.toFixed(d)`（x 有限、|x| < 1e21）。"""
    neg = x < 0  # JS 对 -0 不加负号
    q = Decimal(abs(float(x))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
    s = f"{q:.{d}f}"
    return "-" + s if neg else s


def _ok(x) -> bool:
    return x is not None and not isinstance(x, bool) and math.isfinite(float(x))


def pct(x, d: int = 1, sign: bool = False) -> str:
    """小数 → 百分数文本，如 0.0863 → `+8.6%`（sign=True）。与页面 `pct` 相同。"""
    if not _ok(x):
        return MISSING
    v = to_fixed(float(x) * 100, d)
    neg = v.startswith("-")
    a = v[1:] if neg else v
    if float(a) == 0:
        return to_fixed(0.0, d) + "%"
    return ("-" if neg else ("+" if sign else "")) + a + "%"


def num(x, d: int = 2) -> str:
    """普通数字，如夏普 0.5884 → `0.59`。与页面 `num` 相同。"""
    if not _ok(x):
        return MISSING
    v = to_fixed(float(x), d)
    if float(v) == 0:
        v = to_fixed(0.0, d)
    return v


def money(x) -> str:
    """金额（元），千分位、两位小数。"""
    if not _ok(x):
        return MISSING
    return f"{float(x):,.2f} 元"


def integer(x, unit: str = "") -> str:
    if x is None:
        return MISSING
    return f"{int(x)}{unit}"


def width(s: str) -> int:
    """等宽终端里的显示列数：东亚宽字符（中文、全角标点）算两列。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in s)


def pad(s: str, w: int, align: str = "l") -> str:
    gap = max(0, w - width(s))
    return s + " " * gap if align == "l" else " " * gap + s


MAX_WIDTH = 100
BREAK_AFTER = " ，；、,;"
# 命令片段（`--section warnings`、`--lang en`）整体不拆：选项名和紧跟的一个参数算一个整体
_COMMAND = re.compile(r"--[A-Za-z][\w-]*(?: [A-Za-z0-9][\w,.-]*)?")


def _atoms(line: str) -> list[str]:
    out, i = [], 0
    for m in _COMMAND.finditer(line):
        out += list(line[i : m.start()])
        out.append(m.group())
        i = m.end()
    return out + list(line[i:])


def wrap(line: str, limit: int = MAX_WIDTH, indent: str = "  ") -> list[str]:
    """按显示宽度折行；优先在空格、逗号、分号、顿号之后断开，命令片段不拆开。

    续行带 `indent`。没有合适的断点时在宽度上限处硬断（命令片段整体移到下一行）。
    """
    out: list[str] = []
    cur: list[str] = []
    for atom in _atoms(line):
        if "".join(cur).strip() and width("".join(cur) + atom) > limit:
            cuts = [k for k, x in enumerate(cur) if x in BREAK_AFTER]
            cut = cuts[-1] if cuts else -1
            if cut > len(cur) // 2:
                out.append("".join(cur[: cut + 1]).rstrip())
                rest = "".join(cur[cut + 1 :]).lstrip()
                cur = [indent, *_atoms(rest)] if rest else [indent]
            else:
                out.append("".join(cur).rstrip())
                cur = [indent]
        cur.append(atom)
    out.append("".join(cur))
    return out
