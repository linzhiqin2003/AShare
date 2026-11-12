"""报告的文本版：给在终端里工作的人和 AI 代理看。

`render(data, sections, fmt, lang, start, end)` 是唯一入口；`Tearsheet.text()`、`print(tearsheet)`
和 `python -m ashare.report show` 都走它。三种格式：

- `text`：等宽对齐的纯文本，不用 ANSI 颜色、不用制表线字符，行宽不超过 100 列（中文按两列算）；
- `markdown`：同样的内容排成 Markdown 表格；
- `json`：只含所选小节的原始数值（不取整、不加单位），键名与报告数据一致。

所有数字都取自 `build_tearsheet` 的输出（或按页面公式重算的区间指标，见 `window.py`），
显示精度与页面相同。小节清单见 `SECTIONS`。
"""

from __future__ import annotations

import json

from . import sections as sec
from . import sections_extra  # noqa: F401  注册补充内容的小节
from .fmt import MAX_WIDTH, pad, width, wrap
from .sections import Table

FORMATS = ("text", "markdown", "json")


def _table_text(t: Table, gap: int = 2) -> list[str]:
    head = [h for h, _ in t.cols]
    ws = [max([width(h)] + [width(r[j]) for r in t.rows]) for j, h in enumerate(head)]
    lines = []
    for row in [head, *t.rows]:
        cells = [pad(c, ws[j], t.cols[j][1]) for j, c in enumerate(row)]
        lines.append((" " * gap).join(cells).rstrip())
    if all(width(x) <= MAX_WIDTH for x in lines):
        return lines
    if gap > 1:
        return _table_text(t, gap - 1)
    # 单空格仍放不下（英文表头较长时）：保留第一列，其余列分成几张表
    groups, cur, used = [], [], ws[0]
    for j in range(1, len(head)):
        if cur and used + 1 + ws[j] > MAX_WIDTH:
            groups.append(cur)
            cur, used = [], ws[0]
        cur.append(j)
        used += 1 + ws[j]
    groups.append(cur)
    out = []
    for g in groups:
        sub = Table(
            [t.cols[0]] + [t.cols[j] for j in g],
            [[r[0]] + [r[j] for j in g] for r in t.rows],
        )
        if out:
            out.append("")
        out += _table_text(sub, 2)
    return out


def _table_md(t: Table) -> list[str]:
    def esc(s):
        return s.replace("|", "\\|")

    lines = ["| " + " | ".join(esc(h) for h, _ in t.cols) + " |"]
    lines.append("|" + "|".join("---:" if a == "r" else "---" for _, a in t.cols) + "|")
    lines += ["| " + " | ".join(esc(c) for c in r) + " |" for r in t.rows]
    return lines


def _render_blocks(title: str | None, blocks: list, fmt: str) -> list[str]:
    out = []
    if title:
        out.append(f"### {title}" if fmt == "markdown" else title)
    for b in blocks:
        if isinstance(b, Table):
            if fmt == "markdown":
                if out and out[-1] != "":
                    out.append("")
                out += _table_md(b)
                out.append("")
            else:
                out += _table_text(b)
        elif fmt == "markdown":
            out.append(b + "  " if b else "")
        else:
            out += wrap(b)
    while out and out[-1] == "":
        out.pop()
    return out


def render(
    data: dict,
    sections=None,
    fmt: str = "text",
    *,
    lang: str = "zh",
    start=None,
    end=None,
) -> str:
    """把报告数据渲染成文本。

    sections：小节名或名字列表（见 `SECTIONS`），`None` 为默认摘要 `["summary"]`，`"all"` 为全部。
    fmt：`"text"`、`"markdown"` 或 `"json"`。lang：`"zh"`（中文表头）或 `"en"`（表头和标签用 JSON 键名）。
    start / end：只看这个区间（`YYYY-MM-DD` 或 `YYYYMMDD`）；区间指标按页面「切换区间」的公式重算。
    """
    if fmt not in FORMATS:
        raise ValueError(f"fmt 只能是 {', '.join(FORMATS)}")
    if lang not in ("zh", "en"):
        raise ValueError("lang 只能是 'zh' 或 'en'")
    names = sec.resolve_sections(sections)
    ctx = sec.Context.make(data, lang=lang, start=start, end=end)
    if fmt == "json":
        out = {"range": ctx.range_info()}
        for n in names:
            out[n] = sec.SECTIONS[n].json(ctx)
        return json.dumps(out, ensure_ascii=False, indent=1, allow_nan=False)
    parts = []
    for n in names:
        s = sec.SECTIONS[n]
        blocks = s.blocks(ctx)
        title = None if n == "summary" else (n if lang == "en" else s.title)
        if n == "summary" and fmt == "markdown":
            title = ctx.data["meta"].get("title") or "回测报告"
            blocks = blocks[1:]  # 第一行是标题
        parts.append("\n".join(_render_blocks(title, blocks, fmt)))
    return "\n\n".join(parts) + "\n"
