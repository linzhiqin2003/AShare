"""把报告数据渲染成一个自包含、离线可打开的 HTML 文件。

模板 `templates/tearsheet.html` 随包分发：原生 JS 加内联 SVG，系统字体栈，不引用任何外部资源。
数据以列式 JSON 内嵌在 `<script type="application/json">` 里。
"""

from __future__ import annotations

import json
import math
from importlib import resources
from pathlib import Path

PLACEHOLDER = "__TEARSHEET_DATA__"

# 页面里只画到这些精度；多余的位数只会让文件变大。
_DIGITS = {
    "nav": 6,
    "benchmark_nav": 6,
    "return": 6,
    "benchmark_return": 6,
    "drawdown": 5,
    "benchmark_drawdown": 5,
    "cash_weight": 5,
    "turnover": 5,
}


def _round_list(xs, k):
    if xs is None:
        return None
    out = []
    for x in xs:
        if x is None or (isinstance(x, float) and not math.isfinite(x)):
            out.append(None)
        else:
            v = round(float(x), k)
            out.append(int(v) if v == int(v) else v)
    return out


# 补充内容（对比指数、胜率、个股贡献、尾部、集中度）页面暂不显示，不嵌入，页面与之前逐字节相同
EXTRA_KEYS = ("benchmarks", "win_rates", "contributions", "tails", "concentration")


def compact(tearsheet: dict) -> dict:
    """只为嵌入页面而做的瘦身：按显示精度取整、去掉页面不用的明细、滚动指标的日期改成起始行号。"""
    d = {k: v for k, v in tearsheet.items() if k not in EXTRA_KEYS}
    daily = dict(d["daily"])
    for k, digits in _DIGITS.items():
        daily[k] = _round_list(daily.get(k), digits)
    d["daily"] = daily
    roll = dict(d["rolling"])
    dates = roll.pop("date")
    roll["start_index"] = d["daily"]["date"].index(dates[0]) if dates else None
    roll["length"] = len(dates)
    d["rolling"] = roll
    d["rebalances"] = [
        {k: v for k, v in r.items() if k != "unfilled_detail"} for r in d["rebalances"]
    ]
    return d


def expand(embedded: dict) -> dict:
    """`compact` 的逆：把滚动指标的日期还原出来（测试与外部读取用）。"""
    d = dict(embedded)
    roll = dict(d["rolling"])
    s, n = roll.pop("start_index"), roll.pop("length")
    roll["date"] = [] if s is None else d["daily"]["date"][s : s + n]
    d["rolling"] = roll
    return d


def _template() -> str:
    return (
        resources.files("ashare.report")
        .joinpath("templates/tearsheet.html")
        .read_text(encoding="utf-8")
    )


def to_html(tearsheet: dict) -> str:
    payload = json.dumps(
        compact(tearsheet), ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    # 防止数据里的 "</script>" 或 "<!--" 提前结束脚本块；< 在 JSON 里仍是 "<"。
    payload = payload.replace("<", "\\u003c")
    html = _template()
    title = str(tearsheet["meta"].get("title") or "回测报告")
    title = title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return html.replace("__TEARSHEET_TITLE__", title).replace(PLACEHOLDER, payload)


def render_html(tearsheet: dict, path) -> Path:
    """写出 HTML 文件并返回路径。"""
    path = Path(path)
    path.write_text(to_html(tearsheet), encoding="utf-8")
    return path
