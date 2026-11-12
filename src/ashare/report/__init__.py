"""回测报告：从 `panelbt` 回测结果生成一份自包含、离线可打开的 HTML 报告。

最短用法::

    import ashare.report
    ashare.report.tearsheet(result, market, targets, dates, symbols,
                            data=ashare.data.open(), benchmark="000300.SH",
                            benchmark_kind="price").save("report.html")

只要数据、不要页面时用 `build_tearsheet`；已有数据要出页面用 `render_html`。

在终端里看（给人和 AI 代理）::

    ts = ashare.report.tearsheet(...)
    print(ts)                                   # 默认摘要，40 行以内
    print(ts.text(["yearly", "warnings"]))      # 按小节取；fmt="markdown" / "json"
    ts.save_json("report.json")                 # 之后：python -m ashare.report show report.json

指标口径见 `docs/REPORT_NOTES.md`。
"""

from __future__ import annotations

import json
from pathlib import Path

from .build import Benchmark, ReportError, build_tearsheet
from .compare import Comparison, compare
from .context import context_from_data, index_benchmark, industry_switch_date
from .render import render_html, to_html
from .sections import SECTIONS
from .shape import check_shape
from .text import render

__all__ = [
    "Benchmark",
    "Comparison",
    "ReportError",
    "SECTIONS",
    "Tearsheet",
    "build_tearsheet",
    "compare",
    "context_from_data",
    "index_benchmark",
    "industry_switch_date",
    "render_html",
    "tearsheet",
    "to_html",
]


class Tearsheet:
    """一份报告。`data` 是 `build_tearsheet` 的输出。

    `print(ts)` 打印默认摘要；`ts.text(...)` 取任意小节；`ts.save(path)` 写 HTML 页面；
    `ts.save_json(path)` / `Tearsheet.load(path)` 存取报告数据。
    """

    def __init__(self, data: dict):
        self.data = data

    @property
    def summary(self) -> dict:
        return self.data["summary"]

    def html(self) -> str:
        return to_html(self.data)

    def save(self, path) -> Path:
        return render_html(self.data, path)

    def text(
        self,
        sections=None,
        fmt: str = "text",
        *,
        lang: str = "zh",
        start=None,
        end=None,
    ) -> str:
        """文本版报告。

        sections：小节名或列表——summary、yearly、monthly、drawdowns、rolling、rebalances、
            holdings、industry、costs、warnings 及 `SECTIONS` 里的其他小节，"all" 为全部；
            默认只给摘要。
        fmt："text"（等宽纯文本）、"markdown" 或 "json"（所选小节的原始数值）。
        lang："zh" 中文表头；"en" 表头和标签用 JSON 键名。
        start / end：只看这个区间，区间指标按页面「切换区间」的公式重算。

        >>> print(ts.text(["summary", "warnings"], start="2020-01-01"))  # doctest: +SKIP
        """
        return render(self.data, sections, fmt, lang=lang, start=start, end=end)

    def __str__(self) -> str:
        return render(self.data)

    def __repr__(self) -> str:
        s = self.data["summary"]
        return (
            f"<Tearsheet {self.data['meta'].get('title')!r} "
            f"{s['first_trade_day']}..{s['end']}>"
        )

    def save_json(self, path) -> Path:
        """把报告数据写成 JSON（UTF-8）；之后可用 `Tearsheet.load` 或命令行读回。"""
        path = Path(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, allow_nan=False)
        return path

    @classmethod
    def load(cls, path) -> Tearsheet:
        """读回 `save_json` 写的报告数据；不是 JSON 或形状不对时抛 `ReportError`（说明缺什么）。"""
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except UnicodeDecodeError:
            raise ReportError(f"{path} 不是文本文件，读不出报告数据") from None
        except json.JSONDecodeError as e:
            raise ReportError(
                f"{path} 不是有效的 JSON（第 {e.lineno} 行第 {e.colno} 列）"
            ) from None
        check_shape(data, path)
        return cls(data)


def tearsheet(
    result,
    market,
    targets,
    dates,
    symbols,
    *,
    data=None,
    benchmark=None,
    benchmark_kind: str | None = None,
    benchmark_name: str | None = None,
    benchmarks=None,
    **kwargs,
) -> Tearsheet:
    """回测完一行出报告。

    data: `ashare.data.open()` 的返回值；给了就从数据层补齐证券名称、行业、行业换版日、
        停牌掩码，以及 `benchmark` 指数代码对应的行情（需同时声明 `benchmark_kind`）。
    benchmark: 指数代码（需要 data）或 `Benchmark` 对象。
    benchmarks: 对比指数列表，每项是 `Benchmark`，或 `(指数代码, 类型)` / `(代码, 类型, 名称)`
        （需要 data），如 `[("000905.SH", "price"), ("000852.SH", "price")]`。
    其余关键字参数原样传给 `build_tearsheet`，会覆盖从数据层取到的同名项。
    """
    ctx: dict = {}
    if data is not None:
        code = benchmark if isinstance(benchmark, str) else None
        ctx = context_from_data(
            data,
            dates,
            symbols,
            benchmark=code,
            benchmark_kind=benchmark_kind,
            benchmark_name=benchmark_name,
        )
    elif isinstance(benchmark, str):
        raise ReportError("用指数代码作基准时需要传入 data=ashare.data.open()")
    if isinstance(benchmark, Benchmark):
        ctx["benchmark"] = benchmark
    if benchmarks:
        extra = []
        for b in benchmarks:
            if isinstance(b, Benchmark):
                extra.append(b)
            elif data is None:
                raise ReportError(
                    "用指数代码作对比指数时需要传入 data=ashare.data.open()"
                )
            else:
                extra.append(index_benchmark(data, dates, *b))
        ctx["benchmarks"] = extra
    ctx.update(kwargs)
    return Tearsheet(build_tearsheet(result, market, targets, dates, symbols, **ctx))
