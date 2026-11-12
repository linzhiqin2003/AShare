"""回测报告：从 `panelbt` 回测结果生成一份自包含、离线可打开的 HTML 报告。

最短用法::

    import ashare.report
    ashare.report.tearsheet(result, market, targets, dates, symbols,
                            data=ashare.data.open(), benchmark="000300.SH",
                            benchmark_kind="price").save("report.html")

只要数据、不要页面时用 `build_tearsheet`；已有数据要出页面用 `render_html`。
指标口径见 `docs/REPORT_NOTES.md`。
"""

from __future__ import annotations

from pathlib import Path

from .build import Benchmark, ReportError, build_tearsheet
from .context import context_from_data, industry_switch_date
from .render import render_html, to_html

__all__ = [
    "Benchmark",
    "ReportError",
    "Tearsheet",
    "build_tearsheet",
    "context_from_data",
    "industry_switch_date",
    "render_html",
    "tearsheet",
    "to_html",
]


class Tearsheet:
    """一份报告。`data` 是 `build_tearsheet` 的输出。"""

    def __init__(self, data: dict):
        self.data = data

    @property
    def summary(self) -> dict:
        return self.data["summary"]

    def html(self) -> str:
        return to_html(self.data)

    def save(self, path) -> Path:
        return render_html(self.data, path)


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
    **kwargs,
) -> Tearsheet:
    """回测完一行出报告。

    data: `ashare.data.open()` 的返回值；给了就从数据层补齐证券名称、行业、行业换版日、
        停牌掩码，以及 `benchmark` 指数代码对应的行情（需同时声明 `benchmark_kind`）。
    benchmark: 指数代码（需要 data）或 `Benchmark` 对象。
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
    ctx.update(kwargs)
    return Tearsheet(build_tearsheet(result, market, targets, dates, symbols, **ctx))
