"""多策略对比：同一区间上的指标并排、分年收益并排、两两之间日收益的相关系数。

最短用法::

    import ashare.report
    cmp = ashare.report.compare([ts_a, ts_b], names=["低波", "红利"])
    print(cmp)                    # 并排的表
    cmp.data                      # 对比数据（dict）

各策略的区间不同时取交集（共同的第一个交易日到共同的最后一个交易日），并在输出里写明。
区间指标用页面「切换区间」的公式（`window.metrics`），某个策略的交集正好是它的全区间时直接取
它的 summary，与它自己的报告一致。
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence

from .build import ReportError
from .fmt import MISSING, num, pct
from .render import compact
from .sections import Table
from .text import FORMATS, _render_blocks
from .window import MIN_ANNUAL_DAYS, SUMMARY_KEYS, metrics

ROWS = (
    ("cumulative_return", "累计收益", lambda x: pct(x, 1, True)),
    ("annual_return", "年化收益", lambda x: pct(x, 1, True)),
    ("annual_volatility", "年化波动", lambda x: pct(x, 1)),
    ("sharpe", "夏普", num),
    ("max_drawdown", "最大回撤", lambda x: pct(x, 1)),
    ("excess_annual_return", "年化超额", lambda x: pct(x, 1, True)),
    ("information_ratio", "信息比率", num),
    ("tracking_error", "跟踪误差", lambda x: pct(x, 1)),
    ("annual_turnover", "年化换手（双边）", lambda x: pct(x, 0)),
    ("total_fees_to_initial", "费用 / 期初权益", lambda x: pct(x, 2)),
    ("total_slippage_to_initial", "滑点 / 期初权益", lambda x: pct(x, 2)),
)


def _data(x) -> dict:
    d = getattr(x, "data", x)
    if not isinstance(d, dict) or "daily" not in d:
        raise ReportError("compare 需要 Tearsheet 或 build_tearsheet 的输出")
    return d


def _pearson(x: list[float], y: list[float]) -> float | None:
    n = len(x)
    if n < 2:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y, strict=True))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    if not (sxx > 0 and syy > 0):
        return None
    return sxy / math.sqrt(sxx * syy)


def compare_data(tearsheets: Sequence, names: Sequence[str] | None = None) -> dict:
    """返回对比数据（只含 JSON 基本类型）。"""
    ds = [_data(t) for t in tearsheets]
    if len(ds) < 2:
        raise ReportError("至少需要两份报告")
    if names is None:
        names = [d["meta"].get("title") or f"策略{k + 1}" for k, d in enumerate(ds)]
    names = [str(n) for n in names]
    if len(names) != len(ds) or len(set(names)) != len(names):
        raise ReportError("names 的个数要与报告相同，且不能重复")
    start = max(d["daily"]["date"][0] for d in ds)  # 共同的基准日
    end = min(d["daily"]["date"][-1] for d in ds)
    if start >= end:
        raise ReportError("各报告的区间没有重叠")
    spans, metrics_out, yearly, rets = [], [], {}, []
    for d, name in zip(ds, names, strict=True):
        dates = d["daily"]["date"]
        a = max(i for i, x in enumerate(dates) if x <= start)
        b = max(i for i, x in enumerate(dates) if x <= end)
        if b - a < 1:
            raise ReportError(f"{name} 在共同区间内没有交易日")
        emb = compact(d)
        mm = metrics(d, a, b, emb)
        row = {SUMMARY_KEYS[k]: v for k, v in mm.items() if k in SUMMARY_KEYS}
        row["annualized"] = mm["n"] >= MIN_ANNUAL_DAYS
        row["benchmark"] = (d["meta"].get("benchmark") or {}).get("name")
        row["full_range"] = a == 0 and b == len(dates) - 1
        row["name"] = name
        metrics_out.append(row)
        spans.append(
            {
                "name": name,
                "report_start": dates[0],
                "report_end": dates[-1],
                "start": dates[a],
                "end": dates[b],
            }
        )
        nav = emb["daily"]["nav"]
        by_year: dict[int, tuple[int, int]] = {}
        for i in range(a + 1, b + 1):
            y = int(dates[i][:4])
            by_year[y] = (by_year.get(y, (i, i))[0], i)
        own = {y["year"]: y for y in d["yearly"]}
        for y, (i0, i1) in by_year.items():
            whole = [i for i in range(1, len(dates)) if dates[i][:4] == str(y)]
            clipped = (i0, i1) != (whole[0], whole[-1])
            # 整年都在交集里：取报告自己的分年收益；被截断：按嵌入的净值重算交集内的部分
            yearly.setdefault(y, {})[name] = {
                "return": nav[i1] / nav[i0 - 1] - 1 if clipped else own[y]["return"],
                "partial": bool(own[y]["partial"] or clipped),
            }
        rets.append(
            dict(
                zip(
                    dates[a + 1 : b + 1],
                    emb["daily"]["return"][a + 1 : b + 1],
                    strict=True,
                )
            )
        )
    common = sorted(set.intersection(*(set(r) for r in rets)))
    corr = [
        [
            1.0
            if i == j
            else _pearson([rets[i][t] for t in common], [rets[j][t] for t in common])
            for j in range(len(ds))
        ]
        for i in range(len(ds))
    ]
    aligned = all(
        s["start"] == s["report_start"] and s["end"] == s["report_end"] for s in spans
    )
    return {
        "names": names,
        "start": start,
        "end": end,
        "aligned": aligned,
        "spans": spans,
        "metrics": metrics_out,
        "yearly": [
            {"year": y, **{n: yearly[y].get(n) for n in names}} for y in sorted(yearly)
        ],
        "correlation": corr,
        "common_days": len(common),
    }


class Comparison:
    """`compare` 的结果。`data` 是对比数据；`print(cmp)` 打印并排的表。"""

    def __init__(self, data: dict):
        self.data = data

    def text(self, fmt: str = "text", *, lang: str = "zh") -> str:
        if fmt not in FORMATS:
            raise ValueError(f"fmt 只能是 {', '.join(FORMATS)}")
        if fmt == "json":
            return json.dumps(self.data, ensure_ascii=False, indent=1, allow_nan=False)
        return render_compare(self.data, fmt, lang)

    def __str__(self) -> str:
        return self.text()

    def save_json(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, allow_nan=False)
        return path


def render_compare(c: dict, fmt: str = "text", lang: str = "zh") -> str:
    def L(zh, en):
        return en if lang == "en" else zh

    names = c["names"]
    blocks: list = []
    blocks.append(
        L(
            f"共同区间 {c['start']} 至 {c['end']}（{c['start']} 为基准日），{c['common_days']} 个共同交易日",
            f"start={c['start']} end={c['end']} common_days={c['common_days']}",
        )
    )
    if not c["aligned"]:
        for s in c["spans"]:
            if (s["start"], s["end"]) != (s["report_start"], s["report_end"]):
                blocks.append(
                    L(
                        f"{s['name']} 的报告区间是 {s['report_start']} 至 {s['report_end']}，这里只取交集",
                        f"{s['name']}: report {s['report_start']}..{s['report_end']}, cut to the common range",
                    )
                )
    bn = {m["benchmark"] for m in c["metrics"]}
    if len(bn) > 1:
        blocks.append(
            L(
                "各策略的基准不同："
                + "、".join(
                    f"{m['name']} 对 {m['benchmark'] or '无'}" for m in c["metrics"]
                ),
                "benchmarks differ: "
                + ", ".join(f"{m['name']}={m['benchmark']}" for m in c["metrics"]),
            )
        )
    t = Table([(L("指标", "metric"), "l")] + [(n, "r") for n in names])
    for key, zh, f in ROWS:
        if key == "annual_return" and not all(m["annualized"] for m in c["metrics"]):
            continue
        if all(m.get(key) is None for m in c["metrics"]):
            continue
        t.rows.append([L(zh, key)] + [f(m.get(key)) for m in c["metrics"]])
    blocks += ["", t]
    yt = Table([(L("年份", "year"), "l")] + [(n, "r") for n in names])
    for y in c["yearly"]:
        cells = []
        for n in names:
            v = y.get(n)
            cells.append(
                MISSING
                if v is None
                else pct(v["return"], 1, True) + ("*" if v["partial"] else "")
            )
        yt.rows.append([str(y["year"])] + cells)
    blocks += [
        "",
        L("分年收益（* 不完整或被区间截断）", "yearly return (* partial)"),
        yt,
    ]
    ct = Table([("", "l")] + [(n, "r") for n in names])
    for i, n in enumerate(names):
        ct.rows.append([n] + [num(x) for x in c["correlation"][i]])
    blocks += ["", L("日收益相关系数", "daily return correlation"), ct]
    title = L("策略对比", "comparison")
    return "\n".join(_render_blocks(title, blocks, fmt)) + "\n"


def compare(tearsheets: Sequence, names: Sequence[str] | None = None) -> Comparison:
    """把几份报告放在同一区间上对比。tearsheets：`Tearsheet` 或 `build_tearsheet` 的输出。

    >>> print(ashare.report.compare([ts_a, ts_b], names=["低波", "红利"]))  # doctest: +SKIP
    """
    return Comparison(compare_data(tearsheets, names))
