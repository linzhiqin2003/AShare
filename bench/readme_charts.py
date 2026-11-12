"""Render the two README charts as static SVG (docs/assets/), in Chinese and English.

Numbers are copied from the frozen records so the charts cannot drift from them:
docs/BENCHMARKS.md (engine comparison) and bench/e2e/results/{baseline,round1-after}.json
(end-to-end research run before and after the optimisation round).

    uv run --no-sync python bench/readme_charts.py
"""

import json
import math
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/assets"

STYLE = """
<style>
  .surface { fill: #fcfcfb; } .ink { fill: #0b0b0b; } .ink2 { fill: #52514e; }
  .muted { fill: #898781; } .grid { stroke: #e1e0d9; } .axis { stroke: #c3c2b7; }
  .other { fill: #b9b7ae; } .hero { fill: #2a78d6; }
  .s1 { fill: #2a78d6; } .s2 { fill: #eb6834; } .s3 { fill: #1baf7a; } .s4 { fill: #eda100; }
  .s5 { fill: #b9b7ae; }
  text { font-family: -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei",
         "Noto Sans CJK SC", Helvetica, Arial, sans-serif; }
  .t { font-size: 15px; font-weight: 600; } .l { font-size: 13px; } .s { font-size: 12px; }
  .b { font-weight: 600; }
  @media (prefers-color-scheme: dark) {
    .surface { fill: #1a1a19; } .ink { fill: #f0efec; } .ink2 { fill: #c3c2b7; }
    .grid { stroke: #2c2c2a; } .axis { stroke: #383835; }
    .other { fill: #6f6d66; } .hero { fill: #3987e5; }
    .s1 { fill: #3987e5; } .s2 { fill: #d95926; } .s3 { fill: #199e70; } .s4 { fill: #c98500; }
    .s5 { fill: #6f6d66; }
  }
</style>
"""


def svg(width, height, body, label):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" aria-label="{escape(label)}">'
        f'{STYLE}<rect class="surface" width="{width}" height="{height}" rx="10"/>'
        f"{body}</svg>\n"
    )


def bar(x, y, w, h, cls):
    """Rounded on the data end only, square on the baseline."""
    r = min(4, w / 2, h / 2)
    return (
        f'<path class="{cls}" d="M{x:.1f},{y:.1f}h{w - r:.1f}a{r},{r} 0 0 1 {r},{r}'
        f'v{h - 2 * r:.1f}a{r},{r} 0 0 1 -{r},{r}h-{w - r:.1f}z"/>'
    )


def duration(ms):
    if ms < 100:
        return f"{ms:.1f} ms"
    if ms < 1000:
        return f"{ms:.0f} ms"
    seconds = ms / 1000
    return f"{seconds:.2f} s" if seconds < 10 else f"{seconds:.0f} s"


TEXT = {
    "en": {
        "engines": [
            "panelbt",
            "NumPy, hand-written (sparse holdings)",
            "NumPy, hand-written (full cross-section)",
            "vectorbt (Numba)",
            "vectorbt (Rust engine)",
            "bt",
            "backtrader",
        ],
        "e_title": "One backtest, 4,000 days × 5,500 stocks",
        "e_label": "Backtest wall time by engine, log scale",
        "bars": ["Before", "After", "After, batch + parallel"],
        "stages": ["Risk model", "Factors", "Rank IC", "Optimisation", "Other"],
        "r_title": "One full research run, 4,824 days × 5,923 stocks",
        "r_label": "End-to-end research run before and after",
    },
    "zh": {
        "engines": [
            "panelbt",
            "手写 NumPy（稀疏持仓）",
            "手写 NumPy（全截面）",
            "vectorbt（Numba）",
            "vectorbt（Rust 引擎）",
            "bt",
            "backtrader",
        ],
        "e_title": "一次回测，4000 日 × 5500 只",
        "e_label": "各引擎一次回测的耗时，对数刻度",
        "bars": ["优化前", "优化后", "优化后，批量加并行"],
        "stages": ["风险模型", "算因子", "秩 IC", "组合优化", "其余"],
        "r_title": "一次完整研究，4824 日 × 5923 只",
        "r_label": "一次完整研究在优化前后的耗时",
    },
}


def engines(lang):
    # docs/BENCHMARKS.md, task 1, 4000 x 5500, rebalance every 21 days, the backtest itself.
    t = TEXT[lang]
    values = [2.698, 8.990, 14.496, 1195.775, 1286.401, 7435.173, 330844.682]
    rows = list(zip(t["engines"], values, strict=True))
    width, left, right, top, step = 860, 290, 150, 52, 34
    plot = width - left - right
    height = top + step * len(rows) + 30
    lo, hi = 0.0, 6.0  # 1 ms .. 1e6 ms

    def x(ms):
        return left + (math.log10(ms) - lo) / (hi - lo) * plot

    out = [
        f'<text class="t ink" x="24" y="30">{escape(t["e_title"])}</text>',
    ]
    for k, tick in enumerate(["1 ms", "10 ms", "100 ms", "1 s", "10 s", "100 s"]):
        gx = left + k / (hi - lo) * plot
        out.append(
            f'<line class="grid" x1="{gx:.1f}" y1="{top - 4}" x2="{gx:.1f}" '
            f'y2="{top + step * len(rows)}"/>'
            f'<text class="s muted" x="{gx:.1f}" y="{top + step * len(rows) + 18}" '
            f'text-anchor="middle">{tick}</text>'
        )
    base = rows[0][1]
    for i, (name, ms) in enumerate(rows):
        y = top + i * step + 6
        hero = i == 0
        out.append(
            f'<text class="l {"ink b" if hero else "ink2"}" x="{left - 12}" y="{y + 15}" '
            f'text-anchor="end">{escape(name)}</text>'
        )
        out.append(bar(left, y, x(ms) - left, 22, "hero" if hero else "other"))
        ratio = "" if hero else f" · {ms / base:,.0f}×"
        out.append(
            f'<text class="l {"ink b" if hero else "ink"}" x="{x(ms) + 8:.1f}" y="{y + 15}">'
            f"{duration(ms)}{ratio}</text>"
        )
    out.append(
        f'<line class="axis" x1="{left}" y1="{top - 4}" x2="{left}" y2="{top + step * len(rows)}"/>'
    )
    return svg(width, height, "".join(out), t["e_label"])


def research(lang):
    t = TEXT[lang]
    old = json.loads((ROOT / "bench/e2e/results/baseline.json").read_text())
    new = json.loads((ROOT / "bench/e2e/results/round1-after.json").read_text())

    def steps(report, case):
        found = next(c for c in report["stages"]["e2e"]["cases"] if c["case"] == case)
        s = found["detail"]["steps"]
        parts = [
            s["risk_model_full"],
            s["factor_evaluate"],
            s["factor_rank_ic"],
            s["portfolio_optimize_sequence"],
        ]
        return [*parts, found["wall_min"] - sum(parts)], found["wall_min"]

    bars = [
        (t["bars"][0], *steps(old, "research_once")),
        (t["bars"][1], *steps(new, "research_once")),
        (t["bars"][2], *steps(new, "research_once_fast")),
    ]
    names = t["stages"]
    width, left, right, top, step = 860, 190, 96, 82, 40
    plot = width - left - right
    height = top + step * len(bars) + 30
    scale = plot / 200.0
    out = [
        f'<text class="t ink" x="24" y="30">{escape(t["r_title"])}</text>',
    ]
    lx = 24
    for i, name in enumerate(names):
        out.append(
            f'<rect class="s{i + 1}" x="{lx}" y="46" width="11" height="11" rx="2"/>'
            f'<text class="s ink2" x="{lx + 16}" y="56">{escape(name)}</text>'
        )
        lx += 16 + sum(12.5 if ord(c) > 255 else 6.8 for c in name) + 20
    for tick in range(0, 201, 50):
        gx = left + tick * scale
        out.append(
            f'<line class="grid" x1="{gx:.1f}" y1="{top - 6}" x2="{gx:.1f}" '
            f'y2="{top + step * len(bars)}"/>'
            f'<text class="s muted" x="{gx:.1f}" y="{top + step * len(bars) + 18}" '
            f'text-anchor="middle">{tick} s</text>'
        )
    for i, (name, parts, total) in enumerate(bars):
        y = top + i * step + 6
        out.append(
            f'<text class="l {"ink2" if i == 0 else "ink b"}" x="{left - 12}" y="{y + 19}" '
            f'text-anchor="end">{escape(name)}</text>'
        )
        cursor = left
        for k, value in enumerate(parts):
            w = value * scale
            last = k == len(parts) - 1
            shown = max(w - (0 if last else 2), 0.6)  # 2px surface gap between segments
            if last:
                out.append(bar(cursor, y, shown, 28, f"s{k + 1}"))
            else:
                out.append(
                    f'<rect class="s{k + 1}" x="{cursor:.1f}" y="{y}" width="{shown:.1f}" '
                    'height="28"/>'
                )
            cursor += w
        gain = f" · {bars[0][2] / total:.1f}×" if i else ""
        out.append(
            f'<text class="l ink b" x="{cursor + 8:.1f}" y="{y + 19}">{total:.1f} s{gain}</text>'
        )
    out.append(
        f'<line class="axis" x1="{left}" y1="{top - 6}" x2="{left}" y2="{top + step * len(bars)}"/>'
    )
    return svg(width, height, "".join(out), t["r_label"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for lang, suffix in (("zh", ""), ("en", ".en")):
        (OUT / f"backtest-engines{suffix}.svg").write_text(engines(lang))
        (OUT / f"research-run{suffix}.svg").write_text(research(lang))
    print("wrote", *(p.name for p in sorted(OUT.glob("*.svg"))))


if __name__ == "__main__":
    main()
