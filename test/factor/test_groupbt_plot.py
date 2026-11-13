"""groupbt.plot 的测试：手算样例、随机数据与逐点参考实现。"""

from __future__ import annotations

import importlib.util
import math
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "gbt_plot", ROOT / "src/ashare/factor/groupbt/plot.py"
)
plot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plot)

NS = "{http://www.w3.org/2000/svg}"

REF_LEFT = 64.0
REF_RIGHT_MARGIN = 168.0
REF_TOP_TITLE = 44.0
REF_TOP_PLAIN = 24.0
REF_BOTTOM_MARGIN = 36.0


def parse(svg):
    return ET.fromstring(svg)


def polylines(root):
    return root.findall(f".//{NS}polyline")


def points_of(element):
    return [
        tuple(float(part) for part in chunk.split(","))
        for chunk in element.get("points").split()
    ]


def polyline_points_by_name(root):
    result = {}
    for element in polylines(root):
        result.setdefault(element.get("data-name"), []).append(points_of(element))
    return result


def ref_hex_to_rgb(value):
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def ref_group_colors(count):
    start = (0x08, 0x30, 0x6B)
    end = (0xC6, 0xDB, 0xEF)
    colors = []
    for i in range(count):
        t = i / (count - 1) if count > 1 else 0.0
        rgb = tuple(int(round(start[k] + (end[k] - start[k]) * t)) for k in range(3))
        colors.append("#{:02x}{:02x}{:02x}".format(*rgb))
    return colors


def ref_nice_ticks(lo, hi, target=6):
    if not (math.isfinite(lo) and math.isfinite(hi)):
        lo, hi = 0.0, 1.0
    if hi - lo <= 0.0:
        pad = 1.0 if lo == 0.0 else abs(lo) * 0.1
        lo, hi = lo - pad, hi + pad
    span = hi - lo
    exponent = math.floor(math.log10(span / target))
    best_key = None
    best_ticks = None
    for k in range(exponent - 2, exponent + 2):
        scale = 10.0**k
        for mantissa in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0):
            step = mantissa * scale
            start = math.floor(lo / step + 1e-12) * step
            count = int(math.floor((hi - start) / step + 1e-9)) + 1
            if not 5 <= count <= 7:
                continue
            key = (abs(count - target), step)
            if best_key is None or key < best_key:
                best_key = key
                best_ticks = [start + i * step for i in range(count)]
    if best_ticks is None:
        step = span / 5.0
        best_ticks = [lo + i * step for i in range(6)]
    return [0.0 if abs(v) < 1e-12 else float(v) for v in best_ticks]


def ref_sample_indices(count, limit=2000):
    if count <= limit:
        return list(range(count))
    step = (count - 1) / (limit - 1)
    raw = [int(round(j * step)) for j in range(limit)]
    unique = []
    for value in raw:
        if not unique or value != unique[-1]:
            unique.append(value)
    return unique


def ref_plot_runs(values, indices):
    runs = []
    start = None
    previous = None
    for position, index in enumerate(indices):
        if not math.isfinite(float(values[index])):
            if start is not None:
                runs.append((start, position))
                start = None
            previous = None
            continue
        if start is None:
            start = position
        elif previous is not None and index - previous > 1:
            broken = False
            for j in range(previous + 1, index):
                if not math.isfinite(float(values[j])):
                    broken = True
                    break
            if broken:
                runs.append((start, position))
                start = position
        previous = index
    if start is not None:
        runs.append((start, len(indices)))
    return runs


def ref_layout(width, height, title):
    top = REF_TOP_TITLE if title else REF_TOP_PLAIN
    left = REF_LEFT
    right = float(width) - REF_RIGHT_MARGIN
    bottom = float(height) - REF_BOTTOM_MARGIN
    return left, right, top, bottom


def ref_x(index, count, left, right):
    if count <= 1:
        return (left + right) / 2.0
    return left + index / (count - 1) * (right - left)


def ref_y(value, lo, hi, top, bottom):
    if hi <= lo:
        return (top + bottom) / 2.0
    raw = top + (hi - value) / (hi - lo) * (bottom - top)
    return min(bottom, max(top, raw))


def ref_series_points(values, ticks, count, width, height, title="", limit=2000):
    left, right, top, bottom = ref_layout(width, height, title)
    indices = ref_sample_indices(count, limit)
    runs = ref_plot_runs(values, indices)
    lo, hi = ticks[0], ticks[-1]
    segments = []
    for start, end in runs:
        segment = []
        for position in range(start, end):
            index = indices[position]
            x = ref_x(index, count, left, right)
            y = ref_y(float(values[index]), lo, hi, top, bottom)
            segment.append((float(f"{x:.1f}"), float(f"{y:.1f}")))
        segments.append(segment)
    return segments


def make_dates(count, start=date(2020, 1, 1)):
    return [(start + timedelta(days=i)).strftime("%Y%m%d") for i in range(count)]


def test_helpers_hand_computed():
    assert plot._pct_label(0.12) == "12%"
    assert plot._pct_label(-0.05) == "-5%"
    assert plot._pct_label(0.0) == "0%"
    assert plot._pct_label(0.0123) == "1.23%"
    assert plot._pct_label(-0.00123) == "-0.12%"

    assert plot._group_colors(2) == ["#08306b", "#c6dbef"]
    colors = plot._group_colors(10)
    assert colors[0] == "#08306b"
    assert colors[-1] == "#c6dbef"
    assert sum(ref_hex_to_rgb(colors[0])) < sum(ref_hex_to_rgb(colors[-1]))

    np.testing.assert_allclose(
        plot._nice_ticks(0.0, 0.05),
        [0.0, 0.01, 0.02, 0.03, 0.04, 0.05],
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        plot._sample_indices(10, 5), [0, 2, 4, 7, 9], rtol=0, atol=0
    )
    values = np.array([1.0, np.nan, 2.0, 3.0, np.nan, 4.0])
    assert plot._plot_runs(values, np.arange(len(values))) == [(0, 1), (2, 4), (5, 6)]


def test_helpers_match_reference():
    rng = np.random.default_rng(0)
    for count in (0, 1, 10, 17, 2000, 2001, 5000):
        np.testing.assert_allclose(
            plot._sample_indices(count, 2000),
            ref_sample_indices(count, 2000),
            rtol=0,
            atol=0,
        )

    for count in (1, 2, 5, 10):
        assert plot._group_colors(count) == ref_group_colors(count)

    for _ in range(20):
        lo = float(rng.normal())
        hi = lo + float(rng.uniform(0.001, 2.0))
        np.testing.assert_allclose(
            plot._nice_ticks(lo, hi),
            ref_nice_ticks(lo, hi),
            rtol=0,
            atol=1e-12,
        )

    values = rng.normal(size=77)
    values[[3, 4, 20, 55]] = np.nan
    indices = plot._sample_indices(77, 23)
    assert plot._plot_runs(values, indices) == ref_plot_runs(values, list(indices))

    big = rng.normal(size=5000)
    big[1000:1005] = np.nan
    big[3000] = np.nan
    sampled = plot._sample_indices(5000, 2000)
    assert plot._plot_runs(big, sampled) == ref_plot_runs(big, list(sampled))


def test_empty_and_all_nan():
    dates = ["20200101", "20200102", "20200103", "20200106"]
    nan_groups = [np.full(4, np.nan) for _ in range(5)]
    svg = plot.pnl_chart(nan_groups, np.full(4, np.nan), dates, title="空 & <图>")
    root = parse(svg)
    assert not polylines(root)
    texts = [element.text for element in root.findall(f".//{NS}text")]
    assert "空 & <图>" in texts
    assert any(
        element.tag == f"{NS}rect" and element.get("fill") == "none" for element in root
    )

    empty = parse(plot.pnl_chart([], [], []))
    assert not polylines(empty)
    assert empty.tag == f"{NS}svg"


def test_title_escaping():
    dates = ["20200101", "20200102"]
    groups = [np.array([0.0, 0.01]) for _ in range(5)]
    svg = plot.pnl_chart(groups, np.array([0.0, 0.02]), dates, title="A & B < C > D")
    root = parse(svg)
    texts = [element.text for element in root.findall(f".//{NS}text")]
    assert "A & B < C > D" in texts


def test_small_example_coordinates():
    groups = [
        np.array([0.00, 0.02, 0.04, 0.06]),
        np.array([0.00, 0.01, 0.02, 0.03]),
        np.array([0.00, 0.00, 0.00, 0.00]),
        np.array([0.00, -0.01, -0.02, -0.03]),
        np.array([0.00, -0.02, -0.04, -0.06]),
    ]
    long_short = np.array([0.0, 0.01, 0.03, 0.09])
    dates = ["20200101", "20200102", "20200103", "20200106"]
    root = parse(plot.pnl_chart(groups, long_short, dates, width=500, height=300))
    assert len(polylines(root)) == 6

    by_name = polyline_points_by_name(root)
    np.testing.assert_allclose(
        by_name["G1"][0],
        [(64.0, 168.0), (153.3, 136.0), (242.7, 104.0), (332.0, 72.0)],
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        by_name["G5"][0],
        [(64.0, 168.0), (153.3, 200.0), (242.7, 232.0), (332.0, 264.0)],
        rtol=0,
        atol=0,
        equal_nan=True,
    )
    np.testing.assert_allclose(
        by_name["L/S"][0],
        [(64.0, 168.0), (153.3, 152.0), (242.7, 120.0), (332.0, 24.0)],
        rtol=0,
        atol=0,
        equal_nan=True,
    )

    zero_lines = [
        element
        for element in root.findall(f".//{NS}line")
        if element.get("stroke") == "#aaaaaa"
    ]
    assert len(zero_lines) == 1
    np.testing.assert_allclose(
        [float(zero_lines[0].get("y1"))], [168.0], rtol=0, atol=0
    )


def test_nan_breaks_and_counts():
    dates = ["20200101", "20200102", "20200103", "20200106"]
    groups = [np.zeros(4) for _ in range(5)]
    groups[0] = np.array([0.0, np.nan, 0.04, 0.06])
    groups[4] = np.full(4, np.nan)
    svg = plot.pnl_chart(
        groups, np.zeros(4), dates, extra={"x": np.array([0.0, np.nan, 0.01, 0.02])}
    )
    root = parse(svg)
    by_name = polyline_points_by_name(root)
    assert len(by_name["G1"]) == 2
    assert "G5" not in by_name
    assert len(by_name["x"]) == 2
    assert len(by_name["L/S"]) == 1

    counts = {}
    for element in polylines(root):
        counts[element.get("data-name")] = counts.get(element.get("data-name"), 0) + 1
    assert counts == {"G1": 2, "G2": 1, "G3": 1, "G4": 1, "x": 2, "L/S": 1}


def test_colors_legend_and_order():
    dates = make_dates(4)
    rng = np.random.default_rng(0)
    groups = [np.cumsum(rng.normal(scale=0.01, size=4)) for _ in range(10)]
    long_short = np.cumsum(rng.normal(scale=0.01, size=4))
    root = parse(
        plot.pnl_chart(
            groups, long_short, dates, extra={"bench": np.cumsum(rng.normal(size=4))}
        )
    )
    by_name = {}
    for element in polylines(root):
        by_name.setdefault(element.get("data-name"), []).append(element)

    assert by_name["L/S"][0].get("stroke") == "#d94801"
    assert float(by_name["L/S"][0].get("stroke-width")) == 2.4
    assert by_name["bench"][0].get("stroke") == "#444"
    assert by_name["bench"][0].get("stroke-dasharray") == "6,4"

    g1 = by_name["G1"][0].get("stroke")
    g10 = by_name["G10"][0].get("stroke")
    assert sum(ref_hex_to_rgb(g1)) < sum(ref_hex_to_rgb(g10))
    assert by_name["G1"][0].get("stroke") == "#08306b"
    assert by_name["G10"][0].get("stroke") == "#c6dbef"

    texts = [element.text for element in root.findall(f".//{NS}text")]
    for name in ["G1", "G10", "L/S", "bench"]:
        assert name in texts
    assert polylines(root)[-1].get("data-kind") == "long_short"


def test_random_chart_matches_reference():
    rng = np.random.default_rng(0)
    count = 120
    dates = make_dates(count)
    groups = [np.cumsum(rng.normal(scale=0.01, size=count)) for _ in range(5)]
    long_short = np.cumsum(rng.normal(scale=0.01, size=count))
    benchmark = np.cumsum(rng.normal(scale=0.01, size=count))
    for array in groups + [long_short, benchmark]:
        array[rng.choice(count, size=5, replace=False)] = np.nan
    root = parse(plot.pnl_chart(groups, long_short, dates, extra={"bench": benchmark}))
    actual = polyline_points_by_name(root)

    all_arrays = groups + [long_short, benchmark]
    finite = np.concatenate([a[np.isfinite(a)] for a in all_arrays])
    lo = min(0.0, float(finite.min()))
    hi = max(0.0, float(finite.max()))
    ticks = ref_nice_ticks(lo, hi)

    expected = {}
    for i, array in enumerate(groups):
        expected[f"G{i + 1}"] = ref_series_points(array, ticks, count, 960, 520)
    expected["bench"] = ref_series_points(benchmark, ticks, count, 960, 520)
    expected["L/S"] = ref_series_points(long_short, ticks, count, 960, 520)

    for name, segments in expected.items():
        assert len(actual[name]) == len(segments)
        for got, want in zip(actual[name], segments, strict=True):
            np.testing.assert_allclose(
                np.asarray(got), np.asarray(want), rtol=0, atol=0, equal_nan=True
            )


def test_coordinate_bounds():
    rng = np.random.default_rng(0)
    count = 64
    dates = make_dates(count)
    groups = [np.cumsum(rng.normal(scale=0.02, size=count)) for _ in range(10)]
    long_short = np.cumsum(rng.normal(scale=0.02, size=count))
    root = parse(plot.pnl_chart(groups, long_short, dates, title="t"))

    width = 960.0
    height = 520.0
    for element in root.iter():
        for attr in ("x", "x1", "x2", "cx"):
            if attr in element.attrib:
                value = float(element.get(attr))
                assert 0.0 <= value <= width
        for attr in ("y", "y1", "y2", "cy"):
            if attr in element.attrib:
                value = float(element.get(attr))
                assert 0.0 <= value <= height
        if element.tag == f"{NS}polyline":
            for x, y in points_of(element):
                assert 0.0 <= x <= width
                assert 0.0 <= y <= height


def test_downsample_limit():
    rng = np.random.default_rng(0)
    count = 5000
    dates = make_dates(count)
    groups = [np.cumsum(rng.normal(scale=0.01, size=count)) for _ in range(5)]
    long_short = np.cumsum(rng.normal(scale=0.01, size=count))
    root = parse(plot.pnl_chart(groups, long_short, dates))
    for element in polylines(root):
        assert len(points_of(element)) <= 2000


def test_example_file():
    rng = np.random.default_rng(0)
    dates = []
    current = date(2022, 1, 3)
    while current <= date(2024, 12, 31):
        if current.weekday() < 5:
            dates.append(current.strftime("%Y%m%d"))
        current += timedelta(days=1)
    count = len(dates)
    groups = [np.cumsum(rng.normal(scale=0.008, size=count)) for _ in range(10)]
    long_short = np.cumsum(rng.normal(scale=0.008, size=count))
    svg = plot.pnl_chart(
        groups,
        long_short,
        dates,
        title="Group PnL & <L/S>",
        extra={"benchmark": np.cumsum(rng.normal(scale=0.008, size=count))},
    )
    path = Path("/private/tmp/gbt_plot_example.svg")
    path.write_text(svg, encoding="utf-8")
    root = parse(path.read_text(encoding="utf-8"))
    assert root.tag == f"{NS}svg"
    assert len(polylines(root)) == 12


def test_inputs_unchanged_and_single_point():
    groups = [np.array([0.0, np.nan, 0.02]) for _ in range(5)]
    long_short = np.array([0.0, 0.01, np.nan])
    copies = [array.copy() for array in groups]
    long_short_copy = long_short.copy()
    plot.pnl_chart(groups, long_short, ["20200101", "20200102", "20200103"])
    for array, copy in zip(groups, copies, strict=True):
        np.testing.assert_allclose(array, copy, rtol=0, atol=0, equal_nan=True)
    np.testing.assert_allclose(
        long_short, long_short_copy, rtol=0, atol=0, equal_nan=True
    )

    single = parse(
        plot.pnl_chart(
            [np.array([0.01]) for _ in range(5)],
            np.array([0.02]),
            ["20200101"],
        )
    )
    assert len(polylines(single)) == 6
