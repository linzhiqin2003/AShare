"""分组累计盈亏的 SVG 折线图，仅依赖 numpy 与标准库。"""

from __future__ import annotations

import math
from xml.sax.saxutils import escape

import numpy as np

_GROUP_START = (0x08, 0x30, 0x6B)
_GROUP_END = (0xC6, 0xDB, 0xEF)
_LONG_SHORT = "#d94801"
_EXTRA = "#444"
_FONT = "-apple-system, Helvetica, Arial, sans-serif"
_MAX_POINTS = 2000
_LEFT = 64.0
_RIGHT = 168.0
_TOP_TITLE = 44.0
_TOP_PLAIN = 24.0
_BOTTOM = 36.0


def _rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*tuple(int(round(v)) for v in rgb))


def _group_colors(count):
    """返回由深到浅的蓝紫色序列，G1 最深。"""
    start = np.asarray(_GROUP_START, dtype=float)
    end = np.asarray(_GROUP_END, dtype=float)
    colors = []
    for i in range(count):
        t = i / (count - 1) if count > 1 else 0.0
        colors.append(_rgb_to_hex(start + (end - start) * t))
    return colors


def _pct_label(value):
    """把小数格式化成百分比刻度文本。"""
    pct = value * 100.0
    if abs(pct) < 1e-9:
        return "0%"
    if abs(pct - round(pct)) < 1e-6:
        return f"{int(round(pct))}%"
    if abs(pct * 10.0 - round(pct * 10.0)) < 1e-6:
        return f"{pct:.1f}%"
    return f"{pct:.2f}%"


def _nice_ticks(lo, hi, target=6):
    """返回覆盖 [lo, hi] 的 5 到 7 个整齐刻度。"""
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
            if count < 1 or not 5 <= count <= 7:
                continue
            key = (abs(count - target), step)
            if best_key is None or key < best_key:
                best_key = key
                best_ticks = [start + i * step for i in range(count)]
    if best_ticks is None:
        step = span / 5.0
        best_ticks = [lo + i * step for i in range(6)]
    return [0.0 if abs(v) < 1e-12 else float(v) for v in best_ticks]


def _sample_indices(count, limit=_MAX_POINTS):
    """等间隔取点，最多 limit 个，并保留最后一个点。"""
    if count <= limit:
        return np.arange(count, dtype=int)
    indices = np.rint(np.linspace(0, count - 1, limit)).astype(int)
    return np.unique(indices)


def _plot_runs(values, indices):
    """按有效值分段，返回每段在 indices 中的半开区间。"""
    runs = []
    start = None
    previous = None
    for pos, raw in enumerate(indices):
        index = int(raw)
        if not np.isfinite(values[index]):
            if start is not None:
                runs.append((start, pos))
                start = None
            previous = None
            continue
        if start is None:
            start = pos
        elif previous is not None and index - previous > 1:
            gap = values[previous + 1 : index]
            if gap.size and not bool(np.all(np.isfinite(gap))):
                runs.append((start, pos))
                start = pos
        previous = index
    if start is not None:
        runs.append((start, len(indices)))
    return runs


def _layout(width, height, title):
    top = _TOP_TITLE if title else _TOP_PLAIN
    top = max(0.0, min(top, float(height) - 10.0))
    left = min(_LEFT, max(0.0, float(width) - 2.0))
    right = float(width) - _RIGHT
    if right <= left + 1.0:
        right = left + 1.0
    bottom = float(height) - _BOTTOM
    bottom = max(top + 1.0, min(bottom, float(height) - 1.0))
    return left, right, top, bottom


def _coord(value):
    text = f"{value:.1f}"
    return "0.0" if text == "-0.0" else text


def _fit(values, count):
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == count:
        return array
    out = np.full(count, np.nan)
    used = min(count, array.size)
    if used:
        out[:used] = array[:used]
    return out


def _map_x(index, count, left, right):
    if count <= 1:
        return (left + right) / 2.0
    return left + index / (count - 1) * (right - left)


def _map_y(value, lo, hi, top, bottom):
    if hi <= lo:
        return (top + bottom) / 2.0
    y = top + (hi - value) / (hi - lo) * (bottom - top)
    return min(bottom, max(top, y))


def _text(value):
    return escape(str(value))


def _attr(value):
    return escape(str(value), {'"': "&quot;"})


def pnl_chart(
    groups, long_short, dates, *, title="", extra=None, width=960, height=520
):
    """画分组累计盈亏、多空与额外序列的 SVG。"""
    dates = list(dates)
    count = len(dates)
    group_arrays = [_fit(series, count) for series in groups]
    long_short_array = _fit(long_short, count)
    extra_items = list((extra or {}).items())
    extra_arrays = [(name, _fit(series, count)) for name, series in extra_items]
    all_arrays = group_arrays + [long_short_array] + [a for _, a in extra_arrays]

    left, right, top, bottom = _layout(width, height, title)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:g}" '
        f'height="{height:g}" viewBox="0 0 {width:g} {height:g}" '
        f'font-family="{_FONT}">',
        f'<rect x="0.0" y="0.0" width="{width:g}" height="{height:g}" fill="#ffffff"/>',
    ]

    def frame():
        return (
            f'<rect x="{_coord(left)}" y="{_coord(top)}" '
            f'width="{_coord(right - left)}" height="{_coord(bottom - top)}" '
            'fill="none" stroke="#cccccc" stroke-width="1"/>'
        )

    def title_text():
        if not title:
            return None
        return (
            f'<text x="{_coord(left)}" y="{_coord(min(float(height) - 1.0, 30.0))}" '
            'font-size="15" font-weight="bold" fill="#222222">'
            f"{_text(title)}</text>"
        )

    has_data = count > 0 and any(bool(np.isfinite(a).any()) for a in all_arrays)
    if not has_data:
        label = title_text()
        if label is not None:
            parts.append(label)
        parts.append(frame())
        parts.append("</svg>")
        return "\n".join(parts)

    finite = np.concatenate(
        [a[np.isfinite(a)] for a in all_arrays if np.isfinite(a).any()]
    )
    lo = min(0.0, float(finite.min()))
    hi = max(0.0, float(finite.max()))
    ticks = _nice_ticks(lo, hi)
    axis_lo, axis_hi = ticks[0], ticks[-1]

    for tick in ticks:
        y = _map_y(tick, axis_lo, axis_hi, top, bottom)
        color = "#aaaaaa" if abs(tick) < 1e-12 else "#e6e6e6"
        parts.append(
            f'<line x1="{_coord(left)}" y1="{_coord(y)}" '
            f'x2="{_coord(right)}" y2="{_coord(y)}" '
            f'stroke="{color}" stroke-width="1"/>'
        )

    indices = _sample_indices(count)

    def series_polylines(name, array, color, stroke_width, dash, kind):
        for start, end in _plot_runs(array, indices):
            points = []
            for pos in range(start, end):
                index = int(indices[pos])
                x = _map_x(index, count, left, right)
                y = _map_y(float(array[index]), axis_lo, axis_hi, top, bottom)
                points.append(f"{_coord(x)},{_coord(y)}")
            if not points:
                continue
            attrs = [
                f'points="{" ".join(points)}"',
                'fill="none"',
                f'stroke="{color}"',
                f'stroke-width="{stroke_width:g}"',
                'stroke-linejoin="round"',
                'stroke-linecap="round"',
                f'data-name="{_attr(name)}"',
                f'data-kind="{kind}"',
            ]
            if dash:
                attrs.append(f'stroke-dasharray="{dash}"')
            parts.append("<polyline " + " ".join(attrs) + "/>")

    group_colors = _group_colors(len(group_arrays))
    for i, array in enumerate(group_arrays):
        series_polylines(f"G{i + 1}", array, group_colors[i], 1.4, None, "group")
    for name, array in extra_arrays:
        series_polylines(name, array, _EXTRA, 1.6, "6,4", "extra")
    series_polylines("L/S", long_short_array, _LONG_SHORT, 2.4, None, "long_short")

    parts.append(frame())

    for tick in ticks:
        y = _map_y(tick, axis_lo, axis_hi, top, bottom)
        parts.append(
            f'<line x1="{_coord(max(0.0, left - 4.0))}" y1="{_coord(y)}" '
            f'x2="{_coord(left)}" y2="{_coord(y)}" stroke="#999999" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{_coord(max(0.0, left - 8.0))}" '
            f'y="{_coord(min(float(height) - 1.0, y + 4.0))}" '
            'font-size="11" fill="#555555" text-anchor="end">'
            f"{_pct_label(tick)}</text>"
        )

    year_marks = []
    previous_year = None
    for i in range(count):
        year = str(dates[i])[:4]
        if i == 0 or year != previous_year:
            year_marks.append((i, year))
            previous_year = year
    if len(year_marks) > 12:
        year_marks = year_marks[::2]
    for i, year in year_marks:
        x = _map_x(i, count, left, right)
        tick_y = min(float(height) - 1.0, bottom + 3.0)
        text_y = min(float(height) - 1.0, bottom + 16.0)
        parts.append(
            f'<line x1="{_coord(x)}" y1="{_coord(bottom)}" '
            f'x2="{_coord(x)}" y2="{_coord(tick_y)}" stroke="#999999" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{_coord(x)}" y="{_coord(text_y)}" font-size="11" '
            f'fill="#555555" text-anchor="middle">{_text(year)}</text>'
        )

    legend_items = [(f"G{i + 1}", group_colors[i]) for i in range(len(group_arrays))]
    legend_items.append(("L/S", _LONG_SHORT))
    legend_items.extend((name, _EXTRA) for name, _ in extra_arrays)
    if legend_items:
        legend_x = min(right + 16.0, max(0.0, float(width) - 40.0))
        available = max(1.0, float(height) - 20.0)
        row_height = min(18.0, available / len(legend_items))
        for i, (name, color) in enumerate(legend_items):
            center = 20.0 + row_height * (i + 0.5)
            parts.append(
                f'<rect x="{_coord(legend_x)}" y="{_coord(center - 2.0)}" '
                f'width="16.0" height="4.0" fill="{color}"/>'
            )
            parts.append(
                f'<text x="{_coord(min(float(width) - 1.0, legend_x + 22.0))}" '
                f'y="{_coord(min(float(height) - 1.0, center + 4.0))}" '
                f'font-size="11" fill="#333333">{_text(name)}</text>'
            )

    label = title_text()
    if label is not None:
        parts.insert(2, label)
    parts.append("</svg>")
    return "\n".join(parts)
