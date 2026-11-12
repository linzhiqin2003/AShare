# ruff: noqa: E402
"""第二轮复核后补的测试：深链文本化、缓存键中的求值器实现指纹。"""

from pathlib import Path

import panelops

import ashare

ROOT = Path(__file__).resolve().parents[2]
assert Path(ashare.__file__).resolve().is_relative_to(ROOT)
assert Path(panelops.__file__).resolve().is_relative_to(ROOT)

import numpy as np

import ashare.factor as af
from ashare.factor import engine


class Source:
    def __init__(self, x):
        self.x = x
        self.dates = list(range(x.shape[0]))
        self.symbols = list(range(x.shape[1]))
        self.version = "v1"

    def __getitem__(self, name):
        return self.x


def test_canonical_and_text_of_small_expression():
    x = af.field("x")
    e = af.call("ts_mean", x + 1.0, window=3) * x
    assert str(e) == ("(ts_mean((x + 1.0), min_periods=3, window=3) * x)")
    assert e.canonical() == [
        2,
        "multiply",
        [
            [
                2,
                "ts_mean",
                [
                    [
                        2,
                        "add",
                        [
                            [
                                2,
                                "field",
                                [],
                                [
                                    ("broadcast", False),
                                    ("dtype", "float64"),
                                    ("value", "x"),
                                ],
                            ],
                            [2, "constant", [], [("value", 1.0)]],
                        ],
                        [],
                    ]
                ],
                [("min_periods", 3), ("window", 3)],
            ],
            [
                2,
                "field",
                [],
                [("broadcast", False), ("dtype", "float64"), ("value", "x")],
            ],
        ],
        [],
    ]
    assert str(af.field("m", broadcast=True)) == "m[market]"


def test_canonical_and_text_of_3000_level_chain():
    x = af.field("x")
    e = x
    for i in range(3000):
        e = af.call("ts_delay", e, periods=1) if i % 2 else e + 1.0
    text = str(e)
    assert text.count("ts_delay(") == 1500 and text.count(" + 1.0)") == 1500
    assert text.startswith("ts_delay((ts_delay((") and text.endswith("periods=1)")
    node, depth = e.canonical(), 0
    while node[2]:
        node, depth = node[2][0], depth + 1
    assert depth == 3000 and node[1] == "field"


def test_cache_key_tracks_evaluator_implementation(tmp_path, monkeypatch):
    source = Source(np.arange(24.0).reshape(8, 3))
    expression = af.call("ts_mean", af.field("x"), window=3)
    ev = af.Evaluator(tmp_path)
    first = ev.evaluate(expression, source)
    ev.evaluate(expression, source)
    assert ev.cache_hit
    monkeypatch.setattr(engine, "_implementation_id", lambda: "changed-evaluator")
    again = ev.evaluate(expression, source)
    assert not ev.cache_hit
    np.testing.assert_array_equal(again, first)
    assert len(list(tmp_path.glob("*.npy"))) == 2
