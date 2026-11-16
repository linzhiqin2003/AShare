"""§9.3：审查已核实的普通读取行为，冻结本轮修复前 80 个数组的字节摘要。"""

import hashlib
import json
from pathlib import Path

import polars as pl
import pytest

from ashare.data.spec import open
from test.spec.conftest import assert_good
from test.spec.test_contract import add_field, change, hourly

BASELINE = Path(__file__).with_name("r3_panel_baseline.json")
FIELDS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "has_quote",
    "can_buy",
    "can_sell",
    "can_buy_open",
    "can_sell_open",
]


def prepare(root, scenario):
    if scenario == "missing_null":
        change(
            root,
            lambda f: f.slice(1).with_columns(
                pl.when(pl.int_range(pl.len()) == 4)
                .then(None)
                .otherwise(pl.col("close"))
                .alias("close")
            ),
        )
    elif scenario == "hourly":
        hourly(root)
    elif scenario == "limits":
        add_field(
            root,
            dict(name="suspended", type="boolean"),
            pl.int_range(pl.len()) % 7 == 0,
        )
        add_field(
            root,
            dict(name="limit_up", type="number"),
            pl.when(pl.int_range(pl.len()) % 4 == 0)
            .then(pl.col("open"))
            .otherwise(pl.col("high") + 1),
        )
        add_field(
            root,
            dict(name="limit_down", type="number"),
            pl.when(pl.int_range(pl.len()) % 5 == 0)
            .then(pl.col("open"))
            .otherwise(pl.col("low") - 1),
        )
    assert_good(root)


def capture(root):
    ds = open(root)
    result = []
    for bounds in (
        {},
        dict(start="2024-01-04", end="2024-01-10", symbols=["600002.SH", "000001.SZ"]),
    ):
        panel = ds.panel(FIELDS, **bounds)
        assert panel.version == ds.version
        result.append(
            dict(
                arrays={
                    name: dict(
                        dtype=str(panel[name].dtype),
                        shape=list(panel[name].shape),
                        sha256=hashlib.sha256(panel[name].tobytes()).hexdigest(),
                    )
                    for name in FIELDS
                },
                dates=[str(x) for x in panel.dates],
                symbols=panel.symbols.tolist(),
                metadata={
                    name: dict(desc) for name, desc in panel.field_metadata.items()
                },
            )
        )
    return result


@pytest.mark.parametrize("scenario", ["normal", "missing_null", "hourly", "limits"])
def test_s9_r3_80_ordinary_arrays_bitwise_unchanged(package, scenario):
    prepare(package, scenario)
    expected = json.loads(BASELINE.read_text())[scenario]
    assert len(expected) == 2 and sum(len(x["arrays"]) for x in expected) == 20
    assert (
        capture(package) == expected
    )  # 包含 NaN 字节、轴、元数据；版本另与清单逐次核对
