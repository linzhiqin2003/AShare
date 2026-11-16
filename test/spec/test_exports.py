"""第 10 节：真实写入流程产生的合成夹具导出/读回；NaN 位置也逐格比较。"""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ashare.data.spec import build, open
from ashare.data.spec.export_cn import CNAdapter
from ashare.data.spec.export_crypto import (
    HOUR_MAPPING,
    MINUTE_MAPPING,
    CryptoAdapter,
    MinuteAdapter,
)
from test.spec.conftest import assert_good


@pytest.fixture(scope="module")
def synthetic_cn(tmp_path_factory):
    from test.fixtures.synthetic_datadir import build_datadir

    return build_datadir(tmp_path_factory.mktemp("cn-source"))


def test_s10_cn_bitwise_roundtrip(synthetic_cn, tmp_path):
    adapter = CNAdapter(synthetic_cn.root, start="20240101", end="20241231")
    target = tmp_path / "cn"
    build(adapter, target)
    assert_good(target)
    fields = list(adapter.mapping)
    panel = open(target).panel(fields)
    source = adapter.data.panel(
        [*adapter.mapping.values(), "has_quote"], adapter.dates[0], adapter.dates[-1]
    )
    assert np.array_equal(panel.dates, source.dates)
    assert np.array_equal(panel.symbols, source.symbols)
    assert np.array_equal(panel["has_quote"], source["has_quote"])
    for name, original in adapter.mapping.items():
        # 第 6、9.3 节：源面板缺行处的状态值不进入 bars；有效行逐位不变。
        expected = np.where(source["has_quote"], source[original].astype(float), np.nan)
        np.testing.assert_array_equal(panel[name], expected, strict=True)
    assert open(target).read("corporate_actions").height >= 2
    # §3.3 / R4：仅调整公告列自身的声明，不改变事件内容。
    fields = {
        f["name"]: f
        for r in adapter.describe()["resources"]
        if r["kind"] == "corporate_actions"
        for f in r["schema"]["fields"]
    }
    assert fields["available_at"]["knownAt"] == "end"
    assert all(
        f["knownAt"] == "column:available_at"
        for n, f in fields.items()
        if n != "available_at"
    )


def test_s10_crypto_hourly_bitwise_roundtrip(tmp_path):
    from ashare.data.crypto.dataset import write_panel

    fixture = Path(__file__).parents[1] / "fixtures/crypto"
    inputs = json.loads((fixture / "inputs.json").read_text())
    frame = pl.read_parquet(fixture / "panel.parquet")
    source = write_panel(
        tmp_path / "crypto-source",
        frame,
        profiles=inputs["profiles"],
        events=inputs["events"],
        data=inputs["data"],
        notes=[],
    )
    adapter = CryptoAdapter(source.root)
    target = tmp_path / "crypto-package"
    build(adapter, target)
    assert_good(target)
    panel = open(target).panel(list(HOUR_MAPPING))
    original = source.arrays(
        [*set(HOUR_MAPPING.values()), "quality", "lifecycle"],
        instruments=panel.symbols.tolist(),
    )
    assert panel.symbols.tolist() == sorted(source.meta["instruments"])
    assert np.array_equal(
        panel.dates.astype("datetime64[ms]").astype(np.int64), original["time"]
    )
    quote = (original["lifecycle"] == "active") & ((original["quality"] & 1) == 0)
    np.testing.assert_array_equal(panel["has_quote"], quote)
    for name, raw in HOUR_MAPPING.items():
        expected = np.where(quote, original[raw], np.nan).astype(float)
        np.testing.assert_array_equal(panel[name], expected)


def test_s10_validation_rejects_dropped_source_rows(
    synthetic_cn, tmp_path, monkeypatch
):
    # 来源：验收自审发现仅用输出 has_quote 比对，会把导出漏行误报为通过。
    from bench import validate_data_spec as validation

    monkeypatch.setattr(validation, "ROOT", tmp_path)
    original = CNAdapter.read
    dropped = False

    def drop_one(self, table, partition):
        nonlocal dropped
        frame = original(self, table, partition)
        if table == "bars" and frame.height and not dropped:
            dropped = True
            return frame.slice(1)
        return frame

    monkeypatch.setattr(CNAdapter, "read", drop_one)
    result = validation.validate(
        "cn",
        synthetic_cn.root / "cn",
        "2024-01-01",
        "2024-12-31",
        tmp_path / "tmp/dropped",
    )
    assert dropped
    assert result["status"] == "failed"
    assert result["quote_mask_mismatches"] == 1


def test_s10_minute_bitwise_roundtrip(tmp_path):
    from ashare.data.crypto import minute
    from test.crypto.test_minute import T, built

    source = tmp_path / "minute-source"
    built(source)
    adapter = MinuteAdapter(
        source, start="2025-03-01", end="2025-03-02", symbols=["BTCUSDT"], market="spot"
    )
    target = tmp_path / "minute-package"
    build(adapter, target)
    assert_good(target)
    panel = open(target).panel(list(MINUTE_MAPPING))
    original = minute.bars("BTCUSDT", "spot", "klines", T, T + 86400000, root=source)
    assert panel.shape == (1440, 1)
    assert panel["has_quote"].sum() == original.height == 2
    for name, raw in MINUTE_MAPPING.items():
        expected = np.full((1440, 1), np.nan)
        expected[:2, 0] = original[raw].to_numpy()
        np.testing.assert_array_equal(panel[name], expected)


def test_s10_committed_example_is_complete():
    target = Path(__file__).parents[2] / "docs/examples/datapackage"
    assert_good(target)
    panel = open(target).panel(["open", "close"])
    from ashare.factor import Evaluator, field

    np.testing.assert_array_equal(
        Evaluator().evaluate(field("close") / field("open"), panel),
        panel["close"] / panel["open"],
    )
    assert panel.shape == (40, 3)


def test_s10_cn_zero_limit_is_null(synthetic_cn, monkeypatch):
    # 来源：2024 全市场只读验证发现 23 个 lower_limit=0；正价格域中的无下限应输出 null。
    from ashare.data.reader import Panel

    adapter = CNAdapter(synthetic_cn.root, start="20240101", end="20241231")
    original = adapter.data.panel

    def zero_limit(*args, **kwargs):
        panel = original(*args, **kwargs)
        arrays = dict(panel._arrays)
        arrays["lower_limit"] = arrays["lower_limit"].copy()
        i, j = np.nonzero(panel["has_quote"])
        arrays["lower_limit"][i[0], j[0]] = 0.0
        return Panel(panel.dates, panel.symbols, arrays, panel.version)

    monkeypatch.setattr(adapter.data, "panel", zero_limit)
    frame = adapter.read("bars", adapter.months[0])
    assert frame["limit_down"].null_count() == 1
    assert not (frame["limit_down"] == 0).any()


def test_s9_csv_adapter_example(package, tmp_path, monkeypatch):
    import runpy

    examples = Path(__file__).parents[2] / "docs/examples"
    monkeypatch.syspath_prepend(str(examples))
    adapter_class = runpy.run_path(str(examples / "csv_adapter.py"))["CSVAdapter"]
    source = tmp_path / "csv"
    for path in package.rglob("*.parquet"):
        dest = source / path.relative_to(package).with_suffix(".csv")
        dest.parent.mkdir(parents=True, exist_ok=True)
        pl.read_parquet(path).write_csv(dest)
    target = tmp_path / "csv-package"
    build(adapter_class(source), target)
    assert_good(target)
    np.testing.assert_array_equal(
        open(target).panel("close")["close"], open(package).panel("close")["close"]
    )


def test_s10_hourly_quality_and_suspension_mapping(tmp_path):
    # 来源：第 6、9.3 节，来源停牌位必须接到公共 suspended 才能生成正确成交掩码。
    from ashare.data.crypto.dataset import write_panel

    fixture = Path(__file__).parents[1] / "fixtures/crypto"
    inputs = json.loads((fixture / "inputs.json").read_text())
    frame = pl.read_parquet(fixture / "panel.parquet").with_columns(
        (pl.col("quality") | (4 | 8 | 64 | 256)).alias("quality")
    )
    source = write_panel(
        tmp_path / "source",
        frame,
        profiles=inputs["profiles"],
        events=inputs["events"],
        data=inputs["data"],
        notes=[],
    )
    adapter = CryptoAdapter(source.root)
    bars = adapter.read("bars", adapter.months[0])
    assert "suspended" in bars.columns
    assert bars["suspended"].all()
    original = frame.filter(
        (pl.col("lifecycle") == "active") & ((pl.col("quality") & 1) == 0)
    ).sort(["instrument", "time"])
    expected = original["quality"].to_numpy().astype(np.uint32) * 65536 | np.uint32(
        1 | 2 | 8
    )
    np.testing.assert_array_equal(bars["quality"].to_numpy(), expected)
    target = tmp_path / "package"
    build(adapter, target)
    assert not open(target).panel(["can_buy", "can_sell"])["can_buy"].any()
