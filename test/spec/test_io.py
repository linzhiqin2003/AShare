"""DATA_SPEC 7、9：封存、原子发布、投影读取、缓存与 PanelSource 契约。"""

import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from ashare.data.spec import ValidationError, build, open, seal
from ashare.factor import Evaluator, field
from docs.examples.generate_datapackage import SyntheticAdapter
from test.spec.conftest import assert_good, read_manifest, resign, write_manifest
from test.spec.test_contract import change, hourly


def test_s9_build_existing_empty_directory(tmp_path):
    # 来源：D2 开发时生成示例的首次运行，空目标没有清单而失败。
    target = tmp_path / "empty"
    target.mkdir()
    assert build(SyntheticAdapter(), target)["published"]
    assert_good(target)


def test_s7_seal_reference_hash_and_created_exclusion(package):
    old = read_manifest(package)
    got = seal(package)
    assert got["version"] == old["version"]
    body = {k: v for k, v in got.items() if k not in {"version", "created"}}
    expected = hashlib.sha256(
        json.dumps(
            body, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    assert got["version"] == expected
    assert got["created"] != old["created"]
    assert_good(package)


def test_s7_atomic_seal_failure_keeps_original(package, monkeypatch):
    module = importlib.import_module("ashare.data.spec.seal")
    original = (package / "datapackage.json").read_bytes()

    def failed(*args):
        raise OSError("injected rename failure")

    monkeypatch.setattr(module.os, "replace", failed)
    with pytest.raises(OSError, match="injected"):
        seal(package)
    assert (package / "datapackage.json").read_bytes() == original
    assert sorted(p.name for p in package.iterdir()) == [
        "bars",
        "calendar",
        "datapackage.json",
        "instruments",
    ]


def test_s9_build_unchanged_partitions_and_invalid_not_published(tmp_path):
    adapter = SyntheticAdapter()
    target = tmp_path / "built"
    first = build(adapter, target)
    files = {
        p.relative_to(target): (p.stat().st_ino, p.stat().st_mtime_ns, p.read_bytes())
        for p in target.rglob("*")
        if p.is_file()
    }
    second = build(adapter, target)
    assert first["version"] == second["version"]
    assert second["reused"] == 3 and second["written"] == 0 and not second["published"]
    for name, snapshot in files.items():
        p = target / name
        assert (p.stat().st_ino, p.stat().st_mtime_ns, p.read_bytes()) == snapshot

    class Bad(SyntheticAdapter):
        def read(self, table, partition):
            f = super().read(table, partition)
            return (
                f.with_columns(pl.lit(float("nan")).alias("close"))
                if table == "bars"
                else f
            )

    with pytest.raises(ValidationError) as error:
        build(Bad(), target)
    assert any(
        x["code"] == "nan" and x["status"] == "fail"
        for x in error.value.report["checks"]
    )
    assert (target / "datapackage.json").read_bytes() == files[
        Path("datapackage.json")
    ][2]

    class Changed(SyntheticAdapter):
        def read(self, table, partition):
            f = super().read(table, partition)
            return f.with_columns(pl.col("volume") + 1) if table == "bars" else f

    third = build(Changed(), target)
    assert (
        third["version"] != first["version"]
        and third["reused"] == 2
        and third["written"] == 1
    )
    assert (target / "calendar/all.parquet").stat().st_ino == files[
        Path("calendar/all.parquet")
    ][0]


def test_s9_build_before_exchange_failure_keeps_target(package, monkeypatch):
    module = importlib.import_module("ashare.data.spec.build")
    before = (package / "datapackage.json").read_bytes()

    class Changed(SyntheticAdapter):
        def describe(self):
            result = super().describe()
            result["source"] += " second"
            return result

    def failed(*args):
        raise OSError("injected swap failure")

    monkeypatch.setattr(module, "_exchange", failed)
    with pytest.raises(OSError, match="swap"):
        build(Changed(), package)
    assert (package / "datapackage.json").read_bytes() == before


def test_s9_example_factor_and_readonly(package):
    ds = open(package)
    panel = ds.panel(["open", "close"])
    assert panel.shape == (40, 3)
    assert np.array_equal(
        Evaluator().evaluate(field("close") - field("open"), panel),
        np.full((40, 3), 0.5),
    )
    for value in (panel.dates, panel.symbols, panel["close"], panel["has_quote"]):
        assert not value.flags.writeable
    assert panel.version == ds.version


def test_s9_absent_row_vs_null_no_fill(package):
    change(
        package,
        lambda f: f.slice(1).with_columns(
            pl.when(pl.int_range(pl.len()) == 0)
            .then(None)
            .otherwise(pl.col("close"))
            .alias("close")
        ),
    )
    panel = open(package).panel("close")
    assert not panel["has_quote"][0, 0]
    assert panel["has_quote"][1, 0]
    assert np.isnan(panel["close"][:2, 0]).all()
    assert panel["close"][2, 0] == 10.75


def test_s9_projection_and_inclusive_daily_bounds(package, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("读取接口必须 lazy scan")

    monkeypatch.setattr(pl, "read_parquet", forbidden)
    ds = open(package)
    frame = ds.read("bars", "20240103", "2024-01-04", ["000001.SZ"], ["close"])
    assert frame.columns == ["symbol", "date", "close"] and frame.height == 2
    assert "PROJECT 3/7 COLUMNS" in ds._scan("bars", fields=["close"]).explain()
    panel = ds.panel("close", "20240103", "20240104", ["600002.SH", "000001.SZ"])
    assert panel.shape == (2, 2)
    assert panel.symbols.tolist() == ["600002.SH", "000001.SZ"]


def test_s7_hash_cache_mtime_and_expected_hash(package, monkeypatch):
    module = importlib.import_module("ashare.data.spec.reader")
    module._VERIFIED.clear()
    calls = []
    actual = module._hash_file

    def spy(path):
        calls.append(path)
        return actual(path)

    monkeypatch.setattr(module, "_hash_file", spy)
    ds = open(package)
    ds.read(fields=["close"])
    ds.read(fields=["open"])
    assert len(calls) == 1
    change(package, lambda f: f.with_columns(pl.col("volume") + 1))
    with pytest.raises(ValueError, match="大小|哈希"):
        ds.read()
    new = open(package)
    new.read()
    assert len(calls) >= 2
    m = read_manifest(package)
    m["resources"][0]["files"][0]["hash"] = "sha256:" + "f" * 64
    write_manifest(package, m)
    with pytest.raises(ValueError, match="哈希"):
        open(package).read()


def test_s9_partition_pruning_ignores_unread_corruption(package):
    path = package / "bars/2025.parquet"
    pl.read_parquet(package / "bars/2024.parquet").with_columns(
        pl.col("date").dt.offset_by("1y")
    ).write_parquet(path)
    resign(package)
    path.write_bytes(b"corrupt unread partition")
    ds = open(package)
    assert ds.read(start="2024-01-01", end="2024-12-31", fields="close").height == 120
    with pytest.raises(ValueError, match="大小|哈希"):
        ds.read(start="2025-01-01", end="2025-12-31")


def test_s9_hourly_regular_grid_and_half_open_end(package):
    hourly(package)
    ds = open(package)
    result = ds.panel("close", "2024-01-02", "2024-01-03T01:00:00Z", ["000001.SZ"])
    assert result.shape == (25, 1)
    assert result["has_quote"].sum() == 2
    assert np.isnan(result["close"][1:24]).all()
    assert ds.read(start="2024-01-02", end="2024-01-03").height == 3
    with pytest.raises(TypeError, match="epoch"):
        ds.read(start=1704153600000)


def test_s9_registered_ext_and_pointer(package, tmp_path, monkeypatch):
    root = tmp_path / "data-root"
    ext = root / "ext"
    ext.mkdir(parents=True)
    (ext / "LOCATION.json").write_text(
        json.dumps(
            dict(
                path=str(package.parent),
                moved_at="2026-01-01T00:00:00Z",
                files=3,
                bytes=1,
            )
        )
    )
    monkeypatch.setenv("ASHARE_DATA_DIR", str(root))
    assert open("package").version == open(package).version


def test_s9_info_resolves_default_unit_and_known_at(package):
    info = open(package).info()
    close = next(f for f in info["resources"][0]["fields"] if f["name"] == "close")
    assert close["unit"] == "CNY/share" and close["knownAt"] == "end"
    assert info["resources"][0]["range"] == ["2024-01-02", "2024-02-26"]


def test_s9_cli_check_codes(package, capsys):
    from ashare.data.spec.__main__ import main

    assert main(["check", str(package)]) == 0
    assert json.loads(capsys.readouterr().out)["ok"]
    m = read_manifest(package)
    m["source"] += " changed"
    write_manifest(package, m, version=False)
    assert main(["check", str(package)]) == 1
    capsys.readouterr()
    (package / "datapackage.json").write_text("[")
    assert main(["check", str(package)]) == 2


def test_s9_reuse_compares_schema_not_only_values(tmp_path):
    # 来源：D2 代码审查；Polars equals 对 int/float 同值返回 True。
    target = tmp_path / "types"
    build(SyntheticAdapter(), target)
    path = target / "bars/2024.parquet"
    pl.read_parquet(path).with_columns(pl.col("volume").cast(pl.Int64)).write_parquet(
        path
    )
    resign(target)
    result = build(SyntheticAdapter(), target)
    assert result["written"] == 1
    assert_good(target)


def test_s9_unchanged_content_repairs_invalid_manifest(tmp_path):
    # 来源：D2 代码审查；不能只信旧清单自称的 version 而跳过发布。
    target = tmp_path / "manifest"
    build(SyntheticAdapter(), target)
    m = read_manifest(target)
    m["created"] = "not-a-time"
    write_manifest(target, m)
    result = build(SyntheticAdapter(), target)
    assert result["published"]
    assert_good(target)


def test_s3_2_s9_crypto_calendar_month_grid(package):
    # 来源：D2 自审；规范 1M 与 Polars 的 1mo 语法不同。
    hourly(package)
    path = package / "bars/2024.parquet"
    frame = pl.read_parquet(path).with_columns(pl.col("ts").dt.truncate("1mo"))
    frame.unique(["symbol", "ts"], keep="first").sort(["symbol", "ts"]).write_parquet(
        path
    )
    m = read_manifest(package)
    m["resources"][0]["frequency"] = "1M"
    resign(package, m)
    assert_good(package)
    assert open(package).panel("close", "2024-01-01", "2024-03-01").shape == (2, 3)


def test_s9_panel_does_not_materialize_requested_long_table(package, monkeypatch):
    # 来源：D2 十倍数据量设计审查；panel 不应调用返回整段长表的 read 再复制到数组。
    from ashare.data.spec.reader import Package

    original = Package.read

    def bounded(self, table="bars", *args, **kwargs):
        if table == "bars":
            pytest.fail("panel 必须流式填充目标数组")
        return original(self, table, *args, **kwargs)

    monkeypatch.setattr(Package, "read", bounded)
    panel = open(package).panel("close")
    assert panel.shape == (40, 3)
    assert panel["has_quote"].all()
