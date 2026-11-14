"""`ashare.data` 公开的只读小接口：数据目录、原始库清单、指数简称、字段单位。

来源：2026-11-14 合入 GPU 粗筛时全量测试发现 `test_public_boundary` 自 11-13 平台合入起一直失败——
数据页、状态栏和模拟盘接口直接用了 `ashare.data.store` / `ashare.data.catalog` 的内部名字。
改为经这些公开接口读取；这里守住它们与内部实现给出相同的结果。"""

from ashare import data
from ashare.data.catalog import BENCHMARKS, FIELDS, INDEXES
from ashare.data.store import Store, atomic_json


def test_market_root_resolves_under_the_data_dir(tmp_path):
    assert data.market_root(tmp_path) == tmp_path.resolve() / "cn"
    assert data.market_root(tmp_path, "hk") == tmp_path.resolve() / "hk"


def test_read_manifest_empty_without_a_store_and_read_only(tmp_path):
    assert data.read_manifest(tmp_path) == {}
    assert not (tmp_path / "cn").exists()  # 没有清单时不建目录、不建库
    root = tmp_path / "cn"
    root.mkdir()
    atomic_json(
        root / "manifest.json",
        {"daily/x": dict(dataset="daily", partition="x", status="ok")},
    )
    assert (
        data.read_manifest(tmp_path) == Store(tmp_path, "cn", readonly=True).manifest()
    )
    assert data.read_manifest(tmp_path)["daily/x"]["status"] == "ok"


def test_index_names_cover_benchmarks_and_universe_indexes():
    names = data.index_names()
    for code, name in BENCHMARKS.items():
        assert names[code] == name  # 基准名单的中文简称优先
    for alias, code in INDEXES.items():
        assert code in names
        if code not in BENCHMARKS:
            assert names[code] == alias
    names["000001.SH"] = "改了"
    assert data.index_names()["000001.SH"] == BENCHMARKS["000001.SH"]  # 每次给新字典


def test_field_unit():
    name, spec = next(iter(FIELDS.items()))
    assert data.field_unit(name) == spec.unit
    assert data.field_unit("没有这个字段") == ""
