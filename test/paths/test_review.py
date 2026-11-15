"""D1 独立审查回归；所有外置目录故意与数据集名称不同。"""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ashare import paths

from .test_entries import pointer
from .test_static import violations


@pytest.mark.parametrize("stage", ["before", "temporary", "renamed", "returned"])
@pytest.mark.parametrize("delete_source", [False, True])
def test_move_interrupt_publication(roots, monkeypatch, stage, delete_source):
    root, external = roots
    source, target = root / "dev", external / "renamed-dev"
    source.mkdir()
    (source / "payload").write_bytes(b"preserve source and verified target")
    original_write, original_replace = paths._write_pointer, paths.os.replace

    def write(folder, value):
        if stage == "before":
            raise KeyboardInterrupt
        original_write(folder, value)
        if stage == "returned":
            raise KeyboardInterrupt

    def replace(temporary, published):
        assert json.loads(Path(temporary).read_text())["path"] == str(target)
        assert not Path(published).exists()
        if stage == "temporary":
            raise KeyboardInterrupt
        original_replace(temporary, published)
        if stage == "renamed":
            raise KeyboardInterrupt

    monkeypatch.setattr(paths, "_write_pointer", write)
    monkeypatch.setattr(paths.os, "replace", replace)
    with pytest.raises(KeyboardInterrupt):
        paths.move("dev", target, delete_source=delete_source)
    published = (source / paths.POINTER).exists()
    assert not published or target.is_dir(), "已发布指针不能指向被清掉的目标"
    assert (source / "payload").read_bytes() == b"preserve source and verified target"
    assert not list(source.glob(".LOCATION-*.tmp"))
    assert not (root / ".move-dev.lock").exists()
    if stage in {"renamed", "returned"}:
        assert paths.dataset_dir("dev") == target
        assert (target / "payload").read_bytes() == (source / "payload").read_bytes()
    else:
        assert not published and not target.exists()


def test_datastatus_renamed_dataset(roots):
    import ashare.data as ad
    from ashare.platform.datastatus import DataStatus
    from bench.platform.synthetic import write_panel

    root, external = roots
    target = external / "renamed-cn"
    target.mkdir()
    pointer(root, "cn", target)
    write_panel(root)
    expected = ad.read_manifest(root)
    assert expected
    status = DataStatus(root)
    assert status.root == target
    assert status._manifest() == expected


@pytest.mark.parametrize("explicit_root", [False, True])
def test_cb_panel_renamed_dataset(roots, monkeypatch, explicit_root):
    import asyncio

    from ashare.data.cb_panel import build_cb_arrays
    from ashare.data.fund_sync import sync_funds
    from ashare.data.reader import Panel
    from ashare.data.store import Store
    from bench.platform.synthetic import write_panel

    # 复用现有纯合成行情源；这些旧夹具用同目录导入。
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "data"))
    from futures_fixtures import DAYS
    from test_fund_fixtures import STOCK, FakeFundTushare, stock_price

    root, external = roots
    target = external / "renamed-cn"
    target.mkdir()
    pointer(root, "cn", target)
    symbols = np.array(sorted(STOCK))
    close = np.array([[stock_price(s, i) for s in symbols] for i in range(len(DAYS))])
    write_panel(root, Panel(np.array(DAYS), symbols, {"close": close}, "synthetic-v1"))
    store = Store(root if explicit_root else None)
    asyncio.run(sync_funds(FakeFundTushare(), store, DAYS[0], DAYS[-1]))
    _, arrays, _, _, diagnostics = build_cb_arrays(store, np.array(DAYS))
    assert diagnostics["stock_panel_version"] == "synthetic-v1"
    assert np.isfinite(arrays["stock_close"]).any()


@pytest.mark.parametrize("name", ["platform", "paper"])
def test_cwd_defaults_outside_checkout(roots, monkeypatch, name):
    from ashare.platform.config import platform_dir
    from ashare.platform.paper_api import paper_root

    root, _ = roots
    monkeypatch.delenv("ASHARE_DATA_DIR")
    monkeypatch.chdir(root)
    # 仓库默认根也隔离，旧实现仍会因错选根而失败，不触碰真实库。
    monkeypatch.setattr(paths, "DEFAULT_DATA_ROOT", root / "checkout-data")
    result = platform_dir() if name == "platform" else paper_root()
    assert result == Path("data") / name
    assert not result.exists()


@pytest.mark.parametrize("name", ["platform", "paper"])
def test_cwd_pointer_and_overrides(roots, monkeypatch, name):
    from ashare.platform.config import platform_dir
    from ashare.platform.paper_api import paper_root

    root, external = roots
    monkeypatch.chdir(root)
    monkeypatch.delenv("ASHARE_DATA_DIR")
    monkeypatch.setattr(paths, "DEFAULT_DATA_ROOT", root / "checkout-data")
    target = external / f"renamed-{name}"
    target.mkdir()
    pointer(root / "data", name, target)
    resolve = platform_dir if name == "platform" else paper_root
    assert resolve() == target
    target.rmdir()
    with pytest.raises(paths.DataLocationError, match=name):
        resolve()
    monkeypatch.setenv("ASHARE_DATA_DIR", str(root / "alternative"))
    assert resolve() == root / "alternative" / name
    monkeypatch.setenv(paths.DATASETS[name].env, str(external / "chosen"))
    assert resolve() == external / "chosen"


@pytest.mark.parametrize("module", ["e2e", "rules"])
def test_bench_main_checkout_fallback(roots, monkeypatch, module):
    from bench import rules_examples
    from bench.e2e import common

    root, external = roots
    benchmark = common if module == "e2e" else rules_examples
    monkeypatch.delenv("ASHARE_DATA_DIR")
    monkeypatch.delenv("ASHARE_BENCH_DATA", raising=False)
    monkeypatch.setattr(benchmark, "ROOT", root)
    monkeypatch.setattr(paths, "DEFAULT_DATA_ROOT", root / "data")
    calls = []

    def git(args, **kwargs):
        calls.append((args, kwargs["cwd"]))
        return SimpleNamespace(stdout=str(external / ".git") + "\n")

    monkeypatch.setattr(subprocess, "run", git)
    assert benchmark.default_data() == external / "data"
    assert len(calls) == 1 and calls[0][1] == root
    current = root / "data/cn/panel/current.json"
    current.parent.mkdir(parents=True)
    current.write_text("{}")
    assert benchmark.default_data() == root / "data"
    assert len(calls) == 1
    current.unlink()
    # 有明确指针时绝不因外置盘不可用而回退主检出。
    pointer(root / "data", "cn", external / "missing-cn")
    with pytest.raises(paths.DataLocationError, match="cn"):
        benchmark.default_data()
    assert len(calls) == 1
    monkeypatch.setenv("ASHARE_DATA_DIR", str(root / "configured"))
    assert benchmark.default_data() == root / "configured"
    monkeypatch.setenv("ASHARE_BENCH_DATA", str(root / "bench-configured"))
    assert benchmark.default_data() == root / "bench-configured"


def test_reference_location_error_not_cached(roots, monkeypatch):
    import polars as pl

    import ashare.data as ad
    from ashare.platform.paper_api import Reference
    from bench.platform.synthetic import write_panel

    root, external = roots
    target = external / "renamed-cn"
    pointer(root, "cn", target)
    reference = Reference(root)
    for _ in range(2):
        with pytest.raises(paths.DataLocationError, match="cn"):
            reference.lookup("S0000")
        assert reference.loaded_at == float("-inf")
        assert reference.data is None and reference.names == {}
    target.mkdir()
    write_panel(root)
    data = ad.open(root)
    # 最小行业表；证券名称仍从重定向后的真实合成 Parquet 读取。
    monkeypatch.setattr(
        data,
        "table",
        lambda *a, **kw: pl.DataFrame(
            {
                "symbol": ["S0000"],
                "industry_name": ["合成行业"],
                "effective_through": [None],
            }
        ).lazy(),
    )
    monkeypatch.setattr(ad, "open", lambda *a, **kw: data)
    assert reference.lookup("S0000") == ("合成证券1", "合成行业")


def test_meta_location_error_not_cached(roots):
    from fastapi.testclient import TestClient

    from ashare.platform.app import create_app
    from bench.platform.synthetic import write_panel

    root, external = roots
    target = external / "renamed-cn"
    pointer(root, "cn", target)
    app = create_app(root / "platform", data_root=root)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        for _ in range(2):
            response = client.get("/api/meta")
            assert response.status_code == 503
            assert response.json()["message"] == "数据位置不可用"
        target.mkdir()
        source = write_panel(root)
        response = client.get("/api/meta")
        assert response.status_code == 200
        day = str(source.dates[-1])
        assert response.json()["data_date"] == f"{day[:4]}-{day[4:6]}-{day[6:]}"


def test_reference_benchmark_location_error_not_cached():
    from ashare.platform.paper_api import Reference

    def unavailable(*args, **kwargs):
        raise paths.DataLocationError("数据集 cn 的目标目录不可用")

    reference = Reference()
    # 模拟证券名称已缓存，但后续基准读取才遇到位置错误。
    import time

    reference.loaded_at = time.monotonic()
    reference.names = {"TEST": "旧缓存"}
    reference.data = SimpleNamespace(table=unavailable)
    with pytest.raises(paths.DataLocationError, match="cn"):
        reference.benchmark("20240101", "20240102")
    assert reference.loaded_at == float("-inf")
    assert reference.data is None and reference.names == {}


def test_crypto_bench_cwd_default(roots, monkeypatch):
    import sys

    from bench import crypto_examples

    root, _ = roots
    monkeypatch.chdir(root)
    monkeypatch.delenv("ASHARE_DATA_DIR")
    monkeypatch.setattr(paths, "DEFAULT_DATA_ROOT", root / "checkout-data")
    monkeypatch.setattr(sys, "argv", ["crypto_examples"])
    seen = []

    def opened(path, *args, **kwargs):
        seen.append(path)
        raise StopIteration("已到读取入口")

    monkeypatch.setattr(crypto_examples.crypto, "open", opened)
    with pytest.raises(StopIteration):
        crypto_examples.main()
    assert seen == [Path("data") / "crypto"]


def test_ml_bench_preserves_fallback_root(roots, monkeypatch):
    import importlib.util

    root, external = roots
    fallback = root / "main-checkout"
    target = external / "renamed-dev"
    target.mkdir()
    pointer(fallback, "dev", target)
    file = Path(__file__).resolve().parents[2] / "bench/ml/mlbench.py"

    def selected(path=None, *, checkout=None):
        assert checkout == file.parents[2]
        return fallback

    monkeypatch.setattr(paths, "bench_data_root", selected)
    spec = importlib.util.spec_from_file_location("review_mlbench", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.DATA == fallback
    assert module.ML_ROOT == target / "ml"


@pytest.mark.parametrize(
    "source",
    [
        'x = Path(os.environ.get("ASHARE_DATA_DIR", "data")) / "platform"',
        'x = os.environ.get("ASHARE_DATA_DIR", default="data")',
        'x = os.path.join("data", "crypto")',
    ],
)
def test_static_environment_and_join_defaults(source):
    assert violations(source) == [1]


def test_crypto_minute_registered_cold(roots, monkeypatch):
    root, external = roots
    assert "crypto_minute" in paths.DATASETS
    assert paths.DATASETS["crypto_minute"] == paths.Dataset(False)
    monkeypatch.setenv("CRYPTO_DATA_ROOT", str(external / "unrelated-crypto"))
    source = paths.dataset_dir("crypto_minute", create=True)
    assert source == root / "crypto_minute"
    (source / "minute").write_text("synthetic")
    target = external / "renamed-minute"
    paths.move("crypto_minute", target)
    assert paths.dataset_dir("crypto_minute") == target
    assert (target / "minute").read_text() == "synthetic"
    row = next(row for row in paths.status() if row["name"] == "crypto_minute")
    assert row["readable"] and row["redirected"] and row["path"] == str(target)


@pytest.mark.parametrize(
    "var, call",
    [
        ("ASHARE_DATA_DIR", lambda: paths.dataset_dir("cn")),
        ("ASHARE_DATA_DIR", lambda: paths.data_root()),
        ("ASHARE_PLATFORM_DIR", lambda: paths.dataset_dir("platform")),
        ("ASHARE_PAPER_DIR", lambda: paths.dataset_dir("paper")),
        ("ASHARE_MINING_DIR", lambda: paths.dataset_dir("mining")),
        ("ASHARE_ML_DIR", lambda: paths.dataset_dir("ml")),
        ("CRYPTO_DATA_ROOT", lambda: paths.dataset_dir("crypto")),
        ("ASHARE_BENCH_DATA", lambda: paths.bench_data_root()),
    ],
)
def test_review2_empty_location_variable_is_an_error(monkeypatch, var, call):
    """来源：D1 审查第二轮 P1。设了但为空的位置变量不能静默换目录，统一报错。"""
    monkeypatch.setenv(var, "")
    with pytest.raises(paths.DataLocationError, match=var):
        call()
