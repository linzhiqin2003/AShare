"""D1 基准入口与调度模板回归；HTTP 使用内存替身，不访问服务。"""

import json
import plistlib
import sys

from .test_entries import pointer


def test_platform_verify_default_root(roots, monkeypatch, capsys):
    """D1 开发回归：PanelCache 的 root 参数必填，None 才表示默认入口。"""
    from bench.platform import verify_real
    from bench.platform.synthetic import write_panel

    root, external = roots
    target = external / "renamed-cn"
    target.mkdir()
    pointer(root, "cn", target)
    write_panel(root)

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, *args, **kwargs):
            return self

        get = delete = post

        def raise_for_status(self):
            pass

        def json(self):
            return dict(id="offline", status="complete", runs=[], elapsed=0)

    monkeypatch.setattr(verify_real.httpx, "Client", Client)
    monkeypatch.setattr(sys, "argv", ["verify_real"])
    verify_real.main()
    assert json.loads(capsys.readouterr().out)["job"] == "offline"


def test_service_paths_follow_redirect_and_escape_xml(roots):
    from bench.platform.service import render

    root, external = roots
    pointer(root, "platform", external)
    value = plistlib.loads(
        render(port=8765, subnet="10.77.0.0/24", python="/fake/a & b/python").encode()
    )
    from ashare.platform.config import platform_dir

    assert platform_dir(service=True) == external / "results"
    assert value["StandardOutPath"] == str(external / "logs/platform.out.log")
    assert "'/fake/a & b/python'" in value["ProgramArguments"][2]


def test_service_refuses_missing_target(roots):
    import pytest

    from ashare.paths import DataLocationError
    from bench.platform.service import render

    root, external = roots
    target = external / "missing"
    pointer(root, "platform", target)
    with pytest.raises(DataLocationError):
        render(port=8765, subnet="10.77.0.0/24")
    assert not target.exists()


def test_ml_bench_subdirectory_and_override(roots, monkeypatch):
    import pytest

    from ashare.paths import DataLocationError, bench_ml_dir

    root, external = roots
    pointer(root, "dev", external)
    assert bench_ml_dir() == external / "ml"
    pointer(root, "dev", external / "missing")
    with pytest.raises(DataLocationError):
        bench_ml_dir()
    monkeypatch.setenv("ASHARE_ML_DIR", str(external / "chosen"))
    assert bench_ml_dir() == external / "chosen"


def test_synthetic_panel_writer_respects_market_pointer(roots):
    """D1 扫描回归：合成面板曾自行拼 cn/panel，绕过同根的指针。"""
    import ashare.data as ad
    from bench.platform.synthetic import write_panel

    root, external = roots
    target = external / "renamed-cn"
    target.mkdir()
    pointer(root, "cn", target)
    source = write_panel(root)
    assert ad.open().panel("close")["close"].shape == source.shape
    assert [p.name for p in (root / "cn").iterdir()] == ["LOCATION.json"]


def test_service_does_not_freeze_pointer_as_legacy_override(roots):
    """D1 自查：把指针目标写入专用变量会绕过下次启动时的断盘检查。"""
    from bench.platform.service import render

    root, external = roots
    pointer(root, "platform", external)
    value = plistlib.loads(render(port=8765, subnet="10.77.0.0/24").encode())
    assert "ASHARE_PLATFORM_DIR" not in value["EnvironmentVariables"]
    assert "--service-results" in value["ProgramArguments"][2]


def test_service_legacy_override_is_not_given_extra_subdirectory(roots, monkeypatch):
    from ashare.platform.config import platform_dir
    from bench.platform.service import render

    _, external = roots
    monkeypatch.setenv("ASHARE_PLATFORM_DIR", str(external))
    assert platform_dir(service=True) == external
    value = plistlib.loads(render(port=8765, subnet="10.77.0.0/24").encode())
    assert value["EnvironmentVariables"]["ASHARE_PLATFORM_DIR"] == str(external)
