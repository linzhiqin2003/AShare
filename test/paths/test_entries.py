"""D1 回归：原入口忽略 LOCATION，或在默认参数解析时抢先访问未挂载目录。"""

import json

import pytest


def pointer(root, name, target):
    stub = root / name
    stub.mkdir(parents=True, exist_ok=True)
    (stub / "LOCATION.json").write_text(
        json.dumps(
            dict(
                path=str(target),
                moved_at="2026-11-15T00:00:00+00:00",
                files=1,
                bytes=7,
            )
        )
    )


def entry(name):
    if name == "cn":
        from ashare.data import open

        return open()._store.root
    if name == "crypto":
        from ashare.data.crypto import open

        return open().root
    if name == "ml":
        from ashare.ml import default_root

        return default_root()
    if name == "mining":
        from ashare.factor.mining.ledger import open_ledger

        return open_ledger().path.parent
    if name == "platform":
        from ashare.platform.config import platform_dir

        return platform_dir()
    from ashare.paper.ops.context import Context

    return Context.create().paper


@pytest.mark.parametrize("name", ["cn", "crypto", "ml", "mining", "platform", "paper"])
def test_entries_follow_pointer(roots, name):
    root, external = roots
    target = external / f"renamed-{name}"
    target.mkdir()
    (target / "marker").write_text("outside")
    if name == "crypto":
        version = "a" * 64
        (target / "latest.json").write_text(json.dumps(dict(version=version)))
        folder = target / "panels" / version
        folder.mkdir(parents=True)
        (folder / "meta.json").write_text("{}")
    pointer(root, name, target)
    assert (entry(name) / "marker").read_text() == "outside"


@pytest.mark.parametrize("name", ["cn", "crypto", "ml", "mining", "platform", "paper"])
def test_entries_missing_target_never_fall_back(roots, name):
    from ashare.paths import DataLocationError

    root, external = roots
    target = external / f"renamed-{name}"
    pointer(root, name, target)
    with pytest.raises(DataLocationError, match=name):
        entry(name)
    assert not target.exists()
    assert [p.name for p in (root / name).iterdir()] == ["LOCATION.json"]


def test_paper_read_does_not_resolve_unused_platform(roots, monkeypatch):
    """D1 全量回归：只读模拟账户不能先解析完全未使用的回测结果库。"""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from ashare.paths import DataLocationError
    from ashare.platform import config, paper_api

    def unavailable():
        raise DataLocationError("platform 未挂载")

    monkeypatch.setattr(config, "platform_dir", unavailable)
    app = FastAPI()
    app.include_router(paper_api.router(root=roots[0] / "paper"))
    with TestClient(app) as client:
        assert client.get("/api/paper/accounts").json() == {"items": []}


@pytest.mark.parametrize("kind", ["--launchd", "--systemd"])
def test_paper_schedule_keeps_pointer_dynamic(roots, capsys, kind):
    """D1 调度回归：默认账户位置不固化成专用变量，下次启动仍查指针。"""
    from ashare.paper.ops.cli import main

    root, external = roots
    pointer(root, "paper", external)
    assert main(["install-schedule", kind, "--target", str(root / "schedule")]) == 0
    text = capsys.readouterr().out
    assert "ASHARE_PAPER_DIR" not in text
    if kind == "--launchd":
        assert str(external / ".ops" / "logs") in text
    else:
        # systemd 模板没有 launchd 的 StandardOutPath；应用日志仍在解析后的 .ops。
        from ashare.paper.ops.context import Context

        assert Context.create().ops == external / ".ops"
        assert "-m ashare.paper daily nightly" in text
    assert not (root / "schedule").exists()
