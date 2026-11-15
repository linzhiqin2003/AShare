"""D1 离线验收：所有数据均在临时目录，外置盘也是普通临时目录。"""

import json
import os
from pathlib import Path

import pytest

import ashare.paths as paths

from .test_entries import pointer


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    for spec in paths.DATASETS.values():
        if spec.env:
            monkeypatch.delenv(spec.env, raising=False)
    monkeypatch.delenv("ASHARE_DATA_DIR", raising=False)
    root = tmp_path / "internal"
    root.mkdir()
    monkeypatch.setattr(paths, "DEFAULT_DATA_ROOT", root)
    return root


def test_default_and_create(isolated):
    folder = paths.dataset_dir("dev")
    assert folder == isolated / "dev"
    assert not folder.exists()
    assert paths.dataset_dir("dev", create=True).is_dir()
    with pytest.raises(paths.DataLocationError, match="未登记"):
        paths.dataset_dir("dev/futures", create=True)
    assert not (folder / "futures").exists()


def test_priority(isolated, tmp_path, monkeypatch):
    outside = tmp_path / "external"
    outside.mkdir()
    pointer(isolated, "ml", outside)
    assert paths.dataset_dir("ml") == outside
    replacement = tmp_path / "replacement"
    monkeypatch.setenv("ASHARE_DATA_DIR", str(replacement))
    assert paths.dataset_dir("ml") == replacement / "ml"
    other = tmp_path / "other"
    other.mkdir()
    pointer(replacement, "ml", other)
    assert paths.dataset_dir("ml") == other
    legacy = tmp_path / "legacy"
    monkeypatch.setenv("ASHARE_ML_DIR", str(legacy))
    assert paths.dataset_dir("ml") == legacy
    explicit = tmp_path / "explicit"
    assert paths.dataset_dir("ml", explicit) == explicit
    assert paths.dataset_dir("ml", root=isolated) == outside


@pytest.mark.parametrize("name", ["ml", "mining", "platform", "paper", "crypto"])
def test_each_legacy_override(isolated, tmp_path, monkeypatch, name):
    pointer(isolated, name, tmp_path / "missing")
    override = tmp_path / "override"
    monkeypatch.setenv(paths.DATASETS[name].env, str(override))
    assert paths.dataset_dir(name) == override
    assert paths.dataset_dir(name, tmp_path / "explicit") == tmp_path / "explicit"


@pytest.mark.parametrize("create", [False, True])
def test_missing_target(isolated, tmp_path, create):
    target = tmp_path / "unmounted" / "dev"
    pointer(isolated, "dev", target)
    with pytest.raises(paths.DataLocationError, match="dev") as exc:
        paths.dataset_dir("dev", create=create)
    assert str(target) in str(exc.value)
    assert not target.parent.exists()
    assert sorted(p.name for p in (isolated / "dev").iterdir()) == [paths.POINTER]


@pytest.mark.parametrize(
    "bad",
    [
        "{",
        "[]",
        "{}",
        '{"path": "relative"}',
        '{"path":"/unused","moved_at":"bad","files":0,"bytes":0}',
        '{"path":"/unused","moved_at":"2026-11-15","files":true,"bytes":0}',
    ],
)
def test_invalid_pointer(isolated, bad):
    folder = isolated / "dev"
    folder.mkdir()
    (folder / paths.POINTER).write_text(bad)
    with pytest.raises(paths.DataLocationError, match="dev.*指针无效"):
        paths.dataset_dir("dev", create=True)


def test_target_file_is_not_directory(isolated, tmp_path):
    target = tmp_path / "file"
    target.write_text("x")
    pointer(isolated, "crypto", target)
    with pytest.raises(paths.DataLocationError):
        paths.dataset_dir("crypto", create=True)
    assert target.read_text() == "x"


@pytest.fixture
def moving(isolated, tmp_path):
    source = isolated / "dev"
    (source / "nested" / "empty").mkdir(parents=True)
    (source / "a").write_bytes(b"abc")
    (source / "nested" / "b").write_bytes(b"12345")
    os.utime(source / "a", ns=(1_000_000_000, 2_000_000_000))
    return source, tmp_path / "disk-copy"


@pytest.mark.parametrize("delete_source", [False, True])
@pytest.mark.parametrize("existing_empty", [False, True])
def test_move_success(moving, delete_source, existing_empty):
    source, target = moving
    if existing_empty:
        target.mkdir()
    result = paths.move("dev", target, delete_source=delete_source)
    assert result["files"] == 2
    assert result["bytes"] == 8
    assert result["path"] == str(target)
    assert json.loads((source / paths.POINTER).read_text()) == result
    assert paths.dataset_dir("dev") == target
    assert (target / "a").read_bytes() == b"abc"
    assert (target / "nested" / "b").read_bytes() == b"12345"
    assert (target / "a").stat().st_mtime_ns == 2_000_000_000
    assert (target / "nested" / "empty").is_dir()
    if delete_source:
        assert sorted(p.name for p in source.iterdir()) == [paths.POINTER]
    else:
        assert (source / "a").read_bytes() == b"abc"


@pytest.mark.parametrize(
    "damage", ["content", "size", "copy_failure", "source_change", "publish_failure"]
)
def test_move_failure_is_atomic(moving, monkeypatch, damage):
    source, target = moving
    original = paths.shutil.copy2

    def copy(src, dst, **kwargs):
        original(src, dst, **kwargs)
        if Path(src).name == "a":
            if damage == "content":
                Path(dst).write_bytes(b"xyz")  # 同大小，必须靠 SHA 抓住。
            elif damage == "size":
                Path(dst).write_bytes(b"x")
            elif damage == "source_change":
                Path(src).write_bytes(b"new")
            elif damage == "copy_failure":
                raise OSError("injected copy failure")
        return dst

    monkeypatch.setattr(paths.shutil, "copy2", copy)
    if damage == "publish_failure":

        def fail(*args):
            raise OSError("injected replace failure")

        monkeypatch.setattr(paths.os, "replace", fail)
    with pytest.raises((paths.DataLocationError, OSError)):
        paths.move("dev", target, delete_source=True)
    assert not (source / paths.POINTER).exists()
    assert not target.exists()
    assert (source / "nested" / "b").read_bytes() == b"12345"
    assert not list(source.glob(".LOCATION-*"))
    assert not list(source.parent.glob(".move-*"))


def test_pointer_only_after_verification(moving, monkeypatch):
    source, target = moving
    verify = paths._verify_copy
    seen = []

    def check(*args):
        assert not (source / paths.POINTER).exists()
        seen.append(True)
        verify(*args)

    monkeypatch.setattr(paths, "_verify_copy", check)
    paths.move("dev", target)
    assert seen == [True]


def test_nonempty_target_untouched(moving):
    source, target = moving
    target.mkdir()
    (target / "keep").write_text("keep")
    with pytest.raises(paths.DataLocationError, match="非空"):
        paths.move("dev", target)
    assert (target / "keep").read_text() == "keep"
    assert not (source / paths.POINTER).exists()


@pytest.mark.parametrize(
    "name", [name for name, spec in paths.DATASETS.items() if spec.hot]
)
def test_hot_datasets_refused(isolated, tmp_path, name):
    with pytest.raises(paths.DataLocationError, match="热数据"):
        paths.move(name, tmp_path / "copy")
    assert not (tmp_path / "copy").exists()


def test_force_hot(isolated, tmp_path):
    (isolated / "cn").mkdir()
    paths.move("cn", tmp_path / "copy", force=True)
    assert paths.dataset_dir("cn") == tmp_path / "copy"


def test_move_overlap_symlinks_and_second_move(moving, tmp_path):
    source, target = moving
    for bad in (source, source / "child", source.parent, source.parent / "crypto"):
        with pytest.raises(paths.DataLocationError):
            paths.move("dev", bad)
    (source / "link").symlink_to(source / "a")
    with pytest.raises(paths.DataLocationError, match="符号链接"):
        paths.move("dev", target)
    assert not target.exists()
    (source / "link").unlink()
    paths.move("dev", target)
    with pytest.raises(paths.DataLocationError, match="已有指针"):
        paths.move("dev", tmp_path / "second")


def test_status_missing_and_redirected(isolated, tmp_path, capsys):
    target = tmp_path / "target"
    target.mkdir()
    (target / "a").write_bytes(b"123")
    pointer(isolated, "dev", target)
    pointer(isolated, "crypto", tmp_path / "unmounted")
    rows = {row["name"]: row for row in paths.status()}
    assert rows["dev"]["bytes"] == 3
    assert rows["dev"]["readable"] and rows["dev"]["redirected"]
    assert not rows["crypto"]["readable"] and rows["crypto"]["redirected"]
    assert str(tmp_path / "unmounted") in rows["crypto"]["error"]
    assert paths.main(["status"]) == 1
    output = capsys.readouterr().out
    assert "重定向" in output and "dev" in output


def test_cli_move(moving):
    _, target = moving
    assert paths.main(["move", "dev", str(target), "--delete-source"]) == 0
    assert paths.dataset_dir("dev") == target


def test_explicit_bypasses_invalid_default(isolated, tmp_path):
    pointer(isolated, "crypto", tmp_path / "missing")
    assert paths.dataset_dir("crypto", tmp_path / "chosen") == tmp_path / "chosen"


def test_status_bad_pointer_is_failed_redirect(isolated, capsys):
    """D1 自查：坏指针不能被 status 误报成普通缺目录并成功退出。"""
    folder = isolated / "dev"
    folder.mkdir()
    (folder / paths.POINTER).write_text("{")
    row = next(r for r in paths.status() if r["name"] == "dev")
    assert row["redirected"] and not row["readable"]
    assert str(folder) in row["path"]
    assert paths.main(["status"]) == 1


def test_move_rechecks_pointer_after_lock(moving, monkeypatch):
    """D1 自查：另一进程可能在第一次检查和取得锁之间发布指针。"""
    source, target = moving
    original = paths.os.open

    def acquire(file, *args, **kwargs):
        if Path(file).name == ".move-dev.lock":
            pointer(source.parent, "dev", source.parent.parent / "other")
        return original(file, *args, **kwargs)

    monkeypatch.setattr(paths.os, "open", acquire)
    with pytest.raises(paths.DataLocationError, match="已有指针"):
        paths.move("dev", target)
    assert not target.exists()
    assert json.loads((source / paths.POINTER).read_text())["path"].endswith("other")
