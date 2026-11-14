"""调度模板（2026-11-14 新增，模拟盘每日运行）：路径替换、引号、时区、默认只打印、从不加载任务；
公开仓库里的模板和文档不含本机绝对路径。
"""

import io
import plistlib
import re
import subprocess

import pytest

from ashare.paper.ops import install

REPO = install.REPO


def render(kind, tmp_path, apply=False, **extra):
    out = io.StringIO()
    files = install.install(
        kind,
        apply=apply,
        target=tmp_path / "agents",
        python="/opt/py env/bin/python",
        repo=tmp_path / "repo",
        data_dir=tmp_path / "data dir",
        paper_dir=tmp_path / "paper",
        env_file=tmp_path / "paper.env",
        out=out,
        **extra,
    )
    return files, out.getvalue()


@pytest.fixture(autouse=True)
def no_processes(monkeypatch):
    """安装命令绝不调用 launchctl / systemctl。"""

    def forbidden(*args, **kwargs):
        raise AssertionError("install 不应启动任何进程")

    for name in ("run", "Popen", "call", "check_call", "check_output"):
        monkeypatch.setattr(subprocess, name, forbidden)


def test_launchd_dry_run_renders_valid_plists(tmp_path):
    files, text = render("launchd", tmp_path)
    assert not (tmp_path / "agents").exists()  # 默认只打印
    assert "--dry-run" in text and "launchctl bootstrap" in text
    # 拷贝进 ~/Library/LaunchAgents 和加载都由人执行（2026-11-14 审查）。
    agents = install.launch_agents()
    lines = text.splitlines()
    assert f"cp {tmp_path / 'agents' / 'com.ashare.paper.run.plist'} {agents}/" in lines
    assert any(
        line.startswith("launchctl bootstrap")
        and line.endswith(str(agents / "com.ashare.paper.run.plist"))
        for line in lines
    )
    by_name = {p.name: plistlib.loads(content.encode()) for p, content in files}
    assert set(by_name) == {
        "com.ashare.paper.nightly.plist",
        "com.ashare.paper.run.plist",
    }
    run = by_name["com.ashare.paper.run.plist"]
    assert run["Label"] == "com.ashare.paper.run"
    command = run["ProgramArguments"][2]
    # 带空格的解释器路径被 shell 引用；防睡眠；跑到 17:00。
    assert command.startswith("exec /usr/bin/caffeinate -i '/opt/py env/bin/python'")
    assert command.endswith("-m ashare.paper daily run --until 17:00")
    assert run["EnvironmentVariables"] == {
        "ASHARE_DATA_DIR": str(tmp_path / "data dir"),
        "ASHARE_PAPER_DIR": str(tmp_path / "paper"),
    }
    assert run["StandardErrorPath"].startswith(
        str(tmp_path / "paper" / ".ops" / "logs")
    )
    assert [
        (d["Weekday"], d["Hour"], d["Minute"]) for d in run["StartCalendarInterval"]
    ] == [(w, 8, 55) for w in range(1, 6)]
    nightly = by_name["com.ashare.paper.nightly.plist"]
    assert nightly["ProgramArguments"][2].endswith("daily nightly")
    assert [(d["Hour"], d["Minute"]) for d in nightly["StartCalendarInterval"]] == [
        (7, 30),
        (18, 30),
        (20, 30),
    ]


def test_systemd_units_use_beijing_time_and_persistent_timers(tmp_path):
    with pytest.raises(ValueError):
        render("systemd", tmp_path)  # 解释器路径有空格：systemd 单元不替人加引号
    out = io.StringIO()
    files = install.install(
        "systemd",
        target=tmp_path / "units",
        python="/opt/py/bin/python",
        repo=tmp_path / "repo",
        data_dir=tmp_path / "data",
        paper_dir=tmp_path / "paper",
        env_file=tmp_path / "paper.env",
        out=out,
    )
    units = {p.name: content for p, content in files}
    assert set(units) == set(install.SYSTEMD)
    for name, content in units.items():
        assert "{{" not in content
        if name.endswith(".timer"):
            calendars = re.findall(r"^OnCalendar=(.+)$", content, re.M)
            assert calendars and all(c.endswith(" Asia/Shanghai") for c in calendars)
            assert "Persistent=true" in content
        else:
            assert "Environment=TZ=Asia/Shanghai" in content
            assert f"EnvironmentFile={tmp_path / 'paper.env'}" in content
            assert "ExecStart=/opt/py/bin/python -m ashare.paper daily " in content
    assert (
        "OnCalendar=Mon..Fri *-*-* 08:55:00 Asia/Shanghai"
        in units["ashare-paper-run.timer"]
    )
    assert "SuccessExitStatus=3 4" in units["ashare-paper-nightly.service"]
    # 开机补跑最早从零点跑到 17:00（2026-11-14 审查）：超时要覆盖 17 小时。
    timeout = re.search(
        r"^TimeoutStartSec=(\d+)h$", units["ashare-paper-run.service"], re.M
    )
    assert timeout and int(timeout.group(1)) >= 17
    text = out.getvalue()
    assert "systemctl --user enable --now ashare-paper-run.timer" in text
    assert not (tmp_path / "units").exists()


def test_apply_writes_files_only(tmp_path):
    files, text = render("launchd", tmp_path, apply=True)
    assert sorted(p.name for p in (tmp_path / "agents").iterdir()) == sorted(
        p.name for p, _ in files
    )
    assert "没有加载任何任务" in text
    assert (tmp_path / "paper" / ".ops" / "logs").is_dir()


def test_render_rejects_leftovers_and_newlines():
    with pytest.raises(ValueError):
        install.render("{{UNKNOWN}}", {"PYTHON": "p"}, kind="systemd")
    with pytest.raises(ValueError):
        install.render("{{PYTHON}}", {"PYTHON": "a\nb"}, kind="launchd")
    escaped = install.render(
        "<string>{{REPO}}</string>", {"REPO": "a&b<c"}, kind="launchd"
    )
    assert escaped == "<string>a&amp;b&lt;c</string>"


def test_public_files_have_no_local_absolute_paths():
    """公开仓库：模板与文档只用占位符，不写本机路径、用户名或密钥。"""
    paths = [
        *sorted((REPO / "deploy" / "paper").rglob("*.in")),
        REPO / "docs" / "PAPER_OPS.md",
    ]
    assert len(paths) == 7
    for path in paths:
        text = path.read_text()
        assert not re.search(r"/Users/|/home/[a-z]", text), path
        assert "TUSHARE_TOKEN=" not in text.replace("TUSHARE_TOKEN=…", ""), path


def test_launchd_default_target_is_staging_not_launch_agents(tmp_path, monkeypatch):
    """2026-11-14 审查：~/Library/LaunchAgents 里的 plist 下次登录会被自动加载，--apply 默认不写那里。"""
    monkeypatch.setattr(install.Path, "home", classmethod(lambda cls: tmp_path))
    target = install.default_target("launchd")
    assert target != install.launch_agents()
    assert install.launch_agents() not in target.parents
    out = io.StringIO()
    install.install(
        "launchd",
        apply=True,
        python="/opt/py/bin/python",
        repo=tmp_path / "repo",
        data_dir=tmp_path / "data",
        paper_dir=tmp_path / "paper",
        out=out,
    )
    assert sorted(p.name for p in target.iterdir()) == sorted(install.LAUNCHD)
    assert not install.launch_agents().exists()
    # 显式指定 LaunchAgents 也行，但要提醒会自动加载。
    explicit = io.StringIO()
    install.install(
        "launchd",
        apply=True,
        target=install.launch_agents(),
        python="/opt/py/bin/python",
        repo=tmp_path / "repo",
        data_dir=tmp_path / "data",
        paper_dir=tmp_path / "paper",
        out=explicit,
    )
    assert "下次登录会自动加载" in explicit.getvalue()
    assert not any(line.startswith("cp ") for line in explicit.getvalue().splitlines())
