"""把 ``deploy/paper/`` 里的调度模板换成本机实际路径。

默认只打印（dry run）：每个文件的目标路径、内容，以及装上后要执行的命令。``--apply`` 只把文件
写到目标目录，**从不加载或启动**任务（``launchctl`` / ``systemctl`` 留给人执行）。

launchd 的文件默认写到暂存目录 ``~/.local/share/ashare/paper-schedule/launchd``，不直接写
``~/Library/LaunchAgents``：那里的 plist 在下次登录时会被系统自动加载，写进去就等于装上了
（2026-11-14 审查）。拷贝进去由人执行，命令一并打印。systemd 的单元写进
``~/.config/systemd/user`` 不会生效，要 ``enable`` 定时器才会。
"""

from __future__ import annotations

import os
import shlex
import sys
from datetime import datetime
from pathlib import Path
from xml.sax.saxutils import escape

from ..market import TZ

REPO = Path(__file__).resolve().parents[4]
TEMPLATES = REPO / "deploy" / "paper"
LAUNCHD = ("com.ashare.paper.nightly.plist", "com.ashare.paper.run.plist")
SYSTEMD = (
    "ashare-paper-nightly.service",
    "ashare-paper-nightly.timer",
    "ashare-paper-run.service",
    "ashare-paper-run.timer",
)


def values(*, python, repo, data_dir, paper_dir, env_file) -> dict[str, str]:
    paper = Path(paper_dir)
    return dict(
        PYTHON=str(python),
        REPO=str(repo),
        DATA_DIR=str(data_dir),
        PAPER_DIR=str(paper),
        LOG_DIR=str(paper / ".ops" / "logs"),
        ENV_FILE=str(env_file),
    )


def render(text: str, mapping: dict[str, str], *, kind: str) -> str:
    """launchd：值经 XML 转义，放进 ``zsh -c`` 的路径再按 shell 引用；systemd：值不能含换行。"""
    for key, value in mapping.items():
        if "\n" in value or "\r" in value:
            raise ValueError(f"{key} 不能含换行")
        if kind == "launchd":
            text = text.replace(
                f"exec /usr/bin/caffeinate -i {{{{{key}}}}}",
                f"exec /usr/bin/caffeinate -i {escape(shlex.quote(value))}",
            )
            text = text.replace(f"{{{{{key}}}}}", escape(value))
        else:
            if any(c.isspace() for c in value) and key in ("PYTHON", "REPO"):
                raise ValueError(f"{key} 含空白字符，systemd 单元里需要手工加引号")
            text = text.replace(f"{{{{{key}}}}}", value)
    if "{{" in text or "}}" in text:
        raise ValueError("模板里还有没替换的占位符")
    return text


def plan(kind: str, mapping: dict[str, str], target: Path) -> list[tuple[Path, str]]:
    names = LAUNCHD if kind == "launchd" else SYSTEMD
    folder = TEMPLATES / kind
    return [
        (
            target / name,
            render((folder / f"{name}.in").read_text(), mapping, kind=kind),
        )
        for name in names
    ]


def launch_agents() -> Path:
    """launchd 登录时自动加载的用户任务目录。"""
    return Path.home() / "Library" / "LaunchAgents"


def default_target(kind: str) -> Path:
    home = Path.home()
    if kind == "launchd":
        return home / ".local" / "share" / "ashare" / "paper-schedule" / "launchd"
    return home / ".config" / "systemd" / "user"


def commands(kind: str, files: list[Path]) -> list[str]:
    if kind == "launchd":
        uid = os.getuid()
        agents = launch_agents()
        installed = [agents / p.name for p in files]
        copies = [
            f"cp {shlex.quote(str(p))} {shlex.quote(str(agents))}/"
            for p in files
            if p.parent != agents
        ]
        return (
            copies
            + [
                f"launchctl bootstrap gui/{uid} {shlex.quote(str(p))}"
                for p in installed
            ]
            + [
                f"launchctl print gui/{uid}/{p.name.removesuffix('.plist')}"
                for p in installed
            ]
        )
    timers = [p.name for p in files if p.name.endswith(".timer")]
    return [
        "systemctl --user daemon-reload",
        *(f"systemctl --user enable --now {t}" for t in timers),
        "systemctl --user list-timers 'ashare-paper-*'",
        'loginctl enable-linger "$USER"  # 用户不登录时定时器也照跑',
    ]


def timezone_warning(kind: str) -> str | None:
    """launchd 按本机时区触发；本机不是北京时间时，模板里的钟点要换算。"""
    if kind != "launchd":
        return None
    now = datetime.now()
    local = now.astimezone().utcoffset()
    beijing = now.astimezone(TZ).utcoffset()
    if local != beijing:
        return (
            f"本机时区偏移 {local} 与北京时间 {beijing} 不同：launchd 按本机时间触发，"
            "请把模板里的钟点换算后再装。"
        )
    return None


def install(
    kind: str,
    *,
    apply: bool = False,
    target: Path | None = None,
    python: str | None = None,
    repo: Path | None = None,
    data_dir: Path,
    paper_dir: Path,
    env_file: Path | None = None,
    out=None,
) -> list[tuple[Path, str]]:
    out = out or sys.stdout
    if kind not in ("launchd", "systemd"):
        raise ValueError("只支持 launchd 或 systemd")
    target = target or default_target(kind)
    mapping = values(
        python=python or sys.executable,
        repo=repo or REPO,
        data_dir=data_dir,
        paper_dir=paper_dir,
        env_file=env_file or Path.home() / ".config" / "ashare" / "paper.env",
    )
    files = plan(kind, mapping, target)
    warning = timezone_warning(kind)
    if warning:
        print(f"# 注意：{warning}", file=out)
    for path, text in files:
        print(f"# ===== {path} =====", file=out)
        print(text, file=out)
    if apply:
        if kind == "launchd" and target.resolve() == launch_agents().resolve():
            print(
                "# 注意：写进 ~/Library/LaunchAgents 的任务下次登录会自动加载。",
                file=out,
            )
        target.mkdir(parents=True, exist_ok=True)
        (Path(mapping["LOG_DIR"])).mkdir(parents=True, exist_ok=True)
        for path, text in files:
            path.write_text(text)
        print(f"# 已写入 {len(files)} 个文件；没有加载任何任务。", file=out)
    else:
        print(
            "# 以上为预览（--dry-run，默认）；加 --apply 才写文件，且不会加载任务。",
            file=out,
        )
    print("# 审查后由人执行：", file=out)
    for line in commands(kind, [p for p, _ in files]):
        print(line, file=out)
    return files
