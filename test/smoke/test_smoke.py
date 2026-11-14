"""冒烟测试（`make smoke`，1 分钟内）：导入全部模块、每个命令行入口 `--help`、
平台在合成数据上启动后打一遍主要接口再干净关掉。

只回答「装好了、能起来、主路径通」；数值正确性归集成和端到端测试。
"""

from __future__ import annotations

import importlib
import pkgutil
import re
import subprocess
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PACKAGES = ("ashare", "panelbt", "panelops", "paperbroker", "portopt")
# 可选依赖（pyproject 的 extras / 依赖组，或文档里写明要手动装的）：缺了就跳过依赖它的模块。
OPTIONAL = {
    "databento",  # extra databento
    "blpapi",  # Bloomberg，文档写明手动装
    "lightgbm",  # 依赖组 ml
    "sklearn",  # 依赖组 ml / risk-reference
    "fastapi",  # extra platform
    "uvicorn",  # extra platform
    "starlette",
    "baostock",  # 依赖组 data-verify
    "matplotlib",
    "pandas",
}
CLI_WORKERS = 4


def modules():
    names = []
    for package in PACKAGES:
        module = importlib.import_module(package)
        names.append(package)
        for info in pkgutil.walk_packages(module.__path__, prefix=f"{package}."):
            names.append(info.name)
    return sorted(set(names))


def test_import_every_module():
    """遍历五个包的全部子模块逐个导入；缺可选依赖的跳过并列出，其余任何失败都算错。"""
    failures, skipped = {}, {}
    names = modules()
    assert len(names) > 100
    for name in names:
        try:
            importlib.import_module(name)
        except ModuleNotFoundError as error:
            root = (error.name or "").split(".")[0]
            if root in OPTIONAL and root not in PACKAGES:
                skipped[name] = root
            else:
                failures[name] = repr(error)
        except Exception as error:  # 导入期间的任何异常都要报出来
            failures[name] = f"{type(error).__name__}: {error}"
    assert not failures, failures
    if skipped:
        warnings.warn(f"缺可选依赖，跳过导入：{skipped}", stacklevel=1)
    # 导入的是本工作区的代码，不是别处装的旧版本。
    for package in PACKAGES:
        assert (
            Path(importlib.import_module(package).__file__)
            .resolve()
            .is_relative_to(ROOT)
        ), package


def entry_points():
    """源码里所有 `if __name__ == "__main__"` 的模块都是命令行入口。"""
    found = []
    for base in [ROOT / "src", *(ROOT / "packages").glob("*/python")]:
        for path in base.rglob("*.py"):
            if re.search(r"^if __name__ == .__main__.:", path.read_text(), re.M):
                parts = path.relative_to(base).with_suffix("").parts
                if parts[-1] == "__main__":
                    parts = parts[:-1]
                found.append(".".join(parts))
    return sorted(found)


def run_help(args):
    completed = subprocess.run(
        [sys.executable, "-m", *args, "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return args, completed


def subcommands(usage):
    """argparse 子命令写成 {a,b,c}；取第一组。"""
    match = re.search(r"\{([a-z0-9_,-]+)\}", usage)
    return match.group(1).split(",") if match else []


def test_every_cli_entry_answers_help():
    entries = entry_points()
    assert {"ashare.data", "ashare.report", "ashare.paper", "ashare.platform"} <= set(
        entries
    )
    problems = {}
    with ThreadPoolExecutor(CLI_WORKERS) as pool:
        top = list(pool.map(run_help, [[e] for e in entries]))
        nested = []
        for (entry,), completed in top:
            if completed.returncode != 0 or "usage:" not in completed.stdout:
                problems[entry] = (completed.returncode, completed.stderr[-800:])
                continue
            # paper.smoke 把 smoke 子命令固定好了，--help 显示的就是子命令帮助。
            if entry != "ashare.paper.smoke":
                nested += [[entry, sub] for sub in subcommands(completed.stdout)]
        for args, completed in pool.map(run_help, nested):
            if completed.returncode != 0 or "usage:" not in completed.stdout:
                problems[" ".join(args)] = (
                    completed.returncode,
                    completed.stderr[-800:],
                )
    assert not problems, problems
    assert len(nested) >= 10, nested


def test_platform_starts_serves_and_stops(datadir, tmp_path):
    pytest.importorskip("fastapi", reason="平台是可选依赖：uv sync --extra platform")
    from test.fixtures.platform_server import PlatformServer

    server = PlatformServer(datadir.root, tmp_path / "platform")
    server.start()
    try:
        # 健康检查：数据截至日来自合成面板。
        meta = server.get("/api/meta").json()
        assert meta["data_date"] is not None and meta["running"] == 0
        assert server.get("/api/settings").json()["defaults"]["universe"] == "TOP1800"
        tab = server.post("/api/tabs").json()["id"]
        job = server.post(
            f"/api/tabs/{tab}/runs",
            {"name": "smoke", "source": "-ts_std(returns, 10)", "settings": {}},
        )
        assert job.status_code == 202, job.text
        result = server.wait_job(job.json()["id"], deadline=60)
        assert result["status"] == "complete", server.log_text()
        server.snapshot_children()
        run = result["runs"][0]
        assert run["summary"]["is"]["sharpe"] is not None
        holdings = server.get(f"/api/runs/{run['id']}/holdings").json()
        assert holdings["date"] and holdings["dates"]
        assert server.get(
            f"/api/runs/{run['id']}/holdings", params={"date": holdings["date"]}
        ).json()["items"]
        switched = server.get(
            f"/api/jobs/{job.json()['id']}/benchmark", params={"symbol": "000905.SH"}
        ).json()
        assert switched["symbol"] == "000905.SH" and switched["relatives"]
        library = server.get("/api/library").json()
        assert library["total"] == 1 and library["items"][0]["name"] == "smoke"
    finally:
        stopped = server.close()
    assert stopped["graceful"] and stopped["leftovers"] == {}, stopped
    assert server.seen_children
