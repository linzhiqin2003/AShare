"""命令行端到端：原始库 → `python -m ashare.data build-panel / verify / status` →
研究脚本（因子、分组回测、账户回测、报告）→ `python -m ashare.report show / html`。

每一步都在子进程里跑真命令，检查退出码、产物文件和格式，关键数字与本进程直接调用库函数的结果逐位对照。
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import ashare.data
from ashare.factor import Evaluator, groupbt
from ashare.platform.language import parse
from ashare.report import Tearsheet
from test.fixtures.synthetic_datadir import copy_datadir

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = Path(__file__).with_name("research_cli.py")
SOURCE = "-ts_std(returns, 10)"
# 合成库只生成面板要用的数据集；这两项覆盖检查本来就会失败，其余必须全过。
NOT_GENERATED = {"partition_coverage_fut_daily", "partition_coverage_margin_secs"}
COMPARED_FIELDS = ["close", "adj_factor", "can_buy_open", "industry", "member_csi300"]


def run(*args, timeout=240):
    env = dict(os.environ)
    env.pop("TUSHARE_TOKEN", None)
    completed = subprocess.run(
        [sys.executable, *args],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert completed.returncode == 0, (
        completed.stdout[-3000:] + completed.stderr[-3000:]
    )
    return completed


@pytest.fixture(scope="module")
def workspace(datadir, tmp_path_factory):
    """从原始库开始：复制只读共享库里的原始部分（不带面板），命令行重建。"""
    base = tmp_path_factory.mktemp("cli")
    copy = copy_datadir(datadir, base / "data", panel=False)
    assert not (copy.root / "cn/panel").exists()
    return dict(base=base, data=copy)


@pytest.fixture(scope="module")
def built(workspace):
    output = run(
        "-m", "ashare.data", "--root", str(workspace["data"].root), "build-panel"
    )
    return json.loads(output.stdout)


def test_build_panel_cli_matches_shared_build(built, workspace, datadir):
    assert built["version"] == datadir.version
    assert built["shape"] == [len(datadir.dates), len(datadir.symbols)]
    assert built["reused_rows"] == 0
    fresh = ashare.data.open(workspace["data"].root).panel(COMPARED_FIELDS)
    shared = ashare.data.open(datadir.root).panel(COMPARED_FIELDS)
    for name in COMPARED_FIELDS:
        assert np.asarray(fresh[name]).tobytes() == np.asarray(shared[name]).tobytes()


def test_verify_and_status_cli(built, workspace):
    root = str(workspace["data"].root)
    report = json.loads(run("-m", "ashare.data", "--root", root, "verify").stdout)
    path = Path(report.pop("report"))
    assert path.is_file() and path.is_relative_to(workspace["data"].root)
    failing = {name for name, check in report.items() if check["failed"]}
    assert failing == NOT_GENERATED
    # 边界确实被检查到：停牌、除权、公司行为对账、开盘掩码与 panelbt 一致。
    for name in (
        "suspension_no_trade",
        "ex_dividend_reference",
        "action_reconciliation",
        "open_masks_match_panelbt",
        "price_limits",
    ):
        assert report[name]["checked"] > 0, name
    status = json.loads(run("-m", "ashare.data", "--root", root, "status").stdout)
    assert status


@pytest.fixture(scope="module")
def research(built, workspace):
    out = workspace["base"] / "research"
    output = run(
        str(SCRIPT),
        "--root",
        str(workspace["data"].root),
        "--out",
        str(out),
        "--source",
        SOURCE,
    )
    return dict(out=out, summary=json.loads(output.stdout.strip().splitlines()[-1]))


def test_research_script_artifacts(research, datadir):
    out, summary = research["out"], research["summary"]
    for name in ("groupbt.html", "groupbt.json", "tearsheet.json", "report.html"):
        assert (out / name).stat().st_size > 0, name
    assert json.loads((out / "summary.json").read_text()) == summary
    assert summary["data_version"] == datadir.version
    html = (out / "groupbt.html").read_text(encoding="utf-8")
    assert html.count("<svg") == 2  # 主池 TOP1800 + 子池 TOP800
    sheet = Tearsheet.load(out / "tearsheet.json")
    assert sheet.summary["final_nav"] == pytest.approx(
        summary["final_equity"] / 1e7, abs=5e-7
    )
    assert sheet.summary["final_nav"] == summary["tearsheet_final_nav"]
    assert sheet.summary["trading_days"] > 100
    page = (out / "report.html").read_text(encoding="utf-8")
    assert page.lstrip().lower().startswith("<!doctype html")


def test_groupbt_numbers_equal_direct_library_call(research, datadir):
    """命令行脚本写出的分组多空收益，与本进程直接调库在共享库上算的逐位相同。"""
    record = json.loads((research["out"] / "groupbt.json").read_text())
    data = ashare.data.open(datadir.root)
    expression = parse(SOURCE)[0].expression
    panel = data.panel(sorted(set(groupbt.FIELDS) | set(expression.fields)))
    alpha = Evaluator().evaluate(expression, panel)
    result = groupbt.run(alpha, panel, "TOP1800", rebalance=5)
    assert record["dates"] == [str(d) for d in result.dates]
    expected = [
        None if not np.isfinite(x) else float(x) for x in result.pnl["long_short"]
    ]
    assert record["long_short_pnl"] == expected
    assert record["long_short"] == json.loads(json.dumps(result.table().to_dicts()))
    assert (
        research["summary"]["groupbt_sharpe"] == result.table().to_dicts()[-1]["sharpe"]
    )


def test_report_cli_show_and_html(research):
    path = research["out"] / "tearsheet.json"
    shown = run("-m", "ashare.report", "show", str(path), "--format", "json")
    payload = json.loads(shown.stdout)
    sheet = Tearsheet.load(path)
    assert payload["summary"]["summary"]["final_nav"] == sheet.summary["final_nav"]
    text = run("-m", "ashare.report", "show", str(path), "--section", "all").stdout
    assert "合成数据低波动" in text
    target = research["out"] / "again.html"
    written = run("-m", "ashare.report", "html", str(path), "-o", str(target))
    assert Path(written.stdout.strip()) == target
    assert target.read_text(encoding="utf-8") == (
        research["out"] / "report.html"
    ).read_text(encoding="utf-8")


def test_report_cli_rejects_bad_input(research):
    bad = research["out"] / "broken.json"
    bad.write_text("{not json", encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "-m", "ashare.report", "show", str(bad)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 2
    assert "Traceback" not in completed.stderr
