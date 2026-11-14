"""命令行（2026-11-14 新增，模拟盘每日运行）：经 ``python -m ashare.paper`` 分发、退出码、账户命令。

旧子命令（start、status、replay…）不受影响由 test/paper 里原有测试守住。
"""

import json

import pytest

from ashare.paper.__main__ import main


def call(argv, capsys):
    with pytest.raises(SystemExit) as exit_:
        main(argv)
    out = capsys.readouterr()
    return exit_.value.code, out.out, out.err


def test_account_commands_and_daily_skip(tmp_path, capsys, monkeypatch):
    paper = tmp_path / "paper"
    data = ["--data-dir", str(tmp_path / "data"), "--paper-dir", str(paper)]
    code, out, _ = call(["account", "list", *data], capsys)
    assert code == 0 and json.loads(out) == []
    code, out, _ = call(
        [
            "account",
            "create",
            "demo",
            "--expression",
            "rank(close)",
            "--settings",
            '{"top_n": 5, "rebalance": "week"}',
            "--capital",
            "50",
            "--disabled",
            *data,
        ],
        capsys,
    )
    created = json.loads(out)
    assert (
        code == 0 and created["enabled"] is False and created["initial_cash"] == 500_000
    )
    code, _, err = call(
        ["account", "create", "demo", "--expression", "rank(close)", *data], capsys
    )
    assert code == 2 and json.loads(err) == {"message": "已有同名账户"}
    code, _, err = call(
        [
            "account",
            "create",
            "x",
            "--expression",
            "rank(close)",
            "--settings",
            "{",
            *data,
        ],
        capsys,
    )
    assert code == 2
    # 停用的账户不参与：两个阶段都干净跳过，退出 0。
    for stage in ("signal", "run"):
        code, out, _ = call(["daily", stage, *data], capsys)
        assert code == 0 and json.loads(out)["message"] == "没有启用的账户，跳过"
    code, out, _ = call(["account", "enable", "demo", *data], capsys)
    assert json.loads(out) == {"name": "demo", "enabled": True}
    code, out, _ = call(["account", "show", "demo", *data], capsys)
    shown = json.loads(out)
    assert (
        shown["account"] == "demo"
        and shown["strategy"]["settings"]["rebalance"] == "week"
    )
    assert shown["plan"]["last_signal_day"] is None
    code, out, _ = call(["account", "list", *data], capsys)
    assert [item["account"] for item in json.loads(out)] == ["demo"]
    code, out, _ = call(["daily", "status", *data], capsys)
    status = json.loads(out)
    assert status["alerts"] == [] and set(status["stages"]) == {"signal", "run"}
    assert (paper / ".ops" / "logs").is_dir()
    code, out, _ = call(["daily", "run", "--until", "08:00", *data], capsys)
    assert code == 2


def test_backtest_file_and_install_preview(tmp_path, capsys):
    paper = tmp_path / "paper"
    data = ["--data-dir", str(tmp_path / "data"), "--paper-dir", str(paper)]
    record = tmp_path / "record.json"
    record.write_text(
        json.dumps(
            dict(
                id="0123456789abcdef01234567",
                name="回测 A",
                source="rank(close)",
                settings={"top_n": 7},
                status="complete",
            ),
            ensure_ascii=False,
        )
    )
    code, out, _ = call(
        ["account", "create", "from-bt", "--backtest-file", str(record), *data], capsys
    )
    assert code == 0 and json.loads(out)["origin"]["kind"] == "backtest"
    code, _, err = call(
        [
            "account",
            "create",
            "x",
            "--backtest-file",
            str(record),
            "--settings",
            '{"top_n": 3}',
            *data,
        ],
        capsys,
    )
    assert code == 2  # 从回测建时设置沿用回测，不能另给
    code, out, _ = call(
        ["install-schedule", "--launchd", "--target", str(tmp_path / "agents"), *data],
        capsys,
    )
    assert code == 0 and "com.ashare.paper.run" in out
    assert not (tmp_path / "agents").exists()
