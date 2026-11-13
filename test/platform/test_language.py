"""输入资源边界和白名单：恶意语法永远不会进入求值器。"""

import builtins

import pytest

from ashare.factor.formula import dsl
from ashare.factor.library import Factor
from ashare.platform.config import expand_settings
from ashare.platform.language import ExpressionError, completions, parse


@pytest.mark.parametrize(
    "source",
    [
        "__import__('os').system('true')",
        "().__class__",
        "close.__class__.__mro__",
        "close[0]",
        "[x for x in close]",
        "(lambda: 1)()",
        "import os",
        "from os import system",
        "open('file')",
        "ts_std(returns, window=20)",
        "getattr(close, 'shape')",
        "a.b.c",
        "a = __builtins__",
        "a = 'secret'",
        "a = True",
        "a += 1",
        "(a,b) = close",
        "[20,60]",
        "ts_mean(close, [])",
        "ts_mean(close, [1,2,3,4,5,6,7,8,9])",
        "ts_mean(close, 1e300)",
        "9" * 5000,
        "9" * 1000,
        " " * 16001,
        "rank(" * 50 + "close" + ")" * 50,
        "(" * 500 + "close" + ")" * 500,
        "close ** 100000000",
        "ts_std(returns, 1000000000 * 1000000000)",
        "ts_std(returns, 1 / 0)",
        "ts_std(returns, 999999)",
        "ts_std(returns, -1)",
        "a=1\n" * 300,
        "rank(close)\nwhile True: pass",
        "close < cap < 10",
    ],
)
def test_rejects_unsafe_source_without_execution(source, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("调用了 Python 执行入口")

    # 因子节点已经导入；解析过程不得调用任何动态执行入口。
    monkeypatch.setattr(builtins, "eval", forbidden)
    monkeypatch.setattr(builtins, "exec", forbidden)
    with pytest.raises(ExpressionError) as error:
        parse(source, factors={})
    assert error.value.line >= 1
    assert error.value.column >= 1


def test_scan_cartesian_and_assignment_reuse():
    variants = parse(
        "# 注释\nvol = ts_std(returns, [20, 60])\nalpha = -vol", factors={}
    )
    assert [v.expression.digest for v in variants] == [
        (-dsl.stddev(dsl.returns, w)).digest for w in (20, 60)
    ]
    assert (
        len(
            expand_settings(
                {"group_neutralize": ["industry", "cap+industry"]}, len(variants)
            )
        )
        * len(variants)
        == 4
    )
    assert (
        len(parse("ts_mean(close, [1,2,3,4]) + ts_std(close, [5,6])", factors={})) == 8
    )
    with pytest.raises(ExpressionError, match="最多运行 8 组"):
        parse("ts_mean(close, [1,2,3]) + ts_std(close, [4,5,6])", factors={})
    with pytest.raises(ValueError, match="最多运行 8 组"):
        expand_settings({"group_neutralize": [None, "industry", "cap"]}, 3)


def test_normalization_comments_names_and_registered_factor():
    a = parse("a=rank(close)\na", factors={})[0]
    b = parse(" # 注释\nb = rank( close )", factors={})[0]
    assert a.expression.digest == b.expression.digest
    factors = {"custom": Factor("custom", dsl.rank(dsl.close), "示例")}
    assert parse("-custom", factors=factors)[0].expression == -dsl.rank(dsl.close)
    with pytest.raises(ExpressionError, match="未知函数 ts_stdd") as error:
        parse("# 第一行\na=1\nalpha=ts_stdd(close, 5)", factors={})
    assert (error.value.line, error.value.column) == (3, 7)
    assert parse("n=10+10\n-ts_std(returns,n)", factors={})[
        0
    ].expression == -dsl.stddev(dsl.returns, 20)


@pytest.mark.parametrize(
    "settings",
    [
        {"window": ["20260101", "99991231"]},
        {"TEST": True},
        {"vector_neut": [1]},
        {"delay": 2},
        {"n_groups": True},
        {"trade_limitation": 1},
        {"decay": -1},
        {"universe": "TOP5000"},
        {"group_neutralize": []},
        {"decay": 1.5},
    ],
)
def test_settings_are_strict(settings):
    with pytest.raises(ValueError):
        expand_settings(settings)


def test_completions_include_signatures_and_best_sharpe():
    items = {c["label"]: c for c in completions({"reversal_20": 1.25})}
    assert items["ts_std"]["signature"] == "ts_std(x, window)"
    assert items["returns"]["kind"] == "field"
    assert items["reversal_20"]["hint"] == "1.25"
