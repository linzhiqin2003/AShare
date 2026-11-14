"""大模型研究循环：表达式文本解析（白名单、参数取值、量纲与结构约束）与回写文本。

来源：LLM 挖掘任务书（2026-11-14）——模型写的文本必须在评估前按空间校验，错误要能
原样反馈给模型；解析器不执行任何 Python。
"""

import numpy as np
import pytest

from ashare.factor import call, field
from ashare.factor.mining import Space, canonical
from ashare.factor.mining.llm import ParseError, parse, render
from ashare.factor.mining.llm.parse import MAX_NESTING, MAX_TEXT
from ashare.factor.mining.strategies import random_candidate
from ashare.factor.mining.synthetic import planted_expression

SPACE = Space.default()


def test_accepts_function_and_infix_forms():
    want = canonical(planted_expression())
    for text in (
        "negate(close / ts_delay(close, periods=5))",
        "-(close / ts_delay(close, 5))",
        "  `negate(divide(close, ts_delay(close, periods=5)))`  ",
        "-( close/ts_delay( close , periods = 5 ) )",
    ):
        assert parse(text, SPACE).digest == want.digest, text


def test_nested_calls_and_precedence():
    e = parse("ts_corr(ts_mean(close, window=5) / close, volume, window=20)", SPACE)
    c = field("close")
    want = call("ts_corr", call("ts_mean", c, window=5) / c, field("volume"), window=20)
    assert e.digest == canonical(want).digest
    # a - b - c 是 (a - b) - c；右边要括号才是另一棵树
    big = Space.default(max_nodes=12)
    a = parse("close / open - high / low - vwap / close", big)
    b = parse("close / open - (high / low - vwap / close)", big)
    assert a.digest != b.digest


def test_commutative_operands_are_canonicalised():
    # 可交换算子的操作数顺序不影响结果（空间要求规范顺序，解析时自动排好）
    a = parse("ts_mean(volume, window=5) * ts_std(close, window=20) / amount", SPACE)
    b = parse("ts_std(close, window=20) * ts_mean(volume, window=5) / amount", SPACE)
    assert a.digest == b.digest


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("ts_foo(close, window=5)", "未知算子 ts_foo"),
        ("ts_skew(close, window=5)", "未知算子 ts_skew"),  # CATALOG 里有、空间里没有
        ("cs_rank(ts_mean(pe, window=5))", "未知字段 pe"),
        ("ts_mean(close / open, window=7)", "不在允许的取值里"),
        ("ts_mean(close / open, 7.5)", "不在允许的取值里"),
        ("ts_mean(close / open, span=5)", "没有参数 span"),
        ("ts_mean(close / open)", "缺少参数 window"),
        ("ts_mean(close / open, window=5, window=10)", "重复"),
        ("ts_mean(close / open, 5, 10)", "参数过多"),
        ("ts_mean(window=5, close / open)", "操作数要写在参数前面"),
        ("ts_corr(close, window=10)", "需要 2 个操作数"),
        ("close / open - 1", "数值常量"),
        ("-1 * close / open", "数值常量"),
        ("close", "根节点必须无量纲"),
        ("ts_mean(close, window=20)", "根节点必须无量纲"),
        ("cs_rank(close)", "截面算子的参数必须无量纲"),
        ("close + volume", "量纲不同"),
        ("negate(negate(close / open))", "套在自己上冗余"),
        ("close / close", "两个操作数相同"),
        (
            "ts_mean(ts_mean(ts_mean(ts_mean(close / open, window=5), window=5), "
            "window=5), window=5)",
            "深度",
        ),
        (
            "ts_corr(close / open, high / low, window=10) * ts_rank(volume, window=5)",
            "节点数",
        ),
        ("close(5)", "是字段"),
        ("ts_mean", "后面要跟括号"),
        ("ts_mean(close / open, window=5", "提前结束"),
        ("close / open)", "多余的内容"),
        ("", "为空"),
        ("__import__('os').system('true')", "无法识别的字符"),
        ("close.__class__", "无法识别的字符"),
        ("我觉得应该用动量", "无法识别的字符"),
        ("ts_mean(close / open, window=x)", "必须是数字"),
    ],
)
def test_rejects_with_reason(text, message):
    with pytest.raises(ParseError, match=message):
        parse(text, SPACE)


def test_infix_needs_the_operator_in_space():
    small = Space(fields={"close": "price", "open": "price"},
                  ops={"subtract": {}, "log": {}}, max_depth=2, max_nodes=5)  # fmt: skip
    assert parse("log(close) - log(open)", small)
    with pytest.raises(ParseError, match="算子 divide 不在空间里"):
        parse("close / open", small)
    with pytest.raises(ParseError, match="negate"):
        parse("-(log(close) - log(open))", small)


def test_rejects_non_strings_and_oversized_text():
    for junk in (None, 3, ["close"], {"expression": "close"}):
        with pytest.raises(ParseError):
            parse(junk, SPACE)
    with pytest.raises(ParseError, match=str(MAX_TEXT)):
        parse("close /" + " " * MAX_TEXT + "open", SPACE)
    deep = "(" * (MAX_NESTING + 1) + "close / open" + ")" * (MAX_NESTING + 1)
    with pytest.raises(ParseError, match="嵌套"):
        parse(deep, SPACE)


def test_render_round_trips_random_candidates():
    space = Space.default(max_depth=4, max_nodes=9)
    rng = np.random.default_rng(7)
    for _ in range(400):
        e = random_candidate(space, rng)
        text = render(e)
        assert parse(text, space).digest == e.digest, text


def test_render_uses_only_declared_params():
    text = render(parse("ts_mean(close / open, window=5)", SPACE))
    assert text == "ts_mean(close / open, window=5)"
    assert "min_periods" not in text
