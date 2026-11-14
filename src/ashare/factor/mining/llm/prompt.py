"""研究循环的提示词：模板、各节的拼装、回答的解析。

模板（下面的常量文本）和拼装函数一起定版：``TEMPLATE_VERSION`` 手工递增，``template_sha()``
是全部模板文本的哈希，进入策略的 ``describe()``——改了提示词，搜索身份随之改变。

提示词里只出现：场景说明、空间允许的字段与算子、语法例子、训练段的反馈（适应度、训练段
IC 均值与 t 值、训练段多头端信息比率与超额）、模型自己写的假设与笔记、上一轮的校验错误。
不出现任何日期、验证段与留出段的数字（`feedback_line` 只读白名单字段）。
"""

from __future__ import annotations

import hashlib
import json
import math

from ..space import CATALOG, Unit
from ..strategies import random_candidate
from .parse import ParseError, parse, render

TEMPLATE_VERSION = "llm-research/1"

SYSTEM = """你是 A 股量化研究员，任务是在给定的表达式语言里提出截面选股因子，并根据回测反馈迭代改进。
工作方式：先提出一个可检验的市场假设（为什么某种量价形态会预测下一段收益），再把它落成几个具体的因子表达式；
看到反馈后判断假设是否成立，决定继续细化还是换一个方向，并把学到的东西写进研究笔记。
只用给出的字段、算子和参数取值；表达式只能用到当天收盘及以前的数据（语言里的算子都只往过去看）。
只输出一个 JSON 对象，不要输出其他文字。"""

SCENARIO = """## 场景
- 数据：中国 A 股日频行情（复权价格、成交量、成交额、换手率），全市场股票每天一个截面。
- 因子：每个交易日收盘后对每只股票算出一个数，按截面排序；数越大越看好。
- 检验：在样本的训练段上，因子值最高的一组股票等权持有，下一个交易日成交，按固定间隔调仓，扣交易成本，
  与同样成交规则下持有全体股票等权的基准比较。
- 适应度：{fitness}。先过粗筛：训练段日截面 Rank IC 的 t 值不低于 {screen_t}，否则不回测、记为「未过粗筛」。
- 因子只做多头端选股（A 股做空受限），所以看重排在最前面那一组的表现，而不只是整体相关。
- 每个提交的表达式都算一次试验并计入多重检验，重复或换写法的提交不会带来新信息。"""

FITNESS_TEXT = {
    "information_ratio": "训练段多头端相对基准的超额收益信息比率（年化），越大越好",
    "excess_return": "训练段多头端相对基准的年化超额收益，越大越好",
    "ic": "训练段日截面 Rank IC 的均值，越大越好",
    "ic_t": "训练段日截面 Rank IC 的 t 值，越大越好",
}
UNIT_TEXT = {
    "price": "价格",
    "volume": "成交量",
    "amount": "成交额（价格×量）",
    "ratio": "无量纲",
}
FIELD_TEXT = {
    "open": "复权开盘价",
    "high": "复权最高价",
    "low": "复权最低价",
    "close": "复权收盘价",
    "vwap": "复权成交均价",
    "volume": "复权成交量",
    "amount": "成交额",
    "turnover": "换手率",
}
OP_TEXT = {
    "add": "a + b（量纲相同）",
    "subtract": "a - b（量纲相同；两个同底对数相减得无量纲）",
    "multiply": "a * b",
    "divide": "a / b（价格除以价格得无量纲）",
    "maximum": "逐元素取大（量纲相同）",
    "minimum": "逐元素取小（量纲相同）",
    "negate": "取负，也可写 -x",
    "abs": "绝对值",
    "sign": "符号，结果无量纲",
    "log": "自然对数；log(价格) 是对数量，差分或相减后无量纲",
    "ts_delay": "periods 天前的值",
    "ts_delta": "与 periods 天前的差",
    "ts_mean": "过去 window 天均值（含当天）",
    "ts_sum": "过去 window 天求和",
    "ts_std": "过去 window 天标准差",
    "ts_max": "过去 window 天最大值",
    "ts_min": "过去 window 天最小值",
    "ts_median": "过去 window 天中位数",
    "ts_decay_linear": "过去 window 天线性衰减加权平均",
    "ts_ema": "指数移动平均",
    "ts_rank": "当天值在过去 window 天里的分位，无量纲",
    "ts_zscore": "相对过去 window 天的标准分，无量纲",
    "ts_skew": "过去 window 天偏度，无量纲",
    "ts_slope": "过去 window 天线性回归斜率",
    "ts_argmax": "过去 window 天最大值距今天数",
    "ts_argmin": "过去 window 天最小值距今天数",
    "ts_corr": "两个序列过去 window 天的相关系数，无量纲",
    "cs_rank": "当天截面百分位排名，无量纲",
    "cs_zscore": "当天截面标准分，无量纲",
}
SYNTAX = """## 语法
- 字段直接写名字；算子写成函数调用，操作数在前、参数用关键字写在后面，如 ts_mean(close, window=20)。
- 加减乘除可以写成中缀 a + b、a - b、a * b、a / b（对应 add、subtract、multiply、divide，要求这些算子可用），-x 表示 negate(x)。
- 没有数值常量：数字只能作为算子参数，而且只能取上面列出的值。
- 例子（都满足本空间的约束）：
{examples}"""

RULES = """## 结构约束
- 最大深度 {max_depth}（字段的深度为 0），最多 {max_nodes} 个节点（字段也算）。
- 最终结果必须无量纲（价格水平、成交量水平本身不是候选）；量纲规则见算子说明，加减和取大取小要求两边量纲相同。
- {cs_rule}
- 不允许显然冗余的写法：同一算子直接套自己（如排名套排名、取负套取负）、两个相同的操作数做二元运算。"""

HISTORY = """## 迄今训练段上最好的因子
{best}

## 最近几轮
{rounds}

## 更早提交过的其他表达式（不要再交，也不要只换写法）
{earlier}"""

FEEDBACK_HEAD = """## 上一轮的结果
{last}"""

NOTES = """## 研究笔记（你上一轮写的，下面会让你更新）
{notes}"""

TASK = """## 本轮任务
第 {round} 轮，共 {rounds} 轮。建议本轮{advice}（你可以不采纳，但要在 action 里写明选择）：
- "explore"：提出一个与之前不同的新假设；
- "refine"：围绕当前最好的因子做改进（换窗口、换归一化、换相近的字段、组合两个有效成分）。
提出 {batch} 个互不相同、此前没提交过的表达式。只输出如下 JSON：
{{"action": "explore 或 refine", "hypothesis": "一句话假设", "rationale": "为什么这样设计",
 "factors": [{{"name": "简短英文名", "expression": "表达式", "idea": "这个表达式检验什么"}}],
 "notes": "更新后的研究笔记：哪些方向有效、哪些无效、下一步打算（不超过 {notes_limit} 字）"}}"""

TEMPLATES = (SYSTEM, SCENARIO, json.dumps(FITNESS_TEXT, ensure_ascii=False, sort_keys=True),
             json.dumps(UNIT_TEXT, ensure_ascii=False, sort_keys=True),
             json.dumps(FIELD_TEXT, ensure_ascii=False, sort_keys=True),
             json.dumps(OP_TEXT, ensure_ascii=False, sort_keys=True),
             SYNTAX, RULES, HISTORY, FEEDBACK_HEAD, NOTES, TASK)  # fmt: skip


def template_sha():
    """全部模板文本（含拼装规则的版本号）的哈希。"""
    text = json.dumps([TEMPLATE_VERSION, *TEMPLATES], ensure_ascii=False)
    return hashlib.sha256(text.encode()).hexdigest()


# ---- 空间的说明 -------------------------------------------------------------------


def _unit_name(unit):
    for name, u in (("price", Unit(1, 0)), ("volume", Unit(0, 1)), ("amount", Unit(1, 1)),
                    ("ratio", Unit())):  # fmt: skip
        if unit == u:
            return UNIT_TEXT[name]
    return str(unit)


def describe_space(space):
    """字段与算子两节文本。"""
    fields = [
        f"- {name}：{FIELD_TEXT.get(name, '字段')}，量纲 {_unit_name(unit)}"
        for name, unit in space.fields.items()
    ]
    ops = []
    for name, grid in space.grids.items():
        rule = CATALOG[name]
        values = "；".join(
            f"{p} ∈ {{{', '.join(str(v) for v in sorted({g[p] for g in grid}))}}}"
            for p in rule.params
        )
        args = ", ".join("ab"[i] for i in range(rule.arity))
        sig = (
            f"{name}({args}"
            + (", " + ", ".join(f"{p}=…" for p in rule.params) if rule.params else "")
            + ")"
        )
        ops.append(
            f"- {sig}：{OP_TEXT.get(name, name)}" + (f"；{values}" if values else "")
        )
    return "## 字段\n" + "\n".join(fields) + "\n\n## 算子\n" + "\n".join(ops)


_EXAMPLE_SOURCES = (
    "ts_corr(close, volume, window=20)",
    "negate(ts_delta(log(close), periods=5))",
    "cs_rank(ts_mean(turnover, window=20))",
    "ts_std(close / ts_delay(close, periods=1), window=20)",
    "(ts_max(high, window=20) - close) / close",
    "ts_rank(volume, window=20)",
)


def examples(space, k=4):
    """从空间里挑 k 个合法例子：先试固定的几条，不够再用固定种子随机生成。"""
    out = []
    for text in _EXAMPLE_SOURCES:
        try:
            out.append(render(parse(text, space)))
        except ParseError:
            continue
        if len(out) == k:
            return out
    import numpy as np

    rng = np.random.default_rng(20261114)
    for _ in range(50):
        if len(out) >= k:
            break
        try:
            text = render(random_candidate(space, rng))
        except ValueError:
            break
        if text not in out:
            out.append(text)
    return out


def rules(space):
    if space.allow_level_ranks:
        cs = "截面算子（cs_*）"
    else:
        cs = "截面算子（cs_*）的参数必须无量纲"
    if space.cs_positions == "root":
        cs += "，而且只能出现在最外层"
    return RULES.format(
        max_depth=space.max_depth, max_nodes=space.max_nodes, cs_rule=cs + "。"
    )


# ---- 反馈 -------------------------------------------------------------------------

# 允许进入提示词的反馈字段（全部来自训练段）；其余字段一律丢弃。
FEEDBACK_FIELDS = ("stage", "ic_mean", "ic_t", "long_ir", "long_excess", "duplicate")


def train_only(details):
    """只留白名单字段，数值必须有限；不认识的键（验证段、留出段、日期……）全部丢弃。"""
    out = {}
    for key in FEEDBACK_FIELDS:
        value = (details or {}).get(key)
        if key == "stage":
            if isinstance(value, str):
                out[key] = value
        elif key == "duplicate":
            if isinstance(value, bool):
                out[key] = value
        elif (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
        ):
            out[key] = float(value)
    return out


def _num(x, digits=3):
    return f"{x:.{digits}f}"


def short_outcome(fitness, details):
    """一个候选的简短结果（「更早提交过的」一节用）。"""
    d = train_only(details)
    if fitness is not None and math.isfinite(fitness):
        return f"适应度 {_num(fitness)}"
    if "ic_t" in d:
        return f"未过粗筛（IC t {_num(d['ic_t'], 2)}）"
    return {"screen": "未过粗筛", "failed": "求值失败"}.get(d.get("stage"), "无适应度")


def feedback_line(text, fitness, details):
    """一条候选的反馈：表达式、适应度和训练段指标。"""
    d = train_only(details)
    if fitness is not None and math.isfinite(fitness):
        head = f"适应度 {_num(fitness)}"
    elif d.get("stage") == "screen":
        head = "未过粗筛"
    elif d.get("stage") == "failed":
        head = "求值失败"
    else:
        head = "无适应度"
    extra = []
    if "ic_mean" in d:
        extra.append(f"IC {_num(d['ic_mean'], 4)}")
    if "ic_t" in d:
        extra.append(f"IC t {_num(d['ic_t'], 2)}")
    if "long_ir" in d:
        extra.append(f"多头 IR {_num(d['long_ir'], 2)}")
    if "long_excess" in d:
        extra.append(f"多头超额 {d['long_excess'] * 100:.1f}%")
    if d.get("duplicate"):
        extra.append("与已入选因子高度相关")
    return f"- {text}：{head}" + (f"（{'，'.join(extra)}）" if extra else "")


# ---- 拼装 -------------------------------------------------------------------------


def build_messages(*, space, task, state, round_no, rounds, batch, advice, notes_limit):
    """拼出一轮的 messages（system + user）。state 是 `strategy._State`。"""
    fit = task.get("fitness")
    fitness = FITNESS_TEXT.get(fit, "训练段上自定义的适应度，越大越好")
    complexity = task.get("complexity") or 0
    if complexity:
        fitness += f"；每个节点扣 {complexity:g}"
    screen_t = task.get("screen_t")
    parts = [
        SCENARIO.format(
            fitness=fitness,
            screen_t=_num(screen_t, 2) if screen_t is not None else "门槛",
        ),
        describe_space(space),
        SYNTAX.format(examples="\n".join(f"  - {e}" for e in examples(space))),
        rules(space),
    ]
    if round_no > 1:
        best = (
            "\n".join(feedback_line(t, f, d) for f, t, d in state.best)
            or "（还没有通过粗筛的因子）"
        )
        recent = (
            "\n".join(
                f"- 第 {r['round']} 轮（{r['action']}）{r['hypothesis']} → 有效 {r['valid']} 个、"
                f"评估 {r['evaluated']} 个、本轮最好 {r['best']}"
                for r in state.rounds
            )
            or "（无）"
        )
        shown = {t for _, t, _ in state.best} | set(state.last_exprs)
        earlier = (
            "\n".join(
                f"- {t}：{short}" for t, short in state.submitted if t not in shown
            )
            or "（无）"
        )
        parts.append(HISTORY.format(best=best, rounds=recent, earlier=earlier))
    if state.last:
        parts.append(FEEDBACK_HEAD.format(last="\n".join(state.last)))
    parts.append(NOTES.format(notes=state.notes or "（空）"))
    parts.append(TASK.format(round=round_no, rounds=rounds, advice=advice, batch=batch,
                             notes_limit=notes_limit))  # fmt: skip
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "\n\n".join(parts)}]  # fmt: skip


def messages_sha(messages):
    text = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()


# ---- 回答 -------------------------------------------------------------------------


class AnswerError(ValueError):
    """回答不是约定的 JSON。"""


def parse_answer(text, batch, text_limit=400):
    """模型回答 → dict(action, hypothesis, rationale, factors=[(name, expression, idea)], notes)。

    容忍 ```json 围栏和前后的多余文字：取第一个 ``{`` 起能解析的那个对象。"""
    if not isinstance(text, str) or "{" not in text:
        raise AnswerError("回答里没有 JSON 对象")
    start = text.index("{")
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[start:])
    except ValueError as exc:
        raise AnswerError(
            f"JSON 解析失败：{exc.msg}（第 {exc.pos} 个字符附近）"
        ) from None
    if not isinstance(obj, dict):
        raise AnswerError("JSON 顶层必须是对象")
    factors = obj.get("factors")
    if not isinstance(factors, list):
        raise AnswerError("缺少 factors 列表")

    def s(x, limit=text_limit):
        return x.strip()[:limit] if isinstance(x, str) else ""

    out = []
    for item in factors:
        if isinstance(item, str):
            out.append(("", item, ""))
        elif isinstance(item, dict):
            out.append(
                (s(item.get("name"), 60), item.get("expression"), s(item.get("idea")))
            )
        else:
            out.append(("", None, ""))
    action = s(obj.get("action"), 20).lower()
    return {
        "action": action if action in ("explore", "refine") else "explore",
        "hypothesis": s(obj.get("hypothesis")),
        "rationale": s(obj.get("rationale")),
        "factors": out,
        "extra": max(len(out) - batch, 0),
        "notes": obj.get("notes") if isinstance(obj.get("notes"), str) else None,
    }
