"""大模型研究循环（`LLMResearch`）：校验先于评估、去重、停止条件、费用、转写记录、
只有训练段信息进入提示词、在合成行情上端到端找到埋入的信号。全部用假模型，不联网。

来源：LLM 挖掘任务书（2026-11-14）。变异检查（泄漏守卫、评估前校验、预算停止、去重）
的结果记在 docs/LLM_MINING.md。
"""

import io
import json
import math
import re

import httpx
import numpy as np
import pytest

from ashare.factor.mining import LLMResearch, Space, Split, dedup_key, mine, open_ledger
from ashare.factor.mining.llm import (
    ChatClient,
    LLMError,
    Pricing,
    Reply,
    Usage,
    parse,
    prompt,
    render,
)
from ashare.factor.mining.llm.strategy import check_location
from ashare.factor.mining.synthetic import market, planted_expression

SPACE = Space.default()
PLANTED = "-(close / ts_delay(close, 5))"
A = "ts_corr(close, volume, window=20)"
A_SWAPPED = "ts_corr(volume, close, window=20)"  # ts_corr 对称：去重键相同
B = "negate(ts_delta(log(close), periods=5))"
C = "ts_rank(volume, window=20)"
D = "(ts_max(high, window=20) - close) / close"


def answer(*exprs, notes="笔记", action="explore", hypothesis="假设"):
    return json.dumps({"action": action, "hypothesis": hypothesis, "rationale": "理由",
                       "factors": [{"name": f"f{i}", "expression": e, "idea": "想法"}
                                   for i, e in enumerate(exprs)],
                       "notes": notes}, ensure_ascii=False)  # fmt: skip


class Scripted:
    """按顺序给出预先写好的回答；记录每一轮收到的 messages。"""

    model = "fake-model"

    def __init__(self, answers, usage=(1000, 200, 0), pricing=None):
        self.answers = list(answers)
        self.prompts = []
        self.usage = Usage(*usage)
        self.pricing = pricing or Pricing()

    def describe(self):
        return {"model": self.model}

    def complete(self, messages, *, json_mode=True, seed=None):
        self.prompts.append(messages)
        a = self.answers.pop(0) if self.answers else answer()
        if callable(a):
            a = a(messages)
        if isinstance(a, Exception):
            raise a
        return Reply(a, self.usage, self.pricing.cost(self.usage))

    def user(self, i):
        return self.prompts[i][1]["content"]


class FakeEvaluate:
    """模拟 `mine` 交给策略的 evaluate：按文本给适应度，记下每批；带两个只读入口。"""

    def __init__(self, fitness=None, details=None, root=None):
        self.fitness = fitness or {}
        self.details = details or {}
        self.batches = []
        self.task = {"fitness": "information_ratio", "complexity": 0.0, "screen_t": 1.0,
                     "root": str(root) if root else None}  # fmt: skip

    def __call__(self, batch):
        self.batches.append([render(e) for e in batch])
        return [self.fitness.get(render(e), -math.inf) for e in batch]

    def train_feedback(self, expr):
        return self.details.get(
            render(expr), {"stage": "screen", "ic_mean": 0.001, "ic_t": 0.3}
        )


def norm(text):
    return render(parse(text, SPACE))


def run(client, tmp_path, evaluate=None, **kw):
    s = LLMResearch(client, transcript=tmp_path / "t.jsonl", **kw)
    ev = evaluate or FakeEvaluate()
    s.run(SPACE, ev)
    return s, ev


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


# ---- 校验先于评估 -----------------------------------------------------------------


def test_invalid_expressions_are_not_evaluated_and_fed_back(tmp_path):
    client = Scripted([answer("ts_foo(close, window=5)", "close", "close + volume", A),
                       answer(B)])  # fmt: skip
    s, ev = run(client, tmp_path, rounds=2, batch=4)
    assert ev.batches == [[norm(A)], [norm(B)]]
    second = client.user(1)
    assert "未知算子 ts_foo" in second
    assert "根节点必须无量纲" in second
    assert "量纲不同" in second
    assert s.last_run["invalid"] == 3 and s.last_run["evaluated"] == 2
    rec = records(tmp_path / "t.jsonl")[1]
    assert [i["expression"] for i in rec["invalid"]] == ["ts_foo(close, window=5)", "close",
                                                       "close + volume"]  # fmt: skip


def test_malformed_json_is_fed_back_not_fatal(tmp_path):
    client = Scripted(
        ['我想想……{"factors": [', "```json\n" + answer(A) + "\n```", "[1, 2]"]
    )
    s, ev = run(client, tmp_path, rounds=3, batch=2)
    assert ev.batches == [[norm(A)]]
    assert "无法使用" in client.user(1) and "JSON" in client.user(1)
    assert "无法使用" not in client.user(2)  # 第 3 轮的提示词来自第 2 轮的正常回答
    recs = [r for r in records(tmp_path / "t.jsonl") if r["event"] == "round"]
    assert "answer_error" in recs[0] and "answer_error" not in recs[1]
    assert "answer_error" in recs[2]
    assert s.last_run["rounds"] == 3


def test_extra_and_non_string_factors(tmp_path):
    client = Scripted([json.dumps({"factors": [A, {"expression": 3}, B, C]})])
    s, ev = run(client, tmp_path, rounds=1, batch=3)
    assert ev.batches == [[norm(A), norm(B)]]  # 只看前 batch 个：A、坏条目、B
    rec = records(tmp_path / "t.jsonl")[1]
    assert rec["invalid"][0]["error"] == "表达式必须是字符串"


def test_extra_factors_beyond_batch_are_ignored(tmp_path):
    client = Scripted([answer(A, B, C), answer()])
    s, ev = run(client, tmp_path, rounds=2, batch=2)
    assert ev.batches == [[norm(A), norm(B)]]
    assert "多给的 1 个表达式被忽略" in client.user(1)


# ---- 去重 -------------------------------------------------------------------------


def test_dedup_within_and_across_rounds(tmp_path):
    client = Scripted([answer(A, A_SWAPPED, f"cs_rank({C})"),
                       answer(A, B, C)])  # fmt: skip
    s, ev = run(client, tmp_path, rounds=2, batch=3)
    # 第 1 轮：A 与换序的 A 等价；第 2 轮：A 交过，C 与 cs_rank(C)（根上的单调变换）等价
    assert ev.batches == [[norm(A), norm(f"cs_rank({C})")], [norm(B)]]
    assert s.last_run["duplicates"] == 3
    assert "与第 1 轮提交过的表达式等价" in client.user(1)
    keys = [dedup_key(parse(t, SPACE)) for b in ev.batches for t in b]
    assert len(keys) == len(set(keys))


# ---- 停止条件 ---------------------------------------------------------------------


def test_budget_usd_stops_before_next_round(tmp_path):
    pricing = Pricing(
        input=500.0, cached_input=0.0, output=0.0
    )  # 每轮 1000 token = $0.5
    client = Scripted([answer(A), answer(B), answer(C), answer(D)], pricing=pricing)
    s, ev = run(client, tmp_path, rounds=4, batch=1, budget_usd=1.0)
    assert len(client.prompts) == 2
    assert s.last_run["stop"] == "budget_usd"
    assert s.last_run["cost_usd"] == pytest.approx(1.0)


def test_budget_tokens_stops(tmp_path):
    client = Scripted([answer(A), answer(B), answer(C)], usage=(700, 300, 0))
    s, _ = run(client, tmp_path, rounds=3, batch=1, budget_tokens=1500)
    assert len(client.prompts) == 2 and s.last_run["stop"] == "budget_tokens"
    assert (
        s.last_run["prompt_tokens"] == 1400 and s.last_run["completion_tokens"] == 600
    )


def test_consecutive_rounds_without_valid_candidates_stop(tmp_path):
    client = Scripted([answer("close"), "not json", answer(A), answer("volume")] * 3)
    s, ev = run(client, tmp_path, rounds=10, batch=1, patience=2)
    # 第 1、2 轮都没有可评估的 → 第 2 轮后停
    assert len(client.prompts) == 2 and ev.batches == []
    assert s.last_run["stop"] == "no_valid"


def test_all_duplicates_count_as_empty_round(tmp_path):
    client = Scripted([answer(A), answer(A_SWAPPED), answer(A), answer(B)])
    s, ev = run(client, tmp_path, rounds=4, batch=1, patience=2)
    assert len(client.prompts) == 3 and s.last_run["stop"] == "no_valid"
    assert ev.batches == [[norm(A)]]


def test_rounds_exhausted_and_llm_error(tmp_path):
    client = Scripted([answer(A), answer(B)])
    s, _ = run(client, tmp_path, rounds=2, batch=1)
    assert s.last_run["stop"] == "rounds" and s.last_run["rounds"] == 2
    client = Scripted([answer(A), LLMError("模型接口返回 500")])
    s, ev = run(client, tmp_path, rounds=5, batch=1)
    assert s.last_run["stop"] == "llm_error" and s.last_run["rounds"] == 1
    assert ev.batches == [[norm(A)]]
    assert records(tmp_path / "t.jsonl")[-1]["event"] == "end"


# ---- 提示词：历史、笔记、建议 ----------------------------------------------------


def test_history_best_notes_and_advice(tmp_path):
    fitness = {norm(A): 0.8, norm(B): 1.7, norm(C): -0.2}
    details = {norm(B): {"stage": "backtest", "ic_mean": 0.031, "ic_t": 4.2,
                         "long_ir": 1.7, "long_excess": 0.123}}  # fmt: skip
    client = Scripted([answer(A, B, notes="反转有效"), answer(C, hypothesis="换手", notes=None),
                       answer(D), answer(), answer()])  # fmt: skip
    ev = FakeEvaluate(fitness, details)
    s, _ = run(client, tmp_path, evaluate=ev, rounds=5, batch=2, stall=2, notes_limit=4)
    third = client.user(2)
    best = third.split("## 迄今训练段上最好的因子")[1].split("## 最近几轮")[0]
    assert best.index(norm(B)) < best.index(norm(A))
    assert "适应度 1.700（IC 0.0310，IC t 4.20，多头 IR 1.70，多头超额 12.3%）" in best
    assert "反转有效" in client.user(1) and "反转有效" in third  # 没给新笔记时保留旧的
    assert "笔记" not in client.user(0).split("## 研究笔记")[1].split("## 本轮任务")[0]
    # 第 1 轮建议探索；第 2 轮刚有进步建议细化；第 2、3 轮都没进步后建议探索
    assert "建议本轮探索新假设" in client.user(0)
    assert "建议本轮细化当前最好的因子" in client.user(1)
    assert "建议本轮探索新假设" in client.user(3)
    # notes_limit=4：笔记被截短
    assert records(tmp_path / "t.jsonl")[1]["notes"] == "反转有效"[:4]


# ---- 泄漏守卫 ---------------------------------------------------------------------


def test_only_whitelisted_training_fields_reach_the_prompt(tmp_path):
    details = {norm(A): {"stage": "backtest", "ic_mean": 0.02, "ic_t": 2.5,
                         "valid_ir": 9.87654, "valid": {"ir": 8.7654},
                         "end": "20250315", "long_ir": float("nan")}}  # fmt: skip
    client = Scripted([answer(A), answer()])
    ev = FakeEvaluate({norm(A): 0.5}, details)
    run(client, tmp_path, evaluate=ev, rounds=2, batch=1)
    text = json.dumps(client.prompts, ensure_ascii=False)
    for leak in ("9.87", "8.76", "20250315", "2025", "valid", "nan"):
        assert leak not in text
    assert "IC t 2.50" in client.user(1)
    assert prompt.train_only(details[norm(A)]) == {"stage": "backtest", "ic_mean": 0.02,
                                                    "ic_t": 2.5}  # fmt: skip


def test_mine_feedback_hook_reads_only_training_segment():
    from ashare.factor.mining.search import _train_feedback

    row = {"stage": "backtest", "duplicate_of": None,
           "train": {"ic": {"mean": 0.01, "t": 2.0},
                     "long": {"information_ratio": 1.1, "excess_return": 0.05}},
           "valid": {"ic": {"mean": 0.9, "t": 9.0},
                     "long": {"information_ratio": 9.9, "excess_return": 0.99}}}  # fmt: skip
    out = _train_feedback(row)
    assert out == {"stage": "backtest", "ic_mean": 0.01, "ic_t": 2.0, "long_ir": 1.1,
                   "long_excess": 0.05, "duplicate": False}  # fmt: skip
    assert _train_feedback(None) is None
    assert _train_feedback({"stage": "gpu_screen", "duplicate_of": None, "train": None}) == {
        "stage": "gpu_screen", "ic_mean": None, "ic_t": None, "long_ir": None,
        "long_excess": None, "duplicate": False}  # fmt: skip


# ---- 身份、费用、转写记录 ---------------------------------------------------------


def test_describe_tracks_prompt_and_settings(monkeypatch):
    s = LLMResearch(Scripted([]), rounds=7, batch=5, seed=3, budget_usd=1.5)
    d = s.describe()
    assert d["strategy"] == "llm" and d["model"] == "fake-model"
    assert (d["rounds"], d["batch"], d["seed"], d["budget_usd"]) == (7, 5, 3, 1.5)
    assert d["template"] == prompt.TEMPLATE_VERSION
    before = d["prompt_sha"]
    monkeypatch.setattr(
        prompt, "TEMPLATES", (*prompt.TEMPLATES[:-1], prompt.TASK + "改")
    )
    assert s.describe()["prompt_sha"] != before


def test_transcript_under_root_without_api_key(tmp_path):
    key = "sk-test-SECRET-abcdef"
    body = {"choices": [{"message": {"content": answer(A, B)}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1200, "completion_tokens": 300}}  # fmt: skip
    seen = []

    def handler(request):
        seen.append(request.headers["authorization"])
        return httpx.Response(200, json=body)

    client = ChatClient(api_key=key, transport=httpx.MockTransport(handler),
                        pricing=Pricing(input=1.0, cached_input=0.0, output=2.0))  # fmt: skip
    s = LLMResearch(client, rounds=2, batch=2, clock=lambda: 1_700_000_000.0)
    ev = FakeEvaluate(root=tmp_path / "mining")
    s.run(SPACE, ev)
    path = tmp_path / "mining" / "llm" / s.last_run["transcript"].rsplit("/", 1)[-1]
    assert path.exists() and str(path) == s.last_run["transcript"]
    raw = path.read_text()
    assert key not in raw and seen == [f"Bearer {key}"] * 2
    recs = records(path)
    assert [r["event"] for r in recs] == ["start", "round", "round", "end"]
    r1 = recs[1]
    assert len(r1["prompt_sha"]) == 64 and r1["response"] == answer(A, B)
    assert r1["hypothesis"] == "假设" and r1["results"][0]["expression"] == norm(A)
    # 费用：每轮 (1200 × 1 + 300 × 2) / 1e6
    assert s.last_run["cost_usd"] == pytest.approx(2 * 1800 / 1e6)
    assert recs[-1]["cost_usd"] == pytest.approx(2 * 1800 / 1e6)


def test_transcript_may_not_go_into_the_repository(tmp_path):
    from ashare.factor.mining.llm import strategy as mod

    repo = mod._REPO
    with pytest.raises(ValueError, match="不能写进仓库"):
        check_location(repo / "docs" / "x.jsonl")
    with pytest.raises(ValueError, match="不能写进仓库"):
        LLMResearch(Scripted([]), transcript=repo / "src" / "t.jsonl")
    assert check_location(repo / "data" / "mining" / "llm" / "x.jsonl")
    assert check_location(tmp_path / "x.jsonl")


# ---- 接上 mine：合成行情 ----------------------------------------------------------


def test_finds_planted_signal_end_to_end(tmp_path):
    """假模型先给垃圾和随便的候选、再给一轮坏 JSON，第 3 轮看到反馈后提出埋入的信号。"""
    p = market(t=800, n=100, seed=21, strength=0.003)
    split = Split.recent(p.dates, holdout_years=1, valid_years=1)

    def third(messages):
        user = messages[1]["content"]
        assert "无法使用" in user and norm(A) in user  # 看到了上一轮的错误和更早的结果
        return answer(PLANTED, C, action="refine", hypothesis="短期反转")

    vol = "ts_std(close / ts_delay(close, periods=1), window=20)"
    client = Scripted([answer("ts_bogus(close)", A, "ts_zscore(volume, window=20)"), "{oops",
                       third, answer(PLANTED, vol)])  # fmt: skip
    s = LLMResearch(client, rounds=4, batch=3)
    root = tmp_path / "m"
    res = mine(p, split=split, space=SPACE, strategy=s, root=root)
    planted = dedup_key(parse(PLANTED, SPACE))
    assert planted == dedup_key(planted_expression())
    rows = {r["key"]: r for r in res.candidates}
    assert rows[planted]["stage"] == "backtest"
    assert res.selected[0]["key"] == planted
    assert res.summary["significant"]["ic_holm"] >= 1
    # 每个评估过的模型候选都是台账里的一次试验；不合法的、重复的不是
    assert s.last_run["evaluated"] == 5 and s.last_run["invalid"] == 1
    assert s.last_run["duplicates"] == 1
    assert res.summary["unique"] == 5
    assert open_ledger(root=root).trials() == 5
    # 第 4 轮的提示词里，埋入的信号排在最好的因子第一位，带训练段指标
    best = client.user(3).split("## 迄今训练段上最好的因子")[1].split("## 最近几轮")[0]
    assert best.lstrip().startswith(f"- {norm(PLANTED)}：适应度")
    assert (root / "llm").is_dir() and len(list((root / "llm").glob("*.jsonl"))) == 1


def _leak_panel():
    return market(t=420, n=60, seed=3, strength=0.003)


def _leak_split(p):
    return Split(valid_start=str(p.dates[200]), holdout_start=str(p.dates[320]))


def _leak_run(p, tmp_path, name):
    client = Scripted([answer(A, B, C), answer(PLANTED, D), answer()])
    s = LLMResearch(client, rounds=3, batch=3)
    res = mine(p, split=_leak_split(p), space=SPACE, strategy=s, root=tmp_path / name,
               screen_t=0.5)  # fmt: skip
    return client, res


def _perturb(p, start):
    rng = np.random.default_rng(11)
    changed = {}
    for name in ("open", "high", "low", "close", "vwap"):
        x = np.array(p[name], copy=True)
        x[start:] *= np.exp(rng.standard_normal(x[start:].shape) * 0.05)
        changed[name] = x
    return p.replace(**changed)


def test_prompts_see_only_the_training_segment(tmp_path):
    """改动验证段和留出段的行情：每一轮的提示词逐字不变；验证段指标确实变了（比较是敏感的），
    提示词里也没有训练段结束以后的日期。对照：改动训练段的行情，提示词就会变。"""
    p = _leak_panel()
    split = _leak_split(p)
    _, v, _ = split.rows(p.dates)
    base, res = _leak_run(p, tmp_path, "a")
    moved, res2 = _leak_run(_perturb(p, v), tmp_path, "b")
    assert base.prompts == moved.prompts
    va = {r["key"]: json.dumps(r["valid"], sort_keys=True) for r in res.candidates}
    vb = {r["key"]: json.dumps(r["valid"], sort_keys=True) for r in res2.candidates}
    assert va.keys() == vb.keys() and any(va[k] != vb[k] for k in va)
    assert res.summary["backtested"] > 0
    text = json.dumps(base.prompts, ensure_ascii=False)
    later = {str(d) for d in p.dates[v:]}
    assert not any(d in text for d in re.findall(r"\d{8}", text) if d in later)
    assert not re.search(r"\d{4}-\d{2}-\d{2}", text)
    # 验证段的数字（信息比率、IC）不出现在提示词里
    for r in res.candidates:
        lo = (r["valid"] or {}).get("long") or {}
        ir = lo.get("information_ratio")
        if ir is not None and abs(ir) > 0.05:
            assert f"多头 IR {ir:.2f}" not in text
    # 对照：训练段改了，提示词就不同
    early, _ = _leak_run(_perturb(p, v - 120), tmp_path, "c")
    assert early.prompts != base.prompts


def test_cli_synthetic_run(tmp_path):
    from ashare.factor.mining.llm.__main__ import main

    client = Scripted([answer(A, PLANTED), answer(B)])
    out = io.StringIO()
    got = main(["--synthetic", "--days", "420", "--stocks", "60", "--rounds", "2",
                "--batch", "2", "--root", str(tmp_path / "cli")], client=client, out=out)  # fmt: skip
    text = out.getvalue()
    assert "大模型：2 轮" in text and "转写记录" in text and "费用 $" in text
    assert got["llm"]["evaluated"] == 3
    assert got["summary"]["strategy"]["strategy"] == "llm"
    assert (tmp_path / "cli" / "ledger.jsonl").exists()


def test_earlier_submissions_are_listed_once(tmp_path):
    """评估过但已不在「最好」和「上一轮」里的表达式列在「更早提交过的」一节，避免模型重复。"""
    details = {norm(C): {"stage": "screen", "ic_t": 0.42}}
    client = Scripted([answer(C), answer(A), answer(B), answer()])
    ev = FakeEvaluate({norm(A): 0.3}, details)
    run(client, tmp_path, evaluate=ev, rounds=4, batch=1)
    third = client.user(2).split("## 更早提交过的其他表达式")[1].split("## ")[0]
    assert f"- {norm(C)}：未过粗筛（IC t 0.42）" in third
    assert norm(A) not in third  # 在「最好」与「上一轮」里，不重复列
    fourth = client.user(3).split("## 更早提交过的其他表达式")[1].split("## ")[0]
    assert norm(C) in fourth and norm(B) not in fourth  # B 是上一轮的


def test_gpu_screen_option(tmp_path):
    """``screen="gpu"`` 下照常跑（GPU 判为不过线的只给阶段，没有训练段指标）。"""
    from ashare.factor.mining import gpu

    ok, why = gpu.available()
    if not ok:
        pytest.skip(f"GPU 不可用：{why}")
    p = _leak_panel()
    client = Scripted([answer(A, B, C), answer(PLANTED, D)])
    s = LLMResearch(client, rounds=2, batch=3)
    res = mine(p, split=_leak_split(p), space=SPACE, strategy=s, root=tmp_path / "g",
               screen="gpu", screen_t=0.5)  # fmt: skip
    rows = {r["key"]: r for r in res.candidates}
    assert rows[dedup_key(parse(PLANTED, SPACE))]["stage"] == "backtest"
    assert s.last_run["evaluated"] == 5 and len(client.prompts) == 2


def test_truncated_answer_hint_and_zero_history(tmp_path):
    """回答被截断（finish_reason=length）时提示精简；history=0、recent=0 不保留任何轮次。"""

    class Cut(Scripted):
        def complete(self, messages, *, json_mode=True, seed=None):
            r = super().complete(messages, json_mode=json_mode, seed=seed)
            return Reply(r.text, r.usage, r.cost, finish_reason="length")

    client = Cut([answer(A), answer(B)[:40], answer(C)])
    run(client, tmp_path, rounds=3, batch=1, history=0, recent=0)
    assert "被截断" in client.user(2)
    assert "## 最近几轮\n（无）" in client.user(1) and "第 1 轮（" not in client.user(1)
    assert "## 迄今训练段" not in client.user(0)
