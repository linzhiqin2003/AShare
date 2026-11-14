"""大模型研究循环的跨次记忆（`llm.memory`）：存、取、防泄漏、口径、向量检索与退路。不联网。

来源：2026-11-14 用户要求加跨次记忆（向量模型 qwen/qwen3-embedding-8b，经 OpenRouter）。
首次 20 轮之前的试跑里，第 2、3 轮大多是同一因子的参数变体，每次搜索都从零开始。
"""

import json
import math

import httpx
import numpy as np
import pytest

from ashare.factor.mining import Space
from ashare.factor.mining.llm import LLMError, Pricing, Reply, Usage, parse, render
from ashare.factor.mining.llm.client import EmbeddingClient
from ashare.factor.mining.llm.memory import Context, Memory, recall_text
from ashare.factor.mining.llm.strategy import LLMResearch

SPACE = Space.default()
A = "ts_corr(close, volume, window=20)"
B = "negate(ts_delta(log(close), periods=5))"
C = "ts_rank(volume, window=20)"
SPLIT = {
    "train_start": "20160101",
    "valid_start": "20250102",
    "holdout_start": "20260105",
}


def norm(text):
    return render(parse(text, SPACE))


def answer(*exprs, hypothesis="假设", notes="笔记", action="explore"):
    return json.dumps({"action": action, "hypothesis": hypothesis, "rationale": "理由",
                       "factors": [{"name": f"f{i}", "expression": e, "idea": "想法"}
                                   for i, e in enumerate(exprs)],
                       "notes": notes}, ensure_ascii=False)  # fmt: skip


class Scripted:
    model = "fake-model"

    def __init__(self, answers, cost=0.001):
        self.answers, self.prompts, self.cost = list(answers), [], cost

    def describe(self):
        return {"model": self.model}

    def complete(self, messages, *, json_mode=True, seed=None):
        self.prompts.append(messages)
        a = self.answers.pop(0) if self.answers else answer()
        return Reply(a, Usage(1000, 200, 0), self.cost)

    def user(self, i):
        return self.prompts[i][1]["content"]


class FakeEvaluate:
    def __init__(self, fitness=None, details=None, split=SPLIT, fit_name="information_ratio",
                 evaluation="E1"):  # fmt: skip
        self.fitness, self.details = fitness or {}, details or {}
        self.task = {"fitness": fit_name, "complexity": 0.0, "screen_t": 1.0,
                     "root": None, "split": split, "evaluation": evaluation}  # fmt: skip

    def __call__(self, batch):
        return [self.fitness.get(render(e), -math.inf) for e in batch]

    def train_feedback(self, expr):
        return self.details.get(render(expr), {"stage": "screen", "ic_t": 0.3})


class KeywordEmbedder:
    """按关键词给向量：同一关键词的文本向量相同；记下每次请求的文本。"""

    model = "fake-embed"
    words = ("换手", "动量", "反转", "波动", "低")

    def __init__(self, fail=False):
        self.calls, self.fail = [], fail

    def embed(self, texts):
        if self.fail:
            raise LLMError("向量接口返回 503")
        self.calls.append(list(texts))
        out = []
        for t in texts:
            v = np.array(
                [1.0 if w in t else 0.0 for w in self.words] + [0.01], np.float32
            )
            out.append(v)
        return out, 0.0001 * len(texts)


def research(tmp_path, client, memory, evaluate=None, name="t", **kw):
    s = LLMResearch(client, transcript=tmp_path / f"{name}.jsonl", memory=memory, **kw)
    s.run(SPACE, evaluate or FakeEvaluate())
    return s


def mem(tmp_path, **kw):
    kw.setdefault("tag", "cn")
    return Memory(tmp_path / "memory", **kw)


def section(prompt, title):
    return prompt.split(title)[1].split("\n## ")[0] if title in prompt else None


# ---- 存与取 ------------------------------------------------------------------------


def test_second_run_sees_first_run_best_factors_hypotheses_and_notes(tmp_path):
    ev = FakeEvaluate({norm(A): 0.42})
    research(tmp_path, Scripted([answer(A, B, hypothesis="量价背离预示反转", notes="量价相关有效")]),
             mem(tmp_path), ev, name="r1", rounds=1, batch=2)  # fmt: skip
    client = Scripted([answer(C)])
    s = research(tmp_path, client, mem(tmp_path), ev, name="r2", rounds=1, batch=1)
    first = client.user(0)
    memo = section(first, "## 以往研究")
    assert memo is not None and "以往共 1 次搜索、1 轮、2 个因子" in memo
    assert f"- {norm(A)}：" in memo and "0.420" in memo  # 以往最好的因子及训练段适应度
    assert norm(B) not in memo.split("###")[1]  # 没过粗筛的不进「最好」
    assert "量价背离预示反转" in memo
    assert "量价相关有效" in section(first, "## 研究笔记")  # 接上上一次的笔记
    start = json.loads((tmp_path / "r2.jsonl").read_text().splitlines()[0])
    assert start["memory"]["seeded_notes"] is True
    assert s.describe()["memory"]["tag"] == "cn"


def test_first_run_has_no_memory_section(tmp_path):
    client = Scripted([answer(A)])
    research(tmp_path, client, mem(tmp_path), rounds=1, batch=1)
    assert "## 以往研究" not in client.user(0)


def test_own_run_entries_are_not_recalled(tmp_path):
    """本次搜索的历史已在「最近几轮」里，记忆只取以往搜索的。"""
    client = Scripted([answer(A, hypothesis="第一轮假设"), answer(B)])
    research(
        tmp_path, client, mem(tmp_path), FakeEvaluate({norm(A): 0.3}), rounds=2, batch=1
    )
    assert "## 以往研究" not in client.user(1)


def test_only_train_whitelist_is_stored(tmp_path):
    leak = {"stage": "backtest", "ic_t": 2.0, "valid_ir": 9.9, "holdout_excess": 0.5}
    research(tmp_path, Scripted([answer(A)]), mem(tmp_path),
             FakeEvaluate({norm(A): 0.4}, {norm(A): leak}), rounds=1, batch=1)  # fmt: skip
    raw = (tmp_path / "memory" / "entries.jsonl").read_text()
    assert "valid_ir" not in raw and "holdout_excess" not in raw and "9.9" not in raw
    assert '"ic_t": 2.0' in raw


# ---- 防泄漏与口径 ----------------------------------------------------------------------


def _seed_run(tmp_path, split=SPLIT, tag="cn", fit_name="information_ratio", fit=0.5):
    research(tmp_path, Scripted([answer(A, hypothesis="以往的假设", notes="以往的笔记")]),
             mem(tmp_path, tag=tag), FakeEvaluate({norm(A): fit}, split=split, fit_name=fit_name),
             name=f"seed-{tag}-{split['valid_start']}-{fit_name}", rounds=1, batch=1)  # fmt: skip


def test_entries_whose_train_reaches_current_validation_are_not_recalled(tmp_path):
    later = {**SPLIT, "valid_start": "20260105", "holdout_start": "20260601"}
    _seed_run(
        tmp_path, split=later
    )  # 训练段到 2026-01，越过本次 2025-01-02 的验证段起点
    client = Scripted([answer(C)])
    research(tmp_path, client, mem(tmp_path), name="now", rounds=1, batch=1)
    assert "## 以往研究" not in client.user(0)
    assert "以往的笔记" not in client.user(0)


def test_earlier_validation_start_is_admitted(tmp_path):
    earlier = {**SPLIT, "valid_start": "20240102"}
    _seed_run(tmp_path, split=earlier)
    client = Scripted([answer(C)])
    research(tmp_path, client, mem(tmp_path), name="now", rounds=1, batch=1)
    memo = section(client.user(0), "## 以往研究")
    assert memo is not None and "以往的假设" in memo
    # 训练段不同，口径不同：只列假设，不带数字，也不进「最好」
    assert "评估口径不同" in memo and "0.500" not in memo


def test_other_dataset_tag_is_not_recalled(tmp_path):
    _seed_run(tmp_path, tag="synthetic")
    client = Scripted([answer(C)])
    research(tmp_path, client, mem(tmp_path, tag="cn"), name="now", rounds=1, batch=1)
    assert "## 以往研究" not in client.user(0)


def test_other_fitness_is_listed_without_numbers(tmp_path):
    _seed_run(tmp_path, fit_name="ic_mean")
    client = Scripted([answer(C)])
    research(tmp_path, client, mem(tmp_path), name="now", rounds=1, batch=1)
    memo = section(client.user(0), "## 以往研究")
    assert "以往的假设" in memo and "0.500" not in memo
    assert "以往同口径下还没有通过粗筛的因子" in memo


def test_missing_split_neither_stores_nor_recalls(tmp_path):
    research(tmp_path, Scripted([answer(A)]), mem(tmp_path), FakeEvaluate(split=None),
             rounds=1, batch=1)  # fmt: skip
    assert not (tmp_path / "memory" / "entries.jsonl").exists()
    # 已有可取的以往条目时，没有切分信息的搜索也不取
    _seed_run(tmp_path)
    client = Scripted([answer(C)])
    research(tmp_path, client, mem(tmp_path), FakeEvaluate(split=None), name="nosplit",
             rounds=1, batch=1)  # fmt: skip
    assert "## 以往研究" not in client.user(0) and "以往的笔记" not in client.user(0)


# ---- 向量检索 ----------------------------------------------------------------------


def _ctx(evaluation="E1"):
    return Context("cn", "information_ratio", 0.0, SPLIT["train_start"],
                   SPLIT["valid_start"], evaluation)  # fmt: skip


def _store(m, run, hypothesis, t, fit=0.1):
    ctx = _ctx()
    m._clock = lambda: t
    m.record_round(ctx, run, 1, {"action": "explore", "hypothesis": hypothesis},
                   [{"expression": A, "fitness": fit, "train": {}}])  # fmt: skip
    return ctx


def test_similar_hypotheses_ranked_by_embedding_and_deduplicated(tmp_path):
    e = KeywordEmbedder()
    m = mem(tmp_path, embedder=e, similar_k=2)
    _store(m, "a", "换手率低的股票更好", 1)  # 向量 (换手, 低)
    _store(m, "a2", "低换手率股票表现好", 2)  # 与上一条向量相同 → 只留一条
    _store(m, "b", "换手率高的股票更差", 3)  # 只有「换手」，相似度 0.71
    ctx = _store(m, "c", "短期反转效应", 4)
    rec = m.recall(ctx, "now", "低换手的方向")
    assert rec.mode == "embedding" and len(rec.similar) == 2
    twins = [line for line in rec.similar if "换手率低" in line or "低换手率" in line]
    assert (
        len(twins) == 1 and twins[0] == rec.similar[0]
    )  # 最相近的排第一，近重复只留一条
    assert "换手率高" in rec.similar[1]  # 第二名轮到相似度次高的，而不是近重复
    calls = sum(len(c) for c in e.calls)
    m.recall(ctx, "now", "低换手的方向")  # 条目向量全部命中缓存，只算查询
    assert sum(len(c) for c in e.calls) == calls + 1
    # 新开一个 Memory 也能读回条目向量的缓存
    e2 = KeywordEmbedder()
    mem(tmp_path, embedder=e2).recall(ctx, "now", "低换手的方向")
    assert [len(c) for c in e2.calls] == [1]
    # 查询向量不进缓存：缓存里只有条目
    rows = (tmp_path / "memory" / "vectors.jsonl").read_text().splitlines()
    assert len(rows) == 4


def test_embedding_failure_falls_back_to_recent(tmp_path):
    m = mem(tmp_path, embedder=KeywordEmbedder(fail=True), similar_k=1)
    _store(m, "a", "旧的假设", 1)
    ctx = _store(m, "b", "新的假设", 2)
    rec = m.recall(ctx, "now", "随便")
    assert rec.mode == "recent" and "503" in rec.warning
    assert "新的假设" in rec.similar[0]


def test_recall_cost_counts_toward_budget(tmp_path):
    m = mem(tmp_path, embedder=KeywordEmbedder())
    _store(m, "old", "换手率", 1)
    client = Scripted([answer(A), answer(B), answer(C)], cost=0.0)
    s = research(tmp_path, client, m, rounds=3, batch=1, budget_usd=0.00015)
    assert s.last_run["memory_cost_usd"] > 0
    assert s.last_run["stop"] == "budget_usd"


def test_corrupt_lines_are_skipped(tmp_path):
    m = mem(tmp_path)
    ctx = _store(m, "a", "完整的一条", 1)
    with (tmp_path / "memory" / "entries.jsonl").open("a") as f:
        f.write('{"v": 1, "kind": "round", "run": "x", "hyp')  # 写到一半的行
    m2 = mem(tmp_path)
    rec = m2.recall(ctx, "now", "")
    assert rec.rounds == 1 and m2.skipped_lines == 1
    assert recall_text(rec)["rounds"] == 1


def test_memory_inside_repo_is_rejected():
    from ashare.factor.mining.llm import strategy

    repo = strategy._REPO
    with pytest.raises(ValueError, match="仓库"):
        Memory(repo / "src" / "memory", tag="cn")


# ---- 向量客户端 --------------------------------------------------------------------------


def _embed_client(handler, **kw):
    return EmbeddingClient(api_key="sk-test-EMBED-123456", transport=httpx.MockTransport(handler),
                           sleep=lambda s: None, **kw)  # fmt: skip


def test_embedding_client_orders_by_index_and_uses_reported_cost():
    seen = []

    def handler(request):
        seen.append(json.loads(request.content))
        body = {"data": [{"index": 1, "embedding": [0.0, 1.0]},
                         {"index": 0, "embedding": [1.0, 0.0]}],
                "usage": {"prompt_tokens": 7, "cost": 2e-7}}  # fmt: skip
        return httpx.Response(200, json=body)

    c = _embed_client(handler)
    vectors, cost = c.embed(["甲", "乙"])
    assert seen[0] == {"model": "qwen/qwen3-embedding-8b", "input": ["甲", "乙"]}
    assert vectors[0].tolist() == [1.0, 0.0] and vectors[1].tolist() == [0.0, 1.0]
    assert cost == pytest.approx(2e-7)


@pytest.mark.parametrize("data", [[{"index": 0, "embedding": [1.0]}],
                                  [{"index": 0, "embedding": [1.0]},
                                   {"index": 1, "embedding": [float("nan")]}],
                                  [{"index": 0, "embedding": []},
                                   {"index": 1, "embedding": [1.0]}]])  # fmt: skip
def test_embedding_client_rejects_wrong_count_or_values(data):
    c = _embed_client(
        lambda r: httpx.Response(200, content=json.dumps({"data": data}).encode())
    )
    with pytest.raises(LLMError):
        c.embed(["甲", "乙"])


def test_embedding_errors_hide_key():
    c = _embed_client(
        lambda r: httpx.Response(
            401, json={"error": {"message": "bad sk-test-EMBED-123456"}}
        )
    )
    with pytest.raises(LLMError) as exc:
        c.embed(["甲"])
    assert "sk-test-EMBED" not in str(exc.value)


def test_pricing_unchanged_for_chat():
    assert Pricing().input == 0.3  # 记忆改动没碰聊天客户端的默认价目


def test_first_write_of_a_fresh_memory_is_counted_once(tmp_path):
    """变异检查发现（2026-11-14）：新开的 Memory 先写文件再读，第一条被读进来又追加一次。"""
    m = mem(tmp_path)
    ctx = _store(m, "a", "唯一的一条", 1)
    assert len(m.entries()) == 1
    rec = m.recall(ctx, "now", "")
    assert rec.rounds == 1 and rec.factors == 1 and len(rec.similar) == 1


def test_similar_k_without_embedder_lists_distinct_entries(tmp_path):
    m = mem(tmp_path, similar_k=5)
    _store(m, "a", "第一条", 1)
    ctx = _store(m, "b", "第二条", 2)
    rec = m.recall(ctx, "now", "")
    assert rec.mode == "recent" and len(rec.similar) == 2
    assert "第二条" in rec.similar[0] and "第一条" in rec.similar[1]


def test_memory_reapplies_train_whitelist(tmp_path):
    """记忆自己再过一遍白名单：调用方给了验证段字段也不落盘。"""
    m = mem(tmp_path)
    ctx = Context(
        "cn", "information_ratio", 0.0, SPLIT["train_start"], SPLIT["valid_start"]
    )
    m.record_round(ctx, "a", 1, {"hypothesis": "h"},
                   [{"expression": A, "fitness": 0.2,
                     "train": {"stage": "backtest", "ic_t": 1.5, "valid_ir": 7.7}}])  # fmt: skip
    raw = (tmp_path / "memory" / "entries.jsonl").read_text()
    assert "valid_ir" not in raw and "7.7" not in raw and '"ic_t": 1.5' in raw


# ---- 独立审查发现（2026-11-14）------------------------------------------------------------


def test_cli_tag_separates_synthetic_seeds_and_real_start(tmp_path):
    """审查 P1-1：所有合成行情共用一个标签，种子 1 的提示词会拿到种子 0 的「答案」。"""
    from ashare.factor.mining.llm.__main__ import _memory, parser

    def tag(*argv):
        args = parser().parse_args(
            [*argv, "--root", str(tmp_path), "--embed-model", "off"]
        )
        return _memory(args).tag

    assert tag("--synthetic", "--seed", "0") != tag("--synthetic", "--seed", "1")
    assert tag("--synthetic", "--days", "400") != tag("--synthetic", "--days", "900")
    assert tag("--start", "20160101") != tag("--start", "20200101")
    assert tag("--memory-tag", "x") == "x"


def test_torn_utf8_line_is_skipped(tmp_path):
    """审查 P1-2：写到多字节字符中间断掉的行让整个记忆读不了、之后每次搜索都崩。"""
    m = mem(tmp_path)
    ctx = _store(m, "a", "完整的一条", 1)
    with (tmp_path / "memory" / "entries.jsonl").open("ab") as f:
        f.write('{"v": 1, "hypothesis": "换'.encode()[:-1])
    m2 = mem(tmp_path)
    assert m2.seed_notes(ctx, "now") is None
    assert m2.recall(ctx, "now", "").rounds == 1 and m2.skipped_lines == 1


def test_append_after_torn_line_is_not_lost(tmp_path):
    """审查 P1/P2-3：坏行后面紧接着写的新条目粘在坏行上，一起丢掉。"""
    m = mem(tmp_path)
    ctx = _store(m, "a", "第一条", 1)
    with (tmp_path / "memory" / "entries.jsonl").open("a") as f:
        f.write('{"v": 1, "kind": "round", "hyp')
    _store(mem(tmp_path), "b", "第二条", 2)
    rec = mem(tmp_path, similar_k=5).recall(ctx, "now", "")
    assert rec.rounds == 2 and any("第二条" in line for line in rec.similar)


class FailSecond(KeywordEmbedder):
    def embed(self, texts):
        if self.calls:
            raise LLMError("向量接口返回 503")
        return super().embed(texts)


def test_partial_embedding_cost_is_counted(tmp_path):
    """审查 P2-4：第一批已扣费、第二批失败时，第一批的费用没有计入预算。"""
    m = mem(tmp_path, embedder=FailSecond(), embed_batch=1)
    _store(m, "a", "换手", 1)
    ctx = _store(m, "b", "动量", 2)
    rec = m.recall(ctx, "now", "随便")
    assert rec.mode == "recent" and rec.cost == pytest.approx(0.0001)


def test_evaluation_definition_must_match_for_numbers_and_notes(tmp_path):
    """审查 P2-6：评估定义（持有期、成本、股票池……）不同的数字被放在一起排名。"""
    research(tmp_path, Scripted([answer(A, hypothesis="以往的假设", notes="以往的笔记")]),
             mem(tmp_path), FakeEvaluate({norm(A): 0.5}, evaluation="E0"), name="old",
             rounds=1, batch=1)  # fmt: skip
    client = Scripted([answer(C)])
    research(tmp_path, client, mem(tmp_path), FakeEvaluate(evaluation="E1"), name="new",
             rounds=1, batch=1)  # fmt: skip
    memo = section(client.user(0), "## 以往研究")
    assert "以往的假设" in memo and "评估口径不同" in memo and "0.500" not in memo
    assert "以往的笔记" not in client.user(0)  # 笔记里可能有数字，口径不同不接


def test_same_transcript_path_gets_distinct_run_ids(tmp_path):
    """审查 P2-7：两次搜索用同一个转写文件时 run 相同，以往那次被当成本次而取不到。"""
    research(tmp_path, Scripted([answer(A, hypothesis="上一次的假设")]), mem(tmp_path),
             FakeEvaluate({norm(A): 0.4}), name="same", rounds=1, batch=1)  # fmt: skip
    client = Scripted([answer(C)])
    research(tmp_path, client, mem(tmp_path), name="same", rounds=1, batch=1)
    assert "上一次的假设" in (section(client.user(0), "## 以往研究") or "")


def test_reused_memory_sees_new_entries_and_skips_malformed_rows(tmp_path):
    """审查 P2-8：同一个 Memory 跨多次搜索时看不到别的进程写的条目；缺字段的行让读取崩溃。"""
    m = mem(tmp_path)
    research(tmp_path, Scripted([answer(A)]), m, FakeEvaluate({norm(A): 0.4}), name="r1",
             rounds=1, batch=1)  # fmt: skip
    _store(mem(tmp_path), "other", "别的进程写的假设", 5)
    with (tmp_path / "memory" / "entries.jsonl").open("a") as f:
        f.write('{"v": 1}\n')
    client = Scripted([answer(C)])
    research(tmp_path, client, m, name="r2", rounds=1, batch=1)
    assert "别的进程写的假设" in section(client.user(0), "## 以往研究")
