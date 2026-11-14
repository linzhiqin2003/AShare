"""`LLMResearch`：让大模型按「假设 → 因子 → 回测反馈 → 笔记」的循环提出候选。

思路来自微软 RD-Agent（R&D-Agent-Quant）的研究循环，在本框架里重写：模型只写本仓库的
表达式语言，由 `mine` 的评估器打分；提示词是自己写的。每一轮：

1. 拼提示词（场景、空间、语法例子、迄今最好的因子、最近几轮的假设与结果、上一轮的校验
   错误、模型自己维护的研究笔记），附一个「探索新假设 / 细化当前最好」的建议；
2. 模型回 JSON：假设、理由、若干表达式、更新后的笔记；
3. 每个表达式先解析并按空间校验（`parse`），不合法的不评估（不算试验），错误原样反馈；
   与本次搜索里已经提交过的去重键相同的也不评估；
4. 合法的新候选一批交给 ``evaluate``，拿回训练段适应度；`mine` 提供的训练段细节
   （``evaluate.train_feedback``）只经白名单（`prompt.train_only`）进入下一轮提示词；
5. 每轮写一行转写记录（JSONL）。

停止：轮数用完；累计费用或 token 达到预算（在下一轮开始前判断，最多超出一轮）；连续
``patience`` 轮没有可评估的候选；模型接口报错（已评估的照常记账、出结果）。

模型的回答不是确定性的：同样的设置重跑会提出不同的候选（``seed`` 只在接口支持时传过去）。
转写记录是这次搜索的完整依据；台账照常按试验去重和累计。
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

from ..ledger import default_root
from ..space import dedup_key
from .client import LLMError
from .parse import ParseError, parse, render
from .prompt import (
    TEMPLATE_VERSION,
    AnswerError,
    build_messages,
    feedback_line,
    messages_sha,
    parse_answer,
    short_outcome,
    template_sha,
    train_only,
)

# 仓库根目录：转写记录不许写进仓库（数据目录 data/ 除外，它不入库）
_REPO = Path(__file__).resolve().parents[5]


@dataclass
class _State:
    """跨轮的记忆：提示词只从这里取历史。"""

    best: list = field(default_factory=list)  # [(适应度, 表达式文本, 训练段细节)]
    rounds: list = field(default_factory=list)  # 最近几轮的摘要
    last: list = field(default_factory=list)  # 上一轮的逐条反馈
    last_exprs: list = field(default_factory=list)  # 上一轮评估过的表达式文本
    submitted: list = field(
        default_factory=list
    )  # 最近提交过的 [(表达式文本, 简短结果)]
    notes: str = ""


def _fit(x):
    return float(x) if isinstance(x, (int, float)) and math.isfinite(x) else -math.inf


def _keep_last(items, n):
    del items[: max(len(items) - n, 0)]


def check_location(path):
    """转写记录的位置：不许在仓库里（仓库下的 data/ 不入库，可以）。"""
    path = Path(path).resolve()
    if path.is_relative_to(_REPO) and not path.is_relative_to(_REPO / "data"):
        raise ValueError(
            f"转写记录是私有研究输出，不能写进仓库目录：{path}（用数据根目录或仓库外的目录）"
        )
    return path


class LLMResearch:
    """大模型研究循环策略。

    - client：`ChatClient` 或同接口的对象（``complete``、``model``、``describe``）；
    - rounds：最多几轮；batch：每轮要几个表达式；seed：传给接口（接口支持时）；
    - budget_usd / budget_tokens：累计费用（美元）/ token 上限，None 为不限；
    - patience：连续多少轮没有可评估的候选就停；stall：连续多少轮没有提高最好适应度后建议探索；
    - best_k、history、recent：提示词里列出的最好因子个数、最近几轮个数、更早提交过的表达式
      个数；notes_limit：笔记字数上限；
    - transcript：转写记录文件；默认 ``<mine 的 root>/llm/<时间>-<身份>.jsonl``。

    跑完后 ``last_run`` 里有轮数、停止原因、token、费用、转写记录路径。"""

    def __init__(
        self,
        client,
        rounds=20,
        batch=8,
        seed=0,
        *,
        budget_usd=None,
        budget_tokens=None,
        patience=3,
        stall=2,
        best_k=8,
        history=6,
        recent=40,
        notes_limit=1500,
        transcript=None,
        clock=time.time,
    ):
        if rounds < 1 or batch < 1:
            raise ValueError("rounds、batch 至少为 1")
        self.client = client
        self.rounds, self.batch, self.seed = int(rounds), int(batch), int(seed)
        self.budget_usd = None if budget_usd is None else float(budget_usd)
        self.budget_tokens = None if budget_tokens is None else int(budget_tokens)
        self.patience, self.stall = int(patience), int(stall)
        self.best_k, self.history, self.notes_limit = (
            int(best_k),
            int(history),
            int(notes_limit),
        )
        self.recent = int(recent)
        self.transcript = None if transcript is None else check_location(transcript)
        self._clock = clock
        self.last_run = None

    def describe(self):
        client = self.client.describe() if hasattr(self.client, "describe") else {}
        return {
            "strategy": "llm",
            "model": getattr(self.client, "model", None),
            "client": client,
            "template": TEMPLATE_VERSION,
            "prompt_sha": template_sha(),
            "rounds": self.rounds,
            "batch": self.batch,
            "seed": self.seed,
            "budget_usd": self.budget_usd,
            "budget_tokens": self.budget_tokens,
            "patience": self.patience,
            "stall": self.stall,
            "best_k": self.best_k,
            "history": self.history,
            "recent": self.recent,
            "notes_limit": self.notes_limit,
        }

    # ---- 运行 ---------------------------------------------------------------------

    def _path(self, task):
        if self.transcript is not None:
            return self.transcript
        root = Path(task["root"]) if task.get("root") else default_root()
        ident = hashlib.sha256(
            json.dumps(self.describe(), sort_keys=True, default=str).encode()
        ).hexdigest()[:10]
        stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(self._clock()))
        return check_location(root / "llm" / f"{stamp}-{ident}.jsonl")

    def run(self, space, evaluate):
        task = dict(getattr(evaluate, "task", None) or {})
        details_of = getattr(evaluate, "train_feedback", None)
        path = self._path(task)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = _State()
        seen = {}  # 去重键 → 第几轮提交
        stats = {"rounds": 0, "stop": "rounds", "prompt_tokens": 0, "completion_tokens": 0,
                 "cost_usd": 0.0, "evaluated": 0, "invalid": 0, "duplicates": 0,
                 "transcript": str(path)}  # fmt: skip
        self.last_run = stats
        best_fit, stall, empty = -math.inf, 0, 0

        def write(record):
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

        write({"event": "start", "time": self._clock(), "strategy": self.describe(),
               "task": {k: task.get(k) for k in ("fitness", "complexity", "screen_t")},
               "space": space.digest})  # fmt: skip
        for r in range(1, self.rounds + 1):
            reason = self._over_budget(stats)
            if reason:
                stats["stop"] = reason
                break
            advice = (
                "探索新假设" if r == 1 or stall >= self.stall else "细化当前最好的因子"
            )
            messages = build_messages(space=space, task=task, state=state, round_no=r,
                                      rounds=self.rounds, batch=self.batch, advice=advice,
                                      notes_limit=self.notes_limit)  # fmt: skip
            record = {"event": "round", "round": r, "time": self._clock(),
                      "prompt_sha": messages_sha(messages), "advice": advice}  # fmt: skip
            try:
                reply = self.client.complete(
                    messages, json_mode=True, seed=self.seed * 1000 + r
                )
            except LLMError as exc:
                stats["stop"] = "llm_error"
                stats["error"] = str(exc)
                write({**record, "error": str(exc)})
                break
            stats["rounds"] = r
            stats["prompt_tokens"] += reply.usage.prompt_tokens
            stats["completion_tokens"] += reply.usage.completion_tokens
            stats["cost_usd"] += reply.cost
            record.update(response=reply.text, usage=vars(reply.usage), cost_usd=reply.cost,
                          finish_reason=reply.finish_reason)  # fmt: skip
            try:
                answer = parse_answer(reply.text, self.batch)
            except AnswerError as exc:
                cut = (
                    "（回答超出长度上限被截断：精简 rationale 和 notes）"
                    if (reply.finish_reason == "length")
                    else ""
                )
                state.last = [
                    f"- 上一轮的回答无法使用：{exc}{cut}。只输出约定格式的一个 JSON 对象。"
                ]
                state.last_exprs = []
                record["answer_error"] = str(exc)
                write(record)
                empty += 1
                stall += 1
                if empty >= self.patience:
                    stats["stop"] = "no_valid"
                    break
                continue
            fresh, lines, invalid, dups = [], [], [], []
            for name, text, idea in answer["factors"][: self.batch]:
                try:
                    expr = parse(text, space)
                except ParseError as exc:
                    invalid.append({"expression": text, "error": str(exc)})
                    lines.append(f"- {str(text)[:200]!r}：不合法，{exc}")
                    continue
                key = dedup_key(expr)
                if key in seen:
                    dups.append({"expression": text, "round": seen[key]})
                    lines.append(
                        f"- {render(expr)}：与第 {seen[key]} 轮提交过的表达式等价，未评估"
                    )
                    continue
                seen[key] = r
                fresh.append((name, render(expr), idea, expr))
            if answer["extra"]:
                lines.append(
                    f"- 多给的 {answer['extra']} 个表达式被忽略（每轮只要 {self.batch} 个）"
                )
            fits = list(evaluate([f[3] for f in fresh])) if fresh else []
            results, scored = [], []
            for (name, text, idea, expr), fit in zip(fresh, fits, strict=True):
                details = train_only(details_of(expr)) if details_of is not None else {}
                fit = _fit(fit)
                results.append({"name": name, "expression": text, "idea": idea,
                                "fitness": fit if math.isfinite(fit) else None,
                                "train": details})  # fmt: skip
                scored.append(feedback_line(text, fit, details))
                state.submitted.append((text, short_outcome(fit, details)))
                if math.isfinite(fit):
                    state.best.append((fit, text, details))
            state.best.sort(key=lambda x: (-x[0], x[1]))
            del state.best[self.best_k :]
            round_best = max((_fit(f) for f in fits), default=-math.inf)
            improved = round_best > best_fit
            best_fit = max(best_fit, round_best)
            stall = 0 if improved else stall + 1
            empty = 0 if fresh else empty + 1
            stats["evaluated"] += len(fresh)
            stats["invalid"] += len(invalid)
            stats["duplicates"] += len(dups)
            state.last = scored + lines
            state.last_exprs = [f[1] for f in fresh]
            state.rounds.append({
                "round": r, "action": answer["action"], "hypothesis": answer["hypothesis"][:200],
                "valid": len(fresh) + len(dups), "evaluated": len(fresh),
                "best": f"{round_best:.3f}" if math.isfinite(round_best) else "无",
            })  # fmt: skip
            _keep_last(state.rounds, self.history)
            _keep_last(state.submitted, self.recent)
            if answer["notes"] is not None:
                state.notes = answer["notes"].strip()[: self.notes_limit]
            record.update(action=answer["action"], hypothesis=answer["hypothesis"],
                          rationale=answer["rationale"], notes=state.notes, results=results,
                          invalid=invalid, duplicates=dups)  # fmt: skip
            write(record)
            if empty >= self.patience:
                stats["stop"] = "no_valid"
                break
        write({"event": "end", "time": self._clock(),
               **{k: v for k, v in stats.items() if k != "transcript"}})  # fmt: skip

    def _over_budget(self, stats):
        if self.budget_usd is not None and stats["cost_usd"] >= self.budget_usd:
            return "budget_usd"
        tokens = stats["prompt_tokens"] + stats["completion_tokens"]
        if self.budget_tokens is not None and tokens >= self.budget_tokens:
            return "budget_tokens"
        return None
