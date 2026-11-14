"""跨次研究记忆：把每轮的假设、因子和训练段结果存下来，下一次搜索开始时取回。

借 RD-Agent 知识库的思路（按语义取回相近的以往经验），在本框架里重写：

- 存什么：每轮一条（假设、理由摘要、因子表达式及其训练段适应度与白名单细节），每次搜索结束
  再存一条研究笔记。只有训练段的信息——与提示词同一份白名单（`prompt.train_only`）。
- 取什么（每轮拼提示词前）：同一评估口径下以往训练段最好的因子；与当前方向语义相近的以往
  假设及结果（向量检索，失败或没配向量模型时退回「最近的几条」）；第一轮另外接上上一次
  搜索结束时的研究笔记。
- 防泄漏：以往条目的训练段如果延伸进本次的验证段或留出段（条目的 ``valid_start`` 晚于本次），
  一律不取；数据集标签（``tag``，如 ``cn``、``synthetic``）不同的不取；切分信息缺失的不存不取。
- 口径：只有 适应度定义、复杂度惩罚、训练段起止 都相同的条目，适应度数字才放在一起排名；
  口径不同的以往假设仍可作为「相近假设」出现，但不带数字。

文件（目录默认 ``<mine 的 root>/llm/memory/``，不许在仓库里，``data/`` 除外）：
``entries.jsonl`` 一行一条，追加写、加文件锁，读时跳过写坏的行；``vectors.jsonl`` 缓存向量
（按「模型 + 文本」的哈希，float16）。
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .client import LLMError
from .prompt import feedback_line, train_only

VERSION = 1
_TEXT_LIMIT = 400


@dataclass(frozen=True)
class Context:
    """本次搜索的身份：决定哪些以往条目可以取、哪些数字可以放在一起比。"""

    tag: str
    fitness: str | None
    complexity: float
    train_start: str | None
    valid_start: str | None
    evaluation: str | None = (
        None  # `mine` 的评估定义摘要（持有期、调仓、股票池、成本、基准……）
    )

    @classmethod
    def from_task(cls, tag, task):
        split = task.get("split") or {}
        evaluation = task.get("evaluation")
        return cls(str(tag), None if task.get("fitness") is None else str(task["fitness"]),
                   float(task.get("complexity") or 0.0), split.get("train_start"),
                   split.get("valid_start"),
                   None if evaluation is None else str(evaluation))  # fmt: skip

    @property
    def usable(self):
        return bool(self.valid_start)

    def comparable(self, entry):
        return (self.evaluation is not None
                and entry.get("evaluation") == self.evaluation
                and entry.get("fitness") == self.fitness
                and float(entry.get("complexity") or 0.0) == self.complexity
                and entry.get("train_start") == self.train_start
                and entry.get("valid_start") == self.valid_start)  # fmt: skip

    def admits(self, entry):
        """条目能不能进本次的提示词：同一数据集，且它的训练段没有越过本次的验证段起点。"""
        v = entry.get("valid_start")
        return (
            self.usable
            and entry.get("tag") == self.tag
            and isinstance(v, str)
            and v <= self.valid_start
        )


_KINDS = {"round", "notes"}


def _well_formed(e):
    """手写或写坏的行（缺 run、time、id 等）跳过，不让读取崩溃。"""
    return (
        e.get("v") == VERSION
        and e.get("kind") in _KINDS
        and isinstance(e.get("run"), str)
        and isinstance(e.get("id"), str)
        and isinstance(e.get("time"), (int, float))
        and not isinstance(e.get("time"), bool)
        and (e["kind"] != "notes" or isinstance(e.get("notes"), str))
        and (e["kind"] != "round" or isinstance(e.get("factors", []), list))
    )


def _sha(*parts):
    return hashlib.sha256("\x1f".join(map(str, parts)).encode()).hexdigest()


def _encode(v):
    return base64.b64encode(np.asarray(v, dtype=np.float16).tobytes()).decode()


def _decode(s):
    return np.frombuffer(base64.b64decode(s), dtype=np.float16).astype(np.float32)


def _cos(a, b):
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na == 0 or nb == 0 or a.shape != b.shape:
        return -1.0
    return float(a @ b) / (na * nb)


def _fit(x):
    return float(x) if isinstance(x, (int, float)) and math.isfinite(x) else None


def entry_text(entry):
    """用来算向量的文本：假设 + 理由摘要。"""
    return (
        str(entry.get("hypothesis") or "") + "\n" + str(entry.get("rationale") or "")
    )[:_TEXT_LIMIT]


@dataclass
class Recall:
    """一轮取回的结果：给提示词的各段文本，以及写进转写记录的说明。"""

    runs: int = 0
    rounds: int = 0
    factors: int = 0
    best: list = None  # [(适应度, 表达式, 细节)]
    similar: list = None  # 文本行
    mode: str = "none"  # embedding / recent / none
    picked: list = None  # 取回条目的 id
    warning: str | None = None
    cost: float = 0.0

    @property
    def empty(self):
        return not self.rounds


class Memory:
    """跨次研究记忆。``embedder`` 是 `EmbeddingClient` 或同接口对象（``embed(texts)`` →
    (向量列表, 费用)、``model``），None 时相近假设按时间取最近的。"""

    def __init__(self, path, *, tag, embedder=None, top_k=10, similar_k=6,
                 dedup_cos=0.95, embed_batch=64, clock=time.time):  # fmt: skip
        from .strategy import check_location

        self.path = check_location(path)
        self.tag = str(tag)
        self.embedder = embedder
        self.top_k, self.similar_k = int(top_k), int(similar_k)
        self.dedup_cos, self.embed_batch = float(dedup_cos), int(embed_batch)
        self._clock = clock
        self._entries = None
        self._vectors = None
        self.skipped_lines = 0

    def describe(self):
        return {
            "version": VERSION,
            "tag": self.tag,
            "embed_model": getattr(self.embedder, "model", None),
            "top_k": self.top_k,
            "similar_k": self.similar_k,
            "dedup_cos": self.dedup_cos,
        }

    # ---- 文件 -----------------------------------------------------------------

    @property
    def _entries_file(self):
        return self.path / "entries.jsonl"

    @property
    def _vectors_file(self):
        return self.path / "vectors.jsonl"

    def _append(self, file, rows):
        self.path.mkdir(parents=True, exist_ok=True)
        with open(self.path / ".lock", "a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                # 上一次写到一半断掉、文件不以换行结尾时先补一个换行，否则新行粘在坏行上一起丢
                torn = False
                if file.exists() and file.stat().st_size:
                    with file.open("rb") as f:
                        f.seek(-1, 2)
                        torn = f.read(1) != b"\n"
                with file.open("a", encoding="utf-8") as f:
                    if torn:
                        f.write("\n")
                    for row in rows:
                        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _read(self, file):
        rows = []
        if not file.exists():
            return rows
        with file.open(
            "rb"
        ) as f:  # 按字节读、逐行解码：断在多字节字符中间的行只坏它自己
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line.decode("utf-8"))
                except ValueError:  # 含 UnicodeDecodeError
                    self.skipped_lines += 1
                    continue
                if isinstance(row, dict):
                    rows.append(row)
                else:
                    self.skipped_lines += 1
        return rows

    def entries(self):
        if self._entries is None:
            rows = self._read(self._entries_file)
            self._entries = [e for e in rows if _well_formed(e)]
            self.skipped_lines += len(rows) - len(self._entries)
        return self._entries

    def refresh(self):
        """下次读取时重新读文件（别的进程可能写了新条目）。每次搜索开始时调用。"""
        self._entries = None

    def _vector_cache(self):
        if self._vectors is None:
            self._vectors = {}
            for row in self._read(self._vectors_file):
                try:
                    self._vectors[row["sha"]] = _decode(row["v"])
                except (KeyError, ValueError, TypeError):
                    self.skipped_lines += 1
        return self._vectors

    # ---- 写 -------------------------------------------------------------------

    def _add(self, entry):
        # 先把文件读进来再追加：反过来的话，第一次写入后读文件会把这一条读进来，再追加一次就重复了
        # （2026-11-14 变异检查发现：新开的 Memory 第一条记录被算两遍，被向量去重掩盖）
        entries = self.entries()
        self._append(self._entries_file, [entry])
        entries.append(entry)

    def _base(self, ctx, run, kind):
        return {"v": VERSION, "kind": kind, "run": str(run), "time": self._clock(),
                "tag": ctx.tag, "evaluation": ctx.evaluation, "fitness": ctx.fitness,
                "complexity": ctx.complexity,
                "train_start": ctx.train_start, "valid_start": ctx.valid_start}  # fmt: skip

    def record_round(self, ctx, run, round_no, answer, results, invalid=0):
        """存一轮：只存训练段信息（适应度、白名单细节），表达式之外的文字截断。"""
        if not ctx.usable:
            return None
        factors = [{"expression": str(r["expression"]), "fitness": _fit(r.get("fitness")),
                    "train": train_only(r.get("train") or {})}
                   for r in results]  # fmt: skip
        entry = {**self._base(ctx, run, "round"), "round": int(round_no),
                 "action": str(answer.get("action") or "")[:20],
                 "hypothesis": str(answer.get("hypothesis") or "")[:_TEXT_LIMIT],
                 "rationale": str(answer.get("rationale") or "")[:_TEXT_LIMIT],
                 "factors": factors, "invalid": int(invalid)}  # fmt: skip
        entry["id"] = _sha(entry["run"], entry["round"])[:16]
        self._add(entry)
        return entry

    def record_notes(self, ctx, run, notes):
        if not ctx.usable or not str(notes or "").strip():
            return None
        entry = {**self._base(ctx, run, "notes"), "notes": str(notes)[:4000]}
        entry["id"] = _sha(entry["run"], "notes")[:16]
        self._add(entry)
        return entry

    # ---- 取 -------------------------------------------------------------------

    def seed_notes(self, ctx, run):
        """本次之前、可取的最近一份研究笔记（没有为 None）。"""
        # 笔记里可能写了数字：口径不同的不接
        notes = [e for e in self.entries() if e.get("kind") == "notes" and e["run"] != run
                 and ctx.admits(e) and ctx.comparable(e)]  # fmt: skip
        return max(notes, key=lambda e: e["time"])["notes"] if notes else None

    def _embed(self, texts, spent):
        """条目文本 → 向量（先查缓存，缺的成批请求并写回缓存）。每批的费用一到手就记进
        ``spent``（列表），后面的批次失败时已扣的费用不丢。"""
        model = getattr(self.embedder, "model", "")
        cache = self._vector_cache()
        keys = [_sha(model, t) for t in texts]
        missing = list(dict.fromkeys(k for k in keys if k not in cache))
        text_of = dict(zip(keys, texts, strict=True))
        for i in range(0, len(missing), self.embed_batch):
            chunk = missing[i : i + self.embed_batch]
            vectors, c = self.embedder.embed([text_of[k] for k in chunk])
            spent.append(c)
            rows = []
            for k, v in zip(chunk, vectors, strict=True):
                cache[k] = np.asarray(v, dtype=np.float16).astype(np.float32)
                rows.append({"sha": k, "model": model, "v": _encode(v)})
            self._append(self._vectors_file, rows)
        return [cache[k] for k in keys]

    def recall(self, ctx, run, query):
        """本轮可放进提示词的以往研究（排除本次搜索自己的条目，本次的历史已在提示词里）。"""
        rec = Recall(best=[], similar=[], picked=[])
        if not ctx.usable:
            return rec
        past = [e for e in self.entries() if e.get("kind") == "round" and e["run"] != run
                and ctx.admits(e)]  # fmt: skip
        if not past:
            return rec
        rec.runs = len({e["run"] for e in past})
        rec.rounds = len(past)
        rec.factors = sum(len(e.get("factors") or []) for e in past)
        best = {}
        for e in past:
            if not ctx.comparable(e):
                continue
            for f in e.get("factors") or []:
                fit = _fit(f.get("fitness"))
                if fit is None:
                    continue
                text = f["expression"]
                if text not in best or fit > best[text][0]:
                    best[text] = (fit, text, train_only(f.get("train") or {}))
        rec.best = sorted(best.values(), key=lambda x: (-x[0], x[1]))[: self.top_k]

        order = None
        if self.embedder is not None and str(query or "").strip():
            spent = []
            try:
                vs = self._embed([entry_text(e) for e in past], spent)
                # 查询每轮都不同，不进缓存
                (q,), c = self.embedder.embed([str(query)[:_TEXT_LIMIT]])
                spent.append(c)
                vectors = [q, *vs]
                scores = [_cos(q, v) for v in vs]
                order = sorted(
                    range(len(past)), key=lambda i: (-scores[i], past[i]["id"])
                )
                rec.mode = "embedding"
            except LLMError as exc:
                rec.warning = f"向量检索失败，改按时间取最近的：{exc}"
                vectors = None
            rec.cost = float(sum(spent))
        if order is None:
            order = sorted(
                range(len(past)), key=lambda i: (-past[i]["time"], past[i]["id"])
            )
            rec.mode = "recent"
            vectors = None
        chosen = []
        for i in order:
            if len(chosen) >= self.similar_k:
                break
            if vectors is not None and any(
                _cos(vectors[1 + i], vectors[1 + j]) > self.dedup_cos for j in chosen
            ):
                continue
            chosen.append(i)
        for i in chosen:
            e = past[i]
            rec.picked.append(e["id"])
            rec.similar.append(self._line(ctx, e))
        return rec

    @staticmethod
    def _line(ctx, e):
        factors = e.get("factors") or []
        head = f"- [{e.get('action') or '?'}] {e.get('hypothesis') or '（无假设）'}"
        if not ctx.comparable(e):
            return head + f" → 评估 {len(factors)} 个（评估口径不同，不列数字）"
        scored = [(f["fitness"], f["expression"]) for f in factors
                  if _fit(f.get("fitness")) is not None]  # fmt: skip
        if not scored:
            return head + f" → 评估 {len(factors)} 个，都没有通过粗筛"
        fit, text = max(scored, key=lambda x: (x[0], x[1]))
        return head + f" → 评估 {len(factors)} 个，最好 {fit:.3f}（{text}）"


def recall_text(rec):
    """`Recall` → 提示词里「以往研究」一节的三个占位内容；没有以往条目时返回 None。"""
    if rec is None or rec.empty:
        return None
    best = (
        "\n".join(feedback_line(t, f, d) for f, t, d in rec.best)
        or "（以往同口径下还没有通过粗筛的因子）"
    )
    similar = "\n".join(rec.similar) or "（无）"
    return {"runs": rec.runs, "rounds": rec.rounds, "factors": rec.factors,
            "best": best, "similar": similar}  # fmt: skip


def default_path(root):
    return Path(root) / "llm" / "memory"
