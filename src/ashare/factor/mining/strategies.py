"""三种搜索策略，同一个接口：枚举、随机采样、简单进化。全部确定性。

策略只负责「提出候选」：`run(space, evaluate)` 里按批调用 ``evaluate(表达式列表)``，
拿回同顺序的适应度（越大越好；被粗筛掉、失败的为 −∞）。求值、去重、记账都在
`ashare.factor.mining.search` 里。随机数只来自以种子构造的 `numpy.random.Generator`
（进化按代 ``default_rng([seed, 代号])``），与线程数无关；同一空间、同一种子、同一份
数据 → 同一批候选。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .space import CATALOG, Invalid, canonical, dedup_key, replace, subtrees

# ---- 生成 -------------------------------------------------------------------------


def _unit(space, expr):
    """子表达式的量纲（检查深度、节点、冗余，但不要求无量纲）。"""
    return space._check(expr)[0]


def grow(space, rng, depth, want=None, leaf_probability=0.3, tries=30):
    """随机长一棵深度不超过 depth 的子树；want 为要求的量纲（`Unit`）。失败抛 `Invalid`。"""
    fields = list(space.fields)
    ops = list(space.grids)
    for _ in range(tries):
        try:
            if depth == 0 or rng.random() < leaf_probability:
                choices = [f for f in fields if want is None or space.fields[f] == want]
                if not choices:
                    if depth == 0:
                        raise Invalid("没有该量纲的字段")
                    continue
                return space.leaf(choices[int(rng.integers(len(choices)))])
            op = ops[int(rng.integers(len(ops)))]
            grid = space.grids[op]
            params = grid[int(rng.integers(len(grid)))]
            rule = CATALOG[op]
            if rule.arity == 1:
                args = [grow(space, rng, depth - 1, None, leaf_probability, tries)]
            else:
                a = grow(space, rng, depth - 1, None, leaf_probability, tries)
                same = rule.unit.__name__ == "_same"
                b = grow(
                    space, rng, depth - 1, _unit(space, a) if same else None,
                    leaf_probability, tries,
                )  # fmt: skip
                args = [a, b]
            node = space.build(op, args, params)
            unit = _unit(space, node)
            if want is not None and unit != want:
                continue
            return node
        except Invalid:
            continue
    raise Invalid("生成失败")


def random_candidate(space, rng, tries=200):
    """一个满足空间全部约束（含根无量纲）的随机表达式。"""
    for _ in range(tries):
        try:
            e = canonical(grow(space, rng, space.max_depth))
        except Invalid:
            continue
        if e.op == "field" and space.grids and rng.random() < 0.9:
            continue  # 根上偏向算子：裸字段只占很小一部分
        if space.valid(e):
            return e
    raise Invalid("在给定空间里随机生成不出合法表达式，检查空间是否过窄")


def enumerate_space(space):
    """按深度逐层穷举（确定顺序）：字段 → 每个算子（按名）× 参数组合 × 操作数组合。
    给出满足全部约束、去重键不重复的表达式（生成器，调用方按需截断）。"""
    pool = {0: [space.leaf(f) for f in space.fields]}
    seen = set()
    for e in pool[0]:
        if space.valid(e) and dedup_key(e) not in seen:
            seen.add(dedup_key(e))
            yield e
    known = {e.digest for e in pool[0]}
    for d in range(1, space.max_depth + 1):
        below = [e for k in range(d) for e in pool[k]]
        top = pool[d - 1]
        top_ids = {e.digest for e in top}
        pool[d] = []
        for op, grid in space.grids.items():
            rule = CATALOG[op]
            for params in grid:
                if rule.arity == 1:
                    combos = ((a,) for a in top)
                else:
                    combos = (
                        (a, b)
                        for i, a in enumerate(below)
                        for j, b in enumerate(below)
                        if (a.digest in top_ids or b.digest in top_ids)
                        and (not rule.commutative or i < j)
                    )
                for args in combos:
                    try:
                        node = space.build(op, args, params)
                        _unit(space, node)
                    except (Invalid, ValueError, TypeError):
                        continue
                    if node.digest in known:
                        continue
                    known.add(node.digest)
                    pool[d].append(node)
                    key = dedup_key(node)
                    if key not in seen and space.valid(node):
                        seen.add(key)
                        yield node


def mutate(space, rng, expr, tries=30):
    """变异：换参数、换子树（同量纲重新长）、或提升一个子树为根。失败返回 None。"""
    for _ in range(tries):
        nodes = subtrees(expr)
        path, node = nodes[int(rng.integers(len(nodes)))]
        kind = rng.random()
        try:
            if kind < 0.35 and node.op in space.grids and len(space.grids[node.op]) > 1:
                grid = space.grids[node.op]
                params = grid[int(rng.integers(len(grid)))]
                new = replace(expr, path, space.build(node.op, node.args, params))
            elif kind < 0.85:
                budget = space.max_depth - len(path)
                sub = grow(
                    space, rng, max(budget, 0), _unit(space, node) if path else None
                )
                new = replace(expr, path, sub)
            else:
                if not path and not node.args:
                    continue
                pick = subtrees(node)[1:] or [((), node)]
                new = pick[int(rng.integers(len(pick)))][1]
            new = canonical(new)
        except Invalid:
            continue
        if space.valid(new) and new.digest != expr.digest:
            return new
    return None


def crossover(space, rng, a, b, tries=30):
    """交叉：a 的一棵子树换成 b 里同量纲的一棵子树。失败返回 None。"""
    nodes_b = subtrees(b)
    for _ in range(tries):
        nodes_a = subtrees(a)
        path, node = nodes_a[int(rng.integers(len(nodes_a)))]
        try:
            unit = _unit(space, node)
        except Invalid:
            continue
        options = []
        for _, sub in nodes_b:
            try:
                if _unit(space, sub) == unit:
                    options.append(sub)
            except Invalid:
                continue
        if not options:
            continue
        child = canonical(replace(a, path, options[int(rng.integers(len(options)))]))
        if space.valid(child) and child.digest not in (a.digest, b.digest):
            return child
    return None


# ---- 策略 -------------------------------------------------------------------------


def _fit(x):
    return x if isinstance(x, (int, float)) and math.isfinite(x) else -math.inf


@dataclass(frozen=True)
class Enumerate:
    """小空间穷举：按 `enumerate_space` 的顺序，最多 limit 个，每批 batch 个。"""

    limit: int = 1000
    batch: int = 200

    def describe(self):
        return {"strategy": "enumerate", **asdict(self)}

    def run(self, space, evaluate):
        chunk = []
        for k, e in enumerate(enumerate_space(space)):
            if k >= self.limit:
                break
            chunk.append(e)
            if len(chunk) == self.batch:
                evaluate(chunk)
                chunk = []
        if chunk:
            evaluate(chunk)


@dataclass(frozen=True)
class Random:
    """随机采样 n 个（去重键互不相同），每批 batch 个；种子决定全部候选。"""

    n: int = 200
    seed: int = 0
    batch: int = 100

    def describe(self):
        return {"strategy": "random", **asdict(self)}

    def run(self, space, evaluate):
        rng = np.random.default_rng(self.seed)
        seen, chunk, attempts = set(), [], 0
        while len(seen) < self.n and attempts < 50 * self.n:
            attempts += 1
            e = random_candidate(space, rng)
            key = dedup_key(e)
            if key in seen:
                continue
            seen.add(key)
            chunk.append(e)
            if len(chunk) == self.batch:
                evaluate(chunk)
                chunk = []
        if chunk:
            evaluate(chunk)


@dataclass(frozen=True)
class Evolution:
    """简单进化：初代随机 population 个；之后每代用锦标赛选择（每次抽 tournament 个取
    最好）产生 population 个后代（概率 crossover 交叉，否则变异），下一代取「上一代 ∪
    后代」里适应度最高的 population 个（(μ+λ) 截断，同分按去重键排序）。"""

    population: int = 50
    generations: int = 5
    tournament: int = 3
    crossover: float = 0.5
    seed: int = 0

    def describe(self):
        return {"strategy": "evolution", **asdict(self)}

    def run(self, space, evaluate):
        rng = np.random.default_rng([self.seed, 0])
        pop, seen = [], set()
        attempts = 0
        while len(pop) < self.population and attempts < 50 * self.population:
            attempts += 1
            e = random_candidate(space, rng)
            if dedup_key(e) not in seen:
                seen.add(dedup_key(e))
                pop.append(e)
        fitness = dict(
            zip((dedup_key(e) for e in pop), map(_fit, evaluate(pop)), strict=True)
        )
        for g in range(1, self.generations + 1):
            rng = np.random.default_rng([self.seed, g])
            children, attempts = [], 0
            while len(children) < self.population and attempts < 50 * self.population:
                attempts += 1
                if rng.random() < self.crossover:
                    a, b = self._pick(rng, pop, fitness), self._pick(rng, pop, fitness)
                    child = crossover(space, rng, a, b)
                else:
                    child = mutate(space, rng, self._pick(rng, pop, fitness))
                if child is None or dedup_key(child) in seen:
                    continue
                seen.add(dedup_key(child))
                children.append(child)
            if not children:
                break
            fitness.update(
                zip(
                    (dedup_key(e) for e in children),
                    map(_fit, evaluate(children)),
                    strict=True,
                )
            )
            union = pop + children
            union.sort(key=lambda e: (-fitness[dedup_key(e)], dedup_key(e)))
            pop = union[: self.population]

    def _pick(self, rng, pop, fitness):
        idx = rng.integers(len(pop), size=min(self.tournament, len(pop)))
        best = min(idx, key=lambda i: (-fitness[dedup_key(pop[i])], int(i)))
        return pop[int(best)]
