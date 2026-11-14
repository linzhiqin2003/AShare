"""表达式搜索：切分、两段式评估、去重、记账、多重检验折算、留出段一次性解锁。

数据切分是硬性的：``split`` 必填。搜索只在 ``[0, 留出段起点)`` 这几行的面板视图上进行
（`Truncated`：取数时直接截掉后面的行），训练段算适应度，验证段只用来最后挑选；
留出段（默认最近两年）在搜索和挑选的任何代码路径上都读不到，只能事后用
`SearchResult.unlock_holdout` 显式解锁一次，解锁记进台账。

两段式：先在训练段算 Rank IC（便宜），训练段 IC 的 t 值不低于 ``screen_t`` 的才跑多头端
回测；粗筛掉的同样记一次试验。``screen="gpu"`` 时先在 GPU 上按 float32 整代粗筛，只把可能
过线的送 CPU 按 float64 复算（`ashare.factor.mining.gpu`），结果与 CPU 粗筛相同。
去重：去重键（规范形去掉根上的单调变换）相同的只评一次；数值上与已入选因子（过了粗筛、
不重复的）截面秩相关不低于 ``dedup_threshold`` 的标为重复，不计入「发现」，但照样是一次试验。

```python
from ashare.factor.mining import Random, Split, mine
from ashare.factor.mining.synthetic import market
panel = market(t=900, n=150, strength=0.003)
split = Split.recent(panel.dates, holdout_years=1, valid_years=1)
result = mine(panel, split=split, strategy=Random(n=60, seed=1), root="/tmp/mining-demo")
print(result.text())
```
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import stats
from ..engine import Evaluator
from ..evaluate import FactorEvaluator
from .ledger import code_version, default_root, digest, open_ledger, trial_id
from .space import Space, dedup_key, size
from .strategies import Random

FITNESS = ("information_ratio", "excess_return", "ic", "ic_t")


# ---- 切分 -------------------------------------------------------------------------


def _shift_years(date, years):
    y, md = int(date[:4]) - years, date[4:]
    if md == "0229":
        md = "0228"
    return f"{y:04d}{md}"


@dataclass(frozen=True)
class Split:
    """按日期切成训练、验证、留出三段（YYYYMMDD 字符串，均为该段第一天）。

    train_start 缺省为面板第一天。留出段 = [holdout_start, 面板最后一天]，搜索和挑选都
    碰不到。`Split.recent` 按「最近 holdout_years 年留出、再往前 valid_years 年验证」给出。"""

    valid_start: str
    holdout_start: str
    train_start: str | None = None

    def __post_init__(self):
        for name in ("valid_start", "holdout_start", "train_start"):
            v = getattr(self, name)
            if v is not None and (len(str(v)) != 8 or not str(v).isdigit()):
                raise ValueError(f"{name}: 需要 YYYYMMDD")
        if self.holdout_start <= self.valid_start:
            raise ValueError("holdout_start 必须晚于 valid_start")
        if self.train_start is not None and self.valid_start <= self.train_start:
            raise ValueError("valid_start 必须晚于 train_start")

    @classmethod
    def recent(cls, dates, holdout_years=2, valid_years=2, train_start=None):
        """最近 holdout_years 年作留出段，之前 valid_years 年作验证段，其余为训练段。"""
        dates = np.asarray(dates).astype(str)
        last = dates[-1]
        holdout = _shift_years(last, holdout_years)
        holdout = str(dates[np.searchsorted(dates, holdout, side="right")])
        valid = _shift_years(holdout, valid_years)
        valid = str(dates[np.searchsorted(dates, valid, side="left")])
        return cls(valid, holdout, train_start)

    @classmethod
    def standard(cls):
        """项目统一的切分（`ashare.factor.windows`）：训练 = 样本内 2016-01-01 起，
        验证 = 2025 年，留出 = 2026 年起。与分组回测的窗口是同一套日期。"""
        from ..windows import IS, TEST, VALIDATION

        return cls(VALIDATION[0], TEST[0], IS[0])

    def rows(self, dates):
        """(训练起始行, 验证起始行, 留出起始行)。"""
        dates = np.asarray(dates).astype(str)
        a = (
            0
            if self.train_start is None
            else int(np.searchsorted(dates, self.train_start))
        )
        v = int(np.searchsorted(dates, self.valid_start))
        h = int(np.searchsorted(dates, self.holdout_start))
        if not (a < v < h <= len(dates)):
            raise ValueError("切分超出面板日期范围，或某一段为空")
        if h == len(dates):
            raise ValueError("留出段为空：holdout_start 晚于面板最后一天")
        return a, v, h

    def describe(self):
        return {"train_start": self.train_start, "valid_start": self.valid_start,
                "holdout_start": self.holdout_start}  # fmt: skip


class Truncated:
    """面板的前 rows 行视图：取数时就截掉，后面的行在任何代码路径上都读不到。"""

    def __init__(self, source, rows):
        self._source, self.rows = source, int(rows)
        self.dates = np.asarray(source.dates)[: self.rows]
        self.symbols = np.asarray(source.symbols)
        self.version = f"{source.version}:rows<{self.dates[-1]}"

    def __getitem__(self, name):
        value = self._source[name]
        return np.asarray(value)[: self.rows]


# ---- 指纹（数值去重） -------------------------------------------------------------


def fingerprint(scores, rows):
    """因子在若干行上的截面排名（居中、缺失记 0、每行单位长度），float16。

    两个指纹逐行内积的平均 ≈ 平均日截面秩相关（缺失按中位数处理的近似）。"""
    import panelops

    sub = np.ascontiguousarray(np.asarray(scores, dtype=np.float64)[rows])
    r = panelops.cs_rank(sub, normalize=True)
    r = np.where(np.isfinite(r), r - 0.5, 0.0)
    norm = np.sqrt((r * r).sum(axis=1, keepdims=True))
    r = np.where(norm > 0, r / np.where(norm > 0, norm, 1), 0.0)
    return r.astype(np.float16)


def fingerprint_corr(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float((a * b).sum(axis=1).mean())


# ---- 搜索 -------------------------------------------------------------------------


def _fitness(spec, train, complexity, nodes):
    if callable(spec):
        value = spec(train)
    elif spec == "information_ratio":
        value = (train.get("long") or {}).get("information_ratio")
    elif spec == "excess_return":
        value = (train.get("long") or {}).get("excess_return")
    elif spec == "ic":
        value = train["ic"]["mean"]
    elif spec == "ic_t":
        value = train["ic"]["t"]
    else:
        raise ValueError(f"fitness: 只能是 {FITNESS} 或函数")
    if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
        return -math.inf
    return float(value) - complexity * nodes


def _compact(report, sh, long):
    """台账与结果表里每段只留筛选、折算要用的指标。"""
    out = {}
    for seg, m in report.to_dict()["segments"].items():
        out[seg] = {
            "start": m["start"],
            "end": m["end"],
            "ic": m["ic"][str(sh)],
            "ic_all": {h: {"mean": s["mean"], "t": s["t"]} for h, s in m["ic"].items()},
            "monotonicity": m["groups"]["monotonicity"],
            "long": m.get("long") if long else None,
        }
    return out


def _train_feedback(row):
    """一个候选在本次搜索里的训练段反馈（给策略看）：阶段、训练段 IC 均值与 t 值、训练段
    多头端信息比率与年化超额、是否与已入选因子数值重复。只读 ``row["train"]``，不碰验证段。"""
    if row is None:
        return None
    train = row.get("train") or {}
    ic = train.get("ic") or {}
    long = train.get("long") or {}
    return {
        "stage": row["stage"],
        "ic_mean": _num(ic.get("mean")),
        "ic_t": _num(ic.get("t")),
        "long_ir": _num(long.get("information_ratio")),
        "long_excess": _num(long.get("excess_return")),
        "duplicate": row["duplicate_of"] is not None,
    }


def _num(x):
    return (
        x
        if isinstance(x, (int, float)) and x is not None and math.isfinite(x)
        else None
    )


@dataclass
class SearchResult:
    """一次搜索的结果。candidates：每个不同去重键一行（按提出顺序）；summary：汇总。"""

    candidates: list
    summary: dict
    selected: list
    context: dict = field(repr=False, default_factory=dict)

    def to_dict(self):
        return {"summary": self.summary, "selected": self.selected,
                "candidates": self.candidates}  # fmt: skip

    def text(self, top=10, show_expressions=True):
        """文本摘要。show_expressions=False 时不打印任何具体表达式（只给统计）。"""
        from ...report.fmt import num, pct

        s = self.summary
        lines = [
            f"搜索 {s['search_id'][:12]}  策略 {s['strategy']['strategy']}  空间 {s['space'][:12]}",
            f"切分 训练 {s['split_dates']['train']}  验证 {s['split_dates']['valid']}  "
            f"留出 {s['split_dates']['holdout']}（{'已解锁' if s.get('holdout_unlocked') else '未解锁'}）",
            f"提出 {s['proposed']}  去重键不同 {s['unique']}  粗筛掉 {s['screened_out']}  "
            f"回测 {s['backtested']}  数值重复 {s['duplicates']}  失败 {s['failed']}  "
            f"账本复用 {s['cache_hits']}",
            f"台账累计试验 {s['trials_total']}  本次回测候选的有效试验数约 "
            f"{num(s['effective_trials'], 1)}（{s['effective_method']}）  "
            f"PBO {num(s['pbo'], 2) if s['pbo'] is not None else s.get('pbo_note') or '不适用'}",
            f"显著（α = {s['alpha']}）  训练段 IC 未校正 {s['significant']['ic_raw']}  "
            f"Holm {s['significant']['ic_holm']}  BH {s['significant']['ic_bh']}  "
            f"BY {s['significant']['ic_by']}；"
            f"多头端 PSR {s['significant']['long_psr']}  折算夏普 {s['significant']['long_dsr']}"
            f"（V 取零假设 {s['significant']['long_dsr_null']}）；"
            f"验证段再通过 {s['significant']['valid_dsr']}",
            f"耗时 {num(s['seconds'], 1)} 秒  每秒评估 {num(s['per_second'], 1)} 个",
        ]
        if s.get("gpu"):
            g = s["gpu"]
            lines.append(
                f"GPU 粗筛（{g['device']}）{g['screened']} 个：送 CPU 复算 {g['rechecked']}、"
                f"判为不过线 {g['dropped']}（门槛下放 {g['margin']}）"
            )
        if show_expressions and self.selected:
            lines.append("验证段排名前列（训练段折算夏普 / 验证段折算夏普）：")
            for row in self.selected[:top]:
                tv = row["valid"]["long"] or {}
                lines.append(
                    f"  IR 验证 {num(tv.get('information_ratio'), 2)}  超额 "
                    f"{pct(tv.get('excess_return'))}  DSR {num(row['dsr_train'], 3)} / "
                    f"{num(row['dsr_valid'], 3)}  {row['expr']}"
                )
        return "\n".join(lines)

    def __str__(self):
        return self.text()

    def unlock_holdout(self, panel, *, ledger=None, root=None, top=None):
        """显式解锁留出段：在完整面板上评估入选的候选，只报留出段指标。一次搜索只能
        解锁一次（以台账为准），解锁本身记进台账。panel 必须是搜索时的同一份面板。

        这是纪律，不是技术上的锁：换一个台账或数据根目录、删掉台账里的解锁记录，就能
        再解锁一次。它防的是无意中反复看留出段，防不了有意绕过；留出段只能看一次要靠
        使用者自己遵守，台账只负责留下看过的记录。"""
        ctx = self.context
        if str(panel.version) != ctx["source_version"]:
            raise ValueError("面板版本与搜索时不同")
        led = open_ledger(ledger, root if root is not None else ctx["root"])
        for r in led.records():
            if (
                r.get("event") == "holdout_unlock"
                and r.get("search_id") == self.summary["search_id"]
            ):
                raise RuntimeError("这次搜索的留出段已经解锁过；不能再看一次")
        rows = self.selected[: top or len(self.selected)]
        a, v, h = ctx["rows"]
        t = len(panel.dates)
        options = dict(ctx["options"])
        if options.get("universe") is not None:
            options["universe"] = ctx["universe_full"]
        if "benchmark_full" in ctx:
            options["benchmark"] = ctx["benchmark_full"]
        if options.get("market") is not None:
            options["market"] = ctx["market_full"]
        fe = FactorEvaluator(panel, segments={"holdout": (h, t)}, **options)
        exprs = {row["key"]: ctx["exprs"][row["key"]] for row in rows}
        values = Evaluator().evaluate_many(exprs, panel)
        reports = fe.evaluate_many(values)
        out = []
        for row in rows:
            rep = reports[row["key"]]
            m = _compact(rep, fe.screen_horizon, True)["holdout"]
            lo = m["long"] or {}
            sr = lo.get("sr_period")
            q = (lo.get("var_ratio") or 1.0) if ctx.get("autocorr", True) else 1.0
            psr = (
                stats.deflated_sharpe(
                    sr,
                    lo["days"],
                    lo["skew"] or 0.0,
                    lo["kurt"] or 3.0,
                    n_trials=len(rows),
                    sr_var=q * stats.trial_sharpe_variance([], t=lo["days"]),
                    q=q,
                )  # fmt: skip
                if sr is not None
                else None
            )
            out.append(
                {
                    "key": row["key"],
                    "expr": row["expr"],
                    "holdout": m,
                    "dsr_holdout": psr,
                }
            )
        led.append(
            {
                "event": "holdout_unlock",
                "kind": "factor",
                "search_id": self.summary["search_id"],
                "keys": [row["key"] for row in rows],
                "results": out,
                "code": code_version(),
            }
        )
        self.summary["holdout_unlocked"] = True
        return out


def mine(
    panel,
    *,
    split,
    space=None,
    strategy=None,
    ledger=None,
    root=None,
    fitness="information_ratio",
    complexity=0.0,
    screen_t=1.0,
    dedup_threshold=0.9,
    alpha=0.05,
    select=10,
    existing=None,
    evaluator=None,
    memory_limit=None,
    fingerprint_rows=16,
    progress=None,
    sr_var="null",
    autocorr=True,
    screen="cpu",
    **options,
):
    """在 panel 上按 space 和 strategy 搜索因子表达式。

    - split（必填）：`Split`，留出段之后的数据在搜索中读不到；
    - space：`Space`，默认 `Space.default()`；strategy：`Enumerate`、`Random`、
      `Evolution` 之一，默认 `Random()`；
    - ledger / root：台账（路径或 `Ledger`）与指纹、收益序列的存放目录，默认
      `default_root()`（数据根目录下的 ``mining/``，不入库）；
    - fitness：训练段上的适应度，"information_ratio"（默认，多头端超额的信息比率）、
      "excess_return"、"ic"、"ic_t"，或接收训练段指标 dict 的函数；complexity：每个
      节点扣的分；
    - screen_t：粗筛门槛（训练段 IC 的 t 值，见 `ic_summary`），低于它的不回测。台账里的
      旧记录按这次的门槛重新判：现在过门槛的才复用回测结果，进入折算和挑选；现在不过的
      降回粗筛结果，照样算一次试验；
    - dedup_threshold：与已入选因子的截面秩相关不低于它的标为重复；
    - existing：{名字: Expr}，已有因子，参与去重并在结果里报相关；
    - select：按验证段信息比率挑出的个数；alpha：显著性水平；
    - evaluator：`Evaluator`（可带缓存目录）；memory_limit：批量求值的内存上限；
    - sr_var：折算夏普的试验间夏普方差 V。"null"（默认）：零假设下单个夏普估计量的方差
      1/(T−1)；"empirical"：Bailey & López de Prado 原文的做法，各试验夏普的截面方差——
      候选池里有真信号时它被真信号的离散度撑大，门槛跟着抬高，几乎什么都过不了（见
      MINING_NOTES 的模拟）；"max"：两者取大，最保守。``dsr_train`` 按这里选的算，
      ``dsr_train_null`` 总是按零假设算；
    - autocorr：True（默认）时按超额日收益的序列相关修正夏普的标准误（Lo 2002；每个候选
      乘自己的方差比 q ≥ 1，见 `stats.variance_ratio`，零假设方差相应取 q/(T−1)）；
      False 时按独立同分布算（原文做法，日收益正自相关时偏松）；
    - progress：每记一条台账后调用一次 ``progress(记录)``（测试用来模拟中途被杀）；
    - screen：粗筛在哪里算。"cpu"（默认）：现行做法，每个候选在 CPU 上按 float64 算训练段
      IC。"gpu" 或 `gpu.GpuScreen(...)`：先在 GPU 上按 float32 整代粗筛，GPU t 值不低于
      ``min(screen_t, z_{1−α}) − margin``（边际默认 `gpu.GPU_SCREEN_T_MARGIN` = 0.3）的、
      GPU 算不了的，送 CPU 按 float64 复算并照常判定；其余记一次阶段为 ``gpu_screen`` 的
      试验（不带任何指标，复用台账时重算）。进台账、统计检验和结果的数字全部来自 CPU；
      复算与下一块的 GPU 粗筛重叠进行。GPU 不可用时抛 `gpu.GpuUnavailable`，不退回 CPU；
    - 其余关键字参数交给 `FactorEvaluator`（horizons、rebalance、top、top_fraction、
      groups、universe、costs、config、benchmark、threads ...）。universe、benchmark 给
      完整长度（与 panel 同行数），这里截掉留出段。
    """
    started = time.perf_counter()
    if not isinstance(split, Split):
        raise TypeError("split 必须是 Split：先把最后一段留作最终检验")
    space = space or Space.default()
    strategy = strategy or Random()
    root = Path(root) if root is not None else default_root()
    led = open_ledger(ledger, root)
    if sr_var not in SR_VAR:
        raise ValueError(f"sr_var: 只能是 {SR_VAR}")
    if not callable(fitness) and fitness not in FITNESS:
        raise ValueError(f"fitness: 只能是 {FITNESS} 或函数")
    gpu_options = _screen_options(screen)
    a, v, h = split.rows(panel.dates)
    view = Truncated(panel, h)
    opts = dict(options)
    context_full = {}
    for name in ("universe", "benchmark"):
        if opts.get(name) is not None and not isinstance(opts[name], str):
            full = np.asarray(opts[name])
            if full.shape[0] != len(panel.dates):
                raise ValueError(f"{name}: 行数必须与面板相同")
            context_full[f"{name}_full"] = full
            opts[name] = np.ascontiguousarray(full[:h])
    if opts.get("market") is not None:
        opts["market"] = _truncate_market(opts["market"], h, opts.get("exec", "open"))
        context_full["market_full"] = options["market"]
    fe = FactorEvaluator(view, segments={"train": (a, v), "valid": (v, h)}, **opts)
    ev = evaluator or Evaluator()
    sh = fe.screen_horizon
    gpu = None
    if gpu_options is not None:
        from .gpu import Screener, gate_threshold

        screener = Screener(view, fe.universe, fe.returns[sh], rows=v,
                            memory_limit=gpu_options.memory_limit)  # fmt: skip
        gate_t = gate_threshold(screen_t, alpha, gpu_options.margin)
        gpu = {
            "screener": screener,
            "record": {"device": screener.device, "kernel": screener.kernel_id,
                       "margin": gpu_options.margin,
                       "gate_t": gate_t if math.isfinite(gate_t) else None},
            "gate_t": gate_t,
            "counts": {"rechecked": 0, "dropped": 0},
            "reasons": {},
            "recheck_seconds": 0.0,
        }  # fmt: skip
    # 评估定义：凡是改变指标数值的设置都进来（进入试验编号）。粗筛门槛不进编号（同一个
    # 表达式换门槛仍是同一个试验），而是在复用时按这次的门槛重判，见 `evaluate`。
    definition = {
        "horizons": list(fe.horizons),
        "screen_horizon": sh,
        "rebalance": fe.rebalance,
        "lag": fe.lag,
        "exec": fe.exec,
        "benchmark_mode": fe.benchmark_mode,
        "var_ratio_lags": fe.var_ratio_lags,
        "groups": fe.groups,
        "top": fe.top,
        "top_fraction": fe.top_fraction,
        "universe": digest(np.packbits(fe.universe).tobytes().hex()),
        "benchmark": digest(np.nan_to_num(fe.benchmark, nan=-9.0).tobytes().hex()),
        "costs": _obj_digest(fe.costs),
        "config": _obj_digest(fe.config),
        "split": [a, v, h],
    }
    eval_def = digest(definition)
    data_version = view.version
    fp_rows = np.unique(
        np.linspace(a + (v - a) // 3, v - 1, fingerprint_rows).astype(np.int64)
    )
    search_def = {
        "data": data_version,
        "evaluation": eval_def,
        "space": space.digest,
        "strategy": strategy.describe(),
        "fitness": getattr(fitness, "__qualname__", fitness),
        "complexity": complexity,
        "screen_t": screen_t,
        "dedup_threshold": dedup_threshold,
        "sr_var": sr_var,
        "autocorr": bool(autocorr),
        "existing": sorted((existing or {}).keys()),
        "split": split.describe(),
    }
    search_id = digest(search_def)
    fp_dir, series_dir = root / "fingerprints", root / "series"

    cache = {}
    for r in led.records():
        if (
            r.get("event") == "trial"
            and r.get("kind") == "factor"
            and r.get("status") == "ok"
        ):
            cache[r["trial_id"]] = r

    rows = {}  # 去重键 → 本次搜索的结果行
    exprs = {}
    distinct = []  # (键或已有因子名, 指纹)
    counters = {
        "proposed": 0,
        "cache_hits": 0,
        "failed": 0,
        "eval_batches": 0,
        "evaluated": 0,
    }

    if existing:
        for name, e in existing.items():
            values = ev.evaluate(e, view)
            distinct.append(
                (f"existing:{name}", fingerprint(fe._scores(values), fp_rows))
            )

    def passes(train):
        t = train["ic"]["t"]
        return t is not None and math.isfinite(t) and t >= screen_t

    def dedup(key, fp):
        best, corr = None, -math.inf
        for other, ofp in distinct:
            c = fingerprint_corr(fp, ofp)
            if c > corr:
                best, corr = other, c
        if best is not None and corr >= dedup_threshold:
            return best, corr
        distinct.append((key, fp))
        return None, corr

    def finalize(
        key,
        tid,
        metrics,
        stage,
        status,
        fp,
        error=None,
        cached=False,
        screen=None,
        gpu_note=None,
    ):
        nodes = size(exprs[key])
        fit = -math.inf
        if status == "ok" and stage == "backtest":
            fit = _fitness(fitness, metrics["train"], complexity, nodes)
        dup, corr = (None, None)
        if status == "ok" and stage == "backtest" and fp is not None:
            dup, corr = dedup(key, fp)
        row = {
            "key": key,
            "trial_id": tid,
            "expr": str(exprs[key]),
            "nodes": nodes,
            "stage": stage,
            "status": status,
            "fitness": fit if math.isfinite(fit) else None,
            "duplicate_of": dup,
            "max_corr": _num(corr) if corr is not None else None,
            "train": metrics.get("train") if metrics else None,
            "valid": metrics.get("valid") if metrics else None,
            "cached": cached,
        }
        rows[key] = row
        record = {
            "event": "cache_hit" if cached else "trial",
            "kind": "factor",
            "source": "search",
            "trial_id": tid,
            "search_id": search_id,
            "key": key,
            "expr": row["expr"],
            "nodes": nodes,
            "data_version": data_version,
            "evaluation": eval_def,
            "definition": definition,
            "space": space.digest,
            "strategy": strategy.describe(),
            "split": split.describe(),
            "stage": stage,
            "status": status,
            "metrics": metrics,
            "screen": screen,
            "duplicate_of": dup,
            "code": code_version(),
        }
        if error is not None:
            record["error"] = error
        if gpu_note is not None:
            record["gpu"] = gpu_note
        led.append(record)
        if not cached and status == "ok":
            cache[tid] = record
        if progress is not None:
            progress(record)
        return fit

    def load_fp(tid, key):
        path = fp_dir / f"{tid}.npy"
        if path.exists():
            try:
                return np.load(path, allow_pickle=False)
            except (OSError, ValueError):
                pass
        values = ev.evaluate(exprs[key], view)
        fp = fingerprint(fe._scores(values), fp_rows)
        _save(path, fp)
        return fp

    def reuse(rec):
        """台账里同一试验的记录能怎么用：(metrics, stage, screen) 或 None（要重算）。

        按这次的门槛重判粗筛：过了且有回测结果的复用回测；没过的降回粗筛结果（与在空
        台账上直接用这个门槛跑逐位相同）；过了却只有粗筛结果、或旧记录没有粗筛结果的重算。"""
        if rec is None or rec.get("screen") is None:
            return None
        screen = rec["screen"]
        if not passes(screen["train"]):
            return screen, "screen", screen
        if rec["stage"] == "backtest":
            return rec["metrics"], "backtest", screen
        return None

    def evaluate(batch):
        counters["eval_batches"] += 1
        out_keys = []
        todo = []
        for e in batch:
            counters["proposed"] += 1
            key = dedup_key(e)
            out_keys.append(key)
            if key in rows or key in exprs:
                continue
            exprs[key] = e
            tid = trial_id(key, data_version, eval_def)
            todo.append((key, tid, reuse(cache.get(tid))))
        fresh = [(key, tid) for key, tid, use in todo if use is None]
        if not fresh:
            computed = {}
        elif gpu is None:
            computed = _evaluate_batch(fresh)
        else:
            computed = _gpu_batch(fresh)
        for key, tid, use in todo:  # 按提出顺序落定：去重判定与顺序有关
            if use is not None:
                counters["cache_hits"] += 1
                metrics, stage, screen = use
                fp = load_fp(tid, key) if stage == "backtest" else None
                finalize(key, tid, metrics, stage, "ok", fp, cached=True, screen=screen)
            else:
                metrics, stage, status, fp, error, screen = computed[key]
                if status != "ok":
                    counters["failed"] += 1
                note = None
                if gpu is not None:
                    note = {**gpu["record"], "decision": "drop" if stage == "gpu_screen"
                            else "recheck", "reason": gpu["reasons"].get(key)}  # fmt: skip
                finalize(key, tid, metrics, stage, status, fp, error, screen=screen,
                         gpu_note=note)  # fmt: skip
        return [rows[k]["fitness"] if rows[k]["fitness"] is not None else -math.inf
                for k in out_keys]  # fmt: skip

    def _values(keys):
        """批量求值；某个表达式失败时退回逐个求值，失败的给出异常。"""
        done = set()
        try:
            for k, values in ev.evaluate_many(
                {k: exprs[k] for k in keys}, view, memory_limit=memory_limit
            ):
                done.add(k)
                yield k, values
            return
        except Exception:  # noqa: BLE001 —— 其余的逐个重试，定位失败的那个
            pass
        for k in keys:
            if k in done:
                continue
            try:
                yield k, ev.evaluate(exprs[k], view)
            except Exception as exc:  # noqa: BLE001
                yield k, exc

    def _gpu_batch(fresh):
        """GPU 粗筛整批，可能过线的、GPU 算不了的交给 `_evaluate_batch` 在 CPU 上复算
        （后台线程，与下一块的 GPU 粗筛重叠）；GPU 判为不过线的记成 ``gpu_screen``。"""
        from .gpu import pipeline

        tids = dict(fresh)
        gate_t = gpu["gate_t"]

        def decide(key, outcome):
            if outcome.ic is None:
                gpu["reasons"][key] = outcome.reason
                return True
            t = fe.screen_metrics(outcome.ic)["train"]["ic"]["t"]
            return t is None or not math.isfinite(t) or t >= gate_t

        def recheck(keys):
            t0 = time.perf_counter()
            out = _evaluate_batch([(k, tids[k]) for k in keys], count=False)
            gpu["recheck_seconds"] += time.perf_counter() - t0
            return out

        dropped, computed = pipeline(
            gpu["screener"],
            [(k, exprs[k]) for k, _ in fresh],
            decide,
            recheck,
            gpu_options.chunk,
        )
        counters["evaluated"] += len(fresh)
        gpu["counts"]["rechecked"] += len(computed)
        gpu["counts"]["dropped"] += len(dropped)
        for key in dropped:
            computed[key] = (None, "gpu_screen", "ok", None, None, None)
        return computed

    def _evaluate_batch(fresh, count=True):
        out, partials, targets, passed = {}, {}, [], []
        tids = dict(fresh)
        for key, values in _values([k for k, _ in fresh]):
            if count:
                counters["evaluated"] += 1
            if isinstance(values, Exception):
                out[key] = (None, "failed", "failed", None, type(values).__name__, None)
                continue
            try:
                scores, ic = fe.screen(values)
                metrics = fe.screen_metrics(ic)
                if passes(metrics["train"]):
                    _, partials[key] = fe.cheap(scores=scores, screen_ic=ic)
                    targets.append(fe.targets(scores))
                    fp = fingerprint(scores, fp_rows)
                    _save(fp_dir / f"{tids[key]}.npy", fp)
                    passed.append((key, fp, metrics))
                else:
                    out[key] = (metrics, "screen", "ok", None, None, metrics)
            except Exception as exc:  # noqa: BLE001
                out[key] = (None, "failed", "failed", None, type(exc).__name__, None)
        if passed:
            longs = fe.run_long(targets)
            for (key, fp, screen), long in zip(passed, longs, strict=True):
                rep = fe.finish(key, partials[key], long)
                metrics = _compact(rep, sh, True)
                _save(series_dir / f"{tids[key]}.npy",
                      np.ascontiguousarray(rep.series["excess"][a:v]))  # fmt: skip
                out[key] = (metrics, "backtest", "ok", fp, None, screen)
        return out

    # 给需要反馈细节的策略（`llm.LLMResearch`）的两个只读入口；只含训练段的信息。
    # 其余策略不读它们，结果与没有这两行逐位相同。
    evaluate.task = {
        "fitness": getattr(fitness, "__qualname__", fitness),
        "complexity": complexity,
        "screen_t": screen_t,
        "root": str(root),
        "split": split.describe(),  # 跨次记忆按它判断以往条目会不会越过本次的验证段
        "evaluation": eval_def,  # 跨次记忆只把评估定义相同的数字放在一起比
    }
    evaluate.train_feedback = lambda e: _train_feedback(rows.get(dedup_key(e)))
    strategy.run(space, evaluate)
    elapsed = time.perf_counter() - started
    total = led.trials()
    summary, selected = _summarize(
        rows,
        total,
        alpha,
        select,
        a,
        v,
        h,
        series_dir,
        panel,
        data_version,
        sr_var,
        autocorr,
    )
    summary.update(
        search_id=search_id,
        strategy=strategy.describe(),
        space=space.digest,
        proposed=counters["proposed"],
        cache_hits=counters["cache_hits"],
        failed=counters["failed"],
        evaluated=counters["evaluated"],
        seconds=elapsed,
        per_second=counters["evaluated"] / elapsed if elapsed > 0 else None,
        evaluator_batches=counters["eval_batches"],
        existing_correlation={
            row["key"]: {
                name[len("existing:"):]: fingerprint_corr(load_fp(row["trial_id"], row["key"]), fp)
                for name, fp in distinct if name.startswith("existing:")
            }
            for row in selected
        } if existing else {},
    )  # fmt: skip
    if gpu is not None:
        sc = gpu["screener"]
        summary["gpu"] = {
            **gpu["record"],
            "chunk": gpu_options.chunk,
            "screened": int(sc.counts["screened"]),
            "rechecked": gpu["counts"]["rechecked"],
            "dropped": gpu["counts"]["dropped"],
            "unsupported": int(sc.counts["unsupported"]),
            "float32_range": int(sc.counts["range"]),
            "errors": int(sc.counts["error"]),
            "gpu_seconds": sc.seconds["gpu"],
            "recheck_seconds": gpu["recheck_seconds"],
            "rounding_ties": int(sc.counts["degenerate"]),
            "upload_seconds": sc.seconds["upload"],
            "peak_device_memory": sc.peak_memory,
        }
    context = {
        "source_version": str(panel.version),
        "rows": (a, v, h),
        "options": opts,
        "exprs": exprs,
        "root": root,
        "autocorr": bool(autocorr),
        **context_full,
    }
    return SearchResult(list(rows.values()), summary, selected, context)


def _screen_options(screen):
    """``screen`` 参数 → None（CPU 粗筛）或 `gpu.GpuScreen`。"""
    if isinstance(screen, str):
        if screen == "cpu":
            return None
        if screen == "gpu":
            from .gpu import GpuScreen

            return GpuScreen()
    elif type(screen).__name__ == "GpuScreen":
        from .gpu import GpuScreen

        if isinstance(screen, GpuScreen):
            return screen
    raise ValueError('screen: 只能是 "cpu"、"gpu" 或 gpu.GpuScreen(...)')


def _truncate_market(market, rows, exec="open"):
    """`BacktestMarket` 截到前 rows 行（由它自带的面板按 exec 重建，留出段的行情读不到）。
    原行情的成交价与重建出的不一致（例如按别的 exec 建的）时报错。"""
    import panelbt

    from ...backtest.market import BacktestMarket, market_inputs

    if not isinstance(market, BacktestMarket):
        raise TypeError(
            "market: 需要 ashare.backtest.market_from_panel 的结果（要能截掉留出段）"
        )
    view = Truncated(market.panel, rows)
    inputs = market_inputs(view, exec)
    if not np.array_equal(
        np.asarray(inputs["exec_price"], dtype=np.float64),
        np.asarray(market.market.exec_price, dtype=np.float64)[:rows],
        equal_nan=True,
    ):
        raise ValueError(f"market: 成交价与 exec={exec!r} 不一致")
    return BacktestMarket(
        panelbt.Market(**inputs, validate=False),
        np.asarray(market.dates)[:rows],
        market.symbols,
        market.board,
        view,
    )


def _obj_digest(obj):
    """费用、账户这类数据类的内容摘要（数组按字节）。"""
    from dataclasses import fields, is_dataclass

    def enc(x):
        if is_dataclass(x):
            return [
                type(x).__name__,
                {f.name: enc(getattr(x, f.name)) for f in fields(x)},
            ]
        if isinstance(x, np.ndarray):
            return ["array", str(x.dtype), list(x.shape), digest(x.tobytes().hex())]
        if isinstance(x, (list, tuple)):
            return [enc(v) for v in x]
        if isinstance(x, float) and not math.isfinite(x):
            return repr(x)
        return x

    return digest(enc(obj))


def _save(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{np.random.default_rng().integers(1 << 30)}.tmp")
    with open(tmp, "wb") as f:
        np.save(f, array, allow_pickle=False)
    tmp.replace(path)


SR_VAR = ("max", "empirical", "null")


def _sr_var(mode, srs, t):
    """折算夏普用的试验间夏普方差 V：empirical 为各试验夏普的截面方差；null 为零假设下单个
    夏普估计量的方差 1/(T−1)（默认）；max 取两者大者（最保守）。"""
    if mode == "empirical":
        return stats.trial_sharpe_variance(srs, floor=False)
    if mode == "null":
        return 1.0 / (t - 1) if t >= 2 else math.nan
    return stats.trial_sharpe_variance(srs, t=t)


def _summarize(
    rows,
    total,
    alpha,
    select,
    a,
    v,
    h,
    series_dir,
    panel,
    data_version,
    sr_var="max",
    autocorr=True,
):
    dates = np.asarray(panel.dates).astype(str)
    table = list(rows.values())
    unique = len(table)
    back = [r for r in table if r["stage"] == "backtest" and r["status"] == "ok"]
    distinct = [r for r in back if r["duplicate_of"] is None]
    total = max(int(total), unique)

    # 训练段 IC：全部候选的单侧 p 值，补 NaN 到台账总试验数（之前的试验同样占分母）
    p = [
        (r["train"] or {}).get("ic", {}).get("p") if r["status"] == "ok" else None
        for r in table
    ]
    p = [x if isinstance(x, (int, float)) and x is not None else math.nan for x in p]
    padded = p + [math.nan] * (total - len(p))
    raw = sum(1 for x in p if math.isfinite(x) and x <= alpha)
    holm = int(stats.reject(padded, alpha, "holm").sum()) if padded else 0
    bh = int(stats.reject(padded, alpha, "bh").sum()) if padded else 0
    by = int(stats.reject(padded, alpha, "by").sum()) if padded else 0

    def long(r, seg):
        return (r[seg] or {}).get("long") or {}

    def q_of(lo):
        q = lo.get("var_ratio") if autocorr else None
        return float(q) if isinstance(q, (int, float)) and math.isfinite(q) else 1.0

    def scaled(mode, base, q):
        # 零假设方差随候选的序列相关放大（q/(T−1)）；截面方差是观测值，不再乘
        if mode == "null":
            return base * q
        if mode == "max":
            return max(base, q * base_null) if math.isfinite(base) else base
        return base

    t_train = max((long(r, "train").get("days") or 0 for r in back), default=0)
    srs = [long(r, "train").get("sr_period") for r in back]
    v_train = _sr_var(sr_var, srs, t_train)
    v_null = _sr_var("null", srs, t_train)
    base_null = v_null
    for r in table:
        r["psr_train"] = r["dsr_train"] = r["dsr_train_null"] = r["dsr_valid"] = None
        r["q_train"] = None
    for r in back:
        lo = long(r, "train")
        sr = lo.get("sr_period")
        if sr is None:
            continue
        g3, g4 = lo.get("skew") or 0.0, lo.get("kurt") or 3.0
        q = q_of(lo)
        r["q_train"] = q
        r["psr_train"] = stats.probabilistic_sharpe(sr, lo["days"], g3, g4, q=q)
        r["dsr_train"] = stats.deflated_sharpe(
            sr,
            lo["days"],
            g3,
            g4,
            n_trials=total,
            sr_var=scaled(sr_var, v_train, q),
            q=q,
        )
        r["dsr_train_null"] = stats.deflated_sharpe(
            sr, lo["days"], g3, g4, n_trials=total, sr_var=v_null * q, q=q
        )
    psr_count = sum(1 for r in distinct if (r["psr_train"] or 0) > 1 - alpha)
    dsr_count = sum(1 for r in distinct if (r["dsr_train"] or 0) > 1 - alpha)
    null_count = sum(1 for r in distinct if (r["dsr_train_null"] or 0) > 1 - alpha)

    # 验证段：在不重复的回测候选里按验证段信息比率挑选；折算的试验数 = 参与挑选的个数
    def valid_ir(r):
        x = long(r, "valid").get("information_ratio")
        return x if x is not None else -math.inf

    ranked = sorted(distinct, key=lambda r: (-valid_ir(r), r["key"]))
    t_valid = max((long(r, "valid").get("days") or 0 for r in distinct), default=0)
    v_valid = _sr_var(
        sr_var, [long(r, "valid").get("sr_period") for r in distinct], t_valid
    )
    base_null = _sr_var("null", [], t_valid)
    for r in distinct:
        lo = long(r, "valid")
        if lo.get("sr_period") is not None:
            q = q_of(lo)
            r["dsr_valid"] = stats.deflated_sharpe(
                lo["sr_period"], lo["days"], lo.get("skew") or 0.0, lo.get("kurt") or 3.0,
                n_trials=max(len(distinct), 1), sr_var=scaled(sr_var, v_valid, q), q=q,
            )  # fmt: skip
    selected = ranked[:select]
    passed = sum(
        1 for r in distinct
        if (r["dsr_train"] or 0) > 1 - alpha and (r["dsr_valid"] or 0) > 1 - alpha
    )  # fmt: skip

    # 有效试验数与 PBO：用回测候选的训练段日超额收益（存在 series/ 下，断点续跑也能取到）
    eff, pbo = None, None
    pbo_note = (
        "不适用：没有候选通过粗筛进入回测" if not back
        else "不适用：不重复的回测候选不足 2 个" if len(distinct) < 2
        else None
    )  # fmt: skip
    if len(distinct) >= 2:
        mats = []
        for r in distinct:
            path = series_dir / f"{r['trial_id']}.npy"
            if path.exists():
                mats.append(np.load(path, allow_pickle=False))
        if len(mats) >= 2 and len({len(m) for m in mats}) == 1:
            mat = np.column_stack(mats)
            mat = mat[np.isfinite(mat).all(axis=1)]
            if len(mat) >= 3:
                eff = stats.effective_trials(mat, method="li_ji")
            if len(mat) >= 40:
                pbo = stats.pbo_cscv(mat, 10)["pbo"]
            else:
                pbo_note = "不适用：训练段共同有效的天数不足 40"
        else:
            pbo_note = "不适用：候选的训练段收益序列取不到或长度不一"
    summary = {
        "data_version": data_version,
        "split_dates": {
            "train": f"{dates[a]}–{dates[v - 1]}",
            "valid": f"{dates[v]}–{dates[h - 1]}",
            "holdout": f"{dates[h]}–{dates[-1]}",
        },
        "unique": unique,
        "screened_out": sum(1 for r in table if r["stage"] in ("screen", "gpu_screen")),
        "backtested": len(back),
        "duplicates": len(back) - len(distinct),
        "distinct": len(distinct),
        "trials_total": total,
        "alpha": alpha,
        "sr_var": sr_var,
        "sr_var_train": v_train,
        "effective_trials": eff,
        "effective_method": "li_ji",
        "pbo": pbo,
        "pbo_note": pbo_note,
        "autocorr": bool(autocorr),
        "significant": {
            "ic_raw": raw,
            "ic_holm": holm,
            "ic_bh": bh,
            "ic_by": by,
            "long_psr": psr_count,
            "long_dsr": dsr_count,
            "long_dsr_null": null_count,
            "valid_dsr": passed,
        },
        "holdout_unlocked": False,
    }
    return summary, selected
