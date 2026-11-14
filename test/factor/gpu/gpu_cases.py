"""GPU 粗筛测试的公用部分：小合成面板、覆盖默认搜索空间全部算子的空间、搜索结果的逐位摘要。

摘要（`mine_digest`）覆盖一次 `mine` 的全部可观察输出：候选表、汇总、入选、台账每一行
（去掉时间戳和代码版本）、指纹和收益序列文件的字节。`screen="cpu"` 的冻结摘要
（`fixtures/mining_cpu_digest.json`）由加入 GPU 粗筛之前的代码生成，见 test_gpu_mine.py。
"""

from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from ashare.factor.mining import Evolution, Random, Space, Split, mine, open_ledger
from ashare.factor.mining.synthetic import Frictions, market

# 默认搜索空间的全部算子，外加 CATALOG 里 GPU 也支持的 sign、ts_sum、cs_zscore；窗口取小，
# 让 420 行的小面板上也有足够多的有效值。
SPACE = Space(
    fields={"close": "price", "open": "price", "high": "price", "low": "price",
            "vwap": "price", "volume": "volume", "amount": "amount",
            "turnover": "ratio"},
    ops={"add": {}, "subtract": {}, "multiply": {}, "divide": {}, "negate": {},
         "abs": {}, "log": {}, "sign": {}, "maximum": {}, "minimum": {},
         "ts_delay": {"periods": (1, 5)}, "ts_delta": {"periods": (1, 5)},
         "ts_mean": {"window": (5, 10)}, "ts_sum": {"window": (5,)},
         "ts_std": {"window": (5, 10)}, "ts_max": {"window": (5,)},
         "ts_min": {"window": (5,)}, "ts_rank": {"window": (5,)},
         "ts_zscore": {"window": (10,)}, "ts_decay_linear": {"window": (5,)},
         "ts_corr": {"window": (10,)}, "cs_rank": {}, "cs_zscore": {}},
    max_depth=3,
    max_nodes=6,
)  # fmt: skip


def small_panel(seed=3, strength=0.004):
    return market(t=420, n=60, seed=seed, strength=strength,
                  frictions=Frictions.typical(suspend=0.01))  # fmt: skip


def split_of(panel):
    return Split(valid_start=str(panel.dates[220]), holdout_start=str(panel.dates[340]))


STRATEGIES = {
    "random": lambda: Random(n=48, seed=5, batch=16),
    "evolution": lambda: Evolution(population=20, generations=2, seed=1),
}


def run(panel, root, strategy, **kw):
    kw.setdefault("screen_t", 0.5)
    return mine(panel, split=split_of(panel), space=SPACE, root=root,
                strategy=STRATEGIES[strategy](), **kw)  # fmt: skip


def _canon(x):
    """JSON 可比较形式：浮点保留全部位（repr），NaN、无穷写成字符串。"""
    if isinstance(x, dict):
        return {str(k): _canon(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_canon(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return repr(float(x)) if math.isfinite(x) else str(float(x))
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


def view(result, root, *, ledger_drop=()):
    """一次搜索可观察输出的规范形（不含耗时、时间戳、代码版本）。"""
    summary = {
        k: v for k, v in result.summary.items() if k not in ("seconds", "per_second")
    }
    records = []
    for r in open_ledger(None, root).records():
        r = {k: v for k, v in r.items() if k not in ("time", "code", *ledger_drop)}
        records.append(r)
    files = {}
    for sub in ("fingerprints", "series"):
        d = root / sub
        for p in sorted(d.glob("*.npy")) if d.exists() else []:
            files[f"{sub}/{p.name}"] = hashlib.sha256(p.read_bytes()).hexdigest()
    return _canon({"candidates": result.candidates, "summary": summary,
                   "selected": [r["key"] for r in result.selected],
                   "ledger": records, "files": files})  # fmt: skip


def mine_digest(result, root):
    blob = json.dumps(view(result, root), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()
