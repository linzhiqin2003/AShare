"""去重键审计：键相同的写法数值上是否真的一样；键不同的候选里有多少排序完全相同。

在默认空间里随机生成候选（不按键去重），在真实面板的一段（只读）上求值，比较：

1. 键相同、写法不同的成对候选：逐日截面秩相关的最小值、排名不同的格子比例。键相同就
   应该排序相同（秩相关为 1）；
2. 键不同的候选里，与另一个键的排序完全相同（指纹相关 ≥ 0.9999）的比例——这是去重键
   漏掉的等价写法。新旧两种键（旧键：规范形去掉根上的单调变换；新键：再加上 ts_corr
   对称、非负量上的 abs 去掉）各算一遍。

只写汇总比例，不含表达式或行情数值。

    uv run --no-sync python bench/mining/dedup_audit.py --root <数据根目录> \\
        --out bench/mining/results/dedup_audit.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import panelops


def old_key(expr):
    from ashare.factor.mining.space import MONOTONE_ROOT, canonical

    node = canonical(expr)
    while node.op in MONOTONE_ROOT:
        node = node.args[0]
    return node.digest


def _load():
    out = subprocess.run(["uptime"], capture_output=True, text=True).stdout
    return out.split("load averages:")[-1].strip()


def main(argv=None):
    import ashare.data as ad
    from ashare.factor import Evaluator
    from ashare.factor.evaluate import default_universe
    from ashare.factor.mining import Space
    from ashare.factor.mining import strategies as st
    from ashare.factor.mining.search import fingerprint, fingerprint_corr
    from ashare.factor.mining.space import dedup_key

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--start", default="20230101")
    ap.add_argument("--end", default="20240630")
    ap.add_argument("--draws", type=int, default=20000)
    ap.add_argument("--distinct", type=int, default=1500)
    args = ap.parse_args(argv)
    started = time.time()
    space = Space.default()
    rng = np.random.default_rng(0)
    exprs = {}
    for _ in range(args.draws):
        e = st.random_candidate(space, rng)
        exprs.setdefault(e.digest, e)
    by_new, by_old = defaultdict(list), defaultdict(list)
    for d, e in exprs.items():
        by_new[dedup_key(e)].append(d)
        by_old[old_key(e)].append(d)
    data = ad.open(root=args.root)
    fields = sorted(
        set(space.fields) | {"is_listed", "is_st", "listing_age", "has_quote"}
    )
    panel = data.panel(fields, start=args.start, end=args.end)
    uni = default_universe(panel)
    t = len(panel.dates)
    rows = np.unique(np.linspace(80, t - 1, 32).astype(np.int64))
    ev = Evaluator()

    def scores(e):
        x = np.asarray(ev.evaluate(e, panel), dtype=np.float64)
        return np.ascontiguousarray(np.where(uni & np.isfinite(x), x, np.nan))

    # 1. 键相同、写法不同
    pairs = [(g[0], g[1]) for g in by_new.values() if len(g) >= 2]
    worst, diff_share = [], []
    for a, b in pairs[:300]:
        x, y = scores(exprs[a]), scores(exprs[b])
        ic = panelops.rank_ic(x, y)[80:]
        worst.append(float(np.nanmin(ic)) if np.isfinite(ic).any() else float("nan"))
        rx = panelops.cs_rank(x[rows], normalize=True)
        ry = panelops.cs_rank(y[rows], normalize=True)
        ok = np.isfinite(rx) | np.isfinite(ry)
        diff_share.append(
            float((~np.isclose(rx, ry, equal_nan=True) & ok).sum() / max(ok.sum(), 1))
        )
    merged = {
        "groups_new": sum(1 for g in by_new.values() if len(g) >= 2),
        "groups_old": sum(1 for g in by_old.values() if len(g) >= 2),
        "pairs_checked": len(worst),
        "min_rank_corr": float(np.nanmin(worst)) if worst else None,
        "pairs_rank_corr_below_0.9999": int(sum(1 for w in worst if not w >= 0.9999)),
        "max_rank_diff_share": float(max(diff_share)) if diff_share else None,
    }
    # 2. 键不同但排序相同（指纹）：每个键取一个代表，看有多少代表与另一个代表排序相同
    cache = {}

    def fp(d):
        if d not in cache:
            cache[d] = fingerprint(scores(exprs[d]), rows)
        return cache[d]

    def share(groups):
        keys = list(groups)
        pick = np.random.default_rng(1).choice(
            len(keys), size=min(args.distinct, len(keys)), replace=False
        )
        ds = [groups[keys[i]][0] for i in sorted(pick)]
        same = set()
        for i, a in enumerate(ds):
            for b in ds[i + 1 :]:
                if fingerprint_corr(fp(a), fp(b)) >= 0.9999:
                    same.update((a, b))
        return len(same) / max(len(ds), 1), len(ds)

    share_new, n_new = share(by_new)
    share_old, n_old = share(by_old)
    result = {
        "settings": {"space": "Space.default()", "draws": args.draws,
                     "distinct_checked": [n_new, n_old], "period": f"{args.start[:4]}–{args.end[:4]}",
                     "fingerprint_rows": int(len(rows))},
        "expressions": len(exprs), "keys_new": len(by_new), "keys_old": len(by_old),
        "same_key": merged,
        "identical_rankings_share_new": share_new,
        "identical_rankings_share_old": share_old,
        "load": _load(), "seconds": time.time() - started,
    }  # fmt: skip
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
