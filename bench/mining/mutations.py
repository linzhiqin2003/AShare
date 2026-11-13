"""变异检查：故意改坏因子评估、搜索、台账、统计检验的一处，跑相关测试，必须有测试失败。

每个变异只改一处、改完立即恢复（finally）。先确认测试导入的是本工作区的源码，并确认
未改动时这些测试全过。用法：.venv/bin/python bench/mining/mutations.py [--only 名字,...]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "ashare" / "factor"
TESTS = [
    "test/factor/test_factor_evaluate.py",
    "test/factor/test_mining_stats.py",
    "test/factor/test_mining_space.py",
    "test/factor/test_mining_search.py",
]

MUTATIONS = [
    (
        "holdout_leak",
        "mining/search.py",
        [
            ("    view = Truncated(panel, h)\n", "    view = panel\n"),
            (
                'segments={"train": (a, v), "valid": (v, h)}',
                'segments={"train": (a, v), "valid": (v, len(panel.dates))}',
            ),
        ],
        None,
        "留出段并进验证段（搜索的面板不截断、验证段延到最后一天）",
    ),
    (
        "label_one_day_late",
        "evaluate.py",
        "panel, _labels.Label(horizon=h, lag=self.lag), self.universe",
        "panel, _labels.Label(horizon=h, lag=self.lag + 1), self.universe",
        "收益标签对齐差一天（T+2 开盘买）",
    ),
    (
        "execution_lag_wrong",
        "evaluate.py",
        "return execution_targets(sparse, t=self.shape[0], lag=self.lag)",
        "return execution_targets(sparse, t=self.shape[0], lag=self.lag + 1)",
        "多头端成交晚一天",
    ),
    (
        "segment_not_purged",
        "evaluate.py",
        'h: ic_summary(partial["ic"][h][a : max(a, b - self.span[h])], h)',
        'h: ic_summary(partial["ic"][h][a:b], h)',
        "段内 IC 用了标签跨出段尾的信号日",
    ),
    (
        "costs_not_charged",
        "evaluate.py",
        "            costs = ASHARE.costs(iso_dates(self.dates))\n",
        "            costs = panelbt.Costs(commission_rate=0.0, commission_min=None,"
        " fee_rate_buy=0.0, fee_rate_sell=0.0, slippage=0.0)\n",
        "默认不扣成本",
    ),
    (
        "duplicates_not_ledgered",
        "mining/search.py",
        "        led.append(record)\n        if not cached and status",
        "        if dup is None:\n            led.append(record)\n        if not cached and status",
        "判为数值重复的候选不记进台账（不计入试验数）",
    ),
    (
        "screened_not_ledgered",
        "mining/search.py",
        "        led.append(record)\n        if not cached and status",
        '        if stage == "backtest":\n            led.append(record)\n        if not cached and status',
        "粗筛掉的候选不记进台账",
    ),
    (
        "dsr_uses_search_count",
        "mining/search.py",
        '            sr, lo["days"], g3, g4, n_trials=total, sr_var=v_train\n',
        '            sr, lo["days"], g3, g4, n_trials=unique, sr_var=v_train\n',
        "折算夏普只用本次搜索的试验数而不是台账累计",
    ),
    (
        "evolution_unseeded",
        "mining/strategies.py",
        "            rng = np.random.default_rng([self.seed, g])\n",
        "            rng = np.random.default_rng()\n",
        "进化的每一代不按种子取随机数",
    ),
    (
        "dedup_key_keeps_rank",
        "mining/space.py",
        "    while node.op in MONOTONE_ROOT:\n",
        "    while False:\n",
        "去重键不剥根上的截面排名",
    ),
    (
        "commutative_not_sorted",
        "mining/space.py",
        "        if CATALOG[op].commutative:\n            args.sort(key=lambda a: a.digest)\n",
        "        if False:\n            args.sort(key=lambda a: a.digest)\n",
        "可交换算子的操作数不排序（a+b 与 b+a 算两个）",
    ),
    (
        "units_not_checked",
        "mining/space.py",
        "    if a != b:\n        raise Invalid",
        "    if False:\n        raise Invalid",
        "加减时不检查量纲",
    ),
    (
        "unlock_twice",
        "mining/search.py",
        '                raise RuntimeError("这次搜索的留出段已经解锁过；不能再看一次")',
        "                pass",
        "留出段可以重复解锁",
    ),
    (
        "student_p_to_normal",
        "evaluate.py",
        '"p": p_from_t(t, df=df) if math.isfinite(t) else math.nan,',
        '"p": p_from_t(t) if math.isfinite(t) else math.nan,',
        "IC 的 p 值用正态近似（小样本尾部太薄）",
    ),
    (
        "hac_bartlett",
        "evaluate.py",
        't, _ = hac_t(v, h - 1, "uniform")',
        't, _ = hac_t(v, h - 1, "bartlett")',
        "IC 的 t 值用 Bartlett 核（低估重叠方差）",
    ),
    (
        "holm_multiplier",
        "stats.py",
        "(m - np.arange(m)) * sorted_p",
        "(m - np.arange(m) - 1).clip(1) * sorted_p",
        "Holm 的乘数少 1",
    ),
    (
        "bh_wrong_direction",
        "stats.py",
        "adj = np.minimum.accumulate((scale * m * sorted_p / ranks)[::-1])[::-1]",
        "adj = np.minimum.accumulate(scale * m * sorted_p / ranks)",
        "BH 累计最小值方向反了",
    ),
    (
        "psr_kurtosis_sign",
        "stats.py",
        "    denom = math.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr * sr))\n    return _N.cdf(",
        "    denom = math.sqrt(max(1e-12, 1 + skew * sr + (kurt - 1) / 4 * sr * sr))\n    return _N.cdf(",
        "PSR 分母偏度项符号反了",
    ),
    (
        "emax_no_gamma",
        "stats.py",
        "return math.sqrt(sr_var) * ((1 - EULER_GAMMA) * a + EULER_GAMMA * b)",
        "return math.sqrt(sr_var) * a",
        "期望最大夏普少了欧拉常数修正",
    ),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()
    py = sys.executable
    probe = subprocess.run(
        [py, "-c", "import ashare.factor as m; print(m.__file__)"],
        cwd=ROOT, capture_output=True, text=True, check=True,
    ).stdout.strip()  # fmt: skip
    assert Path(probe).resolve() == (SRC / "__init__.py").resolve(), probe
    cmd = [py, "-m", "pytest", *TESTS, "-q", "-x", "-p", "no:cacheprovider"]
    chosen = set(args.only.split(",")) if args.only else None
    baseline = subprocess.run(
        cmd, cwd=ROOT, capture_output=True, text=True, timeout=args.timeout
    )
    assert baseline.returncode == 0, "未改动时测试必须全过\n" + baseline.stdout[-2000:]
    results = []
    for name, file, old, new, desc in MUTATIONS:
        if chosen and name not in chosen:
            continue
        path = SRC / file
        original = path.read_text()
        pairs = old if isinstance(old, list) else [(old, new)]
        mutated = original
        for a, b in pairs:
            assert mutated.count(a) == 1, f"{name}: 原文出现 {mutated.count(a)} 次"
            mutated = mutated.replace(a, b)
        try:
            path.write_text(mutated)
            t = time.perf_counter()
            run = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                                 timeout=args.timeout)  # fmt: skip
            failed = [
                x for x in run.stdout.splitlines() if x.startswith(("FAILED", "ERROR"))
            ]
            results.append({"name": name, "desc": desc, "caught": run.returncode != 0,
                            "by": failed[:2], "seconds": round(time.perf_counter() - t, 1)})  # fmt: skip
        finally:
            path.write_text(original)
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    out = ROOT / "bench" / "mining" / "results" / "mutations.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not chosen:
        out.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    missed = [r["name"] for r in results if not r["caught"]]
    print("没抓到：", missed or "无")
    sys.exit(1 if missed else 0)


if __name__ == "__main__":
    main()
