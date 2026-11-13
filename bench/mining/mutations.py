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
        "panel, _labels.Label(horizon=h, lag=self.lag, exec=exec), self.universe",
        "panel, _labels.Label(horizon=h, lag=self.lag + 1, exec=exec), self.universe",
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
        "            n_trials=total,\n            sr_var=scaled(sr_var, v_train, q),\n",
        "            n_trials=unique,\n            sr_var=scaled(sr_var, v_train, q),\n",
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
        'out["p"] = p_from_t(t, df=df)',
        'out["p"] = p_from_t(t)',
        "IC 的 p 值用正态近似（小样本尾部太薄）",
    ),
    (
        "ic_t_ignores_overlap",
        "stats.py",
        "return float(v.mean() / math.sqrt(s2 * h / n)), n // h - 1, n",
        "return float(v.mean() / math.sqrt(s2 / n)), n // h - 1, n",
        "重叠持有期的 IC t 值不放大方差（按独立样本算）",
    ),
    (
        "ic_df_not_blocks",
        "stats.py",
        "return float(v.mean() / math.sqrt(s2 * h / n)), n // h - 1, n",
        "return float(v.mean() / math.sqrt(s2 * h / n)), n - 1, n",
        "重叠持有期的 IC 自由度按天数而不是不重叠的块数",
    ),
    (
        "screen_not_purged",
        "evaluate.py",
        "s = ic_summary(ic[a : max(a, b - self.span[sh])], sh)",
        "s = ic_summary(ic[a:b], sh)",
        "粗筛的段内 IC 用了标签跨进下一段的信号日",
    ),
    (
        "bench_same_day_universe",
        "evaluate.py",
        "scores = np.ascontiguousarray(np.where(self.universe, 1.0, np.nan))",
        "scores = np.ascontiguousarray(np.where(np.roll(self.universe, -self.lag, axis=0), 1.0, np.nan))",
        "基准用成交日的股票池（前视一天）",
    ),
    (
        "bench_close_default",
        "evaluate.py",
        "        benchmark=None,\n        existing=None,",
        '        benchmark="close",\n        existing=None,',
        "默认基准退回收盘到收盘的旧口径",
    ),
    (
        "capital_small_default",
        "evaluate.py",
        "CAPITAL = 1e10\n",
        "CAPITAL = 1e8\n",
        "默认账户规模退回 1 亿元（最低佣金让持有全体的基准多扣成本）",
    ),
    (
        "grid_not_aligned",
        "evaluate.py",
        "return days[days >= int(rows[0])]",
        "return np.arange(int(rows[0]), self.shape[0], self.rebalance, dtype=np.int64)",
        "因子调仓日不对齐固定日表（从第一个有值的行起每 rebalance 行）",
    ),
    (
        "capital_overrides_config",
        "evaluate.py",
        "config = config or market.config(initial_cash=self.capital)",
        "config = market.config(initial_cash=self.capital)",
        "传 BacktestMarket 时忽略用户的 config=",
    ),
    (
        "q_no_floor",
        "stats.py",
        "return max(q, 1.0) if floor else q",
        "return q",
        "方差比去掉「不低于 1」的下限（负自相关放松检验）",
    ),
    (
        "old_bench_same_day_universe",
        "evaluate.py",
        "ok = np.asarray(universe, dtype=bool)[:-1] & np.isfinite(ret)",
        "ok = np.asarray(universe, dtype=bool)[1:] & np.isfinite(ret)",
        '旧基准（benchmark="close"）用当天的股票池',
    ),
    (
        "long_start_ignored",
        "evaluate.py",
        'first = a if long is None else max(a, int(long.get("start", 0)))',
        "first = a",
        "多头端指标不从第一次成交那天算起",
    ),
    (
        "exec_not_checked",
        "evaluate.py",
        "            if not np.array_equal(want, got, equal_nan=True):\n",
        "            if False:\n",
        "传入行情的成交价与 exec 不符时不报错",
    ),
    (
        "evaluate_needs_ml",
        "evaluate.py",
        "        from .. import labels as _labels\n",
        "        from ..ml import labels as _labels\n",
        "评估依赖 ashare.ml（没有 ml 依赖组就用不了）",
    ),
    (
        "trial_id_no_definition",
        "mining/ledger.py",
        'return digest(["factor-trial", 1, key, str(data_version), definition])',
        'return digest(["factor-trial", 1, key, str(data_version)])',
        "试验编号不含评估定义（换了成交价、基准、费用仍复用旧结果）",
    ),
    (
        "reuse_ignores_threshold",
        "mining/search.py",
        '        if not passes(screen["train"]):\n            return screen, "screen", screen\n',
        '        if rec["stage"] == "screen" and not passes(screen["train"]):\n'
        '            return screen, "screen", screen\n',
        "复用台账时不按这次的门槛重判旧回测",
    ),
    (
        "holm_not_padded",
        "mining/search.py",
        "    padded = p + [math.nan] * (total - len(p))\n",
        "    padded = p\n",
        "Holm 的分母不补到台账累计试验数",
    ),
    (
        "sharpe_ignores_autocorr",
        "mining/search.py",
        '        q = lo.get("var_ratio") if autocorr else None\n',
        "        q = None\n",
        "折算夏普不按序列相关修正标准误",
    ),
    (
        "evolution_same_stream",
        "mining/strategies.py",
        "            rng = np.random.default_rng([self.seed, g])\n",
        "            rng = np.random.default_rng([self.seed, 1])\n",
        "进化的每一代用同一个随机流",
    ),
    (
        "level_ranks_allowed",
        "mining/space.py",
        "            and not self.allow_level_ranks\n",
        "            and False\n",
        "截面排名可以作用在价格、成交量水平上",
    ),
    (
        "ts_corr_key_ordered",
        "mining/space.py",
        "        if (rule is not None and rule.commutative) or node.op in SYMMETRIC_KEY:\n",
        "        if rule is not None and rule.commutative:\n",
        "去重键不把 ts_corr 当对称",
    ),
    (
        "abs_kept_on_nonneg",
        "mining/space.py",
        '        if node.op == "abs" and nonneg[node.args[0].digest]:\n',
        "        if False:\n",
        "去重键不去掉非负量上的 abs",
    ),
    (
        "ledger_torn_line",
        "../ledger.py",
        '                if f.read(1) != b"\\n":\n',
        "                if False:\n",
        "台账末尾是半行时直接接着写（两条记录粘成一行）",
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
        "    denom = math.sqrt(q * max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr * sr))\n",
        "    denom = math.sqrt(q * max(1e-12, 1 + skew * sr + (kurt - 1) / 4 * sr * sr))\n",
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


def _merge(path, results):
    """把这次跑过的条目并进已有结果：同名的替换，其余保留（--only 只重跑一部分时不丢留档），
    按 MUTATIONS 的顺序排，清单里已删掉的条目不再保留。"""
    old = {}
    if path.exists():
        try:
            old = {r["name"]: r for r in json.loads(path.read_text())}
        except (ValueError, KeyError, TypeError):
            old = {}
    old.update({r["name"]: r for r in results})
    return [old[m[0]] for m in MUTATIONS if m[0] in old]


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
    merged = _merge(out, results)
    out.write_text(json.dumps(merged, ensure_ascii=False, indent=1))
    missed = [r["name"] for r in results if not r["caught"]]
    print("没抓到：", missed or "无")
    sys.exit(1 if missed else 0)


if __name__ == "__main__":
    main()
