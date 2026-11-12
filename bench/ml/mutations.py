"""变异检查：故意改坏 ashare.ml 的一处（重点是时间对齐），跑 test/ml，必须有测试失败。

每个变异只改一处、改完立即恢复（finally）。先确认测试导入的是本工作区的源码。
用法：.venv/bin/python bench/ml/mutations.py [--only 名字,...]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "ashare" / "ml"

MUTATIONS = [
    (
        "label_shift_one_day",
        "labels.py",
        "raw[:m] = adjusted[hold : hold + m] / buy - 1",
        "raw[:m] = adjusted[hold - 1 : hold - 1 + m] / buy - 1",
        "标签少移一天（卖出日提前一天）",
    ),
    (
        "label_buy_one_day_early",
        "labels.py",
        "buy = adjusted[lag : m + lag]",
        "buy = adjusted[lag - 1 : m + lag - 1]",
        "买入价早取一天（信号日当天成交）",
    ),
    (
        "buyable_misaligned",
        "labels.py",
        "buyable[:m] = can_buy[lag : m + lag]",
        "buyable[:m] = can_buy[:m]",
        "可买性取信号日而不是买入日",
    ),
    (
        "blocked_not_excluded",
        "labels.py",
        "            ok &= self.buyable\n",
        "            pass\n",
        "买不进的样本也进训练集",
    ),
    (
        "gap_minus_one",
        "split.py",
        "            v_end = test[0] - gap\n",
        "            v_end = test[0] - gap + 1\n",
        "验证段与预测段的间隔少一天",
    ),
    (
        "valid_overlaps_train",
        "split.py",
        "            train_end = valid[0] - gap\n",
        "            train_end = valid[1]\n",
        "训练集与验证集重叠",
    ),
    (
        "check_off_by_one",
        "walk.py",
        "if last is not None and last + span >= first:",
        "if last is not None and last + span > first:",
        "防前视检查放过等号",
    ),
    (
        "shared_bins_use_all_data",
        "walk.py",
        "        ref_idx = sl.start + np.flatnonzero(trainable[sl])\n        shared = SharedBins",
        "        ref_idx = np.flatnonzero(trainable)\n        shared = SharedBins",
        "共享分箱边界用了全部样本（含未来）",
    ),
    (
        "shared_bins_use_last_fold",
        "walk.py",
        "        f0 = sched.folds[0]\n",
        "        f0 = sched.folds[-1]\n",
        "共享分箱边界用了最后一折的训练集（晚于前面各折的预测期）",
    ),
    (
        "carry_ignores_can_sell",
        "labels.py",
        "ok = np.isnan(sell) & can_sell[rows] & np.isfinite(adjusted[rows])",
        "ok = np.isnan(sell) & np.isfinite(adjusted[rows])",
        "顺延口径不看卖出日能否卖出",
    ),
    (
        "carry_span_not_widened",
        "labels.py",
        "return self.lag + self.horizon + self.carry",
        "return self.lag + self.horizon",
        "顺延时最小间隔没有加上顺延天数",
    ),
    (
        "end_day_ignores_carry",
        "labels.py",
        "            end[:m] = when\n",
        "            end[:m] = np.arange(hold, hold + m)[:, None]\n",
        "实际卖出日记成了不顺延的那天",
    ),
    (
        "lookahead_end_off_by_one",
        "walk.py",
        "if end is not None and end >= first:",
        "if end is not None and end > first:",
        "按实际卖出日的防前视检查放过等号",
    ),
    (
        "zscore_not_applied",
        "labels.py",
        "        target = panelops.cs_zscore(target, mask=universe)\n",
        "        target = target\n",
        "截面标准分目标没有生效",
    ),
    (
        "store_key_missing_preprocess",
        "store.py",
        "            preprocess.digest,\n            version,",
        "            version,",
        "特征库缓存键漏掉预处理定义",
    ),
    (
        "ledger_only_positive_ic",
        "walk.py",
        '    if book:\n        book.append(\n            {\n                "event": "trial",\n                "status": "ok",',
        '    if book and metrics["ic_mean"] > 0:\n        book.append(\n            {\n                "event": "trial",\n                "status": "ok",',
        "账本只记 Rank IC 为正的试验",
    ),
    (
        "seed_not_normalized",
        "walk.py",
        '        "seed": seed_def,\n',
        '        "seed": seed,\n',
        "试验定义记 walk_forward 的 seed 而不是模型实际用的",
    ),
    (
        "custom_identity_module_path",
        "models.py",
        '    return {"class": cls.__qualname__, "source": digest}',
        '    return {"class": f"{cls.__module__}.{cls.__qualname__}", "source": digest}',
        "自定义模型的身份带模块路径",
    ),
    (
        "versions_not_in_run_id",
        "walk.py",
        "run_id = _hash([trial_id, code, versions, threads, fold_workers])",
        "run_id = _hash([trial_id, code, threads, fold_workers])",
        "运行缓存键不含库版本",
    ),
    (
        "failure_not_ledgered",
        "walk.py",
        '        if book:\n            book.append(\n                {\n                    "event": "trial",\n                    "status": "failed",',
        '        if False:\n            book.append(\n                {\n                    "event": "trial",\n                    "status": "failed",',
        "训练失败不入账",
    ),
    (
        "no_constant_prediction_check",
        "walk.py",
        "        if len(finite) > 1 and finite.min() == finite.max():",
        "        if False:",
        "常数预测不报错",
    ),
    (
        "no_empty_fold_check",
        "walk.py",
        "        if not len(train_idx):\n",
        "        if False:\n",
        "训练集为空不报错",
    ),
    (
        "all_missing_feature_allowed",
        "store.py",
        "        if np.isfinite(values[block][universe[block]]).any():\n            return\n",
        "        if True:\n            return\n",
        "特征全缺失不报错",
    ),
    (
        "search_end_optional",
        "search.py",
        "    if end is None:\n        raise ValueError(",
        "    if False:\n        raise ValueError(",
        "超参搜索不要求留出段",
    ),
    (
        "gram_window_one_day_late",
        "walk.py",
        "        w = slice(first_day, last_day)\n",
        "        w = slice(first_day, last_day + 1)\n",
        "岭回归的按日统计量多加了训练期之后的一天",
    ),
    (
        "ridge_no_intercept",
        "models.py",
        "self.intercept_ = float(my - mx @ self.coef_)",
        "self.intercept_ = float(my)",
        "岭回归截距算错",
    ),
    (
        "early_stop_lower_better",
        "models.py",
        'return "rank_ic", mean_rank_ic(preds, y, day, col, n), True',
        'return "rank_ic", mean_rank_ic(preds, y, day, col, n), False',
        "提前停止把 Rank IC 当成越小越好",
    ),
    (
        "ledger_counts_runs_not_definitions",
        "ledger.py",
        '            ids.add(r["trial_id"])',
        '            ids.add(r["run_id"])',
        "账本按运行（含线程数）而不是按定义数试验组数",
    ),
    (
        "winsor_upper_ignored",
        "preprocess.py",
        "x, lower=self.lower, upper=self.upper, mask=mask",
        "x, lower=self.lower, upper=1.0, mask=mask",
        "去极值上界没生效",
    ),
    (
        "ic_misaligned",
        "metrics.py",
        "    ic = panelops.rank_ic(p, y)\n",
        "    ic = panelops.rank_ic(p, np.roll(y, 1, axis=1))\n",
        "逐日 Rank IC 的预测与标签错位",
    ),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()
    py = sys.executable
    probe = subprocess.run(
        [py, "-c", "import ashare.ml as m; print(m.__file__)"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert Path(probe).resolve() == (SRC / "__init__.py").resolve(), probe
    chosen = set(args.only.split(",")) if args.only else None
    baseline = subprocess.run(
        [py, "-m", "pytest", "test/ml", "-q", "-x", "-p", "no:cacheprovider"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=args.timeout,
    )
    assert baseline.returncode == 0, (
        "未改动时 test/ml 必须全过\n" + baseline.stdout[-2000:]
    )
    results = []
    for name, file, old, new, desc in MUTATIONS:
        if chosen and name not in chosen:
            continue
        path = SRC / file
        original = path.read_text()
        assert original.count(old) == 1, f"{name}: 原文出现 {original.count(old)} 次"
        try:
            path.write_text(original.replace(old, new))
            t = time.perf_counter()
            run = subprocess.run(
                [py, "-m", "pytest", "test/ml", "-q", "-x", "-p", "no:cacheprovider"],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=args.timeout,
            )
            failed = [
                line for line in run.stdout.splitlines() if line.startswith("FAILED")
            ]
            results.append(
                {
                    "name": name,
                    "desc": desc,
                    "caught": run.returncode != 0,
                    "by": failed[:2],
                    "seconds": round(time.perf_counter() - t, 1),
                }
            )
        finally:
            path.write_text(original)
        print(json.dumps(results[-1], ensure_ascii=False), flush=True)
    out = ROOT / "bench" / "ml" / "results" / "mutations.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=1))
    missed = [r["name"] for r in results if not r["caught"]]
    print("没抓到：", missed or "无")
    sys.exit(1 if missed else 0)


if __name__ == "__main__":
    main()
