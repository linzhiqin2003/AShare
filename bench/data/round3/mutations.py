"""在临时副本变异；绝不读取真实数据目录，不改工作源码。输出 JSON 可另行保存。"""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CASES = []


def case(name, module, before, after, test):
    CASES.append((name, module, before, after, test))


INC = "test_round3_increment.py"
A = "test_round3_actions.py"
S = "test_round3_storage.py"
B = "test_round3_alias_industry.py"
case(
    "N8 退市变化不回退",
    "panel_increment",
    '("list_date", "delist_date")',
    '("list_date",)',
    INC + "::test_increment_revision_branches[delist]",
)
case(
    "N8b 补录旧证券不回退",
    "panel_increment",
    'prefix = min(prefix, int(np.searchsorted(dates, row["list_date"])))',
    "prefix = prefix",
    INC + "::test_increment_revision_branches[backdated_security]",
)
case(
    "N9 月权重修订不回退",
    "panel_increment",
    'prefix = min(prefix, int(np.searchsorted(dates, month + "01")))',
    "prefix = prefix",
    INC + "::test_cross_month_missing_weight_then_increment",
)
case(
    "N9b 行业修订不回退",
    "panel_increment",
    "prefix = min(prefix, int(np.searchsorted(dates, earliest)))",
    "prefix = prefix",
    INC + "::test_increment_revision_branches[industry]",
)
case(
    "N3a 不试单条",
    "panel_actions",
    "for r in records\n        )",
    "for r in []\n        )",
    A + "::test_economic_dedup_key_and_candidate_order",
)
case(
    "N3b 任取第一条",
    "panel_actions",
    "if abs(theta / ratio - 1) <= tolerance:",
    "if source == 2 or abs(theta / ratio - 1) <= tolerance:",
    A + "::test_economic_dedup_key_and_candidate_order",
)
case(
    "N19 has_quote 恒真",
    "panel",
    'save("has_quote", np.isfinite(destination("close")[prefix:]))',
    'save("has_quote", np.ones(destination("close")[prefix:].shape, bool))',
    INC + "::test_quote_tradable_and_integer_storage",
)
case(
    "N17 恢复无信息 st_known",
    "catalog",
    'FIELDS.pop("st_known")',
    'FIELDS["st_known"] = replace(FIELDS["is_listed"], name="st_known")',
    INC + "::test_quote_tradable_and_integer_storage",
)
case(
    "N12c 同窗口重复减速",
    "sources/tushare",
    "if self.clock() - self.last_reduction >= 60:",
    "if True:",
    S + "::test_zero_month_legacy_freeze_and_repeated_rate",
)
case(
    "K2 新证券只追加末列",
    "panel",
    'old_columns = [list(symbols).index(s) for s in old["symbols"]] if prefix else []',
    'old_columns = list(range(len(old["symbols"]))) if prefix else []',
    INC + "::test_increment_revision_branches[middle_column]",
)
case(
    "R1 去重忽略现金",
    "panel_events",
    'subset = ["ts_code", "ex_date", "_stock_key", "_cash_key"]',
    'subset = ["ts_code", "ex_date", "_stock_key"]',
    A + "::test_economic_dedup_key_and_candidate_order",
)
case(
    "R2 不做百万分位取整",
    "panel_events",
    '(pl.col("stk_div").fill_null(0) * 1e6)',
    '(pl.col("stk_div").fill_null(0) * 1e10)',
    A + "::test_economic_dedup_key_and_candidate_order",
)
case(
    "R3 优先送转丢现金",
    "panel_actions",
    "for shares, money, source in candidates:",
    "for shares, money, source in candidates[2:] + candidates[:2]:",
    A + "::test_economic_dedup_key_and_candidate_order",
)
case(
    "R4 禁用因子合成", "panel_actions", "return ratio, 0.0, 3", "return 1.0, 0.0, 0", A
)
case(
    "R5 不否决重整事件",
    "panel_actions",
    "return 1.0, 0.0, 4 if records else 0",
    'return (1 + records[0]["stk_div"], records[0]["cash_div_tax"], 1) if records else (1.0, 0.0, 0)',
    A,
)
case(
    "R6 缩股截为一",
    "panel_actions",
    "return ratio, 0.0, 3",
    "return max(1.0, ratio), 0.0, 3",
    A,
)
case(
    "R7 不移停牌事件",
    "panel_actions",
    "target = min(nearby, key=lambda i: (abs(i - origin), i))",
    "target = origin",
    A + "::test_suspension_shift_and_fail_closed",
)
case(
    "R8 不拦截对账失败",
    "panel_actions",
    'if checks["failed"] and not allow_unreconciled:',
    "if False:",
    A + "::test_suspension_shift_and_fail_closed",
)
case(
    "R9 不拼旧代码",
    "aliases",
    "mapped = pl.when(pl.col(column) == old).then(pl.lit(new)).otherwise(mapped)",
    "mapped = mapped",
    B,
)
case(
    "R10 因子不对齐",
    "aliases",
    'pl.col("adj_factor") * scales.get(old, 1.0)',
    'pl.col("adj_factor")',
    B,
)
case(
    "R11 取消行业回填",
    "panel_events",
    "output[:cut] = np.where(present, previous, current[:cut])",
    "output[:cut] = previous",
    B,
)
case(
    "R12 回填冒充当时可知",
    "panel_events",
    "np.where(np.isfinite(current[:cut]), 3, 0)",
    "np.where(np.isfinite(current[:cut]), 1, 0)",
    B,
)
case(
    "R13 空月定稿",
    "sync_policy",
    'if entry.get("rows", 0) <= 0:',
    "if False:",
    S + "::test_zero_month_legacy_freeze_and_repeated_rate",
)
case(
    "R14 月初拒绝沿用",
    "panel_quality",
    "if strict:\n                    raise ValueError(message)",
    "if True:\n                    raise ValueError(message)",
    INC + "::test_cross_month_missing_weight_then_increment",
)
case(
    "R15 tradable 忽略零量",
    "panel",
    '(destination("volume")[prefix:] > 0)',
    '(destination("volume")[prefix:] >= 0)',
    INC + "::test_quote_tradable_and_integer_storage",
)
case(
    "R16 GC 无写者互斥",
    "maintenance",
    'store.root / ".raw.lock", blocking=False',
    'store.root / ".unused.lock", blocking=False',
    S + "::test_gc_does_not_remove_live_write",
)
case(
    "R17 源目录读取需要写权限",
    "reader",
    "Store(root, market, readonly=True)",
    "Store(root, market)",
    S + "::test_read_only_and_expired_reader",
)
case(
    "R18 测试不拦真实根",
    "store",
    'if "pytest" in sys.modules:',
    "if False:",
    S + "::test_read_only_and_expired_reader",
)

case(
    "R19 事件缓存忽略修订哈希",
    "panel_cache",
    "previous.get(key) == digest",
    "key in previous",
    INC + "::test_event_cache_hash_invalidation",
)

case(
    "R20 诊断写入非标准NaN",
    "panel_actions",
    "else None,",
    'else float("nan"),',
    A + "::test_missing_reference_diagnostics_are_standard_json",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="只重跑指定编号，多个以逗号分隔")
    args = parser.parse_args()
    selected = [
        c for c in CASES if not args.only or c[0].split()[0] in args.only.split(",")
    ]
    originals = {p: p.read_bytes() for p in (ROOT / "src/ashare/data").rglob("*.py")}
    with tempfile.TemporaryDirectory(prefix="ashare-mutation-") as temporary:
        clone = Path(temporary)
        shutil.copytree(
            ROOT / "src/ashare",
            clone / "src/ashare",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        shutil.copytree(
            ROOT / "test/data",
            clone / "test/data",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        env = dict(
            os.environ,
            PYTHONPATH=str(clone / "src"),
            ASHARE_DATA_DIR=str(clone / "isolated-root"),
            PYTHONDONTWRITEBYTECODE="1",
        )
        env.pop("TUSHARE_TOKEN", None)

        def run(test=""):
            output = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "-x",
                    str(clone / "test/data" / test),
                ],
                cwd=clone,
                env=env,
                text=True,
                capture_output=True,
                timeout=180,
            )
            return output.returncode, output.stdout + output.stderr

        status, log = run()
        assert status == 0, log[-5000:]
        baseline = log.strip().splitlines()[-1]
        results = []
        for label, module, before, after, test in selected:
            path = clone / "src/ashare/data" / f"{module}.py"
            original = path.read_text()
            assert before in original, (label, "找不到变异目标")
            changed = original.replace(before, after)
            compile(changed, str(path), "exec")
            try:
                path.write_text(changed)
                status, output = run(test)
                caught = (
                    status == 1
                    and "FAILED " in output
                    and "ERROR collecting" not in output
                )
                failures = [
                    line for line in output.splitlines() if line.startswith("FAILED ")
                ]
                results.append(
                    dict(
                        case=label,
                        caught=caught,
                        exit=status,
                        tests=failures,
                        summary=output.strip().splitlines()[-1],
                    )
                )
                print(json.dumps(results[-1], ensure_ascii=False), flush=True)
            finally:
                path.write_text(original)
            assert path.read_text() == original
        status, log = run()
        assert status == 0, log[-5000:]
        restored = log.strip().splitlines()[-1]
    assert all(p.read_bytes() == data for p, data in originals.items())
    result = dict(
        baseline=baseline,
        restored=restored,
        results=results,
        source_sha256=hashlib.sha256(b"".join(originals.values())).hexdigest(),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not all(r["caught"] for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
