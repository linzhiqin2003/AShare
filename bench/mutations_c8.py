"""C8 定点变异：先跑变异0、证明源码路径，再在隔离副本逐个运行断言测试。"""

import argparse
import hashlib
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = "src/ashare/data/crypto/metrics.py"
TIMING = "src/ashare/factor/timing/"
TEST_M = "test/crypto/test_metrics_c8.py"
TEST_T = "test/crypto/test_timing_c8.py"
TEST_R = "test/crypto/test_c8_review.py"
TEST_S = "test/crypto/test_c8_recheck.py"
MUTATIONS = [
    (
        "daily_median",
        METRICS,
        "np.median(paired, axis=0)",
        "np.mean(paired, axis=0)",
        TEST_M + "::test_daily_median_uses_paired_rows_and_keeps_outliers",
    ),
    (
        "paired_rows",
        METRICS,
        "np.isfinite(error[a:b]).all(axis=1)",
        "np.isfinite(error[a:b]).any(axis=1)",
        TEST_M + "::test_daily_median_uses_paired_rows_and_keeps_outliers",
    ),
    (
        "alignment_decision",
        METRICS,
        "medians[0] < match and medians[1] > reject",
        "medians[0] > reject and medians[1] < match",
        TEST_M + "::test_alignment_registered_day_boundary",
    ),
    (
        "median_match_threshold",
        METRICS,
        "medians[0] < match",
        "medians[0] <= match",
        TEST_M + "::test_threshold_gray_zone_does_not_vote",
    ),
    (
        "median_reject_threshold",
        METRICS,
        "medians[1] > reject",
        "medians[1] >= reject",
        TEST_M + "::test_threshold_gray_zone_does_not_vote",
    ),
    (
        "quality_threshold",
        METRICS,
        "mismatch = np.isfinite(chosen) & (chosen > MISMATCH_ERROR)",
        "mismatch = np.isfinite(chosen) & (chosen >= MISMATCH_ERROR)",
        TEST_M + "::test_mismatch_threshold_and_correct_rows",
    ),
    (
        "available_at_start",
        METRICS,
        'pl.Series("available_at", times + STEP)',
        'pl.Series("available_at", times)',
        TEST_M + "::test_alignment_registered_day_boundary",
    ),
    (
        "deduplicate_conflict",
        METRICS,
        "if row != seen[t]:",
        "if False:",
        TEST_M + "::test_duplicate_exact_only",
    ),
    (
        "firewall_cutoff",
        TIMING + "view.py",
        'pl.col("available_at") < b',
        'pl.col("available_at") <= b',
        TEST_T + "::test_firewall_all_readers_at_end_excluded",
    ),
    (
        "lookahead_disabled",
        TIMING + "view.py",
        "if left.tobytes() != right.tobytes():",
        "if False:",
        TEST_T + "::test_firewall_and_lookahead_reject_next_day_and_sealed",
    ),
    (
        "cross_boundary_kept",
        TIMING + "inference.py",
        "entries + horizon * DAY <= end",
        "entries <= end",
        TEST_T + "::test_cross_boundary_labels_hand_calculation",
    ),
    (
        "validation_twice",
        TIMING + "ledger.py",
        'if stage == "validation" and any(',
        "if False and any(",
        TEST_T + "::test_validation_once_list_hash_and_frozen_dates",
    ),
    (
        "confirmation_twice",
        TIMING + "ledger.py",
        'if stage == "confirmation" and any(',
        "if False and any(",
        TEST_T + "::test_confirmation_whole_list_once_and_hash",
    ),
    (
        "list_hash_ignored",
        TIMING + "settings.py",
        "if expected_hash is None or digest(payload) != expected_hash:",
        "if False:",
        TEST_T + "::test_validation_once_list_hash_and_frozen_dates",
    ),
    (
        "definition_cache_disabled",
        TIMING + "study.py",
        "if cached is not None:",
        "if False:",
        TEST_T + "::test_mining_repeat_definition_does_not_increment",
    ),
    (
        "definition_names_counted",
        TIMING + "definition.py",
        'fn.name = "_factor"',
        "fn.name = fn.name",
        TEST_T + "::test_definition_normalization_and_parameter_identity",
    ),
    (
        "replace_study_settings",
        TIMING + "study.py",
        "    @property\n    def settings(self):\n        return self._settings",
        "    @property\n    def settings(self):\n        return self._settings\n\n    @settings.setter\n    def settings(self, value):\n        self._settings = value",
        TEST_T + "::test_study_settings_reference_cannot_change_after_validation",
    ),
    (
        "factor_exact_type",
        TIMING + "definition.py",
        "if type(factor) is not Factor:",
        "if not isinstance(factor, Factor):",
        TEST_R + "::test_uncontrolled_callable_rejected_before_cache_and_lists",
    ),
    (
        "execution_immutable",
        TIMING + "definition.py",
        'raise AttributeError("Factor 定义只读")',
        "object.__setattr__(self, name, value)",
        TEST_R + "::test_execution_and_parameters_cannot_be_replaced",
    ),
    (
        "execution_hash_binding",
        TIMING + "definition.py",
        "code=digest(encoded(_code_payload(self._code)))",
        'code="unbound"',
        TEST_R + "::test_hash_follows_actual_executable_even_after_private_replacement",
    ),
    (
        "sustained_opposite",
        METRICS,
        "if run >= OPPOSITE_RUN_LIMIT:",
        "if False:",
        TEST_R + "::test_real_longest_isolated_reverse_and_k_boundary",
    ),
    (
        "opposite_k_too_high",
        METRICS,
        "OPPOSITE_RUN_LIMIT = 2",
        "OPPOSITE_RUN_LIMIT = 3",
        TEST_R + "::test_real_longest_isolated_reverse_and_k_boundary",
    ),
    (
        "opposite_k_too_low",
        METRICS,
        "OPPOSITE_RUN_LIMIT = 2",
        "OPPOSITE_RUN_LIMIT = 1",
        TEST_R + "::test_real_longest_isolated_reverse_and_k_boundary",
    ),
    (
        "minimum_paired",
        METRICS,
        "MIN_PAIRED_ROWS = 120",
        "MIN_PAIRED_ROWS = 119",
        TEST_R + "::test_minimum_paired_daily_samples",
    ),
    (
        "verify_opposite",
        "src/ashare/data/crypto/verify_metrics.py",
        "if run >= 2:",
        "if False:",
        TEST_R + "::test_real_longest_isolated_reverse_and_k_boundary",
    ),
    (
        "verify_k",
        "src/ashare/data/crypto/verify_metrics.py",
        "if run >= 2:",
        "if run >= 3:",
        TEST_R + "::test_real_longest_isolated_reverse_and_k_boundary",
    ),
    (
        "verify_minimum_paired",
        "src/ashare/data/crypto/verify_metrics.py",
        "if len(paired) >= 120:",
        "if len(paired) >= 119:",
        TEST_R + "::test_minimum_paired_daily_samples",
    ),
    (
        "ledger_path_binding",
        TIMING + "ledger.py",
        'if str(path) != settings.value["ledger"]:',
        "if False:",
        TEST_R + "::test_ledger_path_cannot_reset_gates",
    ),
    (
        "test_identity_count",
        TIMING + "settings.py",
        "    def test_hash(self, definition_hash):",
        "    def test_hash(self, definition_hash):\n        return definition_hash",
        TEST_R + "::test_label_horizons_are_three_tests_and_bh",
    ),
    (
        "registered_assignment",
        METRICS,
        "choice = int(day >= switch)",
        "choice = int(day > switch)",
        TEST_M + "::test_alignment_registered_day_boundary",
    ),
    (
        "registered_check",
        METRICS,
        "if vote >= 0 and vote != choice:",
        "if False:",
        TEST_S + "::test_registered_constant_checked_by_daily_median",
    ),
    (
        "cross_midnight_run",
        METRICS,
        '        choice = d["alignment"]\n        block = error[a:b]',
        '        run = 0\n        choice = d["alignment"]\n        block = error[a:b]',
        TEST_S + "::test_reverse_run_crosses_midnight",
    ),
    (
        "switch_window",
        METRICS,
        "if opposite and abs(int(stamp) - switch) <= SWITCH_WINDOW:",
        "if False:",
        TEST_S + "::test_switch_window_rejects_single_reverse",
    ),
    (
        "switch_window_closed",
        METRICS,
        "abs(int(stamp) - switch) <= SWITCH_WINDOW",
        "abs(int(stamp) - switch) < SWITCH_WINDOW",
        TEST_S + "::test_switch_window_rejects_single_reverse",
    ),
    (
        "verify_registered_check",
        "src/ashare/data/crypto/verify_metrics.py",
        "if choice >= 0 and choice != assigned:",
        "if False:",
        TEST_S + "::test_registered_constant_checked_by_daily_median",
    ),
    (
        "verify_cross_midnight",
        "src/ashare/data/crypto/verify_metrics.py",
        "        longest = count = 0",
        "        run = 0\n        longest = count = 0",
        TEST_S + "::test_reverse_run_crosses_midnight",
    ),
    (
        "verify_switch_window",
        "src/ashare/data/crypto/verify_metrics.py",
        "if opposite and abs(stamp - switch) <= 86400000:",
        "if False:",
        TEST_S + "::test_switch_window_rejects_single_reverse",
    ),
    (
        "cache_consistency",
        TIMING + "ledger.py",
        "valid = self.valid_result_record(r) and all(",
        "valid = True or all(",
        TEST_S + "::test_cache_identity_mismatch_recomputed_with_warning",
    ),
    (
        "cache_complete_structure",
        TIMING + "ledger.py",
        "        required = {",
        "        return True\n        required = {",
        TEST_S + "::test_incomplete_cache_is_miss_with_warning",
    ),
    (
        "dates_test_identity",
        TIMING + "settings.py",
        'stages=self.value["stages"],',
        "stages=None,",
        TEST_S + "::test_three_mining_starts_are_three_attempts_and_bh",
    ),
    (
        "recomputation_data_revision",
        TIMING + "study.py",
        "previous is not None and previous != self.data_version",
        "previous is not None",
        TEST_S + "::test_same_test_and_data_with_changed_seed_is_not_revision",
    ),
    (
        "recomputation_requires_result",
        TIMING + "study.py",
        "and self.ledger.valid_result_record(r)",
        "and True",
        TEST_S + "::test_cache_identity_mismatch_recomputed_with_warning",
    ),
    (
        "recomputation_latest_version",
        TIMING + "study.py",
        "for r in reversed(self.ledger.records())",
        "for r in self.ledger.records()",
        TEST_S + "::test_revision_compares_latest_attempt_not_any_old_version",
    ),
]


def run(root, tests, *, timeout=180):
    env = dict(
        os.environ,
        PYTHONPATH=str(root / "src"),
        PYTHONDONTWRITEBYTECODE="1",
        POLARS_MAX_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        VECLIB_MAXIMUM_THREADS="1",
    )
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--tb=short", *tests],
        # 保留原测试树及原生扩展构建校验，只替换 ashare 的源码搜索路径。
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def source_fingerprint():
    h = hashlib.sha256()
    paths = [
        Path(__file__),
        *sorted((ROOT / "src").rglob("*.py")),
        *sorted((ROOT / "test").rglob("*.py")),
    ]
    for path in paths:
        h.update(str(path.relative_to(ROOT)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def load_progress(path, fingerprint):
    if not path.exists():
        return None
    state = json.loads(path.read_text())
    if state.get("fingerprint") != fingerprint:
        raise ValueError("MUTATION_STATE: source/test changed; restart mutation 0")
    if [r["name"] for r in state["mutations"]] != [
        m[0] for m in MUTATIONS[: len(state["mutations"])]
    ]:
        raise ValueError("MUTATION_STATE: non-prefix mutation progress")
    return state


def save_progress(path, state):
    tmp = path.with_suffix(".part")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--restart", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.batch_size <= 10:
        raise ValueError("batch-size must be between 1 and 10")
    imports = []
    for name in (
        "ashare.data.crypto.metrics",
        "ashare.data.crypto.verify_metrics",
        "ashare.factor.timing",
        "ashare.factor.crypto_eval",
    ):
        path = Path(importlib.import_module(name).__file__).resolve()
        if not path.is_relative_to(ROOT / "src"):
            raise RuntimeError("导入的不是本 worktree 源码")
        imports.append(path.relative_to(ROOT).as_posix())
    progress = ROOT / "tmp/c8-mutation-progress.json"
    fingerprint = source_fingerprint()
    state = None if args.restart else load_progress(progress, fingerprint)
    if state is None:
        state = dict(
            fingerprint=fingerprint,
            imports=imports,
            mutation_0_passed=False,
            isolated_control_passed=False,
            mutations=[],
            seconds=0.0,
        )
    began_batch = time.perf_counter()
    previous_seconds = state["seconds"]

    def checkpoint():
        state["seconds"] = previous_seconds + time.perf_counter() - began_batch
        save_progress(progress, state)

    if not state["mutation_0_passed"]:
        baseline = run(ROOT, [TEST_M, TEST_T, TEST_R, TEST_S], timeout=600)
        (ROOT / "tmp/c8-mutation-0.log").write_text(baseline.stdout + baseline.stderr)
        if baseline.returncode:
            raise RuntimeError(
                "变异0失败: " + baseline.stdout[-5000:] + baseline.stderr[-1000:]
            )
        state["mutation_0_passed"] = True
        state["mutation_0_tests"] = int(
            re.search(r"(\d+) passed", baseline.stdout).group(1)
        )
        checkpoint()
        print("mutation 0 passed", state["mutation_0_tests"], flush=True)
    with tempfile.TemporaryDirectory(
        prefix="c8-mutations-", dir=ROOT / "tmp"
    ) as folder:
        isolated = Path(folder)
        shutil.copytree(
            ROOT / "src",
            isolated / "src",
            ignore=shutil.ignore_patterns("__pycache__", "*.so"),
        )
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import ashare.data.crypto.metrics as m; from pathlib import Path; import sys; assert Path(m.__file__).resolve().is_relative_to(Path(sys.argv[1]).resolve())",
                str(isolated),
            ],
            cwd=isolated,
            env=dict(
                os.environ,
                PYTHONPATH=str(isolated / "src"),
                PYTHONDONTWRITEBYTECODE="1",
            ),
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        if not state["isolated_control_passed"]:
            control = run(isolated, [TEST_M, TEST_T, TEST_R, TEST_S], timeout=600)
            (ROOT / "tmp/c8-mutation-control.log").write_text(
                control.stdout + control.stderr
            )
            if control.returncode:
                raise RuntimeError(
                    "隔离副本基线失败: "
                    + control.stdout[-4000:]
                    + control.stderr[-1000:]
                )
            state["isolated_control_passed"] = True
            checkpoint()
            print("isolated control passed", flush=True)
        start = len(state["mutations"])
        for name, file, before, after, test in MUTATIONS[
            start : start + args.batch_size
        ]:
            path = isolated / file
            original = path.read_text()
            if original.count(before) != 1:
                raise RuntimeError("变异锚点不唯一: " + name)
            began = time.perf_counter()
            mutated = original.replace(before, after)
            compile(mutated, file, "exec")
            path.write_text(mutated)
            try:
                result = run(isolated, [test])
                output = result.stdout + result.stderr
                failed = re.findall(r"^FAILED ([^\s]+)", output, re.M)
                invalid = any(
                    x in output
                    for x in (
                        "SyntaxError",
                        "ImportError",
                        "ModuleNotFoundError",
                        "ERROR collecting",
                    )
                )
                killed = result.returncode == 1 and bool(failed) and not invalid
                row = dict(
                    name=name,
                    killed=killed,
                    invalid=invalid,
                    failed_tests=failed,
                    seconds=time.perf_counter() - began,
                )
                (ROOT / "tmp" / f"c8-mutant-{name}.log").write_text(output)
            finally:
                path.write_text(original)
            state["mutations"].append(row)
            checkpoint()
            print(name, "KILLED" if killed else "SURVIVED", flush=True)
    checkpoint()
    if len(state["mutations"]) < len(MUTATIONS):
        print("checkpoint", len(state["mutations"]), "/", len(MUTATIONS), flush=True)
        raise SystemExit(75)
    report = dict(
        state,
        native_build_guard_preserved=True,
        total=len(MUTATIONS),
        killed=sum(r["killed"] for r in state["mutations"]),
    )
    (ROOT / "bench/results/mutations_c8.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    if report["killed"] != report["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
