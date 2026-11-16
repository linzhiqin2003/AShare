"""D2 定向变异：基线 0、临时源码副本、精确错误/警告断言；不改工作源码。"""

import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
TEST = "test/spec/test_contract.py"
TIMING = "test/spec/test_timing.py"
PIT = "test/spec/test_point_in_time.py"
INTEGRITY = "test/spec/test_integrity.py"
REVIEW = "test/spec/test_review_validation.py"
R3_TIMING = "test/spec/test_r3_timing.py"
R3_INTEGRITY = "test/spec/test_r3_integrity.py"
R4 = "test/spec/test_r4_dependencies.py"
R5 = "test/spec/test_r5_availability.py"


def suppress(source, code):
    class Remove(ast.NodeTransformer):
        def __init__(self):
            self.count = 0

        def visit_Call(self, node):
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr in {"frame", "fail"}
                and node.args
                and isinstance(node.args[0], ast.Constant)
                and node.args[0].value == code
            ):
                self.count += 1
                return ast.copy_location(ast.Constant(None), node)
            return self.generic_visit(node)

    remover = Remove()
    tree = remover.visit(ast.parse(source))
    if not remover.count:
        raise AssertionError(f"没有找到变异点 {code}")
    return ast.unparse(ast.fix_missing_locations(tree)) + "\n"


def replacements(source, before, after, count=1):
    if source.count(before) != count:
        raise AssertionError(f"变异锚点数量变化：{before}")
    return source.replace(before, after)


def run(source_root, selection):
    env = os.environ.copy()
    env.update(
        PYTHONPATH=os.pathsep.join([str(source_root), str(ROOT)]),
        POLARS_MAX_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        OMP_NUM_THREADS="2",
        MPLCONFIGDIR=str(ROOT / ".cache/mpl"),
        TMPDIR=str(ROOT / "tmp"),
    )
    expected = source_root / "ashare/__init__.py"
    program = (
        "import ashare, pathlib, pytest; print('ashare.__file__ =', ashare.__file__, flush=True); assert pathlib.Path(ashare.__file__).resolve() == pathlib.Path("
        + repr(str(expected))
        + ").resolve(); raise SystemExit(pytest.main("
        + repr(["-q", "--tb=short", *selection])
        + "))"
    )
    started = perf_counter()
    proc = subprocess.run(
        [sys.executable, "-c", program],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    return proc, perf_counter() - started


def main():
    (ROOT / "tmp").mkdir(exist_ok=True)
    report = {"baseline": {}, "mutations": []}
    proc, seconds = run(ROOT / "src", ["test/spec"])
    print(
        proc.stdout.splitlines()[0] if proc.stdout else "missing import output",
        flush=True,
    )
    report["baseline"] = dict(
        id=0,
        returncode=proc.returncode,
        seconds=seconds,
        imported_worktree=proc.stdout.startswith(
            "ashare.__file__ = " + str(ROOT / "src/ashare/__init__.py")
        ),
        summary=proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "",
    )
    print(report["baseline"]["summary"], flush=True)
    output = ROOT / "bench/results/mutations_data_spec.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if proc.returncode or not report["baseline"]["imported_worktree"]:
        output.write_text(json.dumps(report, indent=2) + "\n")
        print(proc.stdout[-6000:], flush=True)
        raise RuntimeError("变异 0 不通过，不运行其他变异")
    jobs = [
        (
            "timezone",
            "check_schema.py",
            lambda s: replacements(s, 'and dtype.tz == "UTC"', "", 2),
            "test_s3_1_utc_timestamp_positive_and_timezone_negative",
        ),
        (
            "epoch",
            "check_schema.py",
            lambda s: replacements(
                s,
                "if not valid:  # mutation: epoch",
                'if not valid and field.name != "ts":  # mutation: epoch',
            ),
            "test_s3_1_timestamp_integer_epoch_rejected",
        ),
        (
            "nan",
            "check_content.py",
            lambda s: suppress(s, "nan"),
            "test_s6_null_positive_nan_infinity_negative",
        ),
        (
            "primary_key",
            "check_content.py",
            lambda s: suppress(s, "primary_key"),
            "test_s6_pk_unique",
        ),
        (
            "prices",
            "check_content.py",
            lambda s: replacements(s, "complete & invalid", "pl.lit(False)"),
            "test_s6_price_and_quantity_constraints",
        ),
        (
            "lifecycle",
            "check_content.py",
            lambda s: suppress(s, "lifecycle"),
            "test_s4_lifecycle_inclusive",
        ),
        (
            "available_at",
            "check_content.py",
            lambda s: replacements(
                s, "pl.col(target) < _start_expr", "pl.col(target) > _start_expr"
            ),
            "test_s3_3_bars_available_at_lower_bound",
        ),
        (
            "hash",
            "check.py",
            lambda s: replacements(
                s, '"sha256:" + digest.hexdigest() != entry["hash"]', "False"
            ),
            "test_s7_file_integrity",
        ),
        (
            "version",
            "check_structure.py",
            lambda s: replacements(s, 'manifest.get("version") != expected', "False"),
            "test_s7_version_metadata_change",
        ),
        (
            "calendar",
            "check_content.py",
            lambda s: suppress(s, "calendar_membership"),
            "test_s8_4_calendar_membership",
        ),
        (
            "event_announcement_lower_bound",
            "check_content.py",
            lambda s: replacements(
                s,
                'if resource["kind"] == "bars":  # mutation: availability_kind',
                "if True:  # mutation: availability_kind",
            ),
            f"{TIMING}::test_s3_3_announcement_before_event_date",
        ),
        (
            "unknown_grammar",
            "constants.py",
            lambda s: replacements(s, "start|end|unknown|", "start|end|"),
            f"{TIMING}::test_s3_3_unknown_warning_and_typo_rejected",
        ),
        (
            "unknown_warning_missing",
            "check_schema.py",
            lambda s: suppress(s, "known_at_unknown"),
            f"{TIMING}::test_s3_3_unknown_warning_and_typo_rejected",
        ),
        (
            "unknown_warning_fatal",
            "check_schema.py",
            lambda s: replacements(s, "warning=True", "warning=False"),
            f"{TIMING}::test_s3_3_unknown_warning_and_typo_rejected",
        ),
        (
            "unknown_panel_metadata",
            "panel.py",
            lambda s: replacements(
                s,
                "    return SpecPanel(",
                "    for desc in metadata.values():\n"
                '        if desc.get("knownAt") == "unknown":\n'
                '            desc["knownAt"] = "end"\n'
                "    return SpecPanel(",
            ),
            f"{TIMING}::test_s9_unknown_visible_in_info_and_panel",
        ),
        (
            "delay_placement",
            "panel_timing.py",
            lambda s: replacements(
                s,
                "np.searchsorted(\n                    self.decisions, arrival\n                )",
                "i",
            ),
            f"{PIT}::test_s9_delay_calendar_shift_and_left_boundary",
        ),
        (
            "panel_available_at",
            "panel_timing.py",
            lambda s: replacements(
                s,
                "at <= self.decisions[np.minimum(dest, len(dates) - 1)]",
                "np.ones(len(at), dtype=bool)",
            ),
            f"{PIT}::test_s9_review_future_announcement_is_masked",
        ),
        (
            "read_available_at",
            "timing.py",
            lambda s: replacements(
                s,
                'dependency_time("available_at", resource, manifest) <= instant',
                "pl.lit(True)",
            ),
            f"{PIT}::test_s9_review_future_announcement_is_masked",
        ),
        (
            "read_known_at",
            "timing.py",
            lambda s: replacements(
                s, "pl.when(at <= instant)", "pl.when(pl.lit(True))"
            ),
            f"{PIT}::test_s9_as_of_masks_fields_by_known_at",
        ),
        (
            "unknown_panel_rejection",
            "panel.py",
            lambda s: replacements(s, "if unknown and not allow_unknown:", "if False:"),
            f"{PIT}::test_s9_unknown_rejected_unless_explicitly_allowed",
        ),
        (
            "hash_cache_identity",
            "reader.py",
            lambda s: replacements(
                s, ", stat.st_dev, stat.st_ino, stat.st_ctime_ns", ""
            ),
            f"{INTEGRITY}::test_s7_same_size_restored_mtime_cannot_bypass_hash",
        ),
        (
            "verified_file_instance",
            "scan.py",
            lambda s: replacements(
                s,
                "yield payload if payload is not None else stream, payload",
                "yield str(path), str(path)",
            ),
            f"{INTEGRITY}::test_s7_read_uses_verified_file_instance",
        ),
        (
            "negative_price_mask",
            "panel.py",
            lambda s: replacements(s, "opening != 0", "opening > 0"),
            f"{REVIEW}::test_s9_negative_price_trading_mask",
        ),
        (
            "extension_unit_type",
            "check_schema.py",
            lambda s: replacements(
                s,
                "if unit_type is not None and declared_type != unit_type:",
                "if False:",
            ),
            f"{REVIEW}::test_s5_extension_unit_enforces_type",
        ),
        (
            "extension_price_domain",
            "check_content.py",
            lambda s: replacements(
                s, 'if str(d.get("unit")) in PRICE_UNITS', "if False"
            ),
            f"{REVIEW}::test_s6_extension_price_domain",
        ),
        (
            "resource_symbol_units",
            "check.py",
            lambda s: replacements(
                s, 'pl.col("symbol").is_in(sorted(referenced))', "pl.lit(True)"
            ),
            f"{REVIEW}::test_s5_resource_units_ignore_unreferenced_instruments",
        ),
        (
            "event_quality_bits",
            "check_content.py",
            lambda s: replacements(
                s,
                'if "quality" in columns and frame.schema["quality"].is_integer():',
                'if kind == "bars" and "quality" in columns and frame.schema["quality"].is_integer():',
            ),
            f"{REVIEW}::test_s6_event_quality_flags",
        ),
        (
            "quality_flag_declarations",
            "check.py",
            lambda s: replacements(s, "quality_flags(manifest, report)", "None"),
            f"{REVIEW}::test_s6_invalid_quality_flag_key_reports_without_crashing",
        ),
        (
            "panel_grid_rejection",
            "panel_timing.py",
            lambda s: replacements(
                s, "if not np.all(on_grid & (inside | historical)):", "if False:"
            ),
            f"{R3_TIMING}::test_s9_r3_off_grid_public_panel_rejected",
        ),
        (
            "history_grid_rejection",
            "panel_timing.py",
            lambda s: replacements(
                s, "on_grid & (inside | historical)", "(on_grid & inside) | historical"
            ),
            f"{R3_TIMING}::test_s9_r3_off_grid_history_is_not_discardable",
        ),
        (
            "reader_column_dependency",
            "dependencies.py",
            lambda s: replacements(s, 'and known.startswith("column:")', "and False"),
            f"{R4}::test_s3_3_s9_r4_review_variants_rejected[read-direct-end+1d]",
        ),
        (
            "reader_global_dependency",
            "dependencies.py",
            lambda s: replacements(
                s, 'targets |= {"available_at"} & fields.keys()', "targets |= set()"
            ),
            f"{R4}::test_s3_3_s9_r4_review_variants_rejected[read-global-end+1d]",
        ),
        (
            "reader_time_dependency_rule",
            "dependencies.py",
            lambda s: replacements(
                s,
                'if desc.get("type") != "datetime" or desc.get("knownAt") not in (\n            "start",\n            "end",\n        ):',
                "if False:",
            ),
            f"{R4}::test_s3_3_s9_r4_review_variants_rejected",
        ),
        (
            "unknown_value_metadata",
            "dependencies.py",
            lambda s: replacements(
                s, 'result["knownAt"] = "unknown"', 'result["knownAt"] = "end"'
            ),
            f"{R3_TIMING}::test_s9_r3_unknown_value_read_opt_in",
        ),
        (
            "checker_time_dependency_rule",
            "check_schema.py",
            lambda s: suppress(s, "known_at_dependency"),
            f"{R4}::test_s3_3_s9_r4_review_variants_rejected",
        ),
        (
            "row_known_nonbar_time",
            "timing.py",
            lambda s: replacements(
                s,
                'resource["kind"] != "bars"',
                'resource["kind"] in {"events", "corporate_actions"}',
            ),
            f"{R4}::test_s3_3_s9_r4_static_available_at_is_row_known",
        ),
        (
            "dependency_max_own_clock",
            "timing.py",
            lambda s: replacements(s, "pl.max_horizontal(value, own)", "value"),
            f"{R5}::test_s3_3_s9_r5_review_dependency_effective_max",
        ),
        (
            "dependency_max_time_value",
            "timing.py",
            lambda s: replacements(s, "pl.max_horizontal(value, own)", "own"),
            f"{R5}::test_s3_3_s9_r5_review_dependency_effective_max",
        ),
        (
            "column_effective_max",
            "timing.py",
            lambda s: replacements(
                s,
                "return dependency_time(known[7:], resource, manifest)",
                "return pl.col(known[7:])",
            ),
            f"{R5}::test_s3_3_s9_r5_review_dependency_effective_max[end-column-read]",
        ),
        (
            "global_effective_max",
            "timing.py",
            lambda s: replacements(
                s,
                'dependency_time("available_at", resource, manifest) <= instant',
                'pl.col("available_at") <= instant',
            ),
            f"{R5}::test_s3_3_s9_r5_review_dependency_effective_max[end-global-read]",
        ),
        (
            "static_time_reference",
            "timing.py",
            lambda s: replacements(
                s,
                'if known in {"start", "end"} and resource["kind"] != "bars":',
                'if desc["name"] == "available_at" and known in {"start", "end"} and resource["kind"] != "bars":',
            ),
            f"{R5}::test_s3_3_s9_r5_static_default_equals_projection",
        ),
        (
            "null_dependency_max",
            "timing.py",
            lambda s: replacements(
                s,
                "pl.when(value.is_not_null()).then(pl.max_horizontal(value, own))",
                "pl.max_horizontal(value, own)",
            ),
            f"{R5}::test_s3_3_s9_r5_null_time_not_replaced_by_interval_end",
        ),
        (
            "static_dependency_required",
            "check_content.py",
            lambda s: replacements(
                s,
                "invalid = pl.col(target).is_null()",
                'invalid = pl.col(target).is_null() if resource["kind"] == "bars" else pl.lit(False)',
            ),
            f"{R5}::test_s3_3_r5_static_dependency_null_rejected",
        ),
        (
            "cold_snapshot_parse",
            "scan.py",
            lambda s: replacements(
                s,
                "yield payload if payload is not None else stream, payload",
                "yield stream, None",
            ),
            f"{R3_INTEGRITY}::test_s7_r3_review_mutable_stream_race[False]",
        ),
        (
            "cached_post_read_identity",
            "scan.py",
            lambda s: replacements(
                s,
                "if payload is None and identity(os.fstat(stream.fileno())) != before:",
                "if False:",
            ),
            f"{R3_INTEGRITY}::test_s7_r3_cached_identity_checked_after_parquet_read",
        ),
    ]
    for identifier, filename, mutate, test in jobs:
        with tempfile.TemporaryDirectory(
            prefix="ashare-d2-mutant-", dir=ROOT / "tmp"
        ) as temp:
            source_root = Path(temp)
            shutil.copytree(
                ROOT / "src/ashare",
                source_root / "ashare",
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
            path = source_root / "ashare/data/spec" / filename
            path.write_text(mutate(path.read_text()))
            proc, seconds = run(
                source_root, [test if "::" in test else f"{TEST}::{test}"]
            )
            killed = (
                proc.returncode == 1
                and "FAILED " in proc.stdout
                and "ERROR collecting" not in proc.stdout
            )
            report["mutations"].append(
                dict(
                    name=identifier,
                    killed=killed,
                    returncode=proc.returncode,
                    seconds=seconds,
                    test=test,
                )
            )
            print(
                f"{identifier}: {'killed' if killed else 'SURVIVED/ERROR'}", flush=True
            )
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    report["killed"] = sum(x["killed"] for x in report["mutations"])
    report["total"] = len(jobs)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    assert report["killed"] == report["total"], "存在存活变异"


if __name__ == "__main__":
    main()
