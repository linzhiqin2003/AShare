"""Review mutations plus new regressions, with verified native rebuilds and restoration."""

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RS = "crates/paperbroker-core/src/lib.rs"
REF = "packages/paperbroker/python/paperbroker/reference.py"
JRN = "packages/paperbroker/python/paperbroker/journal.py"
RUN = "src/ashare/paper/runner.py"
API = "packages/paperbroker/python/paperbroker/api.py"
CORE_TEST = "packages/paperbroker/tests/test_revision.py::"
ENGINE_TEST = "packages/paperbroker/tests/test_engine.py::"
JOURNAL_TEST = "packages/paperbroker/tests/test_journal.py::"
RUN_TEST = "test/paper/test_revision.py::"


def cases(originals):
    def exact(path, before, after):
        assert originals[path].count(before) == 1, (path, before)
        return path, before, after

    c = "packages/paperbroker/tests/test_round3.py::"
    r = "test/paper/test_round3.py::"
    time_test = c + "test_receipt_backwards_without_journal_and_unsettled_rollover"
    bounded = c + "test_bounded_snapshots_archived_events_and_gc"
    rs_day = "|| (day != self.state.day && self.state.day != self.state.settled_day)"
    ref_day = ' or (day != s["day"] and s["day"] != s["settled_day"])'
    return [
        (
            "R3",
            "auction lock ignores priced empty ask",
            CORE_TEST + "test_auction_limit_lock",
            [exact(RS, "(book <= 0 || ask == 0)", "(book <= 0)")],
        ),
        (
            "R6",
            "receipt rollback allowed in native engine",
            time_test,
            [
                exact(
                    RS,
                    "if received_at.is_some_and(|t| t < self.state.time)",
                    "if false",
                )
            ],
        ),
        ("R10", "unsettled rollover allowed", time_test, [exact(RS, rs_day, "")]),
        (
            "R12",
            "closing baseline retained across days",
            RUN_TEST + "test_two_days_baseline_report_and_calendar_refresh",
            [
                exact(
                    RUN,
                    "                self.close_baseline = {}",
                    "                pass",
                )
            ],
        ),
        (
            "D9",
            "unsettled rollover allowed in both models",
            time_test,
            [exact(RS, rs_day, ""), exact(REF, ref_day, "")],
        ),
        (
            "D10",
            "priced empty lock ignored in both models",
            CORE_TEST + "test_auction_limit_lock",
            [
                exact(RS, "(book <= 0 || ask == 0)", "(book <= 0)"),
                exact(RS, "(book <= 0 || bid == 0)", "(book <= 0)"),
                exact(
                    REF,
                    'if locked and (book <= 0 or available[o["side"]] == 0):',
                    "if locked and book <= 0:",
                ),
            ],
        ),
        (
            "T1",
            "outage checks today only",
            r + "test_outage_actions_in_date_order",
            [
                exact(
                    RUN,
                    "for ex_day in sorted(required & self.schedule.open_days):",
                    "for ex_day in [day]:",
                )
            ],
        ),
        (
            "T2",
            "same security cannot receive consecutive dated actions",
            r + "test_outage_actions_in_date_order",
            [
                exact(
                    RS,
                    "&& self.state.corporate_dates[a.symbol] < day",
                    "&& !self.state.corporate_done[a.symbol]",
                ),
                exact(
                    RS,
                    "self.state.corporate_done[a.symbol] = day == self.state.day;",
                    "self.state.corporate_done[a.symbol] = true;",
                ),
            ],
        ),
        (
            "T3",
            "release arithmetic wraps in extension",
            c + "test_feed_overflow_preflight_does_not_poison_wal",
            [exact("Cargo.toml", "overflow-checks = true", "overflow-checks = false")],
        ),
        (
            "T4",
            "bad quote written before overflow trial",
            c + "test_feed_overflow_preflight_does_not_poison_wal",
            [exact(JRN, "        trial.feed(quotes, received_at)", "        pass")],
        ),
        (
            "T5",
            "stale quote blocks settlement after deadline",
            r + "test_close_deadline_is_per_security",
            [
                exact(
                    RUN,
                    "if stale and now.time().replace(tzinfo=None) < self.settlement_deadline:",
                    "if stale:",
                )
            ],
        ),
        (
            "T6",
            "snapshot retention disabled",
            bounded,
            [
                exact(
                    JRN,
                    "for old in paths[self.broker.snapshot_retention :]:",
                    "for old in []:",
                )
            ],
        ),
        (
            "T7",
            "old quote retention disabled",
            bounded,
            [exact(JRN, "        if not keep:\n", "        if True:\n")],
        ),
        (
            "T8",
            "output event sequence repeats",
            bounded,
            [
                exact(
                    RS,
                    "self.state.event_sequence += 1;",
                    "self.state.event_sequence += 0;",
                )
            ],
        ),
        (
            "T9",
            "cursor returns duplicate boundary event",
            bounded,
            [
                exact(
                    API,
                    '            if e["seq"] > after:\n                events[e["seq"]] = e',
                    '            if e["seq"] >= after:\n                events[e["seq"]] = e',
                )
            ],
        ),
        (
            "T10",
            "mainboard odd lots require complete liquidation",
            c + "test_mainboard_residual_lot",
            [
                exact(
                    RS, "qty % model.step == available % model.step", "qty == available"
                )
            ],
        ),
        (
            "T11",
            "history retained indefinitely",
            bounded,
            [
                exact(
                    API,
                    'self.history = [e for e in self.history if e["day"] == day]',
                    "self.history = list(self.history)",
                )
            ],
        ),
        (
            "T12",
            "early final quote cannot acquire auction volume",
            r + "test_early_close_quote_can_be_enriched_once",
            [exact(RS, "&& !(q[17] > 0", "&& !(false && q[17] > 0")],
        ),
        (
            "T13",
            "rejected targets retry every poll",
            r + "test_clock_rollback_run_survives_and_targets_back_off",
            [
                exact(
                    RUN,
                    "        self.last_target = self.clock.monotonic()\n        results =",
                    "        results =",
                )
            ],
        ),
        (
            "T14",
            "status fsync on every poll",
            r + "test_reports_not_rewritten_status_throttled_gc_cli",
            [
                exact(
                    RUN,
                    "if self.clock.monotonic() - self.last_status >= self.status_interval:",
                    "if True:",
                )
            ],
        ),
        (
            "T15",
            "stale symbols receive closing proxy volume",
            r + "test_close_deadline_is_per_security",
            [
                exact(
                    RUN,
                    "if baseline is not None and i not in stale",
                    "if baseline is not None",
                )
            ],
        ),
        (
            "T16",
            "historical ex-date can be applied twice",
            c + "test_historical_action_idempotence_and_consecutive_dates",
            [
                exact(
                    RS,
                    "&& self.state.corporate_dates[a.symbol] < day",
                    "&& !self.state.corporate_done[a.symbol]",
                )
            ],
        ),
    ]


def main():
    directory = ROOT / ".cache/round3/mutations"
    directory.mkdir(parents=True, exist_ok=True)
    originals = {
        p: (ROOT / p).read_text() for p in (RS, REF, JRN, RUN, API, "Cargo.toml")
    }
    mutations = cases(originals)
    env = os.environ | dict(
        CARGO_HOME=str(ROOT / ".cache/cargo-home"),
        PYO3_PYTHON=str(ROOT / ".venv/bin/python"),
        TMPDIR=str(ROOT / ".cache/round3/tmp"),
        PYTHONDONTWRITEBYTECODE="1",
    )

    def run(args, name):
        result = subprocess.run(
            args,
            cwd=ROOT,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        (directory / (name + ".log")).write_text(result.stdout)
        return result

    def native_hash():
        result = run(
            [
                str(ROOT / ".venv/bin/python"),
                "-c",
                "import hashlib,json,paperbroker._core as c;from pathlib import Path;p=Path(c.__file__).resolve();assert p.is_relative_to(Path.cwd());print(json.dumps({'path':str(p.relative_to(Path.cwd())),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))",
            ],
            "import-hash",
        )
        assert result.returncode == 0
        return json.loads(result.stdout)

    def build(name):
        result = run(
            [
                "uv",
                "run",
                "--no-sync",
                "maturin",
                "develop",
                "--release",
                "--manifest-path",
                "packages/paperbroker/Cargo.toml",
            ],
            name,
        )
        assert result.returncode == 0, result.stdout
        imported = native_hash()
        artifact = ROOT / "target/release/libpaperbroker_native.dylib"
        assert imported["sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
        return imported

    baseline_import = build("baseline-build")
    control = run(["make", "check"], "baseline-check")
    assert control.returncode == 0, control.stdout
    results = []
    try:
        for tag, description, test, edits in mutations:
            for path, before, after in edits:
                file = ROOT / path
                file.write_text(file.read_text().replace(before, after))
            rust = any(p in (RS, "Cargo.toml") for p, _, _ in edits)
            imported = build(tag + "-build") if rust else native_hash()
            assert (
                (imported["sha256"] != baseline_import["sha256"])
                if rust
                else (imported == baseline_import)
            )
            checked = run(
                [
                    str(ROOT / ".venv/bin/python"),
                    "-m",
                    "pytest",
                    test,
                    "-q",
                    "-p",
                    "no:cacheprovider",
                ],
                tag + "-test",
            )
            # A collected test must fail. Build failures and collection errors are not kills.
            caught = (
                checked.returncode == 1
                and "FAILED " in checked.stdout
                and "ERROR collecting" not in checked.stdout
            )
            failed = re.findall(r"^FAILED (\S+)", checked.stdout, re.M)
            results.append(
                dict(
                    id=tag,
                    mutation=description,
                    test=test,
                    caught=caught,
                    native_rebuilt=rust,
                    imported=imported,
                    failed=failed,
                    test_exit=checked.returncode,
                    log_sha256=hashlib.sha256(checked.stdout.encode()).hexdigest(),
                )
            )
            print(json.dumps(results[-1]), flush=True)
            for path in originals:
                (ROOT / path).write_text(originals[path])
            restored = build(tag + "-restore") if rust else native_hash()
            assert restored == baseline_import, (tag, restored, baseline_import)
            assert caught, checked.stdout
    finally:
        for path in originals:
            (ROOT / path).write_text(originals[path])
        assert build("final-restore") == baseline_import
    final = run(["make", "check"], "restored-check")
    assert final.returncode == 0, final.stdout
    result = dict(
        baseline_passed=True,
        restored_check_passed=True,
        baseline_import=baseline_import,
        sources={
            p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in originals
        },
        mutations=results,
    )
    (ROOT / ".cache/round3/mutation-results.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(
        f"All {len(results)} mutations caught; source and extension rebuilt and restored."
    )


if __name__ == "__main__":
    main()
