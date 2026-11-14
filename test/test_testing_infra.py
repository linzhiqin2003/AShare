"""Tests for the test infrastructure: categories (test/conftest.py) and the
metric tools in bench/testing (junit parsing, runtime budgets, flaky
aggregation, metrics page update)."""

import importlib
import sys
import types
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from . import conftest as suite

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "bench" / "testing"


@pytest.fixture
def tool(monkeypatch):
    """Import a bench/testing script without leaking its flat module names."""
    monkeypatch.syspath_prepend(str(TOOLS))
    loaded = []

    def load(name):
        for dep in ("budgets", "common", name):
            monkeypatch.delitem(sys.modules, dep, raising=False)
        module = importlib.import_module(name)
        loaded.append(name)
        return module

    yield load
    for name in ("budgets", "common", *loaded):
        sys.modules.pop(name, None)


class FakeItem:
    def __init__(self, path, *, name="test_x", markers=(), function=None, module=None):
        self.path = Path(path)
        self.name = self.originalname = name
        self._markers = [types.SimpleNamespace(name=m) for m in markers]
        self.function = function
        self.module = module

    def iter_markers(self):
        return iter(self._markers)


def _module(name, file, **attrs):
    module = types.ModuleType(name)
    module.__file__ = file
    module.__dict__.update(attrs)
    return module


PLAIN = _module("test.test_plain", str(ROOT / "test/test_plain.py"))


@pytest.mark.parametrize(
    "directory", ["smoke", "integration", "e2e", "concurrency", "perf"]
)
def test_directory_sets_category(directory):
    item = FakeItem(ROOT / "test" / directory / "test_a.py", module=PLAIN)
    assert suite.categories_for(item) == [directory]


def test_default_category_is_unit():
    assert suite.categories_for(FakeItem(ROOT / "test/test_a.py", module=PLAIN)) == [
        "unit"
    ]


def test_hypothesis_test_is_property():
    function = types.SimpleNamespace(is_hypothesis_test=True)
    item = FakeItem(ROOT / "test/test_a.py", function=function, module=PLAIN)
    assert suite.categories_for(item) == ["property"]


def test_differential_by_file_name_helper_import_and_test_name():
    by_file = _module("test.x", str(ROOT / "test/test_groupbt_bitwise.py"))
    assert suite.categories_for(FakeItem(by_file.__file__, module=by_file)) == [
        "differential"
    ]

    def helper():
        pass

    helper.__module__ = "test.comparison"
    by_import = _module("test.y", str(ROOT / "test/test_y.py"), helper=helper)
    assert suite.categories_for(FakeItem(by_import.__file__, module=by_import)) == [
        "differential"
    ]
    by_name = FakeItem(
        ROOT / "test/test_z.py", name="test_legacy_bitwise_fixture", module=PLAIN
    )
    assert suite.categories_for(by_name) == ["differential"]


def test_property_and_differential_both_marked_property_primary():
    module = _module("test.x", str(ROOT / "test/test_reference_x.py"))
    function = types.SimpleNamespace(is_hypothesis_test=True)
    item = FakeItem(module.__file__, function=function, module=module)
    assert suite.categories_for(item) == ["property", "differential"]


def test_explicit_marker_wins_over_directory():
    item = FakeItem(ROOT / "test/perf/test_a.py", markers=["unit"], module=PLAIN)
    names = suite.categories_for(item)
    assert names[0] == "unit" and "perf" in names


def test_this_test_is_marked_unit(request):
    assert request.node.get_closest_marker("unit") is not None
    assert ("category", "unit") in request.node.user_properties


@settings(
    max_examples=5,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(st.integers())
def test_hypothesis_tests_are_marked_property(request, value):
    assert request.node.get_closest_marker("property") is not None


JUNIT = """<?xml version="1.0" encoding="utf-8"?>
<testsuites><testsuite name="pytest">
<testcase classname="test.test_a" name="test_ok" time="0.5">
  <properties><property name="category" value="unit"/></properties>
</testcase>
<testcase classname="test.test_a" name="test_slow" time="3.0">
  <properties><property name="category" value="unit"/></properties>
</testcase>
<testcase classname="test.test_b" name="test_bad" time="0.1">
  <properties><property name="category" value="property"/></properties>
  <failure message="assert 1 == 2&#10;second line">trace</failure>
</testcase>
<testcase classname="test.test_b" name="test_skip" time="0.0">
  <skipped message="no databento"/>
</testcase>
</testsuite></testsuites>
"""


def test_parse_junit_and_runtime_budget(tool, tmp_path):
    common = tool("common")
    runtime = tool("runtime")
    path = tmp_path / "junit.xml"
    path.write_text(JUNIT)
    records = common.parse_junit(path)
    assert [r["outcome"] for r in records] == ["passed", "passed", "failed", "skipped"]
    assert records[2]["message"] == "assert 1 == 2"
    assert records[3]["category"] == "unknown"
    summary = runtime.summarize(records, wall=4.0)
    assert summary["categories"]["unit"] == {"tests": 2, "seconds": 3.5}
    assert summary["slowest"][0]["id"] == "test.test_a::test_slow"
    assert [r["id"] for r in summary["over_budget"]] == ["test.test_a::test_slow"]


def test_flaky_aggregate_separates_flaky_from_failing(tool):
    flaky = tool("flaky")

    def rec(test_id, outcome):
        return {"id": test_id, "outcome": outcome, "message": "m"}

    runs = [
        [
            rec("a", "passed"),
            rec("b", "failed"),
            rec("c", "passed"),
            rec("s", "skipped"),
        ],
        [
            rec("a", "failed"),
            rec("b", "error"),
            rec("c", "passed"),
            rec("s", "skipped"),
        ],
        [rec("a", "passed"), rec("b", "failed"), rec("c", "passed")],
    ]
    summary = flaky.aggregate(runs)
    assert [r["id"] for r in summary["flaky"]] == ["a"]
    assert [r["id"] for r in summary["failing"]] == ["b"]
    assert summary["missing_in_some_runs"] == ["s"]
    assert summary["tests"] == 4
    assert summary["flaky_rate"] == pytest.approx(0.25)


def test_metrics_doc_replaces_same_day_and_orders_newest_first(tool):
    report = tool("report")
    text = report.update_doc("", "## 2026-11-13\n\nold", "2026-11-13")
    assert text.startswith(report.DOC_HEADER)
    text = report.update_doc(text, "## 2026-11-14\n\nnew", "2026-11-14")
    assert text.index("## 2026-11-14") < text.index("## 2026-11-13")
    text = report.update_doc(text, "## 2026-11-14\n\nnewer", "2026-11-14")
    assert (
        text.count("## 2026-11-14") == 1 and "newer" in text and "\nnew\n" not in text
    )
    assert "old" in text


def test_report_marks_missing_inputs_not_run(tool, tmp_path, monkeypatch):
    report = tool("report")
    monkeypatch.setattr(report, "CACHE", tmp_path)
    result = report.build_report()
    for key in ("coverage", "runtime", "mutation", "flaky", "perf"):
        assert result[key]["status"] == report.NOT_RUN
    section = report.render_section(result, "2026-11-14")
    assert section.count(report.NOT_RUN) == 5


def test_report_reads_mutation_contract(tool, tmp_path, monkeypatch):
    report = tool("report")
    common = tool("common")
    monkeypatch.setattr(report, "CACHE", tmp_path)
    common.write_json(
        tmp_path / "mutation.json",
        {
            "generated": "2026-11-14T10:00:00+08:00",
            "scope": "full",
            "modules": [
                {
                    "module": "panelbt",
                    "language": "rust",
                    "killed": 8,
                    "survived": 2,
                    "timeout": 0,
                    "equivalent": 1,
                    "score": 0.8,
                }
            ],
            "total": {
                "killed": 8,
                "survived": 2,
                "timeout": 0,
                "equivalent": 1,
                "score": 0.8,
            },
        },
    )
    result = report.build_report()
    assert result["mutation"]["score"] == 0.8
    assert "80.0%" in report.render_section(result, "2026-11-14")


def test_developer_environment_does_not_leak_into_tests():
    # 2026-11-14: the developer shell exports TUSHARE_TOKEN (and may export
    # ASHARE_* overrides); tests then ran with a real token and local settings
    # that CI does not have. The suite scrubs them before any test runs.
    import os

    leaked = sorted(
        name
        for name in os.environ
        if name in suite.SCRUBBED_ENV or name.startswith("ASHARE_")
    )
    assert leaked == []


def test_category_selection_with_dash_m():
    # Markers are added at collection time; ``-m`` must see them.
    import subprocess

    out = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            "-m",
            "property",
            str(Path(__file__)),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    ).stdout
    selected = [line for line in out.splitlines() if "::" in line]
    assert selected == [
        "test/test_testing_infra.py::test_hypothesis_tests_are_marked_property"
    ]
