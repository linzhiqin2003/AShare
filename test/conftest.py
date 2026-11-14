"""Test-suite wide setup: build guard and automatic test categories.

Every collected test gets exactly one *primary* category (used for timing
reports, recorded as the junit property ``category``) plus a marker for each
category it belongs to, so ``pytest -m property`` or ``make test-cat CAT=...``
select by category. Rules, in precedence order:

1. an explicit category marker on the test wins;
2. directory: ``test/<smoke|integration|e2e|concurrency|perf>/...``;
3. hypothesis tests are ``property``;
4. differential tests (cross-checks against an independent reference) are
   ``differential``: file names such as ``*bitwise*``/``*reference*``, test
   modules importing the reference helpers listed below, or single tests named
   ``*bitwise*``/``*legacy*``/``*reference*``;
5. everything else is ``unit``.

Property and differential can both apply; the primary category is the first
match in the list above.
"""

import os
import re
from pathlib import Path

import pytest

from .build_guard import assert_current_build

assert_current_build()

ROOT = Path(__file__).resolve().parents[1]

CATEGORIES = {
    "unit": "fast isolated test of one function or class (default)",
    "integration": "several components wired together (test/integration)",
    "e2e": "full research run from CLI or platform API (test/e2e)",
    "smoke": "one-minute sanity sweep (test/smoke)",
    "concurrency": "threads, processes, cancellation, locking (test/concurrency)",
    "perf": "performance regression guards (test/perf)",
    "property": "hypothesis property-based test",
    "differential": "cross-check against an independent reference implementation",
}

# Directories whose tests all belong to one category.
DIRECTORY_CATEGORIES = ("smoke", "integration", "e2e", "concurrency", "perf")

# File names that mark a whole module as differential.
DIFFERENTIAL_FILE = re.compile(
    r"bitwise|reference|crosscheck|exact|independent|differential|parity|comparison"
)

# Test function names that mark a single test as differential (narrower than
# the file pattern: "exact" or "comparison" alone are too common in names).
DIFFERENTIAL_FUNCTION = re.compile(
    r"bitwise|reference|crosscheck|cross_check|differential|parity|legacy|oracle"
)

# Helper modules holding independent reference implementations; a test module
# that imports from them is a differential test.
DIFFERENTIAL_HELPERS = (
    "test.comparison",
    "test.legacy",
    "test.digest",
    "test.events.reference",
    "test.analysis.analysis_reference",
)

PRECEDENCE = (*DIRECTORY_CATEGORIES, "property", "differential", "unit")

# Developer-shell variables that change product behaviour or unlock real data
# sources. Tests must see the same environment as CI, so they are removed for
# the whole session (plus every ``ASHARE_*`` override); tests that need one set
# it with ``monkeypatch.setenv``.
SCRUBBED_ENV = ("TUSHARE_TOKEN", "DATABENTO_API_KEY")


_saved_env = {}


def _scrub_environment():
    for name in list(os.environ):
        if name in SCRUBBED_ENV or name.startswith("ASHARE_"):
            _saved_env[name] = os.environ.pop(name)


def pytest_configure(config):
    for name, description in CATEGORIES.items():
        config.addinivalue_line("markers", f"{name}: {description}")
    # Before xdist starts its workers, so they inherit the scrubbed environment.
    _scrub_environment()


def pytest_unconfigure(config):
    os.environ.update(_saved_env)
    _saved_env.clear()


def _directory_category(path):
    try:
        parts = path.resolve().relative_to(ROOT).parts
    except ValueError:
        return None
    if len(parts) >= 2 and parts[0] == "test" and parts[1] in DIRECTORY_CATEGORIES:
        return parts[1]
    return None


def _module_is_differential(module, cache):
    name = getattr(module, "__name__", "")
    if name in cache:
        return cache[name]
    stem = Path(getattr(module, "__file__", "") or "").stem
    result = bool(DIFFERENTIAL_FILE.search(stem))
    if not result:
        for value in vars(module).values():
            origin = getattr(value, "__module__", None) or getattr(
                value, "__name__", None
            )
            if isinstance(origin, str) and origin.startswith(DIFFERENTIAL_HELPERS):
                result = True
                break
    cache[name] = result
    return result


def categories_for(item, cache=None):
    """Return the categories of ``item``, primary category first."""
    cache = {} if cache is None else cache
    found = {m.name for m in item.iter_markers() if m.name in CATEGORIES}
    explicit = bool(found)
    directory = _directory_category(Path(str(item.path)))
    if directory:
        found.add(directory)
    function = getattr(item, "function", None)
    if getattr(function, "is_hypothesis_test", False):
        found.add("property")
    module = getattr(item, "module", None)
    if module is not None and _module_is_differential(module, cache):
        found.add("differential")
    elif DIFFERENTIAL_FUNCTION.search(getattr(item, "originalname", item.name)):
        found.add("differential")
    if not found:
        found.add("unit")
    if explicit:
        explicit_names = [m.name for m in item.iter_markers() if m.name in CATEGORIES]
        primary = min(explicit_names, key=PRECEDENCE.index)
    else:
        primary = min(found, key=PRECEDENCE.index)
    return [primary, *sorted(found - {primary}, key=PRECEDENCE.index)]


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    # Markers must exist before ``-m`` deselection runs. Conftest hooks already
    # run before the builtin mark plugin; tryfirst keeps it so if that changes.
    cache = {}
    for item in items:
        names = categories_for(item, cache)
        for name in names:
            if not item.get_closest_marker(name):
                item.add_marker(name)
        item.user_properties.append(("category", names[0]))
