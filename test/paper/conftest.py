from pathlib import Path

import paperbroker
import paperbroker._core

import ashare.paper

ROOT = Path(__file__).resolve().parents[2]
assert Path(paperbroker.__file__).resolve().is_relative_to(ROOT)
assert Path(paperbroker._core.__file__).resolve().is_relative_to(ROOT)
assert Path(ashare.paper.__file__).resolve().is_relative_to(ROOT)


def pytest_terminal_summary(terminalreporter):
    skipped = terminalreporter.stats.get("skipped", [])
    terminalreporter.write_line(f"paper fixture/test skips: {len(skipped)}")
    for item in skipped:
        terminalreporter.write_line(str(item.longrepr))
