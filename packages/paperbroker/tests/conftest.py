from pathlib import Path

import paperbroker
import paperbroker._core

ROOT = Path(__file__).resolve().parents[3]
assert Path(paperbroker.__file__).resolve().is_relative_to(ROOT)
assert Path(paperbroker._core.__file__).resolve().is_relative_to(ROOT)
