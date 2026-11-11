from pathlib import Path

import panelbt

assert (
    Path(panelbt.__file__).resolve().is_relative_to(Path(__file__).resolve().parents[1])
)
