"""Execute all README examples with an isolated installed-wheel interpreter."""

import json
import re
from pathlib import Path

import numpy as np
import panelbt
from panelbt.reference import run

assert "site-packages" in panelbt.__file__
root = Path(__file__).resolve().parents[1]
report = {}
for filename in ("README.md", "README.zh-CN.md"):
    path = root / "packages/panelbt" / filename
    namespace = {}
    blocks = re.findall(r"```python\n(.*?)```", path.read_text(), flags=re.S)
    for source in blocks:
        exec(compile(source, filename, "exec"), namespace)
    for market in ("hk", "us"):
        profile = namespace[market + "_style"]
        expected = run(
            profile.market(namespace["prices"], namespace["prices"]),
            namespace["targets"],
            namespace[market + "_costs"],
            namespace[market + "_config"],
        )
        np.testing.assert_allclose(
            namespace[market + "_result"].nav, expected.nav, rtol=1e-9, atol=1e-9
        )
    report[filename] = {
        "blocks_passed": len(blocks),
        "hk_nav": namespace["hk_result"].nav.tolist(),
        "us_nav": namespace["us_result"].nav.tolist(),
    }
print(json.dumps(report, ensure_ascii=False))
