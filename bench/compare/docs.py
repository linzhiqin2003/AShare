"""Render documentation only on explicit request."""

import json

from run import ROOT, render

render(json.loads((ROOT / "bench/results/compare.json").read_text()))
