"""Digest of full run results over the random corpora (bitwise regression aid).

Usage: python bench/step_digest.py OUT.json   (run once per build and diff).
Every float array is hashed by its raw bytes, so any bit change shows up.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import panelbt as bt  # noqa: E402

from test.digest import corpora, digest  # noqa: E402


def main():
    count = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    out = {name: digest(bt.run(*args)) for name, args in corpora(count)}
    Path(sys.argv[1]).write_text(json.dumps(out, indent=0) + "\n")
    print(len(out), "digests")


if __name__ == "__main__":
    main()
