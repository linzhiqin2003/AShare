"""C2 联合验收发现的 pytest 收集命名冲突回归。"""

import subprocess
import sys
from pathlib import Path


def test_crypto_and_existing_data_modules_collect_together():
    root = Path(__file__).parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "test/crypto/test_panel.py",
            "test/crypto/test_verify.py",
            "test/data/test_panel.py",
            "test/data/test_verify.py",
        ],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
