import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parents[1] / "fixtures" / "crypto"


@pytest.fixture
def fixture():
    return json.loads((FIXTURES / "inputs.json").read_text())


@pytest.fixture
def raw_dir():
    return FIXTURES / "raw"
