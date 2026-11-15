import pytest


@pytest.fixture
def roots(tmp_path, monkeypatch):
    root, external = tmp_path / "root", tmp_path / "disk"
    root.mkdir()
    external.mkdir()
    monkeypatch.setenv("ASHARE_DATA_DIR", str(root))
    for key in (
        "ASHARE_ML_DIR",
        "ASHARE_MINING_DIR",
        "ASHARE_PLATFORM_DIR",
        "ASHARE_PAPER_DIR",
        "CRYPTO_DATA_ROOT",
    ):
        monkeypatch.delenv(key, raising=False)
    return root, external
