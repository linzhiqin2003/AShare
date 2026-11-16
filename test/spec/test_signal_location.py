"""D2 任务书已知 bug：季度信号 Python 入口必须遵循 D1 位置解析。"""

import json

import pytest

from ashare.data.crypto.signals import SignalDataset, build_signals, sync_signals
from ashare.paths import DataLocationError
from test.crypto.test_signals_c6 import SyntheticClient


def location(tmp_path, monkeypatch):
    root, disk = tmp_path / "root", tmp_path / "disk"
    disk.mkdir()
    pointer = root / "crypto" / "LOCATION.json"
    pointer.parent.mkdir(parents=True)
    pointer.write_text(
        json.dumps(
            dict(path=str(disk), moved_at="2026-11-15T00:00:00Z", files=0, bytes=0)
        )
    )
    monkeypatch.setenv("ASHARE_DATA_DIR", str(root))
    monkeypatch.delenv("CRYPTO_DATA_ROOT", raising=False)
    return disk


def test_signal_location_pointer_read_and_build(tmp_path, monkeypatch):
    disk = location(tmp_path, monkeypatch)
    original = sync_signals(
        disk, start="2024-01-01", end="2024-01-01T01:00:00Z", client=SyntheticClient()
    )
    assert SignalDataset(None).read().equals(original.read())
    assert build_signals(None).read().equals(original.read())


@pytest.mark.parametrize("entry", [SignalDataset, build_signals])
def test_signal_location_missing_disk_does_not_fallback(tmp_path, monkeypatch, entry):
    disk = location(tmp_path, monkeypatch)
    disk.rmdir()
    with pytest.raises(DataLocationError, match="目标目录不可用"):
        entry(None)
