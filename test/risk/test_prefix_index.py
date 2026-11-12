"""P-008：前缀索引（追加只顺序哈希新的一天）与只读窗口内的快照。"""

import numpy as np
import pytest

import ashare.risk.series as series
from ashare.risk import ModelCache
from ashare.risk.data import MemoryPanel
from ashare.risk.series import model_series
from test.risk.test_risk_review import config
from test.risk.test_series import source


def _same(a, b):
    assert a.date == b.date and a.metadata == b.metadata
    for name in ("B", "F", "D"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


def _spy(monkeypatch):
    """记录顺序链真正重算的日期和打开快照的次数。"""
    chained, loads = [], []
    parts = series._row_parts
    digests = series._row_digests
    inside = []

    def record_parts(source, external, day):
        if not inside:
            chained.append(day)
        return parts(source, external, day)

    def record_digests(*args, **kwargs):
        inside.append(1)
        try:
            return digests(*args, **kwargs)
        finally:
            inside.pop()

    original = ModelCache.load

    def record_load(self, key, **kwargs):
        loads.append(kwargs.get("day"))
        return original(self, key, **kwargs)

    monkeypatch.setattr(series, "_row_parts", record_parts)
    monkeypatch.setattr(series, "_row_digests", record_digests)
    monkeypatch.setattr(ModelCache, "load", record_load)
    return chained, loads


def test_append_hashes_only_the_new_day_and_reads_only_the_window(
    tmp_path, monkeypatch
):
    p, c = source(60, 40), config()  # history=20
    full = list(model_series(p, config=c))
    list(model_series(p, config=c, cache=tmp_path, end=59))
    assert (tmp_path / ".prefix-index").exists()
    chained, loads = _spy(monkeypatch)
    appended = list(model_series(p, config=c, cache=tmp_path, start=59))
    _same(appended[0], full[-1])
    assert chained == [59]
    assert min(d for d in loads if d is not None) >= 59 - 20
    assert len(loads) <= 20 + 3
    chained.clear()
    loads.clear()
    read = list(model_series(p, config=c, cache=tmp_path, start=59))
    _same(read[0], full[-1])
    assert chained == [] and loads == [59]


def test_history_rewrite_is_detected_through_the_index(tmp_path):
    p, c = source(50, 40), config()
    list(model_series(p, config=c, cache=tmp_path))
    fields = {k: v.copy() for k, v in p.fields.items()}
    fields["close"][7, 3] *= 1.01  # 改历史但不改 version
    changed = MemoryPanel(p.dates, p.symbols, p.version, fields)
    expected = list(model_series(changed, config=c))
    actual = list(model_series(changed, config=c, cache=tmp_path, start=45))
    for a, b in zip(actual, expected[45:], strict=True):
        _same(a, b)
    assert (
        actual[-1].metadata["prefix_hash"]
        != list(model_series(p, config=c))[-1].metadata["prefix_hash"]
    )


@pytest.mark.parametrize("damage", ("garbage", "checksum", "short"))
def test_damaged_index_is_ignored(tmp_path, damage):
    p, c = source(40, 30), config()
    expected = list(model_series(p, config=c))
    list(model_series(p, config=c, cache=tmp_path, end=39))
    path = tmp_path / ".prefix-index"
    if damage == "garbage":
        path.write_bytes(b"broken")
    else:
        with np.load(path) as data:
            rows, prefixes, checksum = (
                data["rows"].copy(),
                data["prefixes"].copy(),
                str(data["checksum"]),
            )
        if damage == "checksum":
            prefixes[-1, 0] ^= 1  # 链摘要被改而输入摘要没变：校验和不符
        else:
            rows, prefixes = rows[:10], prefixes[:10]
            checksum = "0" * 64
        with path.open("wb") as stream:
            np.savez(
                stream, rows=rows, prefixes=prefixes, checksum=np.asarray(checksum)
            )
    actual = list(model_series(p, config=c, cache=tmp_path, start=38))
    for a, b in zip(actual, expected[38:], strict=True):
        _same(a, b)


@pytest.mark.parametrize("victim_day", (45, 55))
def test_damaged_snapshot_inside_the_window_moves_the_start(
    tmp_path, monkeypatch, victim_day
):
    """45：预读阶段发现；55：已经交出 50..54 之后才发现，从 55 起重新计划。"""
    p, c = source(60, 40), config()
    full = list(model_series(p, config=c))
    list(model_series(p, config=c, cache=tmp_path, end=59))
    victim = next(tmp_path.glob(f"{victim_day:06d}-*.npz"))
    victim.write_bytes(b"truncated")
    chained, loads = _spy(monkeypatch)
    passes = []
    iterate = series._iterate

    def count(*args, **kwargs):
        passes.append(1)
        return iterate(*args, **kwargs)

    monkeypatch.setattr(series, "_iterate", count)
    appended = list(model_series(p, config=c, cache=tmp_path, start=50))
    # 预读范围内的损坏在计划阶段就处理掉；交出之后才发现的才需要第二遍。
    assert len(passes) == (1 if victim_day < 50 else 2)
    for a, b in zip(appended, full[50:], strict=True):
        _same(a, b)
    assert [m.date for m in appended] == [m.date for m in full[50:]]
    assert min(d for d in loads if d is not None) <= victim_day - 20


def test_shorter_run_keeps_the_longer_index(tmp_path):
    p, c = source(45, 30), config()
    list(model_series(p, config=c, cache=tmp_path))
    rows = ModelCache(tmp_path).load_index()[0]
    list(model_series(p, config=c, cache=tmp_path, end=30, start=29))
    assert len(ModelCache(tmp_path).load_index()[0]) == len(rows) == 45
