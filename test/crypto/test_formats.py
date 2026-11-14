import pytest

from ashare.data.crypto.formats import parse_funding, parse_klines
from ashare.data.crypto.timeutil import HOUR, funding_hour, millis


@pytest.mark.parametrize(
    "name,market,start,rows",
    [
        ("spot_ms.csv", "spot", "2024-12-01", 20),
        ("spot_us.csv", "spot", "2025-01-01", 20),
        ("usdm_header.csv", "usdm", "2022-01-01", 19),
        ("usdm_no_header.csv", "usdm", "2021-12-01", 20),
        ("mark.csv", "usdm", "2024-05-01", 19),
    ],
)
def test_synthetic_archive_formats(raw_dir, name, market, start, rows):
    """来源：R1-16 格式反例；R2 改为同格式的独立合成值。"""
    parsed = parse_klines((raw_dir / name).read_bytes(), market)
    assert len(parsed) == rows
    assert parsed[0]["time"] == millis(start)
    assert parsed[1]["time"] - parsed[0]["time"] == HOUR


def test_synthetic_funding_format_and_jitter(raw_dir):
    rows = parse_funding((raw_dir / "funding.csv").read_bytes())
    assert len(rows) == 19
    assert rows[0]["time"] == millis("2024-05-01")
    assert {r["interval"] for r in rows} == {8}
    assert {r["raw_time"] - r["time"] for r in rows} == {0, 1, 2, 3, 5}
    assert {r["rate"] for r in rows} == {0.00012345, -0.00023456, 0.00034567}
    assert all(
        b["time"] - a["time"] == 8 * HOUR for a, b in zip(rows, rows[1:], strict=False)
    )


def test_units_range_header_and_duplicates(raw_dir):
    payload = (raw_dir / "spot_us.csv").read_bytes()
    with pytest.raises(ValueError, match="毫秒"):
        parse_klines(payload, "usdm")
    with pytest.raises(ValueError, match="文件范围"):
        parse_klines(payload, "spot", start="2024-01-01", end="2024-02-01")
    with pytest.raises(ValueError, match="表头"):
        parse_klines(b"wrong,header\n" + payload, "spot")
    with pytest.raises(ValueError, match="重复"):
        parse_klines(payload + payload, "spot")
    with pytest.raises(ValueError, match="混杂"):
        parse_klines(payload + (raw_dir / "spot_ms.csv").read_bytes(), "spot")


@pytest.mark.parametrize("jitter", [-1000, -2, 0, 3, 1000])
def test_funding_jitter(jitter):
    base = millis("2024-05-01T08:00:00Z")
    assert funding_hour(base + jitter) == base


def test_funding_jitter_and_grid_rejected():
    base = millis("2024-05-01T08:00:00Z")
    with pytest.raises(ValueError, match="1 秒"):
        funding_hour(base + 1001)
    with pytest.raises(ValueError, match="整点"):
        parse_funding(f"{base + HOUR},8,0.0001\n".encode())
