"""Claude C2 返工 R2：公开夹具须为合成值，同时保留逐列的真实格式约束。"""

import csv
import json
import re
from pathlib import Path

import pytest

KLINE_COLUMNS = (
    "open_time,open,high,low,close,volume,close_time,quote_volume,count,"
    "taker_buy_volume,taker_buy_quote_volume,ignore"
).split(",")
FUNDING_COLUMNS = "calc_time,funding_interval_hours,last_funding_rate".split(",")
MARK_INTEGER_PRICES = {
    (1, 2),
    (7, 4),
    (8, 4),
    (11, 2),
    (12, 4),
    (13, 1),
    (13, 2),
    (14, 4),
    (17, 4),
}


@pytest.mark.parametrize(
    "name,header,digits,decimals",
    [
        ("spot_ms.csv", False, 13, [None, 8, 8, 8, 8, 8, None, 8, None, 8, 8, None]),
        ("spot_us.csv", False, 16, [None, 8, 8, 8, 8, 8, None, 8, None, 8, 8, None]),
        ("usdm_header.csv", True, 13, [None, 2, 2, 2, 2, 3, None, 5, None, 3, 5, None]),
        (
            "usdm_no_header.csv",
            False,
            13,
            [None, 2, 2, 2, 2, 3, None, 5, None, 3, 5, None],
        ),
        (
            "mark.csv",
            True,
            13,
            [None, 8, 8, 8, 8, None, None, None, None, None, None, None],
        ),
        ("funding.csv", True, 13, [None, None, 8]),
    ],
)
def test_synthetic_csv_layout_and_values(raw_dir, name, header, digits, decimals):
    meta = json.loads((raw_dir / "provenance.json").read_text())[name]
    assert meta["synthetic"] is True
    assert (
        not {"archive_sha256", "sample_sha256", "checksum_text", "selection"}
        & meta.keys()
    )
    payload = (raw_dir / name).read_bytes()
    assert payload.endswith(b"\n") and b"\r" not in payload
    assert not payload.startswith(b"\xef\xbb\xbf")
    assert len(payload.splitlines()) == 20
    rows = list(csv.reader(payload.decode("ascii").splitlines()))
    columns = FUNDING_COLUMNS if name == "funding.csv" else KLINE_COLUMNS
    assert meta["columns"] == columns
    if header:
        assert rows.pop(0) == columns
    else:
        assert rows[0][0].isdigit()
    for row_number, row in enumerate(rows):
        assert len(row) == len(columns)
        assert re.fullmatch(rf"\d{{{digits}}}", row[0])
        for column, (value, places) in enumerate(zip(row, decimals, strict=True)):
            if name == "mark.csv" and (row_number, column) in MARK_INTEGER_PRICES:
                places = None
            pattern = r"-?\d+" if places is None else rf"-?\d+\.\d{{{places}}}"
            assert re.fullmatch(pattern, value), (name, row_number, column)
        if name == "funding.csv":
            assert row[1] == "8"
            assert int(row[0]) % (8 * 3_600_000) in {0, 1, 2, 3, 5}
        else:
            assert re.fullmatch(rf"\d{{{digits}}}", row[6])
            hour = 3_600_000 * (1000 if digits == 16 else 1)
            assert int(row[0]) % hour == 0
            assert int(row[6]) == int(row[0]) + hour - 1
            # 原始真实价格为数万；合成值由独立常数生成，不能误换回原始行情。
            assert all(90 <= float(row[c]) <= 130 for c in (1, 2, 3, 4))
            assert 0 <= float(row[5]) < 20
            assert 0 <= float(row[7]) < 3000
            if name == "mark.csv":
                assert [row[c] for c in (5, 7, 9, 10, 11)] == ["0"] * 5
                assert row[8] == "3600"


def test_announcement_fixture_is_authored_synthetic_text():
    root = Path(__file__).parents[1] / "fixtures/crypto/announcements"
    a = json.loads((root / "waves.json").read_text())
    assert a["fixture_provenance"]["synthetic"] is True
    assert a["code"] == "synthetic-fake-delisting"
    assert "FAKE" in a["title"]
    body = json.loads(a["body"])
    assert body["node"] == "root" and body["child"]
    assert len(a["body"]) < 2500
    assert "WAVES" not in a["body"]
    assert "Fellow Binancians" not in a["body"]
    assert "raw_response_sha256" not in a["fixture_provenance"]
