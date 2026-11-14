"""离线生成公开格式夹具：只用固定常数，绝不读取真实归档、行情或公告。"""

import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).parent
HOUR = 3_600_000
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
SPECS = {
    "spot_ms.csv": ("spot", "klines", "2024-12-01", False, 1),
    "spot_us.csv": ("spot", "klines", "2025-01-01", False, 1000),
    "usdm_header.csv": ("usdm", "klines", "2022-01-01", True, 1),
    "usdm_no_header.csv": ("usdm", "klines", "2021-12-01", False, 1),
    "mark.csv": ("usdm", "markPriceKlines", "2024-05-01", True, 1),
    "funding.csv": ("usdm", "fundingRate", "2024-05-01", True, 1),
}


def millis(date):
    return int(datetime.fromisoformat(date).replace(tzinfo=UTC).timestamp() * 1000)


def bar(index, t, market, dataset, scale):
    base = Decimal(100 + index)
    prices = [base + Decimal("0.25"), base + 3, base - Decimal("2.5"), base + 1]
    is_mark = dataset == "markPriceKlines"
    price_places, volume_places, quote_places = (
        (8, 8, 8) if market == "spot" else (2, 3, 5)
    )
    if is_mark:
        price_places = 8
    prices = [
        format(
            p,
            ".0f"
            if is_mark and (index, c) in MARK_INTEGER_PRICES
            else f".{price_places}f",
        )
        for c, p in enumerate(prices, 1)
    ]
    volume = Decimal(10) + Decimal(index) / 1000
    taker = Decimal(5) + Decimal(index) / 1000
    quantity = "0" if is_mark else f"{volume:.{volume_places}f}"
    amount = "0" if is_mark else f"{base * volume:.{quote_places}f}"
    taker_quantity = "0" if is_mark else f"{taker:.{volume_places}f}"
    taker_amount = "0" if is_mark else f"{base * taker:.{quote_places}f}"
    return [
        str(t * scale),
        *prices,
        quantity,
        str((t + HOUR) * scale - 1),
        amount,
        "3600" if is_mark else str(index + 1),
        taker_quantity,
        taker_amount,
        "0",
    ]


def write_csvs():
    folder = ROOT / "raw"
    folder.mkdir(exist_ok=True)
    provenance = {}
    for name, (market, dataset, date, header, scale) in SPECS.items():
        funding = dataset == "fundingRate"
        columns = FUNDING_COLUMNS if funding else KLINE_COLUMNS
        rows = [columns] if header else []
        for i in range(20 - int(header)):
            if funding:
                jitter = (0, 1, 5, 2, 3)[i % 5]
                rate = ("0.00012345", "-0.00023456", "0.00034567")[i % 3]
                rows.append([str(millis(date) + i * 8 * HOUR + jitter), "8", rate])
            else:
                rows.append(bar(i, millis(date) + i * HOUR, market, dataset, scale))
        (folder / name).write_bytes(
            ("\n".join(",".join(r) for r in rows) + "\n").encode("ascii")
        )
        prefix = "data/spot" if market == "spot" else "data/futures/um"
        suffix = (
            "{symbol}/{symbol}-fundingRate-{YYYY-MM}.zip"
            if funding
            else "{symbol}/1h/{symbol}-1h-{YYYY-MM}.zip"
        )
        provenance[name] = dict(
            synthetic=True,
            generation="generate_formats.py: independent constants; no real-data input or transformation",
            dataset=f"{market}/{dataset}",
            filename_pattern=f"{prefix}/monthly/{dataset}/{suffix}",
            format_reference="docs/markets/BINANCE.md; C2 offline format inspection, summarized in docs/C2_NOTES.md",
            columns=columns,
            header=header,
            timestamp_unit="us" if scale == 1000 else "ms",
            timestamp_digits=16 if scale == 1000 else 13,
            encoding="ASCII (UTF-8 compatible), no BOM",
            line_ending="LF, including final line",
            data_rows=20 - int(header),
        )
        if funding:
            provenance[name].update(
                decimal_places=dict(last_funding_rate=8),
                funding_interval_hours=8,
                synthetic_jitter_ms=[0, 1, 5, 2, 3],
            )
        elif dataset == "markPriceKlines":
            provenance[name].update(
                decimal_places=dict(
                    ohlc=8, integer_price_cells=sorted(MARK_INTEGER_PRICES)
                ),
                integer_price_cells_indexing="zero-based data row, zero-based CSV column",
                zero_integer_columns=[
                    "volume",
                    "quote_volume",
                    "taker_buy_volume",
                    "taker_buy_quote_volume",
                    "ignore",
                ],
                count=3600,
            )
        else:
            provenance[name]["decimal_places"] = dict(
                ohlc=8 if market == "spot" else 2,
                volume_and_taker_volume=8 if market == "spot" else 3,
                quote_and_taker_quote=8 if market == "spot" else 5,
            )
    (folder / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def write_announcement():
    def paragraph(*texts):
        return dict(
            node="element", tag="p", child=[dict(node="text", text=t) for t in texts]
        )

    body = dict(
        node="root",
        child=[
            paragraph(
                "Invented example for parser testing. These assets and actions are fictional."
            ),
            paragraph(
                "In this exercise, Binance will delist and cease trading on all spot trading pairs at ",
                "2024-06-17 03:00 (UTC)",
                ".",
            ),
            paragraph(
                "The exact trading pairs being removed are: FAKE/USDT, FAKE/BTC, DUMMY/USDT, MOCK/USDT, SAMPLE/USDT."
            ),
            paragraph(
                "For the mock contract, we will close all positions and conduct an automatic settlement on FAKEUSDT at ",
                "2024-06-11 09:00 (UTC)",
                ". This sentence exercises a separate lifecycle.",
            ),
            paragraph(
                "This invented notice was updated on 2024-06-03 for date-only parser coverage."
            ),
        ],
    )
    article = dict(
        code="synthetic-fake-delisting",
        title="Binance Will Delist FAKE, DUMMY, MOCK, SAMPLE on 2024-06-17",
        publishDate=millis("2024-06-03T01:02:03.004"),
        lastUpdateTime=0,
        body=json.dumps(body, ensure_ascii=False, separators=(",", ":")),
        fixture_provenance=dict(
            synthetic=True,
            generation="generate_formats.py: independently authored text and fictional symbols",
            legacy_filename="waves.json is retained only as a fixture path, not as a claim of real content",
            format_features=[
                "JSON string body with node/child/text tree",
                "publishDate milliseconds",
                "lastUpdateTime zero sentinel",
                "was updated on YYYY-MM-DD",
                "spot delist and perpetual settlement grammar",
            ],
        ),
    )
    folder = ROOT / "announcements"
    folder.mkdir(exist_ok=True)
    (folder / "waves.json").write_text(
        json.dumps(article, ensure_ascii=False, indent=2) + "\n"
    )


if __name__ == "__main__":
    write_csvs()
    write_announcement()
