"""离线重建 C1/C2/C3 共用合成夹具；期望数值另有手算断言。"""

import json
from pathlib import Path

import numpy as np

from ashare.data.crypto.events import normalize_event
from ashare.data.crypto.panel import build_frame
from ashare.data.crypto.timeutil import HOUR, millis

ROOT = Path(__file__).parent
START = millis("2024-06-28T00:00:00Z")
END = START + 12 * HOUR


def bar(t, price, volume=10):
    return dict(
        time=t,
        open=price,
        high=price + 2,
        low=price - 2,
        close=price + 1,
        volume=volume,
        quote_volume=volume * price,
        taker_buy_quote_volume=volume * price / 2,
    )


def inputs():
    profiles = []
    data = {}
    for symbol, contract, listed, unit in [
        ("PERPUSDT", "PERPETUAL", START + 2 * HOUR, 0.01),
        ("FUTUSDT_240628", "CURRENT_QUARTER", START - 30 * 24 * HOUR, 0.1),
    ]:
        name = "usdm:" + symbol
        profiles.append(
            dict(
                instrument=name,
                symbol=symbol,
                market="usdm",
                base_asset=symbol.split("USDT")[0],
                coin=symbol.split("USDT")[0],
                coin_units=1,
                quote_asset="USDT",
                margin_asset="USDT",
                underlying_type="COIN",
                contract_type=contract,
                listed_at=listed,
                source_url="fixture://synthetic-instrument",
                multiplier=unit,
                rules=[
                    dict(
                        qty_step=unit,
                        min_qty=unit * 2,
                        min_notional=5,
                        known_at=START - 24 * HOUR,
                        effective_at=START - 24 * HOUR,
                        source="snapshot",
                        source_url="fixture://synthetic-rule",
                    )
                ],
            )
        )
        offset, first = (100, 2) if contract == "PERPETUAL" else (200, 0)
        raw = [bar(START + i * HOUR, offset + i) for i in range(first, 13)]
        data[name] = dict(
            klines=raw,
            markPriceKlines=[bar(r["time"], r["open"] + 0.5, 0) for r in raw],
            indexPriceKlines=[bar(r["time"], r["open"] - 0.5, 0) for r in raw],
        )
        if contract == "PERPETUAL":
            data[name]["fundingRate"] = [
                dict(
                    time=START + i * HOUR,
                    raw_time=START + i * HOUR + 2,
                    interval=4,
                    rate=rate,
                )
                for i, rate in [(4, 0.0001), (8, -0.0002), (12, 0.0003)]
            ]
            data[name]["funding_prices"] = [
                dict(
                    fundingTime=r["raw_time"],
                    fundingRate=str(r["rate"]),
                    markPrice=str(101 + (r["time"] - START) / HOUR),
                )
                for r in data[name]["fundingRate"]
            ]
        else:
            data[name]["deliveries"] = [
                dict(deliveryTime=START + 8 * HOUR, deliveryPrice=199.0)
            ]
    events = [
        normalize_event(
            dict(
                event_id="fixture-quarter-expiry",
                instrument="usdm:FUTUSDT_240628",
                kind="delivery",
                known_at=START - 24 * HOUR,
                effective_at=START + 8 * HOUR,
                source_url="fixture://delivery-schedule",
                original_text="Synthetic quarterly contract expires at 2024-06-28 08:00 UTC.",
            )
        )
    ]
    return dict(profiles=profiles, data=data, events=events, start=START, end=END)


def main():
    fixture = inputs()
    (ROOT / "inputs.json").write_text(json.dumps(fixture, indent=2) + "\n")
    frame, _ = build_frame(**fixture)
    frame.write_parquet(ROOT / "panel.parquet")
    frame.write_csv(ROOT / "panel.csv")
    names = [p["instrument"] for p in fixture["profiles"]]
    fields = [
        "open",
        "high",
        "low",
        "close",
        "volume_engine",
        "settlement",
        "mark_open",
        "mark_high",
        "mark_low",
        "mark_close",
        "funding",
        "qty_step",
        "min_qty",
        "min_notional",
        "tradable",
        "expires",
        "quality",
    ]
    arrays = {
        f: np.column_stack(
            [
                frame.filter(frame["instrument"] == name).sort("time")[f].to_numpy()
                for name in names
            ]
        )
        for f in fields
    }
    np.savez(
        ROOT / "arrays.npz",
        time=np.arange(START, END, HOUR, dtype=np.int64),
        instruments=np.array(names),
        multiplier=np.array([0.01, 0.1]),
        **arrays,
    )


if __name__ == "__main__":
    main()
