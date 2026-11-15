"""C1 algebra checks: 2021-05-19 BTC marks, an hourly funding day, delivery.

BTC source is the official Binance archive, downloaded into memory only.
Execution at mark open, MM=.004, clearance=.02 (announced for 2021-02-07..05-20) and fee=.0004 are explicit
model assumptions, not claims about that historical user's account terms.
"""

import csv
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import panelbt as bt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from test.build_guard import assert_current_build  # noqa: E402
from test.test_crypto import ZERO, scenario  # noqa: E402


def main():
    source = assert_current_build()
    url = "https://data.binance.vision/data/futures/um/daily/markPriceKlines/BTCUSDT/1h/BTCUSDT-1h-2021-05-19.zip"
    raw = urllib.request.urlopen(url, timeout=30).read()
    expected = (
        urllib.request.urlopen(url + ".CHECKSUM", timeout=30).read().decode().split()[0]
    )
    checksum = hashlib.sha256(raw).hexdigest()
    assert checksum == expected
    archive = zipfile.ZipFile(io.BytesIO(raw))
    records = list(
        csv.reader(io.StringIO(archive.read(archive.namelist()[0]).decode()))
    )
    prices = np.array([[float(x) for x in row[:5]] for row in records])
    assert len(prices) == 24
    entry = prices[0, 1]
    initial = entry * 0.1 + entry * 0.0004
    target = np.full((24, 1), np.nan)
    target[0] = 1
    args = scenario(
        prices[:, 1:2],
        target,
        cash=initial,
        high=prices[:, 2:3],
        low=prices[:, 3:4],
        settlement=prices[:, 4:5],
        initial=0.1,
        maint=0.004,
        clearance=0.02,
        fee_rate=0.0004,
    )
    result = bt.run(*args)
    trigger = entry * 0.9 / (1 - 0.004)
    first = next(i for i in range(24) if prices[i, 3] <= trigger)
    assert prices[first, 1] > trigger  # intraday crossing, not an opening gap
    expected_remainder = trigger * (0.004 - 0.0004)
    expected_clearance = min(0.02 * trigger, expected_remainder)
    assert result.liquidation_events[0].day == first
    np.testing.assert_allclose(result.trades[-1].price, trigger, rtol=1e-12)
    np.testing.assert_allclose(
        sum(result.clearance_fee), expected_clearance, rtol=1e-12
    )
    assert result.cash[-1] == 0 and not result.bankruptcy_events
    funding = np.zeros((24, 1))
    funding[[7, 15, 23]] = 0.1
    targets = np.full((24, 1), np.nan)
    targets[0] = 2
    funded = bt.run(*scenario(np.full((24, 1), 100), targets, funding=funding))
    np.testing.assert_allclose(funded.cash[-1], 99.4, rtol=1e-12)
    delivered = bt.run(
        *scenario(
            [[100], [100]],
            [[2], [np.nan]],
            settlement=[[100], [99]],
            low=[[100], [99]],
            expiry=[1],
            delivery_fee_rate=0.0004,
            costs=ZERO,
        )
    )
    np.testing.assert_allclose(delivered.cash[-1], 97.9208, rtol=1e-12)
    report = dict(
        kernel_source=source,
        btc_2021_05_19=dict(
            source=url,
            source_sha256=checksum,
            entry=float(entry),
            minimum_mark=float(prices[:, 3].min()),
            liquidation_utc_hour=first,
            assumptions=dict(
                exec_price="mark_open",
                initial_rate=0.1,
                maintenance=0.004,
                fee_rate=0.0004,
                clearance_rate=0.02,
                funding="disabled",
            ),
            calculated_trigger=float(trigger),
            actual_trigger=result.trades[-1].price,
            clearance=float(expected_clearance),
            ending_cash=float(result.cash[-1]),
        ),
        synthetic_funding_day=dict(
            contracts=2, fee_per_contract=0.1, events=3, ending_cash=99.4
        ),
        synthetic_delivery=dict(
            contracts=2, basis=100, delivery=99, fee=0.0792, ending_cash=97.9208
        ),
    )
    (ROOT / "bench/results/crypto_hand_check.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
