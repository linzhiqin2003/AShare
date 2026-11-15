"""从原始 ZIP/CSV 独立复算 C6 片段；输出误差与样本数，不输出行情。"""

import argparse
import csv
import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np

from ashare.data import crypto
from ashare.data.crypto.signals import SignalDataset
from ashare.data.crypto.timeutil import DAY, HOUR, millis
from ashare.factor.crypto import crypto_factors


def decode(root, references, pattern, funding=False):
    rows = {}
    count = 0
    for key, ref in references.items():
        if pattern not in key or not key.endswith(".zip"):
            continue
        # 独立从原始 bytes 解压；不调用 formats/read_series。
        payload = (root / "objects" / ref["sha256"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == ref["sha256"] == ref["checksum"]
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            names = [n for n in z.namelist() if n.endswith(".csv")]
            assert len(names) == 1
            records = list(
                csv.reader(io.StringIO(z.read(names[0]).decode("utf-8-sig")))
            )
        if not records[0][0].isdigit():
            records = records[1:]
        for r in records:
            stamp = int(r[0])
            if funding:
                t = round(stamp / 3600000) * 3600000
                assert abs(t - stamp) <= 1000
                value = float(r[2])
            else:
                scale = 1000 if stamp >= 10**14 else 1
                t = stamp // scale
                value = float(r[4])
            if t in rows:
                assert rows[t] == value
            rows[t] = value
        count += 1
    return rows, count


def run(root):
    root = Path(root)
    ds = crypto.open(root)
    sig = SignalDataset(root) if (root / "signals/latest.json").exists() else None
    start, end = millis("2024-06-01"), millis("2024-06-08")
    f = crypto_factors(ds, sig, ["BTCUSDT", "ETHUSDT"], start, end)
    frozen = json.loads(
        (root / "raw_versions" / f"{ds.meta['raw_version']}.json").read_text()
    )
    # 月/日补档只取被面板冻结引用的键，防止后来修订混进来。
    inputs = ds.inputs()
    rows = []
    for j, symbol in enumerate(("BTCUSDT", "ETHUSDT")):
        refs = {
            key: value
            for key, value in frozen.items()
            if any(f"{month}.zip" in key for month in ("2024-05", "2024-06"))
        }
        spot, _ = decode(root, refs, "/spot/monthly/klines/" + symbol + "/")
        perp, _ = decode(root, refs, "/futures/um/monthly/klines/" + symbol + "/")
        rates, _ = decode(root, refs, "/fundingRate/" + symbol + "/", funding=True)
        at = f["premium_1d"][0]
        expected = np.array(
            [
                np.mean(
                    [perp[t] / spot[t] - 1 for t in range(int(a) - DAY, int(a), HOUR)]
                )
                for a in at
            ]
        )
        error = float(np.max(np.abs(expected - f["premium_1d"][1][:, j])))
        assert error < 1e-12
        funding = np.array(
            [
                8760
                * sum(rate for t, rate in rates.items() if a - 7 * DAY < t <= a)
                / (7 * 24)
                for a in at
            ]
        )
        ferr = float(np.max(np.abs(funding - f["funding_mean_7d"][1][:, j])))
        assert ferr < 1e-12
        item = dict(
            symbol=symbol,
            hours=len(at),
            premium_max_error=error,
            funding_mean_max_error=ferr,
        )
        if sig is not None:
            metadata = sig.metadata
            sin = sig.inputs()
            errors = []
            cache = {}
            for i, a in enumerate(at):
                candidates = [
                    m
                    for m in metadata
                    if m["pair"] == symbol
                    and m["listed_at"] is not None
                    and a - m["listed_at"] >= 7 * DAY
                    and m["delivery_at"] - a > 336 * HOUR
                ]
                chosen = min(candidates, key=lambda m: m["delivery_at"])
                assert f.selected_contract[1][i, j] == chosen["symbol"]
                if chosen["symbol"] not in cache:
                    cache[chosen["symbol"]] = decode(
                        root / "signals/raw",
                        sin["references"],
                        "/klines/" + chosen["symbol"] + "/",
                    )[0]
                close = cache[chosen["symbol"]][int(a) - HOUR]
                expected = (
                    (close / spot[int(a) - HOUR] - 1)
                    * 8760
                    / ((chosen["delivery_at"] - a) / HOUR)
                )
                if np.isfinite(f["basis_q"][1][i, j]):
                    errors.append(abs(expected - f["basis_q"][1][i, j]))
            assert errors and max(errors) < 1e-12
            item.update(basis_max_error=max(errors), basis_hours=len(errors))
        else:
            item["basis_status"] = "strict_signal_dataset_unavailable"
        # 证明面板资金费不是本复算的输入：只核对原始归档键身份。
        assert inputs["data"]["usdm:" + symbol]["fundingRate"]
        rows.append(item)
    return dict(
        data_version=ds.version,
        signal_version=sig.version if sig else None,
        window=["2024-06-01", "2024-06-08"],
        source="independent raw ZIP/CSV decode",
        results=rows,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="tmp/crypto")
    parser.add_argument(
        "--output", default="bench/results/crypto_factors_hand_check.json"
    )
    args = parser.parse_args()
    result = run(args.data)
    Path(args.output).write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
