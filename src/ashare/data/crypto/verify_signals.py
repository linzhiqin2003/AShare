"""季度信号库独立核验：不调用构建器、生产解析器或质量判据。"""

import csv
import hashlib
import io
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import polars as pl


def verify_signals(root, version=None):
    root = Path(root)
    version = (
        version or json.loads((root / "signals/latest.json").read_text())["version"]
    )
    folder = root / "signals/versions" / version
    payload = (folder / "inputs.json").read_bytes()
    inputs = json.loads(payload)
    errors = []
    if hashlib.sha256(payload).hexdigest() != version:
        errors.append("version")
    meta = json.loads((folder / "meta.json").read_text())
    if meta["metadata"] != inputs["metadata"]:
        errors.append("metadata_digest")
    original = {}
    for key, ref in inputs["references"].items():
        data = (root / "signals/raw/objects" / ref["sha256"]).read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        check = ref["checksum_text"].split()
        if (
            sha != ref["sha256"]
            or sha != ref["checksum"]
            or check != [sha, key.rsplit("/", 1)[1]]
        ):
            errors.append("checksum:" + key)
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            if z.testzip() is not None:
                errors.append("crc:" + key)
            files = [n for n in z.namelist() if n.endswith(".csv")]
            if len(files) != 1:
                raise ValueError("季度 ZIP 必须一个 CSV")
            values = list(csv.reader(io.StringIO(z.read(files[0]).decode("utf-8-sig"))))
        if not values[0][0].isdigit():
            values = values[1:]
        fields = {
            "open": 1,
            "high": 2,
            "low": 3,
            "close": 4,
            "volume": 5,
            "quote_volume": 7,
            "taker_buy_volume": 9,
            "taker_buy_quote_volume": 10,
        }
        decoded = []
        for row in values:
            t, close = int(row[0]), int(row[6])
            if t % 3600000 or close > t + 3600999:
                errors.append("timestamp:" + key)
            q = (
                "invalid_close_time"
                if close < t
                else "partial_halt"
                if close < t + 3598999
                else "normal"
            )
            decoded.append(
                dict(
                    time=t,
                    close_time=close,
                    time_quality=q,
                    count=int(row[8]),
                    **{f: float(row[i]) for f, i in fields.items()},
                    source="archive",
                    source_key=key,
                )
            )
        original[key] = decoded
    frame = pl.read_parquet(folder / "hours.parquet")
    if (
        hashlib.sha256((folder / "hours.parquet").read_bytes()).hexdigest()
        != inputs["frame_sha256"]
    ):
        errors.append("frame_digest")
    checked = 0
    for m in inputs["metadata"]:
        symbol = m["symbol"]
        raw = inputs["raw"][symbol]
        independent = sorted(
            (r for k in m["archives"] for r in original.get(k, [])),
            key=lambda r: r["time"],
        )
        if inputs["references"] and independent != raw:
            errors.append("decode:" + symbol)
        expiry = int(
            datetime.strptime(symbol[-6:], "%y%m%d")
            .replace(tzinfo=UTC, hour=8)
            .timestamp()
            * 1000
        )
        if m["delivery_at"] != expiry:
            errors.append("delivery:" + symbol)
        after = [r for r in raw if r["close_time"] >= expiry]
        if m.get("post_delivery_excluded_rows", 0) != len(after):
            errors.append("post_delivery_count:" + symbol)
        recorded = [
            e
            for e in inputs["exceptions"]
            if e.get("kind") == "post_delivery" and e["symbol"] == symbol
        ]
        if after:
            if (
                len(recorded) != 1
                or recorded[0].get("count") != len(after)
                or recorded[0]["start"] != min(r["time"] for r in after)
                or recorded[0]["end"] != max(r["time"] for r in after) + 3600000
                or not recorded[0].get("reason")
            ):
                errors.append("post_delivery_exception:" + symbol)
        elif recorded:
            errors.append("post_delivery_exception:" + symbol)
        last_trade = max((r["time"] for r in raw if r["volume"] > 0), default=None)
        matured = (
            max(inputs["end"], max((r["close_time"] + 1 for r in raw), default=0))
            >= expiry
        )
        status = (
            "excluded_no_listing"
            if m["listed_at"] is None
            else "pass"
            if matured
            else "pending_not_expired"
        )
        if m.get("last_trade_at") != last_trade or m.get("delivery_check") != status:
            errors.append("delivery_audit:" + symbol)
        if (last_trade is not None and last_trade > expiry) or (
            m["listed_at"] is not None
            and matured
            and (last_trade is None or last_trade < expiry - 86400000)
        ):
            errors.append("last_trade:" + symbol)
        ev = m["evidence"]
        if ev.get("deliveryDate") is not None and int(ev["deliveryDate"]) != expiry:
            errors.append("delivery_evidence:" + symbol)
        earliest = min((r["time"] for r in raw), default=None)
        expected_listed = ev.get("onboardDate", earliest)
        if isinstance(expected_listed, str):
            expected_listed = int(
                datetime.fromisoformat(expected_listed.replace("Z", "+00:00"))
                .replace(tzinfo=UTC)
                .timestamp()
                * 1000
            )
        if expected_listed != m["listed_at"] or (
            earliest is not None and expected_listed > earliest
        ):
            errors.append("listing:" + symbol)
        if m["listed_at"] is None:
            continue
        by_time = {r["time"]: r for r in raw}
        observed = {
            r["time"]: r
            for r in frame.filter(pl.col("symbol") == symbol).iter_rows(named=True)
        }
        lo = max(inputs["start"], (m["listed_at"] + 3599999) // 3600000 * 3600000)
        hi = min(inputs["end"], expiry)
        excluded_times = {r["time"] for r in after}
        if sorted(observed) != [
            t for t in range(lo, hi, 3600000) if t not in excluded_times
        ]:
            errors.append("continuity:" + symbol)
        for t, row in observed.items():
            checked += 1
            value = by_time.get(t)
            if value is None:
                allowed = any(
                    e.get("kind") != "post_delivery"
                    and e["symbol"] == symbol
                    and e["start"] <= t < e["end"]
                    and e.get("reason")
                    for e in inputs["exceptions"]
                )
                if not allowed or row["quality"] != 1 or row["close"] is not None:
                    errors.append("gap:" + symbol)
                continue
            quality = (
                2048
                if value["close_time"] < t
                else 512
                if value["close_time"] < t + 3598999
                else 0
            )
            if min(value[f] for f in ("volume", "quote_volume", "count")) <= 0:
                quality |= 2
            if (
                t - 3600000 in by_time
                and by_time[t - 3600000]["close"] == value["close"]
            ):
                quality |= 4
            if row["quality"] != quality or row["close"] != value["close"]:
                errors.append("quality_or_price:" + symbol)
    return dict(
        version=version,
        verdict="fail" if errors else "pass",
        archives=len(original),
        hours=checked,
        post_delivery_rows=sum(
            m["post_delivery_excluded_rows"] for m in inputs["metadata"]
        ),
        delivery_checks={
            status: sum(m["delivery_check"] == status for m in inputs["metadata"])
            for status in ("pass", "pending_not_expired", "excluded_no_listing")
        },
        errors=errors,
    )
