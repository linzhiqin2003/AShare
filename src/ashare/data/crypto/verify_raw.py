"""原始 ZIP 的独立 CSV 解码与规范化行对账；不复用 formats。"""

import csv
import hashlib
import io
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path


def verify_raw(root, inputs):
    if not inputs.get("raw_version"):
        return [], 0
    root = Path(root)
    manifest = json.loads(
        (root / "raw_versions" / f"{inputs['raw_version']}.json").read_text()
    )
    errors, checked = [], 0
    decoded = {}
    context = inputs.get("recent", {})
    snapshot = context.get("funding_info")
    if snapshot is not None:
        entry = manifest.get(snapshot.get("key"), {})
        payload = (root / "objects" / entry["sha256"]).read_bytes() if entry else None
        if (
            not entry
            or entry["sha256"] != snapshot.get("sha256")
            or entry.get("known_at") != snapshot.get("fetched_at")
            or json.loads(payload) != snapshot.get("records")
        ):
            errors.append(dict(check="inferred_snapshot_provenance"))
    if context.get("source_contract"):
        for p in inputs["profiles"]:
            if p["contract_type"] != "PERPETUAL":
                continue
            name = p["instrument"]
            ref = context.get("funding_price_refs", {}).get(name, {})
            entry = manifest.get(ref.get("key"), {})
            if (
                not entry
                or entry["sha256"] != ref.get("sha256")
                or not ref.get("key", "").startswith(f"funding_prices/{p['symbol']}/")
            ):
                errors.append(dict(check="funding_rest_reference", instrument=name))
                continue
            try:
                original = json.loads((root / "objects" / entry["sha256"]).read_bytes())

                def economic(rows):
                    # 保留毫秒原时刻，不共享 funding_hour、price_map 或构建判据。
                    return sorted(
                        (
                            int(r["fundingTime"]),
                            float(r["fundingRate"]),
                            float(r["markPrice"]) if r.get("markPrice") else None,
                        )
                        for r in rows
                    )

                if economic(original) != economic(
                    inputs["data"][name].get("funding_prices", [])
                ):
                    errors.append(dict(check="funding_rest_decode", instrument=name))
            except (ValueError, TypeError, KeyError):
                errors.append(dict(check="funding_rest_decode", instrument=name))
            for window in context.get("funding_archive_coverage", {}).get(name, []):
                key = window["key"]
                raw = manifest.get(key, {})
                try:
                    label = key.removesuffix(".zip")[-7:]
                    first = datetime.fromisoformat(label + "-01").replace(tzinfo=UTC)
                    following = first.replace(
                        year=first.year + (first.month == 12),
                        month=first.month % 12 + 1,
                    )
                    valid = (
                        f"/monthly/fundingRate/{p['symbol']}/" in key
                        and raw.get("sha256") == window.get("sha256")
                        and window["start"] == int(first.timestamp() * 1000)
                        and window["end"] == int(following.timestamp() * 1000)
                    )
                except (ValueError, KeyError):
                    valid = False
                if not valid:
                    errors.append(
                        dict(
                            check="funding_archive_reference", instrument=name, key=key
                        )
                    )
    for key, e in manifest.items():
        payload = (root / "objects" / e["sha256"]).read_bytes()
        sha = hashlib.sha256(payload).hexdigest()
        if sha != e["sha256"] or (
            e.get("source") == "archive" and sha != e.get("checksum")
        ):
            errors.append(dict(check="checksum", key=key))
        if e.get("source") != "archive":
            continue
        checked += 1
        try:
            with zipfile.ZipFile(io.BytesIO(payload)) as z:
                if z.testzip() is not None:
                    raise ValueError("zip CRC")
                names = [n for n in z.namelist() if n.endswith(".csv")]
                if len(names) != 1:
                    raise ValueError("CSV count")
                csv_rows = list(
                    csv.reader(io.StringIO(z.read(names[0]).decode("utf-8-sig")))
                )
            if csv_rows and not csv_rows[0][0].isdigit():
                csv_rows = csv_rows[1:]
            funding = "/fundingRate/" in key
            rows = {}
            for r in csv_rows:
                ts = int(r[0])
                if funding:
                    t = int((ts + 1_800_000) // 3_600_000) * 3_600_000
                    if abs(t - ts) > 1000:
                        raise ValueError("funding jitter")
                    row = dict(time=t, interval=int(r[1]), rate=float(r[2]))
                else:
                    scale = 1000 if ts >= 100_000_000_000_000 else 1
                    if scale == 1000 and "/futures/" in key:
                        raise ValueError("UM timestamp unit")
                    t = ts // scale
                    close_t = int(r[6]) // scale
                    if t % 3_600_000 or close_t - t > 3_600_999:
                        raise ValueError("kline boundary")
                    row = {
                        f: float(r[i])
                        for f, i in [
                            ("open", 1),
                            ("high", 2),
                            ("low", 3),
                            ("close", 4),
                            ("volume", 5),
                            ("quote_volume", 7),
                            ("taker_buy_quote_volume", 10),
                        ]
                    }
                    row["time"] = t
                    row["close_time"] = close_t
                    row["time_quality"] = (
                        "invalid_close_time"
                        if close_t < t
                        else "partial_halt"
                        if close_t < t + 3_598_999
                        else "normal"
                    )
                if t in rows:
                    raise ValueError("duplicate")
                rows[t] = row
            decoded[key] = rows
        except (ValueError, IndexError, zipfile.BadZipFile):
            errors.append(dict(check="raw_csv", key=key))
    for name, datasets in inputs["data"].items():
        for dataset, rows in datasets.items():
            for row in rows:
                key = row.get("source_key")
                if key not in decoded:
                    continue
                raw = decoded[key].get(row["time"])
                # 旧面板的正常行没有新增时间字段，仍可按旧版读取核对。
                if raw is None or any(
                    row.get(f) != v
                    for f, v in raw.items()
                    if f not in ("close_time", "time_quality")
                    or f in row
                    or raw["time_quality"] != "normal"
                ):
                    errors.append(
                        dict(
                            check="raw_decode",
                            instrument=name,
                            dataset=dataset,
                            time=row["time"],
                        )
                    )
    return errors, checked
