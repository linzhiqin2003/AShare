"""离线审计已校验的季度归档；仅输出计数、元数据冲突，不建立例外版本。"""

import argparse
import csv
import hashlib
import io
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path


def audit(root):
    root = Path(root)
    plan = json.loads((root / "signals/sync_plan.json").read_text())
    summary = dict(
        contracts=len(plan["metadata"]),
        archives=len(plan["references"]),
        contracts_with_post_delivery_rows=0,
        post_delivery_rows=0,
        contracts_with_nonzero_post_delivery_volume=0,
        interior_gaps=0,
        metadata_conflicts=[],
        delivery_window_failures=[],
        delivery_checks={"pass": 0, "pending_not_expired": 0},
        contracts_detail=[],
    )
    for meta in plan["metadata"]:
        symbol = meta["symbol"]
        expiry = int(
            datetime.strptime(symbol[-6:], "%y%m%d")
            .replace(tzinfo=UTC, hour=8)
            .timestamp()
            * 1000
        )
        rows = []
        for key in meta["archives"]:
            ref = plan["references"][key]
            payload = (root / "signals/raw/objects" / ref["sha256"]).read_bytes()
            sha = hashlib.sha256(payload).hexdigest()
            assert sha == ref["sha256"] == ref["checksum"]
            checksum = ref["checksum_text"].split()
            assert checksum == [sha, key.rsplit("/", 1)[1]]
            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                assert archive.testzip() is None
                names = [n for n in archive.namelist() if n.endswith(".csv")]
                assert len(names) == 1
                values = list(
                    csv.reader(io.StringIO(archive.read(names[0]).decode("utf-8-sig")))
                )
            rows.extend(r for r in values if r[0].isdigit())
        times = sorted(int(r[0]) for r in rows)
        assert len(times) == len(set(times))
        after = [r for r in rows if int(r[6]) + 1 > expiry]
        nonzero = sum(float(r[5]) != 0 for r in after)
        last_trade = max((int(r[0]) for r in rows if float(r[5]) > 0), default=None)
        matured = (
            max(plan["end"], max((int(r[6]) + 1 for r in rows), default=0)) >= expiry
        )
        status = "pass" if matured else "pending_not_expired"
        if (last_trade is not None and last_trade > expiry) or (
            matured and (last_trade is None or last_trade < expiry - 86400000)
        ):
            summary["delivery_window_failures"].append(symbol)
        summary["delivery_checks"][status] += 1
        earliest = times[0] if times else None
        ev = meta.get("evidence", {})
        if ev.get("deliveryDate") is not None and ev["deliveryDate"] != expiry:
            summary["metadata_conflicts"].append(symbol + ":deliveryDate")
        if earliest is not None and ev.get("onboardDate", earliest) > earliest:
            summary["metadata_conflicts"].append(symbol + ":onboardDate")
        lo = max(plan["start"], ev.get("onboardDate", earliest) or plan["end"])
        lo = (lo + 3599999) // 3600000 * 3600000
        expected = set(range(lo, min(plan["end"], expiry), 3600000))
        gaps = len(expected - set(times))
        summary["contracts_with_post_delivery_rows"] += bool(after)
        summary["post_delivery_rows"] += len(after)
        summary["contracts_with_nonzero_post_delivery_volume"] += bool(nonzero)
        summary["interior_gaps"] += gaps
        summary["contracts_detail"].append(
            dict(
                symbol=symbol,
                archive_months=len(meta["archives"]),
                post_delivery_rows=len(after),
                post_delivery_rows_with_nonzero_volume=nonzero,
                interior_gaps=gaps,
                delivery_check=status,
            )
        )
    summary["strict_build"] = (
        "failed_as_required"
        if summary["delivery_window_failures"] or summary["metadata_conflicts"]
        else "metadata_checks_passed"
    )
    summary["definition"] = (
        "DESIGN 4.18 v5.1: last traded bar opens in [delivery-24h,delivery]; post_delivery excluded"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="tmp/crypto")
    parser.add_argument("--output", default="bench/results/crypto_signals_audit.json")
    args = parser.parse_args()
    value = audit(args.data)
    Path(args.output).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    print({k: v for k, v in value.items() if k != "contracts_detail"})


if __name__ == "__main__":
    main()
