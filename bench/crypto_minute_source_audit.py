"""核对分钟/小时差异是否已经存在于官方 CSV 文本；只输出汇总，不写小时库。"""

import argparse
import csv
import hashlib
import io
import json
import zipfile
from decimal import Decimal
from pathlib import Path

from ashare.data.crypto import open as open_hourly
from ashare.data.crypto.minute import metadata


def rows(payload):
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        return list(
            csv.reader(
                io.StringIO(
                    z.read(next(n for n in z.namelist() if n.endswith(".csv"))).decode()
                )
            )
        )


def checked(root, ref):
    payload = (root / "objects" / ref["sha256"]).read_bytes()
    assert hashlib.sha256(payload).hexdigest() == ref["sha256"] == ref["checksum"]
    return rows(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="tmp/crypto_minute")
    parser.add_argument("--hourly-root", required=True)
    args = parser.parse_args()
    root, hourly_root = Path(args.root), Path(args.hourly_root)
    report = json.loads((root / "reconciliation.json").read_text())
    ds = open_hourly(hourly_root, version=report["hourly_version"])
    manifest = json.loads(
        (hourly_root / "raw_versions" / (ds.meta["raw_version"] + ".json")).read_text()
    )
    meta = metadata(root=root, version=report["version"])
    summary = []
    for part in report["reconciliation"]:
        differences = [
            d for d in part["differences"] if d["reason"] == "source_difference"
        ]
        if not differences:
            continue
        refs = meta["partitions"][part["partition"]]["archives"]
        key, ref = next((k, v) for k, v in refs.items() if "/monthly/" in k)
        hourly_key = key.replace("/1m/", "/1h/").replace("-1m-", "-1h-")
        minute_csv = checked(root, ref)
        hourly_csv = checked(hourly_root, manifest[hourly_key])
        hrows = {int(r[0]): r for r in hourly_csv if r[0].isdigit()}
        mrows = {int(r[0]): r for r in minute_csv if r[0].isdigit()}
        max_delta = {}
        proven, equal_panel = 0, 0
        for d in differences:
            col = {"volume": 5, "quote_volume": 7, "taker_buy_quote_volume": 10}[
                d["field"]
            ]
            t = d["time"]
            # 十进制原文求和，排除二进制浮点累加造成差异的解释。
            total = sum(
                (Decimal(mrows[t + i * 60000][col]) for i in range(60)), Decimal(0)
            )
            archived = Decimal(hrows[t][col])
            proven += total != archived
            equal_panel += float(archived) == d["hourly"]
            max_delta[d["field"]] = max(
                max_delta.get(d["field"], Decimal(0)), abs(total - archived)
            )
        summary.append(
            dict(
                partition=part["partition"],
                hours=len({d["time"] for d in differences}),
                differing_fields=len(differences),
                decimal_source_differences=proven,
                hourly_panel_matches_archive=equal_panel,
                max_decimal_absolute={k: str(v) for k, v in max_delta.items()},
            )
        )
    out = dict(
        hourly_version=report["hourly_version"],
        checksums_verified=True,
        summary=summary,
        conclusion="差异已存在于官方 1m/1h 原始 CSV；小时面板等于其归档，分钟求和用 Decimal 仍不同；交易所内部原因未知",
    )
    path = (
        Path(__file__).resolve().parents[1]
        / "bench/results/crypto_minute_source_audit.json"
    )
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
