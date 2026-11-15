"""归档与旧推断版本的经济字段对账；差异是持久诊断，不阻塞纠正发布。"""

import json
from pathlib import Path

from ..store import atomic_json
from .dataset import CryptoDataset
from .storage import digest, encoded
from .timeutil import HOUR, funding_hour


def reconcile(ds):
    inputs = ds.inputs()
    for path in sorted((ds.root / "panels").glob("*/meta.json")):
        meta = json.loads(path.read_text())
        if not meta.get("allow_inferred_funding") or meta["version"] == ds.version:
            continue
        old = CryptoDataset(ds.root, meta["version"])
        differences = []
        for name, note in meta["recent"]["instruments"].items():
            if name not in inputs["data"]:
                continue
            source = inputs["data"][name]
            archived = [
                r
                for r in source.get("fundingRate", [])
                if r.get("source") != "inferred_continuation"
            ]
            coverage = (
                inputs.get("recent", {}).get("funding_archive_coverage", {}).get(name)
            )
            # 无原始对象的合成/旧输入仍只审计观测范围；新同步版本按整月声明。
            windows = (
                coverage
                if coverage is not None
                else [
                    dict(
                        start=note["anchor"],
                        end=max((r["time"] for r in archived), default=note["anchor"])
                        + 1,
                    )
                ]
            )
            hi = min(
                meta["end"],
                max((w["end"] - 1 for w in windows), default=note["anchor"]),
            )
            lo = max(note["anchor"], meta["start"])
            if hi <= lo:
                continue

            def covered(t, lo=lo, hi=hi, windows=windows):
                return lo < t <= hi and any(w["start"] <= t < w["end"] for w in windows)

            prices = {
                funding_hour(r["fundingTime"]): float(r["markPrice"])
                for r in source.get("funding_prices", [])
                if r.get("markPrice") and int(r["fundingTime"]) <= hi + 1000
            }
            new_values = {
                r["time"]: dict(
                    time=r["time"],
                    interval=r["interval"],
                    rate=r["rate"],
                    price=prices.get(r["time"]),
                )
                for r in archived
                if covered(r["time"])
            }
            old_values = {
                r["time"] + HOUR: dict(
                    time=r["time"] + HOUR,
                    interval=r["funding_interval_hours"],
                    rate=r["funding_rate"],
                    price=r["funding_price"],
                )
                for r in old.read(instruments=[name]).iter_rows(named=True)
                if r["quality"] & 4096 and covered(r["time"] + HOUR)
            }
            for t in sorted(new_values.keys() | old_values.keys()):
                if old_values.get(t) != new_values.get(t):
                    differences.append(
                        dict(
                            instrument=name,
                            time=t,
                            inferred=old_values.get(t),
                            archive=new_values.get(t),
                        )
                    )
        if differences:
            report = dict(
                inferred_version=old.version,
                archive_version=ds.version,
                differences=differences,
            )
            key = digest(encoded(report))
            atomic_json(ds.root / "funding_reconciliations" / f"{key}.json", report)


def reports(root, version):
    result = []
    for path in sorted((Path(root) / "funding_reconciliations").glob("*.json")):
        report = json.loads(path.read_text())
        if version in (report["inferred_version"], report["archive_version"]):
            result.append(dict(path=str(path.relative_to(root)), **report))
    return result
