"""生产规模合成持仓的原始/压缩存储、单日 HTTP 大小和耗时。"""

import gzip
import json
import statistics
from pathlib import Path
from zipfile import ZipFile

from fastapi.testclient import TestClient

from ashare.platform.analysis import run_variant
from ashare.platform.app import create_app
from ashare.platform.config import DEFAULTS
from ashare.platform.language import parse
from bench.platform.synthetic import panel, write_panel


def main():
    root = Path(".cache/platform/holdings-r5")
    source = panel(n=1800, full=True)
    data = root / "data"
    if not data.exists():
        write_panel(data, source)
    app = create_app(root / "results", data_root=data)
    store = app.state.store
    variant = parse("-ts_std(returns,20)")[0]
    rows = []
    with TestClient(app, base_url="http://127.0.0.1") as client:
        for period in (1, 5, 20):
            settings = DEFAULTS | {"rebalance": period, "group_neutralize": "industry"}
            job, runs = store.begin(
                "measurement", "synthetic", [(variant, settings)], {}
            )
            summary, arrays = run_variant(variant.expression, source, settings)
            run = runs[0]["id"]
            store.save(run, summary, arrays, 0, "synthetic")
            store.finish(job, "complete")
            file = store.root / "holdings" / f"{run}.npz"
            with ZipFile(file) as archive:
                raw = sum(entry.file_size for entry in archive.infolist())
            base = f"/api/runs/{run}/holdings"
            dates = client.get(base).json()
            timings = []
            for _ in range(5):
                response = client.get(base, params={"date": dates["date"]})
                response.raise_for_status()
                timings.append(float(response.headers["server-timing"].split("=")[-1]))
            row = dict(
                rebalance=period,
                days=len(dates["dates"]),
                sparse_raw_bytes=raw,
                sparse_compressed_bytes=file.stat().st_size,
                daily_dense_weights_bytes=len(dates["dates"]) * 1800 * 8,
                http_raw_bytes=len(response.content),
                http_gzip_bytes=len(gzip.compress(response.content, compresslevel=6)),
                http_cold_ms=timings[0],
                http_warm_median_ms=statistics.median(timings[1:]),
                daily_rows=response.json()["total"],
            )
            rows.append(row)
    (root / "metrics.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
