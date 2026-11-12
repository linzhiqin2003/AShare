"""盘后由用户运行：核对收盘代理量、日线及录制延迟；本脚本开发时不联网。"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
from paperbroker.journal import canonical, durable_write

from ashare.paper.market import TZ
from ashare.paper.source import TushareClient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", required=True, type=Path)
    parser.add_argument("--day", required=True)
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path(__file__).resolve().parents[3] / ".cache/paper-tushare.lock",
    )
    args = parser.parse_args()
    now = datetime.now(TZ)
    if args.day == now.strftime("%Y%m%d") and now.hour < 20:
        parser.error("当天的 stk_auction_c 应在 20 点后核对")
    manifest = json.loads((args.directory / "manifest.json").read_text())
    if manifest.get("version") == 3:
        from ashare.paper.__main__ import read_state

        symbols = [
            p["symbol"] for p in read_state(args.directory)["account"]["positions"]
        ]
    else:
        symbols = manifest["symbols"]
    latency = []
    proxy = {}
    batches = 0
    for path in sorted((args.directory / "quotes").glob("*.parquet")):
        table = pl.read_parquet(path)
        if not table.height:
            continue
        received = table["received_at"][0]
        if datetime.fromtimestamp(received / 1000, TZ).strftime("%Y%m%d") != args.day:
            continue
        batches += 1
        if table["source"][0] == "rt_k":
            latency.extend((table["received_at"] - table["time"]).to_list())
        if table["source"][0] == "rt_k_close_volume_proxy":
            proxy.update(
                {
                    symbols[row["symbol"]]: row["auction_volume"]
                    for row in table.iter_rows(named=True)
                }
            )
    with TushareClient(args.lock) as client:
        results = {
            name: client.call(name, {"trade_date": args.day})
            for name in (
                "stk_auction",
                "stk_auction_o",
                "stk_auction_c",
                "daily",
                "stk_limit",
                "suspend_d",
            )
        }
        results["dividend"] = client.call("dividend", {"ex_date": args.day})
    actual = {r["ts_code"]: r["vol"] for r in results["stk_auction_c"]}
    differences = {
        s: dict(
            proxy_volume=v,
            official_volume=actual.get(s),
            difference=None if s not in actual else v - actual[s],
        )
        for s, v in proxy.items()
    }
    report = dict(
        day=args.day,
        batches=batches,
        latency_ms_quantiles=None
        if not latency
        else np.quantile(latency, [0, 0.5, 0.9, 0.99, 1]).tolist(),
        closing_auction_comparison=differences,
        official_rows={name: len(rows) for name, rows in results.items()},
    )
    durable_write(args.directory / f"verify-{args.day}.json", canonical(report))
    durable_write(args.directory / f"official-{args.day}.json", canonical(results))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
