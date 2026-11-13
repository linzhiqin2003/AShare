"""手工验收：HTTP 四组运行与 groupbt 直接调用逐位对拍；输出只进入私有报告。"""

import argparse
import json
import os
import time

import httpx
import numpy as np

from ashare.factor import Evaluator, groupbt
from ashare.platform.language import parse
from ashare.platform.runner import PanelCache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    args = parser.parse_args()
    with httpx.Client(
        base_url=args.url, headers={"Origin": args.url}, timeout=30
    ) as client:
        tab = client.post("/api/tabs").json()["id"]
        try:
            response = client.post(
                f"/api/tabs/{tab}/runs",
                json={
                    "name": "low_vol",
                    "source": "vol = ts_std(returns, [20, 60])\nalpha = -vol",
                    "settings": {"group_neutralize": ["industry", "cap+industry"]},
                },
            )
            response.raise_for_status()
            job = response.json()["id"]
            deadline = time.monotonic() + 1200
            while time.monotonic() < deadline:
                result = client.get(f"/api/jobs/{job}").json()
                if result["status"] != "running":
                    break
                time.sleep(1)
            if result["status"] != "complete":
                raise RuntimeError("HTTP 运行未完成")
        finally:
            client.delete(f"/api/tabs/{tab}").raise_for_status()
    source = PanelCache(os.environ["ASHARE_DATA_DIR"])
    variants = parse("vol = ts_std(returns, [20,60])\nalpha = -vol")
    source.ensure(set(groupbt.FIELDS).union(*(v.expression.fields for v in variants)))
    rows = []
    from ashare.platform.storage import Store

    store = Store(os.environ["ASHARE_PLATFORM_DIR"])
    for run in result["runs"]:
        window = int(run["label"].split(" · ")[0])
        expression = variants[0 if window == 20 else 1].expression
        alpha = Evaluator().evaluate(expression, source)
        direct = groupbt.run(alpha, source, **run["settings"])
        stored = store.series(run["id"])
        keep = direct.dates >= groupbt.IS[0]
        for key in direct.pnl:
            np.testing.assert_array_equal(
                stored[f"is.pnl.{key}"].view(np.uint64),
                direct.pnl[key][keep].view(np.uint64),
            )
        np.testing.assert_array_equal(
            stored["is.groups"].view(np.uint64),
            direct.group_pnl[:, keep].copy().view(np.uint64),
        )
        sharpe = direct.table().to_dicts()[-1]["sharpe"]
        assert run["summary"]["is"]["sharpe"] == sharpe
        rows.append(
            {
                "window": window,
                "neutralize": run["settings"]["group_neutralize"],
                "platform": run["summary"]["is"]["sharpe"],
                "direct": sharpe,
                "bitwise": True,
            }
        )
        del alpha, direct
    print(
        json.dumps(
            {"job": job, "elapsed": result["elapsed"], "rows": rows}, ensure_ascii=False
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
