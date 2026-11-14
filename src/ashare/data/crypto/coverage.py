"""逐数据集覆盖统计；观测范围与完整可回测范围分开。"""


def coverage(data):
    result = []
    for instrument, datasets in sorted(data.items()):
        for dataset, rows in sorted(datasets.items()):
            times = [
                r.get("time", r.get("fundingTime", r.get("deliveryTime"))) for r in rows
            ]
            result.append(
                dict(
                    instrument=instrument,
                    dataset=dataset,
                    rows=len(rows),
                    first=min(times) if times else None,
                    last=max(times) if times else None,
                    sources=sorted({r.get("source", "rest") for r in rows}),
                )
            )
    return result
