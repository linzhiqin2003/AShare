"""国君 191 参考实现的真实面板冒烟：最近 N 个交易日 × 全部股票，逐条记录耗时和有效格比例。

用法（数据目录只读）::

    .venv/bin/python tests/reference_gtja191/smoke_gtja191_ref.py \\
        --root /path/to/AShare/data --days 300 --out smoke.json

有效格比例的分母是「当天有收盘价」的格子。``all_nan``：没有一个有限值；``constant``：
有限值只有一种。结果写成 JSON（每条一行统计）并在终端打印汇总。
"""

import argparse
import json
import sys
import time

import numpy as np


def _inputs_module():
    from ashare.factor.formula import inputs

    return inputs


def run(root, days, warmup, names=None, log=print):
    from ashare.factor.alphas.reference import gtja191_ref as g

    inputs = _inputs_module()
    t0 = time.perf_counter()
    source = inputs.load_inputs(root=root, days=days, warmup=warmup)
    data = inputs.reference_frames(source)
    fields = g._fields(data)
    log(f"取数 {time.perf_counter() - t0:.1f}s，形状 {data['close'].shape}")
    quoted = np.isfinite(fields.C)
    rows = []
    for name in names or sorted(g.REFERENCES):
        t1 = time.perf_counter()
        try:
            values = g.REFERENCES[name](fields).to_numpy(float)
            error = None
        except Exception as exc:  # 冒烟要把每条的报错都记下来
            values = None
            error = f"{type(exc).__name__}: {exc}"
        seconds = time.perf_counter() - t1
        row = {"name": name, "seconds": round(seconds, 3), "error": error}
        if values is not None:
            finite = np.isfinite(values)
            distinct = np.unique(values[finite]).size if finite.any() else 0
            last = finite[-1]
            row.update(
                valid=float(finite[quoted].mean()),
                valid_last_day=float(last[quoted[-1]].mean()),
                outside=int((finite & ~quoted).sum()),
                all_nan=bool(not finite.any()),
                constant=bool(distinct == 1),
                distinct=int(min(distinct, 1_000_000)),
            )
        rows.append(row)
        log(json.dumps(row, ensure_ascii=False))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--days", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--names", default=None, help="逗号分隔，只跑这几条")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    names = args.names.split(",") if args.names else None
    rows = run(args.root, args.days, args.warmup, names)
    with open(args.out, "w") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=1)
    bad = [
        r["name"] for r in rows if r["error"] or r.get("all_nan") or r.get("constant")
    ]
    slow = sorted(rows, key=lambda r: -r["seconds"])[:10]
    print("全 NaN / 全常数 / 报错：", bad)
    print("最慢：", [(r["name"], r["seconds"]) for r in slow])
    print("总耗时：", round(sum(r["seconds"] for r in rows), 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
