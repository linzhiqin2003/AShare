"""比较两份端到端剖析结果：按阶段和用例对齐，给出旧、新、比值与结果摘要是否逐位相同。

python bench/e2e/compare.py                 # 旧 = baseline.json，新 = 最近一轮
python bench/e2e/compare.py OLD.json NEW.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from run import DEFAULT_WORK  # noqa: E402

BASELINE = HERE / "results/baseline.json"


def latest(directory):
    files = sorted(directory.glob("*.json"))
    if not files:
        raise SystemExit(f"{directory} 里没有结果；先运行 make perf-profile")
    return files[-1]


def cases(report):
    out = {}
    for stage, record in report["stages"].items():
        for case in record.get("cases", []):
            out[(stage, case["case"])] = case
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("old", nargs="?", type=Path, default=BASELINE)
    parser.add_argument("new", nargs="?", type=Path)
    args = parser.parse_args()
    work = Path(os.environ.get("ASHARE_E2E_WORK", DEFAULT_WORK))
    new_path = args.new or latest(work / "results")
    old, new = (json.loads(p.read_text()) for p in (args.old, new_path))
    eo, en = old["environment"], new["environment"]
    print(
        f"旧 {args.old}\n   {eo['time']} 提交 {eo['commit'][:10]} 负载 {eo['load'][0]:.1f} 开机 {eo.get('uptime_hours')} h"
    )
    print(
        f"新 {new_path}\n   {en['time']} 提交 {en['commit'][:10]} 负载 {en['load'][0]:.1f} 开机 {en.get('uptime_hours')} h"
    )
    if eo["commit"] != en["commit"] or en.get("dirty"):
        print("注意：提交号不同或新一轮有未提交改动")
    if old["options"] != new["options"]:
        print(f"注意：运行参数不同 {old['options']} → {new['options']}")
    a, b = cases(old), cases(new)
    print(
        f"\n{'阶段/用例':52s} {'旧最小':>9s} {'新最小':>9s} {'比值':>6s} {'旧中位':>9s} {'新中位':>9s} {'比值':>6s}  结果"
    )
    for key in list(a) + [k for k in b if k not in a]:
        x, y = a.get(key), b.get(key)
        name = f"{key[0]}/{key[1]}"
        if x is None or y is None:
            print(
                f"{name:52s} {'只在' + ('新' if x is None else '旧') + '一轮里':>30s}"
            )
            continue
        note = []
        if x.get("scale") != y.get("scale"):
            note.append("规模不同")
        if x.get("summary") is not None or y.get("summary") is not None:
            note.append(
                "逐位相同" if x.get("summary") == y.get("summary") else "结果不同"
            )
        values = []
        for field in ("wall_min", "wall_median"):
            p, q = x.get(field), y.get(field)
            ratio = q / p if p and q is not None else float("nan")
            values += [p or 0.0, q or 0.0, ratio]
        print(
            f"{name:52s} {values[0]:9.4f} {values[1]:9.4f} {values[2]:6.2f} "
            f"{values[3]:9.4f} {values[4]:9.4f} {values[5]:6.2f}  {'，'.join(note)}"
        )
    print(
        f"\n总时长 旧 {old.get('total_seconds', 0):.0f} s，新 {new.get('total_seconds', 0):.0f} s"
    )


if __name__ == "__main__":
    main()
