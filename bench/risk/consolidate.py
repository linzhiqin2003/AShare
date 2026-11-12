"""显式冻结一份最小 JSON 基线；日常重跑不会改写它。"""

import argparse
import json
from pathlib import Path

FILES = (
    "kernels",
    "factor-kernels",
    "gram",
    "windows",
    "varying-regression",
    "pipeline",
    "optimizers",
    "optimizers-auto",
    "rebuild",
    "structure",
    "validation",
    "review-comparisons",
)


def compact(value):
    if isinstance(value, dict):
        return {
            k: compact(v)
            for k, v in value.items()
            if k
            not in (
                "pid",
                "objectives",
                "origins",
                "source_metadata",
                "threadpools",
                "blas",
                "numerical_recoveries",
            )
        }
    if isinstance(value, list):
        return [compact(v) for v in value]
    if isinstance(value, str) and ("/Users/" in value or "/private/" in value):
        return Path(value).name if "\n" not in value else "local diagnostic omitted"
    return value


def assemble(directory, previous=None):
    directory = Path(directory)
    result = {"schema": 2, "measurements": {}}
    for name in FILES:
        path = directory / (name + ".json")
        if path.exists():
            result["measurements"][name] = compact(json.loads(path.read_text()))
    mutations = directory / "mutations/mutations.json"
    if mutations.exists():
        result["mutations"] = compact(json.loads(mutations.read_text()))
    comparisons = directory / "comparisons.json"
    if comparisons.exists():
        result["comparisons"] = compact(json.loads(comparisons.read_text()))
    if previous:
        old = json.loads(Path(previous).read_text())
        result["previous_optimizer"] = {
            "metadata": compact(old.get("metadata", {})),
            "rows": {
                key: {
                    metric: row[metric]
                    for metric in ("single_seconds", "sequential_seconds", "status")
                    if metric in row
                }
                for key, row in old["rows"].items()
                if "parallel" not in key
            },
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--previous", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(assemble(args.input, args.previous), ensure_ascii=False, indent=2)
        + "\n"
    )
