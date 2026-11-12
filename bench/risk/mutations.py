"""真实源码变异：先全绿、逐项失败、恢复源码/本地扩展后再全绿。"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter

from mutation_cases import CASES

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(output, only=None, witnesses=None):
    witnesses = {} if witnesses is None else json.loads(Path(witnesses).read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        CARGO_HOME=str(ROOT / ".cache/cargo-home"),
        PYO3_PYTHON=str(ROOT / ".venv/bin/python"),
        PYTHONDONTWRITEBYTECODE="1",
    )
    native = next((ROOT / "packages/portopt/python/portopt").glob("_core*.so"))
    files = {ROOT / f for _, edits in CASES for f, *_ in edits}
    sources = {f: f.read_bytes() for f in files}
    base_native = sha(native)
    records = []

    def test(label):
        begin = perf_counter()
        proc = subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "pytest",
                *(
                    [witnesses[label]]
                    if label in witnesses
                    else ["test/risk", "test/portfolio"]
                ),
                "-q",
                "-p",
                "no:cacheprovider",
                "--tb=line",
                "-rf",
            ],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
        )
        text = proc.stdout + proc.stderr
        (output / (label + ".txt")).write_text(text)
        failed = [
            line.split(" - ")[0].removeprefix("FAILED ")
            for line in text.splitlines()
            if line.startswith("FAILED ")
        ]
        return {
            "returncode": proc.returncode,
            "tests": failed,
            "caught": proc.returncode == 1
            and bool(failed)
            and "ERROR collecting" not in text,
            "seconds": perf_counter() - begin,
            "tail": text.strip().splitlines()[-1:],
        }

    def build(label):
        proc = subprocess.run(
            ["make", "dev"], cwd=ROOT, env=env, capture_output=True, text=True
        )
        (output / (label + "-build.txt")).write_text(proc.stdout + proc.stderr)
        if proc.returncode:
            raise RuntimeError("扩展构建失败，不算测试捕获")
        proc = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                "import portopt; print(portopt._core.__file__)",
            ],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert Path(proc.stdout.strip()).resolve() == native.resolve()

    baseline = test("baseline")
    assert baseline["returncode"] == 0, baseline
    print("baseline", baseline, flush=True)
    for name, edits in CASES:
        label = name.split()[0]
        if only and label not in only:
            continue
        rust = any(f.endswith(".rs") for f, *_ in edits)
        record = {"name": name, "native": rust}
        try:
            for file, old, new, count in edits:
                path = ROOT / file
                text = path.read_text()
                actual = text.count(old)
                assert actual > 0 and (count < 0 or actual == count), (
                    label,
                    file,
                    old,
                    actual,
                    count,
                )
                path.write_text(text.replace(old, new))
            if rust:
                build(label)
                record["native_changed"] = sha(native) != base_native
                assert record["native_changed"]
            record.update(test(label))
        finally:
            for f in {ROOT / e[0] for e in edits}:
                f.write_bytes(sources[f])
            if rust:
                build(label + "-restore")
                record["native_restored"] = sha(native) == base_native
                assert record["native_restored"]
            record["source_restored"] = all(
                f.read_bytes() == data for f, data in sources.items()
            )
            assert record["source_restored"]
        records.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        (output / "mutations.json").write_text(
            json.dumps(
                dict(baseline=baseline, cases=records), ensure_ascii=False, indent=2
            )
        )
    final = test("restored")
    assert final["returncode"] == 0
    result = dict(
        baseline=baseline,
        cases=records,
        restored=final,
        all_sources_restored=all(f.read_bytes() == data for f, data in sources.items()),
        native_restored=sha(native) == base_native,
    )
    (output / "mutations.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )
    assert all(r["caught"] for r in records), [
        r["name"] for r in records if not r["caught"]
    ]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".cache/risk-mutations"))
    parser.add_argument("--only", nargs="+")
    parser.add_argument(
        "--witnesses",
        type=Path,
        help="已证实可捕获的测试节点映射；前后仍运行完整专项测试",
    )
    args = parser.parse_args()
    run(args.output, args.only, args.witnesses)
