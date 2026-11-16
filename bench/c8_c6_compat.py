"""C6 改动前后逐组复算；每组有限时、可续跑，仅公开汇总与逐位比较。"""

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from ashare.data import crypto
from ashare.data.crypto.signals import SignalDataset
from ashare.data.store import atomic_json
from ashare.factor.crypto import FACTOR_NAMES, crypto_factors, daily_factor
from ashare.factor.crypto_eval import crypto_labels, holm_confirm
from bench.crypto_factors_report import SYMBOLS, clean, evaluate_group


def inputs(root, folder, reference=None):
    folder.mkdir(parents=True, exist_ok=True)
    identity = (
        json.loads((reference / "identity.json").read_text()) if reference else {}
    )
    ds = crypto.open(root, version=identity.get("data_version"))
    sig = SignalDataset(root, version=identity.get("signal_version"))
    f = crypto_factors(ds, sig, SYMBOLS, ds.meta["start"], ds.meta["end"] + 1)
    labels = crypto_labels(ds, SYMBOLS, f["funding_last"])
    data = dict(entries=labels["entry"])
    data.update({name: daily_factor(f[name])[1] for name in FACTOR_NAMES})
    for kind in ("spot", "usdm"):
        data["g_" + kind] = labels["daily"][kind]
        for h in (1, 7, 30):
            data[f"desc_{kind}_{h}"] = labels["descriptive"][h][kind]
            data[f"inf_{kind}_{h}"] = labels["inference"][h][kind]
    np.savez(folder / "inputs.npz", **data)
    (folder / "identity.json").write_text(
        json.dumps(dict(data_version=ds.version, signal_version=sig.version))
    )


def group(folder, symbol, kind, split):
    j = SYMBOLS.index(symbol)
    partial = folder / "partials" / f"{symbol}-{kind}-{split}.json"
    previous = json.loads(partial.read_text()) if partial.exists() else []
    done = {row["horizon"] for row in previous}
    for h in done:
        if {row["factor"] for row in previous if row["horizon"] == h} != set(
            FACTOR_NAMES
        ):
            raise ValueError("C6 检查点因子不完整")
    with np.load(folder / "inputs.npz") as a:
        task = (
            symbol,
            j,
            kind,
            split,
            a["g_" + kind][:, j],
            {name: a[name][:, j] for name in FACTOR_NAMES},
            a["entries"],
            {h: a[f"desc_{kind}_{h}"][:, j] for h in (1, 7, 30)},
            4000,
            True,
        )

        def checkpoint(rows):
            atomic_json(partial, clean(previous + rows))

        rows = evaluate_group(
            task,
            horizons=tuple(h for h in (7, 1, 30) if h not in done),
            checkpoint=checkpoint,
        )
        results = previous + rows
    (folder / f"{symbol}-{kind}-{split}.json").write_text(
        json.dumps(clean(results), ensure_ascii=False, sort_keys=True, allow_nan=False)
    )


def run(root, folder, reference=None):
    if not (folder / "inputs.npz").exists():
        inputs(root, folder, reference)
    tasks = [
        (s, k, phase)
        for s in SYMBOLS
        for k in ("usdm", "spot")
        for phase in ("out", "in")
    ]
    started = time.perf_counter()

    def execute(task):
        name = "-".join(task)
        if (folder / f"{name}.json").exists():
            return
        with (folder / f"{name}.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "bench.c8_c6_compat",
                    "--folder",
                    str(folder),
                    "--group",
                    *task,
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=840,
            )
        print(name + " complete", flush=True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(execute, tasks))
    (folder / "execution.json").write_text(
        json.dumps(
            dict(seconds=time.perf_counter() - started, tests=288, repetitions=4000)
        )
    )


def compare(before, after):
    with np.load(before / "inputs.npz") as a, np.load(after / "inputs.npz") as b:
        same_inputs = set(a.files) == set(b.files) and all(
            a[k].dtype == b[k].dtype
            and a[k].shape == b[k].shape
            and a[k].tobytes() == b[k].tobytes()
            for k in a.files
        )
        input_arrays = len(a.files)
    files = sorted(p.name for p in before.glob("*-*-*.json"))
    same = [
        name
        for name in files
        if (before / name).read_bytes() == (after / name).read_bytes()
    ]
    adjusted = []
    for folder in (before, after):
        rows = [
            row for name in files for row in json.loads((folder / name).read_text())
        ]
        family = {
            (r["symbol"], r["factor"]): r["p"] if r["p"] is not None else np.nan
            for r in rows
            if r["family"] == "confirmation"
        }
        adjusted.append(np.array(list(holm_confirm(family).values())))
    same_holm = adjusted[0].tobytes() == adjusted[1].tobytes()
    result = dict(
        input_arrays=input_arrays,
        input_arrays_bitwise_equal=same_inputs,
        holm_tests=len(adjusted[0]),
        holm_bitwise_equal=same_holm,
        groups=len(files),
        equal_groups=len(same),
        tests=sum(len(json.loads((before / n).read_text())) for n in files),
        repetitions=4000,
    )
    if not same_inputs or not same_holm or len(same) != 8:
        raise ValueError(result)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root")
    p.add_argument("--reference", type=Path)
    p.add_argument("--folder", type=Path)
    p.add_argument("--group", nargs=3)
    p.add_argument("--compare", nargs=2, type=Path)
    a = p.parse_args()
    if a.compare:
        print(json.dumps(compare(*a.compare), indent=2))
    elif a.group:
        group(a.folder, *a.group)
    else:
        run(a.root, a.folder, a.reference)
