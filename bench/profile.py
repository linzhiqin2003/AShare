"""Instrument a disposable source copy; production sources remain untouched.

Reports phase wall time including timer overhead. The separately measured API
minimum is the performance result; phase fractions are diagnostic estimates.
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COPY = ROOT / ".cache/panelbt/profile"


def main():
    COPY.mkdir(parents=True, exist_ok=True)
    for name in ["Cargo.toml", "Cargo.lock"]:
        shutil.copy2(ROOT / name, COPY / name)
    manifest = COPY / "Cargo.toml"
    lines = manifest.read_text().splitlines()
    lines[1] = (
        'members = ["crates/tradecore", "crates/panelbt-core", "packages/panelbt"]'
    )
    manifest.write_text("\n".join(lines) + "\n")
    for name in ["crates/tradecore", "crates/panelbt-core", "packages/panelbt"]:
        shutil.copytree(
            ROOT / name,
            COPY / name,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.so", "target", ".venv"),
        )
    p = COPY / "crates/panelbt-core/src/lib.rs"
    s = p.read_text().replace(
        "pub daily: Vec<f64>,", "pub daily: Vec<f64>,\n    pub profile: [u128;5],"
    )
    s = s.replace(
        "daily: vec![0.0; DAILY * m.t],",
        "daily: vec![0.0; DAILY * m.t],\n        profile: [0;5],",
    )
    s = s.replace(
        "let base = day * m.n;",
        "let base = day * m.n;\n        let mut timer = std::time::Instant::now();",
        1,
    )
    markers = [
        ("        let action_start =", 0),
        ("        let rebalance = cursor", 1),
        ("        orders.clear();", 2),
        ("        let (mut long, mut short)", 3),
        ("        previous = equity;", 4),
    ]
    for marker, index in markers:
        assert marker in s, marker
        s = s.replace(
            marker,
            f"        out.profile[{index}] += timer.elapsed().as_nanos();\n        timer = std::time::Instant::now();\n"
            + marker,
            1,
        )
    p.write_text(s)
    p = COPY / "crates/panelbt-core/src/futures.rs"
    s = p.read_text().replace(
        "daily: vec![0.0; DAILY * m.t],",
        "daily: vec![0.0; DAILY * m.t],\n        profile: [0;5],",
    )
    p.write_text(s)
    p = COPY / "packages/panelbt/src/lib.rs"
    s = p.read_text().replace(
        "let d = PyDict::new(py);",
        'let d = PyDict::new(py);\n            d.set_item("profile", o.profile.to_vec())?;',
    )
    p.write_text(s)
    env = os.environ.copy()
    env["VIRTUAL_ENV"] = str(COPY / ".venv")
    env["PYO3_PYTHON"] = str(COPY / ".venv/bin/python")
    env["CARGO_HOME"] = str(ROOT / ".cache/cargo-home")
    if not (COPY / ".venv/bin/python").exists():
        subprocess.run(
            [
                "uv",
                "venv",
                str(COPY / ".venv"),
                "--python",
                str(ROOT / ".venv/bin/python"),
            ],
            check=True,
        )
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                env["PYO3_PYTHON"],
                "numpy",
                "maturin",
            ],
            check=True,
        )
    subprocess.run(
        [
            str(COPY / ".venv/bin/maturin"),
            "develop",
            "--release",
            "--manifest-path",
            str(COPY / "packages/panelbt/Cargo.toml"),
        ],
        env=env,
        cwd=COPY,
        check=True,
    )
    code = """
import json,sys
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from pair import inputs
import numpy as np
import panelbt as bt
from panelbt.api import _prepare
from panelbt import _core
from time import perf_counter_ns
T,N=4000,5500
m=bt.Market(**inputs(T,N))
days=np.arange(0,T,21,dtype=np.int64)
t=bt.Targets(days,np.arange(len(days)+1,dtype=np.int64)*100,((np.arange(100)[None,:]+np.arange(len(days))[:,None]*37)%N).ravel(),np.full(len(days)*100,.0096))
args=_prepare(m,[t],bt.Costs(tax_sell=.0005),bt.Config(initial_cash=1e7,max_participation=.1))
rows=[]
for _ in range(20):
 start=perf_counter_ns();raw=_core.run(*args,1)[0];elapsed=perf_counter_ns()-start
 rows.append(dict(total_ns=elapsed,phase_ns=raw['profile']))
print(json.dumps(rows))
"""
    output = subprocess.check_output(
        [env["PYO3_PYTHON"], "-c", code, str(ROOT / "bench")], env=env, text=True
    )
    rows = json.loads(output)
    report = dict(
        T=4000,
        N=5500,
        phases=[
            "flows",
            "corporate_actions_and_delists",
            "command_and_valuation",
            "execution",
            "close_and_daily_outputs",
        ],
        samples=rows,
        method="Disposable source instrumentation; two Instant reads per phase per day included; these fractions are diagnostic and not an uninstrumented lower bound.",
    )
    (ROOT / ".cache/panelbt/profile.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report))


if __name__ == "__main__":
    main()
