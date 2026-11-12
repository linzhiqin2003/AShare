"""Reproducible 5,500-symbol ingestion baselines and a 4,800-batch trading day."""

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import paperbroker
from paperbroker import Broker, money

ROOT = Path(__file__).resolve().parents[2]
assert Path(paperbroker.__file__).resolve().is_relative_to(ROOT)


def quotes(n):
    a = np.zeros((n, 18), np.int64)
    a[:, 0] = np.arange(n)
    a[:, 1] = 1
    a[:, 2:7] = money(10)
    a[:, 9] = a[:, 11] = money(10)
    a[:, 10] = a[:, 12] = 10000
    a[:, 13], a[:, 14] = money(11), money(9)
    return a


def run(repeats=7, loops=100, n=5500, batches=4800):
    symbols = [f"S{i}" for i in range(n)]
    a = quotes(n)
    python_state = [None] * n
    numpy_state = np.zeros_like(a)
    numpy_valid = np.zeros(n, bool)
    b = Broker(symbols)
    b.session(0, 1, "continuous")

    def python_feed():
        for row in a:
            i = int(row[0])
            old = python_state[i]
            if old is None or (row[1] > old[1] and row[7] >= old[7]):
                python_state[i] = row.tolist()

    def numpy_feed():
        ids = a[:, 0]
        valid = ~numpy_valid[ids] | (
            (a[:, 1] > numpy_state[ids, 1]) & (a[:, 7] >= numpy_state[ids, 7])
        )
        numpy_state[ids[valid]] = a[valid]
        numpy_valid[ids[valid]] = True

    def rust_feed():
        b.feed(a)

    # Compare full retained quotes before timing. Baselines do not implement orders.
    for fn in (python_feed, numpy_feed, rust_feed):
        fn()
    assert np.array_equal(np.array(python_state), numpy_state)
    assert json.loads(b.dump())["quotes"] == python_state
    results = {}
    for name, fn in (
        ("python_loop", python_feed),
        ("numpy_vectorized", numpy_feed),
        ("rust_kernel", rust_feed),
    ):
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            for _ in range(loops):
                a[:, 1] += 3000
                a[:, 7] += 1000
                fn()
            samples.append((time.perf_counter() - start) * 1000 / loops)
        results[name] = dict(median_ms=statistics.median(samples), samples_ms=samples)
    samples = []
    final = {}
    for _ in range(3):
        broker = Broker(symbols, initial_cash=1_000_000_000)
        broker.session(0, 1, "continuous")
        a = quotes(n)
        broker.feed(a)
        start = time.perf_counter()
        order_count = 0
        for i in range(batches):
            if i % 12 == 0:
                broker.order(int(a[0, 1]) + 1, (i // 12) % n, "buy", 100, 11)
                order_count += 1
            a[:, 1] += 3000
            a[:, 7] += 1000
            broker.feed(a)
        broker.session(int(a[0, 1]) + 1, 1, "closed")
        broker.settle(int(a[0, 1]) + 2, 1)
        elapsed = time.perf_counter() - start
        fs = [e for e in broker.events() if e["type"] == "fill"]
        assert len(fs) == order_count
        assert (
            sum(p["total"] for p in broker.account()["positions"]) == order_count * 100
        )
        samples.append(elapsed)
        final = dict(
            orders=order_count, fills=len(fs), batches=batches, quotes=n * batches
        )
    return dict(
        platform=platform.platform(),
        python=platform.python_version(),
        numpy=np.__version__,
        symbols=n,
        baseline_scope="quote retention with no orders; Python/NumPy retain all 18 columns but omit native input validation and account/event machinery",
        contention="Other workloads may be running; rerun on an idle machine.",
        batch=results,
        replay=dict(
            median_seconds=statistics.median(samples), samples_seconds=samples, **final
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("bench/paper/benchmark-results.json")
    )
    args = parser.parse_args()
    baseline = json.loads(args.output.read_bytes()) if args.output.exists() else {}
    result = baseline | run()
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
