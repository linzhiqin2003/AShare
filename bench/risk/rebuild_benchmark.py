"""全量构建、追加、无损缓存和固定窗口描述子的可复现计时。"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
import portopt
from baseline import metadata
from pipeline_benchmark import Source
from threadpoolctl import threadpool_limits

import ashare.risk.series as series
from ashare.risk.cache import ModelCache
from ashare.risk.data import adjusted_returns
from ashare.risk.fundamental import FundamentalConfig, exposures_at
from ashare.risk.rolling import RollingDescriptors


def digest(model):
    h = hashlib.sha256()
    for name in ("B", "F", "D"):
        h.update(getattr(model, name).tobytes())
    return h.hexdigest()


def descriptor_profile(p, c, external, external_names):
    result = {}
    t = len(p.dates)
    returns = adjusted_returns(p, t - 1)
    ages = np.cumsum(p["is_listed"], axis=0, dtype=np.int32)
    rolling = RollingDescriptors(p, returns, c, ages)
    begin = perf_counter()
    rolling.at(t - 2)
    result["rolling_prefix_seconds"] = perf_counter() - begin
    checkpoint = rolling.checkpoint()
    samples = {}
    byline = {}
    for mode in ("window_reference", "rolling"):
        rows = []
        for _rep in range(3):
            if mode == "rolling":
                rolling.restore(checkpoint)
            begin = perf_counter()
            exposures_at(
                p,
                returns,
                t - 1,
                c,
                external,
                external_names,
                listing_ages=ages,
                rolling=rolling if mode == "rolling" else None,
            )
            rows.append(perf_counter() - begin)
        samples[mode] = rows
        previous = [None, perf_counter()]

        def trace(frame, event, arg, previous=previous, mode=mode):
            if frame.f_code is exposures_at.__code__:
                now = perf_counter()
                if previous[0] is not None:
                    byline.setdefault(mode, {}).setdefault(previous[0], 0.0)
                    byline[mode][previous[0]] += now - previous[1]
                previous[:] = [frame.f_lineno if event == "line" else None, now]
                return trace
            return None

        if mode == "rolling":
            rolling.restore(checkpoint)
        sys.settrace(trace)
        try:
            exposures_at(
                p,
                returns,
                t - 1,
                c,
                external,
                external_names,
                listing_ages=ages,
                rolling=rolling if mode == "rolling" else None,
            )
        finally:
            sys.settrace(None)
    result["descriptor_seconds"] = samples
    result["descriptor_profile_lines"] = byline
    return result


def run(output, t=4000, n=5500, *, keep_cache=False):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261112)
    p = Source(t, n, rng)
    r = rng.normal(0, 0.01, (t, n)) + rng.normal(0, 0.004, (t, 1))
    p.fields["close"] = 10 * np.cumprod(1 + r, axis=0)
    p.fields["adj_factor"] = np.broadcast_to(np.ones(n), (t, n))
    external = np.broadcast_to(rng.normal(size=(n, 4)), (t, n, 4))
    options = dict(
        external=external, external_names=tuple(f"external{i}" for i in range(4))
    )
    c = FundamentalConfig()
    result = {"metadata": metadata(), "shape": [t, n, 45], "phases": {}}
    cache_dir = output.parent / "model-cache"
    cache = ModelCache(cache_dir)
    cache.clear()
    original_threads = portopt.get_num_threads()

    def build(name, **kwargs):
        timings = dict(
            exposures_at=0.0, cross_section_batch=0.0, estimate_covariances=0.0
        )
        originals = {key: getattr(series, key) for key in timings}

        def wrapper(key):
            def wrapped(*args, **kw):
                begin = perf_counter()
                value = originals[key](*args, **kw)
                timings[key] += perf_counter() - begin
                return value

            return wrapped

        begin = perf_counter()
        hashes = []
        with patch.multiple(series, **{key: wrapper(key) for key in timings}):
            for i, model in enumerate(series.model_series(p, **options, **kwargs)):
                hashes.append(digest(model))
                if (i + 1) % 250 == 0:
                    print(name, i + 1, round(perf_counter() - begin, 3), flush=True)
        result["phases"][name] = dict(
            seconds=perf_counter() - begin,
            components=timings,
            threads=portopt.get_num_threads(),
            models=len(hashes),
        )
        print(name, result["phases"][name], flush=True)
        output.write_text(json.dumps(result, indent=2))
        return hashes, model

    with threadpool_limits(limits=1):
        portopt.set_num_threads(portopt.available_threads())
        full, last = build("full", batch_size=64)
        portopt.set_num_threads(4)
        prefix, _ = build("cached_prefix", cache=cache, end=t - 1, batch_size=31)
        assert prefix == full[:-1]
        portopt.set_num_threads(1)
        appended, again = build("append", cache=cache, start=t - 1, batch_size=1)
        assert appended == full[-1:]
        result["bitwise"] = {
            "prefix_models": len(prefix),
            "appended_models": len(appended),
            "arrays_per_model": ["B", "F", "D"],
            "last_sha256": full[-1],
            "different_threads_and_chunk_sizes": True,
        }
        result["cache"] = cache.stats()
        result["cache"]["bytes_per_day"] = result["cache"]["bytes"] / t
        samples = []
        for _ in range(3):
            begin = perf_counter()
            loaded = cache.load(cache.entries[t - 1])
            assert digest(loaded[0]) == full[-1]
            samples.append(perf_counter() - begin)
        result["cache"]["latest_read_seconds"] = samples
        # 历史日不保存 B；读回包括从输入前缀重新生成对应暴露。
        _, _ = build("historical_read", cache=cache, start=t - 2, end=t - 1)
        key = cache.entries[t - 1]
        state_samples = []
        for _ in range(5):
            begin = perf_counter()
            cache.load(key, with_model=False)
            state_samples.append(perf_counter() - begin)
        result["cache"]["state_read_seconds"] = state_samples
        result["exposure_storage"] = {}
        weights = np.full(n, 1 / n)
        reference_variance = last.variance(weights)
        for dtype in (np.float64, np.float32):
            path = output.parent / ("exposure-" + np.dtype(dtype).name + ".npz")
            value = last.B.astype(dtype)
            np.savez_compressed(path, B=value)
            samples = []
            for _ in range(5):
                begin = perf_counter()
                with np.load(path) as data:
                    restored = data["B"].astype(float)
                samples.append(perf_counter() - begin)
            v = restored.T @ weights
            variance = float(v @ last.F @ v + (weights * weights) @ last.D)
            result["exposure_storage"][np.dtype(dtype).name] = {
                "raw_bytes": value.nbytes,
                "compressed_bytes": path.stat().st_size,
                "read_seconds": float(np.median(samples)),
                "max_absolute_exposure_error": float(np.max(np.abs(restored - last.B))),
                "portfolio_variance_relative_error": abs(
                    variance / reference_variance - 1
                ),
            }
        result.update(descriptor_profile(p, c, external, options["external_names"]))
        returns = adjusted_returns(p, t - 1)
        # 追加内核与全量完全相同，单独量单日回归，排除读盘/前缀摘要。
        state = cache.load(key, with_model=False)[1]
        times = []
        for _ in range(20):
            begin = perf_counter()
            portopt.regress(
                returns[-1:],
                again.B,
                state["cap"][None],
                state["estimation"][None],
                np.arange(1, 33),
                1,
            )
            times.append(perf_counter() - begin)
        result["single_day_regression_seconds"] = float(np.median(times))
    portopt.set_num_threads(original_threads)
    if not keep_cache:
        cache.clear()
    result["cache_cleaned_after_measurement"] = not keep_cache
    output.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(".cache/risk-bench/rebuild.json")
    )
    parser.add_argument("--days", type=int, default=4000)
    parser.add_argument("--stocks", type=int, default=5500)
    parser.add_argument("--keep-cache", action="store_true")
    args = parser.parse_args()
    run(args.output, args.days, args.stocks, keep_cache=args.keep_cache)
