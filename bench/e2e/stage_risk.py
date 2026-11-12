"""阶段 4：基本面风险模型在真实面板上的全量重建、追加一天、读缓存和内部各步占比。"""

import hashlib
import shutil
import time

import numpy as np
from common import Phases, digest, measure
from risk_adapter import load

import ashare.data
import ashare.risk.fundamental as fundamental
import ashare.risk.series as series
from ashare.risk import ModelCache, model_series
from ashare.risk.data import FIELDS

MONTH = 21
PERIODS = 61


def scale(ctx):
    """默认全历史全市场；--quick 为最近 300 日。"""
    return {"days": 300 if ctx["quick"] else None, "symbols": None}


def instrument(phases):
    patches = [
        (series, "exposures_at"),
        (series, "cross_section_batch"),
        (series, "estimate_covariances"),
        (series, "cross_sectional_bias"),
        (series, "RiskModel"),
        (fundamental, "factor_covariance"),
        (fundamental, "specific_variance"),
        (series.RollingDescriptors, "at"),
        (series.ModelCache, "save"),
        (series.ModelCache, "load"),
    ]
    original = [(owner, name, getattr(owner, name)) for owner, name in patches]
    for owner, name, fn in original:
        label = name if owner in (series, fundamental) else f"{owner.__name__}.{name}"
        setattr(owner, name, phases.wrap(label, fn))
    return lambda: [setattr(o, n, f) for o, n, f in original]


def prefix_hash_cost(source, end):
    """复现 model_series 逐日输入前缀哈希的工作量，单独计时。"""
    prefix = b""
    for day in range(end):
        h = hashlib.sha256(prefix)
        h.update(str(source.dates[day]).encode())
        for field in FIELDS:
            try:
                value = np.ascontiguousarray(source[field][day])
            except KeyError:
                continue
            h.update(field.encode() + value.dtype.str.encode() + value.tobytes())
        prefix = h.digest()
    return prefix.hex()[:24]


def summarize(model):
    return digest(model.B, model.F, model.D)


def run(ctx):
    sizes = scale(ctx)
    built, (source, config) = measure(
        lambda: load(ctx["data"], sizes["days"], sizes["symbols"]), 1
    )
    t, n = len(source.dates), len(source.symbols)
    shape = {"days": t, "symbols": n, "factors": 1 + len(config.industry_codes) + 1 + 8}
    cases = [{"case": "adapter_load", "scale": shape, **built}]
    cache_dir = ctx["work"] / "risk-cache"
    shutil.rmtree(cache_dir, ignore_errors=True)
    phases = Phases()
    restore = instrument(phases)
    months = {}

    def rebuild():
        last = None
        for day, model in enumerate(
            model_series(source, config=config, cache=ModelCache(cache_dir), end=t - 1)
        ):
            if day % MONTH == (t - 2) % MONTH:
                months[day] = (model.B, model.F, model.D)
                months.pop(min(months)) if len(months) > PERIODS else None
            last = model
        return summarize(last)

    stats, value = measure(rebuild, 1)
    restore()
    cases.append(
        {
            "case": "rebuild_full_with_cache",
            "scale": {**shape, "days": t - 1},
            **stats,
            "summary": {"last_model": value},
            "detail": {"steps": phases.report()},
        }
    )
    handoff = ctx["work"] / "handoff"
    handoff.mkdir(exist_ok=True)
    keep = sorted(months)[-PERIODS:]
    np.savez(
        handoff / "risk_months.npz",
        days=np.array(keep),
        offset=len(ashare.data.open(ctx["data"]).panel("close").dates) - t,
        B=np.array([months[d][0] for d in keep]),
        F=np.array([months[d][1] for d in keep]),
        D=np.array([months[d][2] for d in keep]),
        symbols=np.array(source.symbols),
        version=source.version,
    )
    stats, value = measure(lambda: prefix_hash_cost(source, t - 1), 1)
    cases.append(
        {
            "case": "prefix_hash_replay",
            "scale": shape,
            **stats,
            "summary": {"prefix": value},
        }
    )

    snapshot = ctx["work"] / "risk-cache-snapshot"
    shutil.rmtree(snapshot, ignore_errors=True)
    shutil.copytree(cache_dir, snapshot)

    def reset():
        shutil.rmtree(cache_dir, ignore_errors=True)
        shutil.copytree(snapshot, cache_dir)

    def append():
        models = list(
            model_series(
                source, config=config, cache=ModelCache(cache_dir), start=t - 1
            )
        )
        return summarize(models[-1])

    stats, value = measure(append, ctx["repeat"](2), setup=reset)
    cases.append(
        {"case": "append_one_day", "scale": shape, **stats, "summary": {"model": value}}
    )
    stats, value = measure(append, ctx["repeat"](2))
    cases.append(
        {
            "case": "read_cached_last_day",
            "scale": shape,
            **stats,
            "summary": {"model": value},
        }
    )
    size = sum(p.stat().st_size for p in cache_dir.rglob("*") if p.is_file())
    cases[-1]["detail"] = {"cache_bytes": size, "files": len(list(cache_dir.iterdir()))}
    shutil.rmtree(snapshot, ignore_errors=True)
    shutil.rmtree(cache_dir, ignore_errors=True)
    return cases


def build_months(ctx, days):
    """组合与端到端阶段在没有交接文件时自建：最近 days 日的月末模型。"""
    source, config = load(ctx["data"], days)
    t = len(source.dates)
    out = {}
    start = time.perf_counter()
    for day, model in enumerate(model_series(source, config=config)):
        if day % MONTH == (t - 2) % MONTH:
            out[day] = (model.B, model.F, model.D)
    return source, out, time.perf_counter() - start
