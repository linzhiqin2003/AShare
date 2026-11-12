"""候选表达式对独立参考实现的逐格核对。

    python -m ashare.factor.formula.verify <因子集> [--days 750] [--warmup 260]
        [--symbols N] [--factors 名字,名字] [--option 键=值] [--reference 包.模块]
        [--workers 6] [--out 结果.json]

因子集来自 ``plugins.discover()``（内置的 ``example`` 加上外部因子包挂上来的）；参考实现
默认用因子集登记的模块，``--reference`` 可以换。候选和参考在同一段数据上计算（最近
``days + warmup`` 个交易日），只比较最后 ``days`` 行。判定规则：

- 一致：NaN 掩码逐格相同，且 ``max |a−b|/(1+|b|) ≤ 1e-12``；
- 仅浮点差：掩码相同，且 ``≤ 1e-6``；
- 每天的截面秩相关只报告最小值，不参与判定（浮点差会拆开并列，截面小时秩相关会明显下降）；
- 不一致：其余。

因子集登记的「已知例外」（``known_exceptions``）只对满足明确条件、按正常规则不过关、
且候选恰为声明值的格子生效：这些格子从统计里拿掉、单独计数，结论写成
「一致（已知例外 N 格）」这一类，与纯粹的结论分开统计；不会整条因子豁免。

参考实现给出的 ±inf 按 NaN 处理。结果写成 JSON：运行信息加每个因子一行统计。
"""

import argparse
import importlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import panelops

from ..engine import Evaluator
from . import plugins
from .inputs import PublicSource, load_inputs, reference_frames

EXACT = 1e-12
FLOAT = 1e-6


def _mixed_err(a, b):
    both = np.isfinite(a) & np.isfinite(b)
    diff = np.abs(np.subtract(a, b, where=both, out=np.zeros_like(a)))
    return np.where(both, diff / (1 + np.abs(b, where=both, out=np.ones_like(b))), 0.0)


def _where(dates, symbols, t, j, a, b):
    return {
        "date": None if dates is None else str(dates[t]),
        "symbol": None if symbols is None else str(symbols[j]),
        "candidate": float(a[t, j]),
        "reference": float(b[t, j]),
    }


def apply_exceptions(candidate, reference, exceptions, dates=None, symbols=None):
    """按已知例外拿掉格子；返回 (候选, 参考, 例外记录)，拿掉的格子两边都置 NaN。

    ``exceptions``：``[(KnownException, 条件面板)]``。只拿掉条件为真、按正常规则不过关
    （掩码不同或误差超过「一致」阈值）、且候选逐位等于声明值的格子。
    """
    a = np.array(candidate, dtype=float)
    b = np.array(reference, dtype=float)
    fa, fb = np.isfinite(a), np.isfinite(b)
    failing = (fa != fb) | (_mixed_err(a, b) > EXACT)
    records = []
    for rule, condition in exceptions:
        cond = np.asarray(condition, dtype=float)
        if cond.shape != a.shape:
            raise ValueError(f"{rule.ref} 的条件形状 {cond.shape} 与候选不同")
        holds = np.isfinite(cond) & (cond != 0)
        rescued = failing & holds & (a == rule.candidate)
        cells = np.argwhere(rescued)
        records.append(
            {
                "ref": rule.ref,
                "description": rule.description,
                "cells": int(len(cells)),
                "condition_cells": int(holds.sum()),
                "samples": [_where(dates, symbols, t, j, a, b) for t, j in cells[:5]],
            }
        )
        a[rescued] = np.nan
        b[rescued] = np.nan
        failing &= ~rescued
    return a, b, records


def compare(candidate, reference, dates=None, symbols=None, exceptions=()):
    """两张同形状面板的逐格比较统计；不含结论。``exceptions`` 见 ``apply_exceptions``。"""
    a = np.asarray(candidate, dtype=float)
    b = np.asarray(reference, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"形状不同：{a.shape} 与 {b.shape}")
    records = []
    if exceptions:
        a, b, records = apply_exceptions(a, b, exceptions, dates, symbols)
    fa, fb = np.isfinite(a), np.isfinite(b)
    both = fa & fb
    stats = {
        "cells": int(a.size),
        "valid_cells": int(fa.sum()),
        "reference_valid_cells": int(fb.sum()),
        "mask_mismatch": int((fa != fb).sum()),
        "candidate_only": int((fa & ~fb).sum()),
        "reference_only": int((fb & ~fa).sum()),
        "max_abs_err": 0.0,
        "max_rel_err": 0.0,
        "max_err": 0.0,
        "cells_over_exact": 0,
        "cells_over_float": 0,
        "min_rank_corr": None,
        "worst": None,
        "exception_cells": sum(r["cells"] for r in records),
        "known_exceptions": records,
    }
    if both.any():
        diff = np.abs(np.subtract(a, b, where=both, out=np.zeros_like(a)))
        scale = np.abs(b, where=both, out=np.ones_like(b))
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(both & (scale > 0), diff / scale, 0.0)
            rel = np.where(both & (scale == 0), diff, rel)
        mixed = np.where(both, diff / (1 + scale), 0.0)
        stats["max_abs_err"] = float(diff.max())
        stats["max_rel_err"] = float(rel.max())
        stats["max_err"] = float(mixed.max())
        stats["cells_over_exact"] = int((mixed > EXACT).sum())
        stats["cells_over_float"] = int((mixed > FLOAT).sum())
        t, j = np.unravel_index(int(np.argmax(mixed)), a.shape)
        stats["worst"] = _where(dates, symbols, t, j, a, b)
    aa = np.ascontiguousarray(np.where(fa, a, np.nan))
    bb = np.ascontiguousarray(np.where(fb, b, np.nan))
    daily = panelops.rank_ic(aa, bb)
    if np.isfinite(daily).any():
        stats["min_rank_corr"] = float(np.nanmin(daily))
    if stats["mask_mismatch"]:
        t, j = np.argwhere(fa != fb)[0]
        stats["first_mask_mismatch"] = _where(dates, symbols, t, j, a, b)
    return stats


def base_verdict(stats):
    """按 CONVENTIONS 第 6 节的阈值给结论（不含例外说明）。"""
    if stats["mask_mismatch"]:
        return "不一致"
    if stats["max_err"] <= EXACT:
        return "一致"
    if stats["max_err"] <= FLOAT:
        return "仅浮点差"
    return "不一致"


def verdict(stats):
    """结论；用到已知例外时写成「一致（已知例外 N 格）」这一类。"""
    base = base_verdict(stats)
    n = stats.get("exception_cells", 0)
    if base == "不一致" or not n:
        return base
    return f"{base}（已知例外 {n} 格）"


def category(row):
    """汇总用的类别：一致、一致（含已知例外）、仅浮点差、仅浮点差（含已知例外）、不一致。"""
    base = base_verdict(row)
    if base != "不一致" and row.get("exception_cells", 0):
        return f"{base}（含已知例外）"
    return base


# ---- 参考实现的并行计算（按证券分块，每个进程自己取数，数据层是只读内存映射） ----

_WORKER = {}


def _init_worker(load_kwargs):
    _WORKER["source"] = load_inputs(**load_kwargs)


def _columns(source, columns):
    return PublicSource(
        source.dates,
        source.symbols[columns],
        {
            k: np.ascontiguousarray(np.asarray(v)[:, columns])
            for k, v in source.arrays.items()
        },
        source.version,
        source.benchmark,
        source.benchmark_open,
        source.benchmark_close,
    )


def _compute_many(module, frames, names, options=None):
    """参考模块提供 ``compute_many`` 就用它；否则逐个调用 ``REFERENCES[名字](frames)``。

    ``options`` 是因子集选项（核对时的 ``--option``），原样作为关键字参数传给参考实现；
    没有选项时不传。
    """
    options = options or {}
    if hasattr(module, "compute_many"):
        return module.compute_many(frames, names, **options)
    out = {}
    for name in names:
        result = module.REFERENCES[name](frames, **options)
        values = np.asarray(result, dtype=float)
        if values.shape != frames["close"].shape:
            raise ValueError(f"{name} 的参考结果形状 {values.shape} 与输入不同")
        out[name] = values
    return out


def _reference_chunk(module_name, names, columns, keep, source=None, options=None):
    module = importlib.import_module(module_name)
    source = _WORKER["source"] if source is None else source
    frames = reference_frames(_columns(source, np.asarray(columns)))
    out = _compute_many(module, frames, names, options)
    return {
        name: np.asarray(values, dtype=float)[-keep:] for name, values in out.items()
    }


def reference_many(
    module_name,
    names,
    source,
    keep,
    *,
    workers=0,
    load_kwargs=None,
    chunk=None,
    options=None,
):
    """参考实现模块的 ``compute_many``。

    ``chunk`` 为整数时按证券分块（只适用于纯时序的因子集），``workers > 0``
    时多进程；为 None 时整个截面一次算完（含截面排名、行业中性化的因子集必须如此）。
    """
    n = len(source.symbols)
    chunk = n if chunk is None else chunk
    blocks = [list(range(i, min(n, i + chunk))) for i in range(0, n, chunk)]
    out = {name: np.full((keep, n), np.nan) for name in names}
    if workers and load_kwargs is not None and len(blocks) > 1:
        with ProcessPoolExecutor(
            workers, initializer=_init_worker, initargs=(load_kwargs,)
        ) as pool:
            futures = [
                (
                    b,
                    pool.submit(
                        _reference_chunk, module_name, names, b, keep, None, options
                    ),
                )
                for b in blocks
            ]
            for block, future in futures:
                for name, values in future.result().items():
                    out[name][:, block] = values
    else:
        for block in blocks:
            for name, values in _reference_chunk(
                module_name, names, block, keep, source, options
            ).items():
                out[name][:, block] = values
    return out


def verify(
    candidates,
    reference_module,
    source,
    *,
    keep,
    workers=0,
    load_kwargs=None,
    group=20,
    formulas=None,
    chunk=None,
    exceptions=None,
    options=None,
    log=print,
):
    """逐组计算参考与候选并比较；返回每个因子一行的统计（含结论）。

    ``candidates``：``{名字: Expr}``；``reference_module``：模块名，提供
    ``compute_many(frames, names) -> {名字: ndarray}`` 或 ``REFERENCES = {名字: compute}``
    （``compute(frames) -> DataFrame``），名字与 candidates 相同。``exceptions``：
    ``{名字: (KnownException, ...)}``，条件表达式与候选在同一份数据上求值。
    """
    exceptions = {k: v for k, v in (exceptions or {}).items() if k in candidates}
    names = list(candidates)
    dates, symbols = source.dates[-keep:], source.symbols
    evaluator = Evaluator()
    rows = []
    for start in range(0, len(names), group):
        part = names[start : start + group]
        t0 = time.perf_counter()
        refs = reference_many(
            reference_module,
            part,
            source,
            keep,
            workers=workers,
            load_kwargs=load_kwargs,
            chunk=chunk,
            options=options,
        )
        t1 = time.perf_counter()
        wanted = {n: candidates[n] for n in part}
        for n in part:
            for i, rule in enumerate(exceptions.get(n, ())):
                wanted[f"{n}#exception{i}"] = rule.condition
        values = dict(evaluator.evaluate_many(wanted, source))
        for name in part:
            rules = [
                (rule, values[f"{name}#exception{i}"][-keep:])
                for i, rule in enumerate(exceptions.get(name, ()))
            ]
            stats = compare(values[name][-keep:], refs.pop(name), dates, symbols, rules)
            stats["verdict"] = verdict(stats)
            stats["name"] = name
            if formulas:
                stats["formula"] = formulas.get(name)
            rows.append(stats)
        t2 = time.perf_counter()
        log(
            f"[{start + len(part)}/{len(names)}] 参考 {t1 - t0:.1f}s 候选+比较 {t2 - t1:.1f}s"
        )
    return rows


def summary(rows):
    counts = {}
    for row in rows:
        key = category(row)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _options(pairs):
    options = {}
    for pair in pairs or ():
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise SystemExit(f"--option 的格式是 键=值：{pair!r}")
        options[key] = value
    return options


def _candidates(args):
    """因子集 → (候选, 原文, 参考模块, 分块, 已知例外)。"""
    try:
        chosen = plugins.factor_set(args.set)
    except KeyError as error:
        raise SystemExit(str(error)) from None
    registry = chosen.registry(**_options(args.option))
    wanted = None if not args.factors else set(args.factors.split(","))
    candidates = {
        f.name: f.expression for f in registry if wanted is None or f.name in wanted
    }
    if wanted and len(candidates) != len(wanted):
        raise SystemExit(f"未知因子：{sorted(wanted - set(candidates))}")
    formulas = {f.name: f.description for f in registry}
    module = args.reference or chosen.reference
    if not module:
        raise SystemExit(f"因子集 {args.set} 没有登记参考实现，用 --reference 指定")
    # 换了参考实现就不按证券分块（不知道它是否只有时序运算）。
    chunk = chosen.chunk if not args.reference else None
    return candidates, formulas, module, chunk, dict(chosen.exceptions)


def main(argv=None):
    parser = argparse.ArgumentParser(description="公式型因子：候选对参考实现逐格核对")
    parser.add_argument("set", help="因子集名字（plugins.discover() 里的）")
    parser.add_argument(
        "--reference", help="参考实现模块（提供 REFERENCES 或 compute_many）"
    )
    parser.add_argument(
        "--option",
        action="append",
        help="传给因子集注册表的选项，键=值，可重复",
    )
    parser.add_argument("--days", type=int, default=750)
    parser.add_argument("--warmup", type=int, default=260)
    parser.add_argument(
        "--symbols", type=int, default=None, help="只取前 N 只（试跑用）"
    )
    parser.add_argument("--factors", default=None, help="逗号分隔的因子名")
    parser.add_argument("--benchmark", default="000300.SH")
    parser.add_argument(
        "--root", default=None, help="数据根目录（默认 ASHARE_DATA_DIR 或仓库 data/）"
    )
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2)
    )
    parser.add_argument("--group", type=int, default=20)
    parser.add_argument("--out", default=None, help="JSON 结果路径")
    args = parser.parse_args(argv)

    candidates, formulas, module, chunk, exceptions = _candidates(args)
    load_kwargs = dict(
        root=args.root,
        days=args.days,
        warmup=args.warmup,
        symbols=args.symbols,
        benchmark=args.benchmark,
    )
    started = time.perf_counter()
    source = load_inputs(**load_kwargs)
    print(
        f"数据 {source.version}：{source.shape}，核对最后 {args.days} 行，{len(candidates)} 个因子"
    )
    rows = verify(
        candidates,
        module,
        source,
        keep=args.days,
        workers=args.workers,
        load_kwargs=load_kwargs,
        group=args.group,
        formulas=formulas,
        chunk=chunk,
        exceptions=exceptions,
        options=_options(args.option),
    )
    elapsed = time.perf_counter() - started
    counts = summary(rows)
    result = {
        "set": args.set,
        "options": _options(args.option),
        "data_version": source.version,
        "dates": [str(source.dates[-args.days]), str(source.dates[-1])],
        "symbols": len(source.symbols),
        "warmup": args.warmup,
        "reference": module,
        "panelops_build_id": panelops.build_id,
        "thresholds": {"exact": EXACT, "float": FLOAT},
        "elapsed_seconds": round(elapsed, 1),
        "load_average": list(os.getloadavg()),
        "summary": counts,
        "factors": rows,
    }
    print(
        "结论："
        + "，".join(f"{k} {v}" for k, v in sorted(counts.items()))
        + f"；用时 {elapsed:.0f}s"
    )
    for row in rows:
        if row.get("exception_cells"):
            print(f"  {row['verdict']} {row['name']}")
        if row["verdict"] == "不一致":
            print(
                f"  不一致 {row['name']}：掩码不同 {row['mask_mismatch']}，max_err {row['max_err']:.3g}"
            )
    if args.out:
        with open(args.out, "w") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=1)
        print(f"写入 {args.out}")
    return 0 if "不一致" not in counts else 1


if __name__ == "__main__":
    sys.exit(main())
