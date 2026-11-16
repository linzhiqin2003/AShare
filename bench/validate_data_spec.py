"""D2 只读验证：显式来源路径、临时导出、仅保存汇总数字。"""

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from ashare.data.spec import check, open, seal
from ashare.data.spec.export_cn import CNAdapter
from ashare.data.spec.export_crypto import (
    HOUR_MAPPING,
    MINUTE_MAPPING,
    CryptoAdapter,
    MinuteAdapter,
)

ROOT = Path(__file__).resolve().parents[1]


def timed(function):
    started = time.perf_counter()
    result = function()
    return result, time.perf_counter() - started


def compare(actual, original, quote):
    original = original.astype(np.float64)
    actual = actual.astype(np.float64)
    mismatch = (np.isnan(actual) != np.isnan(original)) | (
        ~np.isnan(actual)
        & ~np.isnan(original)
        & (actual.view(np.uint64) != original.view(np.uint64))
    )
    return dict(
        cells=int(actual.size),
        quote_bitwise_mismatches=int((mismatch & quote).sum()),
        no_quote_mismatches=int((mismatch & ~quote).sum()),
        nan_positions_equal_on_quotes=bool(
            np.array_equal(np.isnan(actual[quote]), np.isnan(original[quote]))
        ),
    )


def validate(family, source, start, end, output):
    if not source.is_dir():
        return dict(status="skipped", reason="指定来源目录不可用，外置盘可能未挂载")
    kwargs = dict(start=start, end=end)
    if family == "cn":
        adapter, setup = timed(lambda: CNAdapter(source.parent, **kwargs))
        mapping = adapter.mapping
    elif family == "crypto":
        adapter, setup = timed(lambda: CryptoAdapter(source))
        mapping = HOUR_MAPPING
    else:
        adapter, setup = timed(
            lambda: MinuteAdapter(source, symbols=["BTCUSDT"], market="spot", **kwargs)
        )
        mapping = MINUTE_MAPPING
    result = dict(status="started", source_year=start[:4], adapter_seconds=setup)
    # 第 9.1 节直写路线：分别测量导出/校验，即使源违反规范也保留隔离诊断包。
    # 输出限定在 worktree/tmp，不走 build 的正式发布路径。
    if not output.resolve().is_relative_to((ROOT / "tmp").resolve()):
        raise ValueError("验证输出只能位于本 worktree tmp")
    started = time.perf_counter()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    description = adapter.describe()
    for table, partition in adapter.partitions():
        path = output / table / f"{partition}.parquet"
        path.parent.mkdir(exist_ok=True)
        adapter.read(table, partition).write_parquet(
            path, compression="zstd", statistics=True
        )
    (output / "datapackage.json").write_text(
        json.dumps(description, ensure_ascii=False)
    )
    sealed = seal(output)
    result["export_seconds"] = time.perf_counter() - started
    result["version"] = sealed["version"]
    result["package_bytes"] = sum(
        p.stat().st_size for p in output.rglob("*") if p.is_file()
    )
    report, result["check_seconds"] = timed(lambda: check(output))
    result["check_exit_code"] = report["exit_code"]
    result["check_failures"] = {
        x["code"]: x["count"] for x in report["checks"] if x["status"] == "fail"
    }
    result["warnings"] = {
        x["code"]: x["count"] for x in report["checks"] if x["status"] == "warning"
    }
    ds = open(output)
    panel, result["panel_seconds"] = timed(
        lambda: ds.panel(list(mapping), start=start, end=end)
    )
    _, result["panel_cached_seconds"] = timed(
        lambda: ds.panel(list(mapping), start=start, end=end)
    )
    result["shape"] = list(panel.shape)
    result["quote_cells"] = int(panel["has_quote"].sum())
    comparison = {}
    if family == "cn":
        axes = adapter.data.panel(["has_quote"], start, end)
        result["axes_equal"] = bool(
            np.array_equal(panel.dates, axes.dates)
            and np.array_equal(panel.symbols, axes.symbols)
        )
        if not result["axes_equal"]:
            return dict(result, status="failed")
        quote = axes["has_quote"]
        for name, raw in mapping.items():
            original = adapter.data.panel(
                [raw], panel.dates[0], panel.dates[-1], panel.symbols
            )[raw]
            comparison[name] = compare(panel[name], original, quote)
            if name in {"limit_up", "limit_down"}:
                zeros = (original == 0) & quote
                comparison[name]["raw_quote_bitwise_mismatches"] = comparison[name][
                    "quote_bitwise_mismatches"
                ]
                comparison[name]["raw_no_quote_mismatches"] = comparison[name][
                    "no_quote_mismatches"
                ]
                comparison[name]["source_zero_to_null"] = int(zeros.sum())
                normalized = np.where(original == 0, np.nan, original)
                comparison[name].update(compare(panel[name], normalized, quote))
    elif family == "crypto":
        original = adapter.data.arrays(
            [*set(mapping.values()), "quality", "lifecycle"],
            start=start,
            end=end,
            instruments=sorted(adapter.data.meta["instruments"]),
        )
        result["axes_equal"] = bool(
            np.array_equal(
                panel.dates.astype("datetime64[ms]").astype(np.int64), original["time"]
            )
            and panel.symbols.tolist() == original["instruments"]
        )
        if not result["axes_equal"]:
            return dict(result, status="failed")
        quote = (original["lifecycle"] == "active") & ((original["quality"] & 1) == 0)
        for name, raw in mapping.items():
            comparison[name] = compare(panel[name], original[raw], quote)
    else:
        from ashare.data.crypto.minute import bars
        from ashare.data.crypto.timeutil import millis

        source_frame = bars("BTCUSDT", "spot", "klines", start, end, root=source)
        result["axes_equal"] = bool(
            np.array_equal(
                panel.dates.astype("datetime64[ms]").astype(np.int64),
                np.arange(millis(start), millis(end), 60000),
            )
            and panel.symbols.tolist() == ["spot:BTCUSDT"]
        )
        if not result["axes_equal"]:
            return dict(result, status="failed")
        indices = np.searchsorted(
            panel.dates.astype("datetime64[ms]").astype(np.int64),
            source_frame["open_time"].to_numpy(),
        )
        quote = np.zeros(panel.shape, dtype=bool)
        quote[indices, 0] = True
        for name, raw in mapping.items():
            original = np.full(panel.shape, np.nan)
            original[indices, 0] = source_frame[raw].to_numpy()
            comparison[name] = compare(panel[name], original, quote)
    result["quote_mask_mismatches"] = int((panel["has_quote"] != quote).sum())
    result["fields"] = comparison
    result["quote_bitwise_mismatches"] = sum(
        x["quote_bitwise_mismatches"] for x in comparison.values()
    )
    result["no_quote_mismatches"] = sum(
        x["no_quote_mismatches"] for x in comparison.values()
    )
    result["status"] = (
        "pass"
        if report["ok"]
        and result["quote_bitwise_mismatches"] == 0
        and result["quote_mask_mismatches"] == 0
        else "failed"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cn", type=Path)
    parser.add_argument("--crypto", type=Path)
    parser.add_argument("--crypto-minute", type=Path)
    args = parser.parse_args()
    output = ROOT / "bench/results/data_spec_validation.json"
    results = json.loads(output.read_text()) if output.exists() else {}
    for family, path, start, end in [
        ("cn", args.cn, "2024-01-01", "2024-12-31"),
        ("crypto", args.crypto, "2025-01-01", "2026-01-01"),
        ("crypto_minute", args.crypto_minute, "2025-01-01", "2026-01-01"),
    ]:
        if path is None:
            continue
        print(f"{family}: starting", flush=True)
        try:
            result = validate(
                family, path, start, end, ROOT / "tmp" / ("data-spec-" + family)
            )
        except Exception as exc:
            result = dict(status="failed", error_type=type(exc).__name__)
            # 不把来源异常里的逐行行情复制到报告。
        results[family] = result
        output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
        print(f"{family}: {result['status']}", flush=True)
    return int(any(r["status"] == "failed" for r in results.values()))


if __name__ == "__main__":
    raise SystemExit(main())
