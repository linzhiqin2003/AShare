"""Freeze compact measurements and render both single/default-thread decisions."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = Path(__file__).parent / "results"


def minimum(rows, names):
    values = [r for r in rows if r["method"] in names and r["status"] == "ok"]
    return min(values, key=lambda r: r["median_ms"]) if values else None


def cell(row):
    if row is None:
        return "—"
    return f"{row['median_ms']:.2f} / {row['incremental_mib']:.1f}"


def selections(rows):
    by = defaultdict(list)
    for row in rows:
        by[row["op"]].append(row)
    result = {}
    for op, values in by.items():
        # The single-thread merge criterion excludes pandas, which remains a reference.
        np_single = (
            "numpy_1" if any(v["method"] == "numpy_1" for v in values) else "numpy"
        )
        one = minimum(values, [np_single, "bottleneck", "polars_1", "polars_long_1"])
        default = minimum(
            values,
            ["numpy", "bottleneck", "pandas", "polars_default", "polars_long_default"],
        )
        r1 = minimum(values, ["rust_1"])
        rd = minimum(values, ["rust_default"])
        n1 = minimum(values, [np_single])
        nd = minimum(values, ["numpy"])
        result[op] = {
            "single_baseline": one["method"] if one else None,
            "default_baseline": default["method"] if default else None,
            "single_ratio": r1["median_ms"] / one["median_ms"] if r1 and one else None,
            "default_ratio": rd["median_ms"] / default["median_ms"]
            if rd and default
            else None,
            "single_runtime": "rust"
            if r1 and (not n1 or r1["median_ms"] <= n1["median_ms"])
            else "numpy",
            "default_runtime": "rust"
            if rd and (not nd or rd["median_ms"] <= nd["median_ms"])
            else "numpy",
        }
    return by, result


def render(report):
    rows = report["measurements"]
    by, decisions = selections(rows)
    lines = [
        "# panelops 第二轮基准",
        "",
        "代表配置为 T=4000、N=5500、W=20、缺失 5%，min_periods=10；截面与逐元素算子没有窗口，滞后算子使用 20 行，因子自相关/分组换手使用 1 行。输入为固定种子的正态分布；逻辑和 where 条件为各约一半的 0/1，缺失保留；power 的负底非整数指数按定义域置缺失。每项独立进程，预热后三次取中位数。单元格为 **毫秒 / 输出之外增量 RSS MiB**。",
        "",
        f"机器：{report['machine']}；CPU 可用数 {report['cpus']}；Python {report['python']}；版本：{report['versions']}。",
        "",
        "机器同时存在其他任务，部分基线测量期间也在编译和做验证；CPU 频率、缓存和分配器状态会干扰数字。冻结值保留实际测量，不根据探索结果择优替换。请在空闲时重测。",
        "",
        "内存口径：先原位生成输入，首调用前后读进程 RSS 高水位，差值减去保留输出的字节数，负值截为 0。它是受页粒度、分配器复用和惰性提交影响的**下界估计**，0 不表示零工作区；Rust 的额外状态上界另列在后文。所有库使用同一估算方法。输入、导入库的常驻内存不算入增量。原始 RSS 也保存在冻结 JSON。",
        "",
        "Polars 1 列的线程池固定为 1；默认列保留库默认配置。宽表和长表都测，表内列出较快布局，全部候选留在 JSON。NumPy 只有截面中性化调用 Accelerate 的线性代数；它另外测 VECLIB_MAXIMUM_THREADS=1。其他 NumPy 算子不调用 BLAS。实际参与线程数通过 macOS libproc 的逐线程 CPU 时间增量计数（阈值 0.25 ms），统计在计时区间之外，包含调用线程。池大小不等于实际参与数。",
        "",
        "pandas 没有原生归约时使用 rolling.apply；MAD/位置归约等回调在大面板可能超过进程时限。— 表示该库没有相应实现或超过时限，具体原因列在后面。基线直接使用各库原生输出布局，可能为 F 连续；Rust 输出固定为 C 连续。",
        "",
        "| 算子 | NumPy | BN | Polars 1（布局） | Polars 默认（布局） | pandas | Rust 1 | Rust 默认 | 选用 1 / 默认 | Rust/最快单线程 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|",
    ]
    for op, values in by.items():
        n = minimum(
            values,
            ["numpy_1" if any(v["method"] == "numpy_1" for v in values) else "numpy"],
        )
        p1 = minimum(values, ["polars_1", "polars_long_1"])
        pd = minimum(values, ["polars_default", "polars_long_default"])

        def observed(v):
            counts = sorted(set(v.get("observed_cpu_threads", []))) if v else []
            return "〔线程 " + "/".join(map(str, counts)) + "〕" if counts else ""

        def plcell(v):
            return (
                cell(v)
                + observed(v)
                + (
                    "（长）"
                    if v and "_long_" in v["method"]
                    else "（宽/原生）"
                    if v
                    else ""
                )
            )

        d = decisions[op]
        lines.append(
            "| "
            + " | ".join(
                [
                    op,
                    cell(n) + observed(n),
                    cell(minimum(values, ["bottleneck"])),
                    plcell(p1),
                    plcell(pd),
                    cell(minimum(values, ["pandas"])),
                    cell(minimum(values, ["rust_1"])),
                    cell(minimum(values, ["rust_default"])),
                    d["single_runtime"] + " / " + d["default_runtime"],
                    f"{d['single_ratio']:.3f}"
                    if d["single_ratio"] is not None
                    else "—",
                ]
            )
            + " |"
        )
    single = sum(
        d["single_ratio"] is not None and d["single_ratio"] <= 1
        for d in decisions.values()
    )
    default = sum(
        d["default_ratio"] is not None and d["default_ratio"] < 1
        for d in decisions.values()
    )
    passes = sum(
        d["single_ratio"] is not None and d["single_ratio"] <= 1.1
        for d in decisions.values()
    )
    lines += [
        "",
        f"按验收口径（NumPy、bottleneck、Polars 1），Rust 单线程最快 **{single}/{len(decisions)}**，在最快单线程基线的 110% 内 **{passes}/{len(decisions)}**；Rust 默认线程快于全部库默认配置 **{default}/{len(decisions)}**。选用列只在可作为运行依赖的 NumPy 和 Rust 间比较；不引入 bottleneck/polars/pandas 运行依赖。",
        "",
        "## 未达到单线程门槛的项目",
        "",
        "| 算子 | Rust 1 ms | 最快单线程基线 | 差距 | 已尝试及剩余原因 |",
        "|---|---:|---|---:|---|",
    ]
    explanations = {
        "ts_std": "二矩专用状态、分离/合并循环、逐行输入切片、无掩码 zip、512 行重建和 1024 列块均实测；保留精确常数判断及误差守卫的额外工作，尚未消除差距。",
        "ts_var": "与 std 共用二矩专用内核；上述数值保障仍有成本。",
        "ts_zscore": "与二矩状态融合，避免生成均值/标准差两份面板；保留当日缺失和误差守卫。",
        "ts_delay": "已用连续切片的单次 zip 遍历替代逐格索引和复制后二次清洗；仍包含无穷值清洗与新数组分配。",
        "ts_delta": "已单次遍历两个连续输入切片；仍包含输入有效性检查。",
        "ts_sum": "已用行流式 SoA、补偿求和、512 行重建、事件恢复；补偿和守卫有额外成本。",
        "ts_mean": "与 sum 同一流式更新；保留补偿求和与窗口有效样本计数。",
        "ts_sumif": "已融合条件筛选和成对有效计数，避免生成条件面板；采用逐行 SoA、补偿更新、512 行兜底和事件恢复。两路历史输入和精度守卫仍有额外成本。",
        "ts_quantile": "小窗口固定有序缓冲，长窗口带子树大小的 treap；已对比逐窗选择和 Vec 插删。",
        "ts_median": "与 quantile 共用顺序统计状态，已试固定有序缓冲与长窗口 treap。",
        "cs_neutralize": "FWL 先行业去均值，再对 K 个暴露做归一化带主元重正交 QR；已去除行业哑变量矩阵。",
        "rank_ic": "采用线程本地向量与排序索引；成对缺失需要在交集上分别排名，不能直接套用全截面排名。",
        "factor_autocorr": "短滞后缓存排序，交集上重新计并列名次；不生成整面板 lag 数组。",
        "group_turnover": "短滞后复用分组标签；大滞后限制缓存占用，改为按需排序。",
        "exp": "融合有限值检查与逐元素 libm exp；NumPy/Accelerate 的向量数学实现可能更快。",
        "power": "融合缺失/定义域检查；标量 powf 未使用平台专有向量数学库。",
        "signed_power": "融合 sign 与 powf；已去掉 sign 的数据相关分支。",
        "sign": "已改为两次比较转浮点的无分支公式。",
    }
    for op, d in decisions.items():
        if d["single_ratio"] is not None and d["single_ratio"] > 1.1:
            values = by[op]
            best = minimum(values, [d["single_baseline"]])
            rust = minimum(values, ["rust_1"])
            lines.append(
                f"| {op} | {rust['median_ms']:.3f} | {best['method']} {best['median_ms']:.3f} ms | +{(d['single_ratio'] - 1) * 100:.1f}% | {explanations.get(op, '已采用单态化入口、连续输出和线程本地状态；当前测量仍落后，保留数值及缺失语义，未降低精度换速度。')} |"
            )
    near = [
        op
        for op, d in decisions.items()
        if d["single_ratio"] is not None and 1 < d["single_ratio"] <= 1.1
    ]
    if near:
        lines += ["", "其余未取得单线程第一、但满足 10% 门槛的项目："]
        for op in near:
            d = decisions[op]
            reason = (
                "已用四矩流式状态、补偿求和和精度守卫"
                if op == "ts_kurt"
                else "已去除有效值暂存向量，采用直接中心化两遍扫描"
                if op == "cs_demean"
                else "已使用上述专用内核"
            )
            lines.append(
                f"- {op}：比 {d['single_baseline']} 慢 {(d['single_ratio'] - 1) * 100:.2f}%；{reason}。仍比 NumPy 快，按允许的运行依赖选择 Rust；不宣称它比所有基线快。"
            )
    lines += [
        "",
        "## 线程实测",
        "",
        "| 实现 | 观测参与线程数（各算子/三次采样并集） | 配置线程池 |",
        "|---|---|---|",
    ]
    for method in dict.fromkeys(r["method"] for r in rows):
        values = [r for r in rows if r["method"] == method and r["status"] == "ok"]
        observed = sorted(
            {v for r in values for v in r.get("observed_cpu_threads", [])}
        )
        pools = (
            sorted({r["polars_pool"] for r in values if "polars_pool" in r})
            if method.startswith("polars")
            else sorted(
                {r["rust_pool"] for r in values if r.get("rust_pool") is not None}
            )
            if method.startswith("rust")
            else []
        )
        lines.append(f"| {method} | {observed} | {pools or '不适用'} |")
    neutral = [
        r
        for r in rows
        if r["op"] == "cs_neutralize"
        and r["method"].startswith("numpy")
        and r["status"] == "ok"
    ]
    for r in neutral:
        lines.append(
            f"\nAccelerate 中性化 {r['method']}：{r['median_ms']:.3f} ms，额外 RSS {r['incremental_mib']:.1f} MiB，实际参与线程 {r['observed_cpu_threads']}，VECLIB_MAXIMUM_THREADS={r['accelerate_limit']}。"
        )
    lines += [
        "",
        "## Rust 输出之外的状态",
        "",
        "没有 Vec<Vec<f64>> 形式的全输出临时面板。列块只保存对最终输出每行的可变切片；并行块互不相交，所有源代码禁止 unsafe。",
        "",
        "| 内核 | 主要状态与上界（不含输出；B 为列块宽，P 为同时运行块数） |",
        "|---|---|",
        "| delay/delta、逐元素 | O(1) 运算状态；Rayon 调度和线程池为常驻开销 |",
        "| sum/mean/count/sumif | 32 B 字节状态/工作块；总峰值约 32·B·P，严重消去时另有每块 O(W) 浮点展开暂存，加输出切片索引 16·T·ceil(N/B) |",
        "| std/var/zscore | 64 B 字节状态/工作块，加同一输出切片索引 |",
        "| EMA/ffill | 24 B 字节状态/工作块，加切片索引 |",
        "| decay | 64 B 字节状态/工作块，加切片索引 |",
        "| 配对/高阶矩/回归 | 每列 5 或 4 个补偿矩、误差界、锚点、计数及常数运行长度；O(B·P)，没有整面板中间值 |",
        "| min/max/arg/idx | 前一块后缀值、下标、有效数共 24·W·B，加当前前缀约 24·B；乘以 P，加切片索引 |",
        "| rank/product | W·B 个 f64 环形状态，加每列计数；O(W·B·P) |",
        "| median/quantile | 小窗口环形数据及固定有序缓冲各 W·B；长窗口 treap 每节点 6 个 8 字节字段；O(W·B·P) |",
        "| MAD | W·B 环形数据及每工作块 W 项选择暂存；O(W·B·P) |",
        "| 截面 | scale/demean/zscore 无掩码时仅标量状态；其余每线程 O(N) 暂存；中性化为 O(N·K+G·K)，不构造 N×G 行业哑变量 |",
        "| CSR targets | 所有调仓日的选择向量目前保留候选容量，最坏约 8·D·N 字节，加每线程 O(N) 排序与 CSR Vec 增长容量；D 为调仓日数 |",
        "| 因子自相关/换手 | 每工作块最多缓存 min(lag,32) 个 O(N) 排序/标签；大滞后不缓存整历史 |",
        "",
        "上述是逻辑工作区公式，解释了 RSS 下界可能低估暂存的情况。默认线程下 B=ceil(N/P) 向上对齐 64，最大 1024；单线程也按最多 1024 列分块。无缺失/无掩码分支只省有效性判断，不改变加减顺序。",
        "",
        "## 不可用或超时记录",
        "",
    ]
    failures = defaultdict(set)
    for r in rows:
        if r["status"] != "ok":
            failures[r["op"]].add(
                r["method"] + ": " + r["status"] + " " + r.get("error", "")
            )
    for op, items in failures.items():
        lines.append("- " + op + "：" + "；".join(sorted(items)))
    lines += [
        "",
        "## 重跑",
        "",
        "```sh",
        "make panelops-bench",
        "make panelops-crosscheck",
        "make panelops-mutate",
        "# Full grid (much longer):",
        "uv run --no-sync python bench/panelops/run.py --full --output .cache/panelops/full.json",
        "# Re-render a frozen report:",
        "uv run --no-sync python bench/panelops/report.py --input bench/panelops/results/baseline.json",
        "```",
        "",
        "结果写入被忽略的 `.cache/panelops/`；比较以 op/method/shape/window/missing 精确匹配，超过 10% 的回退列入 comparison JSON。冻结文件不保存每次计时采样和本机绝对路径。",
    ]
    return "\n".join(lines) + "\n", decisions


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=RESULTS / "baseline.json")
    p.add_argument("--merge", nargs="*", type=Path)
    p.add_argument("--validation", type=Path)
    p.add_argument("--freeze", action="store_true")
    a = p.parse_args()
    if a.merge:
        reports = [json.loads(path.read_text()) for path in a.merge]
        report = reports[0]
        merged = {}
        for r in reports:
            for row in r["measurements"]:
                key = (
                    row["op"],
                    row["method"],
                    tuple(row["shape"]),
                    row["window"],
                    row["missing"],
                )
                merged[key] = row
        from additional_baselines import ELEMENTS
        from panelops.api import CS_OPS, TS_OPS

        order = [
            *("ts_" + o for o in TS_OPS),
            *("cs_" + o for o in CS_OPS),
            "ic",
            "rank_ic",
            "group_returns",
            "group_turnover",
            "factor_autocorr",
            "targets",
            *ELEMENTS,
        ]
        report["measurements"] = sorted(
            merged.values(), key=lambda r: (order.index(r["op"]), r["method"])
        )
    else:
        report = json.loads(a.input.read_text())
    if a.validation:
        report["validation"] = json.loads(a.validation.read_text())
    text, decisions = render(report)
    report["selections"] = decisions
    if a.freeze:
        timestamps = [
            r["measured_at"] for r in report["measurements"] if r.get("measured_at")
        ]
        if timestamps:
            report["measurement_period"] = [min(timestamps), max(timestamps)]
        # Shared build identities are interned rather than repeated hundreds of times.
        ids = sorted(
            {
                r["build_id"]
                for r in report["measurements"]
                if r.get("build_id") and r["method"].startswith("rust")
            }
        )
        report["rust_build_ids"] = ids
        for r in report["measurements"]:
            if "build_id" in r:
                build = r.pop("build_id")
                if r["method"].startswith("rust"):
                    r["build"] = ids.index(build)
            for key in ("repeat", "measured_at"):
                r.pop(key, None)
            for key in ("median_ms", "incremental_mib", "output_mib", "peak_rss_mib"):
                if key in r:
                    r[key] = round(r[key], 6)
        RESULTS.mkdir(parents=True, exist_ok=True)
        a.input.write_text(
            json.dumps(report, separators=(",", ":"), ensure_ascii=False) + "\n"
        )
    (ROOT / "docs/PANELOPS_BENCHMARKS.md").write_text(text)
    for filename, chinese in [("README.md", False), ("README.zh-CN.md", True)]:
        path = ROOT / "packages/panelops" / filename
        summary = [
            "## 实测摘要" if chinese else "## Measured snapshot",
            "",
            "4000×5500，窗口 20，缺失 5%；三次中位数，毫秒。完整线程与内存口径见项目基准文档。"
            if chinese
            else "4000×5500, window 20, 5% missing; median of three runs, milliseconds. See the benchmark report for thread/memory methodology.",
            "",
            "| Operator | Fastest single baseline | Rust 1 | Rust default |",
            "|---|---:|---:|---:|",
        ]
        by, _ = selections(report["measurements"])
        for op in (
            "ts_mean",
            "ts_std",
            "ts_quantile",
            "cs_neutralize",
            "rank_ic",
            "targets",
        ):
            d = decisions[op]
            base = minimum(by[op], [d["single_baseline"]])
            r1 = minimum(by[op], ["rust_1"])
            rd = minimum(by[op], ["rust_default"])
            if base and r1 and rd:
                summary.append(
                    f"| {op} | {base['method']}: {base['median_ms']:.2f} | {r1['median_ms']:.2f} | {rd['median_ms']:.2f} |"
                )
        old = path.read_text()
        start = old.index("<!-- BENCHMARK_SUMMARY:START -->")
        end = old.index("<!-- BENCHMARK_SUMMARY:END -->")
        path.write_text(
            old[:start]
            + "<!-- BENCHMARK_SUMMARY:START -->\n"
            + "\n".join(summary)
            + "\n"
            + old[end:]
        )
    print("Rendered", len(decisions), "operators")


if __name__ == "__main__":
    main()
