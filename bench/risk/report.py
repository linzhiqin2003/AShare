"""由唯一冻结 JSON 生成报告；不外推未测耗时，不把求解失败算成功。"""

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def render(baseline, output):
    data = json.loads(Path(baseline).read_text())
    m = data["measurements"]
    meta = m["rebuild"]["metadata"]
    lines = [
        "# 风险模型与组合优化基准（第二轮）",
        "",
        "所有时间为墙钟秒数。其他 worktree、系统任务及本轮部分构建/测量同时运行，数字受干扰；请空闲时重跑。",
        f"机器：{meta['machine']}，{meta['cpus']} 逻辑核；Python {meta['python'].split()[0]}，NumPy {meta['numpy']}，{meta['rust']}；日期 {meta['date']}。",
        "BLAS 为 Apple Accelerate，实际线程数无法由 threadpoolctl 核实；请求 1 线程不代表生效。Rust 明确测 1/4/15 线程，单个 Clarabel 固定 1 线程；独立批次 4 线程。",
        "",
        "## 风险内核与正式流水线",
        "",
        "T=4000、N=5500、K=45。下面是三次中位数（因子协方差为 20 次）。",
        "",
        "| 环节 | NumPy | Rust 1 | Rust 4 | Rust 默认硬件 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, key, label in (
        ("kernels", "regression", "WLS（NumPy SVD）"),
        ("factor-kernels", "ew_covariance", "EW 因子协方差"),
        ("factor-kernels", "newey_west", "NW 因子协方差"),
        ("kernels", "specific_risk", "特质 EW＋NW"),
        ("kernels", "bias", "滚动偏差统计"),
    ):
        row = m[name]["seconds"][key]
        lines.append(
            f"| {label} | "
            + " | ".join(
                f"{row[x]['median']:.6f}"
                for x in ("numpy", "rust_1", "rust_4", "rust_0")
            )
            + " |"
        )
    lines += [
        "",
        "BLAS Gram/KKT 另作认真基线，不只与 SVD 比较：",
        "",
        "| 路径 | 4000 日秒数 |",
        "|---|---:|",
    ]
    for name, row in m["gram"]["seconds"].items():
        lines.append(f"| {name} | {row['median']:.6f} |")
    lines += [
        "",
        "逐日变化 B（64 日小块生成，包含生成开销）：",
        "",
        "| 路径 | 4000 日秒数 |",
        "|---|---:|",
    ]
    for name, row in m["varying-regression"]["seconds"].items():
        lines.append(f"| {name} | {row['median']:.6f} |")
    rebuild = m["rebuild"]
    lines += [
        "",
        "正式 model_series 全量和追加都调用同一 Rust 回归内核，病态时同一确定性 SVD 回退。线程池按数量复用。",
        "",
        "| 执行形态 | 总秒数 | 暴露 | 回归 | 协方差/特质风险 | 线程 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in rebuild["phases"].items():
        c = row["components"]
        lines.append(
            f"| {name} | {row['seconds']:.4f} | {c['exposures_at']:.4f} | {c['cross_section_batch']:.4f} | {c['estimate_covariances']:.4f} | {row['threads']} |"
        )
    lines += [
        "",
        f"4000 个 B/F/D 的 SHA-256 逐日相同：前 3999 日与追加一天使用不同线程数和分块。末日摘要 `{rebuild['bitwise']['last_sha256']}`。追加耗时含读取历史状态、输入前缀哈希；单日回归内核中位数 {rebuild['single_day_regression_seconds']:.6f} 秒。",
        "",
        "短因子面板仍按实测选择 NumPy（T≤1008，K≤64）；大截面特质风险使用 Rust。边界为工程选择，空闲机器可再测。完整统计 LW＋45 PCA 为 "
        + f"{m['pipeline']['statistical_pca_lw_seconds']:.4f} 秒；这部分保留 BLAS/ARPACK。",
        "",
        "## 描述子分布与增量化",
        "",
        f"末日原窗口算法 {np.median(rebuild['descriptor_seconds']['window_reference']):.6f} 秒；递推算法 {np.median(rebuild['descriptor_seconds']['rolling']):.6f} 秒。beta/HSIGMA、DASTD、momentum、21/63/252 换手均使用固定窗口矩递推；CMRA 使用累计收益端点。",
        "",
        "下表为一次行级剖析，包含追踪器开销，只用于辨识热点，不能替代上面的未插桩计时。递推状态更新集中在单独一行，其他描述子行不重复计入。",
        "",
        "| 阶段 | 原窗口秒数 | 递推秒数 |",
        "|---|---:|---:|",
    ]
    ranges = (
        ("Size/行业/ESTU", 160, 212),
        ("Beta/HSIGMA", 213, 244),
        ("全部递推矩更新", 245, 249),
        ("DASTD", 250, 255),
        ("CMRA", 256, 260),
        ("Momentum", 261, 271),
        ("风格标准化/正交化", 272, 279),
        ("Liquidity（含标准化）", 280, 303),
        ("价值/盈利/拼接", 304, 345),
    )
    for label, first, last in ranges:
        values = [
            sum(
                v
                for k, v in rebuild["descriptor_profile_lines"][mode].items()
                if first <= int(k) <= last
            )
            for mode in ("window_reference", "rolling")
        ]
        lines.append(f"| {label} | {values[0]:.6f} | {values[1]:.6f} |")
    lines += [
        "",
        "CMRA 保存 253 日累计 log 收益的环形状态，以十二个 21 日端点的极差替代反复扫描 252 日；有效计数保留缺失窗口语义。所有长窗行情描述子已增量化。截面中位数/MAD、填补和正交化仍需读取当天截面，可在 panelops 合并后评估融合内核，本轮没有导入该包。",
        "",
        "## 缓存",
        "",
    ]
    cache = rebuild["cache"]
    lines += [
        f"实存 {cache['snapshots']} 日：{cache['bytes']:,} 字节（{cache['bytes'] / 1e9:.3f} GB），平均 {cache['bytes_per_day']:,.0f} 字节/日；最新检查点 {cache['latest_exposure_bytes']:,} 字节。低于 2 GB 目标。历史快照只有无损 F/D 和回归/状态历史；B 只保存最新一天，其余按需重建。",
        f"最新模型读取中位数 {np.median(cache['latest_read_seconds']):.6f} 秒；仅状态 {np.median(cache['state_read_seconds']):.6f} 秒。历史 B 读取需重放描述子前缀，耗时见 historical_read。版本/参数变动清理旧代，历史修订清理旧尾部。",
        "",
        "| B 存储试验 | 原始字节 | 压缩字节 | 读取秒数 | 暴露最大误差 | 等权组合方差相对误差 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in rebuild["exposure_storage"].items():
        lines.append(
            f"| {name} | {row['raw_bytes']} | {row['compressed_bytes']} | {row['read_seconds']:.6f} | {row['max_absolute_exposure_error']:.3g} | {row['portfolio_variance_relative_error']:.3g} |"
        )
    lines += [
        "",
        "float32 仅作试验，正式缓存不降精度；单只组合误差小不代表所有权重均如此。采用按需重建已经满足体积目标。占用统计/清理：`python -m ashare.risk.cache stats|clear .cache/risk-models`。",
        "",
        "## 五种优化调用路径",
        "",
        f"版本：{m['optimizers']['versions']}。K=45；预算、long-only、个股上限、主动个股/行业/风格偏离、单边换手、方向冻结、非对称成本均启用；SOCP 另含 TE。OSQP 不支持 SOC，不另写非线性求解器。",
        "单次为建立会话后的 3 次中位数；200 期 alpha 改变，前一期结果作为下一期持仓。B/F/D 固定以测试复用。包含模型转换、独立自检及重试。被拒绝的期数单列，不能据此宣称更快有效求解。旧值为第一轮冻结基线（共享负载不同）；配对重测另见下表。",
        "",
        "| N | 问题 | 调用 | 旧单次 | 新单次 | 旧200期 | 新200期 | 新拒绝期数 |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    oldrows = data.get("previous_optimizer", {}).get("rows", {})
    for n in (1000, 3000, 5000):
        for scenario in ("qp", "socp"):
            for method in ("cvxpy_clarabel", "cvxpy_osqp", "clarabel", "osqp", "rust"):
                key = f"{n}/{scenario}/{method}"
                row = m["optimizers"]["rows"][key]
                old = oldrows.get(key, {})
                if row.get("status") == "unsupported":
                    numbers = "— | 不支持 | — | 不支持 | —"
                else:
                    a = (
                        f"{old['single_seconds']:.5f}"
                        if "single_seconds" in old
                        else "—"
                    )
                    b = (
                        f"{old['sequential_seconds']:.3f}"
                        if "sequential_seconds" in old
                        else "—"
                    )
                    numbers = f"{a} | {row['single_seconds']:.5f} | {b} | {row['sequential_seconds']:.3f} | {len(row['failures'])}"
                lines.append(f"| {n} | {scenario} | {method} | {numbers} |")
    lines += [
        "",
        "默认按调用形态：单次和依赖前期持仓的序列用 Python Clarabel（持有会话、更新 q/b）；多个独立问题用 Rust solve_many。两者是同一求解器，不再按 N 切换。OSQP 保留 warm-start 和矩阵/分解复用；Clarabel 的数据更新复用对象与稀疏结构，但它本身不提供迭代点热启动。数值失败换等价 CVXPY 表达并重试，只有明确 primal infeasible 才放松经济约束。",
        "",
        "| N | 问题 | Rust 4线程 200独立问题 | auto 200依赖期 | auto拒绝 |",
        "|---:|---|---:|---:|---:|",
    ]
    for n in (1000, 3000, 5000):
        for scenario in ("qp", "socp"):
            parallel = m["optimizers"]["rows"][
                f"{n}/{scenario}/rust_parallel_independent"
            ]["independent_seconds"]
            auto = (
                m.get("optimizers-auto", {})
                .get("rows", {})
                .get(f"{n}/{scenario}/auto", {})
            )
            lines.append(
                f"| {n} | {scenario} | {parallel:.3f} | {auto.get('sequential_seconds', float('nan')):.3f} | {len(auto.get('failures', []))} |"
            )
    structure = m["structure"]
    lines += [
        "",
        "约束配对重测（冷调用含建模，3 次中位数；目标 Hessian 为 N+K 块，TE 为 K+N 因子锥，均无 N×N 协方差）：",
        "",
        "| N/问题 | N维行 | y维行 | A非零元 前→后 | 权重最大差 |",
        "|---|---:|---:|---|---:|",
    ]
    for key, row in structure["factor_constraints"].items():
        old, new = row["stock_rows"], row["factor_y"]
        lines.append(
            f"| {key} | {old['median']:.5f} | {new['median']:.5f} | {old['A_nnz']}→{new['A_nnz']} | {row['max_weight_difference']:.3g} |"
        )
    lines += [
        "",
        "固定 B/F/D 的 15 期（边界行模式也不变时才复用）：",
        "",
        "| N | 每期重建 | 向量更新 | 实际复用期数 |",
        "|---:|---:|---:|---:|",
    ]
    for n in (1000, 3000, 5000):
        old, new = (structure["vector_reuse"][f"{n}/{v}"] for v in (False, True))
        lines.append(
            f"| {n} | {old['seconds']:.4f} | {new['seconds']:.4f} | {new['reused']} |"
        )
    lines += [
        "",
        "实用 OSQP 容差对照（15 期；这是求解器调用的诊断，不降低正式自检 1e-7）：",
        "",
        "| N/调用/容差 | 秒数 | 最大违反 | 超过正式自检的期数 |",
        "|---|---:|---:|---:|",
    ]
    for key, row in structure["practical_osqp"].items():
        lines.append(
            f"| {key} | {row['seconds']:.4f} | {row['max_violation']:.3g} | {row['strict_selfcheck_failures']} |"
        )
    lines += [
        "",
        "强制 auto 原生数值失败，测 CVXPY 等价回退的子进程峰值 RSS：",
        "",
        "| N | 峰值GB | 秒数 |",
        "|---:|---:|---:|",
    ]
    for row in structure["fallback_memory"]:
        lines.append(
            f"| {row['N']} | {row['rss_bytes'] / 1e9:.3f} | {row['seconds']:.4f} |"
        )
    lines += [
        "",
        "修复前 N=1000/SOCP 为 8.11 GB。测试固定 N=1000 <1 GB；显式检查 ignore_dpp=True，避免将来变异测试本身耗尽内存。",
        "",
        "## 统计估计对照",
        "",
        "常相关收缩含本项目有限 N 修正，不能称作原封不动的 LW 2004。下表为每格 60 次蒙特卡洛，损失是 Frobenius 平方损失/oracle；oracle 使用真值，只用于诊断。",
        "",
        "| T/N/因子 | δ本实现 | δLW2004 | δoracle | 损失本实现 | 损失LW2004 | 不收缩 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in m["review-comparisons"]["constant_correlation"]:
        v = row["mean"]
        lines.append(
            f"| {row['T']}/{row['N']}/{row['factors']} | "
            + " | ".join(
                f"{v[key]:.4f}"
                for key in (
                    "delta_ours",
                    "delta_paper",
                    "delta_oracle",
                    "loss_ours",
                    "loss_paper",
                    "loss_unshrunk",
                )
            )
            + " |"
        )
    validation = m["validation"]
    lines += [
        "",
        "合成数据 T=3000/N=500/K=13，前2000日估计、后1000日验证；等权、NW=0、regime关闭以隔离抽样误差。容差沿用第一轮：F 16%、D 8%、组合12%、收益 RMSE .002。",
        "",
        "| 模型 | F误差 | D误差 | 组合方差误差 | 公共协方差误差 | 因子收益RMSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in validation["models"].items():
        r = row["recovery"]
        factor_error = (
            f"{r['factor_covariance_relative_frobenius']:.3%}"
            if "factor_covariance_relative_frobenius" in r
            else "旋转不唯一"
        )
        return_error = (
            f"{r['factor_returns_rmse']:.6g}" if "factor_returns_rmse" in r else "—"
        )
        lines.append(
            f"| {name} | {factor_error} | {r['specific_relative_mae']:.3%} | {r['portfolio_variance_relative_mae']:.3%} | {r['common_covariance_relative_frobenius']:.3%} | {return_error} |"
        )
    lines += [
        "",
        "偏差统计：均值 / 重叠95%覆盖 / 不重叠95%覆盖；真模型也用同一实现收益，帮助区分有限样本与模型误差。",
        "",
        "| 模型 | 因子组合 | 随机组合 | 市值组合 | 单股 |",
        "|---|---:|---:|---:|---:|",
    ]
    biases = {name: row["bias"] for name, row in validation["models"].items()}
    biases["true_model"] = validation["truth_bias"]
    for name, report in biases.items():
        numbers = [
            f"{report[g]['mean']:.4f}/{report[g]['within_95pct']:.2%}/{report[g]['non_overlapping']['within_95pct']:.2%}"
            for g in ("factor", "random", "cap_weighted", "single_stock")
        ]
        lines.append(f"| {name} | " + " | ".join(numbers) + " |")
    reference = m["review-comparisons"]["bias"]
    lines += [
        "",
        f"IID 真波动、不重叠 {reference['iid_nonoverlap']['windows']} 窗口：覆盖 {reference['iid_nonoverlap']['coverage']:.4f}，均值 {reference['iid_nonoverlap']['mean']:.4f}。六种子的完整真值/估计对照保存在冻结 JSON；重叠窗口比例不是独立重复试验的置信度。",
        "",
        "| 组合 | 真模型重叠覆盖 | 真模型不重叠 | 估计重叠 | 估计不重叠 |",
        "|---|---:|---:|---:|---:|",
    ]
    for group in ("factor", "random", "cap_weighted", "single_stock"):
        nums = [
            np.mean([v[group + "/" + tag][metric] for v in reference["seeds"].values()])
            for tag in ("truth", "estimated")
            for metric in ("overlap_coverage", "nonoverlap_coverage")
        ]
        lines.append(f"| {group} | " + " | ".join(f"{v:.3%}" for v in nums) + " |")
    lines += [
        "",
        "## 重跑与入库内容",
        "",
        "```sh",
        "make dev && make check",
        "make risk-bench   # 完整规模，含实际200期；可能耗时较长",
        "make risk-mutate",
        "make risk-validate",
        "uv run --no-sync python bench/risk/run.py --baseline bench/risk/results/baseline.json --output .cache/idle-risk-bench",
        "uv run --no-sync python -m ashare.risk.validate --input /path/panel.npz --start 600 --cache .cache/risk-models --output .cache/risk-validation",
        "```",
        "",
        "results/ 只入库 baseline.json；含当前测量、旧优化器最小对照、变异及生成此文档必需数据。日常输出、运行日志、模型缓存均在 .cache/，不自动覆盖冻结文件。重跑生成 comparison.json 与独立 Markdown；显式更新基线用 consolidate.py。",
    ]
    Path(output).write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline", type=Path, default=ROOT / "bench/risk/results/baseline.json"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "docs/RISK_BENCHMARKS.md")
    args = parser.parse_args()
    render(args.baseline, args.output)
