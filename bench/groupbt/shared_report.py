"""由本地性能和验收记录生成中文报告，只摘取计数、耗时与内存。"""

import argparse
import json
import re
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path(".cache/groupbt-fusion"))
    parser.add_argument("-o", type=Path, required=True)
    args = parser.parse_args()
    if not args.o.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
        parser.error("报告只能写入本 worktree")

    def load(name):
        return json.loads((args.cache / name).read_text())

    ablation = load("ablation.json")
    copy = load("no-copy-paired.json")
    evaluation = load("evaluation.json")
    market_peak = ablation["market_preparation"]["peak_allocated_bytes"]
    resident = {
        f"{version}-{mode}": load(f"rss-{version}-{mode}.json")["peak_rss_gb"]
        for version in ("reference", "current")
        for mode in ("mapped", "expression")
    }
    mutations = load("mutations.json")
    assert mutations.pop("restored") and all(v["caught"] for v in mutations.values())
    rows = {
        (r["version"], r["scenario"], r["concurrency"], r["ops_threads"]): r
        for r in load("grid/grid.json")
    }
    pipeline = {
        (r["version"], r["scenario"]): r["wall_seconds"]
        for r in load("pipeline/grid.json")
    }
    summaries = len(list((args.cache / "platform/baseline").glob("*/snapshot/*.json")))
    lines = [
        "# 分组回测与平台：共享优化报告",
        "",
        "基于 `a5ea630`，未提交、未推送。真实数据只读；参考包、真实输出和测量原始记录仅存于被忽略的 `.cache/`。",
        "",
        "## 逐项结果",
        "",
        "配对测量先后顺序交替，以下为中位数。内核测试不含表达式、统计或落盘；平台测试经真实 HTTP 入口。",
        "",
        "| 项目 | 之前 | 之后 | 保留的改动 |",
        "|---|---:|---:|---|",
    ]

    def row(title, values, note):
        lines.append(
            f"| {title} | {values['before'] * 1000:.1f} ms | {values['after'] * 1000:.1f} ms | {note} |"
        )

    row("P-021 单组 IS＋验证", copy["median"], "只读底层缓冲区免拷贝")
    row(
        "P-022 四变体行情准备",
        ablation["market_preparation"]["median"],
        "窗口市值与股票池共享",
    )
    row(
        "P-022 四次内核，仅切换均值缓存",
        ablation["exposure_means"]["median"],
        "池、日期、风格均值只计算一次",
    )
    row(
        "P-023 四种记账配置",
        ablation["batch_targets"]["median"],
        "每日目标、价格历史和共同账本共享",
    )
    for windows in ("2", "4"):
        row(
            f"P-024 {windows} 个表达式窗口",
            evaluation[windows]["median"],
            "复用已有公共子式求值器",
        )
    for scenario, title in (("four", "4 组"), ("eight", "8 组")):
        row(
            f"P-025 {title}，仅切换调度",
            {
                label: pipeline[(version, scenario)]
                for label, version in (("before", "reference"), ("after", "current"))
            },
            "同池交错回测与统计",
        )
    lines += [
        "",
        "P-025 独立测量保持其他共享改动不变，冷运行后五次取中位；与后面的主干整体对照分开测量，不将收益相加。",
        f"单组 tracemalloc 分配峰值：{copy['peak_allocated_bytes']['before'] / 2**20:.1f} → {copy['peak_allocated_bytes']['after'] / 2**20:.1f} MiB，主要省掉约 99 MiB 的因子窗口。",
        f"普通只读 NumPy 视图及 memoryview 包装仍复制，避免可写别名污染。四变体准备阶段分配峰值从 {market_peak['before'] / 2**20:.1f} 降至 {market_peak['after'] / 2**20:.1f} MiB，主要省掉重复 logcap 与池掩码。",
        "tracemalloc 不包含全部 Rust 分配与 mmap 驻留，不能将它当作 RSS。",
        f"单组新进程（记录持仓、使用预存只读信号）峰值 RSS 为 {resident['reference-mapped']:.3f} → {resident['current-mapped']:.3f} GB；若同进程还先求表达式，整体高水位为 {resident['reference-expression']:.3f} → {resident['current-expression']:.3f} GB。前者不代表完整研究进程的峰值。",
        "",
        "## 平台与线程预算",
        "",
        "15 核、24 GiB；每个版本、每个配置先冷启动一次，再运行三次取中位。旧版源码和编译包冻结于本工作树缓存，逐格旧版、新版交替。",
        "回测和统计共用一个 Python 线程池，Rust 仍使用既有 Rayon 池；自动预算规则保持不变。",
        "",
        "| 自动设置 | 之前 | 之后 | 协调进程峰值 RSS：之前 → 之后 |",
        "|---|---:|---:|---:|",
    ]
    for scenario, title in (("four", "4 组"), ("eight", "8 组")):
        a, b = (rows[(v, scenario, 0, 8)] for v in ("reference", "current"))
        lines.append(
            f"| {title} | {a['wall_seconds']:.3f} s | {b['wall_seconds']:.3f} s | {a['coordinator_peak_bytes'] / 2**30:.2f} → {b['coordinator_peak_bytes'] / 2**30:.2f} GiB |"
        )
    lines += [
        "",
        "RSS 包含 tab 协调进程的数据映射、信号、IC 中间值、原生内核及分配器高水位；当前环境无法读取完整进程树 RSS，因此不冒充进程树总量。",
        "串行配置下保留多份信号 mmap 会提高 RSS；完整并发矩阵和选择依据见 `PLATFORM.md`，没有把输入扫描时间当作内存带宽瓶颈的证据。",
        "",
        "## 正确性与变异检查",
        "",
        "- 合成逐位矩阵 441/441 全过；另增共享池、窗口、风险向量、调仓、持仓、线程池和表达式 exact 参数测试。",
        "- 真实数据" + (args.cache / "real-bitwise.md").read_text().strip(),
    ]
    lines.extend(
        "- 平台 " + line.split("；各次耗时")[0] + "。"
        for line in (args.cache / "platform-final.md").read_text().splitlines()
    )
    lines += [
        f"- 平台逐数组检查 dtype、shape、bytes，保留 NaN 位和负零；{summaries} 份原始摘要 JSON 完全相同。日期、权重、因子、行业、证券轴等持仓数组全部纳入，等待后台写入完成后才比对。",
        f"- {len(mutations)}/{len(mutations)} 变异被捕获：只读标志与 memoryview 别名、池键、窗口键、表达式窗口、暴露池索引、调仓周期、删除零项。源码和扩展均已恢复。",
    ]
    text = (args.cache / "make-check-final.log").read_text()
    summary = re.findall(r"=+ (\d+ passed, .*? in .*?) =+", text)
    assert summary, "最终 make check 尚未完成"
    lines += [
        "- `make check` 全绿（含 Rust fmt / clippy / test）：" + summary[-1] + "。",
        "",
        "## 保守取舍与遗留",
        "",
        "1. 不采用删零后求和。合成八元素反例的最大绝对差为 1；压缩改变 NumPy 的成对求和树。保留全长加权和，只共享均值，没有放宽逐位断言。",
        "2. 未新增一次扫描多个标准差窗口的内核。现有公共子式共享已有实测收益，标准差各窗口仍使用原更新及重置规则，exact 参数不变。",
        "3. IS 与验证各自截断并独立处理活动区间，不建立跨止日的全尺寸排名或风险缓存。均值缓存按窗口和完整池集合绑定，异构池集合不再拼接复用。",
        "4. 常规平台任务仍包含各信号的截面排序、权重推进、风险历史、IC 和落盘。四组尚未达到此前 1 秒目标；同进程先求表达式的单组整体 RSS 也未达到 1.5 GB。只读 mmap 和共享市值解决的是重复分配，不能消除这些工作。",
        "",
        "## 复跑",
        "",
        "`bench/groupbt/README.md` 记录超时包装和命令；本报告由 `python -m bench.groupbt.shared_report -o docs/GROUPBT_SHARED_REPORT.md` 生成。",
        "逐项记录为 [P-021 至 P-025](PERF_LOG.md)，原始文件在 `.cache/groupbt-fusion/`；真实行情数字不进入公开文档。",
        "",
    ]
    args.o.write_text("\n".join(lines))


if __name__ == "__main__":
    main()
