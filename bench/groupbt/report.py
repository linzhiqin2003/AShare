"""从缓存中的性能记录生成中文报告；不读取或输出行情收益值。"""

import argparse
import json
import pstats
import re
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-o", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if not args.o.resolve().is_relative_to(root):
        parser.error("报告必须写在本 worktree")
    cache = root / ".cache/groupbt"
    folder = cache / "final"
    environment = (
        json.loads((folder / "environment.json").read_text())
        if (folder / "environment.json").exists()
        else {}
    )
    environment_text = "；".join(f"{key} {value}" for key, value in environment.items())
    check_log = cache / "make-check-final.log"
    if not check_log.exists():
        check_log = cache / "make-check.log"
    checks = (
        re.findall(r"=+ (\d+ passed[^\n]*?) =+", check_log.read_text())
        if check_log.exists()
        else []
    )
    validation = checks[-1] if checks else "完整检查尚未结束"

    def read(name):
        return json.loads((folder / f"{name}.json").read_text())

    def median(xs):
        return statistics.median(xs)

    rows = []
    warm = {}
    peaks = {}
    for backend, title in (
        ("main", "冻结 main"),
        ("simulation", "只改模拟"),
        ("current", "全部优化"),
    ):
        runs = [read(f"{backend}-{i}") for i in range(3)]
        cold = [r["elapsed"][0] for r in runs]
        values = [s for r in runs for s in r["elapsed"][1:]]
        warm[backend] = median(values)
        peaks[backend] = max(r["peak_rss_gb"] for r in runs)
        rows.append(
            f"| {title} | {median(cold):.3f} | {median(values):.3f} | {min(values):.3f}–{max(values):.3f} | {peaks[backend]:.3f} |"
        )
    phase = []
    for backend in ("main", "simulation", "current"):
        r = read(f"{backend}-profile")
        total = r["elapsed"][0]
        if backend == "current":
            sums = [sum(row[k] for row in r["native_stages"]) for k in range(5)]
            values = [sums[0] + sums[2], sums[1], sums[3], sums[4]]
        else:
            stats = pstats.Stats(str(folder / f"{backend}-profile.prof")).stats

            def cum(name, stats=stats):
                return sum(v[3] for k, v in stats.items() if k[2] == name)

            def own(name, stats=stats):
                return sum(v[2] for k, v in stats.items() if k[2] == name)

            values = [
                cum("build_weights") + cum("assign_groups") + cum("industry_groups"),
                cum("_adjusted") + cum("_forward_return") + cum("risk_factors"),
                cum("batch")
                if backend == "simulation"
                else cum("simulate") + cum("group_targets") + own("long_short_targets"),
                cum("risk_exposure"),
            ]
        phase.append([*values, total - sum(values), total])
    stages = [
        "中性化、分组与权重构造",
        "价格、持有期收益、风险因子",
        "批量模拟与换手",
        "暴露归约",
        "其余准备与结果组装",
        "剖析总时间",
    ]
    stage_table = "\n".join(
        f"| {name} | {phase[0][i]:.3f} | {phase[1][i]:.3f} | {phase[2][i]:.3f} |"
        for i, name in enumerate(stages)
    )
    threads = []
    for count in (1, 2, 4, 8):
        r = read(f"threads-{count}")
        threads.append(
            f"| {count} | {median(r['elapsed']):.3f} | {min(r['elapsed']):.3f}–{max(r['elapsed']):.3f} | {r['peak_rss_gb']:.3f} |"
        )
    platforms = []
    for count in (1, 2, 4):
        r = read(f"platform-{count}")
        platforms.append(
            f"| {count} | {median(r['elapsed']):.3f} | {r['peak_rss_gb']:.3f} |"
        )
    coverage = []
    for data, label in (("synthetic", "合成面板"), ("real", "真实面板")):
        r = json.loads((cache / data / "compare-current.json").read_text())
        coverage.append(
            f"- {label}：**{r['passed']}/{r['total']} 逐位一致**，对照耗时 {r['elapsed']:.3f} 秒。"
        )
    mutation = json.loads((cache / "mutation.json").read_text())
    traces = [read(b + "-trace") for b in ("main", "current")]
    memory = f"tracemalloc 峰值：{traces[0]['traced_peak_gb']:.3f} → {traces[1]['traced_peak_gb']:.3f} GB。它仅覆盖可追踪分配，未计入全部 Rust 分配和映射页。"
    result = f"""# 分组回测提速与内存报告

本次只修改当前 worktree，未提交、未推送。真实数据只读；原始基准与剖析均留在被忽略的 `.cache/groupbt/`。

## 结果

单组「IS TOP1800 带 TOP800 子池＋VALIDATION 不带子池」后续调用中位
**{warm["main"]:.3f} → {warm["current"]:.3f} 秒**；单组峰值 RSS
**{peaks["main"]:.3f} → {peaks["current"]:.3f} GB**。速度按下表实测判断，内存仍未达到 1.5 GB。
四组场景与线程扩展性见后表；内部 Rayon 与外部 Python 线程都参与计算。

## 测量口径

同一面板、同一个 20 日波动率因子，默认行业中性化和交易限制，10 组，未开启可选权重组合。
不计读面板与表达式求值，计时包含两次完整 groupbt.run。RSS 为进程高水位，包含输入和表达式准备。
三个版本交替运行，每个版本三次独立进程，每个进程调用三轮；第一轮与后两轮分别报告。
四组是 20/60 日波动率 × industry/cap+industry，表达式均在计时前求值。
环境：{environment_text}。
这与已经准备好稀疏目标、行情可复用的 panelbt 成交基准约 2 ms 的工作量不同：
本任务包含全部截面运算、多个每日调仓组合、子池及暴露，均计入总时间。

| 阶段 | 第一轮中位（s） | 后续中位（s） | 后续范围（s） | 最高 RSS（GB） |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

## 逐阶段耗时

单位秒。单独剖析轮，包含计时器开销，不与上面的无剖析中位数混用。
前两版使用 cProfile 累计耗时；最终版使用原生阶段墙钟，跨 worker 的时间不累加。
截面在最终版中融合为一个阶段；价格阶段含风险因子构造，暴露行只含归约。

| 阶段 | 改之前 | 只改模拟之后 | 全部改完之后 |
|---|---:|---:|---:|
{stage_table}

## 具体改动

- P-017：先测 NumPy，去掉重复复制及随后丢弃的归一化权重，保留逐行归约顺序。
- P-018：纯权重记账并入 panelbt，提供 run_weights / run_weights_batch；共享分组标签，
  共用交易方向判断，保留 NumPy 参考实现。批量计算释放 GIL（PyO3 当前名称 py.detach）。
- P-019：窗口前没有持仓的行不再参与分组或模拟；价格前填保留完整历史语义，保留衰减历史分母。
- P-020：panelops-core 提供 NumPy 顺序的截面归约；64 行工作块内批量计算排名、中性化、
  模拟与暴露。组合内部逐日推进，独立组合、独立截面并行。每池一次排名供全部组使用；
  全市场中性化和当天风险向量跨子池共享。价格有限历史缓冲、暴露即时归约避免整张临时面板。

公开 groupbt.run 参数和统计口径保持不变。run_weights 与成交模式的现金、股数、费用、
手数和价格漂移模型有明确区别。开盘权限复用数据字段，收盘复用 panelbt 的 price_permissions；
收盘不增加开盘一字板判断，以保持旧口径。Python 包仍只依赖 NumPy，不互相 import；
截面代码通过 Rust crate 静态复用。

## 逐位验证

{chr(10).join(coverage)}

384 = 三个池 × 四种中性化 × decay 0/3 × open/close × 是否权重组合 × 是否子池 × IS/VALIDATION。
比较所有组收益、多空、换手、暴露、子池、年/季表、曲线、字典、文本、SVG 和 HTML。
断言为 dtype、shape、bytes 相同，包含 NaN 位和有符号零。额外测试覆盖向量中性化、5 组、
9001 列、长缺价、负衰减沿用旧语义、前后空窗口和 GroupingError 行号。

实现时三次暂停修正：标量 dot 的有符号零（最大绝对误差 0）；裁剪历史漏掉衰减分母
（合成最大误差 0.0029948754066186287）；把 prev 与 held 合并导致负零丢失（最大误差 0）。
均修正实现后重新对照，没有放宽断言。浮点求和保留 NumPy 次序，点积调用当前 NumPy
float64 内核；窄范围 C API 桥接的 unsafe 独立封装并写明安全条件。

变异：Rust 换手乘 0.5，冻结 main 对照抓到多空换手差异；恢复源码、重建后 384 组通过。
自动记录 caught={mutation["caught"]}、restored={mutation["restored"]}，耗时 {mutation["elapsed"]:.3f} 秒。

全仓 `make check`：**{validation}**。包含 Ruff、Rust fmt、clippy 和 Rust test。
7 项跳过来自仓库既有的可选 databento 依赖规则，本任务没有新增跳过或放宽断言。
原生源码包也已打包检查，包含新权重内核、共享截面内核及 NumPy 参考实现。

## 内存归因

旧版每组反复展开 (T,N) 目标及移位副本，模拟逐日创建 compact 临时数组；更大的存活峰值
在风险暴露处：四张风险矩阵及其移位副本、持仓、掩码、去均值数组同时存在。它不是同时
保存 22 份全部持仓，而是大面板的重复构造及暴露阶段的大量重叠。
{memory}
最终版常驻工作缓冲按 N、240 日价格历史、64 日工作块和可选 decay 历史增长；不会为每组
复制整张 (T,N) 目标或持仓。进程 RSS 仍包含只读映射的输入页、因子结果、数学库和线程池。
本次全历史输入主要是 float64，字段逻辑大小约 1.715 GB，与任务描述中 0.24 GB 的面板规模
不同。当前 RSS 超标已保留，后续需要梳理输入映射和表达式准备的生命周期；本次不强制回收
调用方共享的输入映射或修改全局内存分配器。

## 线程扩展性

同进程跑 8 组，每组包含 IS 带子池和 VALIDATION。每格三轮，中位与范围均保留。
内部共享 Rayon 池；外部线程数不是机器上计算线程总数。

| 外部线程 | 8 组总耗时中位（s） | 范围（s） | 峰值 RSS（GB） |
|---|---:|---:|---:|
{chr(10).join(threads)}

四组平台场景：

| 外部线程 | 四组总耗时中位（s） | 峰值 RSS（GB） |
|---|---:|---:|
{chr(10).join(platforms)}

## 遗留与复跑

1. 峰值 RSS 尚未达到 1.5 GB，不能宣称所有目标已完成。首次调用的缺页和线程初始化使耗时
   高于后续调用，具体范围见表。
2. 不使用跨 run 的隐式缓存，输入可变且 IS/VALIDATION 必须隔离；子池只在同一次调用内共享。
   不过滤全市场排名来代替各池排名，减均值的浮点舍入可能改变并列关系。
3. 市值 log 仍在 NumPy 批量计算，保留数学库逐位结果；逐日循环已在 Rust 内执行。
4. NumPy 和 BLAS 环境改变时重新生成基准；当前对照缓存检查版本、因子字节和源文件哈希。
5. 全量检查记录见 `.cache/groupbt/make-check-final.log`；复跑命令见 `bench/groupbt/README.md`。

可用 `python -m bench.groupbt.report -o docs/GROUPBT_PERF_REPORT.md` 重新生成本报告。
逐项记录见 `docs/perf_log.d/P-017*` 至 `P-020*`，总表见 `docs/PERF_LOG.md`。
"""
    args.o.parent.mkdir(parents=True, exist_ok=True)
    args.o.write_text(result)
    print(f"报告已写入 {args.o}")


if __name__ == "__main__":
    main()
