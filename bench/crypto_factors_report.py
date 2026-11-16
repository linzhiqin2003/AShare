"""严格面板上的 C6 报告；4000 次公共随机数模拟，仅输出汇总。"""

import argparse
import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from ashare.data import crypto
from ashare.data.crypto.signals import SignalDataset
from ashare.factor.crypto import FACTOR_NAMES, crypto_factors, daily_factor
from ashare.factor.crypto_eval import (
    CONFIRM_FACTORS,
    CONFIRM_FAMILY,
    SimulationBank,
    _batch_components,
    _component_t,
    calibrate,
    conservative_test,
    crypto_labels,
    descriptive,
    hodrick_1b,
    holm_confirm,
    power_curve,
    quantile_description,
    sample_mask,
)

SYMBOLS = ("BTCUSDT", "ETHUSDT")


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def self_check():
    bank = SimulationBank(
        np.random.default_rng(618).normal(size=3000), 1000, phi_grid=[0]
    )
    mask = np.ones(1000, bool)
    t = _component_t(
        _batch_components(*bank.basis(0), bank.returns[:, 1:], mask, mask, 7), 0
    )
    rate = float(np.mean(np.abs(t) > 1.959963984540054))
    if not 0.035 <= rate <= 0.065:
        raise ValueError(f"phi=delta=0 自检拒绝率异常: {rate}")
    return dict(
        phi=0,
        delta=0,
        repetitions=4000,
        normal_5pct_rejection_rate=rate,
        range=[0.035, 0.065],
        pass_check=True,
    )


def evaluate_group(task, *, horizons=(7, 1, 30), checkpoint=None):
    from threadpoolctl import threadpool_limits

    symbol, j, kind, split, gfull, daily, entries, desc, repetitions, power = task
    started = time.monotonic()
    results = []
    with threadpool_limits(limits=1):
        # 单日合法轴裁剪后，h 天标签由相同逐日序列重建；跨界自然不能组成完整窗口。
        use = sample_mask(entries, 1, split)
        ge = gfull[use]
        calibration = {}
        for name in FACTOR_NAMES:
            try:
                calibration[name] = calibrate(daily[name], gfull, entries)
            except ValueError as exc:
                calibration[name] = str(exc)
        possible = next((v for v in calibration.values() if isinstance(v, dict)), None)
        bank = (
            SimulationBank(
                possible["eps"],
                len(ge),
                repetitions=repetitions,
                seed=6182026 + j * 10 + (kind == "spot"),
            )
            if possible
            else None
        )
        for h in horizons:
            for name in FACTOR_NAMES:
                x = daily[name]
                xe = x[use]
                confirm = (
                    split == "out"
                    and kind == "usdm"
                    and h == 7
                    and name in CONFIRM_FACTORS
                )
                observed = hodrick_1b(xe, ge, h)
                desc_y = desc[h].copy()
                desc_y[~sample_mask(entries, h, split)] = np.nan
                row = dict(
                    symbol=symbol,
                    factor=name,
                    kind=kind,
                    horizon=h,
                    split=split,
                    family="confirmation" if confirm else "探索，不作推断",
                    **observed,
                    descriptive=descriptive(x, desc_y),
                    quantiles=quantile_description(x, desc_y, entries, h),
                )
                cal = calibration[name]
                if (
                    not isinstance(cal, dict)
                    or observed["n"] < 100
                    or not np.isfinite(observed["t"])
                ):
                    row.update(
                        p=np.nan,
                        reason=cal
                        if isinstance(cal, str)
                        else "有效样本不足 100 或统计量退化",
                    )
                else:
                    xm, gm = np.isfinite(xe), np.isfinite(ge)
                    table, details = conservative_test(
                        bank,
                        xm,
                        gm,
                        h,
                        cal["delta_grid"],
                        observed["t"],
                        full_refinement=confirm,
                    )
                    row.update(details)
                    row["calibration"] = {
                        k: v for k, v in cal.items() if k not in ("eps", "delta_grid")
                    }
                    if confirm and power:
                        row["power"] = power_curve(bank, table, xm, gm, h, cal)
                results.append(row)
            print(
                f"{symbol} {kind} {split} h={h}: {len(results)} tests, {time.monotonic() - started:.1f}s",
                flush=True,
            )
            if checkpoint is not None:
                checkpoint(results)
    return results


def run(root, *, repetitions=4000, power=True, progress=None, workers=2):
    started = time.monotonic()
    ds = crypto.open(root)
    if ds.meta.get("allow_inferred_funding"):
        raise ValueError("报告必须使用严格面板")
    signals = (
        SignalDataset(root) if (Path(root) / "signals/latest.json").exists() else None
    )
    factors = crypto_factors(ds, signals, SYMBOLS, ds.meta["start"], ds.meta["end"] + 1)
    labels = crypto_labels(ds, SYMBOLS, factors["funding_last"])
    days, x0 = daily_factor(factors["funding_last"])
    daily = {name: daily_factor(factors[name])[1] for name in FACTOR_NAMES}
    entries = labels["entry"]
    results = []
    header = dict(
        data_version=ds.version,
        signal_version=signals.version if signals else None,
        signal_status="verified_input_available"
        if signals
        else "unavailable_signal_dataset",
        complete=signals is not None,
        definition=factors.identity[2],
        start=ds.meta["start"],
        end=ds.meta["end"],
        trade_hour=0,
        split="2024-01-01T00:00:00Z",
        simulation_repetitions=repetitions,
        simulation_workers=workers,
        evaluation_definition="crypto-eval-v2-calendar-labels-standardized-qr-1b",
        seed_base=6182026,
        phi_grid_count=37,
        delta_grid_count=11,
        gamma=0.001,
        confirmation_refinement="union of all once-refined interior neighborhoods",
        power_effect_unit="7*b*in_sample_sd_x; also report single-initial-shift effect range",
        self_check=self_check(),
        factor_coverage={
            name: [int(np.isfinite(daily[name][:, j]).sum()) for j in range(2)]
            for name in FACTOR_NAMES
        },
        assumptions=[
            "季度冻结为连续相邻同收盘第二小时起",
            "严格面板质量合并位保守排除整行",
            "推断样本至少 100 日；校准至少 30 对",
            "平稳后向创新尾方差截断 ≤ 1e-20",
        ],
        limitations=[
            "网格最大值近似连续上确界",
            "独立重抽丢掉波动聚集",
            "只有 BTC/ETH 两个币",
            "样本外仅从 2024-01-01 至严格面板终点",
            "Fisher 区间在非高斯样本下是工作模型近似",
            "资金费结算与同刻成交零延迟近似",
        ],
    )
    tasks = []
    for j, symbol in enumerate(SYMBOLS):
        for kind in ("usdm", "spot"):
            for split in ("out", "in"):
                tasks.append(
                    (
                        symbol,
                        j,
                        kind,
                        split,
                        labels["daily"][kind][:, j],
                        {name: value[:, j] for name, value in daily.items()},
                        entries,
                        {h: labels["descriptive"][h][kind][:, j] for h in (1, 7, 30)},
                        repetitions,
                        power,
                    )
                )
    if not 1 <= workers <= 2:
        raise ValueError("workers 必须在 1..2")
    if workers == 1:
        for task in tasks:
            results.extend(evaluate_group(task))
    else:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            futures = [pool.submit(evaluate_group, task) for task in tasks]
            for future in as_completed(futures):
                results.extend(future.result())
                if progress:
                    Path(progress).write_text(
                        json.dumps(
                            clean(dict(**header, tests=results)), ensure_ascii=False
                        )
                    )
    results.sort(
        key=lambda r: (
            r["symbol"],
            r["kind"],
            r["split"],
            r["horizon"],
            FACTOR_NAMES.index(r["factor"]),
        )
    )
    confirmed = {
        (r["symbol"], r["factor"]): r["p"]
        for r in results
        if r["family"] == "confirmation"
    }
    adjusted = holm_confirm(confirmed)
    for row in results:
        if row["family"] == "confirmation":
            row["holm_p"] = adjusted[(row["symbol"], row["factor"])]
            row["significant"] = bool(
                np.isfinite(row["holm_p"]) and row["holm_p"] <= 0.05
            )
    header.update(
        tests=results,
        confirmation_family_size=len(CONFIRM_FAMILY),
        runtime_seconds=time.monotonic() - started,
    )
    for field, filename in [
        ("strategy_example", "crypto_factor_example.json"),
        ("raw_recalculation", "crypto_factors_hand_check.json"),
    ]:
        path = Path(__file__).parent / "results" / filename
        if path.exists():
            value = json.loads(path.read_text())
            if value["data_version"] == ds.version and (
                field != "raw_recalculation"
                or value["signal_version"] == header["signal_version"]
            ):
                header[field] = value
    audit = Path(__file__).parent / "results/crypto_signals_audit.json"
    if audit.exists():
        header["signal_archive_audit"] = json.loads(audit.read_text())
    if signals:
        verification = Path(root) / "signals/verification" / f"{signals.version}.json"
        if verification.exists():
            header["signal_verification"] = json.loads(verification.read_text())
    return clean(header)


def markdown(report):
    lines = [
        "# 资金费率与基差时序因子报告",
        "",
        "设计口径：DESIGN 4.18 v5.1。描述收益和可加推断收益分开；只有固定八项样本外确认族使用 Holm。其余均为探索，不作推断。",
        "",
        f"严格面板 `{report['data_version']}`；季度信号 `{report['signal_version']}`；因子 `{report['definition']}`。",
        "",
        f"UTC 覆盖 {np.datetime64(report['start'], 'ms')} 至 {np.datetime64(report['end'], 'ms')}，日频 00:00；2024-01-01 切分，跨界标签排除。因子预热可使用此前历史。",
        "",
        f"每格 {report['simulation_repetitions']} 次；φ 全域 37 点，样本内 δ 99.9% Fisher 区间 11 点，公共随机数、内点加密、最终加 γ=.001。",
        "",
        (
            "季度信号数据集不可用；季度相关结果缺失，C6 的真实季度报告尚未完成。"
            if report["signal_version"] is None
            else "季度信号数据集版本与质量例外已冻结。"
        ),
        "",
        "## 固定八项确认族",
        "",
        "|币|因子|n|β|t|保守 p|Holm p|显著|MDE 正/负：持续七日增加 1 sd 的等价效应（对数收益）|相邻搜索点及最低功效（正/负）|",
        "|---|---|---:|---:|---:|---:|---:|---|---|---|",
    ]

    def fmt(v):
        return "不可计算" if v is None else f"{v:.6g}"

    for row in report["tests"]:
        if row["family"] != "confirmation":
            continue
        mde = []
        brackets = []
        for direction in ("positive", "negative"):
            value = row.get("power", {}).get(direction, {}).get("minimum_detectable")
            mde.append(
                "不可计算"
                if "power" not in row
                else "搜索范围不可检出"
                if value is None
                else fmt(value)
            )
            bracket = row.get("power", {}).get(direction, {}).get("search_bracket", {})

            def show_point(point):
                return (
                    "无"
                    if point is None
                    else f"{point['effect']:g}（{point['worst_power']:.3%}）"
                )

            brackets.append(
                f"{show_point(bracket.get('previous'))} → {show_point(bracket.get('reached'))}"
            )
        conclusion = (
            "不可计算"
            if row.get("holm_p") is None
            else "是"
            if row.get("significant")
            else "否"
        )
        lines.append(
            f"|{row['symbol']}|{row['factor']}|{row['n']}|{fmt(row['beta'])}|{fmt(row['t'])}|{fmt(row['p'])}|{fmt(row.get('holm_p'))}|{conclusion}|{' / '.join(mde)}|{' / '.join(brackets)}|"
        )
    nonmono = [
        r
        for r in report["tests"]
        if r.get("monotone_phi") is False or r.get("monotone_abs_delta") is False
    ]
    lines += [
        "",
        f"p 曲面存在非单调的检验：{len(nonmono)} 项，逐项 φ 与 |δ| 单调性、最大点、加密情况见 JSON。不能用边界值代替全域搜索。",
        "",
        "功效使用逐日备择生成再聚合；每一项预计算零假设 |t*| 分布后查表，以保守 p≤0.00625 的完整规则取所有参数格最低拒绝率。效应网格正负分开，目标 80%；MDE 为搜索网格上的最小值。",
        "相邻搜索点同时列出最低功效：MDE 是首个达到80%的离散点，不能视为精确连续阈值。搜索上限为10，功效算法未改变。",
        "",
        "## 全部因子与探索检验",
        "",
        "JSON 包含 2 币 × 12 因子 × 3 期限 × 2 收益 × 2 切分 = 288 项的样本数、Spearman、β/t、保守 p、滚动分位统计及重叠数；无逐行行情。分位阈值仅取此前365天，至少180个有效值，并列归下组。",
        "",
        "## 自检与性能",
        "",
        f"φ=δ=0 自检：{report['self_check']}。本次报告耗时 {report['runtime_seconds']:.2f} 秒，{report['simulation_workers']} 个模拟进程。"
        + (
            "尚缺季度输入，不能作为完整报告的性能验收。"
            if not report["signal_version"]
            else ""
        ),
        "",
        "## 策略例子（回溯探索）",
        "",
        "冻结阈值 ±2、对应方向减半，缺因子不调整；与 C3 趋势例同区间、同120日预热。详见 `bench/results/crypto_factor_example.json`。确认需要 2026-11-15 之后新数据。",
        "",
        "## 局限",
        "",
    ]
    lines += ["- " + v for v in report["limitations"]]
    lines += [
        "",
        "## 全因子样本外七日永续概览",
        "",
        "确认族以外行均为探索，不作推断；Spearman 使用描述收益，t/p 使用可加推断收益。",
        "",
        "|币|因子|n|Spearman|t|保守p|类别|",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in report["tests"]:
        if row["split"] == "out" and row["kind"] == "usdm" and row["horizon"] == 7:
            lines.append(
                f"|{row['symbol']}|{row['factor']}|{row['n']}|{fmt(row['descriptive']['spearman'])}|{fmt(row['t'])}|{fmt(row['p'])}|{row['family']}|"
            )
    example = report.get("strategy_example")
    if example:
        lines += [
            "",
            "## 回溯策略对照数字",
            "",
            "同区间、同120日预热、阈值±2和减半规则均未调参。",
            "",
            "|账户|版本|终值净值|年化|最大回撤|资金费支出 USDT|费用 USDT|",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for kind, pair in example["strategies"].items():
            for label in ("base", "overlay"):
                v = pair[label]
                lines.append(
                    f"|{kind}|{label}|{v['final_nav']:.6f}|{v['annualized']:.2%}|{v['max_drawdown']:.2%}|{v['funding_paid']:.4f}|{v['fees']:.4f}|"
                )
    hand = report.get("raw_recalculation")
    if hand:
        lines += [
            "",
            "## 原始归档独立复算",
            "",
            "2024-06-01 至 2024-06-08，按 ZIP/CSV 独立解码，不使用生产解析器或面板费率现金。",
        ]
        for v in hand["results"]:
            lines.append(
                f"- {v['symbol']}：{v['hours']} 小时，溢价最大误差 {v['premium_max_error']:.3g}，资金费均值最大误差 {v['funding_mean_max_error']:.3g}；季度基差：{v.get('basis_max_error', v.get('basis_status'))}。"
            )
    lines += [
        "",
        "MDE 主值为持续7日增加一个样本内标准差的 h·b·sd 等价效应；单次初始冲击的效应依赖 φ，JSON 同时给出 b·sd·Σφ^j 的范围。确认项采用所有可能的一次内点加密格的并集，观察和功效共用同一更保守固定规则。",
    ]
    audit = report.get("signal_archive_audit")
    if audit:
        lines += [
            "",
            "## 季度归档与交割核对",
            "",
            f"{audit['contracts']}期、{audit['archives']}个月归档，原始CHECKSUM与CRC核对；{audit['contracts_with_post_delivery_rows']}期共{audit['post_delivery_rows']}行自动登记为 post_delivery 并剔除，原始归档完整保留，逐期计数进入内容版本。",
            "",
            "已到期合约最后成交K线开盘严格落在[交割−24小时，交割]；整点成交只参与交割核对，收盘晚于交割的K线不进入信号。未到期合约保留 pending_not_expired 状态，不能把尚未发生的最后成交当成已核验。",
            "",
            f"独立校验：{report.get('signal_verification', {}).get('verdict', '见信号版本校验记录')}；交割状态计数：{audit.get('delivery_checks', {})}；生命周期内缺小时 {audit['interior_gaps']}。",
        ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="tmp/crypto")
    parser.add_argument("--repetitions", type=int, default=4000)
    parser.add_argument("--no-power", action="store_true")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--output", default="bench/results/crypto_factors_report.json")
    parser.add_argument("--markdown", default="docs/crypto_factors_report.md")
    args = parser.parse_args()
    report = run(
        args.data,
        repetitions=args.repetitions,
        power=not args.no_power,
        workers=args.workers,
        progress="tmp/c6-report-progress.json",
    )
    Path(args.output).write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    Path(args.markdown).write_text(markdown(report))


if __name__ == "__main__":
    main()
