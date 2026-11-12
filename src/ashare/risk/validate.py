"""风险检验命令：合成真值还原，或读取统一字段的 NPZ 面板。"""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.stats import chi2

from .data import MemoryPanel, adjusted_returns
from .engine import cross_section_batch, rolling_bias
from .fundamental import FundamentalConfig, estimate_covariances
from .model import RiskModel
from .series import model_series
from .statistical import statistical_model


def bias_summary(realized, predicted, window=252):
    values = rolling_bias(realized, predicted, window)
    non_overlapping = values[::window]
    valid = np.isfinite(values)
    interval = np.sqrt(chi2.ppf([0.025, 0.975], window - 1) / (window - 1))
    if not valid.any():
        return {
            "mean": None,
            "within_95pct": None,
            "windows": 0,
            "interval": interval.tolist(),
            "non_overlapping": {"mean": None, "within_95pct": None, "windows": 0},
        }
    selected = values[valid]
    return {
        "mean": float(selected.mean()),
        "within_95pct": float(
            np.mean((selected >= interval[0]) & (selected <= interval[1]))
        ),
        "windows": int(len(selected)),
        "interval": interval.tolist(),
        "non_overlapping": {
            "mean": float(np.nanmean(non_overlapping))
            if np.isfinite(non_overlapping).any()
            else None,
            "within_95pct": float(
                np.mean(
                    (non_overlapping[np.isfinite(non_overlapping)] >= interval[0])
                    & (non_overlapping[np.isfinite(non_overlapping)] <= interval[1])
                )
            )
            if np.isfinite(non_overlapping).any()
            else None,
            "windows": int(np.isfinite(non_overlapping).sum()),
        },
    }


def portfolio_prediction(model, weights):
    y = model.B.T @ weights
    return np.sqrt(
        np.maximum(np.sum(y * (model.F @ y), axis=0) + model.D @ (weights * weights), 0)
    )


def common_covariance_error(model, truth):
    """用小型 Gram 矩阵比较公共协方差，允许 PCA 因子旋转，不展开 N×N。"""
    gg = model.B.T @ model.B
    tt = truth.B.T @ truth.B
    gt = model.B.T @ truth.B
    estimated_norm = np.trace(model.F @ gg @ model.F @ gg)
    true_norm = np.trace(truth.F @ tt @ truth.F @ tt)
    cross = np.trace(model.F @ gt @ truth.F @ gt.T)
    return float(np.sqrt(max(estimated_norm + true_norm - 2 * cross, 0) / true_norm))


def synthetic(seed=711, t=3000, n=500, k=13, train=2000):
    rng = np.random.default_rng(seed)
    cap = np.exp(rng.normal(10, 1, n))
    inds = np.eye(8)[np.arange(n) % 8]
    style = rng.normal(size=(n, k - 9))
    style -= np.average(style, axis=0, weights=cap)
    b = np.column_stack((np.ones(n), inds, style))
    transform = np.eye(k)
    industry_weight = cap @ inds / cap.sum()
    transform[1:9, 1:9] -= industry_weight[None, :]
    variances = np.linspace(0.00003, 0.0001, k)
    covariance = (transform * variances) @ transform.T
    d = rng.uniform(0.00015, 0.0004, n)
    factor = rng.normal(size=(t, k)) * np.sqrt(variances) @ transform.T
    noise = rng.normal(size=(t, n)) * np.sqrt(d)
    returns = factor @ b.T + noise
    estimated, residual = cross_section_batch(
        returns[:train],
        b,
        np.broadcast_to(cap, (train, n)),
        np.ones((train, n), bool),
        np.arange(1, 9),
    )
    config = FundamentalConfig(
        vol_half_life=1e12,
        corr_half_life=1e12,
        specific_half_life=1e12,
        specific_corr_half_life=1e12,
        vol_lags=0,
        corr_lags=0,
        specific_lags=0,
        shrinkage=0,
        regime=False,
        history=train,
    )
    f, dd, _, _ = estimate_covariances(estimated, residual, b, cap, config)
    fundamental = RiskModel(
        tuple(map(str, range(n))), b, tuple(f"f{i}" for i in range(k)), f, dd
    )
    statistical = statistical_model(returns[:train], half_life=np.inf, components=k - 1)
    true = RiskModel(fundamental.symbols, b, fundamental.factors, covariance, d)
    portfolios = {
        "factor": np.linalg.pinv(b).T,
        "random": rng.normal(size=(n, 100)) / np.sqrt(n),
        "cap_weighted": (cap / cap.sum())[:, None],
    }
    reports = {}
    common_true = factor[:train] @ b.T
    for name, model in (("fundamental", fundamental), ("statistical", statistical)):
        recovery = {
            "common_covariance_relative_frobenius": common_covariance_error(
                model, true
            ),
            "total_variance_relative_mae": float(
                np.mean(
                    np.abs(
                        (np.sum((model.B @ model.F) * model.B, axis=1) + model.D)
                        / (np.sum((b @ covariance) * b, axis=1) + d)
                        - 1
                    )
                )
            ),
            "specific_relative_mae": float(np.mean(np.abs(model.D / d - 1))),
        }
        random = portfolios["random"]
        recovery["portfolio_variance_relative_mae"] = float(
            np.mean(
                np.abs(
                    (
                        portfolio_prediction(model, random)
                        / portfolio_prediction(true, random)
                    )
                    ** 2
                    - 1
                )
            )
        )
        if name == "fundamental":
            recovery["factor_covariance_relative_frobenius"] = float(
                np.linalg.norm(f - covariance) / np.linalg.norm(covariance)
            )
            recovery["factor_returns_rmse"] = float(
                np.sqrt(np.mean((estimated - factor[:train]) ** 2))
            )
            common = estimated @ b.T
        else:
            scores = returns[:train] @ model.B
            common = scores @ model.B.T
        recovery["common_returns_relative_rmse"] = float(
            np.linalg.norm(common - common_true) / np.linalg.norm(common_true)
        )
        bias = {}
        for group, w in portfolios.items():
            realized = returns[train:] @ w
            predicted = np.broadcast_to(portfolio_prediction(model, w), realized.shape)
            bias[group] = bias_summary(realized, predicted)
        stock_vol = np.sqrt(np.sum((model.B @ model.F) * model.B, axis=1) + model.D)
        bias["single_stock"] = bias_summary(
            returns[train:], np.broadcast_to(stock_vol, returns[train:].shape)
        )
        reports[name] = {"recovery": recovery, "bias": bias}
    truth_bias = {}
    for group, weights in portfolios.items():
        realized = returns[train:] @ weights
        truth_bias[group] = bias_summary(
            realized,
            np.broadcast_to(portfolio_prediction(true, weights), realized.shape),
        )
    truth_bias["single_stock"] = bias_summary(
        returns[train:],
        np.broadcast_to(
            np.sqrt(np.sum((b @ covariance) * b, axis=1) + d), returns[train:].shape
        ),
    )
    return {
        "truth_bias": truth_bias,
        "seed": seed,
        "shape": [t, n, k],
        "train": train,
        "holdout": t - train,
        "config": asdict(config),
        "sampling_error": "IID variance relative SE sqrt(2/2000)=3.16%; factor covariance Frobenius order sqrt(K/T)=8.06%; WLS residual degrees-of-freedom bias K/N=2.6%.",
        "tolerances": {
            "common_covariance_relative_frobenius": 0.16,
            "factor_covariance_relative_frobenius": 0.16,
            "specific_relative_mae": 0.08,
            "portfolio_variance_relative_mae": 0.12,
            "bias_mean_interval": [0.85, 1.15],
        },
        "models": reports,
    }


def validate_panel(source, output, *, start=600, window=252, cache=None):
    """真实数据入口；风险预测取 t，检验收益取 t+1，避免同日拟合偏差。"""
    if len(source.dates) - start - 1 < window:
        raise ValueError("检验区间不足一个完整窗口")
    rng = np.random.default_rng(819)
    n = len(source.symbols)
    random = rng.normal(size=(n, 100)) / np.sqrt(n)
    returns = adjusted_returns(source, len(source.dates) - 1)
    try:
        returns -= source["risk_free"]
    except KeyError:
        pass
    all_results = {}
    for kind in ("fundamental", "statistical"):
        actual = {
            key: [] for key in ("factor", "random", "cap_weighted", "single_stock")
        }
        forecasts = {key: [] for key in actual}
        for day, model in enumerate(
            model_series(
                source, kind=kind, start=start, end=len(source.dates) - 1, cache=cache
            ),
            start=start,
        ):
            cap = np.asarray(source["total_mv"][day], dtype=float)
            cap = np.where(
                np.isfinite(cap) & (cap > 0) & source["is_listed"][day], cap, 0
            )
            covered = cap > 0
            factor = np.zeros_like(model.B)
            if covered.any():
                factor[covered] = np.linalg.pinv(model.B[covered]).T
            weights = {
                "factor": factor,
                "random": np.where(covered[:, None], random, 0),
                "cap_weighted": (cap / max(cap.sum(), 1))[:, None],
            }
            next_return = returns[day + 1].copy()
            # 实现组合按收盘估值；停牌但有合法收盘价仍有可观测收益。
            # 缺价不填零：只要组合持有缺价证券，该日组合检验样本即缺失。
            for name, w in weights.items():
                r = np.nan_to_num(next_return) @ w
                r[
                    (~np.isfinite(next_return)).astype(float) @ (np.abs(w) > 1e-14) > 0
                ] = np.nan
                actual[name].append(r)
                forecasts[name].append(portfolio_prediction(model, w))
            actual["single_stock"].append(next_return)
            forecasts["single_stock"].append(
                np.sqrt(np.sum((model.B @ model.F) * model.B, axis=1) + model.D)
            )
        all_results[kind] = {
            "bias": {
                name: bias_summary(
                    np.asarray(actual[name]), np.asarray(forecasts[name]), window
                )
                for name in actual
            }
        }
    result = {
        "version": source.version,
        "start": str(source.dates[start]),
        "models": all_results,
    }
    write_report(result, output)
    return result


def write_report(result, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2)
    )
    lines = [
        "# 风险模型检验",
        "",
        "预测均使用检验收益之前的模型。置信区间是 IID 正态、已知预测波动的名义 95% 区间；重叠窗口不独立。",
        "",
        "| 模型 | 组合 | 偏差均值 | 区间覆盖比例 | 有效窗口 |",
        "|---|---|---:|---:|---:|",
    ]
    reports = dict(result["models"])
    if "truth_bias" in result:
        reports["true_model"] = {"bias": result["truth_bias"]}
    for name, report in reports.items():
        for group, row in report["bias"].items():
            mean, coverage = row["mean"], row["within_95pct"]
            lines.append(
                f"| {name} | {group} | {mean if mean is None else f'{mean:.5f}'} | {coverage if coverage is None else f'{coverage:.2%}'} | {row['windows']} |"
            )
            independent = row.get("non_overlapping")
            if independent is not None:
                mean, coverage = independent["mean"], independent["within_95pct"]
                lines.append(
                    f"| {name}（不重叠） | {group} | {mean if mean is None else f'{mean:.5f}'} | {coverage if coverage is None else f'{coverage:.2%}'} | {independent['windows']} |"
                )
    for name, report in result["models"].items():
        if "recovery" in report:
            lines.extend(
                (
                    "",
                    f"{name} 还原误差：",
                    "",
                    "```json",
                    json.dumps(report["recovery"], indent=2),
                    "```",
                )
            )
    (output / "validation.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, default=Path(".cache/risk-validation"))
    parser.add_argument("--start", type=int, default=600)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    if args.input:
        with np.load(args.input, allow_pickle=False) as data:
            source = MemoryPanel(
                data["dates"],
                tuple(data["symbols"]),
                str(data["version"]),
                {
                    name: data[name]
                    for name in data.files
                    if name not in ("dates", "symbols", "version")
                },
            )
        validate_panel(source, args.output, start=args.start, cache=args.cache)
    else:
        result = synthetic()
        write_report(result, args.output)
    print(args.output / "validation.md")


if __name__ == "__main__":
    main()
