"""LW 2004 常相关公式的独立蒙特卡洛及真模型偏差统计对照。"""

import json
from pathlib import Path

import numpy as np
from baseline import metadata
from scipy.stats import chi2

from ashare.risk.engine import cross_section_batch
from ashare.risk.fundamental import FundamentalConfig, estimate_covariances
from ashare.risk.kernels import rolling_bias
from ashare.risk.statistical import shrinkage_components


def lw2004(x):
    """Honey, I Shrunk the Sample Covariance Matrix, Appendix B, constant correlation."""
    t, n = x.shape
    y = x - x.mean(0)
    s = y.T @ y / t
    sd = np.sqrt(np.diag(s))
    corr = s / np.outer(sd, sd)
    average = (corr.sum() - n) / (n * (n - 1))
    target = average * np.outer(sd, sd)
    np.fill_diagonal(target, np.diag(s))
    square = y**2
    pi = square.T @ square / t - s**2
    theta = (y**3).T @ y / t - np.diag(s)[:, None] * s
    rho = np.trace(pi) + average * np.sum(
        sd[None, :] / sd[:, None] * theta * (1 - np.eye(n))
    )
    strength = np.clip(
        (pi.sum() - rho) / np.linalg.norm(s - target, "fro") ** 2 / t, 0, 1
    )
    return strength, target, s


def monte_carlo():
    rng = np.random.default_rng(42)
    rows = []
    for t, n, kind in ((120, 60, 1), (250, 100, 1), (120, 60, 3), (500, 50, 3)):
        values = []
        for _ in range(60):
            sd = rng.uniform(0.01, 0.04, n)
            loading = (
                rng.uniform(0.5, 1.5, (n, 1)) * 0.012
                if kind == 1
                else rng.normal(size=(n, 3)) * 0.008
            )
            truth = loading @ loading.T + np.diag(sd**2)
            x = rng.multivariate_normal(np.zeros(n), truth, size=t)
            paper, target, sample = lw2004(x)
            _, _, meta = shrinkage_components(x, np.inf, "constant_correlation")
            ours = meta["shrinkage"]
            oracle = np.clip(
                np.sum((sample - target) * (sample - truth))
                / np.sum((sample - target) ** 2),
                0,
                1,
            )

            def loss(d, target=target, sample=sample, truth=truth):
                return np.linalg.norm(d * target + (1 - d) * sample - truth) ** 2

            values.append(
                [
                    ours,
                    paper,
                    oracle,
                    loss(ours) / loss(oracle),
                    loss(paper) / loss(oracle),
                    loss(0) / loss(oracle),
                ]
            )
        rows.append(
            dict(
                T=t,
                N=n,
                factors=kind,
                repetitions=60,
                mean=dict(
                    zip(
                        (
                            "delta_ours",
                            "delta_paper",
                            "delta_oracle",
                            "loss_ours",
                            "loss_paper",
                            "loss_unshrunk",
                        ),
                        np.mean(values, axis=0).tolist(),
                        strict=True,
                    )
                ),
            )
        )
    return rows


def bias_reference():
    window = 252
    lo, hi = np.sqrt(chi2.ppf([0.025, 0.975], window - 1) / (window - 1))
    rng = np.random.default_rng(0)
    z = rng.normal(size=(400, window, 50)).std(axis=1, ddof=1)
    result = {
        "iid_nonoverlap": dict(
            windows=z.size,
            coverage=float(np.mean((z >= lo) & (z <= hi))),
            mean=float(z.mean()),
        ),
        "seeds": {},
    }
    for seed in (711, 1, 2, 3, 4, 5):
        rng = np.random.default_rng(seed)
        t, n, k, train = 3000, 500, 13, 2000
        cap = np.exp(rng.normal(10, 1, n))
        inds = np.eye(8)[np.arange(n) % 8]
        style = rng.normal(size=(n, k - 9))
        style -= np.average(style, axis=0, weights=cap)
        b = np.column_stack((np.ones(n), inds, style))
        transform = np.eye(k)
        transform[1:9, 1:9] -= (cap @ inds / cap.sum())[None, :]
        var = np.linspace(0.00003, 0.0001, k)
        f = (transform * var) @ transform.T
        d = rng.uniform(0.00015, 0.0004, n)
        realized_f = rng.normal(size=(t, k)) * np.sqrt(var) @ transform.T
        r = realized_f @ b.T + rng.normal(size=(t, n)) * np.sqrt(d)
        estimated, residuals = cross_section_batch(
            r[:train],
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
        fh, dh, _, _ = estimate_covariances(estimated, residuals, b, cap, config)
        groups = {
            "factor": np.linalg.pinv(b).T,
            "random": rng.normal(size=(n, 100)) / np.sqrt(n),
            "cap_weighted": (cap / cap.sum())[:, None],
            "single_stock": np.eye(n),
        }
        rows = {}
        for group, w in groups.items():
            realized = r[train:] @ w
            for tag, ff, dd in (("truth", f, d), ("estimated", fh, dh)):
                y = b.T @ w
                sigma = np.sqrt(np.sum(y * (ff @ y), axis=0) + dd @ (w * w))
                values = rolling_bias(
                    realized, np.broadcast_to(sigma, realized.shape), window
                )
                nonoverlap = values[::window]
                rows[group + "/" + tag] = dict(
                    mean=float(values.mean()),
                    overlap_coverage=float(np.mean((values >= lo) & (values <= hi))),
                    nonoverlap_coverage=float(
                        np.mean((nonoverlap >= lo) & (nonoverlap <= hi))
                    ),
                )
        result["seeds"][str(seed)] = rows
    return result


def run(output):
    result = dict(
        metadata=metadata(), constant_correlation=monte_carlo(), bias=bias_reference()
    )
    Path(output).write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    path = Path(".cache/risk-bench/review-comparisons.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    run(path)
