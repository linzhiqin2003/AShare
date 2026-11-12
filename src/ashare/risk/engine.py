"""基准选定的执行内核；numpy 实现在 kernels 中独立保留。"""

import numpy as np
import portopt
from portopt import get_num_threads, set_num_threads  # noqa: F401

from . import kernels


def _short_factor_panel(returns):
    shape = np.shape(returns)
    return len(shape) == 2 and shape[0] <= 1008 and shape[1] <= 64


def cross_section(returns, exposures, market_cap, estimation, industry_indices=()):
    f, residual = cross_section_batch(
        np.asarray(returns)[None, :],
        exposures,
        np.asarray(market_cap)[None, :],
        np.asarray(estimation)[None, :],
        industry_indices,
    )
    return f[0], residual[0]


def cross_section_batch(
    returns, exposures, market_cap, estimation, industry_indices=(), threads=0
):
    return portopt.regress(
        returns, exposures, market_cap, estimation, industry_indices, threads
    )


def ew_variance(returns, half_life=252.0, lags=0):
    if _short_factor_panel(returns):
        return kernels.ew_variance(returns, half_life, lags)
    return portopt.ew_variance(returns, half_life, lags, threads=0)


def rolling_bias(realized, predicted, window=252):
    return portopt.rolling_bias(realized, predicted, window, threads=0)


def ew_covariance(returns, half_life=252.0, lags=0):
    if _short_factor_panel(returns):
        return kernels.ew_covariance(returns, half_life, lags)
    return portopt.ew_covariance(returns, half_life, lags, threads=0)


def factor_covariance(
    returns, vol_half_life=84.0, corr_half_life=504.0, vol_lags=5, corr_lags=2
):
    from .kernels import psd

    corr = ew_covariance(returns, corr_half_life, corr_lags)
    sd = np.sqrt(np.maximum(np.diag(corr), 1e-30))
    corr /= np.outer(sd, sd)
    vol = np.sqrt(ew_variance(returns, vol_half_life, vol_lags))
    return psd(corr * np.outer(vol, vol))
