"""Stateless time-series diagnostics: Augmented Dickey-Fuller (ADF), Ljung-Box Q, and Durbin-Watson."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import stats

from pyscio.core.errors import SampleSizeError
from pyscio.core.guards import clamp_p, clean_1d
from pyscio.core.types import StationarityResult


def durbin_watson(residuals: np.ndarray | Sequence[float]) -> float:
    """Compute the Durbin-Watson autocorrelation statistic $DW \\in [0, 4]$ on regression residuals."""
    e = clean_1d(residuals, name="residuals")
    if len(e) < 2:
        raise SampleSizeError("durbin_watson requires at least 2 observations.", n=len(e), min_n=2)
    denom = float(np.sum(e**2))
    if denom <= 1e-15:
        return 2.0
    diff_sq = float(np.sum(np.diff(e) ** 2))
    return float(diff_sq / denom)


def ljung_box(
    x: np.ndarray | Sequence[float],
    lags: int = 10,
    alpha: float = 0.05,
) -> StationarityResult:
    """Compute the Ljung-Box portmanteau test for serial autocorrelation up to `lags`.

    Under $H_0$, the series exhibits zero autocorrelation up to lag $m$ (white noise).
    """
    arr = clean_1d(x, name="series")
    n = len(arr)
    max_lag = min(max(1, int(lags)), n - 2)
    if n < 4 or max_lag < 1:
        raise SampleSizeError("ljung_box requires at least 4 observations.", n=n, min_n=4)

    centered = arr - np.mean(arr)
    c0 = float(np.sum(centered**2))
    if c0 <= 1e-15:
        return StationarityResult(
            stat=0.0,
            p=1.0,
            lags=max_lag,
            n_obs=n,
            is_stationary=True,
            method="Ljung-Box Q",
        )

    q_sum = 0.0
    for k in range(1, max_lag + 1):
        rk = float(np.sum(centered[k:] * centered[:-k])) / c0
        q_sum += (rk**2) / float(n - k)

    q_stat = float(n * (n + 2) * q_sum)
    p_raw = float(stats.chi2.sf(q_stat, df=max_lag))
    p_val, _ = clamp_p(p_raw)

    return StationarityResult(
        stat=q_stat,
        p=p_val,
        lags=max_lag,
        n_obs=n,
        is_stationary=p_val < alpha,
        method="Ljung-Box Q",
    )


def _mackinnon_adf_p_constant(tau: float) -> float:
    """Approximate MacKinnon (1994, 2010) p-value for ADF unit-root test with constant drift.

    Critical values (with constant): 1% = -3.430, 5% = -2.861, 10% = -2.567, 50% = -1.57, 90% = -0.44.
    """
    # Calibrated MacKinnon quantiles (tau -> p)
    quantiles_tau = np.array(
        [-4.50, -3.96, -3.430, -3.12, -2.861, -2.567, -1.95, -1.57, -0.44, 0.60, 2.00]
    )
    quantiles_p = np.array(
        [0.0002, 0.0015, 0.010, 0.025, 0.050, 0.100, 0.309, 0.500, 0.900, 0.988, 0.999]
    )
    if tau <= quantiles_tau[0]:
        # Extrapolate left tail smoothly via log-linear decay
        slope = (np.log(quantiles_p[1]) - np.log(quantiles_p[0])) / (
            quantiles_tau[1] - quantiles_tau[0]
        )
        return float(np.clip(quantiles_p[0] * np.exp(slope * (tau - quantiles_tau[0])), 1e-16, 1.0))
    if tau >= quantiles_tau[-1]:
        return 0.999
    return float(np.interp(tau, quantiles_tau, quantiles_p))


def adf_test(
    x: np.ndarray | Sequence[float],
    max_lags: int | None = None,
    alpha: float = 0.05,
) -> StationarityResult:
    """Compute the Augmented Dickey-Fuller (ADF) unit-root test for stationarity.

    Estimates $\\Delta y_t = \\alpha + \\gamma y_{t-1} + \\sum_{i=1}^p \\delta_i \\Delta y_{t-i} + \\varepsilon_t$.
    Under $H_0: \\gamma = 0$, the series has a unit root (non-stationary); rejecting $H_0$
    ($\\tau < \\tau_{\\text{crit}}$, $p < \\alpha$) indicates stationarity.
    """
    y = clean_1d(x, name="series")
    n = len(y)
    if n < 8:
        raise SampleSizeError("adf_test requires at least 8 observations.", n=n, min_n=8)

    if max_lags is None:
        # Schwert (1989) rule of thumb capped for sample size
        p_lags = min(int(12.0 * ((n / 100.0) ** 0.25)), max(0, (n - 5) // 2))
    else:
        p_lags = max(0, min(int(max_lags), (n - 4) // 2))

    dy = np.diff(y)
    # Effective sample starts at index p_lags in dy
    target_dy = dy[p_lags:]
    y_lag1 = y[p_lags:-1]
    n_eff = len(target_dy)

    cols = [np.ones(n_eff, dtype=np.float64), y_lag1]
    for lag in range(1, p_lags + 1):
        cols.append(dy[p_lags - lag : len(dy) - lag])

    x_mat = np.column_stack(cols)
    q_mat, r_mat = np.linalg.qr(x_mat, mode="reduced")
    beta = np.linalg.solve(r_mat, q_mat.T @ target_dy)
    resid = target_dy - x_mat @ beta
    df_resid = max(1, n_eff - x_mat.shape[1])
    s2 = float(np.sum(resid**2)) / float(df_resid)

    r_inv = np.linalg.inv(r_mat)
    se_gamma = float(np.sqrt(max(s2 * np.sum(r_inv[1, :] ** 2), 1e-15)))
    tau_stat = float(beta[1] / se_gamma)

    p_raw = _mackinnon_adf_p_constant(tau_stat)
    p_val, _ = clamp_p(p_raw)

    return StationarityResult(
        stat=tau_stat,
        p=p_val,
        lags=p_lags,
        n_obs=n_eff,
        is_stationary=p_val < alpha,
        method="Augmented Dickey-Fuller (ADF)",
    )
