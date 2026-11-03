"""Pure correlation kernels (Pearson r, Spearman rho, Kendall tau-b, Partial correlation)."""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

from pyscio.core.errors import SampleSizeError, StatError
from pyscio.core.guards import check_paired, clean_1d
from pyscio.core.types import CorrelationResult


def pearson_r(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    conf_level: float = 0.95,
) -> CorrelationResult:
    """Compute Pearson product-moment correlation coefficient $r$ with Fisher $z$-transform CI."""
    arr_x = clean_1d(x, name="x")
    arr_y = clean_1d(y, name="y")
    check_paired(arr_x, arr_y, min_n=3, check_diff_var=False)

    n = len(arr_x)
    dof = float(n - 2)
    res = stats.pearsonr(arr_x, arr_y)
    r = float(np.clip(res.statistic, -1.0, 1.0))
    p = float(res.pvalue)

    denom = max(1.0 - r**2, 1e-15)
    t_stat = float(r * math.sqrt(dof / denom))

    if n > 3 and abs(r) < 0.999999:
        z = math.atanh(r)
        se_z = 1.0 / math.sqrt(n - 3.0)
        z_crit = float(stats.norm.ppf(1.0 - (1.0 - conf_level) / 2.0))
        ci = (float(math.tanh(z - z_crit * se_z)), float(math.tanh(z + z_crit * se_z)))
    else:
        ci = (r, r)

    return CorrelationResult(r=r, stat=t_stat, p=p, dof=dof, ci=ci, method="Pearson r")


def spearman_r(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    conf_level: float = 0.95,
) -> CorrelationResult:
    """Compute Spearman rank-order correlation $\\rho$ with Bonett-Wright $z$-transform CI."""
    arr_x = clean_1d(x, name="x")
    arr_y = clean_1d(y, name="y")
    check_paired(arr_x, arr_y, min_n=3, check_diff_var=False)

    n = len(arr_x)
    dof = float(n - 2)
    res = stats.spearmanr(arr_x, arr_y)
    rho = float(np.clip(res.statistic, -1.0, 1.0))
    p = float(res.pvalue)

    denom = max(1.0 - rho**2, 1e-15)
    t_stat = float(rho * math.sqrt(dof / denom))

    if n > 3 and abs(rho) < 0.999999:
        z = math.atanh(rho)
        # Bonett & Wright (2000) variance adjustment for Spearman rank correlation
        se_z = math.sqrt((1.0 + (rho**2) / 2.0) / (n - 3.0))
        z_crit = float(stats.norm.ppf(1.0 - (1.0 - conf_level) / 2.0))
        ci = (float(math.tanh(z - z_crit * se_z)), float(math.tanh(z + z_crit * se_z)))
    else:
        ci = (rho, rho)

    return CorrelationResult(r=rho, stat=t_stat, p=p, dof=dof, ci=ci, method="Spearman rho")


def kendall_tau(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    conf_level: float = 0.95,
) -> CorrelationResult:
    """Compute Kendall's rank correlation $\\tau_b$ with Fieller-Hartley-Pearson $z$-transform CI."""
    arr_x = clean_1d(x, name="x")
    arr_y = clean_1d(y, name="y")
    check_paired(arr_x, arr_y, min_n=4, check_diff_var=False)

    n = len(arr_x)
    res = stats.kendalltau(arr_x, arr_y)
    tau = float(np.clip(res.statistic, -1.0, 1.0))
    p = float(res.pvalue)

    se_tau = math.sqrt((2.0 * (2.0 * n + 5.0)) / (9.0 * n * (n - 1.0)))
    z_stat = float(tau / se_tau)

    if n > 4 and abs(tau) < 0.999999:
        z = math.atanh(tau)
        # Fieller, Hartley, & Pearson (1957) variance in Fisher z space for Kendall's tau
        se_z = math.sqrt(0.437 / (n - 4.0))
        z_crit = float(stats.norm.ppf(1.0 - (1.0 - conf_level) / 2.0))
        ci = (float(math.tanh(z - z_crit * se_z)), float(math.tanh(z + z_crit * se_z)))
    else:
        ci = (tau, tau)

    return CorrelationResult(
        r=tau, stat=z_stat, p=p, dof=float(n - 2), ci=ci, method="Kendall tau-b"
    )


def partial_corr(
    x: np.ndarray | list[float],
    y: np.ndarray | list[float],
    covar: np.ndarray | list[float] | list[list[float]],
    *,
    method: str = "pearson",
    conf_level: float = 0.95,
) -> CorrelationResult:
    """Compute partial correlation $r_{xy \\cdot Z}$ controlling for covariates via QR projection."""
    arr_x = np.asarray(x, dtype=np.float64)
    arr_y = np.asarray(y, dtype=np.float64)
    arr_z = np.asarray(covar, dtype=np.float64)

    if arr_x.ndim != 1 or arr_y.ndim != 1:
        raise StatError("x and y must be 1D arrays for partial_corr.")
    if arr_z.ndim == 1:
        arr_z = arr_z[:, None]
    elif arr_z.ndim != 2:
        raise StatError("covar must be a 1D or 2D array of covariates.")

    if len(arr_x) != len(arr_y) or len(arr_x) != arr_z.shape[0]:
        raise SampleSizeError("x, y, and covar must have identical sample lengths.")

    valid = np.isfinite(arr_x) & np.isfinite(arr_y) & np.all(np.isfinite(arr_z), axis=1)
    vx = arr_x[valid]
    vy = arr_y[valid]
    vz = arr_z[valid]

    n, k = vz.shape
    dof = float(n - 2 - k)
    if dof < 1:
        raise SampleSizeError(
            f"Insufficient observations (n={n}) for partial correlation with {k} covariates."
        )

    m_low = method.lower()
    if m_low == "spearman":
        vx = stats.rankdata(vx)
        vy = stats.rankdata(vy)
        vz = np.column_stack([stats.rankdata(vz[:, j]) for j in range(k)])
        label = "Partial Spearman rho"
    elif m_low == "pearson":
        label = "Partial Pearson r"
    else:
        raise StatError(
            f"Unsupported partial_corr method '{method}'; choose 'pearson' or 'spearman'."
        )

    z_aug = np.column_stack([np.ones(n, dtype=np.float64), vz])
    q_mat, _ = np.linalg.qr(z_aug, mode="reduced")
    res_x = vx - q_mat @ (q_mat.T @ vx)
    res_y = vy - q_mat @ (q_mat.T @ vy)

    denom_xy = float(np.sqrt(np.sum(res_x**2) * np.sum(res_y**2)))
    if denom_xy <= 1e-15:
        raise StatError("Residual variance is zero after partialling out covariates.")

    r = float(np.clip(np.sum(res_x * res_y) / denom_xy, -1.0, 1.0))
    t_stat = float(r * math.sqrt(dof / max(1.0 - r**2, 1e-15)))
    p = float(2.0 * stats.t.sf(abs(t_stat), df=dof))

    if n - 3 - k > 0 and abs(r) < 0.999999:
        z_val = math.atanh(r)
        se_z = (
            math.sqrt((1.0 + (r**2) / 2.0) / (n - 3.0 - k))
            if m_low == "spearman"
            else 1.0 / math.sqrt(n - 3.0 - k)
        )
        z_crit = float(stats.norm.ppf(1.0 - (1.0 - conf_level) / 2.0))
        ci = (float(math.tanh(z_val - z_crit * se_z)), float(math.tanh(z_val + z_crit * se_z)))
    else:
        ci = (r, r)

    return CorrelationResult(r=r, stat=t_stat, p=p, dof=dof, ci=ci, method=label)
