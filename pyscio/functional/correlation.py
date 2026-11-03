"""Pure correlation kernels (Pearson r and Spearman rho) with exact z-transformed CIs."""

from __future__ import annotations

import math

import numpy as np
from scipy import stats

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
