"""Robust estimators of location/scale and Yuen-Welch trimmed two-sample t-test."""

from __future__ import annotations

import math

import numpy as np
from scipy import stats
from scipy.stats import mstats

from pyscio.core.errors import StatError
from pyscio.core.guards import check_groups, check_sample_size, clean_1d
from pyscio.core.types import TTestResult


def trimmed_mean(
    x: np.ndarray | list[float],
    trim: float = 0.1,
    *,
    proportiontocut: float | None = None,
) -> float:
    """Compute the symmetrically trimmed sample mean."""
    if proportiontocut is not None:
        trim = proportiontocut
    arr = clean_1d(x, name="x")
    if not (0.0 <= trim < 0.5):
        raise StatError(f"trim proportion must be in [0.0, 0.5), got {trim}.")
    return float(stats.trim_mean(arr, proportiontocut=trim))


def winsorized_mean(
    x: np.ndarray | list[float],
    trim: float = 0.1,
    *,
    limits: tuple[float, float] | float | None = None,
    proportiontocut: float | None = None,
) -> float:
    """Compute the Winsorized sample mean by clamping extreme tails."""
    arr = clean_1d(x, name="x")
    if proportiontocut is not None:
        lim: tuple[float, float] = (float(proportiontocut), float(proportiontocut))
    elif limits is not None:
        lim = (float(limits), float(limits)) if isinstance(limits, (int, float)) else limits
    else:
        lim = (float(trim), float(trim))
    w = mstats.winsorize(arr, limits=lim)
    return float(np.mean(w))


def winsorized_var(x: np.ndarray | list[float], trim: float = 0.2) -> float:
    """Compute the sample Winsorized variance (ddof=1)."""
    arr = clean_1d(x, name="x")
    w = np.asarray(mstats.winsorize(arr, limits=(trim, trim)), dtype=np.float64)
    return float(np.var(w, ddof=1))


def huber_location(
    x: np.ndarray | list[float],
    c: float = 1.345,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> tuple[float, float]:
    """Compute Huber's M-estimator of location and MAD-normalized scale via IRLS."""
    arr = clean_1d(x, name="x")
    check_sample_size(arr, min_n=3, name="x")
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med))) * 1.482602218505602
    if mad <= 1e-15:
        sd = float(np.std(arr, ddof=1))
        if sd <= 1e-15:
            return med, 0.0
        mad = sd

    mu = med
    for _ in range(max_iter):
        u = (arr - mu) / mad
        weights = np.where(np.abs(u) <= c, 1.0, c / np.abs(u))
        w_sum = float(np.sum(weights))
        if w_sum <= 0.0:
            break
        mu_new = float(np.sum(weights * arr) / w_sum)
        if abs(mu_new - mu) <= tol * mad:
            mu = mu_new
            break
        mu = mu_new

    return mu, mad


def yuen_t(
    a: np.ndarray | list[float],
    b: np.ndarray | list[float],
    trim: float = 0.2,
    alternative: str = "two-sided",
    conf_level: float = 0.95,
) -> TTestResult:
    """Yuen-Welch trimmed two-sample t-test for heavy-tailed and skewed distributions."""
    if not (0.0 <= trim < 0.5):
        raise StatError(f"trim proportion must be in [0.0, 0.5), got {trim}.")

    g1 = clean_1d(a, name="sample_a")
    g2 = clean_1d(b, name="sample_b")
    check_groups(g1, g2, min_n=5)

    n1, n2 = len(g1), len(g2)
    g1_cut = math.floor(trim * n1)
    g2_cut = math.floor(trim * n2)

    h1 = n1 - 2 * g1_cut
    h2 = n2 - 2 * g2_cut
    if h1 < 2 or h2 < 2:
        raise StatError("Sample size too small for the requested trimming proportion.")

    tm1 = trimmed_mean(g1, trim=trim)
    tm2 = trimmed_mean(g2, trim=trim)
    diff = tm1 - tm2

    wv1 = winsorized_var(g1, trim=trim)
    wv2 = winsorized_var(g2, trim=trim)

    d1 = ((n1 - 1.0) * wv1) / (h1 * (h1 - 1.0))
    d2 = ((n2 - 1.0) * wv2) / (h2 * (h2 - 1.0))
    se2 = d1 + d2
    se = math.sqrt(se2) if se2 > 0 else 0.0

    denom = (d1**2) / (h1 - 1.0) + (d2**2) / (h2 - 1.0)
    dof = (se2**2) / denom if denom > 0 else float(h1 + h2 - 2)
    stat = diff / se if se > 0 else 0.0

    if alternative == "two-sided":
        p = 2.0 * float(stats.t.sf(abs(stat), df=dof))
    elif alternative == "less":
        p = float(stats.t.cdf(stat, df=dof))
    else:
        p = float(stats.t.sf(stat, df=dof))

    t_crit = float(stats.t.ppf(1.0 - (1.0 - conf_level) / 2.0, df=dof))
    ci = (diff - t_crit * se, diff + t_crit * se)

    return TTestResult(stat=stat, p=p, dof=dof, ci=ci, mean_diff=diff, se_diff=se)
