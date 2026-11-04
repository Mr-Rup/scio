"""Rouder's JZS (Jeffreys-Zellner-Siow) Default Bayes Factor for t-tests (Feature F9)."""

from __future__ import annotations

import math

import numpy as np
from scipy import integrate

from pyscio.core.errors import StatError
from pyscio.core.guards import (
    check_groups,
    check_paired,
    check_sample_size,
    check_variance,
    clean_1d,
)
from pyscio.core.types import BayesFactorResult


def interpret_bf(bf10: float) -> str:
    """Classify Bayes Factor BF10 evidence strength using Jeffreys (1961) / Wagenmakers scale."""
    if bf10 >= 100.0:
        return "extreme evidence for H1"
    if bf10 >= 30.0:
        return "very strong evidence for H1"
    if bf10 >= 10.0:
        return "strong evidence for H1"
    if bf10 >= 3.0:
        return "moderate evidence for H1"
    if bf10 > 1.0:
        return "anecdotal evidence for H1"
    if bf10 == 1.0:
        return "no evidence (neutral)"

    bf01 = 1.0 / max(bf10, 1e-300)
    if bf01 >= 100.0:
        return "extreme evidence for H0"
    if bf01 >= 30.0:
        return "very strong evidence for H0"
    if bf01 >= 10.0:
        return "strong evidence for H0"
    if bf01 >= 3.0:
        return "moderate evidence for H0"
    return "anecdotal evidence for H0"


def _jzs_bf10_from_t(t_stat: float, dof: float, n_eff: float, r_scale: float) -> float:
    """Compute exact JZS Bayes Factor BF10 via 1D Gaussian quadrature (Rouder et al., 2009)."""
    if dof < 1.0 or r_scale <= 0.0:
        raise StatError("Degrees of freedom must be >= 1 and r_scale must be > 0.")

    t2 = t_stat**2
    log_null = -0.5 * (dof + 1.0) * math.log1p(t2 / dof)

    r2 = r_scale**2

    def integrand(g: float) -> float:
        if g <= 0.0:
            return 0.0
        ng = n_eff * r2 * g
        log_term1 = -0.5 * math.log1p(ng)
        log_term2 = -0.5 * (dof + 1.0) * math.log1p(t2 / ((1.0 + ng) * dof))
        log_prior = -0.5 * math.log(2.0 * math.pi) - 1.5 * math.log(g) - 1.0 / (2.0 * g)
        log_val = (log_term1 + log_term2 + log_prior) - log_null
        if log_val > 700.0:
            return math.exp(700.0)
        if log_val < -700.0:
            return 0.0
        return math.exp(log_val)

    val, _ = integrate.quad(integrand, 0.0, np.inf, limit=200)
    return float(max(val, 1e-300))


def bayes_factor_t(
    a: np.ndarray | list[float],
    b: np.ndarray | list[float] | None = None,
    *,
    paired: bool = False,
    popmean: float = 0.0,
    r_scale: float = 0.7071067811865476,
) -> BayesFactorResult:
    """Compute Rouder's JZS Default Bayes Factor ($BF_{10}$ and $BF_{01}$) for one-sample, paired, or two-sample t-tests.

    Parameters
    ----------
    a : array-like
        First sample vector.
    b : array-like, optional
        Second sample vector (omit for one-sample test against `popmean`).
    paired : bool, default False
        Whether `a` and `b` are paired observations.
    popmean : float, default 0.0
        Null population mean for one-sample test.
    r_scale : float, default sqrt(2)/2 (~0.7071)
        Cauchy prior scale parameter $r$ on standardized effect size $\\delta$.
    """
    g1 = clean_1d(a, name="sample_a")
    if b is None:
        check_sample_size(g1, min_n=3, name="sample_a")
        check_variance(g1, name="sample_a")
        n = float(len(g1))
        dof = n - 1.0
        se = float(np.std(g1, ddof=1) / math.sqrt(n))
        t_stat = float((np.mean(g1) - popmean) / se)
        n_eff = n
    else:
        g2 = clean_1d(b, name="sample_b")
        if paired:
            check_paired(g1, g2, check_diff_var=True)
            diff = g1 - g2
            check_sample_size(diff, min_n=3, name="differences")
            n = float(len(diff))
            dof = n - 1.0
            se = float(np.std(diff, ddof=1) / math.sqrt(n))
            t_stat = float(np.mean(diff) / se)
            n_eff = n
        else:
            check_groups(g1, g2)
            check_sample_size(g1, min_n=3, name="sample_a")
            check_sample_size(g2, min_n=3, name="sample_b")
            check_variance(g1, name="sample_a")
            check_variance(g2, name="sample_b")
            n1, n2 = float(len(g1)), float(len(g2))
            dof = n1 + n2 - 2.0
            v1, v2 = float(np.var(g1, ddof=1)), float(np.var(g2, ddof=1))
            sp2 = ((n1 - 1.0) * v1 + (n2 - 1.0) * v2) / dof
            n_eff = (n1 * n2) / (n1 + n2)
            t_stat = float((np.mean(g1) - np.mean(g2)) / math.sqrt(sp2 / n_eff))

    bf10 = _jzs_bf10_from_t(t_stat, dof, n_eff, r_scale)
    bf01 = 1.0 / bf10
    return BayesFactorResult(
        bf10=bf10,
        bf01=bf01,
        t_stat=t_stat,
        dof=dof,
        r_scale=r_scale,
        evidence=interpret_bf(bf10),
    )
