"""Resampling, Monte Carlo / Exact Permutation, BCa Bootstrap, and TOST Equivalence Kernels."""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy import stats

from pyscio.core.guards import check_groups, check_paired, check_sample_size, clamp_p, clean_1d
from pyscio.core.results import TestResult
from pyscio.functional.effects import cohen_d, cohen_d_ci, interpret_effect
from pyscio.functional.tests import NonParametricResult, one_sample_t, paired_t, welch_t


def permutation_test(
    a: np.ndarray | list[float],
    b: np.ndarray | list[float],
    n_resamples: int = 10000,
    paired: bool = False,
    alternative: str = "two-sided",
    seed: int | None = None,
) -> NonParametricResult:
    """Perform an exact or Monte Carlo permutation test for difference in means.

    If the total number of exact combinations (or $2^N$ sign flips for paired samples)
    is $\\le n_{resamples}$, computes the exact permutation p-value; otherwise uses
    vectorized Monte Carlo resampling with Phipson-Smyth (+1) continuity adjustment.
    """
    g1 = clean_1d(a, name="sample_a")
    g2 = clean_1d(b, name="sample_b")
    rng = np.random.default_rng(seed)

    if paired:
        check_paired(g1, g2, check_diff_var=False)
        diffs = g1 - g2
        n = len(diffs)
        obs_stat = float(np.mean(diffs))

        if 2**n <= n_resamples and n <= 16:
            # Exact sign-flip enumeration
            signs = np.array(list(itertools.product([-1.0, 1.0], repeat=n)), dtype=np.float64)
            perm_stats = (signs @ diffs) / n
            exact = True
        else:
            signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float64), size=(n_resamples, n))
            perm_stats = (signs @ diffs) / n
            exact = False
    else:
        check_groups(g1, g2)
        n1, n2 = len(g1), len(g2)
        n_total = n1 + n2
        pooled = np.concatenate([g1, g2])
        obs_stat = float(np.mean(g1) - np.mean(g2))

        n_comb = math.comb(n_total, n1) if n_total <= 24 else n_resamples + 1
        if n_comb <= n_resamples:
            # Exact combination enumeration
            total_sum = float(np.sum(pooled))
            comb_indices = np.fromiter(
                itertools.chain.from_iterable(itertools.combinations(range(n_total), n1)),
                dtype=np.intp,
                count=n_comb * n1,
            ).reshape(n_comb, n1)
            sum1 = np.sum(pooled[comb_indices], axis=1)
            sum2 = total_sum - sum1
            perm_stats = (sum1 / n1) - (sum2 / n2)
            exact = True
        else:
            # Fast vectorized Monte Carlo permutation
            total_sum = float(np.sum(pooled))
            # Process in chunks of 2500 to keep peak memory small
            chunk_size = min(2500, n_resamples)
            perm_list: list[np.ndarray] = []
            remaining = n_resamples
            while remaining > 0:
                cur = min(chunk_size, remaining)
                idx = rng.permuted(np.broadcast_to(pooled, (cur, n_total)), axis=1)[:, :n1]
                s1 = np.sum(idx, axis=1)
                s2 = total_sum - s1
                perm_list.append((s1 / n1) - (s2 / n2))
                remaining -= cur
            perm_stats = np.concatenate(perm_list)
            exact = False

    if alternative == "two-sided":
        exceed = int(np.sum(np.abs(perm_stats) >= abs(obs_stat) - 1e-15))
    elif alternative == "greater":
        exceed = int(np.sum(perm_stats >= obs_stat - 1e-15))
    else:
        exceed = int(np.sum(perm_stats <= obs_stat + 1e-15))

    m = len(perm_stats)
    p = float(exceed / m) if exact else float((exceed + 1.0) / (m + 1.0))
    return NonParametricResult(stat=obs_stat, p=min(1.0, p))


def bootstrap_ci(
    data: np.ndarray | list[float],
    stat_func: Callable[[np.ndarray], Any] = np.mean,
    n_boot: int = 5000,
    conf_level: float = 0.95,
    method: str = "bca",
    seed: int | None = None,
) -> tuple[float, float]:
    """Compute Bias-Corrected and Accelerated (BCa) or percentile bootstrap confidence interval."""
    arr = clean_1d(data, name="data")
    check_sample_size(arr, min_n=3, name="data")
    n = len(arr)
    rng = np.random.default_rng(seed)

    theta_hat = float(stat_func(arr))

    # Generate bootstrap resamples
    boot_indices = rng.integers(0, n, size=(n_boot, n))
    boot_samples = arr[boot_indices]

    if stat_func is np.mean:
        boot_stats = np.mean(boot_samples, axis=1)
    elif stat_func is np.median:
        boot_stats = np.median(boot_samples, axis=1)
    else:
        boot_stats = np.array([float(stat_func(row)) for row in boot_samples], dtype=np.float64)

    alpha = 1.0 - conf_level
    if method != "bca":
        low, high = np.percentile(boot_stats, [100.0 * (alpha / 2.0), 100.0 * (1.0 - alpha / 2.0)])
        return (float(low), float(high))

    # BCa bias-correction parameter z0
    prop_less = float(np.mean(boot_stats < theta_hat))
    prop_less = float(np.clip(prop_less, 1.0 / (n_boot + 1.0), n_boot / (n_boot + 1.0)))
    z0 = float(stats.norm.ppf(prop_less))

    # Jackknife acceleration parameter a_hat
    jack_stats = np.empty(n, dtype=np.float64)
    for i in range(n):
        jack_slice = np.delete(arr, i)
        jack_stats[i] = float(stat_func(jack_slice))

    jack_mean = float(np.mean(jack_stats))
    diff_jack = jack_mean - jack_stats
    num = float(np.sum(diff_jack**3))
    den = float(6.0 * (np.sum(diff_jack**2) ** 1.5))
    a_hat = num / den if den > 1e-15 else 0.0

    z_low = float(stats.norm.ppf(alpha / 2.0))
    z_high = float(stats.norm.ppf(1.0 - alpha / 2.0))

    def bca_quantile(z_alpha: float) -> float:
        numer = z0 + z_alpha
        denom = 1.0 - a_hat * numer
        if abs(denom) < 1e-12:
            return 0.5
        adj_z = z0 + numer / denom
        return float(np.clip(stats.norm.cdf(adj_z), 0.0, 1.0))

    q_low = bca_quantile(z_low) * 100.0
    q_high = bca_quantile(z_high) * 100.0
    low, high = np.percentile(boot_stats, [min(q_low, q_high), max(q_low, q_high)])
    return (float(low), float(high))


def tost(
    a: np.ndarray | list[float],
    b: np.ndarray | list[float] | None = None,
    low: float = -0.5,
    high: float = 0.5,
    bound: float | None = None,
    alpha: float = 0.05,
    paired: bool = False,
    popmean: float = 0.0,
) -> TestResult:
    """Two One-Sided Tests (TOST) for statistical equivalence within bounds [low, high].

    Tests the composite null hypothesis $H_0: \\Delta \\le \\text{low} \\lor \\Delta \\ge \\text{high}$
    against the equivalence alternative $H_1: \\text{low} < \\Delta < \\text{high}$.
    """
    from pyscio.core.errors import StatError

    if bound is not None:
        if bound <= 0:
            raise StatError(f"Equivalence bound must be strictly positive, got {bound}.")
        low = -float(bound)
        high = float(bound)
    if low >= high:
        raise StatError(
            f"Equivalence lower bound ({low}) must be strictly less than upper bound ({high})."
        )

    g1 = clean_1d(a, name="sample_a")
    conf_2alpha = max(0.5, 1.0 - 2.0 * alpha)

    if b is None:
        res_base = one_sample_t(g1, popmean=popmean, conf_level=conf_2alpha)
        diff = res_base.mean_diff
        se = res_base.se_diff
        dof = res_base.dof
        ci_90 = res_base.ci
        eff = float((np.mean(g1) - popmean) / np.std(g1, ddof=1))
        eff_ci = cohen_d_ci(g1, conf_level=1.0 - alpha, popmean=popmean)
        x_label, y_label = "sample", f"bounds=[{low}, {high}]"
    elif paired:
        g2 = clean_1d(b, name="sample_b")
        res_base = paired_t(g1, g2, conf_level=conf_2alpha)
        diff = res_base.mean_diff
        se = res_base.se_diff
        dof = res_base.dof
        ci_90 = res_base.ci
        d_arr = g1 - g2
        eff = float(np.mean(d_arr) / np.std(d_arr, ddof=1))
        eff_ci = cohen_d_ci(g1, g2, conf_level=1.0 - alpha, paired=True)
        x_label, y_label = "paired_a vs paired_b", f"bounds=[{low}, {high}]"
    else:
        g2 = clean_1d(b, name="sample_b")
        res_base = welch_t(g1, g2, conf_level=conf_2alpha)
        diff = res_base.mean_diff
        se = res_base.se_diff
        dof = res_base.dof
        ci_90 = res_base.ci
        eff = cohen_d(g1, g2)
        eff_ci = cohen_d_ci(g1, g2, conf_level=1.0 - alpha)
        x_label, y_label = "group_a vs group_b", f"bounds=[{low}, {high}]"

    # Lower bound one-sided t-test (H1: diff > low)
    t_lower = (diff - low) / se if se > 0 else 0.0
    p_lower = float(stats.t.sf(t_lower, df=dof))

    # Upper bound one-sided t-test (H1: diff < high)
    t_upper = (diff - high) / se if se > 0 else 0.0
    p_upper = float(stats.t.cdf(t_upper, df=dof))

    # Overall TOST p-value is max(p_lower, p_upper)
    p_tost_raw = max(p_lower, p_upper)
    t_tost = t_lower if p_lower >= p_upper else t_upper
    p_val, p_display = clamp_p(p_tost_raw)

    return TestResult(
        test_name="TOST Equivalence Test",
        x_name=x_label,
        y_name=y_label,
        stat=t_tost,
        p=p_val,
        p_display=p_display,
        dof=dof,
        ci=ci_90,
        ci_level=conf_2alpha,
        sig=p_val < alpha,
        alpha=alpha,
        effect=eff,
        metric="Cohen's d",
        magnitude=interpret_effect("cohen", eff),
        effect_ci=eff_ci,
        checks=[],
        posthoc=None,
        warns=[],
    )
