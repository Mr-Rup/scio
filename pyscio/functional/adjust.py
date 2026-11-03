"""Multiple testing corrections (Benjamini-Hochberg FDR, Bonferroni, Holm) and power analysis."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import optimize, stats

from pyscio.core.errors import StatError


def fdr_bh(
    p_values: Sequence[float] | np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg False Discovery Rate (FDR) procedure."""
    p = np.asarray(p_values, dtype=np.float64)
    m = len(p)
    if m == 0:
        return np.array([], dtype=bool), np.array([], dtype=np.float64)
    if m == 1:
        return np.array([p[0] <= alpha], dtype=bool), np.clip(p, 0.0, 1.0)

    sort_idx = np.argsort(p)
    sorted_p = p[sort_idx]

    ranks = np.arange(1, m + 1, dtype=np.float64)
    raw_q = sorted_p * (m / ranks)

    q_mono = np.minimum.accumulate(raw_q[::-1])[::-1]
    q_clamped = np.clip(q_mono, 0.0, 1.0)

    orig_idx = np.argsort(sort_idx)
    q_values = q_clamped[orig_idx]
    reject = q_values <= alpha

    return reject, q_values


def bonferroni(
    p_values: Sequence[float] | np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Bonferroni Family-Wise Error Rate (FWER) single-step correction."""
    p = np.asarray(p_values, dtype=np.float64)
    m = len(p)
    if m == 0:
        return np.array([], dtype=bool), np.array([], dtype=np.float64)

    p_adj = np.clip(p * m, 0.0, 1.0)
    reject = p_adj <= alpha
    return reject, p_adj


def holm(
    p_values: Sequence[float] | np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Holm-Bonferroni step-down FWER procedure."""
    p = np.asarray(p_values, dtype=np.float64)
    m = len(p)
    if m == 0:
        return np.array([], dtype=bool), np.array([], dtype=np.float64)
    if m == 1:
        return np.array([p[0] <= alpha], dtype=bool), np.clip(p, 0.0, 1.0)

    sort_idx = np.argsort(p)
    sorted_p = p[sort_idx]

    multipliers = np.arange(m, 0, -1, dtype=np.float64)
    raw_adj = sorted_p * multipliers

    adj_mono = np.maximum.accumulate(raw_adj)
    adj_clamped = np.clip(adj_mono, 0.0, 1.0)

    orig_idx = np.argsort(sort_idx)
    p_adj = adj_clamped[orig_idx]
    reject = p_adj <= alpha

    return reject, p_adj


def power_t(n: int | float, d: float, alpha: float = 0.05) -> float:
    """Compute statistical power for a two-sample t-test with equal group sizes $n$."""
    if n < 2 or abs(d) == 0.0:
        return alpha

    dof = 2.0 * float(n) - 2.0
    delta = abs(d) * np.sqrt(float(n) / 2.0)
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df=dof))

    if delta - t_crit > 35.0:
        return 1.0

    p_upper = float(stats.nct.sf(t_crit, df=dof, nc=delta))
    p_lower = float(stats.nct.cdf(-t_crit, df=dof, nc=delta))
    power = p_upper + p_lower
    if np.isnan(power):
        power = float(stats.norm.sf(t_crit - delta) + stats.norm.cdf(-t_crit - delta))
    return float(np.clip(power, 0.0, 1.0))


def power_anova(k: int, n_per_group: int | float, eta_squared: float, alpha: float = 0.05) -> float:
    """Compute statistical power for One-Way ANOVA across $k$ groups of size $n$."""
    if k < 2 or n_per_group < 2 or eta_squared <= 0.0:
        return alpha

    eta_clamped = min(eta_squared, 0.999)
    f_stat = np.sqrt(eta_clamped / (1.0 - eta_clamped))

    total_n = float(k) * float(n_per_group)
    df_between = float(k - 1)
    df_within = total_n - float(k)

    nc_lambda = total_n * (f_stat**2)
    f_crit = stats.f.ppf(1.0 - alpha, dfn=df_between, dfd=df_within)

    power = float(1.0 - stats.ncf.cdf(f_crit, dfn=df_between, dfd=df_within, nc=nc_lambda))
    return float(np.clip(power, 0.0, 1.0))


def solve_power_t(
    *,
    effect: float | None = None,
    n: int | float | None = None,
    power: float | None = None,
    alpha: float = 0.05,
) -> float:
    """Bidirectional solver for two-sample t-test power, sample size per group $n$, or Cohen's $d$.

    Leave exactly one of `effect`, `n`, or `power` as None to solve for that quantity.
    """
    none_count = sum(x is None for x in (effect, n, power))
    if none_count != 1:
        raise StatError("Exactly one of `effect`, `n`, or `power` must be None in solve_power_t.")

    if power is None:
        assert n is not None and effect is not None
        return power_t(n, effect, alpha=alpha)

    if not (alpha < power < 1.0):
        raise StatError(
            f"Target power ({power}) must lie strictly between alpha ({alpha}) and 1.0."
        )

    if n is None:
        assert effect is not None
        if abs(effect) <= 1e-9:
            raise StatError("Effect size d must be non-zero to solve for sample size n.")
        res_n = optimize.brentq(lambda x: power_t(x, effect, alpha=alpha) - power, 2.001, 1e7)
        return float(np.ceil(float(res_n)))

    assert n is not None
    if n < 2:
        raise StatError("Sample size n per group must be >= 2 to solve for effect size.")
    res_d = optimize.brentq(lambda d_val: power_t(n, d_val, alpha=alpha) - power, 1e-6, 20.0)
    return float(res_d)
