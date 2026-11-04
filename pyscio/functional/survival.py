"""Stateless survival analysis kernels: Kaplan-Meier product-limit estimator and Mantel-Cox Log-Rank test."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import stats

from pyscio.core.errors import SampleSizeError, StatError
from pyscio.core.guards import clamp_p
from pyscio.core.types import LogRankResult, SurvivalCurveResult


def _validate_surv_arrays(
    time: np.ndarray | Sequence[float],
    event: np.ndarray | Sequence[int | float | bool],
    name: str = "survival",
) -> tuple[np.ndarray, np.ndarray]:
    t_arr = np.asarray(time, dtype=np.float64)
    e_arr = np.asarray(event, dtype=np.float64)
    if t_arr.ndim != 1 or e_arr.ndim != 1 or len(t_arr) != len(e_arr):
        raise StatError(f"{name}: `time` and `event` must be 1D arrays of equal length.")
    valid = np.isfinite(t_arr) & np.isfinite(e_arr)
    t_clean = t_arr[valid]
    e_clean = (e_arr[valid] > 0).astype(np.int64)
    if len(t_clean) < 2:
        raise SampleSizeError(
            f"{name}: requires at least 2 valid observations, got {len(t_clean)}.",
            n=len(t_clean),
            min_n=2,
        )
    if np.any(t_clean < 0):
        raise StatError(f"{name}: survival times must be non-negative.")
    return t_clean, e_clean


def kaplan_meier(
    time: np.ndarray | Sequence[float],
    event: np.ndarray | Sequence[int | float | bool],
    conf_level: float = 0.95,
) -> SurvivalCurveResult:
    """Compute the Kaplan-Meier product-limit survival curve with Kalbfleisch-Prentice log-log CIs.

    Parameters
    ----------
    time : array-like
        Observed follow-up times ($t_i \\ge 0$).
    event : array-like
        Event indicator ($1$ = event occurred, $0$ = right-censored).
    conf_level : float, default 0.95
        Confidence level for Greenwood log-log survival bounds.
    """
    t_arr, e_arr = _validate_surv_arrays(time, event, name="kaplan_meier")
    unique_times = np.unique(t_arr)

    times_out: list[float] = [0.0] if unique_times[0] > 0.0 else []
    surv_out: list[float] = [1.0] if unique_times[0] > 0.0 else []
    ci_lo_out: list[float] = [1.0] if unique_times[0] > 0.0 else []
    ci_hi_out: list[float] = [1.0] if unique_times[0] > 0.0 else []
    at_risk_out: list[int] = [len(t_arr)] if unique_times[0] > 0.0 else []
    events_out: list[int] = [0] if unique_times[0] > 0.0 else []

    s_cur = 1.0
    greenwood_sum = 0.0
    z_crit = float(stats.norm.ppf(1.0 - (1.0 - conf_level) / 2.0))

    for tj in unique_times:
        n_j = int(np.sum(t_arr >= tj))
        d_j = int(np.sum((t_arr == tj) & (e_arr == 1)))
        if d_j > 0 and n_j > 0:
            s_cur *= 1.0 - (d_j / n_j)
            if n_j > d_j:
                greenwood_sum += d_j / (n_j * (n_j - d_j))

        if s_cur <= 0.0:
            ci_l, ci_h = 0.0, 0.0
        elif s_cur >= 1.0 or greenwood_sum <= 0.0:
            ci_l, ci_h = 1.0, 1.0
        else:
            # Kalbfleisch-Prentice log(-log(S)) transform keeps CI strictly in [0, 1]
            log_s = np.log(s_cur)
            se_log_neg_log = np.sqrt(greenwood_sum) / abs(log_s)
            theta = np.exp(z_crit * se_log_neg_log)
            ci_l = float(np.clip(s_cur**theta, 0.0, 1.0))
            ci_h = float(np.clip(s_cur ** (1.0 / theta), 0.0, 1.0))

        times_out.append(float(tj))
        surv_out.append(float(s_cur))
        ci_lo_out.append(ci_l)
        ci_hi_out.append(ci_h)
        at_risk_out.append(n_j)
        events_out.append(d_j)

    surv_np = np.asarray(surv_out, dtype=np.float64)
    below_half = np.flatnonzero(surv_np <= 0.5)
    median_surv = float(times_out[int(below_half[0])]) if len(below_half) > 0 else float("nan")

    return SurvivalCurveResult(
        times=tuple(times_out),
        survival_prob=tuple(surv_out),
        ci_low=tuple(ci_lo_out),
        ci_high=tuple(ci_hi_out),
        n_at_risk=tuple(at_risk_out),
        n_events=tuple(events_out),
        median_survival=median_surv,
    )


def logrank_test(
    time_a: np.ndarray | Sequence[float],
    event_a: np.ndarray | Sequence[int | float | bool],
    time_b: np.ndarray | Sequence[float],
    event_b: np.ndarray | Sequence[int | float | bool],
) -> LogRankResult:
    """Compute the Mantel-Cox Log-Rank test comparing two right-censored survival distributions."""
    ta, ea = _validate_surv_arrays(time_a, event_a, name="group_a")
    tb, eb = _validate_surv_arrays(time_b, event_b, name="group_b")

    all_times = np.concatenate([ta, tb])
    all_events = np.concatenate([ea, eb])
    event_times = np.unique(all_times[all_events == 1])

    if len(event_times) == 0:
        return LogRankResult(
            stat=0.0,
            p=1.0,
            dof=1,
            observed=(0.0, 0.0),
            expected=(0.0, 0.0),
            hazard_ratio=1.0,
        )

    obs_a = float(np.sum(ea))
    obs_b = float(np.sum(eb))
    exp_a = 0.0
    exp_b = 0.0
    var_sum = 0.0

    for tj in event_times:
        n1_j = float(np.sum(ta >= tj))
        n2_j = float(np.sum(tb >= tj))
        n_j = n1_j + n2_j
        if n_j <= 0:
            continue
        d1_j = float(np.sum((ta == tj) & (ea == 1)))
        d2_j = float(np.sum((tb == tj) & (eb == 1)))
        d_j = d1_j + d2_j

        e1_j = d_j * (n1_j / n_j)
        e2_j = d_j * (n2_j / n_j)
        exp_a += e1_j
        exp_b += e2_j

        if n_j > 1.0:
            var_sum += (n1_j * n2_j * d_j * (n_j - d_j)) / (n_j**2 * (n_j - 1.0))

    chisq = float((obs_a - exp_a) ** 2 / var_sum) if var_sum > 1e-15 else 0.0
    p_raw = float(stats.chi2.sf(chisq, df=1)) if var_sum > 1e-15 else 1.0
    p_val, _ = clamp_p(p_raw)

    hr = (
        float((obs_a / exp_a) / (obs_b / exp_b))
        if (exp_a > 0 and exp_b > 0 and obs_b > 0)
        else float("nan")
    )

    return LogRankResult(
        stat=chisq,
        p=p_val,
        dof=1,
        observed=(obs_a, obs_b),
        expected=(exp_a, exp_b),
        hazard_ratio=hr,
    )
