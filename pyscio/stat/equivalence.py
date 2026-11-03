"""High-level TOST equivalence wrapper returning a rich TestResult container."""

from __future__ import annotations

import numpy as np

from pyscio.core.guards import clamp_p, clean_1d
from pyscio.core.results import TestResult
from pyscio.functional.effects import cohen_d, cohen_d_ci, interpret_effect
from pyscio.functional.equivalence import tost_kernel


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
    """Two One-Sided Tests (TOST) for statistical equivalence within bounds [low, high]."""
    k_res = tost_kernel(
        a, b=b, low=low, high=high, bound=bound, alpha=alpha, paired=paired, popmean=popmean
    )
    g1 = clean_1d(a, name="sample_a")
    conf_2alpha = max(0.5, 1.0 - 2.0 * alpha)

    if b is None:
        sd = float(np.std(g1, ddof=1))
        eff = float((np.mean(g1) - popmean) / sd) if sd > 0 else 0.0
        eff_ci = cohen_d_ci(g1, conf_level=1.0 - alpha, popmean=popmean)
        x_label = "sample"
    elif paired:
        g2 = clean_1d(b, name="sample_b")
        d_arr = g1 - g2
        sd_d = float(np.std(d_arr, ddof=1))
        eff = float(np.mean(d_arr) / sd_d) if sd_d > 0 else 0.0
        eff_ci = cohen_d_ci(g1, g2, conf_level=1.0 - alpha, paired=True)
        x_label = "paired_a vs paired_b"
    else:
        g2 = clean_1d(b, name="sample_b")
        eff = cohen_d(g1, g2)
        eff_ci = cohen_d_ci(g1, g2, conf_level=1.0 - alpha)
        x_label = "group_a vs group_b"

    p_val, p_display = clamp_p(k_res.p)
    return TestResult(
        test_name="TOST Equivalence Test",
        x_name=x_label,
        y_name=f"bounds=[{k_res.low}, {k_res.high}]",
        stat=k_res.stat,
        p=p_val,
        p_display=p_display,
        dof=k_res.dof,
        ci=k_res.ci,
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
