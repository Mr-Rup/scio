"""Pure Two One-Sided Tests (TOST) equivalence kernel returning lightweight TOSTResult."""

from __future__ import annotations

import numpy as np
from scipy import stats

from pyscio.core.errors import StatError
from pyscio.core.guards import clean_1d
from pyscio.core.types import TOSTResult
from pyscio.functional.tests import one_sample_t, paired_t, welch_t


def tost_kernel(
    a: np.ndarray | list[float],
    b: np.ndarray | list[float] | None = None,
    low: float = -0.5,
    high: float = 0.5,
    bound: float | None = None,
    alpha: float = 0.05,
    paired: bool = False,
    popmean: float = 0.0,
) -> TOSTResult:
    """Two One-Sided Tests (TOST) kernel for statistical equivalence within [low, high].

    Returns a pure `TOSTResult` NamedTuple without importing `pyscio.core.results`.
    """
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
    elif paired:
        g2 = clean_1d(b, name="sample_b")
        res_base = paired_t(g1, g2, conf_level=conf_2alpha)
    else:
        g2 = clean_1d(b, name="sample_b")
        res_base = welch_t(g1, g2, conf_level=conf_2alpha)

    diff = res_base.mean_diff
    se = res_base.se_diff
    dof = res_base.dof
    ci_90 = res_base.ci

    t_lower = (diff - low) / se if se > 0 else 0.0
    p_lower = float(stats.t.sf(t_lower, df=dof))

    t_upper = (diff - high) / se if se > 0 else 0.0
    p_upper = float(stats.t.cdf(t_upper, df=dof))

    p_tost = max(p_lower, p_upper)
    t_tost = t_lower if p_lower >= p_upper else t_upper

    return TOSTResult(
        stat=t_tost,
        p=p_tost,
        dof=dof,
        ci=ci_90,
        mean_diff=diff,
        se_diff=se,
        low=low,
        high=high,
    )
