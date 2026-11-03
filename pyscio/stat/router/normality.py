"""Normality evaluation with Large-N Central Limit Theorem (CLT) and Pathology guards."""

from __future__ import annotations

import numpy as np

from pyscio.core.constants import CLT_MAX_KURT, CLT_MAX_SKEW, CLT_MIN_N
from pyscio.core.results import CheckResult
from pyscio.functional.checks import check_normality, check_pathology


def evaluate_normality_with_clt(
    arr: np.ndarray,
    name: str = "sample",
    alpha: float = 0.05,
) -> tuple[bool, CheckResult]:
    """Check normality with the Large-N Central Limit Theorem (CLT) and Pathology guards."""
    n_res = check_normality(arr, alpha=alpha)
    n = len(arr)
    skew = abs(n_res.skewness)
    kurt = abs(n_res.excess_kurtosis)

    has_path, path_reasons = check_pathology(
        arr, skewness=n_res.skewness, excess_kurtosis=n_res.excess_kurtosis
    )
    clt_holds = n >= CLT_MIN_N and skew < CLT_MAX_SKEW and kurt < CLT_MAX_KURT and not has_path

    if n_res.is_normal and not has_path:
        passed = True
        details = f"{n_res.test_used}: p={n_res.p:.4g} >= {alpha}"
        clt_note = None
    elif clt_holds:
        passed = True
        details = (
            f"{n_res.test_used} rejected (p={n_res.p:.4g}), but N={n:,} >= {CLT_MIN_N}, "
            f"skewness={n_res.skewness:.2f}, kurtosis={n_res.excess_kurtosis:.2f}"
        )
        clt_note = (
            "Central Limit Theorem holds (N >= 500, moderate skew/kurtosis); parametric test robust"
        )
    else:
        passed = False
        path_suffix = f"; pathology: {', '.join(path_reasons)}" if has_path else ""
        details = (
            f"{n_res.test_used} rejected (p={n_res.p:.4g} < {alpha}, "
            f"skewness={n_res.skewness:.2f}, kurtosis={n_res.excess_kurtosis:.2f}{path_suffix})"
        )
        clt_note = None

    check = CheckResult(
        name=f"Normality ({name})",
        stat=n_res.stat,
        p=n_res.p,
        passed=passed,
        details=details,
        clt_note=clt_note,
    )
    return passed, check
