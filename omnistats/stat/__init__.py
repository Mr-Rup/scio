"""Autonomous statistical decision lattice, hypothesis testing router, and discovery screening."""

from __future__ import annotations

from omnistats.models import (
    anova_table,
    ate,
    covariate_balance,
    coxph,
    logistic,
    mixedlm,
    ols,
    parse_formula,
    poisson,
)
from omnistats.stat.compare import compare
from omnistats.stat.equivalence import tost
from omnistats.stat.power import power_anova, power_t, solve_power, solve_power_t
from omnistats.stat.router import (
    evaluate_normality_with_clt,
    route_contingency,
    route_correlation,
    route_independent_two_sample,
    route_k_sample,
    route_one_sample,
    route_paired_k_sample,
    route_paired_two_sample,
)
from omnistats.stat.screen import screen

__all__ = [
    "anova_table",
    "ate",
    "compare",
    "covariate_balance",
    "coxph",
    "evaluate_normality_with_clt",
    "logistic",
    "mixedlm",
    "ols",
    "parse_formula",
    "poisson",
    "power_anova",
    "power_t",
    "route_contingency",
    "route_correlation",
    "route_independent_two_sample",
    "route_k_sample",
    "route_one_sample",
    "route_paired_k_sample",
    "route_paired_two_sample",
    "screen",
    "solve_power",
    "solve_power_t",
    "tost",
]
