"""Autonomous statistical decision lattice, hypothesis testing router, and discovery screening."""

from __future__ import annotations

from pyscio.models import logistic, ols, parse_formula
from pyscio.stat.compare import compare
from pyscio.stat.equivalence import tost
from pyscio.stat.power import power_anova, power_t, solve_power, solve_power_t
from pyscio.stat.router import (
    evaluate_normality_with_clt,
    route_contingency,
    route_correlation,
    route_independent_two_sample,
    route_k_sample,
    route_one_sample,
    route_paired_k_sample,
    route_paired_two_sample,
)
from pyscio.stat.screen import screen

__all__ = [
    "compare",
    "evaluate_normality_with_clt",
    "logistic",
    "ols",
    "parse_formula",
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
