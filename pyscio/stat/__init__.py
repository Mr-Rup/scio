"""Autonomous statistical decision lattice, hypothesis testing router, and discovery screening."""

from __future__ import annotations

from pyscio.stat.compare import compare
from pyscio.stat.power import power_anova, power_t
from pyscio.stat.regression import logistic, ols, parse_formula
from pyscio.stat.router import (
    evaluate_normality_with_clt,
    route_contingency,
    route_independent_two_sample,
    route_k_sample,
    route_one_sample,
    route_paired_two_sample,
)
from pyscio.stat.screen import screen

__all__ = [
    "compare",
    "screen",
    "ols",
    "logistic",
    "parse_formula",
    "power_t",
    "power_anova",
    "route_independent_two_sample",
    "route_paired_two_sample",
    "route_one_sample",
    "route_k_sample",
    "route_contingency",
    "evaluate_normality_with_clt",
]
