"""Autonomous statistical decision lattice router subpackage."""

from __future__ import annotations

from omnistats.stat.router.k_sample import (
    route_contingency,
    route_k_sample,
    route_paired_k_sample,
)
from omnistats.stat.router.normality import evaluate_normality_with_clt
from omnistats.stat.router.two_sample import (
    route_correlation,
    route_independent_two_sample,
    route_one_sample,
    route_paired_two_sample,
)

__all__ = [
    "evaluate_normality_with_clt",
    "route_contingency",
    "route_correlation",
    "route_independent_two_sample",
    "route_k_sample",
    "route_one_sample",
    "route_paired_k_sample",
    "route_paired_two_sample",
]
