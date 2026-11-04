"""Statistical power and sample size calculations for t-tests and One-Way ANOVA."""

from __future__ import annotations

from omnistats.functional.adjust import power_anova, power_t, solve_power_t

solve_power = solve_power_t

__all__ = ["power_anova", "power_t", "solve_power", "solve_power_t"]
