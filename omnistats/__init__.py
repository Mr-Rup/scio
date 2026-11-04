"""omnistats: Autonomous Statistical Inference Engine & One-Tap EDA.

Canonical imports:
    import omnistats as sc
    import omnistats.functional as F
"""

from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

from omnistats.accessor import register_accessor

if TYPE_CHECKING:
    from omnistats import core, eda, functional, integrations, models, stat
    from omnistats.eda import read
    from omnistats.integrations.plotting import plot_raincloud
    from omnistats.stat import (
        anova_table,
        ate,
        compare,
        covariate_balance,
        coxph,
        logistic,
        mixedlm,
        ols,
        poisson,
        screen,
        solve_power,
        tost,
    )

    F = functional

try:
    __version__ = version("omnistats")
except PackageNotFoundError:
    __version__ = "0.3.0"

# Register the DataFrame accessor: df.scio
register_accessor()

__all__ = [
    "F",
    "__version__",
    "anova_table",
    "ate",
    "compare",
    "core",
    "covariate_balance",
    "coxph",
    "eda",
    "functional",
    "integrations",
    "logistic",
    "mixedlm",
    "models",
    "ols",
    "plot_raincloud",
    "poisson",
    "read",
    "screen",
    "solve_power",
    "stat",
    "tost",
]


def __getattr__(name: str) -> Any:
    """Lazy PEP 562 module attribute loader with globals() caching for zero repeat overhead."""
    if name in ("F", "functional"):
        mod = importlib.import_module("omnistats.functional")
        globals()[name] = mod
        return mod
    if name in ("core", "models", "eda", "stat", "integrations"):
        mod = importlib.import_module(f"omnistats.{name}")
        globals()[name] = mod
        return mod
    if name == "read":
        eda_mod = importlib.import_module("omnistats.eda")
        val = eda_mod.read
        globals()[name] = val
        return val
    if name == "plot_raincloud":
        plot_mod = importlib.import_module("omnistats.integrations.plotting")
        val = plot_mod.plot_raincloud
        globals()[name] = val
        return val
    if name in (
        "anova_table",
        "ate",
        "compare",
        "covariate_balance",
        "coxph",
        "screen",
        "ols",
        "logistic",
        "mixedlm",
        "poisson",
        "tost",
        "solve_power",
    ):
        stat_mod = importlib.import_module("omnistats.stat")
        val = getattr(stat_mod, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module 'omnistats' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(__all__)
