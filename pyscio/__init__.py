"""pyscio: Autonomous Statistical Inference Engine & One-Tap EDA.

Canonical imports:
    import pyscio as sc
    import pyscio.functional as F
"""

from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

from pyscio.accessor import register_accessor

if TYPE_CHECKING:
    from pyscio import core, eda, functional, models, stat
    from pyscio.eda import read
    from pyscio.stat import (
        anova_table,
        compare,
        logistic,
        ols,
        poisson,
        screen,
        solve_power,
        tost,
    )

    F = functional

try:
    __version__ = version("pyscio")
except PackageNotFoundError:
    __version__ = "0.2.0.dev0"

# Register the DataFrame accessor: df.scio
register_accessor()

__all__ = [
    "F",
    "__version__",
    "anova_table",
    "compare",
    "core",
    "eda",
    "functional",
    "logistic",
    "models",
    "ols",
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
        mod = importlib.import_module("pyscio.functional")
        globals()[name] = mod
        return mod
    if name in ("core", "models", "eda", "stat"):
        mod = importlib.import_module(f"pyscio.{name}")
        globals()[name] = mod
        return mod
    if name == "read":
        eda_mod = importlib.import_module("pyscio.eda")
        val = eda_mod.read
        globals()[name] = val
        return val
    if name in (
        "anova_table",
        "compare",
        "screen",
        "ols",
        "logistic",
        "poisson",
        "tost",
        "solve_power",
    ):
        stat_mod = importlib.import_module("pyscio.stat")
        val = getattr(stat_mod, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module 'pyscio' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(__all__)
