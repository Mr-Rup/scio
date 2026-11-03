"""pyscio: Autonomous Statistical Inference Engine & One-Tap EDA.

Canonical imports:
    import pyscio as sc
    import pyscio.functional as F
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Any

from pyscio.accessor import register_accessor

if TYPE_CHECKING:
    from pyscio import core, eda, functional, models, stat
    from pyscio.eda import read
    from pyscio.stat import compare, logistic, ols, screen, tost

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
    "compare",
    "core",
    "eda",
    "functional",
    "logistic",
    "models",
    "ols",
    "read",
    "screen",
    "stat",
    "tost",
]


def __getattr__(name: str) -> Any:
    """Lazy PEP 562 module attribute loader for fast package startup."""
    if name in ("F", "functional"):
        from pyscio import functional

        return functional
    if name == "core":
        from pyscio import core

        return core
    if name == "models":
        from pyscio import models

        return models
    if name == "eda":
        from pyscio import eda

        return eda
    if name == "stat":
        from pyscio import stat

        return stat
    if name == "read":
        from pyscio.eda import read

        return read
    if name in ("compare", "screen", "ols", "logistic", "tost"):
        from pyscio import stat

        return getattr(stat, name)
    raise AttributeError(f"module 'pyscio' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(__all__)
