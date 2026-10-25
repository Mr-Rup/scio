"""pyscio: Autonomous Statistical Inference Engine & One-Tap EDA.

Canonical imports:
    import pyscio as sc
    import pyscio.functional as F
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from pyscio import eda, functional, stat
from pyscio.accessor import register_accessor

# Pure mathematical functional kernels alias
F = functional

try:
    __version__ = version("pyscio")
except PackageNotFoundError:
    __version__ = "0.1.0.dev0"

# Register the DataFrame accessor: df.scio
register_accessor()

# Top-level aliases
read = eda.read
compare = stat.compare
screen = stat.screen

__all__ = [
    "__version__",
    "F",
    "functional",
    "eda",
    "stat",
    "read",
    "compare",
    "screen",
]
