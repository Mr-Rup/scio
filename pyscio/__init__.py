"""pyscio: Backwards-compatibility shim re-exporting omnistats."""

from __future__ import annotations

import sys
import warnings

import omnistats

warnings.warn(
    "The 'pyscio' package has been renamed to 'omnistats'. "
    "Please update your imports: 'import omnistats as os' (or 'import omnistats as omni').",
    UserWarning,
    stacklevel=2,
)

# Mirror all attributes from omnistats
sys.modules["pyscio"] = omnistats

__version__ = omnistats.__version__
F = omnistats.F
__all__ = omnistats.__all__


def __getattr__(name: str) -> object:
    return getattr(omnistats, name)
