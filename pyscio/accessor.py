"""Re-export Pandas DataFrame accessor from pyscio.integrations.pandas."""

from __future__ import annotations

from pyscio.integrations.pandas import ScioAccessor, register_accessor

__all__ = ["ScioAccessor", "register_accessor"]
