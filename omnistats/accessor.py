"""Re-export Pandas DataFrame accessor from omnistats.integrations.pandas."""

from __future__ import annotations

from omnistats.integrations.pandas import ScioAccessor, register_accessor

__all__ = ["ScioAccessor", "register_accessor"]
