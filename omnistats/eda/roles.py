"""Re-export semantic column classification from omnistats.core.roles."""

from __future__ import annotations

from omnistats.core.roles import Role, classify, classify_series

__all__ = ["Role", "classify", "classify_series"]
