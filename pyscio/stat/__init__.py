"""Autonomous statistical decision lattice, hypothesis testing router, and discovery screening."""

from __future__ import annotations

from typing import Any

__all__ = ["compare", "screen"]


def compare(data: Any, x: str, y: str, **kwargs: Any) -> Any:
    """Autonomous hypothesis testing comparing target variable `y` across groups `x`."""
    raise NotImplementedError("pyscio.stat.compare is implemented in Phase 5.")


def screen(data: Any, y: str, **kwargs: Any) -> Any:
    """Multi-feature discovery screening against target outcome `y` with FDR control."""
    raise NotImplementedError("pyscio.stat.screen is implemented in Phase 5.")
