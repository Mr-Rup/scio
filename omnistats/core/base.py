"""Abstract base class for all omnistats result containers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd
    from rich.console import Console, ConsoleOptions, RenderResult


class BaseResult(ABC):
    """Abstract base class providing standard exports and multi-format rendering for results."""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Convert the result into a clean, JSON-serializable dictionary."""
        raise NotImplementedError

    @abstractmethod
    def to_df(self) -> pd.DataFrame:
        """Convert the result into a structured Pandas DataFrame."""
        raise NotImplementedError

    @abstractmethod
    def summary(self, format: str = "rich") -> str:
        """Return a formatted string representation of the result.

        Parameters
        ----------
        format : str
            One of 'rich', 'markdown', 'latex', 'text'.
        """
        raise NotImplementedError

    @abstractmethod
    def _repr_html_(self) -> str:
        """Interactive rich HTML card representation for Jupyter and notebook environments."""
        raise NotImplementedError

    @abstractmethod
    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        """Rich console protocol method for automatic colorful terminal printing."""
        raise NotImplementedError

    def __str__(self) -> str:
        """Return human-readable plain text summary."""
        return self.summary(format="text")

    def __repr__(self) -> str:
        """Return developer-friendly string representation."""
        cls_name = self.__class__.__name__
        attrs = ", ".join(f"{k}={v!r}" for k, v in list(self.to_dict().items())[:4])
        return f"{cls_name}({attrs}, ...)"
