"""CLI entry point for pyscio."""

from __future__ import annotations

__all__ = ["main"]


def main() -> int:
    """CLI runner entry point."""
    from rich.console import Console

    console = Console()
    console.print(
        "[bold cyan]pyscio[/bold cyan]: Autonomous Statistical Inference Engine & One-Tap EDA"
    )
    console.print(
        "Run [green]python -m pyscio --help[/green] or import in your Python environment."
    )
    return 0
