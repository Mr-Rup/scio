"""CLI entry point for pyscio."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console

import pyscio as sc

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    """CLI runner entry point."""
    console = Console()
    parser = argparse.ArgumentParser(
        prog="pyscio",
        description="pyscio: Autonomous Statistical Inference Engine & Universal EDA",
    )
    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version=f"%(prog)s {sc.__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Subcommand to execute")

    # Subcommand: profile
    profile_parser = subparsers.add_parser(
        "profile",
        help="Generate comprehensive exploratory data analysis report from a file",
    )
    profile_parser.add_argument(
        "file", type=str, help="Path to dataset (CSV, TSV, Parquet, Excel, JSON)"
    )
    profile_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional path to export standalone interactive HTML report (e.g. report.html)",
    )
    profile_parser.add_argument(
        "--title",
        "-t",
        type=str,
        default="pyscio EDA Report",
        help="Title for the report",
    )

    # Subcommand: screen
    screen_parser = subparsers.add_parser(
        "screen",
        help="Screen multiple features against a target outcome with FDR control",
    )
    screen_parser.add_argument(
        "file", type=str, help="Path to dataset (CSV, TSV, Parquet, Excel, JSON)"
    )
    screen_parser.add_argument(
        "--target",
        "-y",
        type=str,
        required=True,
        help="Target dependent variable name to screen against",
    )
    screen_parser.add_argument(
        "--alpha",
        "-a",
        type=float,
        default=0.05,
        help="False Discovery Rate threshold (default: 0.05)",
    )
    screen_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Optional path to export discovery leaderboard as CSV (e.g. leaderboard.csv)",
    )

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if args.command is None:
        console.print(
            "[bold cyan]pyscio[/bold cyan]: Autonomous Statistical Inference Engine & One-Tap EDA"
        )
        console.print(f"Version: [green]{sc.__version__}[/green]")
        console.print("\nRun [bold green]pyscio --help[/bold green] for available commands:")
        console.print(
            "  [cyan]pyscio profile <file> [-o report.html][/cyan] : Generate full EDA report"
        )
        console.print(
            "  [cyan]pyscio screen <file> -y <target> [-o out.csv][/cyan] : Multi-feature screening"
        )
        return 0

    if args.command == "profile":
        path = Path(args.file)
        if not path.exists():
            console.print(
                f"[bold red]Error:[/bold red] Dataset file not found at: [yellow]{path}[/yellow]"
            )
            return 1

        with console.status(f"[bold green]Ingesting and profiling {path.name}...[/bold green]"):
            df = sc.read(path)
            report = sc.eda.profile(df, title=args.title)

        console.print(report)

        if args.output:
            out_path = Path(args.output)
            report.save_html(out_path)
            console.print(
                f"[bold green]Report saved successfully to:[/bold green] [cyan]{out_path.resolve()}[/cyan]"
            )
        return 0

    if args.command == "screen":
        path = Path(args.file)
        if not path.exists():
            console.print(
                f"[bold red]Error:[/bold red] Dataset file not found at: [yellow]{path}[/yellow]"
            )
            return 1

        with console.status(
            f"[bold green]Ingesting and screening features in {path.name}...[/bold green]"
        ):
            df = sc.read(path)
            report_screen = sc.screen(df, y=args.target, alpha=args.alpha)

        console.print(report_screen)

        if args.output:
            out_path = Path(args.output)
            report_screen.leaderboard().to_csv(out_path, index=False)
            console.print(
                f"[bold green]Leaderboard saved successfully to:[/bold green] [cyan]{out_path.resolve()}[/cyan]"
            )
        return 0

    return 0
