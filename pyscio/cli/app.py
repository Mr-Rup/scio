"""CLI entry point for pyscio."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from rich.console import Console

from pyscio.eda import profile, read
from pyscio.stat import compare, ols, screen

__all__ = ["main"]


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser for pyscio."""
    parser = argparse.ArgumentParser(
        prog="scio",
        description="pyscio: Autonomous Statistical Inference Engine & One-Tap EDA",
    )
    subparsers = parser.add_subparsers(dest="command")

    # profile subcommand
    p_prof = subparsers.add_parser("profile", help="Run exploratory data analysis on a dataset")
    p_prof.add_argument("path", type=str, help="Path to input CSV/Parquet/Excel/JSON file")
    p_prof.add_argument(
        "-o", "--output", type=str, default=None, help="Output file path (.html/.json)"
    )

    # compare subcommand
    p_comp = subparsers.add_parser("compare", help="Run autonomous hypothesis testing")
    p_comp.add_argument("path", type=str, help="Path to input dataset")
    p_comp.add_argument("-x", type=str, required=True, help="Grouping column or Wilkinson formula")
    p_comp.add_argument("-y", type=str, default=None, help="Target outcome column")
    p_comp.add_argument("-o", "--output", type=str, default=None, help="Output file path")

    # screen subcommand
    p_scr = subparsers.add_parser(
        "screen", help="Screen all features against target outcome with FDR"
    )
    p_scr.add_argument("path", type=str, help="Path to input dataset")
    p_scr.add_argument(
        "-y", "--target", dest="target", type=str, required=True, help="Target column"
    )
    p_scr.add_argument("-o", "--output", type=str, default=None, help="Output CSV/HTML path")

    # ols subcommand
    p_ols = subparsers.add_parser("ols", help="Fit Ordinary Least Squares (HC3) regression")
    p_ols.add_argument("path", type=str, help="Path to input dataset")
    p_ols.add_argument("-f", "--formula", type=str, required=True, help="Wilkinson formula string")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI runner entry point."""
    console = Console()
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if not args.command:
        console.print(
            "[bold cyan]pyscio[/bold cyan]: Autonomous Statistical Inference Engine & One-Tap EDA"
        )
        parser.print_help()
        return 0

    df = read(args.path)

    if args.command == "profile":
        rep = profile(df)
        console.print(rep)
        if args.output:
            out_path = Path(args.output)
            if out_path.suffix.lower() == ".html":
                out_path.write_text(rep._repr_html_(), encoding="utf-8")
            else:
                rep.to_df().to_csv(out_path, index=False)
        return 0

    if args.command == "compare":
        res = compare(df, x=args.x, y=args.y)
        console.print(res)
        if args.output:
            out_path = Path(args.output)
            if out_path.suffix.lower() == ".html":
                out_path.write_text(res._repr_html_(), encoding="utf-8")
            else:
                res.to_df().to_csv(out_path, index=False)
        return 0

    if args.command == "screen":
        scr = screen(df, y=args.target)
        console.print(scr)
        if args.output:
            out_path = Path(args.output)
            if out_path.suffix.lower() == ".html":
                out_path.write_text(scr._repr_html_(), encoding="utf-8")
            else:
                scr.leaderboard_df.to_csv(out_path, index=False)
        return 0

    if args.command == "ols":
        reg = ols(df, formula=args.formula)
        console.print(reg)
        return 0

    return 0
