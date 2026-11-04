"""Tests for omnistats command-line interface."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from omnistats.cli.app import main


def test_cli_help(capsys: pytest.CaptureFixture[str]) -> None:
    # Running without arguments prints welcome and basic help
    code = main([])
    assert code == 0


def test_cli_profile_and_export(tmp_path: Path) -> None:
    # Create test CSV
    csv_path = tmp_path / "sample.csv"
    df = pd.DataFrame(
        {
            "a": np.arange(10, dtype=float),
            "b": ["X", "Y"] * 5,
        }
    )
    df.to_csv(csv_path, index=False)

    out_html = tmp_path / "report.html"

    # Profile command with output
    code = main(["profile", str(csv_path), "-o", str(out_html)])
    assert code == 0
    assert out_html.exists()
    content = out_html.read_text(encoding="utf-8")
    assert "Exploratory Data Analysis Report" in content


def test_cli_screen_and_export(tmp_path: Path) -> None:
    csv_path = tmp_path / "screen_sample.csv"
    df = pd.DataFrame(
        {
            "target": ["Control", "Treatment"] * 20,
            "val1": np.random.normal(10, 2, 40),
            "val2": np.random.normal(50, 5, 40),
        }
    )
    df.to_csv(csv_path, index=False)

    out_csv = tmp_path / "leaderboard.csv"

    # Screen command with output
    code = main(["screen", str(csv_path), "-y", "target", "-o", str(out_csv)])
    assert code == 0
    assert out_csv.exists()
    lb = pd.read_csv(out_csv)
    assert "feature" in lb.columns
    assert "significant_fdr" in lb.columns
