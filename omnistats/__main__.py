"""Command-line execution entry point (`python -m omnistats`)."""

from __future__ import annotations

import sys

from omnistats.cli.app import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
