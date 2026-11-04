"""Enforce module line budgets and layered architecture import contracts for omnistats.

Standards enforced (from framework_specs/10_ENGINEERING_STANDARDS.md):
1. Module size limits:
   - omnistats/**/*.py: soft 300 lines, hard 400 lines
   - tests/**/*.py: hard 450 lines
2. Layered import contracts:
   - L0 (omnistats.core): never imports functional, models, stat, eda, integrations, cli
   - L1 (omnistats.functional): never imports core.results, models, stat, eda, integrations, cli
   - L2 (omnistats.models): never imports stat, eda, integrations, cli
   - L3 (omnistats.stat / omnistats.eda): independent (stat never imports eda; eda never imports stat)
"""

from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Zero legacy allowlisted files — all size and layer contracts enforced strictly
LEGACY_SIZE_ALLOWLIST: set[str] = set()

LEGACY_IMPORT_ALLOWLIST: set[tuple[str, str]] = set()

FORBIDDEN_PREFIXES: dict[str, tuple[str, ...]] = {
    "core": (
        "omnistats.functional",
        "omnistats.models",
        "omnistats.stat",
        "omnistats.eda",
        "omnistats.integrations",
        "omnistats.cli",
    ),
    "functional": (
        "omnistats.core.results",
        "omnistats.models",
        "omnistats.stat",
        "omnistats.eda",
        "omnistats.integrations",
        "omnistats.cli",
    ),
    "models": (
        "omnistats.stat",
        "omnistats.eda",
        "omnistats.integrations",
        "omnistats.cli",
    ),
    "stat": (
        "omnistats.eda",
        "omnistats.integrations",
        "omnistats.cli",
    ),
    "eda": (
        "omnistats.stat",
        "omnistats.models",
        "omnistats.integrations",
        "omnistats.cli",
    ),
}


def check_file_sizes(strict: bool = False) -> list[str]:
    """Return list of size-budget violations across omnistats/ and tests/."""
    errors: list[str] = []
    for folder, hard_limit in (("omnistats", 400), ("tests", 450)):
        base = ROOT / folder
        if not base.exists():
            continue
        for path in sorted(base.rglob("*.py")):
            rel = path.relative_to(ROOT).as_posix()
            n_lines = len(path.read_text(encoding="utf-8").splitlines())
            if n_lines > hard_limit:
                if not strict and rel in LEGACY_SIZE_ALLOWLIST:
                    continue
                errors.append(f"[SIZE] {rel}: {n_lines} lines exceeds hard limit ({hard_limit})")
    return errors


def _extract_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.append(node.module)
    return imports


def check_layer_imports(strict: bool = False) -> list[str]:
    """Return list of layer-architecture import violations in omnistats/."""
    errors: list[str] = []
    pkg_root = ROOT / "omnistats"
    for path in sorted(pkg_root.rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        parts = path.relative_to(pkg_root).parts
        if len(parts) < 2:
            continue
        subpkg = parts[0]
        forbidden = FORBIDDEN_PREFIXES.get(subpkg, ())
        if not forbidden:
            continue
        for imp in _extract_imports(path):
            for bad_prefix in forbidden:
                if imp == bad_prefix or imp.startswith(bad_prefix + "."):
                    if not strict and (rel, bad_prefix) in LEGACY_IMPORT_ALLOWLIST:
                        continue
                    errors.append(f"[LAYER] {rel} (in '{subpkg}') imports forbidden module '{imp}'")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify omnistats module size and layer contracts."
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Disallow legacy allowlisted files (enforced from R4 onward).",
    )
    args = parser.parse_args()

    violations = check_file_sizes(strict=args.strict) + check_layer_imports(strict=args.strict)
    if violations:
        for v in violations:
            print(v, file=sys.stderr)
        return 1

    print("OK: All module size budgets and layer architecture contracts satisfied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
