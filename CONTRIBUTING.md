# Contributing to omnistats

Thank you for contributing to **`omnistats`**! We follow strict engineering, mathematical, and architectural standards to keep the library fast, modular, and trustworthy.

## 1. Development Setup

We recommend [`uv`](https://docs.astral.sh/uv/) for environment and dependency management:

```bash
git clone https://github.com/Mr-Rup/scio.git
cd scio
uv sync --extra dev
```

## 2. Engineering & Architectural Rules

1. **Module Size Budgets:**
   - Library files (`omnistats/**/*.py`): soft limit **300 lines**, hard limit **400 lines**.
   - Test files (`tests/**/*.py`): hard limit **450 lines**.
2. **5-Layer Downward Dependency Graph:**
   - `L0 (omnistats.core)` $\leftarrow$ `L1 (omnistats.functional)` $\leftarrow$ `L2 (omnistats.models)` $\leftarrow$ `L3 (omnistats.stat | omnistats.eda)` $\leftarrow$ `L4 (omnistats.integrations | omnistats.cli)`.
   - Lower layers must never import higher layers, and `omnistats.stat` and `omnistats.eda` must remain independent.
3. **Pure Stateless Kernels (`omnistats.functional`):**
   - Kernels return lightweight `NamedTuple`s from `omnistats.core.types` and never import `omnistats.core.results`.
4. **Strict Typing & Error Hierarchy:**
   - All functions require complete type annotations (`mypy --strict`).
   - Raise domain errors from `omnistats.core.errors` (`StatError`, `SampleSizeError`, `ZeroVarError`, `DtypeError`, `MissingDataError`), which dual-inherit standard Python exceptions.

## 3. Pre-Submission Verification Gate

Run all checks locally before opening a Pull Request:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy omnistats
uv run python scripts/check_module_size.py --strict
uv run pytest
```
