# pyscio

[![CI](https://github.com/Mr-Rup/scio/actions/workflows/ci.yml/badge.svg)](https://github.com/Mr-Rup/scio/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/pyscio.svg)](https://pypi.org/project/pyscio/)
[![Python](https://img.shields.io/pypi/pyversions/pyscio.svg)](https://pypi.org/project/pyscio/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checked: mypy](https://img.shields.io/badge/type%20checked-mypy%20strict-blue.svg)](https://github.com/python/mypy)

**`pyscio`** is an autonomous statistical inference engine, one-tap exploratory data analysis (EDA) suite, and pure numerical hypothesis testing framework for Python.

Designed from first principles to be the modern statistical companion to **Pandas**, **NumPy**, and **PyTorch**, `pyscio` bridges the gap between raw data and statistically sound, peer-reviewed scientific conclusions.

---

## ⚡ Highlights

* **Pure Stateless Mathematical Kernels (`pyscio.functional` as `F`):** PyTorch-style pure NumPy/SciPy kernels (`F.welch_t`, `F.mann_whitney`, `F.tukey`, `F.cohen_d`, `F.cliffs_d`, `F.fdr_bh`) with zero classes, zero side-effects, and zero visualization dependencies.
* **Universal Pandas EDA (`pyscio.eda`):** The comprehensive architectural evolution of [`polars-datapulse`](https://github.com/Mr-Rup/polars-datapulse) ported to a universal Pandas + NumPy engine. Includes 8-role semantic classification (`numeric`, `discrete`, `binary`, `categorical`, `temporal`, `text`, `id`, `constant`), vectorized profiling, 0–100 data quality scoring, correlation/VIF matrices, and zero-CDN standalone interactive HTML reports.
* **One-Tap Pandas Accessor (`df.scio`):** Run full EDA, hypothesis comparisons, or multi-feature screening directly on any Pandas DataFrame with clean method chaining.
* **Autonomous Decision Lattice:** Automatically inspects distribution shape (Shapiro-Wilk / D'Agostino $K^2$), evaluates homoscedasticity (Levene / Bartlett), checks sample constraints, and routes to the mathematically optimal test.
* **Large-$N$ Central Limit Theorem (CLT) Guard:** Prevents statistical over-rejection on large samples ($N \ge 500$) when skewness ($|\gamma_1| < 1.0$) and kurtosis ($|\gamma_2| < 2.0$) are well-behaved, routing cleanly to Welch's $t$-test with an explicit CLT assumption audit note.
* **$O(N \log N)$ Cliff's $\delta$:** Computes the non-parametric Cliff's $\delta$ effect size in $O((N_1 + N_2) \log(N_1 + N_2))$ time via the Mann-Whitney $U$ identity $\delta = \frac{2U}{n_1 n_2} - 1$ rather than naive $O(N_1 N_2)$ loops.
* **Confidence Intervals & Automatic Post-Hoc Pairwise Tests:** Every hypothesis result includes exact $(1 - \alpha)$ confidence intervals. Significant $K > 2$ group tests automatically trigger Tukey HSD (ANOVA), Games-Howell (Welch's ANOVA), or Dunn's test with Holm/BH adjustment (Kruskal-Wallis).
* **Multi-Feature Discovery Screening:** Test dozens of features against a target outcome with automatic Benjamini-Hochberg False Discovery Rate (FDR) control and an effect-size leaderboard.
* **Zero-Config Dual Rendering:** Interactive HTML cards in Jupyter/Colab notebooks (`_repr_html_()`) and Rich tables in terminals (`__rich_console__()`).
* **Strict R 4.3 Parity:** All statistical outputs and effect sizes are benchmarked against R golden reference vectors to $\le 10^{-6}$ numerical tolerance.

---

## 📦 Installation

Using `uv` (recommended):
```bash
uv add pyscio
```

Using `pip`:
```bash
pip install pyscio
```

Requirements: Python `>= 3.11`.

---

## 🚀 Quickstart

### 1. One-Tap Exploratory Data Analysis (EDA)

```python
import pandas as pd
import pyscio as sc

df = sc.read("clinical_trial.csv")  # Auto-sniffs CSV/TSV, Parquet, Excel, JSON

# Generate a complete structural and quality report
report = df.scio.eda()

# Render summary to terminal
print(report.summary())

# Export a standalone, self-contained interactive HTML report
report.to_html("eda_report.html")
```

### 2. Autonomous Hypothesis Comparison

Let `pyscio` check assumptions, test for variance homogeneity, evaluate the Central Limit Theorem, and choose the most powerful valid test:

```python
# Compare continuous recovery days across treatment arms
result = df.scio.compare(x="treatment_arm", y="recovery_days")

# Display the rich statistical card
result.summary()
```

Output:
```text
┌─ Hypothesis Test Result ────────────────────────────────────────────────────────┐
│ Target: recovery_days ~ treatment_arm                                            │
│ Test: Welch's Two-Sample t-test (CLT Guard Active)                              │
│ Statistic: t = 3.4812, p = 0.00062, df = 142.8                                  │
│ 95% Confidence Interval: [1.84, 6.12]                                           │
│ Effect Size: Hedges' g = 0.582 (Medium)                                          │
│ Assumptions: Normality (p=0.082, OK), Homoscedasticity (p=0.014, Heteroscedastic)│
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 3. Multi-Feature Discovery Screening with FDR Control

Screen all candidate biomarkers or features against an outcome in one line:

```python
screen = df.scio.screen(y="remission")

# Print ranked discoveries with Benjamini-Hochberg q-values
print(screen.leaderboard())
```

### 4. Pure Stateless Math Kernels (`pyscio.functional`)

If you are building custom algorithms or pipelines, use `pyscio.functional` (`F`) for fast, pure, deterministic numerical kernels:

```python
import pyscio.functional as F

# Compute Welch's t-test
stat, p_val, dof = F.welch_t(group_a, group_b)

# Standardized effect sizes
d = F.cohen_d(group_a, group_b)
delta = F.cliffs_d(group_a, group_b)  # O(N log N)

# Multiple testing adjustments
rejected, q_values = F.fdr_bh(p_values, alpha=0.05)
```

---

## 🏛️ Architecture & Directory Layout

```text
pyscio/
├── pyproject.toml             # Hatchling + hatch-vcs, Python >= 3.11
├── LICENSE                    # Apache-2.0
├── .github/
│   └── workflows/
│       ├── ci.yml             # Ubuntu, macOS, Windows | Python 3.11, 3.12, 3.13
│       └── publish.yml        # Tag v* -> PyPI OIDC Trusted Publishing
├── tests/
│   ├── conftest.py            # R 4.3 golden reference fixtures
│   ├── test_core.py
│   ├── test_functional.py
│   ├── test_r_parity.py
│   ├── test_eda.py
│   └── test_stat.py
└── pyscio/
    ├── __init__.py            # Top-level exports: read, eda, stat, F, test, compare, screen
    ├── py.typed               # Strict PEP 561 typing marker
    ├── accessor.py            # Pandas DataFrame .scio accessor
    ├── core/                  # BaseResult, TestResult, CheckResult, guards, exceptions
    ├── functional/            # Pure stateless kernels: checks, tests, effects, posthoc, adjust
    ├── eda/                   # Pandas+NumPy DataPulse: io, roles, profile, quality, rel, view
    ├── stat/                  # Autonomous decision lattice, router, compare, screen, power
    └── cli/                   # Lightweight CLI runner
```

---

## 🔬 Mathematical & Numerical Rigor Standards

1. **P-Value Underflow Clamp:** All $p$-values below floating point machine epsilon (`2.220446049250313e-16`) are clamped to avoid numerical zero hazards and displayed as `p < 2.22e-16`.
2. **Sample Size & Variance Guards:** Tests raise `SampleSizeError` for $N < 3$, emit a low-power warning for $3 \le N < 15$, and raise `ZeroVarError` when variance is zero.
3. **Dual Representation:** Every result container implements `_repr_html_()` for Jupyter/Colab notebooks and `__rich_console__()` for Rich terminal sessions.
4. **R 4.3 Parity:** Verified against R `stats` golden vectors across Student's $t$, Welch's $t$, Mann-Whitney $U$, Wilcoxon, ANOVA, Kruskal-Wallis, $\chi^2$, and Fisher's Exact Test.

---

## 📄 License

Distributed under the terms of the [Apache License, Version 2.0](LICENSE).
