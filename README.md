# pyscio

[![CI](https://github.com/Mr-Rup/scio/actions/workflows/ci.yml/badge.svg)](https://github.com/Mr-Rup/scio/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/pyscio.svg)](https://pypi.org/project/pyscio/)
[![Python](https://img.shields.io/pypi/pyversions/pyscio.svg)](https://pypi.org/project/pyscio/)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checked: mypy](https://img.shields.io/badge/type%20checked-mypy%20strict-blue.svg)](https://github.com/python/mypy)

**`pyscio`** is an autonomous statistical inference engine, one-tap exploratory data analysis (EDA) suite, multivariate regression toolkit, and pure numerical hypothesis testing framework for Python (`>= 3.11`).

---

## ⚡ Highlights

* **Pure Stateless Mathematical Kernels (`pyscio.functional` as `F`):** PyTorch-style pure NumPy/SciPy kernels (`F.welch_t`, `F.mann_whitney`, `F.tukey`, `F.cohen_d`, `F.cliffs_d`, `F.fdr_bh`, `F.bootstrap_ci`, `F.yuen_t`) returning lightweight `NamedTuple`s with zero classes and zero side-effects.
* **Autonomous Decision Lattice & Pathology Guards (`pyscio.stat`):** Inspects distribution shape (Shapiro-Wilk / D'Agostino $K^2$), checks for **point-mass zero-inflation**, **heavy-tailed / Cauchy regimes**, and **Sarle's bimodality ($BC > 0.555$)** before permitting a Large-$N$ ($N \ge 500$) Central Limit Theorem override, evaluates homoscedasticity (Levene / Brown-Forsythe), and routes to the mathematically optimal test.
* **Exact Non-Central $t$ Effect Size CIs:** Computes exact Non-Central $t$-distribution confidence intervals (`F.cohen_d_ci`, `F.hedges_g_ci`) via Brent root-finding and $z$-transformed CIs for $O(N \log N)$ Cliff's $\delta$ (`F.cliffs_d_ci`), surfacing `effect_ci` on every `TestResult`.
* **Tensor-Broadcasted Post-Hoc Pairwise Tests:** Significant $K > 2$ omnibus tests automatically trigger vectorized Tukey HSD (ANOVA), Games-Howell (Welch's ANOVA), or Dunn's test with BH/Holm adjustment (Kruskal-Wallis).
* **Wilkinson Formulas & Robust Multivariate Models (`pyscio.models`):** Run `sc.compare(df, "y ~ x1 + C(group)")`, Ordinary Least Squares with MacKinnon-White **HC0–HC3** heteroscedasticity-robust standard errors (`sc.ols`), and Newton-Raphson IRLS binary logistic regression with Odds Ratios (`sc.logistic`).
* **Resampling, Equivalence, & Robust Statistics:** Exact & vectorized Monte Carlo `F.permutation_test`, Bias-Corrected and Accelerated ($BC_a$) `F.bootstrap_ci`, Two One-Sided Tests for equivalence (`sc.tost`), Huber M-estimators (`F.huber_location`), and Yuen-Welch trimmed $t$-tests (`F.yuen_t`).
* **Universal Pandas EDA (`pyscio.eda`) & Accessor (`df.scio`):** 8-role semantic column classification, vectorized profiling, 0–100 data health scoring, correlation/VIF matrices, and standalone interactive HTML reports.

---

## 📦 Installation

```bash
uv add pyscio
# or
pip install pyscio
```

---

## 🚀 Quickstart

### 1. One-Tap Exploratory Data Analysis (EDA)

```python
import pyscio as sc

df = sc.read("clinical_trial.csv")  # Auto-sniffs CSV/TSV, Parquet, Excel, JSON
report = df.scio.eda()
print(report.summary())
report.to_html("eda_report.html")
```

### 2. Autonomous Hypothesis Comparison & Wilkinson Formulas

```python
# Compare continuous outcome across treatment arms (auto-routes & computes exact effect CIs)
result = df.scio.compare(x="treatment_arm", y="recovery_days")
print(result.summary())

# Or use Wilkinson formula syntax (bivariate or multivariate ANCOVA/OLS)
res_formula = df.scio.compare("recovery_days ~ treatment_arm + baseline_score")
```

### 3. Multivariate Regression (OLS HC3 & IRLS Logistic)

```python
# Ordinary Least Squares with MacKinnon-White HC3 robust standard errors
ols_res = sc.ols(df, "recovery_days ~ age * C(treatment_arm)", robust_se="HC3")
print(ols_res.summary())

# Binary Logistic Regression with Odds Ratios and McFadden Pseudo-R²
log_res = sc.logistic(df, "remission ~ age + C(treatment_arm)")
```

### 4. Stateless Math Kernels (`pyscio.functional as F`)

```python
import pyscio.functional as F

# Welch's t-test returns a 6-field TTestResult NamedTuple
res = F.welch_t(group_a, group_b)
print(res.stat, res.p, res.dof, res.ci)

# Effect sizes with exact Non-Central t CIs & O(N log N) Cliff's delta
g, g_ci = F.hedges_g(group_a, group_b), F.hedges_g_ci(group_a, group_b)
delta, d_ci = F.cliffs_d(group_a, group_b), F.cliffs_d_ci(group_a, group_b)

# BCa Bootstrap CI, TOST Equivalence, & Yuen-Welch Robust Trimmed t-Test
ci_bca = F.bootstrap_ci(group_a, n_resamples=5000, method="bca", seed=42)
eq_res = sc.tost(group_a, group_b, bound=0.5)
yuen_res = F.yuen_t(group_a, group_b, trim=0.2)
```

---

## 🏛️ 5-Layer Architecture

```text
pyscio/
├── pyproject.toml             # Hatchling + hatch-vcs, Python >= 3.11
├── mkdocs.yml                 # MkDocs-Material documentation config
├── docs/                      # Quickstart, mathematical theory, & API reference
├── scripts/
│   └── check_module_size.py   # Enforces module size budgets & 5-layer import contracts
├── tests/                     # Unit, R 4.3 parity, Hypothesis property, & API snapshot tests
└── pyscio/
    ├── __init__.py            # PEP 562 lazy exports: read, compare, screen, ols, logistic, tost, F
    ├── __main__.py            # CLI execution (`python -m pyscio`)
    ├── core/                  # L0: constants, errors, guards, types, roles, results/
    ├── functional/            # L1: pure stateless kernels (checks, tests, effects, posthoc, resample, robust)
    ├── models/                # L2: Wilkinson formula parser, OLS (HC0-HC3), IRLS Logistic
    ├── stat/                  # L3: Autonomous router/ subpackage, compare, screen, equivalence, power
    ├── eda/                   # L3: Multi-format I/O, profiler, quality, correlations/VIF
    ├── integrations/          # L4: Pandas DataFrame `.scio` accessor
    └── cli/                   # L4: Command-line interface (`pyscio`)
```

---

## 📄 License & Citation

Distributed under the [Apache License, Version 2.0](LICENSE). See [`CITATION.cff`](CITATION.cff) for academic citation metadata.
