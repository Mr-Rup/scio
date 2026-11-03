# Quickstart Guide

## Installation

```bash
uv add pyscio
# or
pip install pyscio
```

## 1. One-Tap Exploratory Data Analysis

```python
import pyscio as sc

df = sc.read("clinical_trial.csv")
eda_report = df.scio.eda()
print(eda_report.summary())
eda_report.to_html("eda_report.html")
```

## 2. Autonomous Hypothesis Testing & Formula Interface

```python
# Standard group comparison (checks normality, CLT, pathology, and variance homogeneity)
res = df.scio.compare(x="treatment_arm", y="recovery_days")
print(res.summary())

# Wilkinson formula syntax (single predictor or multivariate ANCOVA/OLS)
res_formula = df.scio.compare("recovery_days ~ treatment_arm + baseline_score")
```

## 3. Multivariate Regression (OLS HC3 & IRLS Logistic)

```python
# Ordinary Least Squares with MacKinnon-White HC3 heteroscedasticity-robust standard errors
ols_res = sc.ols(df, "recovery_days ~ age * C(treatment_arm)", robust_se="HC3")
print(ols_res.summary())

# Binary Logistic Regression with Odds Ratios and McFadden's Pseudo-R²
log_res = sc.logistic(df, "remission ~ age + C(treatment_arm)")
print(log_res.to_df())
```

## 4. Stateless Functional Kernels (`pyscio.functional as F`)

```python
import pyscio.functional as F

# Welch's t-test returning a 6-field TTestResult NamedTuple
t_res = F.welch_t(group_a, group_b)
print(t_res.stat, t_res.p, t_res.dof, t_res.ci)

# Exact Non-Central t Confidence Intervals for Hedges' g and Cohen's d
g = F.hedges_g(group_a, group_b)
g_ci = F.hedges_g_ci(group_a, group_b, conf_level=0.95)

# BCa Bootstrap CI & Yuen-Welch Robust Trimmed t-Test
ci_bca = F.bootstrap_ci(group_a, n_resamples=5000, method="bca", seed=42)
yuen_res = F.yuen_t(group_a, group_b, trim=0.2)
```
