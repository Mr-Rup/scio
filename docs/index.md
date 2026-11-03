# pyscio Documentation

**`pyscio`** is an autonomous statistical inference engine, one-tap exploratory data analysis (EDA) suite, multivariate regression toolkit, and pure numerical hypothesis testing framework for Python (`>= 3.11`).

## Architectural Layers

`pyscio` enforces a strict 5-layer downward dependency graph:

1. **`pyscio.core` (L0):** Immutable result containers (`TestResult`, `RegressionResult`, `EDAReport`, `ScreenReport`), typed domain exceptions (`StatError`, `SampleSizeError`, `ZeroVarError`), 8-role semantic column classification (`Role`), and HTML/Rich/LaTeX/Markdown renderers.
2. **`pyscio.functional` as `F` (L1):** Stateless, deterministic NumPy/SciPy mathematical kernels returning lightweight `NamedTuple`s (`TTestResult`, `ANOVAResult`, `NonParametricResult`, `TOSTResult`, `CorrelationResult`).
3. **`pyscio.models` (L2):** Wilkinson formula design matrix parser (`parse_formula`), Ordinary Least Squares (`ols` with MacKinnon-White HC0–HC3 robust standard errors), and Newton-Raphson IRLS binary logistic regression (`logistic`).
4. **`pyscio.stat` & `pyscio.eda` (L3):** Autonomous assumption-checked decision router (`compare`, `screen`, `tost`) and vectorized exploratory data analysis (`read`, `profile`, `audit_quality`, `correlations`, `vif`).
5. **`pyscio.integrations` & `pyscio.cli` (L4):** Native Pandas DataFrame `.scio` accessor (`df.scio`) and the `pyscio` command-line interface.

## Feature Matrix

| Capability | High-Level API (`sc.*` / `df.scio.*`) | Stateless Kernel (`pyscio.functional as F`) |
| :--- | :--- | :--- |
| **Two-Sample Parametric** | `sc.compare(df, x="arm", y="val")` | `F.student_t`, `F.welch_t`, `F.paired_t`, `F.one_sample_t` |
| **Two-Sample Non-Parametric** | Auto-routed on normality/pathology failure | `F.mann_whitney`, `F.wilcoxon`, `F.permutation_test` |
| **K-Group Omnibus & Post-Hoc** | Auto-triggers Tukey, Games-Howell, or Dunn | `F.anova`, `F.welch_anova`, `F.kruskal`, `F.tukey`, `F.games_howell`, `F.dunn` |
| **Effect Sizes & Exact CIs** | Populates `.effect` and `.effect_ci` | `F.cohen_d`, `F.cohen_d_ci`, `F.hedges_g`, `F.hedges_g_ci`, `F.cliffs_d`, `F.cliffs_d_ci` |
| **Correlation & Contingency** | Auto-routed for numeric~numeric or categoricals | `F.pearson_r`, `F.spearman_r`, `F.chisq`, `F.fisher`, `F.cramers_v`, `F.odds_ratio` |
| **Multivariate Regression** | `sc.ols(df, "y ~ x1 * x2 + C(grp)")`, `sc.logistic` | `pyscio.models.ols`, `pyscio.models.logistic` |
| **Resampling & Equivalence** | `sc.tost(a, b, bound=0.5)` | `F.bootstrap_ci` ($BC_a$), `F.permutation_test`, `F.tost` |
| **Robust Statistics** | Outlier-resistant location & inference | `F.trimmed_mean`, `F.winsorized_mean`, `F.huber_location`, `F.yuen_t` |
