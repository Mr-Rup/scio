# pyscio Documentation

**`pyscio`** is an autonomous statistical inference engine, one-tap exploratory data analysis (EDA) suite, multivariate regression toolkit, and pure numerical hypothesis testing framework for Python (`>= 3.11`).

## Architectural Layers

`pyscio` enforces a strict 5-layer downward dependency graph:

1. **`pyscio.core` (L0):** Immutable result containers (`TestResult`, `RegressionResult`, `EDAReport`, `ScreenReport`, `SVGFigure`), typed domain exceptions (`StatError`, `SampleSizeError`, `ZeroVarError`), 8-role semantic column classification (`Role`), universal DataFrame coercion (`ensure_pandas_df`), and HTML/Rich/LaTeX/Markdown/APA/SVG renderers.
2. **`pyscio.functional` as `F` (L1):** Stateless, deterministic NumPy/SciPy mathematical kernels returning lightweight `NamedTuple`s (`TTestResult`, `ANOVAResult`, `RMAnovaResult`, `NonParametricResult`, `TOSTResult`, `CorrelationResult`, `BayesFactorResult`).
3. **`pyscio.models` (L2):** Wilkinson formula design matrix parser (`parse_formula`), Ordinary Least Squares (`ols` with QR decomposition and MacKinnon-White HC0–HC3 robust standard errors), Type II & Type III Factorial ANOVA (`anova_table`), Newton-Raphson IRLS binary logistic regression (`logistic`), Poisson log-linear GLM (`poisson` with QMLE HC3 SEs & IRR), and Linear Mixed-Effects Models (`mixedlm` via Profiled REML/ML).
4. **`pyscio.stat` & `pyscio.eda` (L3):** Autonomous assumption-checked decision router (`compare`, `screen`, `tost`, `solve_power`) and vectorized exploratory data analysis (`read`, `profile`, `audit_quality`, `correlations`, `vif`).
5. **`pyscio.integrations` & `pyscio.cli` (L4):** Native Pandas (`df.scio`) and Polars (`pl_df.scio`) DataFrame accessors, dual-backend statistical plotting (`plot_raincloud`, `.plot()`), and the `pyscio` command-line interface.

## Feature Matrix

| Capability | High-Level API (`sc.*` / `df.scio.*`) | Stateless Kernel (`pyscio.functional as F`) |
| :--- | :--- | :--- |
| **Two-Sample Parametric** | `sc.compare(df, x="arm", y="val")` | `F.student_t`, `F.welch_t`, `F.paired_t`, `F.one_sample_t` |
| **Two-Sample Non-Parametric** | Auto-routed on normality/pathology failure | `F.mann_whitney`, `F.wilcoxon`, `F.permutation_test` |
| **K-Group Omnibus & Post-Hoc** | Auto-triggers Tukey, Games-Howell, or Dunn | `F.anova`, `F.welch_anova`, `F.kruskal`, `F.tukey`, `F.games_howell`, `F.dunn` |
| **Repeated Measures & Paired** | `sc.compare(df, x="cond", y="val", paired=True)` | `F.rm_anova` (Greenhouse-Geisser), `F.friedman`, `F.kendalls_w`, `F.mcnemar` |
| **Factorial ANOVA / ANCOVA** | `sc.anova_table(df, "y ~ C(A) * C(B)", typ=2)` | `pyscio.models.anova_table` (Type II & Type III SS) |
| **Linear, GLM, & Mixed Models** | `sc.ols`, `sc.logistic`, `sc.poisson`, `sc.mixedlm` | QR HC0–HC3 & Newey-West HAC OLS, IRLS Logistic/Poisson, Profiled REML LMM |
| **Survival Analysis** | `sc.coxph(df, "Surv(time, event) ~ x1 + x2")` | `F.kaplan_meier` (Greenwood log-log CI), `F.logrank_test`, Harrell's $C$-index |
| **Causal Inference** | `sc.ate(df, y, treat, covars)`, `sc.covariate_balance` | Doubly Robust AIPW, Hajek IPTW, Standardized Mean Difference (SMD) |
| **Time-Series Diagnostics** | `sc.ols(df, "y ~ x", robust_se="HAC")` | `F.adf_test` (Unit Root), `F.ljung_box` ($Q$-test), `F.durbin_watson` |
| **Effect Sizes & Exact CIs** | Populates `.effect` and `.effect_ci` | `F.cohen_d_ci`, `F.hedges_g_ci`, `F.cliffs_d_ci`, `F.eta_sq`, `F.omega_sq` |
| **Correlation & Partial Corr** | `sc.compare(df, x="num1", y="num2")` | `F.pearson_r`, `F.spearman_r`, `F.kendall_tau`, `F.partial_corr` |
| **Bayesian & Power Analysis** | `sc.solve_power(effect=0.5, power=0.8)` | `F.bayes_factor_t` (JZS $BF_{10}$), `F.interpret_bf`, `F.solve_power_t` |
| **Resampling & Equivalence** | `sc.tost(a, b, bound=0.5)` | `F.bootstrap_ci` ($BC_a$), `F.permutation_test`, `F.tost` |
| **Robust Statistics** | Outlier-resistant location & inference | `F.trimmed_mean`, `F.winsorized_mean`, `F.huber_location`, `F.yuen_t` |
| **Visualization & Polars** | `result.plot()`, `sc.plot_raincloud`, `pl_df.scio` | `pyscio[plot]` (`matplotlib` + `SVGFigure`), `pyscio[polars]` |
