# Changelog

All notable changes to **`omnistats`** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-11-04

### Changed
- **Official Rebranding to `omnistats`:**
  - Package renamed to **`omnistats`** on PyPI (`pip install omnistats`).
  - Primary namespace: `import omnistats as os` or `import omnistats as omni`, with `import omnistats.functional as F`.
  - Added full backwards-compatibility shim for `pyscio` (`import pyscio as sc` continues to work with a deprecation notice).
  - DataFrame accessors registered under both `df.omni` (primary) and `df.scio` (alias) across Pandas and Polars.

## [0.3.0] - 2026-11-04

### Added
- **Survival & Time-to-Event Analysis (S1 — `F.kaplan_meier`, `F.logrank_test`, `sc.coxph`):**
  - Kaplan-Meier product-limit survival estimator (`F.kaplan_meier`) with Greenwood variance, Kalbfleisch-Prentice $[0, 1]$ log-log confidence intervals, and median survival time.
  - Mantel-Cox Log-Rank test (`F.logrank_test`) for right-censored survival comparisons with Mantel-Haenszel Hazard Ratios.
  - Cox Proportional Hazards regression (`sc.coxph(df, "Surv(time, event) ~ x1 + x2")`, `df.scio.coxph`, and automatic `sc.compare(df, "Surv(time, event) ~ ...")` routing) via Breslow partial likelihood Newton-Raphson with Hazard Ratios ($\text{HR} = \exp(\beta)$) and Harrell's Concordance Index ($C$-statistic).
- **Causal Inference & Propensity Adjustment (S2 — `sc.ate`, `sc.covariate_balance`):**
  - Robins-Rotnitzky-Zhao **Doubly Robust Augmented Inverse Probability Weighting (`sc.ate(method="aipw")`)** and **Hajek IPTW (`sc.ate(method="iptw")`)** with semiparametric influence-function standard errors and positivity overlap guards.
  - Standardized Mean Difference covariate balance auditing (`sc.covariate_balance`, `df.scio.covariate_balance`) comparing unweighted vs. propensity-weighted $|\text{SMD}| < 0.10$.
- **Time-Series Stationarity, Autocorrelation, & HAC Standard Errors (S3 — `F.adf_test`, `F.ljung_box`, `F.durbin_watson`, `sc.ols(robust_se="HAC")`):**
  - Augmented Dickey-Fuller (`F.adf_test`) unit-root test with Schwert lag selection and MacKinnon response-surface $p$-values.
  - Ljung-Box portmanteau serial autocorrelation test (`F.ljung_box`) and Durbin-Watson residual autocorrelation statistic (`F.durbin_watson`).
  - Newey-West Heteroscedasticity and Autocorrelation Consistent (**HAC**) Bartlett-kernel covariance estimator in `sc.ols(..., robust_se="HAC", hac_lags=...)`.

## [0.2.0] - 2026-11-04

### Added
- **5-Layer Modular Architecture & Enforcement (R0–R6):** Split large modules into `omnistats/core/results/`, `omnistats/models/`, `omnistats/stat/router/`, and `omnistats/integrations/`, enforced by `scripts/check_module_size.py --strict` in CI.
- **Lazy Package Startup (PEP 562, R5/R8):** `import omnistats` loads submodules on demand via `importlib.import_module` with `globals()` caching.
- **Distributional Pathology Guards (`F.check_pathology`, `F.sarle_bimodality`):** Screens for point-mass/zero-inflation ($\ge 30\%$ ties), heavy-tailed/Cauchy regimes ($\text{SD}/(\text{IQR}/1.34898) > 2.2$), and bimodal mixtures (Sarle's $BC > 0.555$) before permitting Large-$N$ CLT overrides.
- **Exact Non-Central $t$ Effect Size Confidence Intervals (`F.cohen_d_ci`, `F.hedges_g_ci`, `F.cliffs_d_ci`):** Populates `TestResult.effect_ci` across two-sample, paired, and one-sample designs.
- **Tensor-Broadcasted Post-Hoc Pairwise Algorithms (`F.tukey`, `F.games_howell`, `F.dunn`):** Replaced Python pair loops with `np.triu_indices` upper-triangle tensor broadcasting and `np.add.reduceat`.
- **Wilkinson Formula Engine & Multivariate Models (`sc.ols`, `sc.logistic`, `sc.poisson`, `sc.anova_table`, `sc.mixedlm`):**
  - Ordinary Least Squares (`sc.ols`) with QR decomposition and MacKinnon-White HC0–HC3 robust standard errors.
  - Type II & Type III Factorial ANOVA / ANCOVA tables (`sc.anova_table`) with partial $\eta_p^2$ and $\omega^2$.
  - Binary Logistic Regression (`sc.logistic`) via Newton-Raphson IRLS with Odds Ratios.
  - Poisson Log-Linear GLM (`sc.poisson`) via IRLS with QMLE HC3 sandwich SEs, Incidence Rate Ratios ($\text{IRR}$), and Pearson overdispersion diagnostics.
  - Linear Mixed-Effects Models (`sc.mixedlm` and `sc.compare(df, "y ~ x + (1 | cluster)")`) via exact Profiled REML / ML with QR whitening, Intraclass Correlation Coefficient ($\text{ICC}$), and Nakagawa's Marginal ($R^2_m$) & Conditional ($R^2_c$) $R^2$.
- **Resampling, Equivalence, & Robust Statistics:** Added `F.permutation_test`, `F.bootstrap_ci` ($BC_a$), `sc.tost` / `F.tost`, `F.trimmed_mean`, `F.winsorized_mean`, `F.huber_location`, and `F.yuen_t`.
- **Complete Correlation Suite (`F.pearson_r`, `F.spearman_r`, `F.kendall_tau`, `F.partial_corr`):** `sc.compare(df, x="num1", y="num2")` automatically routes continuous-vs-continuous pairs to Pearson $r$ or Spearman $\rho$ with exact $z$-transformed CIs; added Kendall's $\tau_b$ and QR-projected partial correlation.
- **Paired Categorical & Repeated-Measures K-Sample (`F.mcnemar`, `F.rm_anova`, `F.friedman`, `F.kendalls_w`):** Added exact/asymptotic McNemar test for paired $2\times 2$ tables and autonomous repeated-measures routing (`sc.compare(df, x="condition", y="score", paired=True)`) between One-Way Repeated-Measures ANOVA (with Greenhouse-Geisser sphericity adjustment $\hat{\varepsilon}_{GG}$) and Friedman's test (with Kendall's $W$).
- **Default Bayes Factors (`F.bayes_factor_t`, `F.interpret_bf`):** Added Rouder's JZS (Jeffreys-Zellner-Siow) Bayes Factor ($BF_{10}$ and $BF_{01}$) for one-sample, paired, and two-sample $t$-tests via 1D Gaussian quadrature.
- **APA 7th Edition Reporting (`.apa()`):** Added `.apa()` and `summary(format="apa")` to `TestResult` and `RegressionResult`.
- **Bidirectional Power & Sample Size Solver (`sc.solve_power`, `F.solve_power_t`):** Solves for required sample size $n$ per group, statistical power, or minimum detectable effect size $d$.
- **Ecosystem Extras (`omnistats[plot]`, `omnistats[polars]`):**
  - Dual-backend `.plot()` method on `TestResult`, `RegressionResult`, `ScreenReport`, and `EDAReport`, plus `sc.plot_raincloud` / `df.scio.plot_raincloud` supporting both `matplotlib` and zero-dependency `SVGFigure`.
  - Universal DataFrame coercion (`ensure_pandas_df`) and `PolarsScioNamespace` (`pl_df.scio`) supporting Pandas, Polars (`DataFrame` & `LazyFrame`), Narwhals, PyArrow, and Python dicts.
- **Documentation, Benchmarks, & Governance Suite:** Added MkDocs-Material documentation under `docs/`, micro-benchmark suite under `benchmarks/bench_kernels.py`, `CITATION.cff`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, and `SECURITY.md`.

## [0.1.0] - 2026-11-03

### Added
- Initial release of `omnistats` with `omnistats.core`, `omnistats.functional`, `omnistats.eda`, `omnistats.stat`, Pandas `.scio` accessor, and Rich CLI.
