# API Reference Manual

## Top-Level Namespace (`omnistats`)

- `sc.read(source, **kwargs)` — Multi-format tabular reader (CSV/TSV, Parquet, Excel, JSON).
- `sc.compare(data, x, y, *, alpha=0.05, paired=False, alternative="two-sided", mu=0.0, formula=None)` — Autonomous hypothesis testing router.
- `sc.screen(df, y, *, alpha=0.05, features=None)` — Multi-feature discovery screening with Benjamini-Hochberg FDR control.
- `sc.ols(df, formula, *, robust_se="HC3", alpha=0.05)` — Ordinary Least Squares regression with HC0–HC3 robust standard errors.
- `sc.logistic(df, formula, *, alpha=0.05, max_iter=50, tol=1e-6)` — Binary logistic regression via Newton-Raphson IRLS.
- `sc.tost(a, b=None, *, low=-0.5, high=0.5, bound=None, alpha=0.05, paired=False, popmean=0.0)` — Two One-Sided Tests for statistical equivalence.

## Functional Kernels (`omnistats.functional as F`)

### Parametric & Non-Parametric Tests
- `F.student_t(a, b, alternative="two-sided", conf_level=0.95) -> TTestResult`
- `F.welch_t(a, b, alternative="two-sided", conf_level=0.95) -> TTestResult`
- `F.paired_t(a, b, alternative="two-sided", conf_level=0.95) -> TTestResult`
- `F.one_sample_t(a, popmean=0.0, alternative="two-sided", conf_level=0.95) -> TTestResult`
- `F.anova(*groups) -> ANOVAResult`
- `F.welch_anova(*groups) -> ANOVAResult`
- `F.mann_whitney(a, b, alternative="two-sided") -> NonParametricResult`
- `F.wilcoxon(a, b=None, alternative="two-sided") -> NonParametricResult`
- `F.kruskal(*groups) -> NonParametricResult`
- `F.chisq(observed, correct=True) -> ChiSqResult`
- `F.fisher(table, alternative="two-sided") -> FisherResult`

### Assumption & Pathology Checks
- `F.shapiro(x, alpha=0.05) -> NormalityResult`
- `F.dagostino(x, alpha=0.05) -> NormalityResult`
- `F.check_normality(x, alpha=0.05) -> NormalityResult`
- `F.sarle_bimodality(x) -> float`
- `F.check_pathology(x, tie_threshold=0.30, tail_ratio_max=2.2, bimodality_max=0.555) -> tuple[bool, list[str]]`
- `F.levene(*groups, center="median", alpha=0.05) -> HomoscedasticityResult`
- `F.bartlett(*groups, alpha=0.05) -> HomoscedasticityResult`

### Standardized Effect Sizes & Exact CIs
- `F.cohen_d(a, b) -> float` and `F.cohen_d_ci(a, b=None, conf_level=0.95, paired=False, popmean=0.0) -> tuple[float, float]`
- `F.hedges_g(a, b) -> float` and `F.hedges_g_ci(a, b, conf_level=0.95) -> tuple[float, float]`
- `F.cliffs_d(a, b) -> float` and `F.cliffs_d_ci(a, b, conf_level=0.95) -> tuple[float, float]`
- `F.eta_sq(*groups) -> float`, `F.eps_sq(*groups) -> float`, `F.cramers_v(table) -> float`, `F.odds_ratio(table) -> OddsRatioResult`

### Post-Hoc, Resampling, Correlation, & Robust Kernels
- `F.tukey(*groups, names=None, alpha=0.05) -> pd.DataFrame`
- `F.games_howell(*groups, names=None, alpha=0.05) -> pd.DataFrame`
- `F.dunn(*groups, names=None, p_adjust="fdr_bh", alpha=0.05) -> pd.DataFrame`
- `F.pearson_r(x, y, conf_level=0.95) -> CorrelationResult`
- `F.spearman_r(x, y, conf_level=0.95) -> CorrelationResult`
- `F.permutation_test(a, b, n_resamples=10000, paired=False, alternative="two-sided", seed=None) -> NonParametricResult`
- `F.bootstrap_ci(data, stat_func=np.mean, n_resamples=5000, conf_level=0.95, method="bca", seed=None) -> tuple[float, float]`
- `F.trimmed_mean(x, trim=0.1) -> float`, `F.winsorized_mean(x, trim=0.1) -> float`, `F.huber_location(x) -> tuple[float, float]`, `F.yuen_t(a, b, trim=0.2) -> TTestResult`
