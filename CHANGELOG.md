# Changelog

All notable changes to **`pyscio`** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **5-Layer Modular Architecture & Enforcement:** Split large modules into `pyscio/core/results/`, `pyscio/models/`, `pyscio/stat/router/`, and `pyscio/integrations/`, enforced by `scripts/check_module_size.py` in CI.
- **Lazy Package Startup (PEP 562):** `import pyscio` now loads submodules on demand via `__getattr__`.
- **Distributional Pathology Guards (`F.check_pathology`, `F.sarle_bimodality`):** Screens for point-mass/zero-inflation ($\ge 30\%$ ties), heavy-tailed/Cauchy regimes ($\text{SD}/(\text{IQR}/1.34898) > 2.2$), and bimodal mixtures (Sarle's $BC > 0.555$) before permitting Large-$N$ CLT overrides.
- **Exact Non-Central $t$ Effect Size Confidence Intervals (`F.cohen_d_ci`, `F.hedges_g_ci`, `F.cliffs_d_ci`):** Populates `TestResult.effect_ci` across two-sample, paired, and one-sample designs.
- **Tensor-Broadcasted Post-Hoc Pairwise Algorithms (`F.tukey`, `F.games_howell`, `F.dunn`):** Replaced Python pair loops with `np.triu_indices` upper-triangle tensor broadcasting and `np.add.reduceat`.
- **Wilkinson Formula Engine & Multivariate Models (`sc.ols`, `sc.logistic`):** Supports formulas (`"y ~ x1 * x2 + C(group)"`), MacKinnon-White HC0–HC3 robust standard errors, and Newton-Raphson IRLS binary logistic regression.
- **Resampling, Equivalence, & Robust Statistics:** Added `F.permutation_test`, `F.bootstrap_ci` ($BC_a$), `sc.tost` / `F.tost`, `F.trimmed_mean`, `F.winsorized_mean`, `F.huber_location`, and `F.yuen_t`.
- **Continuous Correlation Routing (`F.pearson_r`, `F.spearman_r`):** `sc.compare(df, x="num1", y="num2")` now automatically routes continuous-vs-continuous pairs to Pearson $r$ or Spearman $\rho$ with exact $z$-transformed CIs.
- **Documentation & Governance Suite:** Added MkDocs-Material documentation under `docs/`, `CITATION.cff`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, and `SECURITY.md`.

## [0.1.0] - 2026-11-03

### Added
- Initial release of `pyscio` with `pyscio.core`, `pyscio.functional`, `pyscio.eda`, `pyscio.stat`, Pandas `.scio` accessor, and Rich CLI.
