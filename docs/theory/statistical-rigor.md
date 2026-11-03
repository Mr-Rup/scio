# Mathematical Rigor & Statistical Guards

## 1. Distributional Pathology Guards & Large-$N$ CLT

For large sample sizes ($N \ge 500$), formal normality tests (Shapiro-Wilk, D'Agostino-Pearson $K^2$) reject normality for negligible deviations. However, naive moment-based Central Limit Theorem (CLT) overrides ($|\gamma_1| < 1.0 \land |\gamma_2| < 2.0$) can be fooled by symmetric heavy-tailed, zero-inflated, or bimodal distributions.

`pyscio` enforces a 3-stage pathology screen (`F.check_pathology`) before permitting a CLT override:

1. **Point-Mass / Zero-Inflation Guard:** Rejects parametric routing if the maximum single-value tie proportion $\max_v \frac{\#\{x_i = v\}}{n} \ge 0.30$.
2. **Heavy-Tailed / Cauchy Scale Guard:** Compares sample standard deviation against the Gaussian-calibrated interquartile scale:
   $$R_{\text{tail}} = \frac{s}{\text{IQR} / 1.34898}$$
   If $R_{\text{tail}} > 2.2$ or the fraction of observations outside Tukey's outer fences $[Q_1 - 3\text{IQR}, Q_3 + 3\text{IQR}]$ exceeds $2.5\%$, parametric routing is blocked.
3. **Sarle's Bimodality Coefficient ($BC$):**
   $$BC = \frac{\gamma_1^2 + 1}{\gamma_2 + 3 \frac{(n-1)^2}{(n-2)(n-3)}}$$
   If $BC > 0.555$ alongside platykurtic excess kurtosis ($\gamma_2 < -0.5$), the distribution is flagged as bimodal/multimodal and routed to non-parametric inference.

## 2. Exact Non-Central $t$ Confidence Intervals for Effect Sizes

Rather than relying on symmetric normal approximations for Cohen's $d$ and Hedges' $g$, `pyscio` inverts the cumulative distribution function of the Non-Central $t$-distribution ($F_{t,\nu}(\cdot; \lambda)$) via Brent's root-finding method (`scipy.optimize.brentq`):

$$F_{t,\nu}(t_{\text{obs}}; \lambda_L) = 1 - \frac{\alpha}{2}, \qquad F_{t,\nu}(t_{\text{obs}}; \lambda_U) = \frac{\alpha}{2}$$

where $\lambda = d \sqrt{\frac{n_1 n_2}{n_1 + n_2}}$ for independent two-sample designs.

## 3. MacKinnon-White HC3 Robust Covariance

In `sc.ols`, default standard errors use the finite-sample bias-corrected HC3 jackknife sandwich estimator:

$$\hat{\Sigma}_{\text{HC3}} = (X^T X)^{-1} X^T \operatorname{diag}\!\left(\frac{e_i^2}{(1 - h_{ii})^2}\right) X (X^T X)^{-1}$$

where $h_{ii} = x_i^T (X^T X)^{-1} x_i$ is the $i$-th diagonal element of the projection (hat) matrix.
