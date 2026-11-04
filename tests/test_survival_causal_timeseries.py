"""Tests for S1 (Survival Analysis), S2 (Causal Inference AIPW/IPTW), and S3 (Time-Series & HAC SEs)."""

from __future__ import annotations

import numpy as np
import pandas as pd

import omnistats as sc
import omnistats.functional as F


def test_s1_kaplan_meier_and_logrank() -> None:
    """Verify Kaplan-Meier product-limit survival curve, Greenwood log-log CIs, and Mantel-Cox Log-Rank test."""
    rng = np.random.default_rng(77)
    # Control arm: higher hazard (shorter survival); Treatment arm: lower hazard (longer survival)
    t_ctrl = rng.exponential(scale=5.0, size=60)
    e_ctrl = rng.binomial(1, 0.85, size=60)
    t_trt = rng.exponential(scale=12.0, size=60)
    e_trt = rng.binomial(1, 0.80, size=60)

    km = F.kaplan_meier(t_ctrl, e_ctrl)
    assert len(km.times) == len(km.survival_prob)
    assert km.survival_prob[0] == 1.0
    assert all(
        0.0 <= lo <= s <= hi <= 1.0
        for lo, s, hi in zip(km.ci_low, km.survival_prob, km.ci_high, strict=False)
    )
    assert np.isfinite(km.median_survival) and km.median_survival > 0.0

    lr = F.logrank_test(t_ctrl, e_ctrl, t_trt, e_trt)
    assert lr.stat > 6.0
    assert lr.p < 0.02
    assert lr.hazard_ratio > 1.3


def test_s1_cox_proportional_hazards_and_compare_routing() -> None:
    """Verify Cox PH Breslow partial likelihood, Hazard Ratio recovery, C-index, and compare() routing."""
    rng = np.random.default_rng(2026)
    n = 160
    treatment = rng.binomial(1, 0.5, size=n)
    age = rng.normal(0.0, 1.0, size=n)
    # True log-hazard: -0.9 * treatment + 0.6 * age (treatment reduces hazard; age increases hazard)
    lin_pred = -0.9 * treatment + 0.6 * age
    true_rate = 0.1 * np.exp(lin_pred)
    surv_time = rng.exponential(scale=1.0 / true_rate)
    censor_time = rng.uniform(5.0, 40.0, size=n)
    obs_time = np.minimum(surv_time, censor_time)
    status = (surv_time <= censor_time).astype(int)

    df = pd.DataFrame({"time": obs_time, "status": status, "treatment": treatment, "age": age})

    cox_res = sc.coxph(df, "Surv(time, status) ~ treatment + age")
    assert "Cox Proportional Hazards" in cox_res.model_type
    assert cox_res.p < 1e-4
    # r_squared stores Harrell's C-index
    assert cox_res.r_squared > 0.65

    trt_row = cox_res.coef_df[cox_res.coef_df["term"] == "treatment"].iloc[0]
    age_row = cox_res.coef_df[cox_res.coef_df["term"] == "age"].iloc[0]
    # Treatment reduces hazard (coef < 0, HR < 1)
    assert float(trt_row["coef"]) < -0.3
    assert float(trt_row["hazard_ratio"]) < 0.75
    # Age increases hazard (coef > 0, HR > 1)
    assert float(age_row["coef"]) > 0.3
    assert float(age_row["hazard_ratio"]) > 1.35

    # Compare() automatic Surv(time, status) ~ ... formula routing
    cmp_cox = df.scio.compare("Surv(time, status) ~ treatment + age")
    assert "Cox Proportional Hazards" in cmp_cox.test_name
    assert cmp_cox.sig


def test_s2_causal_inference_aipw_iptw_and_balance() -> None:
    """Verify Doubly Robust AIPW and IPTW recover true causal ATE under strong observational confounding."""
    rng = np.random.default_rng(314)
    n = 400
    # Confounder X strongly drives both treatment assignment T and outcome Y
    confounder = rng.normal(0.0, 1.0, size=n)
    ps_true = 1.0 / (1.0 + np.exp(-1.3 * confounder))
    treatment = rng.binomial(1, ps_true)
    # True causal ATE = 3.0, confounding slope = 4.0
    true_ate = 3.0
    outcome = 10.0 + true_ate * treatment + 4.0 * confounder + rng.normal(0.0, 1.0, size=n)

    df = pd.DataFrame({"y": outcome, "treat": treatment, "confounder": confounder})

    # Naive unadjusted difference in means is heavily biased (> 5.0 instead of 3.0)
    naive_diff = float(df[df["treat"] == 1]["y"].mean() - df[df["treat"] == 0]["y"].mean())
    assert naive_diff > 4.5

    # Doubly Robust AIPW recovers true ATE ~ 3.0
    aipw_res = df.scio.ate(outcome="y", treatment="treat", covariates=["confounder"], method="aipw")
    assert aipw_res.sig
    assert aipw_res.ci is not None
    assert aipw_res.ci[0] < true_ate < aipw_res.ci[1]

    # IPTW also recovers true ATE ~ 3.0
    iptw_res = sc.ate(df, outcome="y", treatment="treat", covariates=["confounder"], method="iptw")
    assert iptw_res.sig
    assert iptw_res.ci is not None
    assert 2.2 < (iptw_res.ci[0] + iptw_res.ci[1]) / 2.0 < 3.8

    # Covariate balance check: unweighted |SMD| is large (> 0.8), weighted |SMD| is balanced (< 0.15)
    bal_df = df.scio.covariate_balance(treatment="treat", covariates=["confounder"])
    assert float(bal_df["smd_unweighted"].abs().max()) > 0.7
    assert float(bal_df["smd_weighted"].abs().max()) < 0.15


def test_s3_timeseries_stationarity_and_newey_west_hac() -> None:
    """Verify ADF unit-root test, Ljung-Box Q test, Durbin-Watson, and Newey-West HAC OLS standard errors."""
    rng = np.random.default_rng(909)
    n = 200
    white_noise = rng.normal(0.0, 1.0, size=n)
    random_walk = np.cumsum(white_noise)

    # 1. ADF test: white noise is stationary (p < 0.01); random walk is non-stationary (p > 0.05)
    adf_wn = F.adf_test(white_noise, max_lags=2)
    assert adf_wn.is_stationary
    assert adf_wn.p < 0.01

    adf_rw = F.adf_test(random_walk, max_lags=2)
    assert not adf_rw.is_stationary
    assert adf_rw.p > 0.05

    # 2. Ljung-Box test & Durbin-Watson on AR(1) vs white noise
    ar1 = np.zeros(n, dtype=np.float64)
    for t in range(1, n):
        ar1[t] = 0.8 * ar1[t - 1] + white_noise[t]

    lb_wn = F.ljung_box(white_noise, lags=5)
    assert lb_wn.p > 0.05

    lb_ar1 = F.ljung_box(ar1, lags=5)
    assert lb_ar1.p < 0.001

    dw_wn = F.durbin_watson(white_noise)
    dw_ar1 = F.durbin_watson(ar1)
    assert 1.6 < dw_wn < 2.4
    assert dw_ar1 < 0.8

    # 3. OLS with Newey-West HAC robust standard errors
    x_ts = np.linspace(0.0, 10.0, n)
    y_ts = 2.0 + 1.5 * x_ts + ar1
    ts_df = pd.DataFrame({"x": x_ts, "y": y_ts})
    hac_res = sc.ols(ts_df, "y ~ x", robust_se="HAC", hac_lags=5)
    assert "Newey-West HAC SE" in hac_res.model_type
    assert hac_res.robust_se == "HAC(L=5)"
    assert hac_res.p < 1e-6
