"""APA 7th Edition statistical reporting formatters for TestResult and RegressionResult."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyscio.core.results.inference import TestResult
    from pyscio.core.results.regression import RegressionResult


def _format_apa_p(p: float) -> str:
    """Format a p-value in APA 7th Edition style (no leading zero, clamped at < .001)."""
    if p < 0.001:
        return "p < .001"
    return f"p = {p:.3f}".replace("0.", ".")


def format_test_apa(res: TestResult) -> str:
    """Generate an APA 7th Edition statistical report string for a TestResult."""
    name_low = res.test_name.lower()
    p_str = _format_apa_p(res.p)

    # Determine statistic symbol & degrees of freedom formatting
    if "t-test" in name_low or "t test" in name_low or "tost" in name_low:
        sym = "t"
    elif "anova" in name_low:
        sym = "F"
    elif "kruskal" in name_low:
        sym = "H"
    elif "mann-whitney" in name_low:
        sym = "U"
    elif "wilcoxon" in name_low:
        sym = "W"
    elif "chi-squared" in name_low or "mcnemar" in name_low or "friedman" in name_low:
        sym = "χ²"
    elif "pearson" in name_low:
        sym = "r"
    elif "spearman" in name_low:
        sym = "r_s"
    elif "kendall" in name_low:
        sym = "τ_b"
    else:
        sym = "stat"

    if isinstance(res.dof, tuple):
        dof_str = ", ".join(f"{int(d)}" if float(d).is_integer() else f"{d:.2f}" for d in res.dof)
        stat_part = f"{sym}({dof_str}) = {res.stat:.2f}"
    elif res.dof is not None:
        d_val = int(res.dof) if float(res.dof).is_integer() else round(float(res.dof), 2)
        if sym in ("r", "r_s") and res.effect is not None:
            stat_part = f"{sym}({d_val}) = {res.effect:.2f}"
        else:
            stat_part = f"{sym}({d_val}) = {res.stat:.2f}"
    else:
        stat_part = f"{sym} = {res.stat:.2f}"

    parts = [f"{res.test_name}: {stat_part}", p_str]

    if res.effect is not None and res.metric is not None and sym not in ("r", "r_s"):
        eff_sym = "d"
        m_low = res.metric.lower()
        if "hedges" in m_low:
            eff_sym = "g"
        elif "cliff" in m_low:
            eff_sym = "δ"
        elif "eta" in m_low:
            eff_sym = "η²"
        elif "epsilon" in m_low:
            eff_sym = "ε²"
        elif "cram" in m_low:
            eff_sym = "V"
        elif "kendall" in m_low:
            eff_sym = "W"
        elif "odds" in m_low:
            eff_sym = "OR"
        parts.append(f"{eff_sym} = {res.effect:.2f}")

    ci_tuple = res.effect_ci if res.effect_ci is not None else res.ci
    if ci_tuple is not None:
        pct = round(res.ci_level * 100)
        parts.append(f"{pct}% CI [{ci_tuple[0]:.2f}, {ci_tuple[1]:.2f}]")

    return ", ".join(parts)


def format_regression_apa(res: RegressionResult) -> str:
    """Generate an APA 7th Edition omnibus summary string for a RegressionResult."""
    p_str = _format_apa_p(res.p)
    if "logistic" in res.model_type.lower():
        return (
            f"{res.model_type} ({res.formula}): χ²({res.df_model}, N = {res.n_obs}) = {res.stat:.2f}, "
            f"{p_str}, McFadden R² = {res.r_squared:.3f}, AIC = {res.aic:.1f}"
        )
    return (
        f"{res.model_type} ({res.formula}): F({res.df_model}, {res.df_resid}) = {res.stat:.2f}, "
        f"{p_str}, R² = {res.r_squared:.3f}, adj. R² = {res.adj_r_squared:.3f}"
    )
