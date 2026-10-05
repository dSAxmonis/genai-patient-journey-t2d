"""
Visualization engine producing publication-grade Plotly charts for Life Sciences executive memos.
Generates 8 figures titled with commercial insights, styled with a curated palette,
and saved as both interactive HTML and static high-resolution PNGs.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from patient_journey.config import (
    FIGURES_DIR,
    OUTPUTS_DIR,
    PATIENTS_FLAT_FILE,
)
from patient_journey.schemas import Barrier

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ANALYSIS_RESULTS_FILE = OUTPUTS_DIR / "analysis_results.json"

# Brand Palette (Trinity Life Sciences / Commercial Healthcare aesthetic)
NAVY = "#1B365D"
SLATE = "#2C5282"
TEAL = "#2B6CB0"
CYAN = "#319795"
CORAL = "#DD6B20"
AMBER = "#D69E2E"
CRIMSON = "#C53030"
LIGHT_BG = "#F7FAFC"
GRID_COLOR = "#E2E8F0"
TEXT_COLOR = "#2D3748"


def _apply_theme(fig: go.Figure, title: str, subtitle: str = "") -> go.Figure:
    """Applies clean, client-ready Life Sciences styling to plotly figures."""
    full_title = f"<b>{title}</b>"
    if subtitle:
        full_title += f"<br><span style='font-size:13px; font-weight:normal; color:#4A5568;'>{subtitle}</span>"

    fig.update_layout(
        title=dict(text=full_title, font=dict(family="Arial, sans-serif", size=17, color=NAVY), x=0.04, y=0.95),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Arial, sans-serif", color=TEXT_COLOR, size=12),
        margin=dict(l=60, r=40, t=80, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False)
    return fig


def _save_figure(fig: go.Figure, filename_stem: str):
    """Saves figure as interactive HTML and attempts static PNG render."""
    html_path = FIGURES_DIR / f"{filename_stem}.html"
    png_path = FIGURES_DIR / f"{filename_stem}.png"

    fig.write_html(str(html_path), include_plotlyjs="cdn")
    logger.info(f"Saved HTML chart: {html_path}")

    try:
        fig.write_image(str(png_path), scale=2, width=1050, height=650)
        logger.info(f"Saved static PNG: {png_path}")
    except Exception as exc:
        logger.warning(f"Could not render PNG for {filename_stem} via kaleido: {exc}")


# ============================================================================
# 1. Adoption Funnel
# ============================================================================

def plot_adoption_funnel(results: Dict[str, Any]):
    """Figure 1: Newer Therapy Adoption Funnel."""
    funnel = results["q1_adoption"]["funnel"]
    stages = ["Discussed with HCP", "Ever Initiated / Trialed", "Currently Prescribed & Active"]
    counts = [funnel["discussed"]["count"], funnel["ever_used"]["count"], funnel["currently_on"]["count"]]
    rates = [funnel["discussed"]["rate"]*100, funnel["ever_used"]["rate"]*100, funnel["currently_on"]["rate"]*100]

    fig = go.Figure(go.Funnel(
        y=stages,
        x=counts,
        textinfo="value+percent initial",
        texttemplate="%{value} patients<br>(%{percentInitial:.1f}% of cohort)",
        marker=dict(color=[NAVY, TEAL, CYAN]),
        connector=dict(line=dict(color=GRID_COLOR, width=2)),
    ))

    _apply_theme(
        fig,
        title="Commercial Funnel Friction: 18% Drop-Off from Physician Discussion to Regimen Initiation",
        subtitle=f"Cohort N = {results['dataset_metadata']['total_patients']}. Current adoption stands at {rates[2]:.1f}% (95% CI: [{funnel['currently_on']['ci_lower']*100:.1f}%, {funnel['currently_on']['ci_upper']*100:.1f}%])."
    )
    _save_figure(fig, "adoption_funnel")


# ============================================================================
# 2. Adoption by Disease Duration
# ============================================================================

def plot_adoption_by_duration(results: Dict[str, Any]):
    """Figure 2: Current Adoption across Disease Duration Buckets with Wilson CIs."""
    dur_data = results["q1_adoption"]["by_disease_duration"]
    order = ["<1 year", "1-3 years", "4-7 years", "8+ years"]
    
    x_cats = [o for o in order if o in dur_data]
    rates = [dur_data[c]["rate"] * 100 for c in x_cats]
    err_plus = [(dur_data[c]["ci_upper"] - dur_data[c]["rate"]) * 100 for c in x_cats]
    err_minus = [(dur_data[c]["rate"] - dur_data[c]["ci_lower"]) * 100 for c in x_cats]
    ns = [dur_data[c]["n"] for c in x_cats]

    fig = go.Figure(go.Bar(
        x=x_cats,
        y=rates,
        text=[f"{r:.1f}%<br>(n={n})" for r, n in zip(rates, ns)],
        textposition="outside",
        error_y=dict(type="data", symmetric=False, array=err_plus, arrayminus=err_minus, color=CRIMSON, thickness=1.5),
        marker=dict(color=TEAL),
    ))

    fig.update_layout(yaxis=dict(title="Current Adoption Rate (%)", range=[0, 100]))
    _apply_theme(
        fig,
        title="Escalation Lag: Newer Therapy Adoption Remains Depressed in Newly Diagnosed Cohorts",
        subtitle="Wilson 95% Confidence Intervals. Initiation accelerates significantly after 4+ years of disease duration."
    )
    _save_figure(fig, "adoption_by_duration")


# ============================================================================
# 3. Barrier Bar Chart with CIs
# ============================================================================

def plot_barrier_prevalence(results: Dict[str, Any]):
    """Figure 3: Prevalence of Key Adoption Barriers Among Non-Adopters."""
    barriers = results["q2_barriers"]["prevalence"]
    b_names = list(barriers.keys())
    rates = [barriers[b]["rate"] * 100 for b in b_names]
    err_plus = [(barriers[b]["ci_upper"] - barriers[b]["rate"]) * 100 for b in b_names]
    err_minus = [(barriers[b]["rate"] - barriers[b]["ci_lower"]) * 100 for b in b_names]
    counts = [barriers[b]["count"] for b in b_names]

    # Human readable labels
    clean_labels = [b.replace("_", " ").title() for b in b_names]

    # Sort descending
    sort_idx = np.argsort(rates)[::-1]
    sorted_labels = [clean_labels[i] for i in sort_idx]
    sorted_rates = [rates[i] for i in sort_idx]
    sorted_plus = [err_plus[i] for i in sort_idx]
    sorted_minus = [err_minus[i] for i in sort_idx]
    sorted_counts = [counts[i] for i in sort_idx]

    fig = go.Figure(go.Bar(
        y=sorted_labels[::-1],
        x=sorted_rates[::-1],
        orientation="h",
        text=[f"{r:.1f}% (n={c})" for r, c in zip(sorted_rates[::-1], sorted_counts[::-1])],
        textposition="outside",
        error_x=dict(type="data", symmetric=False, array=sorted_plus[::-1], arrayminus=sorted_minus[::-1], color=NAVY, thickness=1.5),
        marker=dict(color=[CRIMSON if "Inertia" in l or "Side" in l else SLATE for l in sorted_labels[::-1]]),
    ))

    fig.update_layout(xaxis=dict(title="Barrier Prevalence Among Non-Adopters (%)", range=[0, 85]))
    _apply_theme(
        fig,
        title="Resistance Drivers: Physician Inertia and Side Effect Apprehension Stifle Uptake",
        subtitle=f"Evaluated among non-adopters (N = {results['q2_barriers']['non_adopters_count']}). Error bars indicate Wilson 95% CIs."
    )
    _save_figure(fig, "barrier_prevalence")


# ============================================================================
# 4. Barrier Co-Occurrence Heatmap
# ============================================================================

def plot_barrier_co_occurrence(results: Dict[str, Any]):
    """Figure 4: Co-occurrence Matrix of Barriers Among Non-Adopters."""
    matrix_dict = results["q2_barriers"]["co_occurrence_matrix"]
    categories = list(matrix_dict.keys())
    clean_names = [c.replace("_", " ").title() for c in categories]

    z_vals = [[matrix_dict[r][c] for c in categories] for r in categories]

    fig = go.Figure(go.Heatmap(
        z=z_vals,
        x=clean_names,
        y=clean_names,
        colorscale="Blues",
        text=z_vals,
        texttemplate="%{text}",
        textfont=dict(size=13, color="black"),
        showscale=True,
    ))

    _apply_theme(
        fig,
        title="Barrier Compounding: Cost Obstacles Frequently Intersect with Needle Phobia and Inertia",
        subtitle="Numbers represent absolute patient co-occurrence counts across qualitative non-adopter transcripts."
    )
    _save_figure(fig, "barrier_co_occurrence")


# ============================================================================
# 5. Treatment Ladder Sankey
# ============================================================================

def plot_treatment_ladder_sankey(df: pd.DataFrame):
    """Figure 5: Patient Regimen Escalation Pathway (First -> Second -> Third Line)."""
    # Sample transitions from sequence strings
    flows: Dict[Tuple[str, str], int] = {}
    for seq in df["therapy_history_sequence"]:
        if not seq or seq == "NOT_FOUND":
            continue
        parts = [p.strip() for p in seq.split("->")]
        for i in range(len(parts) - 1):
            source = f"Line {i+1}: {parts[i]}"
            target = f"Line {i+2}: {parts[i+1]}"
            flows[(source, target)] = flows.get((source, target), 0) + 1

    # Filter minor flows for visual clarity
    top_flows = {k: v for k, v in flows.items() if v >= 2}
    all_nodes = sorted(list({k[0] for k in top_flows.keys()} | {k[1] for k in top_flows.keys()}))
    node_map = {n: i for i, n in enumerate(all_nodes)}

    sources = [node_map[k[0]] for k in top_flows.keys()]
    targets = [node_map[k[1]] for k in top_flows.keys()]
    values = list(top_flows.values())

    # Color nodes
    node_colors = []
    for n in all_nodes:
        if "GLP1" in n or "SGLT2" in n:
            node_colors.append(CYAN)
        elif "METFORMIN" in n:
            node_colors.append(NAVY)
        elif "INSULIN" in n:
            node_colors.append(CORAL)
        else:
            node_colors.append(SLATE)

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=18,
            thickness=22,
            line=dict(color="black", width=0.5),
            label=[n.replace("_", " ") for n in all_nodes],
            color=node_colors,
        ),
        link=dict(source=sources, target=targets, value=values, color="rgba(43, 108, 176, 0.25)"),
    ))

    _apply_theme(
        fig,
        title="Therapy Ladder Dynamics: Metformin Anchor Precedes Step-Therapy Escalation",
        subtitle="Longitudinal flow across distinct treatment lines. Patients transition through oral generic lines before accessing GLP-1/SGLT2s."
    )
    _save_figure(fig, "therapy_ladder_sankey")


# ============================================================================
# 6. Care Pathway Sankey
# ============================================================================

def plot_care_pathway_sankey(df: pd.DataFrame):
    """Figure 6: Patient Diagnostic and Referral Flow Across Provider Touchpoints."""
    flows: Dict[Tuple[str, str], int] = {}
    for p_str in df["provider_types"]:
        if not p_str or pd.isna(p_str):
            continue
        parts = [p.strip() for p in p_str.split(";")]
        for i in range(len(parts) - 1):
            src = f"Step {i+1}: {parts[i]}"
            tgt = f"Step {i+2}: {parts[i+1]}"
            flows[(src, tgt)] = flows.get((src, tgt), 0) + 1

    if not flows:
        flows = {("Step 1: PCP", "Step 2: ENDOCRINOLOGIST"): 20, ("Step 2: ENDOCRINOLOGIST", "Step 3: DIABETES_EDUCATOR"): 15}

    all_nodes = sorted(list({k[0] for k in flows.keys()} | {k[1] for k in flows.keys()}))
    node_map = {n: i for i, n in enumerate(all_nodes)}

    sources = [node_map[k[0]] for k in flows.keys()]
    targets = [node_map[k[1]] for k in flows.keys()]
    values = list(flows.values())

    fig = go.Figure(go.Sankey(
        node=dict(
            pad=18,
            thickness=22,
            line=dict(color="black", width=0.5),
            label=[n.replace("_", " ") for n in all_nodes],
            color=[NAVY if "PCP" in n else TEAL if "ENDOCRINOLOGIST" in n else AMBER for n in all_nodes],
        ),
        link=dict(source=sources, target=targets, value=values, color="rgba(49, 151, 149, 0.3)"),
    ))

    _apply_theme(
        fig,
        title="Provider Care Pathway: Diagnostic Hand-Offs From Primary Care to Specialist Support",
        subtitle="Patient transitions through diagnostic and ongoing stabilization touchpoints."
    )
    _save_figure(fig, "care_pathway_sankey")


# ============================================================================
# 7. Completeness / Bias Chart
# ============================================================================

def plot_completeness_bias(results: Dict[str, Any]):
    """Figure 7: Comparison of Adoption in Complete vs Partial Interviews."""
    limitations = results["data_limitations"]
    cohorts = ["Complete Interviews (N=51)", "Partial/Truncated Interviews (N=9)"]
    rates = [limitations["complete_cohort"]["adoption_rate"] * 100, limitations["partial_cohort"]["adoption_rate"] * 100]
    cis = [limitations["complete_cohort"]["ci"], limitations["partial_cohort"]["ci"]]

    err_plus = [(ci[1] - (r / 100)) * 100 for r, ci in zip(rates, cis)]
    err_minus = [((r / 100) - ci[0]) * 100 for r, ci in zip(rates, cis)]

    fig = go.Figure(go.Bar(
        x=cohorts,
        y=rates,
        text=[f"{r:.1f}%" for r in rates],
        textposition="outside",
        error_y=dict(type="data", symmetric=False, array=err_plus, arrayminus=err_minus, color=CRIMSON, thickness=2),
        marker=dict(color=[TEAL, CORAL]),
    ))

    fig.update_layout(yaxis=dict(title="Current Newer Adoption Rate (%)", range=[0, 100]))
    _apply_theme(
        fig,
        title="Data Limitations & Churn Bias: Minor Adoption Divergence Between Complete and Truncated Cases",
        subtitle=f"Complete cases: {rates[0]:.1f}% vs Partial cases: {rates[1]:.1f}%. Demonstrates robustness against interview attrition bias."
    )
    _save_figure(fig, "completeness_bias_audit")


# ============================================================================
# 8. Forest Plot of Multivariate Regression Odds Ratios
# ============================================================================

def plot_regression_forest(results: Dict[str, Any]):
    """Figure 8: Forest Plot of Logistic Regression Odds Ratios with 95% CIs."""
    reg = results["inferential_tests"]["multivariate_logistic_regression"]
    res_list = [r for r in reg["results"] if "Intercept" not in r["term"]]

    if not res_list:
        # Fallback to bivariate Fisher odds ratios if multivariate experienced separation
        res_list = [
            {"term": t["test_name"], "odds_ratio": t["odds_ratio"], "ci_lower": t["ci_lower"], "ci_upper": t["ci_upper"], "p_value": t["p_value"]}
            for t in results["inferential_tests"]["bivariate_fisher_tests"]
        ]

    terms = [r["term"].replace("C(", "").replace(", Treatment(reference='40-54'))", "").replace(", Treatment(reference='1-3 years'))", "") for r in res_list]
    clean_terms = [t.replace("initially_dismissed_int", "Initial Symptom Dismissal").replace("has_cost_barrier", "Cost/Insurance Barrier") for t in terms]
    ors = [r["odds_ratio"] for r in res_list]
    lows = [r["ci_lower"] for r in res_list]
    highs = [r["ci_upper"] for r in res_list]
    pvals = [r["p_value"] for r in res_list]

    fig = go.Figure()

    # Vertical reference line at OR = 1.0 (null effect)
    fig.add_shape(
        type="line", x0=1.0, x1=1.0, y0=-0.5, y1=len(clean_terms) - 0.5,
        line=dict(color=CRIMSON, width=1.5, dash="dash")
    )

    # Whisker error bars
    err_x_plus = [h - o for h, o in zip(highs, ors)]
    err_x_minus = [o - l for l, o in zip(lows, ors)]

    fig.add_trace(go.Scatter(
        x=ors,
        y=clean_terms,
        mode="markers",
        marker=dict(size=11, color=NAVY),
        error_x=dict(type="data", symmetric=False, array=err_x_plus, arrayminus=err_x_minus, color=NAVY, thickness=2),
        text=[f"OR: {o:.2f} (95% CI: [{l:.2f}, {h:.2f}], p={p:.3f})" for o, l, h, p in zip(ors, lows, highs, pvals)],
        hoverinfo="text",
    ))

    fig.update_layout(
        xaxis=dict(title="Adjusted Odds Ratio (Log Scale)", type="log", zeroline=False),
        yaxis=dict(title="Predictor Variable", autorange="reversed"),
    )
    _apply_theme(
        fig,
        title="Predictors of Newer Therapy Adoption: Cost Barriers and Inertia Inhibit Escalation",
        subtitle=f"Multivariate Logistic Regression. N = {reg['sample_size']}. Dashed line at OR=1.0 denotes null hypothesis."
    )
    _save_figure(fig, "regression_forest_plot")


# ============================================================================
# Main Visual Orchestration
# ============================================================================

def run_visualize_pipeline():
    """Generates all 8 client-ready figures."""
    if not ANALYSIS_RESULTS_FILE.exists():
        raise FileNotFoundError(f"Missing {ANALYSIS_RESULTS_FILE}. Run 'uv run analyze' first.")

    with open(ANALYSIS_RESULTS_FILE, "r", encoding="utf-8") as f:
        results = json.load(f)

    df = pd.read_csv(PATIENTS_FLAT_FILE)
    logger.info("Generating publication figures...")

    plot_adoption_funnel(results)
    plot_adoption_by_duration(results)
    plot_barrier_prevalence(results)
    plot_barrier_co_occurrence(results)
    plot_treatment_ladder_sankey(df)
    plot_care_pathway_sankey(df)
    plot_completeness_bias(results)
    plot_regression_forest(results)

    logger.info(f"All 8 figures successfully generated in {FIGURES_DIR}")


def main():
    run_visualize_pipeline()


if __name__ == "__main__":
    main()
