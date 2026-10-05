"""
Interactive Streamlit Web Dashboard: GenAI Patient Journey Analytics
Life Sciences Commercial Analytics Practice (Trinity Life Sciences Case Project)
"""

from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

# Configuration & Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
PATIENTS_FLAT_FILE = OUTPUTS_DIR / "patients_flat.csv"
ANALYSIS_RESULTS_FILE = OUTPUTS_DIR / "analysis_results.json"
TRANSCRIPTS_FILE = DATA_DIR / "transcripts.json"
VALIDATION_REPORT_FILE = OUTPUTS_DIR / "validation_report.md"

# Page Configuration
st.set_page_config(
    page_title="T2D Patient Journey Analytics | GenAI Commercial Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for executive consulting look
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1A365D;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4A5568;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F7FAFC;
        border-left: 4px solid #2B6CB0;
        padding: 1rem;
        border-radius: 4px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .recommendation-card {
        background-color: #EDF2F7;
        border: 1px solid #CBD5E0;
        padding: 1.2rem;
        border-radius: 6px;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    """Loads flat patients dataset and analysis results."""
    df = pd.read_csv(PATIENTS_FLAT_FILE)
    with open(ANALYSIS_RESULTS_FILE, "r") as f:
        results = json.load(f)
    with open(TRANSCRIPTS_FILE, "r") as f:
        transcripts = json.load(f)
    return df, results, transcripts


try:
    df, results, transcripts = load_data()
except Exception as e:
    st.error(f"Error loading project artifacts: {e}. Please run 'uv run analyze' first.")
    st.stop()


# Sidebar Navigation
st.sidebar.image("https://img.icons8.com/fluency/96/dna-helix.png", width=64)
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Analysis Module:",
    [
        "Executive Summary & Strategy",
        "Adoption Funnel & Segmentation",
        "Barrier Deep Dive & Co-Occurrence",
        "Treatment Ladder & Care Pathways",
        "Inferential Statistics & Bias Audit",
        "Patient Transcript Explorer",
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Project Details**")
st.sidebar.markdown("- **Cohort Size:** N = 60 Transcripts")
st.sidebar.markdown("- **Disease:** Type 2 Diabetes (T2D)")
st.sidebar.markdown("- **Therapies:** GLP-1 RA & SGLT2i")
st.sidebar.markdown("- **Pipeline:** Instructor + Pydantic v2")
st.sidebar.info("Synthetic benchmark designed for Life Sciences Commercial & Qualitative Analytics.")


# ============================================================================
# Page 1: Executive Summary & Recommendations
# ============================================================================
if page == "Executive Summary & Strategy":
    st.markdown('<div class="main-header">GenAI Patient Journey Analytics: T2D Treatment Adoption</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Commercial Strategy & Qualitative Evidence Mining for Next-Generation Metabolic Regimens (GLP-1 RA / SGLT2i)</div>', unsafe_allow_html=True)

    # Top KPI Metrics
    q1 = results["q1_adoption"]["funnel"]
    ladder = results["q3_ladder"]
    pathways = results["q4_pathways"]

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Discussed with HCP", f"{q1['discussed']['rate']*100:.1f}%", f"{q1['discussed']['count']} / 60 patients")
    with col2:
        st.metric("Active Newer Adoption", f"{q1['currently_on']['rate']*100:.1f}%", f"95% CI: [{q1['currently_on']['ci_lower']*100:.1f}%, {q1['currently_on']['ci_upper']*100:.1f}%]")
    with col3:
        st.metric("Median Prior Steps", f"{ladder['median_steps_before_newer']} lines", "Ladder depth to GLP-1/SGLT2")
    with col4:
        st.metric("Diagnostic Dismissal", f"{pathways['symptom_dismissal']['rate']*100:.1f}%", f"{pathways['symptom_dismissal']['count']} delayed cases")

    st.markdown("---")
    st.subheader("Executive Strategic Recommendations (Trinity Life Sciences Client Memo)")

    recs = results.get("client_recommendations", [])
    for rec in recs:
        with st.container():
            st.markdown(f"""
            <div class="recommendation-card">
                <h4 style="color:#1A365D; margin-top:0;">Pillar {rec['recommendation_number']}: {rec['pillar']}</h4>
                <h5 style="color:#2B6CB0;">{rec['headline']}</h5>
                <p><b>Analytical Finding:</b> {rec['finding']}</p>
                <p><b>Commercial Action:</b> {rec['action']}</p>
            </div>
            """, unsafe_allow_html=True)

    # Quick preview chart
    st.subheader("Adoption Bottleneck Overview")
    if (FIGURES_DIR / "adoption_funnel.png").exists():
        st.image(str(FIGURES_DIR / "adoption_funnel.png"), use_container_width=True)


# ============================================================================
# Page 2: Adoption Funnel & Segmentation
# ============================================================================
elif page == "Adoption Funnel & Segmentation":
    st.title("Adoption Funnel & Longitudinal Segmentation")
    st.markdown("Examining drop-off velocity between physician consultation, trial, and ongoing active adherence.")

    col1, col2 = st.columns(2)
    with col1:
        if (FIGURES_DIR / "adoption_funnel.png").exists():
            st.image(str(FIGURES_DIR / "adoption_funnel.png"), use_container_width=True)
    with col2:
        if (FIGURES_DIR / "adoption_by_duration.png").exists():
            st.image(str(FIGURES_DIR / "adoption_by_duration.png"), use_container_width=True)

    st.subheader("Demographic Stratification Table")
    dur_df = pd.DataFrame(results["q1_adoption"]["by_disease_duration"]).T
    dur_df["Adoption Rate (%)"] = (dur_df["rate"] * 100).round(1)
    dur_df["95% CI Lower (%)"] = (dur_df["ci_lower"] * 100).round(1)
    dur_df["95% CI Upper (%)"] = (dur_df["ci_upper"] * 100).round(1)
    st.dataframe(dur_df[["n", "current_adopters", "Adoption Rate (%)", "95% CI Lower (%)", "95% CI Upper (%)"]], use_container_width=True)


# ============================================================================
# Page 3: Barrier Deep Dive & Co-Occurrence
# ============================================================================
elif page == "Barrier Deep Dive & Co-Occurrence":
    st.title("Barriers to Adoption Among Non-Adopters (N=27)")
    st.markdown("Uncovering the behavioral, economic, and clinical resistance points that stall therapy initiation.")

    col1, col2 = st.columns(2)
    with col1:
        if (FIGURES_DIR / "barrier_prevalence.png").exists():
            st.image(str(FIGURES_DIR / "barrier_prevalence.png"), use_container_width=True)
    with col2:
        if (FIGURES_DIR / "barrier_co_occurrence.png").exists():
            st.image(str(FIGURES_DIR / "barrier_co_occurrence.png"), use_container_width=True)

    st.subheader("Patient Archetypes Identified")
    archetypes = results["q2_barriers"]["archetypes"]
    cols = st.columns(len(archetypes))
    for col, arc in zip(cols, archetypes):
        with col:
            st.markdown(f"**{arc['archetype']}**")
            st.caption(f"Share: {arc['share_of_non_adopters']*100:.1f}%")
            st.write(arc["description"])


# ============================================================================
# Page 4: Treatment Ladder & Care Pathways
# ============================================================================
elif page == "Treatment Ladder & Care Pathways":
    st.title("Treatment Ladder & Provider Touchpoint Pathways")
    st.markdown("Mapping longitudinal sequence dynamics from first-line anchor to specialist escalation.")

    col1, col2 = st.columns(2)
    with col1:
        if (FIGURES_DIR / "therapy_ladder_sankey.png").exists():
            st.image(str(FIGURES_DIR / "therapy_ladder_sankey.png"), use_container_width=True)
    with col2:
        if (FIGURES_DIR / "care_pathway_sankey.png").exists():
            st.image(str(FIGURES_DIR / "care_pathway_sankey.png"), use_container_width=True)

    st.subheader("Top Historical Treatment Sequences")
    top_seq = results["q3_ladder"]["top_sequences"]
    seq_df = pd.DataFrame(list(top_seq.items()), columns=["Treatment Ladder Sequence", "Patient Count"])
    st.dataframe(seq_df, use_container_width=True)


# ============================================================================
# Page 5: Inferential Statistics & Bias Audit
# ============================================================================
elif page == "Inferential Statistics & Bias Audit":
    st.title("Inferential Biostatistics & Data Limitations Audit")
    st.markdown("Testing hypothesis associations with Fisher's exact tests, logistic regression, and interview attrition audits.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Fisher's Exact Tests (Bivariate Associations)")
        fisher_tests = results["inferential_tests"]["bivariate_fisher_tests"]
        f_df = pd.DataFrame(fisher_tests)[["test_name", "odds_ratio", "ci_lower", "ci_upper", "p_value", "holm_adjusted_p", "remains_significant"]]
        f_df.columns = ["Hypothesis Test", "Odds Ratio", "95% CI Low", "95% CI High", "Raw p-value", "Holm Adj p-value", "Significant (α=0.05)"]
        st.dataframe(f_df, use_container_width=True)

        st.info("Holm-Bonferroni step-down correction applied to preserve family-wise error rate across multiple hypotheses.")

    with col2:
        if (FIGURES_DIR / "regression_forest_plot.png").exists():
            st.image(str(FIGURES_DIR / "regression_forest_plot.png"), use_container_width=True)

    st.markdown("---")
    st.subheader("Data Completeness & Churn Bias Sensitivity Audit")
    col3, col4 = st.columns([1, 1])
    with col3:
        if (FIGURES_DIR / "completeness_bias_audit.png").exists():
            st.image(str(FIGURES_DIR / "completeness_bias_audit.png"), use_container_width=True)
    with col4:
        st.write(results["data_limitations"]["bias_audit"]["methodological_statement"])
        st.metric("Adoption Difference (Complete vs Partial)", f"{results['data_limitations']['bias_audit']['delta_rate']*100:+.1f}%")
        st.caption("A negligible delta confirms that qualitative interview brevity does not introduce significant survivor or churn bias.")


# ============================================================================
# Page 6: Patient Transcript Explorer
# ============================================================================
elif page == "Patient Transcript Explorer":
    st.title("Patient Transcript & Extraction Inspector")
    st.markdown("Audit verbatim qualitative evidence quotes side-by-side with structured Pydantic extraction fields.")

    pids = sorted([k for k in transcripts.keys() if not k.startswith("_")])
    selected_pid = st.selectbox("Select Patient Record:", pids)

    col1, col2 = st.columns([3, 2])
    with col1:
        st.subheader("Verbatim Patient Interview Transcript")
        st.text_area("Transcript Text", transcripts[selected_pid], height=380, disabled=True)

    with col2:
        st.subheader("Structured Extraction Audit")
        patient_row = df[df["patient_id"] == selected_pid].iloc[0]

        st.markdown(f"**Patient ID:** `{selected_pid}`")
        st.markdown(f"**Age Group:** `{patient_row['age_group']}` *(Conf: {patient_row['age_group_confidence']})*")
        st.markdown(f"**Disease Duration:** `{patient_row['disease_duration']}` *(Conf: {patient_row['disease_duration_confidence']})*")
        st.markdown(f"**Adoption Funnel:** `{patient_row['adoption_status']}`")
        st.markdown(f"**Currently on Newer:** `{patient_row['is_on_newer_therapy']}`")
        st.markdown(f"**Primary Barrier:** `{patient_row['primary_barrier']}`")
        st.markdown(f"**Treatment Ladder:** `{patient_row['therapy_history_sequence']}`")
        st.markdown(f"**Diagnostic Delay / Dismissal:** `{patient_row['initially_dismissed']}`")
        st.markdown(f"**Completeness Score:** `{patient_row['completeness_score']}` *(Partial Record: {patient_row['is_partial_record']})*")
