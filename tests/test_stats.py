"""
Tests for Stage 5 inferential statistics, Wilson intervals, Fisher's exact tests, and funnel math.
"""

import pytest
import numpy as np
import pandas as pd
from scipy import stats

from patient_journey.stats_tests import (
    wilson_score_interval,
    fishers_exact_test_2x2,
    apply_holm_bonferroni,
    fit_adoption_logistic_regression,
)
from patient_journey.analyze import analyze_adoption_funnel, analyze_barriers_among_non_adopters


def test_wilson_ci_known_reference_value():
    """
    Verify Wilson score interval against known standard statistical value:
    For k = 10, n = 50 (p = 0.20), at 95% confidence (z = 1.96),
    Wilson interval lower ~ 0.1124, upper ~ 0.3296.
    """
    p, lower, upper = wilson_score_interval(10, 50, confidence=0.95)
    assert p == 0.20
    assert pytest.approx(lower, abs=0.01) == 0.112
    assert pytest.approx(upper, abs=0.01) == 0.330
    assert lower < p < upper


def test_wilson_ci_boundary_conditions():
    """Verify Wilson interval handles edge cases without dividing by zero."""
    # Zero total
    p0, l0, u0 = wilson_score_interval(0, 0)
    assert p0 == 0.0 and l0 == 0.0 and u0 == 0.0

    # 0 successes out of 20
    p_zero, l_zero, u_zero = wilson_score_interval(0, 20)
    assert p_zero == 0.0
    assert l_zero == 0.0
    assert u_zero > 0.0  # Wilson interval upper bound is non-zero (rule of three)

    # All successes out of 20
    p_all, l_all, u_all = wilson_score_interval(20, 20)
    assert p_all == 1.0
    assert l_all < 1.0
    assert u_all == 1.0


def test_fishers_exact_test_wrapper_basic():
    """Verify Fisher's exact test on known 2x2 contingency table."""
    # Table with significant association
    table = [[15, 2], [3, 14]]
    res = fishers_exact_test_2x2(table, feature_name="Biomarker", outcome_name="Response")

    assert res["odds_ratio"] > 1.0
    assert res["ci_lower"] > 1.0
    assert res["p_value"] < 0.01
    assert res["is_significant"] is True
    assert "statistically significant" in res["interpretation"]


def test_fishers_exact_test_zero_cell_continuity_adjustment():
    """Verify Haldane-Anscombe 0.5 correction prevents division by zero when a cell is 0."""
    table_zero = [[10, 0], [5, 10]]
    res = fishers_exact_test_2x2(table_zero)

    assert not np.isinf(res["odds_ratio"])
    assert not np.isnan(res["odds_ratio"])
    assert res["ci_lower"] > 0
    assert res["ci_upper"] > res["ci_lower"]


def test_holm_bonferroni_adjustment():
    """Verify Holm-Bonferroni step-down threshold calculations and ranking."""
    tests = [
        {"test_name": "Test A", "p_value": 0.04},
        {"test_name": "Test B", "p_value": 0.005},
        {"test_name": "Test C", "p_value": 0.12},
    ]
    adj = apply_holm_bonferroni(tests, alpha=0.05)

    # Smallest p-value (Test B, p=0.005) ranked 1st
    assert adj[0]["test_name"] == "Test B"
    assert adj[0]["rank"] == 1
    assert adj[0]["holm_threshold"] == pytest.approx(0.05 / 3, abs=0.001)
    assert adj[0]["remains_significant"] is True

    # Test C ranked 3rd
    assert adj[2]["test_name"] == "Test C"
    assert adj[2]["remains_significant"] is False


def test_funnel_math_and_dropoffs():
    """Verify funnel calculation logic on a simulated dataframe."""
    df_sim = pd.DataFrame({
        "has_glp1_or_sglt2_discussed": [True, True, True, True, False],  # 4/5
        "has_glp1_or_sglt2_ever": [True, True, True, False, False],      # 3/5
        "is_on_newer_therapy": [True, True, False, False, False],         # 2/5
        "disease_duration": ["1-3 years"] * 5,
        "age_group": ["40-54"] * 5,
    })

    funnel = analyze_adoption_funnel(df_sim)
    assert funnel["funnel"]["discussed"]["count"] == 4
    assert funnel["funnel"]["ever_used"]["count"] == 3
    assert funnel["funnel"]["currently_on"]["count"] == 2
    assert funnel["dropoff"]["discussed_to_ever_dropoff"] == 0.25
    assert funnel["dropoff"]["ever_to_current_dropoff"] == pytest.approx(0.333, abs=0.01)


def test_logistic_regression_execution():
    """Verify multivariate logistic regression runs and produces expected structure."""
    from patient_journey.config import PATIENTS_FLAT_FILE
    if PATIENTS_FLAT_FILE.exists():
        df_real = pd.read_csv(PATIENTS_FLAT_FILE)
        res = fit_adoption_logistic_regression(df_real)
        assert res["model_status"] in ("SUCCESS", "FAILED_OR_SINGULAR")
        assert "caution" in res
        assert res["sample_size"] == len(df_real)
    else:
        np.random.seed(42)
        n = 60
        df_sim = pd.DataFrame({
            "is_on_newer_therapy": np.random.choice([True, False], size=n),
            "age_group": np.random.choice(["18-39", "40-54", "55-64", "65+"], size=n),
            "disease_duration": np.random.choice(["<1 year", "1-3 years", "4-7 years", "8+ years"], size=n),
            "initially_dismissed": np.random.choice([True, False], size=n),
            "all_barriers": np.random.choice(["COST_INSURANCE", "FEAR_SIDE_EFFECTS", "NONE"], size=n),
        })
        res = fit_adoption_logistic_regression(df_sim)
        assert res["model_status"] in ("SUCCESS", "FAILED_OR_SINGULAR")
        assert "caution" in res

