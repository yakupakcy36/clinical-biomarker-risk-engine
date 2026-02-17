"""Unit tests for clinical cohort generator."""

import pandas as pd
from src.generator import generate_clinical_cohort


def test_clinical_cohort_generation():
    df = generate_clinical_cohort(n_patients=200, seed=42)
    assert len(df) == 200
    expected_cols = [
        "age", "has_diabetes", "has_hypertension", "systolic_bp",
        "heart_rate", "spo2", "troponin_i", "bnp", "creatinine",
        "crp", "lactate", "deterioration_30d"
    ]
    for c in expected_cols:
        assert c in df.columns

    # Physiological bounds
    assert (df["age"] >= 18).all()
    assert (df["systolic_bp"] > 0).all()
    assert (df["heart_rate"] > 0).all()
    assert (df["spo2"] <= 100.0).all()
    assert (df["troponin_i"] > 0).all()
    assert df["deterioration_30d"].isin([0, 1]).all()
