"""Unit tests for clinical feature extractor."""

import numpy as np
from src.generator import generate_clinical_cohort
from src.features import ClinicalFeatureExtractor


def test_clinical_feature_extraction():
    df = generate_clinical_cohort(n_patients=100, seed=42)
    extractor = ClinicalFeatureExtractor()
    X, y = extractor.extract_features(df)

    assert len(X) == 100
    assert len(y) == 100
    assert "shock_index" in X.columns
    assert "cardio_renal_stress" in X.columns
    assert "hypoxia_burden" in X.columns
    assert not X.isna().any().any()
    # Shock index must be positive
    assert (X["shock_index"] > 0).all()
