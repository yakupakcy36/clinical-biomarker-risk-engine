"""Unit tests for calibration metrics, ECE, DCA, & clinical model."""

import tempfile
from pathlib import Path
import numpy as np
import pytest
from src.generator import generate_clinical_cohort
from src.model import (
    compute_ece,
    compute_decision_curves,
    train_clinical_risk_engine,
    ClinicalRiskEngineBundle,
)


def test_ece_computation():
    # Perfectly calibrated dummy predictions
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.05, 0.05, 0.95, 0.95])
    ece = compute_ece(y_true, y_prob, n_bins=5)
    assert ece < 0.10


def test_dca_computation():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9, 0.1, 0.7])
    curves = compute_decision_curves(y_true, y_prob, [0.1, 0.3, 0.5])
    assert len(curves) == 3
    for c in curves:
        assert "net_benefit_model" in c
        assert "net_benefit_treat_all" in c


def test_model_training_and_serialization():
    df = generate_clinical_cohort(n_patients=800, seed=42)
    bundle, metrics = train_clinical_risk_engine(df)

    assert metrics["brier_score"] < 0.25
    assert metrics["roc_auc"] > 0.60
    assert metrics["ece"] < 0.15

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "bundle.joblib"
        bundle.save(str(path))
        loaded = ClinicalRiskEngineBundle.load(str(path))
        assert loaded.test_brier_score == bundle.test_brier_score
