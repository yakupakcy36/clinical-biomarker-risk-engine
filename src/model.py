"""Calibrated Clinical Risk Classifier, Decision Curve Analysis (DCA), & ECE Metrics."""

import os
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler

from src.config import ClinicalRiskConfig, DEFAULT_CLINICAL_CONFIG
from src.features import ClinicalFeatureExtractor


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE) across partitioned probability bins."""
    bin_limits = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        low, high = bin_limits[i], bin_limits[i + 1]
        mask = (y_prob >= low) & (y_prob < high) if i < n_bins - 1 else (y_prob >= low) & (y_prob <= high)
        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            ece += (bin_count / n) * abs(bin_acc - bin_conf)

    return float(ece)


def compute_decision_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: List[float],
) -> List[Dict[str, float]]:
    """
    Computes Net Clinical Benefit across decision threshold probabilities:
    Net Benefit = (TP / N) - (FP / N) * (p_t / (1 - p_t))
    """
    n = len(y_true)
    prevalence = float(np.mean(y_true))
    curves: List[Dict[str, float]] = []

    for pt in thresholds:
        tp = np.sum((y_prob >= pt) & (y_true == 1))
        fp = np.sum((y_prob >= pt) & (y_true == 0))
        weight = pt / (1.0 - pt)

        nb_model = (tp / n) - (fp / n) * weight
        nb_all = prevalence - (1.0 - prevalence) * weight
        nb_none = 0.0

        curves.append(
            {
                "threshold_probability": round(pt, 3),
                "net_benefit_model": round(float(nb_model), 4),
                "net_benefit_treat_all": round(float(nb_all), 4),
                "net_benefit_treat_none": 0.0,
            }
        )

    return curves


@dataclass
class PatientRiskAssessment:
    predicted_risk_pct: float
    risk_tier: str  # LOW, MODERATE, HIGH
    recommended_clinical_action: str
    primary_risk_drivers: List[str]
    brier_reliability_score: float


class ClinicalRiskEngineBundle:
    """Calibrated probability risk engine with Decision Curve Analysis verification."""

    def __init__(
        self,
        scaler: StandardScaler,
        calibrated_model: CalibratedClassifierCV,
        feature_names: List[str],
        test_brier_score: float,
        test_ece: float,
        test_roc_auc: float,
        dca_curves: List[Dict[str, float]],
        config: ClinicalRiskConfig = DEFAULT_CLINICAL_CONFIG,
    ):
        self.scaler = scaler
        self.calibrated_model = calibrated_model
        self.feature_names = feature_names
        self.test_brier_score = test_brier_score
        self.test_ece = test_ece
        self.test_roc_auc = test_roc_auc
        self.dca_curves = dca_curves
        self.config = config

    def predict_patient_risk(self, patient_df: pd.DataFrame) -> PatientRiskAssessment:
        """Assesses calibrated deterioration risk and identifies primary biomarker drivers."""
        extractor = ClinicalFeatureExtractor()
        X, _ = extractor.extract_features(patient_df)
        X_aligned = X[self.feature_names]
        X_scaled = self.scaler.transform(X_aligned)

        # Predict calibrated probability
        prob = float(self.calibrated_model.predict_proba(X_scaled)[0, 1])

        # Stratify risk tier
        if prob < self.config.low_risk_threshold:
            tier = "LOW"
            action = "Routine ward observation; schedule standard 12-hour repeat vitals."
        elif prob < self.config.high_risk_threshold:
            tier = "MODERATE"
            action = "Continuous telemetry monitoring; initiate proactive hemodynamic and lab re-check in 4 hours."
        else:
            tier = "HIGH"
            action = "Urgent Rapid Response Team (RRT) alert; consider immediate ICU transfer and arterial line placement."

        # Identify primary physiological drivers
        drivers: List[str] = []
        p = patient_df.iloc[0]
        if p.get("lactate", 0.0) >= 2.5:
            drivers.append(f"Elevated Serum Lactate ({p['lactate']:.2f} mmol/L) indicating cellular hypoperfusion")
        if p.get("troponin_i", 0.0) >= 0.08:
            drivers.append(f"Myocardial injury elevation: Troponin I ({p['troponin_i']:.3f} ng/mL)")
        if p.get("spo2", 100.0) <= 90.0:
            drivers.append(f"Severe refractory hypoxia: SpO2 ({p['spo2']:.1f}%)")
        if p.get("systolic_bp", 120.0) <= 90.0 or (p.get("heart_rate", 70.0) / max(p.get("systolic_bp", 1.0), 1.0)) >= 0.9:
            drivers.append(f"Hypotensive / Circulatory shock index ({p.get('heart_rate', 70) / p.get('systolic_bp', 120):.2f})")
        if p.get("creatinine", 1.0) >= 2.0:
            drivers.append(f"Acute kidney injury: Creatinine ({p['creatinine']:.2f} mg/dL)")

        if not drivers:
            drivers.append("Cumulative multi-biomarker age and comorbid baseline factors")

        return PatientRiskAssessment(
            predicted_risk_pct=round(prob * 100.0, 1),
            risk_tier=tier,
            recommended_clinical_action=action,
            primary_risk_drivers=drivers,
            brier_reliability_score=round(self.test_brier_score, 4),
        )

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> "ClinicalRiskEngineBundle":
        return joblib.load(filepath)


def train_clinical_risk_engine(
    cohort_df: pd.DataFrame,
    config: ClinicalRiskConfig = DEFAULT_CLINICAL_CONFIG,
) -> Tuple[ClinicalRiskEngineBundle, Dict[str, Any]]:
    """Trains calibrated risk model and computes Brier, ECE, and DCA Net Benefit metrics."""
    extractor = ClinicalFeatureExtractor()
    X, y = extractor.extract_features(cohort_df)

    # 80/20 train/test split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx].to_numpy(), y.iloc[split_idx:].to_numpy()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Base Logistic Estimator with L2 regularization
    base_clf = LogisticRegression(C=0.75, max_iter=1000, random_state=42)

    # Isotonic Probability Calibration with 3-fold cross validation
    calibrated_clf = CalibratedClassifierCV(
        estimator=base_clf,
        method=config.calibration_method,
        cv=3,
    )
    calibrated_clf.fit(X_train_scaled, y_train)

    # Evaluation on held-out test cohort
    probs_test = calibrated_clf.predict_proba(X_test_scaled)[:, 1]
    brier = float(brier_score_loss(y_test, probs_test))
    ece = compute_ece(y_test, probs_test, n_bins=config.n_bins_ece)
    roc_auc = float(roc_auc_score(y_test, probs_test))
    pr_auc = float(average_precision_score(y_test, probs_test))

    # Decision Curve Analysis
    dca = compute_decision_curves(y_test, probs_test, list(config.dca_thresholds))

    bundle = ClinicalRiskEngineBundle(
        scaler=scaler,
        calibrated_model=calibrated_clf,
        feature_names=list(X.columns),
        test_brier_score=brier,
        test_ece=ece,
        test_roc_auc=roc_auc,
        dca_curves=dca,
        config=config,
    )

    metrics = {
        "brier_score": round(brier, 4),
        "ece": round(ece, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "dca_curves": dca,
    }
    return bundle, metrics
