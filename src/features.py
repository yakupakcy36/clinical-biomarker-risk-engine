"""Clinical feature engineering & physiological composite indices."""

from typing import Tuple, List
import numpy as np
import pandas as pd


class ClinicalFeatureExtractor:
    """Computes established ICU shock indices, log-transformed lab values, and risk scores."""

    def extract_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Builds composite clinical indicators and extracts training matrix."""
        data = df.copy()

        # 1. Shock Index (Heart Rate / Systolic Blood Pressure)
        # Clinical normal: 0.5 - 0.7. Values > 0.9 indicate occult circulatory shock.
        data["shock_index"] = data["heart_rate"] / np.maximum(data["systolic_bp"], 50.0)

        # 2. Hypoxia Burden
        data["hypoxia_burden"] = np.maximum(0.0, 95.0 - data["spo2"])

        # 3. Log-Transformed Non-Linear Biomarkers
        data["log_troponin"] = np.log1p(data["troponin_i"])
        data["log_bnp"] = np.log1p(data["bnp"])
        data["log_crp"] = np.log1p(data["crp"])
        data["log_creatinine"] = np.log1p(data["creatinine"])

        # 4. Cardio-Renal Syndrome & Metabolic Cross-Product Interactions
        data["cardio_renal_stress"] = data["log_troponin"] * data["log_creatinine"]
        data["septic_inflammatory_load"] = data["lactate"] * (data["crp"] / 10.0)

        # 5. Age-Comorbidity Risk Index
        data["age_decade"] = (data["age"] / 10.0).astype(float)
        data["comorbidity_score"] = data["has_diabetes"] + data["has_hypertension"]

        feature_cols = [
            "age",
            "has_diabetes",
            "has_hypertension",
            "systolic_bp",
            "heart_rate",
            "spo2",
            "troponin_i",
            "bnp",
            "creatinine",
            "crp",
            "lactate",
            "shock_index",
            "hypoxia_burden",
            "log_troponin",
            "log_bnp",
            "log_crp",
            "log_creatinine",
            "cardio_renal_stress",
            "septic_inflammatory_load",
            "age_decade",
            "comorbidity_score",
        ]

        X = data[feature_cols].copy()
        y = data["deterioration_30d"] if "deterioration_30d" in data.columns else pd.Series(0, index=data.index)

        return X, y
