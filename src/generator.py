"""Realistic synthetic clinical cohort and longitudinal biomarker generator."""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
import pandas as pd


def generate_clinical_cohort(
    n_patients: int = 2500,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generates synthetic patient Electronic Health Record (EHR) episodes with
    physiologically correlated laboratory biomarkers, hemodynamic vitals, and 30-day outcomes.
    """
    np.random.seed(seed)

    # 1. Demographics & Comorbidities
    age = np.random.normal(63.0, 14.0, size=n_patients)
    age = np.clip(age, 22.0, 92.0).astype(int)

    has_diabetes = np.random.binomial(1, 0.28, size=n_patients)
    has_hypertension = np.random.binomial(1, 0.45, size=n_patients)

    # 2. Hemodynamic Vital Signs
    # Systolic BP (lower in shock, higher in hypertension)
    sbp_base = 125.0 + (12.0 * has_hypertension)
    systolic_bp = np.clip(np.random.normal(sbp_base, 18.0), 75.0, 205.0)

    # Heart Rate (higher in distress / sepsis)
    heart_rate = np.clip(np.random.normal(78.0, 15.0), 45.0, 148.0)

    # Pulse Oximetry SpO2 (%)
    spo2 = np.clip(100.0 - np.random.exponential(2.8, size=n_patients), 82.0, 100.0)

    # 3. Laboratory Biomarkers (Right-skewed log-normal distributions)
    # Troponin I (ng/mL) - myocardial infarction / necrosis
    troponin_i = np.random.lognormal(mean=-2.8, sigma=1.1, size=n_patients)
    troponin_i = np.clip(troponin_i, 0.01, 15.0)

    # BNP (pg/mL) - ventricular wall stress / congestive heart failure
    bnp = np.random.lognormal(mean=4.2, sigma=1.2, size=n_patients)
    bnp = np.clip(bnp, 15.0, 4200.0)

    # Serum Creatinine (mg/dL) - renal impairment
    creatinine = np.random.lognormal(mean=0.05, sigma=0.45, size=n_patients)
    creatinine = np.clip(creatinine, 0.5, 6.8)

    # C-Reactive Protein (mg/L) - acute inflammation
    crp = np.random.lognormal(mean=1.8, sigma=1.2, size=n_patients)
    crp = np.clip(crp, 0.5, 180.0)

    # Serum Lactate (mmol/L) - anaerobic cellular metabolism / septic shock
    lactate = np.random.lognormal(mean=0.35, sigma=0.55, size=n_patients)
    lactate = np.clip(lactate, 0.6, 9.5)

    # 4. Latent Physiological Hazard Log-Odds Link
    # Realistic multivariable clinical scoring function
    z = (
        -3.5
        + 0.045 * (age - 60.0)
        + 0.85 * np.log1p(troponin_i)
        + 0.55 * np.log1p(bnp / 100.0)
        + 0.75 * np.log1p(creatinine)
        + 0.50 * np.log1p(crp / 10.0)
        + 1.15 * np.maximum(0.0, lactate - 2.0)
        + 0.050 * np.maximum(0.0, 92.0 - spo2)
        + 0.040 * np.maximum(0.0, 90.0 - systolic_bp)
        + 0.45 * has_diabetes
        + 0.35 * has_hypertension
    )

    # True latent probability via logistic sigmoid
    true_prob = 1.0 / (1.0 + np.exp(-z))
    deterioration_outcome = np.random.binomial(1, true_prob)

    df = pd.DataFrame(
        {
            "age": age,
            "has_diabetes": has_diabetes,
            "has_hypertension": has_hypertension,
            "systolic_bp": np.round(systolic_bp, 1),
            "heart_rate": np.round(heart_rate, 1),
            "spo2": np.round(spo2, 1),
            "troponin_i": np.round(troponin_i, 3),
            "bnp": np.round(bnp, 1),
            "creatinine": np.round(creatinine, 2),
            "crp": np.round(crp, 2),
            "lactate": np.round(lactate, 2),
            "deterioration_30d": deterioration_outcome,
        }
    )
    return df
