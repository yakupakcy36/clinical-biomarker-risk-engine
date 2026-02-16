"""Pydantic schemas for the Clinical Biomarker Risk API."""

from typing import List, Dict
from pydantic import BaseModel, Field


class PatientVitalsRequest(BaseModel):
    age: int = Field(default=65, ge=18, le=105, description="Patient age in years")
    has_diabetes: int = Field(default=0, ge=0, le=1)
    has_hypertension: int = Field(default=1, ge=0, le=1)
    systolic_bp: float = Field(default=120.0, ge=50.0, le=240.0, description="Systolic Blood Pressure (mmHg)")
    heart_rate: float = Field(default=75.0, ge=30.0, le=220.0, description="Heart Rate (bpm)")
    spo2: float = Field(default=96.0, ge=60.0, le=100.0, description="Oxygen Saturation SpO2 (%)")
    troponin_i: float = Field(default=0.02, ge=0.001, le=50.0, description="Cardiac Troponin I (ng/mL)")
    bnp: float = Field(default=80.0, ge=5.0, le=10000.0, description="B-Type Natriuretic Peptide (pg/mL)")
    creatinine: float = Field(default=1.0, ge=0.2, le=15.0, description="Serum Creatinine (mg/dL)")
    crp: float = Field(default=4.0, ge=0.1, le=400.0, description="C-Reactive Protein (mg/L)")
    lactate: float = Field(default=1.2, ge=0.2, le=20.0, description="Serum Lactate (mmol/L)")


class PatientRiskResponse(BaseModel):
    predicted_risk_pct: float
    risk_tier: str
    recommended_clinical_action: str
    primary_risk_drivers: List[str]
    brier_reliability_score: float


class DcaPoint(BaseModel):
    threshold_probability: float
    net_benefit_model: float
    net_benefit_treat_all: float
    net_benefit_treat_none: float = 0.0


class DcaEvaluationResponse(BaseModel):
    test_roc_auc: float
    test_brier_score: float
    test_ece: float
    decision_curves: List[DcaPoint]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    calibration_method: str
    brier_score: float
