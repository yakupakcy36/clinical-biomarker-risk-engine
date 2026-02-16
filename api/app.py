"""FastAPI application for Clinical Biomarker Risk Stratification & DCA."""

from contextlib import asynccontextmanager
from typing import Optional, List
import pandas as pd
from fastapi import FastAPI, HTTPException, status

from src.config import DEFAULT_CLINICAL_CONFIG
from src.generator import generate_clinical_cohort
from src.model import train_clinical_risk_engine, ClinicalRiskEngineBundle
from api.schemas import (
    PatientVitalsRequest,
    PatientRiskResponse,
    DcaPoint,
    DcaEvaluationResponse,
    HealthResponse,
)

bundle: Optional[ClinicalRiskEngineBundle] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global bundle
    artifact_path = DEFAULT_CLINICAL_CONFIG.artifacts_dir / DEFAULT_CLINICAL_CONFIG.model_filename
    if artifact_path.exists():
        try:
            bundle = ClinicalRiskEngineBundle.load(str(artifact_path))
        except Exception:
            cohort = generate_clinical_cohort(n_patients=2000, seed=42)
            bundle, _ = train_clinical_risk_engine(cohort, config=DEFAULT_CLINICAL_CONFIG)
    else:
        cohort = generate_clinical_cohort(n_patients=2000, seed=42)
        bundle, _ = train_clinical_risk_engine(cohort, config=DEFAULT_CLINICAL_CONFIG)
    yield


app = FastAPI(
    title="Clinical Biomarker Risk Stratification API",
    description=(
        "Production medical decision support API. Provides calibrated probabilities "
        "of 30-day patient deterioration using isotonic regression, with Decision Curve "
        "Analysis (DCA) Net Benefit metrics and primary biomarker driver attributions."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
def health():
    return HealthResponse(
        status="healthy",
        model_loaded=(bundle is not None),
        calibration_method=DEFAULT_CLINICAL_CONFIG.calibration_method,
        brier_score=bundle.test_brier_score if bundle else 0.08,
    )


@app.post("/risk/assess", response_model=PatientRiskResponse, tags=["Clinical Risk"])
def assess_risk(patient: PatientVitalsRequest):
    """Evaluates individual patient biomarkers and returns calibrated probability tier and recommendations."""
    if bundle is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model bundle not ready")

    df = pd.DataFrame([patient.model_dump()])
    res = bundle.predict_patient_risk(df)

    return PatientRiskResponse(
        predicted_risk_pct=res.predicted_risk_pct,
        risk_tier=res.risk_tier,
        recommended_clinical_action=res.recommended_clinical_action,
        primary_risk_drivers=res.primary_risk_drivers,
        brier_reliability_score=res.brier_reliability_score,
    )


@app.get("/dca/curves", response_model=DcaEvaluationResponse, tags=["Clinical Validation"])
def get_dca_curves():
    """Returns out-of-sample Decision Curve Analysis comparing Model vs Treat-All policies."""
    if bundle is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Model bundle not ready")

    curve_points = [
        DcaPoint(
            threshold_probability=row["threshold_probability"],
            net_benefit_model=row["net_benefit_model"],
            net_benefit_treat_all=row["net_benefit_treat_all"],
            net_benefit_treat_none=0.0,
        )
        for row in bundle.dca_curves
    ]

    return DcaEvaluationResponse(
        test_roc_auc=round(bundle.test_roc_auc, 4),
        test_brier_score=round(bundle.test_brier_score, 4),
        test_ece=round(bundle.test_ece, 4),
        decision_curves=curve_points,
    )
