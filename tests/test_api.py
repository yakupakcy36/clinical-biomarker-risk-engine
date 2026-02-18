"""Integration tests for Clinical Risk API endpoints."""

import pytest
from fastapi.testclient import TestClient
from api.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["brier_score"] > 0


def test_risk_assess_low_risk(client):
    payload = {
        "age": 35,
        "has_diabetes": 0,
        "has_hypertension": 0,
        "systolic_bp": 120.0,
        "heart_rate": 70.0,
        "spo2": 99.0,
        "troponin_i": 0.01,
        "bnp": 35.0,
        "creatinine": 0.8,
        "crp": 1.2,
        "lactate": 0.9
    }
    response = client.post("/risk/assess", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["risk_tier"] in ["LOW", "MODERATE"]
    assert data["predicted_risk_pct"] < 35.0


def test_risk_assess_high_risk(client):
    payload = {
        "age": 82,
        "has_diabetes": 1,
        "has_hypertension": 1,
        "systolic_bp": 80.0,
        "heart_rate": 135.0,
        "spo2": 85.0,
        "troponin_i": 4.5,
        "bnp": 2500.0,
        "creatinine": 4.2,
        "crp": 120.0,
        "lactate": 5.8
    }
    response = client.post("/risk/assess", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["risk_tier"] in ["MODERATE", "HIGH"]
    assert any("Lactate" in d or "Troponin" in d or "shock" in d.lower() for d in data["primary_risk_drivers"])


def test_risk_assess_batch(client):
    payload = {
        "patients": [
            {
                "patient_id": "WARD-01",
                "age": 40,
                "has_diabetes": 0,
                "has_hypertension": 0,
                "systolic_bp": 120.0,
                "heart_rate": 72.0,
                "spo2": 99.0,
                "troponin_i": 0.01,
                "bnp": 30.0,
                "creatinine": 0.8,
                "crp": 1.0,
                "lactate": 0.9
            },
            {
                "patient_id": "WARD-02",
                "age": 85,
                "has_diabetes": 1,
                "has_hypertension": 1,
                "systolic_bp": 82.0,
                "heart_rate": 130.0,
                "spo2": 84.0,
                "troponin_i": 3.8,
                "bnp": 2100.0,
                "creatinine": 4.1,
                "crp": 110.0,
                "lactate": 5.4
            }
        ]
    }
    response = client.post("/risk/assess/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert data[0]["patient_id"] == "WARD-01"
    assert data[1]["patient_id"] == "WARD-02"


def test_dca_curves_endpoint(client):
    response = client.get("/dca/curves")
    assert response.status_code == 200
    data = response.json()
    assert data["test_roc_auc"] > 0.60
    assert len(data["decision_curves"]) > 0
