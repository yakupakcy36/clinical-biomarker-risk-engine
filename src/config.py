"""Clinical cohort and risk stratification configuration."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class ClinicalRiskConfig:
    # Risk Stratification Probability Tiers
    low_risk_threshold: float = 0.15
    high_risk_threshold: float = 0.35

    # Decision Curve Analysis (DCA) evaluation grid
    dca_thresholds: tuple = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)

    # Calibration hyperparameters
    calibration_method: str = "isotonic"  # 'isotonic' or 'sigmoid'
    n_bins_ece: int = 10  # Expected Calibration Error bins

    # Storage paths
    artifacts_dir: Path = Path("artifacts")
    model_filename: str = "clinical_risk_engine_bundle.joblib"


DEFAULT_CLINICAL_CONFIG = ClinicalRiskConfig()
