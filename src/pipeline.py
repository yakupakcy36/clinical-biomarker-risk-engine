"""CLI runner for clinical risk calibration, Brier scoring, & Decision Curve Analysis."""

import argparse
from pathlib import Path
from src.config import DEFAULT_CLINICAL_CONFIG
from src.generator import generate_clinical_cohort
from src.model import train_clinical_risk_engine, ClinicalRiskEngineBundle


def run_pipeline(n_patients: int = 2500, save_artifact: bool = True):
    print("=" * 68)
    print("CLINICAL BIOMARKER RISK ENGINE: PROBABILITY CALIBRATION & DCA")
    print("=" * 68)

    print(f"[1/4] Simulating longitudinal cohort of {n_patients} hospital admissions...")
    cohort_df = generate_clinical_cohort(n_patients=n_patients, seed=42)
    det_rate = (cohort_df["deterioration_30d"].mean()) * 100.0
    print(f"      Cohort 30-day deterioration event rate: {det_rate:.1f}%")

    print("\n[2/4] Fitting Isotonic Calibrated Classifier with 3-Fold Cross-Validation...")
    bundle, metrics = train_clinical_risk_engine(cohort_df, config=DEFAULT_CLINICAL_CONFIG)

    print("\n[3/4] Calibration & Discrimination Benchmark on Held-Out Test Set:")
    print(f"      ROC-AUC (Discrimination)        : {metrics['roc_auc']:.4f}")
    print(f"      PR-AUC (Precision-Recall)       : {metrics['pr_auc']:.4f}")
    print(f"      Brier Score (Reliability)       : {metrics['brier_score']:.4f} (Lower is better)")
    print(f"      Expected Calibration Error (ECE): {metrics['ece']:.4f}")

    print("\n[4/4] Decision Curve Analysis (DCA) Net Benefit Comparison:")
    print("      Threshold (pt) | Net Benefit (Model) | Net Benefit (Treat All) | Recommendation")
    print("      " + "-" * 68)
    for row in metrics["dca_curves"][:6]:
        pt = row["threshold_probability"]
        nb_m = row["net_benefit_model"]
        nb_a = row["net_benefit_treat_all"]
        rec = "Use Model" if nb_m > nb_a and nb_m > 0 else "Treat None / Clinical Review"
        print(f"          {pt:4.2f}       |       {nb_m:6.4f}        |         {nb_a:6.4f}         | {rec}")

    if save_artifact:
        out_path = DEFAULT_CLINICAL_CONFIG.artifacts_dir / DEFAULT_CLINICAL_CONFIG.model_filename
        print(f"\n[Artifact] Saving calibrated model bundle to: {out_path}")
        bundle.save(str(out_path))

    print("\n[SUCCESS] Clinical Risk Engine pipeline finished successfully.")
    return bundle, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clinical Risk Engine CLI")
    parser.add_argument("--patients", type=int, default=2500, help="Cohort size")
    args = parser.parse_args()
    run_pipeline(n_patients=args.patients)
