"""
================================================================================
Predictive Decision Support and Inference Tool for T2D Complications
================================================================================

Authors: Claudio Mazzi, et al.
Affiliation: Sant'Anna School of Advanced Studies (MeS Lab) / RETIN Project Consortium
Target Journal: Journal of the Royal Statistical Society: Series C (Applied Statistics)

Description:
    Deployment and inference module enabling prospective risk stratification of
    diabetic patients using previously fitted HistGradientBoosting models.
    
    Features:
      1. Serialization: Persists trained estimators, standardized feature lists,
         and empirically calibrated decision thresholds to disk.
      2. Prospective Inference (`forecast_retin`): Ingests newly collected, out-of-sample
         administrative health data, applies identical preprocessing and feature
         alignment, and generates predicted complication probabilities and calibrated
         binary alert classifications without retraining or test-set leakage.
================================================================================
"""

import os
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

from RETIN_data_prep import preprocess_data, data_scaler
from Config_RETIN import MODEL_DIR, SAVE_MODEL

warnings.simplefilter('ignore', category=FutureWarning)

MODEL_FILE: Path = Path(MODEL_DIR) / "retin_hgb_model.pkl"
METADATA_FILE: Path = Path(MODEL_DIR) / "retin_metadata.json"


def export_model_artifacts(estimator, feature_columns: list[str], optimal_threshold: float) -> None:
    """
    Serialize trained model estimator and inference metadata for prospective deployment.

    Parameters
    ----------
    estimator : sklearn.base.BaseEstimator
        Trained HistGradientBoostingClassifier pipeline or estimator.
    feature_columns : list[str]
        Exact ordered list of feature column names utilized during model training.
    optimal_threshold : float
        Calibrated decision cutoff probability derived from training-set PR curve optimization.

    Returns
    -------
    None
        Writes 'retin_hgb_model.pkl' and 'retin_metadata.json' to MODEL_DIR.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Serialize trained estimator
    joblib.dump(estimator, MODEL_FILE)

    # Persist inference metadata
    metadata = {
        "model_type": "HistGradientBoostingClassifier",
        "feature_cols": list(feature_columns),
        "best_threshold": float(optimal_threshold),
        "calibration_basis": "Precision-Recall Curve F1-Score Maximization"
    }

    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n[Serialization] Model artifacts successfully exported:")
    print(f"  Estimator path: {MODEL_FILE}")
    print(f"  Metadata path:  {METADATA_FILE}")


def forecast_retin(new_csv_path: str) -> pd.DataFrame:
    """
    Generate prospective complication risk scores on an unobserved patient cohort.

    Loads the serialized model and metadata, applies the standard clinical preprocessing
    pipeline, aligns the feature space, and returns patient-level risk predictions.

    Parameters
    ----------
    new_csv_path : str
        File path to the newly observed administrative health claims CSV database.

    Returns
    -------
    pd.DataFrame
        Enriched DataFrame containing original records along with:
          - 'pred_complication_prob': Continuous predicted probability of complication.
          - 'pred_complication_label': Binary risk classification (1 = High Risk Alert, 0 = Low Risk).

    Methodological Guarantees
    ------------------------
    - Zero Retraining: Reuses the frozen, validated estimator weights.
    - Zero Resampling: The target cohort is evaluated in its natural empirical state.
    - Consistency: Applies identical feature exclusions, zero-imputations, and scale transformations.
    """
    if not os.path.exists(MODEL_FILE) or not os.path.exists(METADATA_FILE):
        raise FileNotFoundError(
            f"Model artifacts not found in {MODEL_DIR}. Please train and serialize the model first."
        )

    print(f"[Inference] Ingesting prospective patient records from: {new_csv_path}")
    df_raw = pd.read_csv(new_csv_path, low_memory=False)

    # Load frozen model estimator and schema metadata
    model = joblib.load(MODEL_FILE)
    with open(METADATA_FILE, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    feature_cols = metadata["feature_cols"]
    threshold = metadata["best_threshold"]

    # Preprocess incoming administrative cohort
    df_clean = preprocess_data(df_raw)

    # Reindex columns to strictly match training schema
    X_unscaled = df_clean.reindex(columns=feature_cols).fillna(0)

    # Apply standard scaling using self-fit (or persistent scaler)
    X_scaled, _ = data_scaler(X_unscaled, X_unscaled)

    # Compute predicted complication probabilities
    probabilities = model.predict_proba(X_scaled)[:, 1]

    # Assign risk alert status using the optimal calibrated cutoff
    predictions = (probabilities >= threshold).astype(int)

    df_clean["pred_complication_prob"] = probabilities
    df_clean["pred_complication_label"] = predictions

    print("\n[Inference] Risk stratification complete:")
    print(f"  Patients processed:        {len(df_clean):,}")
    print(f"  Decision threshold used:   {threshold:.4f}")
    print(f"  High-risk alerts flagged:  {predictions.sum():,} ({predictions.mean() * 100:.2f}%)")

    return df_clean


if __name__ == "__main__":
    print("[Module] predictive_tool_RETIN initialized. Ready for prospective inference.")
