"""
================================================================================
Empirical Benchmarking: Non-Linear Boosting vs. Penalized Logistic Baselines
================================================================================

Authors: Claudio Mazzi, et al.
Affiliation: Sant'Anna School of Advanced Studies (MeS Lab) / RETIN Project Consortium
Target Journal: Journal of the Royal Statistical Society: Series C (Applied Statistics)

Description:
    This script implements the empirical benchmark comparison between the proposed
    HistGradientBoostingClassifier architecture and standard penalized linear
    classifiers (L1 Lasso, L2 Ridge, with and without training-stage resampling).
    
    Evaluated Classifiers on Holdout Test Cohort:
      1. Logistic Regression (L2 Ridge, balanced class weights, unaugmented train)
      2. Logistic Regression (L2 Ridge, trained on ADASYN + Tomek resampled train)
      3. Logistic Regression (L1 Lasso, trained on ADASYN + Tomek resampled train)
      4. HistGradientBoostingClassifier (proposed non-linear ensemble)
      
    Reported Metrics:
      - Area Under the Precision-Recall Curve (PR-AUC / Average Precision)
      - Area Under the Receiver Operating Characteristic Curve (ROC-AUC)
      - Brier Score (Mean squared probabilistic calibration error)
      - Calibrated Optimal Threshold (maximizing F1-score on the PR curve)
      - Sensitivity / Recall for the complication class (COMP = 1)
      - Positive Predictive Value / Precision for the complication class
      - F1-Score and Balanced Accuracy
================================================================================
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, precision_recall_curve, f1_score,
    recall_score, precision_score
)
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import TomekLinks

from RETIN_data_prep import preprocess_data, data_scaler


def run_benchmark(db_filename: str = "DB_ML_TY.csv", output_csv: str = "benchmark_tuscany_baseline.csv") -> pd.DataFrame:
    """
    Execute out-of-sample benchmark comparison across linear and boosting architectures.

    Parameters
    ----------
    db_filename : str, default="DB_ML_TY.csv"
        Path to the raw regional cohort dataset (e.g., Tuscany).
    output_csv : str, default="benchmark_tuscany_baseline.csv"
        Destination filename for persisting benchmark comparison metrics.

    Returns
    -------
    pd.DataFrame
        Comparative evaluation metrics table across all candidate architectures.
    """
    # Locate dataset within current directory or project root
    candidate_paths = [
        Path(db_filename),
        Path(__file__).resolve().parent.parent / db_filename,
        Path(__file__).resolve().parent / db_filename
    ]
    db_path = None
    for p in candidate_paths:
        if p.exists():
            db_path = p
            break

    if db_path is None:
        raise FileNotFoundError(f"Source cohort database '{db_filename}' not found.")

    print("=" * 80)
    print(f"[Benchmark] Loading cohort records from: {db_path}")
    df_raw = pd.read_csv(db_path, low_memory=False)

    # Preprocess raw records and construct binary complication target (COMP)
    df = preprocess_data(df_raw)
    X = df.drop("COMP", axis=1)
    y = df["COMP"]

    print(f"[Benchmark] Preprocessed sample size: N = {X.shape[0]:,}, Features: P = {X.shape[1]}")
    print(f"[Benchmark] Empirical complication prevalence: {y.mean() * 100:.2f}%")

    # --------------------------------------------------------------------------
    # 1. 70/30 STRATIFIED TRAIN / HOLDOUT TEST SPLIT
    # --------------------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_test = X_test.reindex(columns=X_train.columns)

    # Standardize continuous features while preserving binary flags
    X_train_scaled, X_test_scaled = data_scaler(X_train, X_test)

    # --------------------------------------------------------------------------
    # 2. TWO-STAGE RESAMPLING (TRAINING SET ONLY)
    # --------------------------------------------------------------------------
    print("\n[Benchmark] Applying ADASYN + Tomek Links resampling strictly to training partition...")
    resampling_pipeline = Pipeline([
        ('adasyn', ADASYN(sampling_strategy='minority', random_state=42)),
        ('tomek', TomekLinks(sampling_strategy='auto'))
    ])
    X_train_res, y_train_res = resampling_pipeline.fit_resample(X_train_scaled, y_train)

    print(f"  Training cohort prior to resampling: N = {X_train.shape[0]:,}")
    print(f"  Training cohort after resampling:    N = {X_train_res.shape[0]:,}")
    print(f"  Holdout test cohort (unaugmented):   N = {X_test.shape[0]:,}")

    # --------------------------------------------------------------------------
    # 3. MODEL TRAINING AND PREDICTION
    # --------------------------------------------------------------------------
    predicted_probabilities = {}

    # Model 1: Penalized Logistic Regression (L2 Ridge, balanced class weights, unaugmented)
    print("\n[1/4] Fitting Logistic Regression (L2 Ridge, balanced weights, unaugmented train)...")
    lr_balanced = LogisticRegression(
        penalty='l2', C=1.0, max_iter=1000, random_state=42, class_weight='balanced'
    )
    lr_balanced.fit(X_train_scaled, y_train)
    predicted_probabilities['LogReg (Ridge, balanced weights)'] = lr_balanced.predict_proba(X_test_scaled)[:, 1]

    # Model 2: Penalized Logistic Regression (L2 Ridge, on ADASYN+Tomek resampled train)
    print("[2/4] Fitting Logistic Regression (L2 Ridge, on ADASYN+Tomek train)...")
    lr_res = LogisticRegression(penalty='l2', C=1.0, max_iter=1000, random_state=42)
    lr_res.fit(X_train_res, y_train_res)
    predicted_probabilities['LogReg (Ridge, resampled train)'] = lr_res.predict_proba(X_test_scaled)[:, 1]

    # Model 3: Penalized Logistic Regression (L1 Lasso, on ADASYN+Tomek resampled train)
    print("[3/4] Fitting Logistic Regression (L1 Lasso, on ADASYN+Tomek train)...")
    lr_lasso = LogisticRegression(penalty='l1', solver='saga', C=0.5, max_iter=1000, random_state=42)
    lr_lasso.fit(X_train_res, y_train_res)
    predicted_probabilities['LogReg (Lasso, resampled train)'] = lr_lasso.predict_proba(X_test_scaled)[:, 1]

    # Model 4: HistGradientBoostingClassifier (proposed non-linear architecture)
    print("[4/4] Fitting HistGradientBoostingClassifier (proposed ensemble)...")
    hgb = HistGradientBoostingClassifier(
        learning_rate=0.2, max_depth=7, max_iter=300, random_state=42, class_weight='balanced'
    )
    hgb.fit(X_train_res, y_train_res)
    predicted_probabilities['HistGradientBoosting (Paper)'] = hgb.predict_proba(X_test_scaled)[:, 1]

    # --------------------------------------------------------------------------
    # 4. PERFORMANCE EVALUATION ACROSS METRICS
    # --------------------------------------------------------------------------
    benchmark_metrics = {}

    for model_name, probs in predicted_probabilities.items():
        prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_test, probs)
        f1_arr = 2.0 * (prec_arr * rec_arr) / (prec_arr + rec_arr + 1e-8)
        best_idx = np.argmax(f1_arr[:-1])
        optimal_threshold = thresh_arr[best_idx]
        binary_predictions = (probs >= optimal_threshold).astype(int)

        benchmark_metrics[model_name] = {
            'ROC-AUC': roc_auc_score(y_test, probs),
            'PR-AUC (AP)': average_precision_score(y_test, probs),
            'Brier Score': brier_score_loss(y_test, probs),
            'Optimal Threshold': optimal_threshold,
            'F1-Score (COMP=1)': f1_score(y_test, binary_predictions),
            'Recall (COMP=1)': recall_score(y_test, binary_predictions),
            'Precision (COMP=1)': precision_score(y_test, binary_predictions),
            'Balanced Accuracy': balanced_accuracy_score(y_test, binary_predictions)
        }

    results_df = pd.DataFrame(benchmark_metrics).T
    print("\n" + "=" * 80)
    print("BENCHMARK COMPARISON RESULTS (HOLDOUT TEST COHORT)")
    print("=" * 80)
    print(results_df.round(4).to_string())

    results_df.to_csv(output_csv)
    print(f"\n[Benchmark] Results exported successfully to: {output_csv}")

    return results_df


if __name__ == "__main__":
    run_benchmark()
