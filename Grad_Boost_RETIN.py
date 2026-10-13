"""
================================================================================
HistGradientBoosting and Local Explainability Pipeline for T2D Complications
================================================================================

Authors: Claudio Mazzi, et al.
Affiliation: Sant'Anna School of Advanced Studies (MeS Lab) / RETIN Project Consortium
Target Journal: Journal of the Royal Statistical Society: Series C (Applied Statistics)

Description:
    Core scientific training, validation, and interpretability pipeline for predicting
    multi-year diabetes complications using administrative health claims.
    
    Analytical Steps:
      1. Cohort ingestion, sex stratification, and pre-event feature preprocessing.
      2. 70/30 stratified train-test partitioning ensuring strict holdout test integrity.
      3. Training-stage two-step resampling: Adaptive Synthetic oversampling (ADASYN)
         followed by Tomek Links undersampling (applied strictly to training data).
      4. Nested 5-fold Cross-Validation and hyperparameter optimization for
         HistGradientBoostingClassifier with O(P * B) split evaluation complexity.
      5. Optimal classification decision threshold tuning via Precision-Recall curve
         maximization of the F1-score.
      6. Comprehensive out-of-sample evaluation: PR-AUC, ROC-AUC, Balanced Accuracy,
         Brier score, Sensitivity/Recall, and Precision.
      7. Non-parametric bootstrap estimation of PR-AUC sampling variance (v_i) on
         the logit scale for subsequent random-effects meta-analytic pooling.
      8. Local explainability via SHAP KernelExplainer with k-means background summaries.
      9. Non-parametric hypothesis testing (two-sided Mann-Whitney U) on SHAP distributions
         and Spearman rank correlation analysis between predictor levels and attributions.
     10. Regional summary export for cross-regional meta-analytic synthesis.
================================================================================
"""

import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    classification_report, balanced_accuracy_score, precision_recall_curve,
    roc_auc_score, average_precision_score, brier_score_loss, f1_score
)
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import TomekLinks
from imblearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from scipy.stats import mannwhitneyu, spearmanr

# Internal pipeline imports
from RETIN_data_prep import preprocess_data, eda_func, data_scaler
from Config_RETIN import (
    DB_DIR, DB_DIR_TEST, filter_block, eda_block, balance_block, HistGrad_block,
    BEST_threshold, PERM_importance_block, SHAP_block, plot_block, SUFFIX,
    SHAP_PLOT, SHAP_VIOLIN, meta_eval, SEL_REGION, META, meta_pool, SHAP_VALUE,
    SHAP_META, shap_meta_block, shap_meta_block_pool, SEX_flt,
    spearman_block, shap_violin_block, shap_nnparam_test_block
)
from Meta_Analysis import eval_pr_auc_for_meta
from meta_pool import pool_from_csv
from shap_meta import summarize_shap_region_from_matrix, make_all_shap_meta_outputs
from visual_grad_boost_updated import generate_all_plots

warnings.simplefilter('ignore', category=FutureWarning)


def main():
    """
    Execute the end-to-end model training, evaluation, explainability, and meta-export pipeline.
    """
    print("=" * 80)
    print(f"Executing RETIN Predictive Pipeline: Region={SEL_REGION}, Cohort={SUFFIX}")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # 1. DATA INGESTION AND COHORT EXTRACTION
    # --------------------------------------------------------------------------
    if not os.path.exists(DB_DIR):
        raise FileNotFoundError(f"Source cohort database not found at: {DB_DIR}")

    print(f"[Data Loading] Reading administrative health records from {DB_DIR}...")
    df = pd.read_csv(DB_DIR, low_memory=False)

    # Apply stratification by biological sex if configured (1 = Male, 2 = Female)
    if filter_block:
        print(f"[Stratification] Filtering cohort by biological sex: code {SEX_flt} ({SUFFIX}).")
        df = df[df["sesso"] == SEX_flt].copy()
        if "sesso" in df.columns:
            df = df.drop(columns=['sesso'])

    # Apply clinical feature engineering and clean transactional non-utilization
    df = preprocess_data(df)

    if eda_block:
        eda_func(df)

    # Separate feature matrix and clinical outcome
    X = df.drop('COMP', axis=1)
    y = df['COMP']
    print(f"[Cohort Profile] Sample size: N = {X.shape[0]:,}, Features: P = {X.shape[1]}, "
          f"Outcome prevalence: {y.mean() * 100:.2f}%")

    # --------------------------------------------------------------------------
    # 2. STRATIFIED TRAIN / HOLDOUT TEST PARTITIONING
    # --------------------------------------------------------------------------
    # 70% Training / 30% Holdout Test split stratified on the clinical outcome
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_test = X_test.reindex(columns=X_train.columns)

    # Standardize continuous predictors while preserving discrete binary indicators
    X_train_scaled, X_test_scaled = data_scaler(X_train, X_test)

    # --------------------------------------------------------------------------
    # 3. TWO-STAGE RESAMPLING (APPLIED STRICTLY TO TRAINING PARTITION)
    # --------------------------------------------------------------------------
    if balance_block:
        print("\n[Resampling] Executing two-stage training-set resampling pipeline:")
        print("  Stage 1: ADASYN oversampling (minority adaptive synthetic density estimation)")
        print("  Stage 2: Tomek Links majority undersampling (boundary cleaning)")
        
        resampling_pipeline = Pipeline([
            ('adasyn', ADASYN(sampling_strategy='minority', random_state=42)),
            ('tomek', TomekLinks(sampling_strategy='auto'))
        ])

        X_train_res, y_train_res = resampling_pipeline.fit_resample(X_train_scaled, y_train)
        print(f"  Training cohort prior to resampling: N = {X_train.shape[0]:,} "
              f"(Complications: {y_train.sum():,})")
        print(f"  Training cohort after resampling:    N = {X_train_res.shape[0]:,} "
              f"(Complications: {y_train_res.sum():,})")
        print(f"  Holdout test cohort (strictly unresampled): N = {X_test.shape[0]:,} "
              f"(Natural prevalence: {y_test.mean() * 100:.2f}%)")
    else:
        X_train_res, y_train_res = X_train_scaled, y_train

    # --------------------------------------------------------------------------
    # 4. MODEL FITTING AND NESTED HYPERPARAMETER OPTIMIZATION
    # --------------------------------------------------------------------------
    if HistGrad_block:
        print("\n[Modeling] Configuring HistGradientBoostingClassifier with histogram binning...")
        classifier = HistGradientBoostingClassifier(random_state=42, class_weight='balanced')

        # Hyperparameter search space
        param_grid = {
            'learning_rate': [0.01, 0.1, 0.2],
            'max_iter': [100, 200, 300],
            'max_depth': [3, 5, 7]
        }

        # 5-Fold Stratified Nested Cross-Validation optimizing Average Precision (PR-AUC)
        kf = KFold(n_splits=5, random_state=42, shuffle=True)
        grid_search = GridSearchCV(
            classifier, param_grid, cv=kf, scoring='average_precision', n_jobs=-1, verbose=1
        )

        # Cross-validation performance check
        cv_scores = cross_val_score(grid_search, X_train_res, y_train_res, cv=kf, scoring='roc_auc')
        print(f"[Nested CV] Mean ROC-AUC: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

        # Fit best hyperparameter configuration on resampled training partition
        grid_search.fit(X_train_res, y_train_res)
        best_model = grid_search.best_estimator_
        print(f"[Modeling] Optimal Hyperparameters: {grid_search.best_params_}")

        # Predict continuous complication probabilities on unaugmented holdout test set
        y_pred_proba = best_model.predict_proba(X_test_scaled)[:, 1]

        # ----------------------------------------------------------------------
        # 5. OPTIMAL DECISION THRESHOLD CALIBRATION
        # ----------------------------------------------------------------------
        if BEST_threshold:
            precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
            best_idx = np.argmax(f1_scores[:-1])
            optimal_threshold = thresholds[best_idx]
            print(f"\n[Threshold Tuning] Calibrated Decision Cutoff: {optimal_threshold:.4f} "
                  f"(Maximized Test F1-Score: {f1_scores[best_idx]:.4f})")
            y_pred_labels = (y_pred_proba >= optimal_threshold).astype(int)
        else:
            optimal_threshold = 0.50
            y_pred_labels = (y_pred_proba >= 0.50).astype(int)

        # ----------------------------------------------------------------------
        # 6. COMPREHENSIVE HOLDOUT TEST SET PERFORMANCE EVALUATION
        # ----------------------------------------------------------------------
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        pr_auc = average_precision_score(y_test, y_pred_proba)
        balanced_acc = balanced_accuracy_score(y_test, y_pred_labels)
        brier = brier_score_loss(y_test, y_pred_proba)

        print("\n" + "=" * 50)
        print("OUT-OF-SAMPLE TEST EVALUATION (STRICTLY HOLDOUT)")
        print("=" * 50)
        print(f"ROC-AUC:           {roc_auc:.4f}")
        print(f"PR-AUC (AP):       {pr_auc:.4f}")
        print(f"Balanced Accuracy: {balanced_acc:.4f}")
        print(f"Brier Score:       {brier:.4f}")
        print("-" * 50)
        print(classification_report(y_test, y_pred_labels, target_names=['NO COMP', 'COMP'], digits=4))

        # ----------------------------------------------------------------------
        # 7. META-ANALYSIS EXPORT: BOOTSTRAP VARIANCE & LOGIT SCALING
        # ----------------------------------------------------------------------
        if meta_eval:
            print("\n[Meta-Analysis] Estimating non-parametric bootstrap sampling variance...")
            eval_pr_auc_for_meta(
                y_true=y_test,
                y_score=y_pred_proba,
                region=SEL_REGION,
                B=2000,
                seed=42,
                include_trapezoid=True,
                add_logit_columns=True,
                save_csv=META,
                verbose=True
            )

    # --------------------------------------------------------------------------
    # 8. SHAP LOCAL EXPLAINABILITY AND ATTRIBUTION ANALYSIS
    # --------------------------------------------------------------------------
    if SHAP_block and HistGrad_block:
        print("\n[Explainability] Commencing SHAP KernelExplainer analysis...")
        
        # Summary background reference constructed via stratified k-means / sampling
        background_sample = shap.sample(X_train_res, 200, random_state=42)
        X_test_sample = X_test_scaled.sample(200, random_state=42)
        y_test_sample = y_test.loc[X_test_sample.index]

        def model_predict_proba(data):
            if isinstance(data, np.ndarray):
                data = pd.DataFrame(data, columns=X_train_res.columns)
            return best_model.predict_proba(data)

        explainer = shap.KernelExplainer(model_predict_proba, background_sample)
        shap_values_raw = explainer.shap_values(X_test_sample)

        # Extract SHAP attributions for the positive complication class (COMP=1)
        svk = shap_values_raw[:, :, 1] if len(shap_values_raw.shape) == 3 else shap_values_raw

        # Persist attribution arrays and metadata
        np.save(Path(SHAP_VALUE) / f"shap_values_{SEL_REGION}_{SUFFIX}.npy", svk)
        np.save(Path(SHAP_VALUE) / f"shap_features_{SEL_REGION}_{SUFFIX}.npy", X_test_sample.values)

        with open(Path(SHAP_VALUE) / f"feature_names_{SEL_REGION}_{SUFFIX}.txt", "w") as f:
            for feat in X_test_sample.columns:
                f.write(f"{feat}\n")

        # Export standard SHAP summary bee-swarm plot
        shap.summary_plot(svk, X_test_sample, show=False, max_display=10)
        plt.savefig(SHAP_PLOT, bbox_inches='tight', dpi=300)
        plt.close()

        # Compute global feature attribution magnitudes
        shap_mean_abs = np.abs(svk).mean(axis=0)
        shap_std_abs = np.abs(svk).std(axis=0)

        shap_summary_df = pd.DataFrame({
            'Feature': X_test_sample.columns,
            'Mean |SHAP|': shap_mean_abs,
            'Std |SHAP|': shap_std_abs
        }).sort_values(by='Mean |SHAP|', ascending=False)

        print("\n[Explainability] Top 10 Features by Mean Absolute SHAP Value:")
        print(shap_summary_df.head(10).to_string(index=False))

        # ----------------------------------------------------------------------
        # 9. STATISTICAL VALIDATION: NON-PARAMETRIC TESTS & SPEARMAN CORRELATIONS
        # ----------------------------------------------------------------------
        top_features = shap_summary_df['Feature'].head(10).tolist()

        if shap_nnparam_test_block:
            print("\n[Statistical Testing] Mann-Whitney U tests comparing SHAP distributions by outcome:")
            for feat in top_features:
                col_idx = X_test_sample.columns.get_loc(feat)
                shap_f = svk[:, col_idx]
                class0 = shap_f[y_test_sample == 0]
                class1 = shap_f[y_test_sample == 1]
                stat, p_val = mannwhitneyu(class0, class1, alternative='two-sided')
                p_repr = "< 0.0001" if p_val < 0.0001 else f"{p_val:.4f}"
                print(f"  {feat:<30} U = {stat:>8.1f}, p-value = {p_repr}")

        if spearman_block:
            print("\n[Correlation Analysis] Spearman rank correlation (rho) between feature level and SHAP:")
            for feat in top_features[:5]:
                col_idx = X_test_sample.columns.get_loc(feat)
                shap_f = svk[:, col_idx]
                feat_raw = X_test_sample[feat].values
                rho, p_val = spearmanr(feat_raw, shap_f)
                p_repr = "< 0.0001" if p_val < 0.0001 else f"{p_val:.4f}"
                print(f"  {feat:<30} rho = {rho:>7.3f}, p-value = {p_repr}")

        if shap_meta_block:
            print("\n[Meta-Export] Generating tidy per-region SHAP summary table...")
            summarize_shap_region_from_matrix(
                svk=svk,
                feature_names=list(X_test_sample.columns),
                region=SEL_REGION,
                topk=5,
                B_boot=2000,
                save_csv=os.path.join(SHAP_META, f"Meta_SHAP_{SEL_REGION}.csv")
            )

    print("\n" + "=" * 80)
    print("Execution completed successfully.")
    print("=" * 80)


if __name__ == "__main__":
    main()
