"""
================================================================================
Data Preprocessing and Feature Engineering Module for Administrative Health Data
================================================================================

Authors: Claudio Mazzi, et al.
Affiliation: Sant'Anna School of Advanced Studies (MeS Lab) / RETIN Project Consortium
Target Journal: Journal of the Royal Statistical Society: Series C (Applied Statistics)

Description:
    This module implements the end-to-end data preparation pipeline for routine
    Administrative Health Data (AHD). It handles:
      1. Extraction of patient-level demographic, comorbidity, and inpatient burden metrics.
      2. Encoding of binary process-of-care adherence indicators aligned with
         Italian Standards for the Care of Diabetes Mellitus (AMD-SID).
      3. Aggregation of clinical event entropy metrics reflecting health system contact fragmentation.
      4. Definition of the composite clinical outcome (COMP) across macrovascular,
         microvascular, and acute complications without lookahead leakage.
      5. Systematic zero-filling for transactional health claims non-utilization.
      6. Separation and standard scaling of continuous features while preserving binary flags.
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from Config_RETIN import EDA_PLOT, SUMMARY

# ==============================================================================
# CLINICAL FEATURE DEFINITIONS AND METRIC GROUPINGS
# ==============================================================================

# Disaggregated complication indicator labels derived from inpatient hospital discharge records (SDO):
# AMP: Lower-extremity amputation
# AMP_MAJ: Major amputation
# AMP_MIN: Minor amputation
# ICTUS: Ischemic or hemorrhagic stroke
# IMA: Acute myocardial infarction
# SCO: Heart failure decompensation
# INS: Chronic kidney failure (dialysis or stage 5 CKD)
# RET: Proliferative diabetic retinopathy
# ALTRO: Other severe diabetes-related vascular events
# MORTE: All-cause mortality (excluding trauma/accidental causes)
# ULC_PIEDE: Diabetic foot ulceration
# CANC: Diabetes-associated oncological inpatient admissions
# INF_PIEDE: Diabetic foot soft tissue/bone infection
# CHARCOT: Charcot neuroarthropathy
# RIV: Lower-extremity revascularization procedure
COMPLICATION_LABELS = [
    'AMP', 'AMP_MAJ', 'AMP_MIN', 'ICTUS', 'IMA', 'SCO', 'INS', 'RET',
    'ALTRO', 'MORTE', 'ULC_PIEDE', 'CANC', 'INF_PIEDE', 'CHARCOT', 'RIV'
]

# Non-predictive administrative identifiers, redundant date fields, and intermediate variables
# excluded from the final predictive feature space:
COLUMNS_TO_DROP = {
    'idars', 'DIAB_T1', 'DIAB_T2', 'Max_event_spa', 'Max_event_ph', 'Max_event_ese',
    'min_data_spa', 'max_data_spa', 'media_data_spa', 'median_data_spa',
    'min_data_sdo', 'max_data_sdo', 'media_data_sdo', 'mean_DRG_weight',
    'IQR_DRG', 'media_peso_drg_no_DH', 'entropy_ICD9CM_2', 'Diff_cod_ICD9CM_2', 'guideline'
}

# Binary process-of-care indicators, pharmacotherapy markers, and demographic indicator variables.
# These columns are preserved as discrete binary indicators and exempted from continuous z-score scaling:
BINARY_AND_DUMMY_FEATURES = [
    'V_DIAB',               # Annual outpatient specialist diabetology consultation
    'emoglobina_glicata',   # Annual glycated hemoglobin (HbA1c) testing adherence
    'profilo_lipidico',     # Annual lipid profile testing adherence
    'microalbuminuria',     # Annual microalbuminuria monitoring adherence
    'visita_occhio',        # Annual specialist ophthalmological examination
    'creatinina',           # Annual serum creatinine / eGFR renal assessment
    'sesso',                # Biological sex (1 = male, 2 = female; recoded to 0/1 where applicable)
    'INSULINA',             # Prescription of insulin or insulin analogues
    'OTHER_PH'              # Prescription of non-insulin glucose-lowering pharmaceuticals
]


def eda_func(df: pd.DataFrame) -> None:
    """
    Perform Exploratory Data Analysis (EDA) on the cohort DataFrame.

    Computes descriptive statistics, assesses outcome imbalance, and exports
    a distribution plot and comprehensive summary table.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing preprocessed patient records and the 'COMP' target.

    Returns
    -------
    None
        Artifacts are written to disk: pie chart of outcome distribution (EDA_PLOT)
        and summary metrics CSV (SUMMARY).
    """
    print("--- Exploratory Data Analysis (EDA) ---")
    print(df.info())
    print(df.describe())

    # Target class distribution analysis (COMP: 0 = No Complication, 1 = Complication)
    comp_counts = df['COMP'].value_counts()
    fig, ax = plt.subplots(figsize=(7, 7))
    comp_counts.plot(
        kind='pie',
        labels=['No Complication', 'Complication'],
        autopct='%1.1f%%',
        colors=['#6baed6', '#fb6a4a'],
        textprops={'fontsize': 12},
        ax=ax
    )
    ax.set_ylabel('')
    ax.set_title("Diabetes Complication Outcome Distribution", fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig(EDA_PLOT, dpi=300)
    plt.close(fig)

    # Export comprehensive distributional summary and transaction completeness table
    desc_stats = df.describe(include='all').transpose()
    desc_stats["missing_values"] = df.isnull().sum()
    desc_stats["unique_values"] = df.nunique()
    desc_stats.to_csv(SUMMARY, index=True)
    print(f"[EDA] Summary table successfully exported to {SUMMARY}")


def preprocess_data(df: pd.DataFrame, target_subset: str = "ALL") -> pd.DataFrame:
    """
    Preprocess raw Administrative Health Data for supervised machine learning.

    Executes cohort inclusion filtering, column pruning, binary composite outcome
    construction, age range restriction, and transactional non-utilization imputation.

    Parameters
    ----------
    df : pd.DataFrame
        Raw cohort DataFrame containing longitudinal health claims records.
    target_subset : str, default="ALL"
        Outcome aggregation mode:
          - "ALL": Aggregates all micro- and macrovascular complications into binary COMP.
          - "CARD": Restricts complication definition to major adverse cardiovascular events (MACE).

    Returns
    -------
    pd.DataFrame
        Cleaned, feature-engineered DataFrame ready for training/test partitioning.

    Methodological Notes
    -------------------
    - Lookahead Data Leakage Prevention: All predictor features in the source database
      were aggregated strictly prior to the date of the first complication ($t < T_{event}$).
    - Transactional Imputation: Absence of claims, prescriptions, or exemptions represents
      genuine non-utilization within administrative health claims; missing entries are
      therefore systematically imputed with 0.
    """
    print("[Preprocessing] Commencing cohort filtering and feature restructuring...")

    # 1. Restrict cohort to individuals with verified Type-2 Diabetes
    if 'DIAB_T2' in df.columns:
        df = df.loc[df['DIAB_T2'] == 1].copy()

    # 2. Remove redundant administrative identifiers and intermediate tracking fields
    cols_to_remove = [c for c in COLUMNS_TO_DROP if c in df.columns]
    df = df.drop(columns=cols_to_remove, axis=1)

    # 3. Construct the binary clinical outcome (COMP)
    existing_complication_cols = [c for c in COMPLICATION_LABELS if c in df.columns]
    
    if target_subset.upper() == "CARD":
        # Cardiovascular-specific outcome subset: AMP, IMA, SCO, ICTUS, ALTRO
        card_cols = [c for c in ['AMP', 'IMA', 'SCO', 'ICTUS', 'ALTRO'] if c in df.columns]
        df['COMP'] = (df[card_cols].max(axis=1) > 0).astype(int)
    else:
        # Comprehensive composite outcome: any microvascular, macrovascular, or fatal event
        df['COMP'] = (df[existing_complication_cols].max(axis=1) > 0).astype(int)

    # 4. Remove granular disaggregated complication columns to prevent label leakage
    df = df.drop(columns=existing_complication_cols, axis=1)

    # 5. Restrict cohort to adult patients within clinically plausible age bounds [18, 100]
    if 'eta' in df.columns:
        df = df[(df['eta'].notna()) & (df['eta'] >= 18) & (df['eta'] <= 100)].copy()

    # 6. Impute transactional non-utilization with zero (complete administrative billing logs)
    df_processed = df.fillna(0).copy()

    # Verify absence of residual missing values
    if df_processed.isna().any().any():
        nan_cols = df_processed.columns[df_processed.isna().any()].tolist()
        print(f"[Warning] Residual NaN values detected in columns: {nan_cols}")

    print(f"[Preprocessing] Completed. Final cohort shape: {df_processed.shape}, "
          f"Outcome prevalence (COMP=1): {df_processed['COMP'].mean():.4f}")

    return df_processed


def data_scaler(x_train: pd.DataFrame, x_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Standardize continuous predictor features via z-score scaling without test-set leakage.

    Fits StandardScaler exclusively on the training partition and transforms both
    training and test partitions. Binary guideline indicators and categorical flags
    are strictly exempted to preserve discrete interpretations.

    Parameters
    ----------
    x_train : pd.DataFrame
        Training feature matrix.
    x_test : pd.DataFrame
        Holdout test feature matrix.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Scaled (x_train_scaled, x_test_scaled) feature DataFrames with aligned column indices.
    """
    # Identify continuous columns to scale (all features excluding discrete binary indicators)
    non_scaling_cols = [c for c in BINARY_AND_DUMMY_FEATURES if c in x_train.columns]
    continuous_cols = x_train.columns.difference(non_scaling_cols).tolist()

    print(f"[Scaling] Standardizing {len(continuous_cols)} continuous features via StandardScaler.")
    print(f"[Scaling] Preserving {len(non_scaling_cols)} discrete binary indicators unscaled.")

    scaler = StandardScaler()

    x_train_scaled = x_train.copy()
    x_test_scaled = x_test.copy()

    # Fit scaler strictly on training data to prevent out-of-sample data leakage
    x_train_scaled[continuous_cols] = scaler.fit_transform(x_train[continuous_cols])
    x_test_scaled[continuous_cols] = scaler.transform(x_test[continuous_cols])

    return x_train_scaled, x_test_scaled
