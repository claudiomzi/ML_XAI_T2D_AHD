"""
================================================================================
Configuration Module for Multi-Regional Type-2 Diabetes Complication Risk
Prediction and Explainability Framework (RETIN Project)
================================================================================

Authors: Claudio Mazzi, et al.
Affiliation: Sant'Anna School of Advanced Studies (MeS Lab) / RETIN Project Consortium
Target Journal: Journal of the Royal Statistical Society: Series C (Applied Statistics)

Description:
    Centralized configuration module providing global execution switches, cohort
    stratification parameters, input/output filesystem paths, and algorithmic 
    hyperparameter flags for the decentralized machine learning and meta-analytic
    pipeline across Italian Administrative Health Data (AHD).
================================================================================
"""

import os
from pathlib import Path

# ==============================================================================
# 1. REGIONAL COHORT AND STRATIFICATION CONFIGURATION
# ==============================================================================

# Target region selector:
# 'TY' = Tuscany (Toscana)
# 'LY' = Lombardy (Lombardia)
# 'SY' = Sicily (Sicilia)
SEL_REGION: str = "TY"

# Test region selector (used when performing cross-regional transfer evaluation)
SEL_REGION_TEST: str = "TY"

# Stratification flag by biological sex:
# filter_block = True: Stratifies analysis by biological sex
# filter_block = False: Evaluates the combined population cohort (Global)
filter_block: bool = False

# Biological sex filter code (administrative coding: 1 = Male, 2 = Female):
SEX_flt: int = 2

if filter_block:
    SUFFIX: str = "F" if SEX_flt == 2 else "M"
else:
    SUFFIX: str = "Global"


# ==============================================================================
# 2. PIPELINE EXECUTION TOGGLES
# ==============================================================================

# Exploratory Data Analysis & Target Distribution Pie Charts
eda_block: bool = False

# Supervised Machine Learning Pipeline Controls:
balance_block: bool = True           # Apply two-stage training-set resampling (ADASYN + Tomek Links)
HistGrad_block: bool = True          # Train HistGradientBoostingClassifier with nested CV
BEST_threshold: bool = True          # Compute optimal classification threshold via F1/PR curve

# Model Serialization Flag (for prospective clinical decision support tool)
SAVE_MODEL: bool = False

# Meta-Analysis Evaluation & Pooling:
meta_eval: bool = True               # Compute bootstrap PR-AUC and logit variance for meta-analysis
meta_pool: bool = False              # Pool cross-regional PR-AUC via random-effects HKSJ
shap_meta_block: bool = False        # Export local SHAP statistics for meta-analytic synthesis
shap_meta_block_pool: bool = False   # Pool cross-regional SHAP magnitudes and frequencies

# Feature Attribution & Statistical Explainability:
PERM_importance_block: bool = False  # Compute Permutation Feature Importance
SHAP_block: bool = True              # Execute KernelExplainer SHAP attribution analysis
shap_nnparam_test_block: bool = True # Run two-sided Mann-Whitney U tests on SHAP distributions
shap_violin_block: bool = False      # Generate publication-quality SHAP violin plots
spearman_block: bool = True          # Compute Spearman rank correlations between features and SHAP values
plot_block: bool = False             # Generate supplemental diagnostic performance figures


# ==============================================================================
# 3. DIRECTORY AND PATH CONFIGURATION
# ==============================================================================

# Resolve base directories dynamically relative to this script's location
BASE_DIR: Path = Path(__file__).resolve().parent.parent

# Primary data directories (customizable or dynamically falling back to workspace paths)
DATA_DIR: Path = BASE_DIR
MODEL_DIR: Path = BASE_DIR / "GradientBoosting"
OUTPUT_DIR: Path = BASE_DIR / "FINAL_RUN"
SHAP_VALUE: Path = MODEL_DIR / "SHAP_VALUE"
SHAP_META: Path = OUTPUT_DIR / "META"

# Ensure output directories exist upon initialization
for directory in [MODEL_DIR, OUTPUT_DIR, SHAP_VALUE, SHAP_META, OUTPUT_DIR / "EDA", OUTPUT_DIR / "PLOT"]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset paths for train and out-of-sample evaluation
DB_DIR: str = str(DATA_DIR / f"DB_ML_{SEL_REGION}.csv")
DB_DIR_TEST: str = str(DATA_DIR / f"DB_ML_{SEL_REGION_TEST}.csv")

# Specific output artifact filenames
EDA_PLOT: str = str(OUTPUT_DIR / "EDA" / f"COMP_PERC_{SEL_REGION}_{SUFFIX}.png")
SUMMARY: str = str(OUTPUT_DIR / "EDA" / f"Summary_{SEL_REGION}_{SUFFIX}.csv")
SHAP_PLOT: str = str(OUTPUT_DIR / "PLOT" / f"SHAP_{SEL_REGION}_{SUFFIX}.png")
SHAP_VIOLIN: str = str(OUTPUT_DIR / "PLOT" / f"Violin_SHAP_{SEL_REGION}_{SUFFIX}_{{feat}}.png")
PLOT: str = str(OUTPUT_DIR / "PLOT" / f"PLOT_{SEL_REGION}_{SUFFIX}.pdf")
META: str = str(SHAP_META / f"Meta_{SUFFIX}.csv")
