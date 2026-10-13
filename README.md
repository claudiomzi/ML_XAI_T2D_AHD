# Machine Learning and Explainable AI for Type-2 Diabetes Complications: A Multi-Regional Study using Administrative Health Data

[![Journal](https://img.shields.io/badge/Target%20Journal-JRSS%20Series%20C%20(Applied%20Statistics)-blue.svg)](https://rss.onlinelibrary.wiley.com/journal/14679876)
[![Python](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/Access-Private%20Repository%20(On%20Request)-lightgrey.svg)](#data-governance-and-repository-access)

This repository contains the complete statistical computing and machine learning pipeline developed for the study:  
> **"Machine Learning and Explainable AI for Type-2 Diabetes Management: A Multi-Regional Study using Administrative Health Data"**  
> *Submitted for peer review to the Journal of the Royal Statistical Society: Series C (Applied Statistics).*

---

## 1. Data Governance and Repository Access

Due to strict European and Italian data privacy regulations (**EU General Data Protection Regulation, GDPR**) governing pseudonymized regional Administrative Health Data (AHD), individual-level records cannot be deposited in public repositories. 

This repository is maintained as a **private repository**. Full access to the codebase, analysis pipelines, and synthetic verification scripts will be granted by the corresponding author upon reasonable academic request for research verification, reproducibility checks, and peer-review audit.

* **Corresponding Author:** Claudio Mazzi ([claudio.mazzi@santannapisa.it](mailto:claudio.mazzi@santannapisa.it))
* **Institution:** Management and Health Laboratory (MeS Lab), Institute of Management, Sant'Anna School of Advanced Studies, Pisa, Italy.

---

## 2. Pipeline Architecture and Module Overview

The codebase is organized into modular Python scripts designed to execute within decentralized regional computing environments where individual records cannot be pooled across regions:

```
SCRIPT/
├── Config_RETIN.py           # Centralized configuration, regional flags, paths, and execution toggles
├── RETIN_data_prep.py        # Clinical feature engineering, AMD-SID guideline encoding, zero-imputation, and scaling
├── Grad_Boost_RETIN.py       # Main model training: nested CV, ADASYN + Tomek, threshold tuning, and SHAP attribution
├── benchmark_baseline.py     # Out-of-sample benchmark against penalized logistic regression (L1/L2)
├── Meta_Analysis.py          # Stratified bootstrap sampling variance estimation (v_i) and logit transformation
├── meta_pool.py              # Random-effects meta-analysis with Hartung-Knapp-Sidik-Jonkman (HKSJ) adjustment
├── shap_meta.py              # Cross-regional explainability synthesis via RE-HKSJ and hierarchical Beta-Binomial models
├── predictive_tool_RETIN.py  # Model serialization and prospective inference engine (forecast_retin)
├── JRS_SHAP.py               # Multi-panel publication-grade SHAP visualization builder
└── visual_grad_boost_updated.py # Auxiliary diagnostic performance plotting (PR curves, ROC curves, calibration)
```

### Detailed Functional Breakdown:

1. **`Config_RETIN.py`**:
   - Manages global regional selectors (`SEL_REGION = "TY"`, `"LY"`, `"SY"`).
   - Manages biological sex stratification (`SEX_flt = 1` for Males, `SEX_flt = 2` for Females, or `Global`).
   - Sets execution switches for training, resampling, threshold tuning, explainability, and meta-analytic pooling.

2. **`RETIN_data_prep.py`**:
   - Preprocesses longitudinal claims records covering hospital discharge data (SDO), outpatient specialist services, and pharmaceuticals.
   - Extracts process-of-care adherence markers aligned with national clinical recommendations (*Italian Standards for the Care of Diabetes Mellitus, AMD-SID*): HbA1c, lipid panel, microalbuminuria, eye exams, creatinine/eGFR, and specialist diabetology visits.
   - Operates under a dynamic pre-event observation window ($t < T_{\text{event}}$) to completely eliminate lookahead data leakage.
   - Standardizes continuous features via `StandardScaler` (fitted exclusively on training data to prevent out-of-sample leakage) while preserving discrete binary indicators.

3. **`Grad_Boost_RETIN.py`**:
   - Implements a 70% training / 30% holdout test partition stratified on the outcome.
   - Applies a two-step resampling pipeline exclusively to the training partition: Adaptive Synthetic oversampling (`ADASYN`) followed by `TomekLinks` cleaning.
   - Trains `HistGradientBoostingClassifier` with $O(P \cdot B)$ split complexity per tree (histogram binning with $B \le 256$), optimized via 5-fold nested cross-validation over learning rate, maximum depth, and boosting iterations.
   - Calibrates optimal decision thresholds by maximizing the $F_1$-score along the holdout Precision-Recall curve.
   - Computes local feature attributions using SHAP `KernelExplainer` with $k$-means background summarization.
   - Validates feature drivers via two-sided Mann-Whitney U hypothesis testing and Spearman rank correlation ($\rho$).

4. **`benchmark_baseline.py`**:
   - Validates the empirical necessity of non-linear gradient boosting by benchmarking against standard penalized linear classifiers (L1 Lasso, L2 Ridge, with and without training-stage resampling).
   - Demonstrates that `HistGradientBoostingClassifier` achieves a +16.8% relative improvement in PR-AUC and reduces the Brier score by nearly half compared to linear baselines.

5. **`Meta_Analysis.py` & `meta_pool.py`**:
   - Computes the empirical sampling variance $v_i$ for PR-AUC on the logit scale using non-parametric stratified bootstrap ($B = 2,000$) via the delta method:
     $$v_i = \frac{\text{Var}^*(\text{PR-AUC}_i)}{\overline{\text{PR-AUC}}_i^2 \, (1 - \overline{\text{PR-AUC}}_i)^2}$$
   - Synthesizes regional estimates using random-effects meta-analysis with the Hartung-Knapp-Sidik-Jonkman (HKSJ) adjustment with $df = k - 1 = 2$ degrees of freedom to account for small-$k$ parameter uncertainty.

6. **`shap_meta.py`**:
   - Synthesizes feature attribution magnitudes across regional models using log-transformed random-effects HKSJ models.
   - Pools patient-level top-5 feature appearance frequencies using a hierarchical Beta-Binomial model (BB1N) fitted via Maximum Likelihood Estimation with numerical Hessian inversion ($df = 2K - 2$).

7. **`predictive_tool_RETIN.py`**:
   - Provides model persistence (`retin_hgb_model.pkl`, `retin_metadata.json`).
   - Implements `forecast_retin(new_csv_path)` to generate prospective complication risk predictions on unobserved patient cohorts without retraining or data leakage.

---

## 3. Installation and Computational Environment

The pipeline requires **Python 3.10+** and standard open-source scientific computing libraries.

### Recommended Conda Environment Setup:
```bash
conda create -n retin_ml python=3.10 -y
conda activate retin_ml

pip install numpy==1.24.3 pandas==2.0.3 scikit-learn==1.3.0 \
            imbalanced-learn==0.11.0 scipy==1.11.1 shap==0.42.1 \
            matplotlib==3.7.2 seaborn==0.12.2 joblib==1.3.1
```

---

## 4. Execution Workflow

### A. Run Main Scientific Model Training and SHAP Analysis:
```bash
cd SCRIPT
python Grad_Boost_RETIN.py
```
*Outputs: Evaluates the 30% holdout test set, displays PR-AUC, ROC-AUC, Balanced Accuracy, and Brier score, saves SHAP beeswarm plots, and appends regional meta-analytic metrics to CSV.*

### B. Run Empirical Benchmark against Penalized Logistic Baselines:
```bash
python benchmark_baseline.py
```
*Outputs: Compares HistGradientBoosting against Lasso and Ridge classifiers and exports `benchmark_tuscany_baseline.csv`.*

### C. Run Cross-Regional Meta-Analytic Synthesis:
Once per-region runs (`TY`, `LY`, `SY`) are completed:
```bash
python -c "from meta_pool import pool_from_csv; res = pool_from_csv('../FINAL_RUN/META/Meta_Global.csv'); print(res)"
```

### D. Run Prospective Risk Scoring on Unseen Cohorts:
```python
from predictive_tool_RETIN import forecast_retin

# Score an updated/prospective cohort using frozen model weights
scored_df = forecast_retin("new_patient_cohort.csv")
scored_df.to_csv("patient_risk_predictions.csv", index=False)
```

---

## 5. Citation

If you use or adapt this methodology, please cite our corresponding manuscript:
```bibtex
@article{mazzi2026t2d,
  title={Machine Learning and Explainable AI for Type-2 Diabetes Management: A Multi-Regional Study using Administrative Health Data},
  author={Mazzi, Claudio and Seghieri, Chiara and Molinari, Roberto and Berta, Paolo and Ferrante, Mauro},
  journal={Journal of the Royal Statistical Society: Series C (Applied Statistics)},
  year={2026},
  note={Under Review}
}
```
