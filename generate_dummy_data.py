"""
================================================================================
Synthetic Dummy Data Generator for Schema Verification and Code Execution
================================================================================

Authors: Claudio Mazzi, et al.
Affiliation: Sant'Anna School of Advanced Studies (MeS Lab) / RETIN Project Consortium
Target Journal: Journal of the Royal Statistical Society: Series C (Applied Statistics)

Description:
    Generates a completely fictitious, synthetic dataset matching the exact schema,
    variable names, and data types of the Italian Administrative Health Data (AHD)
    cohorts.
    
    Purpose:
      Allows external researchers and reviewers granted private repository access
      to test, execute, and verify the computational pipeline without requiring
      access to real, privacy-restricted patient records.
================================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path


def generate_synthetic_cohort(n_samples: int = 1000, seed: int = 42, output_path: str = "sample_dummy_data.csv") -> pd.DataFrame:
    """
    Generate synthetic health claims records mimicking the empirical distribution.

    Parameters
    ----------
    n_samples : int, default=1000
        Number of synthetic patient records to generate.
    seed : int, default=42
        Random seed for exact reproducibility.
    output_path : str, default="sample_dummy_data.csv"
        Destination path for saving the synthetic CSV file.

    Returns
    -------
    pd.DataFrame
        Synthetic patient DataFrame.
    """
    rng = np.random.default_rng(seed)

    # Demographic attributes
    age = rng.normal(loc=71, scale=11, size=n_samples).clip(25, 95).astype(int)
    sex = rng.choice([1, 2], size=n_samples, p=[0.53, 0.47])  # 1 = Male, 2 = Female

    # Diagnostic chronic indicators
    diab_t2 = np.ones(n_samples, dtype=int)
    comorbidity_count = rng.poisson(lam=3.2, size=n_samples).clip(0, 15)

    # Inpatient healthcare utilization (SDO)
    admissions = rng.poisson(lam=0.4, size=n_samples).clip(0, 10)
    has_admission = (admissions > 0).astype(int)
    
    drg_min = (rng.uniform(0.5, 2.0, size=n_samples) * has_admission).round(3)
    drg_max = (drg_min + rng.uniform(0.0, 1.5, size=n_samples) * has_admission).round(3)
    drg_median = ((drg_min + drg_max) / 2.0).round(3)
    avg_sdo_interval = (rng.uniform(30, 365, size=n_samples) * has_admission).round(1)

    # Service code diversity counts
    diff_codpres = rng.poisson(lam=8, size=n_samples)
    diff_diadim = (rng.poisson(lam=1.2, size=n_samples) * has_admission).astype(int)
    diff_dia1 = (rng.poisson(lam=1.5, size=n_samples) * has_admission).astype(int)
    diff_dia2 = (rng.poisson(lam=0.8, size=n_samples) * has_admission).astype(int)
    diff_codchi = (rng.poisson(lam=0.5, size=n_samples) * has_admission).astype(int)
    diff_codchi2 = (rng.poisson(lam=0.2, size=n_samples) * has_admission).astype(int)
    diff_codchi3 = (rng.poisson(lam=0.1, size=n_samples) * has_admission).astype(int)

    # Pharmacological utilization
    diff_minsan = rng.poisson(lam=12, size=n_samples).clip(1, 40)
    diff_atc5 = rng.poisson(lam=6, size=n_samples).clip(1, 20)

    # Entropy metrics (Shannon variability)
    entropy_codpres = rng.uniform(0.0, 3.5, size=n_samples).round(4)
    entropy_diadim = (rng.uniform(0.0, 2.0, size=n_samples) * has_admission).round(4)
    entropy_dia1 = (rng.uniform(0.0, 2.0, size=n_samples) * has_admission).round(4)
    entropy_dia2 = (rng.uniform(0.0, 1.5, size=n_samples) * has_admission).round(4)
    entropy_codchi = (rng.uniform(0.0, 1.5, size=n_samples) * has_admission).round(4)
    entropy_codchi2 = (rng.uniform(0.0, 1.0, size=n_samples) * has_admission).round(4)
    entropy_codchi3 = (rng.uniform(0.0, 0.8, size=n_samples) * has_admission).round(4)
    entropy_atc5 = rng.uniform(0.5, 3.0, size=n_samples).round(4)
    entropy_minsan = rng.uniform(0.8, 3.8, size=n_samples).round(4)

    # Clinical guideline adherence (AMD-SID binary indicators)
    hba1c = rng.choice([0, 1], size=n_samples, p=[0.42, 0.58])
    lipid = rng.choice([0, 1], size=n_samples, p=[0.48, 0.52])
    eye = rng.choice([0, 1], size=n_samples, p=[0.72, 0.28])
    microalb = rng.choice([0, 1], size=n_samples, p=[0.65, 0.35])
    creat = rng.choice([0, 1], size=n_samples, p=[0.38, 0.62])
    diab_visit = rng.choice([0, 1], size=n_samples, p=[0.60, 0.40])

    # Pharmacotherapy flags
    insulin = rng.choice([0, 1], size=n_samples, p=[0.82, 0.18])
    other_ph = rng.choice([0, 1], size=n_samples, p=[0.25, 0.75])

    # Disaggregated complication indicators (simulating ~13% total composite prevalence)
    complication_probs = {
        'ICTUS': 0.035, 'IMA': 0.030, 'SCO': 0.040, 'INS': 0.025,
        'AMP': 0.005, 'AMP_MAJ': 0.002, 'AMP_MIN': 0.003, 'RET': 0.020,
        'ALTRO': 0.015, 'MORTE': 0.030, 'ULC_PIEDE': 0.012, 'CANC': 0.018,
        'INF_PIEDE': 0.008, 'CHARCOT': 0.001, 'RIV': 0.005
    }
    
    complications_dict = {}
    for comp_name, prob in complication_probs.items():
        complications_dict[comp_name] = rng.choice([0, 1], size=n_samples, p=[1.0 - prob, prob])

    # Assemble complete DataFrame
    data_dict = {
        'DIAB_T2': diab_t2,
        'eta': age,
        'sesso': sex,
        'tot_pat': comorbidity_count,
        'Max_event_sdo': admissions,
        'Diff_cod_codpres': diff_codpres,
        'Diff_cod_diadim': diff_diadim,
        'Diff_cod_dia1': diff_dia1,
        'Diff_cod_dia2': diff_dia2,
        'Diff_cod_codchi': diff_codchi,
        'Diff_cod_codchi2': diff_codchi2,
        'Diff_cod_codchi3': diff_codchi3,
        'media_data_sdo': avg_sdo_interval,
        'Diff_cod_cod_minsan': diff_minsan,
        'Diff_cod_atc5': diff_atc5,
        'min_DRG': drg_min,
        'max_DRG': drg_max,
        'median_DRG_weight': drg_median,
        'entropy_Codpres': entropy_codpres,
        'entropy_diadim': entropy_diadim,
        'entropy_dia1': entropy_dia1,
        'entropy_dia2': entropy_dia2,
        'entropy_Codchi': entropy_codchi,
        'entropy_Codchi2': entropy_codchi2,
        'entropy_Codchi3': entropy_codchi3,
        'entropy_ATC5': entropy_atc5,
        'entropy_cod_minsan': entropy_minsan,
        'emoglobina_glicata': hba1c,
        'profilo_lipidico': lipid,
        'visita_occhio': eye,
        'microalbuminuria': microalb,
        'creatinina': creat,
        'V_DIAB': diab_visit,
        'INSULINA': insulin,
        'OTHER_PH': other_ph,
        **complications_dict
    }

    df_synthetic = pd.DataFrame(data_dict)
    
    # Calculate composite outcome prevalence
    comp_cols = list(complication_probs.keys())
    composite_comp = (df_synthetic[comp_cols].max(axis=1) > 0).astype(int)
    
    df_synthetic.to_csv(output_path, index=False)
    print(f"[Synthetic Data] Generated {n_samples:,} synthetic records -> {output_path}")
    print(f"[Synthetic Data] Simulated composite outcome prevalence: {composite_comp.mean() * 100:.2f}%")

    return df_synthetic


if __name__ == "__main__":
    generate_synthetic_cohort(n_samples=1000, seed=42, output_path="sample_dummy_data.csv")
