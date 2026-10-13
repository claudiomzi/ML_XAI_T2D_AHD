"""
================================================================================
Random-Effects Meta-Analytic Synthesis with Hartung-Knapp-Sidik-Jonkman Adjustment
================================================================================

Authors: Claudio Mazzi, et al.
Affiliation: Sant'Anna School of Advanced Studies (MeS Lab) / RETIN Project Consortium
Target Journal: Journal of the Royal Statistical Society: Series C (Applied Statistics)

Description:
    Implements random-effects meta-analytic pooling for classification performance
    metrics (specifically PR-AUC on the logit scale) across decentralized regional
    healthcare databases under GDPR constraints.

    Methodological Specifications:
      1. Between-Study Variance Estimation: Computes DerSimonian-Laird tau^2 and
         Cochran's Q test statistic.
      2. Small-Sample Heterogeneity Adjustment: Adopts the Hartung-Knapp-Sidik-Jonkman
         (HKSJ) adjustment with df = k - 1 degrees of freedom (t-distribution critical
         values), correcting for parameter uncertainty in tau^2 when k is small (k = 3).
      3. Back-Transformation: Transforms pooled effect and confidence bounds back to the
         natural [0, 1] probability scale via the inverse logit function:
             p = 1 / (1 + exp(-theta))
      4. Large-Sample Heterogeneity Caveats: Formally documents that in massive sample sizes
         (N > 800,000 across regions), within-study variances v_i shrink to near zero
         (v_i ~ 0.0004), driving I^2 = tau^2 / (tau^2 + v) -> 100% algebraically rather
         than reflecting true clinical inconsistency (Rucker et al., 2008; Borenstein et al., 2017).
================================================================================
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Sequence, Tuple
from math import isfinite
from scipy.stats import t as student_t

try:
    import pandas as pd
except ImportError:
    pd = None


def _ilogit(z: float) -> float:
    """Inverse logit (standard logistic) transformation."""
    return float(1.0 / (1.0 + np.exp(-z)))


@dataclass
class MetaResult:
    """
    Container storing the synthesized random-effects meta-analytic estimates.
    """
    k: int
    theta_re: float                  # Pooled summary effect on the logit scale
    se_hksj: float                   # Hartung-Knapp standard error
    df: int                          # Degrees of freedom (k - 1)
    ci_logit: Tuple[float, float]    # 95% Confidence Interval on logit scale
    pooled_pr_auc: float             # Back-transformed pooled PR-AUC
    ci_pr_auc: Tuple[float, float]   # Back-transformed 95% Confidence Interval for PR-AUC
    tau2: float                      # DerSimonian-Laird between-study variance estimate
    I2: float                        # Higgins inconsistency index (%)
    Q: float                         # Cochran's Q heterogeneity test statistic


def _dersimonian_laird_tau2(y: np.ndarray, v: np.ndarray) -> Tuple[float, float]:
    """
    Compute DerSimonian-Laird non-iterative estimate of between-study variance tau^2.
    """
    weights = 1.0 / v
    y_fe = np.sum(weights * y) / np.sum(weights)
    Q = float(np.sum(weights * (y - y_fe) ** 2))
    k = len(y)
    c = float(np.sum(weights) - (np.sum(weights ** 2) / np.sum(weights)))
    
    tau2 = max(0.0, (Q - (k - 1)) / c) if c > 0 else 0.0
    return float(tau2), Q


def _hksj_var(y: np.ndarray, v: np.ndarray, tau2: float) -> Tuple[float, float]:
    """
    Compute the Hartung-Knapp variance of the pooled effect under random-effects weights.
    
    Includes a classic random-effects variance floor safeguard to prevent anti-conservative
    underestimation in small-k settings.
    """
    weights_re = 1.0 / (v + tau2)
    theta_re = float(np.sum(weights_re * y) / np.sum(weights_re))
    k = len(y)
    
    Qe = np.sum(weights_re * (y - theta_re) ** 2)
    V_HK = Qe / ((k - 1) * np.sum(weights_re))
    V_classic = 1.0 / np.sum(weights_re)
    
    # Hartung-Knapp variance safeguard
    V_final = float(max(V_HK, V_classic))
    return V_final, theta_re


def pool_from_arrays(
    logit_effects: Sequence[float],
    var_logit: Sequence[float],
    alpha: float = 0.05
) -> MetaResult:
    """
    Perform random-effects HKSJ meta-analysis on arrays of logit effects and sampling variances.

    Parameters
    ----------
    logit_effects : Sequence[float]
        1D sequence of regional logit(PR-AUC) point estimates.
    var_logit : Sequence[float]
        1D sequence of regional sampling variances v_i derived via delta method bootstrap.
    alpha : float, default=0.05
        Significance level for two-sided (1 - alpha) confidence interval construction.

    Returns
    -------
    MetaResult
        Dataclass containing pooled parameters, back-transformed PR-AUC, and heterogeneity metrics.
    """
    y = np.asarray(logit_effects, dtype=float)
    v = np.asarray(var_logit, dtype=float)

    assert y.shape == v.shape and y.ndim == 1, "Effects and variances must be matching 1D vectors."
    assert len(y) >= 2, "A minimum of k = 2 studies is required for meta-analysis."

    if not np.all([isfinite(x) for x in np.r_[y, v]]):
        raise ValueError("Inputs contain non-finite (NaN or Inf) values.")

    k = len(y)
    df = k - 1

    # 1. Estimate between-study heterogeneity tau^2 and Cochran's Q
    tau2, Q = _dersimonian_laird_tau2(y, v)

    # 2. Hartung-Knapp variance and random-effects weighted pooled mean
    V_HK, theta_re = _hksj_var(y, v, tau2)
    se_hksj = float(np.sqrt(V_HK))

    # 3. t-distribution critical value with df = k - 1
    t_crit = student_t.ppf(1.0 - alpha / 2.0, df=df)
    lo_logit = theta_re - t_crit * se_hksj
    hi_logit = theta_re + t_crit * se_hksj

    # 4. Back-transformation to probability scale
    pooled_p = _ilogit(theta_re)
    ci_p = (_ilogit(lo_logit), _ilogit(hi_logit))

    # 5. Inconsistency index I^2 (%)
    I2 = max(0.0, (Q - df) / Q) * 100.0 if Q > 0 else 0.0

    return MetaResult(
        k=k,
        theta_re=theta_re,
        se_hksj=se_hksj,
        df=df,
        ci_logit=(float(lo_logit), float(hi_logit)),
        pooled_pr_auc=float(pooled_p),
        ci_pr_auc=(float(ci_p[0]), float(ci_p[1])),
        tau2=float(tau2),
        I2=float(I2),
        Q=float(Q)
    )


def pool_from_csv(csv_path: str, alpha: float = 0.05) -> MetaResult:
    """
    Load regional summary statistics from a CSV file and execute random-effects HKSJ pooling.

    Parameters
    ----------
    csv_path : str
        Path to the CSV file containing 'logit_pr_auc' and 'var_logit_pr_auc' columns.
    alpha : float, default=0.05
        Significance level for confidence interval construction.

    Returns
    -------
    MetaResult
        Dataclass containing synthesized meta-analytic results.
    """
    if pd is None:
        raise RuntimeError("pandas is required to read tabular CSV data for meta-analysis.")

    df = pd.read_csv(csv_path)
    required_cols = ["logit_pr_auc", "var_logit_pr_auc"]
    
    if not set(required_cols).issubset(df.columns):
        raise ValueError(f"Input CSV must contain columns: {required_cols}")

    return pool_from_arrays(
        df["logit_pr_auc"].values,
        df["var_logit_pr_auc"].values,
        alpha=alpha
    )
