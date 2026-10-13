"""
================================================================================
Evaluation and Bootstrap Sampling Variance Estimation for Meta-Analysis
================================================================================

Authors: Claudio Mazzi, et al.
Affiliation: Sant'Anna School of Advanced Studies (MeS Lab) / RETIN Project Consortium
Target Journal: Journal of the Royal Statistical Society: Series C (Applied Statistics)

Description:
    This module computes non-parametric out-of-sample discriminative metrics
    and derives the sampling variance necessary for random-effects meta-analysis
    under decentralized data protection constraints.
    
    Mathematical Formulation:
      1. Point Discrimination: Precision-Recall Area Under Curve (PR-AUC) via Average
         Precision (AP) over holdout test sets.
      2. Non-parametric Stratified Bootstrap: Draws B independent resamples (keeping
         the positive and negative class counts fixed) to obtain empirical standard errors
         and 95% percentile confidence intervals without assuming Gaussianity.
      3. Logit Transformation and Delta Method: Because PR-AUC is bounded within (0, 1),
         effects are transformed to the unbounded logit scale:
             y_i = logit(PR-AUC_i) = ln( PR-AUC_i / (1 - PR-AUC_i) )
         The corresponding sampling variance v_i is computed using the first-order
         Taylor series (Delta method):
             v_i = Var*(PR-AUC_i) / [ PR-AUC_i^2 * (1 - PR-AUC_i)^2 ]
================================================================================
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Optional, Literal, Dict, Any
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, auc
from Config_RETIN import META

try:
    import pandas as pd
except ImportError:
    pd = None


@dataclass
class PRAUCResult:
    """
    Data container storing regional PR-AUC discriminative metrics and meta-analytic inputs.
    """
    region: str
    n_test: int
    prevalence: float
    pr_auc_ap: float
    pr_auc_ap_se: float
    pr_auc_ap_ci_lo: float
    pr_auc_ap_ci_hi: float
    pr_auc_trap: Optional[float] = None
    logit_pr_auc: Optional[float] = None
    var_logit_pr_auc: Optional[float] = None

    def to_row(self) -> Dict[str, Any]:
        """Convert dataclass to a flat dictionary for tabular persistence."""
        return asdict(self)


def _compute_point(y_true: np.ndarray, y_score: np.ndarray, method: Literal["ap", "trap"]) -> float:
    """
    Compute point estimate of the Area Under the Precision-Recall Curve.

    Parameters
    ----------
    y_true : np.ndarray
        Binary ground-truth clinical outcome array {0, 1}.
    y_score : np.ndarray
        Predicted continuous risk probabilities.
    method : Literal["ap", "trap"]
        "ap" for scikit-learn Average Precision (interpolated step function);
        "trap" for trapezoidal integration along empirical PR coordinates.
    """
    if method == "ap":
        return float(average_precision_score(y_true, y_score))
    elif method == "trap":
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        return float(auc(recall, precision))
    else:
        raise ValueError("Method must be 'ap' (Average Precision) or 'trap' (Trapezoidal AUC).")


def _bootstrap_pr_auc(
    y_true: np.ndarray,
    y_score: np.ndarray,
    B: int,
    rng: np.random.Generator,
    method: Literal["ap", "trap"]
) -> np.ndarray:
    """
    Execute stratified non-parametric bootstrap resampling on the test set.

    Preserves exact positive and negative sample sizes to maintain the natural class ratio.
    """
    pos_idx = np.flatnonzero(y_true == 1)
    neg_idx = np.flatnonzero(y_true == 0)
    n_pos, n_neg = len(pos_idx), len(neg_idx)

    if n_pos == 0 or n_neg == 0:
        raise ValueError("Both positive and negative samples required for stratified bootstrap.")

    boot_estimates = np.empty(B, dtype=float)
    for b in range(B):
        sample_pos = rng.choice(pos_idx, size=n_pos, replace=True)
        sample_neg = rng.choice(neg_idx, size=n_neg, replace=True)
        sample_idx = np.concatenate([sample_pos, sample_neg])
        boot_estimates[b] = _compute_point(y_true[sample_idx], y_score[sample_idx], method)

    return boot_estimates


def _logit(x: float, eps: float = 1e-10) -> float:
    """Compute bounded logit transformation: logit(x) = ln(x / (1 - x))."""
    x_clipped = float(np.clip(x, eps, 1.0 - eps))
    return float(np.log(x_clipped / (1.0 - x_clipped)))


def _var_logit_from_se(point_est: float, se_est: float, eps: float = 1e-10) -> float:
    """
    Derive the asymptotic sampling variance of logit(PR-AUC) via the Delta method:
        Var(logit(p)) = Var(p) / [ p^2 * (1 - p)^2 ]
    """
    p = float(np.clip(point_est, eps, 1.0 - eps))
    return float((se_est ** 2) / (p ** 2 * (1.0 - p) ** 2))


def eval_pr_auc_for_meta(
    y_true: np.ndarray,
    y_score: np.ndarray,
    region: str,
    *,
    B: int = 2000,
    seed: int = 42,
    include_trapezoid: bool = True,
    add_logit_columns: bool = True,
    save_csv: Optional[str] = META,
    verbose: bool = True
) -> PRAUCResult:
    """
    Evaluate PR-AUC with stratified bootstrap standard errors and logit delta-method variance.

    Parameters
    ----------
    y_true : np.ndarray
        1D array of ground truth binary labels {0, 1}.
    y_score : np.ndarray
        1D array of continuous predicted probabilities.
    region : str
        Region identifier code ('TY', 'LY', 'SY').
    B : int, default=2000
        Number of bootstrap replicates (>= 1,000 recommended for stable variance estimation).
    seed : int, default=42
        Random seed for exact numerical reproducibility.
    include_trapezoid : bool, default=True
        Compute auxiliary trapezoidal PR-AUC for sensitivity verification.
    add_logit_columns : bool, default=True
        Derive logit-transformed estimate and delta-method sampling variance.
    save_csv : Optional[str], default=META
        Filesystem path to append or create the regional summary CSV table.
    verbose : bool, default=True
        Print formatted statistical output to stdout.

    Returns
    -------
    PRAUCResult
        Dataclass containing regional point estimates, standard errors, and meta-analytic inputs.
    """
    y_true_arr = np.asarray(y_true).ravel()
    y_score_arr = np.asarray(y_score).ravel()

    if y_true_arr.shape[0] != y_score_arr.shape[0]:
        raise ValueError("Length mismatch between ground truth labels and predicted scores.")

    n_test = int(y_true_arr.shape[0])
    prevalence = float(np.mean(y_true_arr == 1))
    rng = np.random.default_rng(seed)

    # 1. Point estimate (Average Precision)
    ap_point = _compute_point(y_true_arr, y_score_arr, method="ap")

    # 2. Stratified non-parametric bootstrap resampling
    boots = _bootstrap_pr_auc(y_true_arr, y_score_arr, B=B, rng=rng, method="ap")
    ap_se = float(boots.std(ddof=1))
    ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5])
    ap_ci_lo, ap_ci_hi = float(ci_lo), float(ci_hi)

    # 3. Auxiliary trapezoidal calculation
    pr_trap = _compute_point(y_true_arr, y_score_arr, method="trap") if include_trapezoid else None

    # 4. Meta-analytic logit scale transformation and delta-method variance
    logit_val = var_logit = None
    if add_logit_columns:
        logit_val = _logit(ap_point)
        var_logit = _var_logit_from_se(ap_point, ap_se)

    result = PRAUCResult(
        region=region,
        n_test=n_test,
        prevalence=prevalence,
        pr_auc_ap=ap_point,
        pr_auc_ap_se=ap_se,
        pr_auc_ap_ci_lo=ap_ci_lo,
        pr_auc_ap_ci_hi=ap_ci_hi,
        pr_auc_trap=pr_trap,
        logit_pr_auc=logit_val,
        var_logit_pr_auc=var_logit
    )

    # Append row to CSV database for downstream meta-analytic synthesis
    if save_csv is not None:
        if pd is None:
            raise RuntimeError("pandas is required to persist results to CSV.")
        try:
            df_existing = pd.read_csv(save_csv)
            # Remove previous run for the same region if present
            df_existing = df_existing[df_existing["region"] != region]
            df_updated = pd.concat([df_existing, pd.DataFrame([result.to_row()])], ignore_index=True)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            df_updated = pd.DataFrame([result.to_row()])

        df_updated.to_csv(save_csv, index=False)
        if verbose:
            print(f"[Meta-Export] Successfully written row for region '{region}' to {save_csv}")

    if verbose:
        print(f"[{region}] Holdout test sample: N = {n_test:,}, Baseline prevalence = {prevalence:.4f}")
        print(f"[{region}] PR-AUC (Average Precision): {ap_point:.4f} +/- {ap_se:.4f} (95% CI: [{ap_ci_lo:.4f}, {ap_ci_hi:.4f}])")
        if add_logit_columns:
            print(f"[{region}] logit(PR-AUC): {logit_val:.4f} | Sampling Variance v_i: {var_logit:.6f}")

    return result
