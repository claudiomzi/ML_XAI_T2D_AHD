"""
================================================================================
Multi-Regional Explainability Synthesis: SHAP Random-Effects & Beta-Binomial
================================================================================

Authors: Claudio Mazzi, et al.
Affiliation: Sant'Anna School of Advanced Studies (MeS Lab) / RETIN Project Consortium
Target Journal: Journal of the Royal Statistical Society: Series C (Applied Statistics)

Description:
    This module provides the statistical infrastructure for synthesizing local
    model explanations (SHAP values) across decentralized regional healthcare cohorts
    without exchanging individual patient records (GDPR compliance).

    Analytical Architecture:
      1. Per-Region Feature Summarization (`summarize_shap_region_from_matrix`):
         Extracts mean absolute SHAP magnitudes (mean |SHAP|), bootstrap standard errors,
         patient-level Top-k appearance frequencies, and directional attribution signs.
      2. Feature Attribution Magnitude Synthesis (`pool_shap_magnitude`):
         Executes random-effects meta-analysis with Hartung-Knapp-Sidik-Jonkman (HKSJ)
         adjustment on log-transformed positive attributions: log(mean |SHAP| + c).
      3. Patient-Level Feature Frequency Synthesis (`pool_shap_topk`):
         Models discrete appearance counts via a hierarchical Beta-Binomial model (BB1N):
             X_ij ~ Binomial(n_i, P_ij),   P_ij ~ Beta(alpha_j, beta_j)
         where alpha_j = mu_j * (1 - rho_j) / rho_j, beta_j = (1 - mu_j) * (1 - rho_j) / rho_j.
         Fits common pooled probability mu_j and intra-cluster correlation/overdispersion
         parameter rho_j via maximum likelihood estimation with numerical Hessian inversion
         and t-based confidence intervals (df = 2K - 2).
================================================================================
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln
from scipy.stats import t as student_t


# ==============================================================================
# 1. PER-REGION LOCAL EXPLAINABILITY SUMMARIZATION
# ==============================================================================

def summarize_shap_region_from_matrix(
    svk: np.ndarray,
    feature_names: List[str],
    region: str,
    topk: int = 5,
    B_boot: int = 0,
    seed: int = 42,
    save_csv: Optional[str] = None
) -> pd.DataFrame:
    """
    Construct a tidy summary table of local SHAP attributions from a regional matrix.

    Computes global importance magnitudes, patient-level top-k ranking frequencies,
    and directional sign proportions without requiring individual record exchange.

    Parameters
    ----------
    svk : np.ndarray
        Array of local SHAP values for the positive class (COMP=1), shape (N_patients, P_features).
    feature_names : List[str]
        Ordered list of predictor feature names.
    region : str
        Regional health authority identifier ('TY', 'LY', 'SY').
    topk : int, default=5
        Threshold for evaluating top-ranking predictors per patient profile.
    B_boot : int, default=0
        Number of bootstrap iterations for standard error estimation (0 uses analytic SE = SD / sqrt(N)).
    seed : int, default=42
        Random seed for bootstrap reproducibility.
    save_csv : Optional[str], default=None
        Path to persist the regional summary CSV table.

    Returns
    -------
    pd.DataFrame
        Aggregated summary table containing feature-level counts and summary statistics.
    """
    S = np.asarray(svk, dtype=float)
    n_patients, n_features = S.shape
    assert len(feature_names) == n_features, "Feature names length must match SHAP matrix columns."

    abs_S = np.abs(S)
    mean_abs = abs_S.mean(axis=0)

    # Standard error calculation: patient-level bootstrap or analytic approximation
    if B_boot and B_boot > 0:
        rng = np.random.default_rng(seed)
        boot_matrix = np.empty((B_boot, n_features), dtype=float)
        indices = np.arange(n_patients)
        for b in range(B_boot):
            boot_idx = rng.choice(indices, size=n_patients, replace=True)
            boot_matrix[b] = abs_S[boot_idx].mean(axis=0)
        se_abs = boot_matrix.std(axis=0, ddof=1)
    else:
        se_abs = abs_S.std(axis=0, ddof=1) / np.sqrt(n_patients)

    # Compute patient-level Top-k appearance frequency
    k_bound = min(int(topk), n_features)
    top_indices = np.argpartition(abs_S, -k_bound, axis=1)[:, -k_bound:]
    top_mask = np.zeros_like(abs_S, dtype=bool)
    row_indices = np.repeat(np.arange(n_patients), k_bound)
    top_mask[row_indices, top_indices.flatten()] = True
    
    topk_success = top_mask.sum(axis=0)
    topk_trials = np.full(n_features, n_patients, dtype=int)

    # Directional sign consistency (SHAP > 0: risk escalation)
    dir_pos_success = (S > 0).sum(axis=0)
    dir_trials = np.full(n_features, n_patients, dtype=int)

    summary_df = pd.DataFrame({
        "region": region,
        "feature": feature_names,
        "n_patients": n_patients,
        "mean_abs_shap": mean_abs,
        "se_mean_abs_shap": se_abs,
        "topk_success": topk_success,
        "topk_trials": topk_trials,
        "dir_pos_success": dir_pos_success,
        "dir_trials": dir_trials,
        "topk_k": k_bound
    })

    if save_csv:
        summary_df.to_csv(save_csv, index=False)
        print(f"[{region}] Exported regional SHAP summary table -> {save_csv}")

    return summary_df


# ==============================================================================
# 2. RANDOM-EFFECTS META-ANALYSIS ON LOG-ATTRIBUTION MAGNITUDES
# ==============================================================================

@dataclass
class REHKSJResult:
    """Container for random-effects meta-analytic synthesis of feature magnitudes."""
    k: int
    theta_re: float
    se_hksj: float
    df: int
    ci_param: Tuple[float, float]
    pooled_back: float
    ci_back: Tuple[float, float]
    tau2: float
    I2: float
    Q: float


def _hksj_pool(y: np.ndarray, v: np.ndarray, alpha: float = 0.05) -> Tuple[float, float, float, float, float]:
    """Execute DerSimonian-Laird tau^2 and Hartung-Knapp variance pooling."""
    w = 1.0 / v
    ybar_fe = np.sum(w * y) / np.sum(w)
    Q = float(np.sum(w * (y - ybar_fe) ** 2))
    c = float(np.sum(w) - (np.sum(w ** 2) / np.sum(w)))
    tau2 = max(0.0, (Q - (len(y) - 1)) / c) if c > 0 else 0.0

    w_re = 1.0 / (v + tau2)
    theta_re = float(np.sum(w_re * y) / np.sum(w_re))

    Qe = np.sum(w_re * (y - theta_re) ** 2)
    V_HK = Qe / ((len(y) - 1) * np.sum(w_re))
    V_classic = 1.0 / np.sum(w_re)
    V_HK = float(max(V_HK, V_classic))

    return theta_re, np.sqrt(V_HK), tau2, Q, V_HK


def re_hksj_on_log(values: np.ndarray, se_values: np.ndarray, alpha: float = 0.05, eps: float = 1e-6) -> REHKSJResult:
    """
    Perform random-effects meta-analysis on log-transformed positive attributions: log(mean |SHAP| + eps).
    """
    x = np.asarray(values, float) + eps
    se = np.asarray(se_values, float)
    y = np.log(x)
    v = (se / x) ** 2

    theta_re, se_hksj, tau2, Q, _ = _hksj_pool(y, v, alpha=alpha)
    k = len(y)
    df = k - 1
    t_crit = student_t.ppf(1.0 - alpha / 2.0, df)
    lo, hi = theta_re - t_crit * se_hksj, theta_re + t_crit * se_hksj

    pooled_val = float(np.exp(theta_re) - eps)
    ci_val = (float(np.exp(lo) - eps), float(np.exp(hi) - eps))
    I2 = max(0.0, (Q - df) / Q) * 100.0 if Q > 0 else 0.0

    return REHKSJResult(
        k=k,
        theta_re=float(theta_re),
        se_hksj=float(se_hksj),
        df=df,
        ci_param=(float(lo), float(hi)),
        pooled_back=pooled_val,
        ci_back=ci_val,
        tau2=float(tau2),
        I2=float(I2),
        Q=float(Q)
    )


# ==============================================================================
# 3. HIERARCHICAL BETA-BINOMIAL POOLING FOR DISCRETE PROPORTIONS
# ==============================================================================

@dataclass
class BBResult:
    """Container for common-parameter Beta-Binomial meta-analysis estimates."""
    k: int
    mu_hat: float
    ci: Tuple[float, float]
    se_logit_mu: float
    logit_mu_hat: float
    rho_hat: float
    df: int


def _neg_loglik_bb(params: np.ndarray, x: np.ndarray, n: np.ndarray, eps: float = 1e-9) -> float:
    """
    Compute negative log-likelihood of common-parameter Beta-Binomial model across regional clusters.
    
    Reparameterization:
      eta = logit(mu), gamma = logit(rho)
      alpha = mu * (1 - rho) / rho, beta = (1 - mu) * (1 - rho) / rho
    """
    eta, gamma = params
    mu = float(np.clip(1.0 / (1.0 + np.exp(-eta)), eps, 1.0 - eps))
    rho = float(np.clip(1.0 / (1.0 + np.exp(-gamma)), eps, 1.0 - eps))

    alpha = mu * (1.0 - rho) / rho
    beta = (1.0 - mu) * (1.0 - rho) / rho

    log_likelihood = np.sum(betaln(x + alpha, n - x + beta) - betaln(alpha, beta))
    return -float(log_likelihood)


def _numerical_hessian(func, x0: np.ndarray, h: float = 1e-4) -> np.ndarray:
    """Compute central-difference numerical Hessian matrix for information inversion."""
    x0 = np.asarray(x0, float)
    f0 = func(x0)
    H = np.zeros((2, 2), float)
    for i in range(2):
        e = np.zeros(2)
        e[i] = 1.0
        f_plus = func(x0 + h * e)
        f_minus = func(x0 - h * e)
        H[i, i] = (f_plus - 2.0 * f0 + f_minus) / (h ** 2)

    e1 = np.array([1.0, 0.0])
    e2 = np.array([0.0, 1.0])
    f_pp = func(x0 + h * e1 + h * e2)
    f_pm = func(x0 + h * e1 - h * e2)
    f_mp = func(x0 - h * e1 + h * e2)
    f_mm = func(x0 - h * e1 - h * e2)
    H[0, 1] = H[1, 0] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h ** 2)
    return H


def beta_binomial_pool(successes: List[int], trials: List[int], alpha: float = 0.05) -> BBResult:
    """
    Fit common-parameter Beta-Binomial model via MLE and construct t-based confidence intervals.

    Parameters
    ----------
    successes : List[int]
        Counts of positive events (e.g., patient-level top-k ranking appearances).
    trials : List[int]
        Total sample sizes per regional study cohort.
    alpha : float, default=0.05
        Significance level.

    Returns
    -------
    BBResult
        Dataclass containing pooled probability mu_hat, overdispersion rho_hat, and CI bounds.
    """
    x = np.asarray(successes, dtype=float)
    n = np.asarray(trials, dtype=float)
    K = len(x)
    df = 2 * K - 2

    # Initialization
    eps = 1e-9
    mu0 = np.clip(x.sum() / n.sum(), eps, 1.0 - eps)
    rho0 = 0.05
    x0 = np.array([np.log(mu0 / (1.0 - mu0)), np.log(rho0 / (1.0 - rho0))], float)

    # Maximum Likelihood Optimization
    obj = lambda par: _neg_loglik_bb(par, x, n, eps=eps)
    res = minimize(obj, x0, method="L-BFGS-B")
    eta_hat, gamma_hat = res.x

    logit_mu = float(eta_hat)
    mu_hat = float(1.0 / (1.0 + np.exp(-eta_hat)))
    rho_hat = float(1.0 / (1.0 + np.exp(-gamma_hat)))

    # Compute observed Fisher information via numerical Hessian
    H = _numerical_hessian(obj, np.array([eta_hat, gamma_hat]), h=1e-4)
    try:
        cov = np.linalg.inv(H)
        se_eta = float(np.sqrt(max(cov[0, 0], 1e-12)))
    except np.linalg.LinAlgError:
        se_eta = 1.0

    # Student-t confidence interval on the logit scale
    t_crit = student_t.ppf(1.0 - alpha / 2.0, df=df)
    lo_eta = logit_mu - t_crit * se_eta
    hi_eta = logit_mu + t_crit * se_eta
    inv_logit = lambda z: float(1.0 / (1.0 + np.exp(-z)))
    ci_prob = (inv_logit(lo_eta), inv_logit(hi_eta))

    return BBResult(
        k=K,
        mu_hat=mu_hat,
        ci=ci_prob,
        se_logit_mu=se_eta,
        logit_mu_hat=logit_mu,
        rho_hat=rho_hat,
        df=df
    )


# ==============================================================================
# 4. HIGH-LEVEL DRIVERS FOR CROSS-REGIONAL SYNTHESIS
# ==============================================================================

def pool_shap_magnitude(csv_paths: List[str], alpha: float = 0.05, eps: float = 1e-6) -> pd.DataFrame:
    """Pool feature importance magnitudes across regional CSV outputs."""
    df_all = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)
    results = []
    
    for feat, group in df_all.groupby("feature", sort=False):
        vals = group["mean_abs_shap"].to_numpy(float)
        ses = group["se_mean_abs_shap"].to_numpy(float)
        valid_mask = np.isfinite(vals) & np.isfinite(ses)
        vals, ses = vals[valid_mask], ses[valid_mask]
        
        if len(vals) < 2:
            continue
            
        res = re_hksj_on_log(vals, ses, alpha=alpha, eps=eps)
        results.append({
            "feature": feat,
            "k": res.k,
            "pooled_mean_abs_shap": res.pooled_back,
            "ci_low": res.ci_back[0],
            "ci_high": res.ci_back[1],
            "tau2": res.tau2,
            "I2": res.I2
        })

    return pd.DataFrame(results).sort_values("pooled_mean_abs_shap", ascending=False)


def pool_shap_topk(csv_paths: List[str], alpha: float = 0.05) -> pd.DataFrame:
    """Pool patient-level top-k appearance frequencies via Beta-Binomial modeling."""
    df_all = pd.concat([pd.read_csv(p) for p in csv_paths], ignore_index=True)
    results = []

    for feat, group in df_all.groupby("feature", sort=False):
        succ = group["topk_success"].to_numpy(int)
        trials = group["topk_trials"].to_numpy(int)
        valid_mask = np.isfinite(succ) & np.isfinite(trials) & (trials > 0)
        succ, trials = succ[valid_mask], trials[valid_mask]

        if len(succ) < 2:
            continue

        bb = beta_binomial_pool(succ.tolist(), trials.tolist(), alpha=alpha)
        results.append({
            "feature": feat,
            "k": bb.k,
            "pooled_topk_rate": bb.mu_hat,
            "ci_low": bb.ci[0],
            "ci_high": bb.ci[1],
            "rho_hat": bb.rho_hat,
            "df": bb.df
        })

    return pd.DataFrame(results).sort_values("pooled_topk_rate", ascending=False)


def make_all_shap_meta_outputs(csv_paths: List[str], out_prefix: str = "shap_meta") -> Dict[str, pd.DataFrame]:
    """Execute end-to-end synthesis of SHAP magnitudes and frequencies across regions."""
    mag_df = pool_shap_magnitude(csv_paths)
    mag_df.to_csv(f"{out_prefix}_magnitude.csv", index=False)

    topk_df = pool_shap_topk(csv_paths)
    topk_df.to_csv(f"{out_prefix}_topk.csv", index=False)

    print(f"[Synthesis] Completed. Exported '{out_prefix}_magnitude.csv' and '{out_prefix}_topk.csv'.")
    return {"magnitude": mag_df, "topk": topk_df}
