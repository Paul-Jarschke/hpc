"""
Tidy summary tables for ONE mixture-HBMNL fit (harness glue, not vendored).

Collapses the study's separate fit/analysis phases into one: given the in-memory
posterior samples + the dataset's ground truth, build the tidy one-row-per-X
DataFrames that the HPC gather step concatenates. Reuses the label-switching-aware
logic from the vendored `src/analysis.py`.

Tables (each carries all `cond` columns, so every row is self-identifying):
  diagnostics    1 row    : runtime, label-invariant rhat_max / ess_min, sampling errors
  pvec_summary   K rows    : component weights, rank-matched to truth (descending)
  mu_summary     K*P rows  : component means, model->true matched via assignment
  sigma_summary  K*P rows  : component variances (Sigma diagonal), same matching as mu
  recovery       P rows    : unit-level beta recovery vs truth (bias / rmse / coverage)

`cond` is a dict of the run's condition columns (dataset_key, k_true, k_model,
sampler, n_chains, ...). It is prepended to every output row.
"""

import re

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .analysis import (
    _recover_pvec,
    _sigma_from_latent,
    invariant_convergence_summary,
)


def _rows(cond, records):
    """Prepend the condition columns to every metric record -> tidy DataFrame."""
    return pd.DataFrame([{**cond, **r} for r in records])


def parse_sampling_errors(log_text):
    """Count 'Errors per chain for kernel_NN: a / b transitions' lines in a goose log."""
    total = 0
    for line in log_text.splitlines():
        if "Errors per chain" not in line:
            continue
        m = re.search(r":\s*(\d+)\s*/\s*\d+\s*transitions", line)
        if m:
            total += int(m.group(1))
    return total


def _component_assignment(post_mu_mean, true_mu, K, K_true):
    """Map each model component k -> matched true component (or None for spurious),
    by minimum squared distance between posterior-mean and true means."""
    mapping = {k: None for k in range(K)}
    if true_mu is None:
        return mapping
    cost = np.sum((post_mu_mean[:, None, :] - true_mu[None, :K_true, :]) ** 2, axis=-1)
    row_ind, col_ind = linear_sum_assignment(cost)
    for r, c in zip(row_ind, col_ind):
        mapping[int(r)] = int(c)
    return mapping


def build_summaries(posterior_samples, raw, cond, runtime_s, sampling_log_text=""):
    """Return {table_name: DataFrame} for one fit."""
    K = int(cond["k_model"])
    K_true = int(raw["K"])
    P = int(raw["n_params"])
    param_names = list(raw["param_names"])

    def _opt(key):
        v = raw.get(key)
        return np.asarray(v) if v is not None else None

    true_pvec = _opt("TRUE_PVEC")
    true_mu = _opt("TRUE_MU_K")
    true_sigma = _opt("TRUE_SIGMA_K")
    true_beta = _opt("TRUE_BETA")

    tables = {}

    # ---- diagnostics (1 row) : label-invariant convergence + timing ----------
    try:
        conv = invariant_convergence_summary(posterior_samples, include_cov=True)
        rhat_max = float(np.nanmax(conv["rhat"].values))
        ess_min = float(np.nanmin(conv["ess"].values))
    except Exception:
        rhat_max, ess_min = np.nan, np.nan
    tables["diagnostics"] = _rows(cond, [{
        "runtime_s": round(float(runtime_s), 2),
        "rhat_max": rhat_max,
        "ess_min": ess_min,
        "n_sampling_errors": parse_sampling_errors(sampling_log_text),
        "converged": bool(rhat_max < 1.1) if np.isfinite(rhat_max) else None,
    }])

    # ---- pvec_summary (K rows, sorted descending by posterior mean) ----------
    pvec = _recover_pvec(posterior_samples).reshape(-1, K)
    order = np.argsort(pvec.mean(axis=0))[::-1]
    true_desc = np.sort(true_pvec)[::-1] if true_pvec is not None else None
    pvec_rows = []
    for rank, k in enumerate(order):
        draws = pvec[:, k]
        lo, hi = np.percentile(draws, [2.5, 97.5])
        rec = {
            "rank": int(rank), "component": int(k),
            "post_mean": float(draws.mean()), "post_std": float(draws.std()),
            "ci_low": float(lo), "ci_high": float(hi),
        }
        if true_desc is not None:
            if rank < K_true:
                rec["true_pvec"] = float(true_desc[rank])
                rec["true_in_ci"] = bool(lo <= true_desc[rank] <= hi)
            else:
                rec["true_pvec"] = np.nan
                rec["true_in_ci"] = np.nan
        pvec_rows.append(rec)
    tables["pvec_summary"] = _rows(cond, pvec_rows)

    # ---- model->true component matching (shared by mu & sigma) ---------------
    mu = np.asarray(posterior_samples["mu_k"]).reshape(-1, K, P)
    post_mu_mean = mu.mean(axis=0)
    mapping = _component_assignment(post_mu_mean, true_mu, K, K_true)

    # ---- mu_summary (K*P rows) -----------------------------------------------
    mu_rows = []
    for k in range(K):
        tk = mapping[k]
        for p in range(P):
            d = mu[:, k, p]
            rec = {
                "component": int(k), "matched_true": tk, "param": param_names[p],
                "post_mean": float(d.mean()), "post_std": float(d.std()),
            }
            rec["true_value"] = float(true_mu[tk, p]) if tk is not None else np.nan
            rec["diff_abs"] = float(abs(true_mu[tk, p] - d.mean())) if tk is not None else np.nan
            mu_rows.append(rec)
    tables["mu_summary"] = _rows(cond, mu_rows)

    # ---- sigma_summary (K*P rows, Sigma diagonal variances) ------------------
    Sigma = _sigma_from_latent(np.asarray(posterior_samples["sigma_inv_chol_k_latent"]))
    Sigma = Sigma.reshape(-1, K, P, P)
    diag = np.diagonal(Sigma, axis1=-2, axis2=-1)          # (R, K, P)
    diag_mean, diag_std = diag.mean(axis=0), diag.std(axis=0)
    true_var = np.diagonal(true_sigma, axis1=-2, axis2=-1) if true_sigma is not None else None
    sig_rows = []
    for k in range(K):
        tk = mapping[k]
        for p in range(P):
            rec = {
                "component": int(k), "matched_true": tk, "param": param_names[p],
                "post_var_mean": float(diag_mean[k, p]), "post_var_std": float(diag_std[k, p]),
            }
            if tk is not None and true_var is not None:
                rec["true_var"] = float(true_var[tk, p])
                rec["diff_abs"] = float(abs(true_var[tk, p] - diag_mean[k, p]))
            else:
                rec["true_var"] = np.nan
                rec["diff_abs"] = np.nan
            sig_rows.append(rec)
    tables["sigma_summary"] = _rows(cond, sig_rows)

    # ---- recovery (P rows, unit-level beta vs truth) -------------------------
    if true_beta is not None and "beta_i" in posterior_samples:
        n_units = true_beta.shape[0]
        beta = np.asarray(posterior_samples["beta_i"]).reshape(-1, n_units, P)
        bmean = beta.mean(axis=0)
        blo = np.percentile(beta, 2.5, axis=0)
        bhi = np.percentile(beta, 97.5, axis=0)
        rec_rows = []
        for p in range(P):
            err = bmean[:, p] - true_beta[:, p]
            cov = np.mean((true_beta[:, p] >= blo[:, p]) & (true_beta[:, p] <= bhi[:, p]))
            rec_rows.append({
                "param": param_names[p],
                "bias": float(err.mean()),
                "rmse": float(np.sqrt((err ** 2).mean())),
                "coverage_95": float(cov),
            })
        tables["recovery"] = _rows(cond, rec_rows)

    return tables
