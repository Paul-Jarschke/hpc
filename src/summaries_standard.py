"""
Per-run tidy summary tables for the standard_model experiment - the on-node summariser
for the SINGLE-normal-component HBMNL (Rossi section 5.4; jobs 200-202).

Mirror of src/summaries.py (the mixture summariser), specialised to K = 1: there is no
component axis, no pvec, no label switching - so no ECR/weights/pvec tables, and every
per-parameter convergence number is directly meaningful. Every numeric quantity the
study's standard analysis notebook (standard_analysis_template.ipynb) and the standard
model-comparison notebook inspect is written as a tidy per-run table:

  * mu_recovery      - population mean vs TRUE_MU (summarize_mu, numeric form)
  * sigma_recovery   - POSTERIOR SIGMA: every lower-triangle element's posterior
                       mean/std/CI vs TRUE_SIGMA plus the empirical covariance of the
                       true unit betas (plot_final_covariance_complete, numeric form)
  * delta_recovery   - Delta elements vs TRUE_DELTA (shared with the mixture pipeline)
  * beta_recovery /
    beta_summary     - unit-level coefficients vs TRUE_BETA (shared)
  * convergence      - arviz R-hat/ESS for mu, Sigma elements, tr(Sigma), Delta and the
                       two notebook-inspected households' betas (first + last unit)
  * moments          - E[beta] and Var[beta] vs truth (Rossi Eq. 5.5.2 with K=1)
  * marginal_distances / marginal_diagnostics - marginal densities vs the TRUE DGP on
                       BOTH grid scenarios (full / chebyshev), via the K=1 packaging of
                       marginal_comparison (load_sampler_standard convention)
  * runs, diagnostics - run rollup + per-kernel sampler diagnostics (NUTS/HMC)

Posterior keys expected (plain, no _k suffix): mu (C,S,P), sigma_inv_chol_latent
(C,S,P(P+1)/2), beta_i (C,S,N,P), Delta (C,S,D,P) when demographics are present.
"""

import numpy as np
import pandas as pd
import arviz as az

from . import analysis
from . import marginal_comparison as mc
from .summaries import (
    delta_summary_rows, beta_summary_rows,
    beta_recovery_rows, delta_recovery_rows, diagnostics_rows,
)

# Identifying columns every row carries (same as the mixture pipeline, so shared
# gather/plot tooling can treat both experiments uniformly; scenario == "standard",
# k_true == k_model == 1 throughout).
COND_KEYS = ("dataset_key", "scenario", "k_true", "data_seed", "k_model", "sampler", "n_chains")

TABLE_NAMES = ("runs", "convergence", "moments", "mu_recovery", "sigma_recovery",
               "delta_recovery", "beta_recovery", "beta_summary", "diagnostics",
               "marginal_distances", "marginal_diagnostics")


# --------------------------------------------------------------------------- #
# In-memory model packaging (K=1 component axis, pvec == 1), mirroring
# marginal_comparison.load_sampler_standard for an in-memory posterior dict.
# --------------------------------------------------------------------------- #
def build_model_standard(post, name, duration_s=None):
    """duration_s (the fit's total wall-clock, incl. warmup) rides along so
    marginal_comparison.functional_diagnostics can report ESS_bulk/s and ESS_tail/s."""
    mu = np.asarray(post["mu"])[:, :, None, :]                                   # (C,S,1,P)
    Sigma = np.asarray(analysis._sigma_from_latent(
        np.asarray(post["sigma_inv_chol_latent"])[:, :, None, :]))               # (C,S,1,P,P)
    std = np.sqrt(np.clip(np.diagonal(Sigma, axis1=3, axis2=4), 0.0, None))      # (C,S,1,P)
    pvec = np.ones(mu.shape[:3])                                                 # (C,S,1)
    return {"name": name, "mu": mu, "pvec": pvec, "Sigma": Sigma, "std": std,
            "is_mcmc": True, "duration_s": duration_s}


# --------------------------------------------------------------------------- #
# Parameter-recovery tables
# --------------------------------------------------------------------------- #
def mu_recovery_rows(post, truth, cond, param_names):
    """Population mean mu vs TRUE_MU: one row per parameter (summarize_mu, numeric)."""
    mu = np.asarray(post["mu"])                                # (C,S,P)
    P = mu.shape[-1]
    flat = mu.reshape(-1, P)
    true_mu = np.asarray(truth["TRUE_MU"], dtype=float).ravel()
    rows = []
    for p in range(P):
        d = flat[:, p]
        lo, hi = np.percentile(d, [2.5, 97.5])
        tv = float(true_mu[p])
        rows.append({**cond, "param": param_names[p],
                     "post_mean": float(d.mean()), "post_std": float(d.std()),
                     "ci_low": float(lo), "ci_high": float(hi), "true_value": tv,
                     "abs_diff": abs(tv - float(d.mean())),
                     "in_ci": bool(lo <= tv <= hi)})
    return rows


def sigma_recovery_rows(model, truth, cond, param_names):
    """POSTERIOR SIGMA vs TRUE_SIGMA, one row per lower-triangle element (incl. diagonal),
    plus the empirical covariance of the true unit betas - the numeric form of the
    notebook's plot_final_covariance_complete cell."""
    Sigma = np.asarray(model["Sigma"])[:, :, 0]                # (C,S,P,P)
    P = Sigma.shape[-1]
    flat = Sigma.reshape(-1, P, P)
    true_sig = np.asarray(truth["TRUE_SIGMA"], dtype=float)
    emp = np.cov(np.asarray(truth["TRUE_BETA"], dtype=float), rowvar=False)
    rows = []
    for i in range(P):
        for j in range(i + 1):                                 # lower triangle incl. diagonal
            d = flat[:, i, j]
            lo, hi = np.percentile(d, [2.5, 97.5])
            tv = float(true_sig[i, j])
            rows.append({**cond, "row": param_names[i], "col": param_names[j],
                         "post_mean": float(d.mean()), "post_std": float(d.std()),
                         "ci_low": float(lo), "ci_high": float(hi),
                         "true_value": tv, "empirical": float(emp[i, j]),
                         "abs_diff": abs(tv - float(d.mean())),
                         "in_ci": bool(lo <= tv <= hi)})
    return rows


# --------------------------------------------------------------------------- #
# Convergence (K=1 -> per-parameter R-hat/ESS is directly meaningful)
# --------------------------------------------------------------------------- #
def _rhat_ess(a):
    return float(az.rhat(np.asarray(a))), float(az.ess(np.asarray(a)))


def convergence_rows(post, model, cond, param_names, demo_names):
    """arviz R-hat/ESS over the real (chains, draws) series for: mu[p], every
    lower-triangle Sigma element, tr(Sigma), Delta[d,p], and the betas of the two
    households the notebook inspects (first and last unit)."""
    rows = []
    mu = np.asarray(post["mu"])                                # (C,S,P)
    P = mu.shape[-1]
    for p in range(P):
        r, e = _rhat_ess(mu[:, :, p])
        rows.append({**cond, "quantity": f"mu:{param_names[p]}", "rhat": r, "ess": e})

    Sigma = np.asarray(model["Sigma"])[:, :, 0]                # (C,S,P,P)
    for i in range(P):
        for j in range(i + 1):
            r, e = _rhat_ess(Sigma[:, :, i, j])
            rows.append({**cond, "quantity": f"Sigma:{param_names[i]},{param_names[j]}",
                         "rhat": r, "ess": e})
    r, e = _rhat_ess(np.trace(Sigma, axis1=-2, axis2=-1))
    rows.append({**cond, "quantity": "tr(Sigma)", "rhat": r, "ess": e})

    if "Delta" in post:
        Delta = np.asarray(post["Delta"])                      # (C,S,D,P)
        D = Delta.shape[-2]
        for dd in range(D):
            for p in range(P):
                r, e = _rhat_ess(Delta[:, :, dd, p])
                rows.append({**cond, "quantity": f"Delta:{demo_names[dd]}:{param_names[p]}",
                             "rhat": r, "ess": e})

    beta = np.asarray(post["beta_i"])                          # (C,S,N,P)
    N = beta.shape[-2]
    for unit in (0, N - 1):
        for p in range(P):
            r, e = _rhat_ess(beta[:, :, unit, p])
            rows.append({**cond, "quantity": f"beta[unit{unit}]:{param_names[p]}",
                         "rhat": r, "ess": e})
    return rows


# --------------------------------------------------------------------------- #
# THE per-run entry point: every tidy table for ONE standard-model run.
# --------------------------------------------------------------------------- #
def per_run_tables(post, meta, truth, diag=None):
    """Build every PER-RUN tidy table from one standard-model run's posterior + truth.

    post  : posterior dict (mu, sigma_inv_chol_latent, beta_i, [Delta]) - plain keys.
    meta  : that run's meta.json dict (COND_KEYS + n_params + runtime/rollup fields).
    truth : the dataset's ground-truth dict (param_names, demo_names, Z, TRUE_*).
    diag  : optional sampler-diagnostics dict (None for bayesm).

    Returns (tables, model): `tables` maps each name in TABLE_NAMES to a list of row
    dicts; `model` is the in-memory K=1 model dict (marginal_comparison convention).
    """
    cond = {k: meta[k] for k in COND_KEYS}
    P = int(meta["n_params"])
    param_names = list(truth["param_names"])
    D = int(truth.get("n_demos", 0) or 0)
    demo_names = list(truth.get("demo_names", [f"z{d + 1}" for d in range(D)]))
    tables = {}

    model = build_model_standard(post, cond["sampler"], duration_s=meta.get("runtime_s"))

    # convergence first, so its rollup can ride along in the runs row.
    crows = convergence_rows(post, model, cond, param_names, demo_names)
    tables["convergence"] = crows
    rhat_max = float(np.nanmax([r["rhat"] for r in crows]))
    ess_min = float(np.nanmin([r["ess"] for r in crows]))

    tables["runs"] = [{**cond, "runtime_s": meta.get("runtime_s"),
                       "n_sampling_errors": meta.get("n_sampling_errors"),
                       "rhat_max": rhat_max, "ess_min": ess_min,
                       "n_divergent": meta.get("n_divergent"),
                       "max_treedepth": meta.get("max_treedepth"),
                       "max_leapfrog": meta.get("max_leapfrog"),
                       "mean_acceptance": meta.get("mean_acceptance")}]

    # moments (Rossi Eq. 5.5.2 with K=1: E[beta]=E[mu], Var = E[Sigma] + Var[mu]) vs true.
    mean, var = mc.mixture_moments(model)
    tmean, tvar = mc.mixture_moments(mc.true_dgp_standard(truth))
    tables["moments"] = [{**cond, "param": param_names[j], "mix_mean": float(mean[j]),
                          "mix_var": float(var[j, j]), "true_mix_mean": float(tmean[j]),
                          "true_mix_var": float(tvar[j, j])} for j in range(P)]

    # parameter recovery vs ground truth.
    tables["mu_recovery"] = mu_recovery_rows(post, truth, cond, param_names)
    tables["sigma_recovery"] = sigma_recovery_rows(model, truth, cond, param_names)
    tables["delta_recovery"] = delta_recovery_rows(post, truth, cond, param_names)
    tables["beta_recovery"] = beta_recovery_rows(post, truth, cond, param_names)
    tables["beta_summary"] = beta_summary_rows(
        post.get("beta_i"), truth.get("TRUE_BETA"), param_names, cond=cond)

    # sampler diagnostics (per kernel/block); empty for bayesm.
    tables["diagnostics"] = diagnostics_rows(diag, cond)

    # marginal-density comparison vs the TRUE DGP marginal, on the SAME two grid
    # scenarios as the mixture pipeline ("full" + "chebyshev"); with K=1 the whole
    # marginal_comparison machinery applies unchanged.
    true_model = mc.true_dgp_standard(truth)
    grid_scenarios = {
        "full":      mc.build_grids_full([model], true_model, n_grid=1000, n_sigma=6),
        "chebyshev": mc.build_grids_chebyshev([model], true_model, n_grid=1000, k=5.0),
    }
    mdist = []
    for grid_name, grids in grid_scenarios.items():
        for (_, param), r in mc.distance_table([model], true_model, grids, param_names).iterrows():
            mdist.append({**cond, "grid": grid_name, "param": param, **r.to_dict()})
    tables["marginal_distances"] = mdist

    # marginal convergence: Goose-identical arviz diagnostics on grid-free functionals
    # (mean, sd, q05/q50/q95) of each per-draw marginal N(mu, Sigma). Rhat is rank split-
    # R-hat; ESS_bulk / ESS_tail the bulk/tail ESS; ESS_bulk/s and ESS_tail/s the effective
    # draws per fit-second (from runtime_s via build_model_standard). Replaces the former
    # density_series_diagnostics / moment_series_diagnostics rhat/ess tables.
    mdiag = []
    for (param, functional), r in mc.functional_diagnostics(model, param_names).iterrows():
        mdiag.append({**cond, "param": param, "functional": functional, **r.to_dict()})
    tables["marginal_diagnostics"] = mdiag

    return tables, model
