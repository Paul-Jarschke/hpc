"""
Per-run tidy summary tables for the k5model_mixture experiment - the on-node summariser.

Every table that can be computed from ONE run's posterior + that run's ground truth
(recovery of Delta / beta / mu / Sigma / weights, ECR report, convergence, mixture
moments, sampler diagnostics) is built here, so the SAME code runs two ways:

  * on the compute node inside run.qmd, right after sampling -> writes tiny CSVs to
    out/<table>/<run_key>.csv (Tier 1: every run, no posterior leaves the node);
  * post-hoc in analysis/post_process.py over the saved posteriors (Tier 2 subset),
    which adds only the cross-sampler marginal comparison (the one analysis that needs
    several samplers' chains together).

This is the `summaries.py` referenced in src/README.md. The recovery math is unchanged
from the original post_process.py; it reuses analysis.* (label-invariant convergence,
Delta/beta element summaries), label_switching.* (ECR.iterative.1 relabel + component
convergence) and marginal_comparison.* (Rossi mixture moments). per_run_tables() returns
the per-run tables plus the in-memory mixture `model` dict so the caller can run the
cross-sampler step without rebuilding it.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import analysis
from . import label_switching as ls
from . import marginal_comparison as mc

# Identifying columns every row carries so the per-run CSVs concatenate cleanly. Must match
# the keys present in each run's meta.json (and analysis/post_process.COND_KEYS).
COND_KEYS = ("dataset_key", "scenario", "k_true", "data_seed", "k_model", "sampler", "n_chains")

# The full set of per-run table names per_run_tables() returns, in output order. (The
# cross-sampler marginal tables are NOT here - they need several runs together.)
TABLE_NAMES = ("runs", "ecr_report", "weights", "convergence", "moments",
               "mu_recovery", "sigma_recovery", "delta_recovery", "beta_recovery",
               "beta_summary", "diagnostics")


# --------------------------------------------------------------------------- #
# In-memory model + component matching
# --------------------------------------------------------------------------- #
def build_model(post, name):
    """Replicate marginal_comparison.load_sampler from an in-memory posterior dict."""
    mu = np.asarray(post["mu_k"])
    pvec = np.asarray(analysis._recover_pvec(post))
    Sigma = np.asarray(analysis._sigma_from_latent(np.asarray(post["sigma_inv_chol_k_latent"])))
    std = np.sqrt(np.clip(np.diagonal(Sigma, axis1=3, axis2=4), 0.0, None))
    return {"name": name, "mu": mu, "pvec": pvec, "Sigma": Sigma, "std": std, "is_mcmc": True}


def _match_components(post_mu_mean, true_mu, K_true):
    """Hungarian-match model components -> true components on squared mu distance.
    Returns {model_slot: true_idx} for the K_true matched slots (analysis.summarize_mu_k)."""
    cost = np.sum((post_mu_mean[:, None, :] - true_mu[None, :K_true, :]) ** 2, axis=-1)  # (K,K_true)
    row, col = linear_sum_assignment(cost)
    return {int(r): int(c) for r, c in zip(row, col)}


# --------------------------------------------------------------------------- #
# Parameter-recovery tables (ECR-relabeled posterior, so per-component means are
# not corrupted by label switching).
# --------------------------------------------------------------------------- #
def mu_recovery_rows(relabeled, truth, cond, K, K_true, param_names):
    """Per-component mu_k recovery vs TRUE_MU_K. Returns (rows, mapping) so the same
    model->true matching is reused for Sigma."""
    mu = np.asarray(relabeled["mu_k"])                          # (C,S,K,P)
    P = mu.shape[-1]
    flat = mu.reshape(-1, K, P)
    post_mean = flat.mean(axis=0)                              # (K,P)
    true_mu = np.asarray(truth["TRUE_MU_K"], dtype=float)      # (K_true,P)
    mapping = _match_components(post_mean, true_mu, K_true)
    rows = []
    for k in range(K):
        tk = mapping.get(k)
        for p in range(P):
            d = flat[:, k, p]
            lo, hi = np.percentile(d, [2.5, 97.5])
            tv = float(true_mu[tk, p]) if tk is not None else np.nan
            rows.append({**cond, "slot": k, "matched_true": (tk if tk is not None else np.nan),
                         "live": tk is not None, "param": param_names[p],
                         "post_mean": float(d.mean()), "post_std": float(d.std()),
                         "ci_low": float(lo), "ci_high": float(hi), "true_value": tv,
                         "abs_diff": (abs(tv - float(d.mean())) if tk is not None else np.nan),
                         "in_ci": (bool(lo <= tv <= hi) if tk is not None else np.nan)})
    return rows, mapping


def sigma_recovery_rows(relabeled, truth, cond, K, K_true, mapping, param_names):
    """Per-component Sigma_k recovery vs TRUE_SIGMA_K, plus the empirical covariance of
    the true betas assigned to that component (analysis.recover_covariance_matrices)."""
    Sigma = np.asarray(relabeled["Sigma"])                     # (C,S,K,P,P)
    P = Sigma.shape[-1]
    flat = Sigma.reshape(-1, K, P, P)
    true_sig = np.asarray(truth["TRUE_SIGMA_K"], dtype=float)  # (K_true,P,P)
    tbeta = np.asarray(truth["TRUE_BETA"], dtype=float)        # (N,P)
    tind = np.asarray(truth["TRUE_INDICATORS"]).ravel()        # (N,)
    rows = []
    for k in range(K):
        tk = mapping.get(k)
        emp = None
        if tk is not None:
            sub = tbeta[tind == tk]
            if sub.shape[0] > P:
                emp = np.cov(sub, rowvar=False)
        for i in range(P):
            for j in range(i + 1):                             # lower triangle incl. diagonal
                d = flat[:, k, i, j]
                lo, hi = np.percentile(d, [2.5, 97.5])
                tv = float(true_sig[tk, i, j]) if tk is not None else np.nan
                rows.append({**cond, "slot": k, "matched_true": (tk if tk is not None else np.nan),
                             "live": tk is not None, "row": param_names[i], "col": param_names[j],
                             "post_mean": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
                             "true_value": tv, "empirical": (float(emp[i, j]) if emp is not None else np.nan),
                             "abs_diff": (abs(tv - float(d.mean())) if tk is not None else np.nan),
                             "in_ci": (bool(lo <= tv <= hi) if tk is not None else np.nan)})
    return rows


def delta_recovery_rows(post, truth, cond, param_names):
    """Delta (demographic shift) recovery vs TRUE_DELTA. Delegates to analysis.delta_summary_rows."""
    if "Delta" not in post or truth.get("TRUE_DELTA") is None:
        return []
    D = np.asarray(post["Delta"]).shape[-2]                    # (C,S,D,P)
    demo_names = list(truth.get("demo_names", [f"demo{d}" for d in range(D)]))
    return analysis.delta_summary_rows(post["Delta"], truth["TRUE_DELTA"],
                                       demo_names, param_names, cond=cond)


def beta_recovery_rows(post, truth, cond, param_names):
    """Per-parameter beta_i recovery vs TRUE_BETA: bias / RMSE / 95% coverage aggregated
    over the N units (the compact 4-rows/run summary)."""
    beta = np.asarray(post["beta_i"])                          # (C,S,N,P)
    N, P = beta.shape[-2], beta.shape[-1]
    flat = beta.reshape(-1, N, P)
    bmean = flat.mean(axis=0)                                  # (N,P)
    lo, hi = np.percentile(flat, [2.5, 97.5], axis=0)          # (N,P) each
    tbeta = np.asarray(truth["TRUE_BETA"], dtype=float)        # (N,P)
    rows = []
    for p in range(P):
        err = bmean[:, p] - tbeta[:, p]
        cov = np.mean((lo[:, p] <= tbeta[:, p]) & (tbeta[:, p] <= hi[:, p]))
        rows.append({**cond, "param": param_names[p], "bias": float(err.mean()),
                     "rmse": float(np.sqrt((err ** 2).mean())),
                     "mean_abs_err": float(np.abs(err).mean()), "coverage95": float(cov)})
    return rows


def diagnostics_rows(diag, cond, max_treedepth=10):
    """Per-kernel sampler-diagnostics summary from an in-memory diagnostics dict (same
    structure as out/diagnostics/<key>.pkl: keys 'kernels_by_pos_key' + 'transition_infos').
    NUTS: divergences, treedepth saturation, leapfrog steps; all: acceptance. None -> []."""
    if not diag:
        return []
    inv = {v: k for k, v in diag.get("kernels_by_pos_key", {}).items()}   # kernel_id -> block name
    rows = []
    for kid, fl in diag.get("transition_infos", {}).items():
        row = {**cond, "block": inv.get(kid, kid), "kernel_id": kid}
        if "divergent" in fl:
            dv = np.asarray(fl["divergent"])
            row["n_divergent"], row["frac_divergent"] = int(dv.sum()), float(dv.mean())
        if "treedepth" in fl:
            td = np.asarray(fl["treedepth"])
            row["max_treedepth"], row["mean_treedepth"] = int(td.max()), float(td.mean())
            row["frac_max_treedepth"] = float(np.mean(td >= max_treedepth))
        if "leapfrog" in fl:
            lp = np.asarray(fl["leapfrog"])
            row["max_leapfrog"], row["mean_leapfrog"] = int(lp.max()), float(lp.mean())
        if "acceptance_prob" in fl:
            row["mean_acceptance"] = float(np.asarray(fl["acceptance_prob"]).mean())
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Partition recovery (allocation accuracy + adjusted Rand)
# --------------------------------------------------------------------------- #
def _contingency(true_lab, pred_lab):
    import pandas as pd
    ct = pd.factorize(true_lab)[0]
    cp = pd.factorize(pred_lab)[0]
    cont = np.zeros((ct.max() + 1, cp.max() + 1), dtype=np.int64)
    np.add.at(cont, (ct, cp), 1)
    return cont


def _adjusted_rand(true_lab, pred_lab):
    """Adjusted Rand index (label-permutation invariant) - no scikit-learn dependency."""
    from scipy.special import comb
    cont = _contingency(true_lab, pred_lab)
    n = int(true_lab.size)
    sum_c = comb(cont, 2).sum()
    sum_a = comb(cont.sum(axis=1), 2).sum()
    sum_b = comb(cont.sum(axis=0), 2).sum()
    expected = sum_a * sum_b / comb(n, 2)
    maxi = 0.5 * (sum_a + sum_b)
    return float(1.0 if maxi == expected else (sum_c - expected) / (maxi - expected))


def allocation_accuracy(post, report, truth, K, Z):
    """Partition recovery: reconstruct hard allocations z (Rossi 5.5.19), relabel them with
    the ECR permutations so labels are consistent across draws, take each unit's modal
    component, and score against TRUE_INDICATORS - Hungarian-matched accuracy + adjusted Rand."""
    tind = truth.get("TRUE_INDICATORS")
    if tind is None:
        return {"alloc_accuracy": np.nan, "adjusted_rand": np.nan}
    tind = np.asarray(tind).ravel()
    try:
        z, _ = ls.reconstruct_allocations(post, Z=Z)            # (C,S,N) raw labels
    except Exception:
        return {"alloc_accuracy": np.nan, "adjusted_rand": np.nan}
    C, S, N = z.shape
    z_flat = z.reshape(C * S, N).astype(np.int64)
    perm = report.get("permutations")
    if perm is not None and np.asarray(perm).shape[-1] == K:
        inv = np.argsort(np.asarray(perm), axis=2).reshape(C * S, K)   # raw label -> ECR slot
        z_flat = np.take_along_axis(inv, z_flat, axis=1)
    modal = np.array([np.bincount(z_flat[:, i], minlength=K).argmax() for i in range(N)])
    cont = _contingency(tind, modal)
    row, col = linear_sum_assignment(-cont)
    return {"alloc_accuracy": float(cont[row, col].sum() / N),
            "adjusted_rand": _adjusted_rand(tind, modal)}


# --------------------------------------------------------------------------- #
# THE per-run entry point: every tidy table for ONE run.
# --------------------------------------------------------------------------- #
def per_run_tables(post, meta, truth, diag=None):
    """Build every PER-RUN tidy table from one run's posterior + ground truth.

    post  : posterior dict (mu_k, sigma_inv_chol_k_latent, pvec, beta_i, [Delta]).
    meta  : that run's meta.json dict (COND_KEYS + n_params + the diag/runtime rollup fields).
    truth : the dataset's ground-truth dict (param_names, Z, TRUE_*).
    diag  : optional in-memory sampler-diagnostics dict (out/diagnostics/<key>.pkl shape);
            None for samplers without transition diagnostics (e.g. bayesm).

    Returns (tables, model): `tables` maps each name in TABLE_NAMES to a list of row dicts;
    `model` is the in-memory mixture model dict (reused by the caller's cross-sampler step).
    """
    cond = {k: meta[k] for k in COND_KEYS}
    K, K_true, P = int(meta["k_model"]), int(meta["k_true"]), int(meta["n_params"])
    param_names = list(truth["param_names"])
    Z = np.asarray(truth["Z"])
    tables = {}

    # runs: one row, the meta rollup (runtime, errors, invariant rhat/ess, diag rollup).
    tables["runs"] = [{**cond, "runtime_s": meta.get("runtime_s"),
                       "n_sampling_errors": meta.get("n_sampling_errors"),
                       "invariant_rhat_max": meta.get("invariant_rhat_max"),
                       "invariant_ess_min": meta.get("invariant_ess_min"),
                       "n_divergent": meta.get("n_divergent"),
                       "max_treedepth": meta.get("max_treedepth"),
                       "max_leapfrog": meta.get("max_leapfrog"),
                       "mean_acceptance": meta.get("mean_acceptance")}]

    # ECR.iterative.1 relabeling + honest verdict against the invariant gate.
    relabeled, report = ls.relabel_run(post, K=K, Z=Z, K_true=K_true)
    gate = analysis.invariant_convergence_summary(post, include_cov=True)
    verdict = ls.classify_outcome(report, gate)
    alloc = allocation_accuracy(post, report, truth, K, Z)
    tables["ecr_report"] = [{**cond, "converged": report["converged"], "n_iter": report["n_iter"],
                             "switching_rate": report["switching_rate"], "verdict": verdict["verdict"],
                             "gate_passed": verdict["gate_passed"],
                             "invariant_pvec_sorted_rhat": verdict["invariant_pvec_sorted_rhat"],
                             "invariant_Eu_rhat": verdict["invariant_Eu_rhat"], **alloc}]

    # ECR-relabeled component weights (slots ordered by descending weight).
    pvec_re = np.asarray(relabeled["pvec"]).reshape(-1, K)
    true_desc = np.sort(np.asarray(truth["TRUE_PVEC"]).ravel())[::-1]
    wrows = []
    for slot in range(K):
        d = pvec_re[:, slot]
        lo, hi = np.percentile(d, [2.5, 97.5])
        tw = float(true_desc[slot]) if slot < K_true else np.nan
        wrows.append({**cond, "slot": slot, "post_mean": float(d.mean()), "post_std": float(d.std()),
                      "ci_low": float(lo), "ci_high": float(hi), "true_weight": tw,
                      "in_ci": (bool(lo <= tw <= hi) if slot < K_true else np.nan)})
    tables["weights"] = wrows

    # convergence: label-invariant gate + per-component (raw) + per-component AFTER relabel
    # (ESS should recover for the live slots - the signal that justifies ECR).
    crows = []
    for q, r in gate.iterrows():
        crows.append({**cond, "scope": "invariant", "quantity": q, "live": np.nan,
                      "rhat": float(r["rhat"]), "ess": float(r["ess"])})
    for _, r in ls.component_convergence_table(post, K, K_true=K_true).iterrows():
        crows.append({**cond, "scope": "component", "quantity": f"slot{int(r['slot'])}:{r['quantity']}",
                      "live": bool(r["live"]), "rhat": float(r["rhat"]), "ess": float(r["ess"])})
    for _, r in ls.component_convergence_table(relabeled, K, K_true=K_true).iterrows():
        crows.append({**cond, "scope": "component_after", "quantity": f"slot{int(r['slot'])}:{r['quantity']}",
                      "live": bool(r["live"]), "rhat": float(r["rhat"]), "ess": float(r["ess"])})
    tables["convergence"] = crows

    # mixture moments (Rossi 5.5.2) vs true.
    model = build_model(post, cond["sampler"])
    mean, var = mc.mixture_moments(model)
    tmean, tvar = mc.mixture_moments(mc.true_dgp_model(truth))
    tables["moments"] = [{**cond, "param": param_names[j], "mix_mean": float(mean[j]),
                          "mix_var": float(var[j, j]), "true_mix_mean": float(tmean[j]),
                          "true_mix_var": float(tvar[j, j])} for j in range(P)]

    # parameter recovery vs ground truth (relabeled for mu/Sigma; identified for Delta/beta).
    mu_rows, mapping = mu_recovery_rows(relabeled, truth, cond, K, K_true, param_names)
    tables["mu_recovery"] = mu_rows
    tables["sigma_recovery"] = sigma_recovery_rows(relabeled, truth, cond, K, K_true, mapping, param_names)
    tables["delta_recovery"] = delta_recovery_rows(post, truth, cond, param_names)
    tables["beta_recovery"] = beta_recovery_rows(post, truth, cond, param_names)
    tables["beta_summary"] = analysis.beta_summary_rows(
        post.get("beta_i"), truth.get("TRUE_BETA"), param_names, cond=cond)

    # sampler diagnostics (per kernel/block); empty for samplers without transition infos.
    tables["diagnostics"] = diagnostics_rows(diag, cond)

    return tables, model
