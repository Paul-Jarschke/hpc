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
from the original post_process.py; it reuses analysis.* (label-invariant convergence),
label_switching.* (ECR relabeling of pvec only - mu_k/Sigma are not relabeled; see
label_switching.py's module docstring) and marginal_comparison.* (Rossi mixture moments).
per_run_tables() returns the per-run tables plus the in-memory mixture `model` dict so
the caller can run the cross-sampler step without rebuilding it.

No mu_recovery / sigma_recovery tables: upstream (@ 9cde043) stopped relabeling mu_k/
Sigma entirely - only pvec is ECR-relabeled now, since all other inference uses
label-invariant functionals (marginal density, mixture moments). Per-component mu/Sigma
recovery vs ground truth is therefore no longer computed here either, matching upstream.

The marginal-density tables are computed on the TWO grid scenarios of the study's
full_marginal_comparison notebook (upstream @ 893e63f): "full" (build_grids_full,
raw mu +/- 6 sigma envelope over every component incl. surplus ones) and "chebyshev"
(build_grids_chebyshev, aggregate mixture-moment mean +/- 5 sigma window, >= 96% of
each model's own marginal mass by Chebyshev's inequality). Every distance /
density-diagnostic row carries a `grid` column naming its scenario.

delta_summary_rows / beta_summary_rows / pvec_mean_table live HERE (harness glue):
they were part of the previously vendored analysis.py / label_switching.py but were
removed upstream; the vendored modules stay byte-identical to upstream, so the tidy
per-element tables they produced are kept as local functions instead.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from . import analysis
from . import label_switching as ls
from . import marginal_comparison as mc

# Identifying columns every row carries so the per-run CSVs concatenate cleanly. Must match
# the keys present in each run's meta.json (and analysis/post_process.COND_KEYS).
COND_KEYS = ("dataset_key", "scenario", "k_true", "data_seed", "k_model", "sampler", "n_chains")

# The full set of per-run table names per_run_tables() returns, in output order. (The
# cross-sampler marginal tables are NOT here - they need several runs together.)
TABLE_NAMES = ("runs", "ecr_report", "weights", "pvec_means", "convergence", "moments",
               "delta_recovery", "beta_recovery",
               "beta_summary", "diagnostics", "marginal_distances", "marginal_diagnostics")


# --------------------------------------------------------------------------- #
# In-memory model + component matching
# --------------------------------------------------------------------------- #
def build_model(post, name, duration_s=None):
    """Replicate marginal_comparison.load_sampler from an in-memory posterior dict.
    duration_s (the fit's total wall-clock, incl. warmup) rides along so
    marginal_comparison.functional_diagnostics can report ESS_bulk/s and ESS_tail/s."""
    mu = np.asarray(post["mu_k"])
    pvec = np.asarray(analysis._recover_pvec(post))
    Sigma = np.asarray(analysis._sigma_from_latent(np.asarray(post["sigma_inv_chol_k_latent"])))
    std = np.sqrt(np.clip(np.diagonal(Sigma, axis1=3, axis2=4), 0.0, None))
    return {"name": name, "mu": mu, "pvec": pvec, "Sigma": Sigma, "std": std,
            "is_mcmc": True, "duration_s": duration_s}


# --------------------------------------------------------------------------- #
# Glue kept from the previously vendored modules (removed upstream; see module
# docstring). Bodies are unchanged from the pre-893e63f vendored versions.
# --------------------------------------------------------------------------- #
def delta_summary_rows(delta_draws, true_delta, demo_names, param_names,
                       cond=None, ci=(2.5, 97.5)):
    """Tidy per-element posterior summary of the demographic shift matrix Delta.

    One row per Delta element (demo d, param p): posterior mean/std, credible-interval
    bounds (default 95%) and the ground-truth TRUE_DELTA value. Numeric, file-friendly
    form of the notebook's generate_delta_summaries / plot_delta_distributions.
    bias = post_mean - true_value is SIGNED (positive = overestimate)."""
    if delta_draws is None:
        return []
    Delta = np.asarray(delta_draws)
    if Delta.ndim < 2:
        return []
    D, P = Delta.shape[-2], Delta.shape[-1]
    if D == 0:
        return []
    flat = Delta.reshape(-1, D, P)
    lo_q, hi_q = ci
    cond = dict(cond or {})
    td = None if true_delta is None else np.asarray(true_delta, dtype=float)
    if demo_names is None:
        demo_names = [f"demo{d}" for d in range(D)]
    rows = []
    for dd in range(D):
        for p in range(P):
            draws = flat[:, dd, p]
            lo, hi = np.percentile(draws, [lo_q, hi_q])
            mean = float(draws.mean())
            tv = float(td[dd, p]) if td is not None else float("nan")
            rows.append({**cond, "demo": demo_names[dd], "param": param_names[p],
                         "post_mean": mean, "post_std": float(draws.std()),
                         "ci_low": float(lo), "ci_high": float(hi), "true_value": tv,
                         "bias": ((mean - tv) if td is not None else float("nan")),
                         "in_ci": (bool(lo <= tv <= hi) if td is not None else None)})
    return rows


def beta_summary_rows(beta_draws, true_beta, param_names, cond=None, ci=(2.5, 97.5)):
    """Tidy per-element posterior summary of the unit-level coefficients beta_i.

    One row per (unit i, param p): posterior mean/std, CI bounds, TRUE_BETA value and
    signed bias. beta_i is individually identified and unaffected by component label
    switching, so no ECR relabeling is needed. N * P rows per run (e.g. 300 * 4)."""
    if beta_draws is None:
        return []
    beta = np.asarray(beta_draws)
    if beta.ndim < 2:
        return []
    N, P = beta.shape[-2], beta.shape[-1]
    flat = beta.reshape(-1, N, P)
    mean = flat.mean(axis=0)                              # (N,P)
    std = flat.std(axis=0)                                # (N,P)
    lo, hi = np.percentile(flat, list(ci), axis=0)        # (N,P) each
    cond = dict(cond or {})
    tb = None if true_beta is None else np.asarray(true_beta, dtype=float)
    rows = []
    for i in range(N):
        for p in range(P):
            m = float(mean[i, p])
            tv = float(tb[i, p]) if tb is not None else float("nan")
            rows.append({**cond, "unit": i, "param": param_names[p],
                         "post_mean": m, "post_std": float(std[i, p]),
                         "ci_low": float(lo[i, p]), "ci_high": float(hi[i, p]),
                         "true_value": tv,
                         "bias": ((m - tv) if tb is not None else float("nan")),
                         "in_ci": (bool(lo[i, p] <= tv <= hi[i, p]) if tb is not None else None)})
    return rows


def pvec_mean_table(posterior_samples, pvec_after, K):
    """Mean component weight per slot, BEFORE and AFTER ECR relabeling.

    `pvec_after` is the relabeled pvec array (C,S,K) returned by
    label_switching.relabel_pvec - not a full posterior dict.

    Each stage is ranked INDEPENDENTLY by descending mean weight - read BY RANK within
    a stage, NOT row-to-row across stages (before relabeling the raw labels have no
    stable identity; that IS label switching)."""
    stages = [("before", np.asarray(analysis._recover_pvec(posterior_samples))),
              ("after",  np.asarray(pvec_after))]
    rows = []
    for stage, pvec in stages:
        C, S, _ = pvec.shape
        means = np.sort(pvec.reshape(C * S, K).mean(axis=0))[::-1]   # descending
        for rank, m in enumerate(means):
            rows.append({"stage": stage, "rank": rank, "pvec_mean": float(m)})
    return pd.DataFrame(rows)


def delta_recovery_rows(post, truth, cond, param_names):
    """Delta (demographic shift) recovery vs TRUE_DELTA. Delegates to delta_summary_rows."""
    if "Delta" not in post or truth.get("TRUE_DELTA") is None:
        return []
    D = np.asarray(post["Delta"]).shape[-2]                    # (C,S,D,P)
    demo_names = list(truth.get("demo_names", [f"demo{d}" for d in range(D)]))
    return delta_summary_rows(post["Delta"], truth["TRUE_DELTA"],
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
        z = ls.reconstruct_allocations(post, Z=Z)                # (C,S,N) raw labels
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

    # ECR relabeling of pvec (only pvec - mu_k/Sigma stay unrelabeled; all other
    # inference uses label-invariant functionals) + honest verdict against the gate.
    pvec_after, report = ls.relabel_pvec(post, K=K, Z=Z, K_true=K_true)
    gate = analysis.invariant_convergence_summary(post, include_cov=True)
    verdict = ls.classify_outcome(report, gate)
    alloc = allocation_accuracy(post, report, truth, K, Z)
    tables["ecr_report"] = [{**cond, "converged": report["converged"], "n_iter": report["n_iter"],
                             "switching_rate": report["switching_rate"], "verdict": verdict["verdict"],
                             "gate_passed": verdict["gate_passed"],
                             "invariant_pvec_sorted_rhat": verdict["invariant_pvec_sorted_rhat"],
                             **alloc}]

    # ECR-relabeled component weights (slots ordered by descending weight).
    pvec_re = np.asarray(pvec_after).reshape(-1, K)
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

    # pvec mean weights per component, BEFORE and AFTER relabeling (each ranked by descending
    # mean weight) - logged for every run so the weight distribution can be studied across seeds.
    pmrows = []
    for _, r in pvec_mean_table(post, pvec_after, K).iterrows():
        pmrows.append({**cond, "stage": r["stage"], "rank": int(r["rank"]),
                       "pvec_mean": float(r["pvec_mean"])})
    tables["pvec_means"] = pmrows

    # convergence: label-invariant gate + per-slot pvec (raw) + per-slot pvec AFTER relabel
    # (ESS should recover for the live slots - the signal that justifies ECR). Only pvec is
    # diagnosed here (mu_k/Sigma are not relabeled upstream anymore, so per-component mu/Sigma
    # R-hat/ESS would be label-switched noise, not a meaningful diagnostic).
    pvec_before = np.asarray(analysis._recover_pvec(post))
    crows = []
    for q, r in gate.iterrows():
        crows.append({**cond, "scope": "invariant", "quantity": q, "live": np.nan,
                      "rhat": float(r["rhat"]), "ess": float(r["ess"])})
    for _, r in ls.pvec_convergence_table(pvec_before, K, K_true=K_true).iterrows():
        crows.append({**cond, "scope": "component", "quantity": f"slot{int(r['slot'])}:pvec",
                      "live": bool(r["live"]), "rhat": float(r["rhat"]), "ess": float(r["ess"])})
    for _, r in ls.pvec_convergence_table(pvec_after, K, K_true=K_true).iterrows():
        crows.append({**cond, "scope": "component_after", "quantity": f"slot{int(r['slot'])}:pvec",
                      "live": bool(r["live"]), "rhat": float(r["rhat"]), "ess": float(r["ess"])})
    tables["convergence"] = crows

    # mixture moments (Rossi 5.5.2) vs true.
    model = build_model(post, cond["sampler"], duration_s=meta.get("runtime_s"))
    mean, var = mc.mixture_moments(model)
    tmean, tvar = mc.mixture_moments(mc.true_dgp_model(truth))
    tables["moments"] = [{**cond, "param": param_names[j], "mix_mean": float(mean[j]),
                          "mix_var": float(var[j, j]), "true_mix_mean": float(tmean[j]),
                          "true_mix_var": float(tvar[j, j])} for j in range(P)]

    # parameter recovery vs ground truth (Delta/beta are individually identified, no ECR needed).
    tables["delta_recovery"] = delta_recovery_rows(post, truth, cond, param_names)
    tables["beta_recovery"] = beta_recovery_rows(post, truth, cond, param_names)
    tables["beta_summary"] = beta_summary_rows(
        post.get("beta_i"), truth.get("TRUE_BETA"), param_names, cond=cond)

    # sampler diagnostics (per kernel/block); empty for samplers without transition infos.
    tables["diagnostics"] = diagnostics_rows(diag, cond)

    # marginal-density comparison vs the TRUE DGP marginal (Rossi Eq. 5.5.19), on the SAME
    # two grid scenarios as the study's full_marginal_comparison notebook: "full" (raw
    # mu +/- 6 sigma envelope over every component incl. surplus ones) and "chebyshev"
    # (exact pooled-marginal mean/std via mc._marginal_moments, +/- 5 sigma window;
    # Chebyshev's inequality guarantees >= 96% of each model's own mass before the union).
    # Single-model grids -> fully run-independent (needs no sibling samplers), so every run
    # logs both on-node. Distances per parameter: Hellinger, KL(model||true), JSD, TVD,
    # Wasserstein-1. retained_mass_model/retained_mass_true (mc.retained_mass) report the
    # REALISED mass of each side's own marginal inside the (possibly union-widened) window -
    # the exact counterpart to the theoretical Chebyshev guarantee. The `grid` column names
    # the scenario.
    true_model = mc.true_dgp_model(truth)
    grid_scenarios = {
        "full":      mc.build_grids_full([model], true_model, n_grid=1000, n_sigma=6),
        "chebyshev": mc.build_grids_chebyshev([model], true_model, n_grid=1000, k=5.0),
    }
    mdist = []
    for grid_name, grids in grid_scenarios.items():
        rm_model = mc.retained_mass(model, grids)
        rm_true = mc.retained_mass(true_model, grids)
        for (_, param), r in mc.distance_table([model], true_model, grids, param_names).iterrows():
            j = param_names.index(param)
            mdist.append({**cond, "grid": grid_name, "param": param,
                          "retained_mass_model": float(rm_model[j]),
                          "retained_mass_true": float(rm_true[j]),
                          **r.to_dict()})
    tables["marginal_distances"] = mdist

    # marginal convergence: Goose-identical arviz diagnostics on grid-free functionals
    # of each per-draw label-invariant marginal (Rossi Eq. 5.5.19). functional_diagnostics
    # gives rank split-R-hat (Rhat) plus bulk/tail ESS (ESS_bulk, ESS_tail) - the exact
    # calls in liesel.goose.summary_m - for the mean, sd and q05/q50/q95 of every marginal;
    # ESS_bulk/s and ESS_tail/s (effective draws per fit-second) come from the run's
    # wall-clock, carried on `model` via build_model(duration_s=runtime_s). One grid-free
    # pass, so no `grid` column here (contrast marginal_distances). Replaces the former
    # density_series_diagnostics / moment_series_diagnostics rhat/ess tables.
    mdiag = []
    for (param, functional), r in mc.functional_diagnostics(model, param_names).iterrows():
        mdiag.append({**cond, "param": param, "functional": functional, **r.to_dict()})
    tables["marginal_diagnostics"] = mdiag

    return tables, model
