"""
Batch post-processing for the k5model_mixture experiment.

Reads every run's saved FULL posterior (jobs/00[4-7]*-k5-*/out[-test]/posterior_raw/*.pkl
plus its meta.json), reproduces the study analyses, and writes tidy CSVs to
data/out/k5model_mixture/:

  runs.csv               1 row/run : conditions, runtime_s, errors, invariant rhat/ess + diag rollup
  ecr_report.csv         1 row/run : ECR converged / switching_rate / honest verdict / gate
  weights.csv            K rows/run: ECR-relabeled component weights vs true (rank-matched)
  convergence.csv        rows/run  : label-invariant + per-component rhat/ess (before AND after relabel)
  mu_recovery.csv        K*P rows  : ECR-relabeled mu_k vs TRUE_MU_K (Hungarian-matched)
  sigma_recovery.csv     lower-tri : ECR-relabeled Sigma_k vs TRUE_SIGMA_K (+ empirical cov)
  delta_recovery.csv     D*P rows  : Delta (demographic shift) vs TRUE_DELTA
  beta_recovery.csv      P rows/run: per-unit beta_i bias / RMSE / 95% coverage vs TRUE_BETA
  diagnostics.csv        rows/kernel: sampler diagnostics (divergences, treedepth, leapfrog, accept)
  moments.csv            P rows/run: mixture mean/var (Rossi 5.5.2) vs true
  marginal_distances.csv per (dataset,chains,sampler,param): Hellinger/KL/JSD/TVD/W1 vs true
  marginal_diagnostics.csv per (dataset,chains,sampler,param): density+moment series ess/rhat

All heavy lifting reuses the vendored study code in src/ (label_switching = ECR.iterative.1,
marginal_comparison = Rossi 5.5.19/5.5.2, analysis = label-invariant convergence).

Run with the project venv from the repo root:
    .venv/Scripts/python.exe analysis/post_process.py             # real runs (out/)
    .venv/Scripts/python.exe analysis/post_process.py --testing   # local (out-test/)
"""

import argparse
import json
import pickle
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

# numpy>=2.0 renamed trapz -> trapezoid; the vendored marginal_comparison.py still calls
# np.trapz. Restore it here (harness pins numpy 2.4.1) without editing the vendored file.
if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src import analysis
from src import label_switching as ls
from src import marginal_comparison as mc

EXPERIMENT = "k5model_mixture"
# 004-007 = liesel (nuts/hmc); 008-009 = bayesm. Widened so the byte-compatible
# bayesm posterior_raw.pkl is swept by the same pipeline (sampler col distinguishes them).
JOB_GLOB = "jobs/00[4-9]*-k5-*"
COND_KEYS = ("dataset_key", "scenario", "k_true", "data_seed", "k_model", "sampler", "n_chains")


def discover_runs(testing):
    out = "out-test" if testing else "out"
    runs = []
    for job in sorted(REPO.glob(JOB_GLOB)):
        pdir, mdir = job / out / "posterior_raw", job / out / "meta"
        if not pdir.exists():
            continue
        for pkl in sorted(pdir.glob("*.pkl")):
            meta_f = mdir / (pkl.stem + ".json")
            if meta_f.exists():
                runs.append((pkl, meta_f))
    return runs


def load_truth(dataset_key):
    with open(REPO / "data" / "in" / EXPERIMENT / f"{dataset_key}.json") as f:
        return json.load(f)


def build_model(post, name):
    """Replicate marginal_comparison.load_sampler from an in-memory posterior dict."""
    mu = np.asarray(post["mu_k"])
    pvec = np.asarray(analysis._recover_pvec(post))
    Sigma = np.asarray(analysis._sigma_from_latent(np.asarray(post["sigma_inv_chol_k_latent"])))
    std = np.sqrt(np.clip(np.diagonal(Sigma, axis1=3, axis2=4), 0.0, None))
    return {"name": name, "mu": mu, "pvec": pvec, "Sigma": Sigma, "std": std, "is_mcmc": True}


# --------------------------------------------------------------------------- #
# Parameter-recovery tables (analysis.ipynb), computed on the ECR-RELABELED
# posterior so per-component means are not corrupted by label switching.
# --------------------------------------------------------------------------- #
def _match_components(post_mu_mean, true_mu, K_true):
    """Hungarian-match model components -> true components on squared mu distance.
    Returns {model_slot: true_idx} for the K_true matched slots (analysis.summarize_mu_k)."""
    cost = np.sum((post_mu_mean[:, None, :] - true_mu[None, :K_true, :]) ** 2, axis=-1)  # (K,K_true)
    row, col = linear_sum_assignment(cost)
    return {int(r): int(c) for r, c in zip(row, col)}


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
    """Delta (demographic shift) recovery vs TRUE_DELTA (analysis.generate_delta_summaries)."""
    if "Delta" not in post or truth.get("TRUE_DELTA") is None:
        return []
    Delta = np.asarray(post["Delta"])                          # (C,S,D,P)
    D, P = Delta.shape[-2], Delta.shape[-1]
    flat = Delta.reshape(-1, D, P)
    tdelta = np.asarray(truth["TRUE_DELTA"], dtype=float)      # (D,P)
    demo_names = list(truth.get("demo_names", [f"demo{d}" for d in range(D)]))
    rows = []
    for dd in range(D):
        for p in range(P):
            d = flat[:, dd, p]
            lo, hi = np.percentile(d, [2.5, 97.5])
            tv = float(tdelta[dd, p])
            rows.append({**cond, "demo": demo_names[dd], "param": param_names[p],
                         "post_mean": float(d.mean()), "post_std": float(d.std()),
                         "ci_low": float(lo), "ci_high": float(hi), "true_value": tv,
                         "abs_diff": abs(tv - float(d.mean())), "in_ci": bool(lo <= tv <= hi)})
    return rows


def beta_recovery_rows(post, truth, cond, param_names):
    """Per-unit beta_i recovery vs TRUE_BETA: per-parameter bias / RMSE / 95% coverage
    of the unit-level posterior-mean betas (analysis.plot_beta_scatter, made numeric)."""
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


def diagnostics_rows(diag_pkl, cond, max_treedepth=10):
    """Per-kernel sampler-diagnostics summary from out/diagnostics/<key>.pkl.
    NUTS: divergences, treedepth (saturation at max_treedepth), leapfrog steps; all: acceptance."""
    if not diag_pkl.exists():
        return []
    d = pickle.load(open(diag_pkl, "rb"))
    inv = {v: k for k, v in d.get("kernels_by_pos_key", {}).items()}   # kernel_id -> block name
    rows = []
    for kid, fl in d.get("transition_infos", {}).items():
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


def per_run(pkl, meta_f, acc):
    meta = json.load(open(meta_f))
    cond = {k: meta[k] for k in COND_KEYS}
    post = pickle.load(open(pkl, "rb"))
    truth = load_truth(cond["dataset_key"])
    K, K_true, P = int(cond["k_model"]), int(cond["k_true"]), int(meta["n_params"])
    param_names = list(truth["param_names"])
    Z = np.asarray(truth["Z"])

    acc["runs"].append({**cond, "runtime_s": meta.get("runtime_s"),
                        "n_sampling_errors": meta.get("n_sampling_errors"),
                        "invariant_rhat_max": meta.get("invariant_rhat_max"),
                        "invariant_ess_min": meta.get("invariant_ess_min"),
                        "n_divergent": meta.get("n_divergent"),
                        "max_treedepth": meta.get("max_treedepth"),
                        "max_leapfrog": meta.get("max_leapfrog"),
                        "mean_acceptance": meta.get("mean_acceptance")})

    # ---- ECR.iterative.1 relabeling + honest verdict against the invariant gate ----
    relabeled, report = ls.relabel_run(post, K=K, Z=Z, K_true=K_true)
    gate = analysis.invariant_convergence_summary(post, include_cov=True)
    verdict = ls.classify_outcome(report, gate)
    acc["ecr_report"].append({**cond, "converged": report["converged"], "n_iter": report["n_iter"],
                              "switching_rate": report["switching_rate"], "verdict": verdict["verdict"],
                              "gate_passed": verdict["gate_passed"],
                              "invariant_pvec_sorted_rhat": verdict["invariant_pvec_sorted_rhat"],
                              "invariant_Eu_rhat": verdict["invariant_Eu_rhat"]})

    # ---- ECR-relabeled component weights (slots ordered by descending weight) ----
    pvec_re = np.asarray(relabeled["pvec"]).reshape(-1, K)
    true_desc = np.sort(np.asarray(truth["TRUE_PVEC"]).ravel())[::-1]
    for slot in range(K):
        d = pvec_re[:, slot]
        lo, hi = np.percentile(d, [2.5, 97.5])
        tw = float(true_desc[slot]) if slot < K_true else np.nan
        acc["weights"].append({**cond, "slot": slot, "post_mean": float(d.mean()),
                               "post_std": float(d.std()), "ci_low": float(lo), "ci_high": float(hi),
                               "true_weight": tw,
                               "in_ci": (bool(lo <= tw <= hi) if slot < K_true else np.nan)})

    # ---- convergence: label-invariant gate + per-component (raw) ----
    for q, r in gate.iterrows():
        acc["convergence"].append({**cond, "scope": "invariant", "quantity": q,
                                   "live": np.nan, "rhat": float(r["rhat"]), "ess": float(r["ess"])})
    for _, r in ls.component_convergence_table(post, K, K_true=K_true).iterrows():
        acc["convergence"].append({**cond, "scope": "component",
                                   "quantity": f"slot{int(r['slot'])}:{r['quantity']}",
                                   "live": bool(r["live"]), "rhat": float(r["rhat"]), "ess": float(r["ess"])})
    # AFTER relabeling: ESS should recover for the live slots — the signal that justifies ECR.
    for _, r in ls.component_convergence_table(relabeled, K, K_true=K_true).iterrows():
        acc["convergence"].append({**cond, "scope": "component_after",
                                   "quantity": f"slot{int(r['slot'])}:{r['quantity']}",
                                   "live": bool(r["live"]), "rhat": float(r["rhat"]), "ess": float(r["ess"])})

    # ---- mixture moments (Rossi 5.5.2) vs true ----
    model = build_model(post, cond["sampler"])
    mean, var = mc.mixture_moments(model)
    tmean, tvar = mc.mixture_moments(mc.true_dgp_model(truth))
    for j in range(P):
        acc["moments"].append({**cond, "param": param_names[j],
                               "mix_mean": float(mean[j]), "mix_var": float(var[j, j]),
                               "true_mix_mean": float(tmean[j]), "true_mix_var": float(tvar[j, j])})

    # ---- parameter recovery vs ground truth (relabeled; the analysis.ipynb tables) ----
    mu_rows, mapping = mu_recovery_rows(relabeled, truth, cond, K, K_true, param_names)
    acc["mu_recovery"].extend(mu_rows)
    acc["sigma_recovery"].extend(sigma_recovery_rows(relabeled, truth, cond, K, K_true, mapping, param_names))
    acc["delta_recovery"].extend(delta_recovery_rows(post, truth, cond, param_names))
    acc["beta_recovery"].extend(beta_recovery_rows(post, truth, cond, param_names))

    # ---- sampler diagnostics summary (per kernel/block; from out/diagnostics/<key>.pkl) ----
    diag_pkl = pkl.parent.parent / "diagnostics" / (pkl.stem + ".pkl")
    acc["diagnostics"].extend(diagnostics_rows(diag_pkl, cond))

    # stash for the cross-sampler marginal comparison
    acc["_groups"].setdefault((cond["dataset_key"], cond["n_chains"]), []).append(
        {"cond": cond, "model": model, "truth": truth, "param_names": param_names, "K_true": K_true})
    return meta["run_key"], report["switching_rate"], verdict["verdict"]


def cross_sampler(acc):
    """Per (dataset, n_chains): every sampler's marginal vs the TRUE DGP marginal."""
    for (dkey, nch), entries in acc["_groups"].items():
        models = [e["model"] for e in entries]
        truth, param_names, K_true = entries[0]["truth"], entries[0]["param_names"], entries[0]["K_true"]
        grids = mc.build_grids(models, K_true)            # fitted-model support only (per the module)
        true_model = mc.true_dgp_model(truth)
        for (sampler, param), r in mc.distance_table(models, true_model, grids, param_names).iterrows():
            acc["marginal_distances"].append({"dataset_key": dkey, "n_chains": nch,
                                               "sampler": sampler, "param": param, **r.to_dict()})
        for e in entries:
            s = e["cond"]["sampler"]
            for param, r in mc.density_series_diagnostics(e["model"], grids, param_names).iterrows():
                acc["marginal_diagnostics"].append({"dataset_key": dkey, "n_chains": nch, "sampler": s,
                                                    "param": param, "kind": "density", **r.to_dict()})
            for (param, moment), r in mc.moment_series_diagnostics(e["model"], param_names).iterrows():
                acc["marginal_diagnostics"].append({"dataset_key": dkey, "n_chains": nch, "sampler": s,
                                                    "param": param, "kind": f"moment_{moment}", **r.to_dict()})


def main(testing):
    runs = discover_runs(testing)
    print(f"found {len(runs)} run(s) [{'out-test' if testing else 'out'}]")
    acc = {k: [] for k in ("runs", "ecr_report", "weights", "convergence",
                           "mu_recovery", "sigma_recovery", "delta_recovery", "beta_recovery",
                           "diagnostics", "moments", "marginal_distances", "marginal_diagnostics")}
    acc["_groups"] = {}

    for pkl, meta_f in runs:
        try:
            key, sw, verdict = per_run(pkl, meta_f, acc)
            print(f"  OK  {key:32s} switch={sw:.2f}  {verdict[:34]}")
        except Exception:
            print(f"  FAIL {pkl.name}\n{traceback.format_exc()}")

    try:
        cross_sampler(acc)
    except Exception:
        print(f"  cross-sampler FAILED\n{traceback.format_exc()}")

    outdir = REPO / "data" / "out" / EXPERIMENT
    outdir.mkdir(parents=True, exist_ok=True)
    acc.pop("_groups")
    print()
    for name, rows in acc.items():
        df = pd.DataFrame(rows)
        df.to_csv(outdir / f"{name}.csv", index=False)
        print(f"wrote {name}.csv  ({df.shape[0]} rows, {df.shape[1]} cols)")
    print(f"\n-> {outdir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Post-process k5model_mixture runs.")
    ap.add_argument("--testing", action="store_true", help="use out-test/ instead of out/")
    main(ap.parse_args().testing)
