"""
Batch post-processing for the k5model_mixture experiment.

Reads every run's saved FULL posterior (jobs/00[4-7]*-k5-*/out[-test]/posterior_raw/*.pkl
plus its meta.json), reproduces the study analyses, and writes tidy CSVs to
data/out/k5model_mixture/:

  runs.csv               1 row/run : conditions, runtime_s, n_sampling_errors
  ecr_report.csv         1 row/run : ECR converged / switching_rate / honest verdict / gate
  weights.csv            K rows/run: ECR-relabeled component weights vs true (rank-matched)
  convergence.csv        rows/run  : label-invariant + per-component rhat/ess
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
JOB_GLOB = "jobs/00[4-7]*-k5-*"
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


def per_run(pkl, meta_f, acc):
    meta = json.load(open(meta_f))
    cond = {k: meta[k] for k in COND_KEYS}
    post = pickle.load(open(pkl, "rb"))
    truth = load_truth(cond["dataset_key"])
    K, K_true, P = int(cond["k_model"]), int(cond["k_true"]), int(meta["n_params"])
    param_names = list(truth["param_names"])
    Z = np.asarray(truth["Z"])

    acc["runs"].append({**cond, "runtime_s": meta.get("runtime_s"),
                        "n_sampling_errors": meta.get("n_sampling_errors")})

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

    # ---- mixture moments (Rossi 5.5.2) vs true ----
    model = build_model(post, cond["sampler"])
    mean, var = mc.mixture_moments(model)
    tmean, tvar = mc.mixture_moments(mc.true_dgp_model(truth))
    for j in range(P):
        acc["moments"].append({**cond, "param": param_names[j],
                               "mix_mean": float(mean[j]), "mix_var": float(var[j, j]),
                               "true_mix_mean": float(tmean[j]), "true_mix_var": float(tvar[j, j])})

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
    acc = {k: [] for k in ("runs", "ecr_report", "weights", "convergence", "moments",
                           "marginal_distances", "marginal_diagnostics")}
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
