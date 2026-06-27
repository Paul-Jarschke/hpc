"""
Batch post-processing for the k5model_mixture experiment.

Reads every run's saved FULL posterior (jobs/00[4-9]*-k5-*/out[-test]/posterior_raw/*.pkl
plus its meta.json), builds the per-run tidy tables via src.summaries.per_run_tables (the
SAME code each run can write on-node), then adds the cross-sampler marginal comparison
(the one analysis that needs several samplers' chains together), and writes tidy CSVs to
data/out/k5model_mixture/:

  runs, ecr_report, weights, convergence, mu_recovery, sigma_recovery, delta_recovery,
  beta_recovery, beta_summary, diagnostics, moments  (per-run, from src.summaries)
  marginal_distances, marginal_diagnostics            (cross-sampler, Rossi 5.5.19/5.5.2)

In the Tier-1/Tier-2 design the per-run tables are written on-node for every run, so this
script is only needed for the Tier-2 subset that keeps full posteriors (for the marginal
comparison). The per-run tables it produces are byte-identical to the on-node CSVs because
both call src.summaries.

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

from src import marginal_comparison as mc
from src import summaries as smry

EXPERIMENT = "k5model_mixture"
# 004-007 = liesel (nuts/hmc); 008-009 = bayesm. Widened so the byte-compatible
# bayesm posterior_raw.pkl is swept by the same pipeline (sampler col distinguishes them).
JOB_GLOB = "jobs/00[4-9]*-k5-*"


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


def per_run(pkl, meta_f, acc):
    """Build every per-run table for one run (via src.summaries) and stash its in-memory
    model for the cross-sampler step."""
    meta = json.load(open(meta_f))
    post = pickle.load(open(pkl, "rb"))
    truth = load_truth(meta["dataset_key"])
    diag_pkl = pkl.parent.parent / "diagnostics" / (pkl.stem + ".pkl")
    diag = pickle.load(open(diag_pkl, "rb")) if diag_pkl.exists() else None

    tables, model = smry.per_run_tables(post, meta, truth, diag)
    for name, rows in tables.items():
        acc[name].extend(rows)

    cond = {k: meta[k] for k in smry.COND_KEYS}
    acc["_groups"].setdefault((cond["dataset_key"], cond["n_chains"]), []).append(
        {"cond": cond, "model": model, "truth": truth,
         "param_names": list(truth["param_names"]), "K_true": int(meta["k_true"])})
    rep = tables["ecr_report"][0]
    return meta["run_key"], rep["switching_rate"], rep["verdict"]


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
    acc = {k: [] for k in (*smry.TABLE_NAMES, "marginal_distances", "marginal_diagnostics")}
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
