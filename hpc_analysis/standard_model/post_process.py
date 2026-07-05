"""
Batch post-processing for the STANDARD-model experiments (jobs 200-202).

Reads every run's saved FULL posterior (<JOB_GLOB>/out[-test]/posterior_raw/*.pkl plus its
meta.json) and re-derives every per-run tidy table via src.summaries_standard.per_run_tables
(the SAME code each run writes on-node), writing concatenated CSVs to data/out/<out-name>/:

  runs, convergence, moments, mu_recovery, sigma_recovery, delta_recovery, beta_recovery,
  beta_summary, diagnostics, marginal_distances, marginal_diagnostics

ALL of these (including the marginal distances vs the true DGP on BOTH grid scenarios -
"full" and "chebyshev", see src/summaries_standard.py - and the marginal-density ESS/R-hat)
are produced per-run by src.summaries_standard, so this script only concatenates what each
run already computes on-node - the gathered CSVs are byte-identical to the per-run
out/<table>/*.csv files. No ECR/weights/pvec tables here: K = 1 has no label switching.

Defaults target the 2-chain standard jobs 200-202 (posterior format: PLAIN keys mu /
sigma_inv_chol_latent / beta_i / Delta, no component axis; datasets in
data/in/standard_model/).

Run with the project venv from the repo root:
    .venv/Scripts/python.exe hpc_analysis/standard_model/post_process.py             # real runs (out/)
    .venv/Scripts/python.exe hpc_analysis/standard_model/post_process.py --testing   # local (out-test/)
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

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src import summaries_standard as smry

# Datasets live here regardless of which job family is gathered.
DATA_IN = "standard_model"
# 200 = bayesm (R, rhierMnlRwMixture with ncomp=1, squeezed to plain keys);
# 201 = hmc; 202 = nuts. All share the plain-key posterior_raw.pkl format
# (sampler col distinguishes them).
JOB_GLOB = "jobs/20[0-2]-standard-*"
OUT_NAME = "standard_model"


def discover_runs(testing, job_glob):
    out = "out-test" if testing else "out"
    runs = []
    for job in sorted(REPO.glob(job_glob)):
        pdir, mdir = job / out / "posterior_raw", job / out / "meta"
        if not pdir.exists():
            continue
        for pkl in sorted(pdir.glob("*.pkl")):
            meta_f = mdir / (pkl.stem + ".json")
            if meta_f.exists():
                runs.append((pkl, meta_f))
    return runs


def load_truth(dataset_key):
    with open(REPO / "data" / "in" / DATA_IN / f"{dataset_key}.json") as f:
        return json.load(f)


def per_run(pkl, meta_f, acc):
    """Re-derive every per-run table for one run via src.summaries_standard (same code
    as on-node)."""
    meta = json.load(open(meta_f))
    post = pickle.load(open(pkl, "rb"))
    truth = load_truth(meta["dataset_key"])
    diag_pkl = pkl.parent.parent / "diagnostics" / (pkl.stem + ".pkl")
    diag = pickle.load(open(diag_pkl, "rb")) if diag_pkl.exists() else None

    tables, _ = smry.per_run_tables(post, meta, truth, diag)
    for name, rows in tables.items():
        acc[name].extend(rows)
    run = tables["runs"][0]
    return meta["run_key"], run["rhat_max"], run["ess_min"]


def main(testing, job_glob, out_name):
    runs = discover_runs(testing, job_glob)
    print(f"found {len(runs)} run(s) [{'out-test' if testing else 'out'}] for {job_glob}")
    acc = {k: [] for k in smry.TABLE_NAMES}

    for pkl, meta_f in runs:
        try:
            key, rhat_max, ess_min = per_run(pkl, meta_f, acc)
            print(f"  OK  {key:40s} rhat_max={rhat_max:.3f}  ess_min={ess_min:.0f}")
        except Exception:
            print(f"  FAIL {pkl.name}\n{traceback.format_exc()}")

    outdir = REPO / "data" / "out" / out_name
    outdir.mkdir(parents=True, exist_ok=True)
    print()
    for name, rows in acc.items():
        df = pd.DataFrame(rows)
        df.to_csv(outdir / f"{name}.csv", index=False)
        print(f"wrote {name}.csv  ({df.shape[0]} rows, {df.shape[1]} cols)")
    print(f"\n-> {outdir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Post-process standard-model runs.")
    ap.add_argument("--testing", action="store_true", help="use out-test/ instead of out/")
    ap.add_argument("--glob", default=JOB_GLOB, help=f"job dir glob (default: {JOB_GLOB})")
    ap.add_argument("--out-name", default=OUT_NAME,
                    help=f"subdir of data/out/ for the gathered CSVs (default: {OUT_NAME})")
    args = ap.parse_args()
    main(args.testing, args.glob, args.out_name)
