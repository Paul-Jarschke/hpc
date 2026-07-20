# Re-derive every per-run table from the saved posterior_raw pickles
# via src.summaries.per_run_tables (same code the runs use on-node),
# concat into data/out/<out-name>/*.csv. Defaults: jobs 100-103 ->
# mixture_c2; old 004-009 via --glob (their c1 rhats gather as NaN).
# run: .venv/Scripts/python.exe hpc_analysis/mixture_models/post_process.py

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

from src import summaries as smry

# Datasets live here regardless of which job family is gathered.
DATA_IN = "k5model_mixture"
# 100 = bayesm (R); 101 = bayesm_gibbs replication; 102 = hmc; 103 = nuts. All share the
# byte-compatible posterior_raw.pkl format (sampler col distinguishes them).
JOB_GLOB = "jobs/10[0-3]_mixture_*"
OUT_NAME = "mixture_c2"


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
    meta = json.load(open(meta_f))
    post = pickle.load(open(pkl, "rb"))
    truth = load_truth(meta["dataset_key"])
    diag_pkl = pkl.parent.parent / "diagnostics" / (pkl.stem + ".pkl")
    diag = pickle.load(open(diag_pkl, "rb")) if diag_pkl.exists() else None

    tables, _ = smry.per_run_tables(post, meta, truth, diag)
    for name, rows in tables.items():
        acc[name].extend(rows)
    rep = tables["ecr_report"][0]
    return meta["run_key"], rep["switching_rate"], rep["verdict"]


def main(testing, job_glob, out_name):
    runs = discover_runs(testing, job_glob)
    print(f"found {len(runs)} run(s) [{'out-test' if testing else 'out'}] for {job_glob}")
    acc = {k: [] for k in smry.TABLE_NAMES}

    for pkl, meta_f in runs:
        try:
            key, sw, verdict = per_run(pkl, meta_f, acc)
            print(f"  OK  {key:32s} switch={sw:.2f}  {verdict[:34]}")
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
    ap = argparse.ArgumentParser(description="Post-process mixture runs.")
    ap.add_argument("--testing", action="store_true", help="use out-test/ instead of out/")
    ap.add_argument("--glob", default=JOB_GLOB, help=f"job dir glob (default: {JOB_GLOB})")
    ap.add_argument("--out-name", default=OUT_NAME,
                    help=f"subdir of data/out/ for the gathered CSVs (default: {OUT_NAME})")
    args = ap.parse_args()
    main(args.testing, args.glob, args.out_name)
