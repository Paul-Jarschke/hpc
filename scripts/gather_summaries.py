"""
Gather the on-node per-run summary CSVs into aggregated tables - the Tier-1 local gather.

Each run writes tidy per-run tables to jobs/<job>/out/<table>/<run_key>.csv (see
src/summaries.py, called from every run.qmd). This concatenates them across all jobs into
data/out/k5model_mixture/<table>.csv, one file per table, ready for analysis/plotting.

It replaces the heavy analysis/post_process.py: every per-run table - including the
marginal distances (each fit vs the true DGP) and the marginal-density ESS/R-hat - is now
written on-node by src.summaries, so this concatenation produces the full set, no posteriors
needed. (post_process.py only re-derives the same tables from saved posteriors as a fallback.)

Defaults target the 2-chain jobs 100-103 -> data/out/mixture_c2/. The older k5 jobs:
    .venv/Scripts/python.exe scripts/gather_summaries.py --glob "jobs/00[4-9]*-k5-*" --out-name k5model_mixture

Run locally after downloading the summary CSVs (download_all.py):
    .venv/Scripts/python.exe scripts/gather_summaries.py            # real out/
    .venv/Scripts/python.exe scripts/gather_summaries.py --testing  # out-test/
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.summaries import TABLE_NAMES   # the per-run tables written on-node

JOB_GLOB = "jobs/10[0-3]_mixture_*"
OUT_NAME = "mixture_c2"


def main(testing, job_glob, out_name):
    out = "out-test" if testing else "out"
    outdir = REPO / "data" / "out" / out_name
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"gathering on-node summaries from {job_glob}/{out}/<table>/ -> {outdir}")
    total = 0
    for table in TABLE_NAMES:
        files = sorted(REPO.glob(f"{job_glob}/{out}/{table}/*.csv"))
        if not files:
            print(f"  {table:16s}: (no files)")
            continue
        df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
        df.to_csv(outdir / f"{table}.csv", index=False)
        total += len(files)
        print(f"  {table:16s}: {len(files):4d} run-files -> {df.shape[0]:7d} rows")
    print(f"\n-> {outdir}")
    if total == 0:
        print("NOTE: found no per-run summary CSVs - have the runs been rendered + downloaded?")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Gather on-node per-run summary CSVs.")
    ap.add_argument("--testing", action="store_true", help="use out-test/ instead of out/")
    ap.add_argument("--glob", default=JOB_GLOB, help=f"job dir glob (default: {JOB_GLOB})")
    ap.add_argument("--out-name", default=OUT_NAME,
                    help=f"subdir of data/out/ for the gathered CSVs (default: {OUT_NAME})")
    args = ap.parse_args()
    main(args.testing, args.glob, args.out_name)
