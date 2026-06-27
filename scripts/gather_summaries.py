"""
Gather the on-node per-run summary CSVs into aggregated tables - the Tier-1 local gather.

Each run writes tidy per-run tables to jobs/<job>/out/<table>/<run_key>.csv (see
src/summaries.py, called from every run.qmd). This concatenates them across all jobs into
data/out/k5model_mixture/<table>.csv, one file per table, ready for analysis/plotting.

It replaces the heavy analysis/post_process.py for everything EXCEPT the cross-sampler
marginal tables (marginal_distances / marginal_diagnostics), which need raw chains from
several samplers together and so are produced by post_process.py on the Tier-2 subset.

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

EXPERIMENT = "k5model_mixture"
JOB_GLOB = "jobs/00[4-9]*-k5-*"


def main(testing):
    out = "out-test" if testing else "out"
    outdir = REPO / "data" / "out" / EXPERIMENT
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"gathering on-node summaries from {JOB_GLOB}/{out}/<table>/ -> {outdir}")
    total = 0
    for table in TABLE_NAMES:
        files = sorted(REPO.glob(f"{JOB_GLOB}/{out}/{table}/*.csv"))
        if not files:
            print(f"  {table:16s}: (no files)")
            continue
        df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
        df.to_csv(outdir / f"{table}.csv", index=False)
        total += len(files)
        print(f"  {table:16s}: {len(files):4d} run-files -> {df.shape[0]:7d} rows")
    print(f"\n-> {outdir}   (marginal_distances / marginal_diagnostics come from "
          f"post_process.py on the Tier-2 posterior subset)")
    if total == 0:
        print("NOTE: found no per-run summary CSVs - have the runs been rendered + downloaded?")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Gather on-node per-run summary CSVs.")
    ap.add_argument("--testing", action="store_true", help="use out-test/ instead of out/")
    main(ap.parse_args().testing)
