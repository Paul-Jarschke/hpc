"""
Batch data generator for the `standard_model` experiment.

Experiment design
-----------------
The STANDARD (single-normal-component, no mixture) HBMNL, Rossi (2006) section 5.4:

    beta_i = mu + Z[i] @ Delta + u_i,   u_i ~ N(0, Sigma)

One scenario ("standard", the DGP defaults: 300 units x 50 obs, 4 alternatives,
2 demographics), replicated over 100 seeds - the same replicate-seed design as the
mixture experiment, so downstream analysis can report distributions over datasets
instead of a single seed-42 anecdote (which is all the study repo generated).

The DGP is deterministic in `seed`, so re-running is idempotent: datasets that already
exist on disk are kept byte-identical and skipped (only genuinely new seeds are written),
while the manifest is always rewritten to cover every seed in SEEDS.

Output (one self-describing JSON per dataset, ground truth included):
    data/in/standard_model/std_s{SEED:02d}.json
    data/in/standard_model/manifest.csv      # index joined later by the fit params.csv

The DGP is vendored verbatim from HierarchicalBayesianMultinomialLogit @ 893e63f (see
data/dgp/dgp.py: generate_standard_simulated_data). `data/in/` is gitignored; this
script + data/dgp/ are the tracked, regenerable source.

Run with the project's Python environment (which has numpy/jax), from anywhere:
    .venv/Scripts/python.exe data/generate_standard_data.py
    .venv/Scripts/python.exe data/generate_standard_data.py --limit 2   # smoke test
"""

import argparse
import contextlib
import csv
import io
import json
import os
import sys

import numpy as np

# Make the vendored DGP package importable regardless of the current working directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from dgp.dgp import generate_standard_simulated_data, save_to_json  # noqa: E402

# --------------------------------------------------------------------------------------
# Experiment configuration
# --------------------------------------------------------------------------------------
EXPERIMENT = "standard_model"
# 1..105: 100 target seeds + a small buffer. bayesm's rhierMnlRwMixture (ncomp=1) refuses
# a dataset unless EVERY alternative is chosen at least once; with a single homogeneous
# component a dominated alternative can occasionally get zero choices (the mixture grid's
# kt1_s70 case). Screened below (all_alts_chosen); params.R keeps the first 100 fittable.
SEEDS = range(1, 106)  # 1..105 inclusive (100 target + 5 backfill buffer)

# The scenario config = the vendored DGP's own defaults (single source of truth).
SCENARIO = {
    "n_units": 300,
    "n_obs":   50,
    "n_alts":  4,
    "n_demos": 2,
}

OUT_DIR = os.path.join(SCRIPT_DIR, "in", EXPERIMENT)

MANIFEST_FIELDS = [
    "dataset_key", "scenario", "k_true", "data_seed",
    "n_units", "n_obs", "n_alts", "n_demos",
    "n_alts_chosen", "all_alts_chosen", "file",
]


def count_alts_chosen(y) -> int:
    """How many distinct alternatives appear in the pooled choice vector y.

    bayesm::rhierMnlRwMixture requires this to equal n_alts (every alternative chosen at
    least once across all units/observations); otherwise it aborts. The gradient samplers
    (NUTS/HMC) have no such constraint. y is 0-indexed in the dataset JSON; only the count
    of distinct values matters here, so the offset is irrelevant.
    """
    return int(np.unique(np.asarray(y).ravel()).size)


def main(limit: int | None = None) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    seeds = list(SEEDS)
    if limit is not None:
        seeds = seeds[:limit]

    total = len(seeds)
    manifest_rows = []

    for i, seed in enumerate(seeds, start=1):
        dataset_key = f"std_s{seed:02d}"
        fname = f"{dataset_key}.json"
        fpath = os.path.join(OUT_DIR, fname)

        print(f"[{i:3d}/{total}] {dataset_key}  (scenario=standard, seed={seed})")

        if os.path.exists(fpath):
            print("        exists -> skipping generation (kept byte-identical)")
            with open(fpath) as fh:           # reload only to screen fittability
                y = json.load(fh)["y"]
        else:
            data = generate_standard_simulated_data(**SCENARIO, seed=seed)
            # save_to_json prints a line per file; keep batch output quiet without editing
            # the vendored code.
            with contextlib.redirect_stdout(io.StringIO()):
                save_to_json(data, fpath)
            y = data["y"]

        n_chosen = count_alts_chosen(y)
        fittable = int(n_chosen == SCENARIO["n_alts"])
        if not fittable:
            print(f"        NOT bayesm-fittable: only {n_chosen}/{SCENARIO['n_alts']} "
                  f"alternatives chosen (will be screened out of params.csv)")

        # Always record in the manifest so it spans every seed in SEEDS, even skipped ones.
        manifest_rows.append({
            "dataset_key":     dataset_key,
            "scenario":        "standard",
            "k_true":          1,
            "data_seed":       seed,
            "n_units":         SCENARIO["n_units"],
            "n_obs":           SCENARIO["n_obs"],
            "n_alts":          SCENARIO["n_alts"],
            "n_demos":         SCENARIO["n_demos"],
            "n_alts_chosen":   n_chosen,
            "all_alts_chosen": fittable,
            "file":            fname,
        })

    manifest_path = os.path.join(OUT_DIR, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nDone: {len(manifest_rows)} dataset(s) + manifest.csv -> {OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate standard_model datasets.")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only generate the first N datasets (for smoke testing).",
    )
    args = parser.parse_args()
    main(limit=args.limit)
