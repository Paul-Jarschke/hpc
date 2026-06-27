"""
Batch data generator for the `k5model_mixture` experiment.

Experiment design
-----------------
For each true number of mixture components K_TRUE in {1, 2, 3, 5}, draw 100 replicate
datasets (seeds 1..100). Every dataset is later fit with an *overspecified* K_MODEL = 5
model (the fitting strategy lives in the job's params.csv, not here).

Because `Z` and `Delta_true` are drawn before any K-dependent branching in the DGP and
have identical shapes across scenarios, reusing seeds 1..100 across K_TRUE gives a *paired*
design: the demographic structure is held fixed across K_TRUE for a matched seed.

The DGP is deterministic in `seed`, so re-running is idempotent: datasets that already
exist on disk are kept byte-identical and skipped (only genuinely new seeds are written),
while the manifest is always rewritten to cover every seed in SEEDS.

Output (one self-describing JSON per dataset, ground truth included):
    data/in/k5model_mixture/kt{K_TRUE}_s{SEED:02d}.json
    data/in/k5model_mixture/manifest.csv      # index joined later by the fit params.csv

The DGP is vendored verbatim from HierarchicalBayesianMNL @ 12ca13b (see data/dgp/).
`data/in/` is gitignored; this script + data/dgp/ are the tracked, regenerable source.

Run with the project's Python environment (which has numpy/jax), from anywhere:
    .venv/Scripts/python.exe data/generate_mixture_data.py
    .venv/Scripts/python.exe data/generate_mixture_data.py --limit 2   # smoke test
"""

import argparse
import contextlib
import csv
import io
import os
import sys

# Make the vendored DGP package importable regardless of the current working directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from dgp.dgp import generate_mixture_simulated_data, save_to_json  # noqa: E402
from dgp.experiment_configs import SCENARIOS  # noqa: E402

# --------------------------------------------------------------------------------------
# Experiment configuration
# --------------------------------------------------------------------------------------
EXPERIMENT = "k5model_mixture"
SEEDS = range(1, 101)  # 1..100 inclusive

# K_TRUE -> study scenario name (the vendored SCENARIOS dict is the single source of
# truth for n_units/n_obs/n_alts/n_demos/custom_pvec).
KTRUE_TO_SCENARIO = {
    1: "1comp",
    2: "2comp_equal",
    3: "3comp_equal",
    5: "5comp_equal",
}

OUT_DIR = os.path.join(SCRIPT_DIR, "in", EXPERIMENT)

MANIFEST_FIELDS = [
    "dataset_key", "scenario", "k_true", "data_seed",
    "n_units", "n_obs", "n_alts", "n_demos", "custom_pvec", "file",
]


def main(limit: int | None = None) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    jobs = [
        (k_true, scenario, seed)
        for k_true, scenario in KTRUE_TO_SCENARIO.items()
        for seed in SEEDS
    ]
    if limit is not None:
        jobs = jobs[:limit]

    total = len(jobs)
    manifest_rows = []

    for i, (k_true, scenario, seed) in enumerate(jobs, start=1):
        base_cfg = SCENARIOS[scenario]
        cfg = {**base_cfg, "seed": seed}  # override the scenario's default seed

        dataset_key = f"kt{k_true}_s{seed:02d}"
        fname = f"{dataset_key}.json"
        fpath = os.path.join(OUT_DIR, fname)

        print(f"[{i:3d}/{total}] {dataset_key}  (scenario={scenario}, K_true={k_true}, seed={seed})")

        if os.path.exists(fpath):
            print("        exists -> skipping generation (kept byte-identical)")
        else:
            data = generate_mixture_simulated_data(**cfg)
            # save_to_json prints a line per file; keep batch output quiet without editing
            # the vendored code.
            with contextlib.redirect_stdout(io.StringIO()):
                save_to_json(data, fpath)

        # Always record in the manifest so it spans every seed in SEEDS, even skipped ones.
        manifest_rows.append({
            "dataset_key": dataset_key,
            "scenario":    scenario,
            "k_true":      k_true,
            "data_seed":   seed,
            "n_units":     base_cfg["n_units"],
            "n_obs":       base_cfg["n_obs"],
            "n_alts":      base_cfg["n_alts"],
            "n_demos":     base_cfg["n_demos"],
            "custom_pvec": base_cfg["custom_pvec"],
            "file":        fname,
        })

    manifest_path = os.path.join(OUT_DIR, "manifest.csv")
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\nDone: {len(manifest_rows)} dataset(s) + manifest.csv -> {OUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate k5model_mixture datasets.")
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only generate the first N datasets (for smoke testing).",
    )
    args = parser.parse_args()
    main(limit=args.limit)
