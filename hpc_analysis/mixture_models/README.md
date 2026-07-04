# hpc_analysis

Analysis pipeline for the NEW 2-chain mixture jobs **100-103** (`jobs/10[0-3]_mixture_*`),
comparing FOUR samplers: `bayesm` (R), `bayesm_gibbs` (the Python Gibbs replication,
labeled "Replication" in figures), `hmc`, and `nuts`. All runs use n_chains = 2 and
K_MODEL = 5 over the same four scenarios (k_true in {1, 2, 3, 5}) and dataset keys as
the legacy study.

New in this arm: the marginal-comparison metrics are computed on TWO evaluation-grid
scenarios, keyed by the `grid` column - `full` and `chebyshev` in
`marginal_distances.csv`; in `marginal_diagnostics.csv` density-series rows carry
`full`/`chebyshev` while the grid-independent moment-series rows carry `moments`.
Every marginal table/plot is therefore produced once per grid (filename suffix
`_full` / `_chebyshev`).

- **Inputs:** gathered tidy CSVs in `data/out/mixture_c2/` (refreshed via
  `scripts/gather_summaries.py`, or regenerated from the raw posteriors with
  `hpc_analysis/post_process.py`).
- **Outputs:** figures and tables under `hpc_analysis/mixture_models/out/`.
- Entry points: `make_plots.py`, `make_tables.py`, `marginal_winrate.py` (each runnable
  with `.venv/Scripts/python.exe hpc_analysis/<script>.py` from the repo root).

The sibling `analysis/` folder is the UNTOUCHED legacy k5 pipeline (jobs 004-009,
three samplers, single-grid marginals) and its archived results - do not mix the two.
