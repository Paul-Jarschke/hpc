# hpc_analysis

Post-modeling analysis for the CURRENT generation of HPC runs (jobs `100+`, vendored
from HierarchicalBayesianMultinomialLogit @ 893e63f). The legacy `analysis/` folder is
frozen: it holds the scripts and results of the old `004`-`009` (k5model_mixture) runs
and must not be modified.

One subfolder per model family, each self-contained (scripts + `out/` results):

- **`mixture_models/`** - the mixture HBMNL comparison (jobs `100_mixture_bayesm`,
  `101_mixture_replicate`, `102_mixture_hmc`, `103_mixture_nuts`; 2 chains,
  K_MODEL = 5, four samplers incl. the bayesm_gibbs replication, marginal metrics on
  the two grid scenarios `full` / `chebyshev`).
- **`standard_model/`** - placeholder for the upcoming standard HBMNL runs (single
  normal component, no mixture); to be populated when those jobs exist.

Data flow: HPC runs write per-run summary CSVs -> `scripts/gather_summaries.py`
concatenates them into `data/out/<out-name>/` -> the scripts here read those tables
and write figures/tables into `<subfolder>/out/`.
