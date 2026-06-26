# Shared model/sampler/analysis code

This `src/` package is imported by all `k5model_mixture` fit jobs (`jobs/004`–`007`).
It lives at the repo root (not per-job) so the four near-identical jobs share one copy.

How it's found:
- **Locally:** `run.qmd` puts the repo root (`JOB_PATH/../..`) on `sys.path`.
- **On the HPC:** each job's `hpc/template.sh.j2` symlinks it into the node-local rundir
  (`ln -s $HOME/<proj>/src $RUNDIR/src`), and `run.qmd` puts the rundir on `sys.path`.

Contents:
- `mixturemodel.py`, `analysis.py`, `inference/{nuts,hmc}.py` — **vendored verbatim** from
  HierarchicalBayesianMNL @ `12ca13b`. Do not edit; re-vendor from upstream if needed.
- `summaries.py` — **harness glue** (not vendored): builds the 5 tidy output tables
  (`diagnostics`, `pvec_summary`, `mu_summary`, `sigma_summary`, `recovery`) from the
  posterior samples + the dataset's ground truth.
