# Shared model/sampler/analysis code

This `src/` package is imported by all mixture fit jobs (`jobs/100`-`103`, previously
`jobs/004`-`009`). It lives at the repo root (not per-job) so the near-identical jobs
share one copy.

How it's found:
- **Locally:** `run.qmd` puts the repo root (`JOB_PATH/../..`) on `sys.path`.
- **On the HPC:** each job's `hpc/template.sh.j2` symlinks it into the node-local rundir
  (`ln -s $HOME/<proj>/src $RUNDIR/src`), and `run.qmd` puts the rundir on `sys.path`.

Contents:
- `mixturemodel.py`, `bayesm_mixture_model.py`, `analysis.py`, `label_switching.py`,
  `marginal_comparison.py`, `inference/{nuts,hmc,init,bayesm_gibbs}.py` - **vendored
  verbatim** from HierarchicalBayesianMultinomialLogit @ `893e63f`
  (https://github.com/Paul-Jarschke/HierarchicalBayesianMultinomialLogit).
  Do not edit; re-vendor from upstream if needed. Highlights of that revision:
  - NUTS/HMC use bayesm's own initialisation scheme per chain (`inference/init.py`)
    and a dense mass matrix on the Sigma block.
  - `inference/bayesm_gibbs.py` - line-faithful JAX/Goose port of
    `bayesm::rhierMnlRwMixture` on the augmented model with explicit allocations
    (`bayesm_mixture_model.py`); the "replication" arm (job 101).
  - `marginal_comparison.py` gained the two grid scenarios `build_grids_full`
    (raw +/- 6 sigma envelope) and `build_grids_chebyshev` (aggregate-moment
    mean +/- 5 sigma window, >= 96% mass by Chebyshev's inequality).
- `__init__.py` - **harness patch** (not vendored): restores the `np.trapz` alias that
  numpy >= 2.0 removed but the vendored `marginal_comparison.py` still calls.
- `summaries.py` - **harness glue** (not vendored): builds the tidy per-run output
  tables (see `TABLE_NAMES`) from the posterior samples + the dataset's ground truth,
  including the marginal-distance / density-diagnostic tables on BOTH grid scenarios
  (`grid` column: `full` / `chebyshev`). Also hosts `delta_summary_rows`,
  `beta_summary_rows` and `pvec_mean_table`, which upstream removed from the vendored
  modules but the harness tables still use.
- `bayesm_sampler.R` - vendored + lightly adapted from the study's
  `run_single_bayesm_experiment.R` (function-wrapped so knitr jobs can `source()` it;
  sampling logic identical, verified against upstream @ 893e63f).
- `bayesm_convert.py` - harness module: rebuilds the canonical `posterior_raw.pkl`
  (TFP `FillScaleTriL` latent) from the R bridge files; conversion math identical to
  the study's `run_single_bayesm_experiment.py`. Pinned by
  `tests/test_bayesm_convert.py`.
