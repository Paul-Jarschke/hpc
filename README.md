# Sampler comparison for hierarchical Bayesian multinomial logit models

This repository holds the computational harness for a simulation study that compares
MCMC samplers for hierarchical Bayesian multinomial logit (HBMNL) models with
mixture-of-normals heterogeneity. Every fit is an independent job run on an HPC
cluster; the per-run outputs are gathered into tidy tables and turned into the figures
and tables reported in the thesis.

The statistical model, samplers and marginal-comparison metrics are vendored verbatim
from the study repository
[HierarchicalBayesianMultinomialLogit](https://github.com/Paul-Jarschke/HierarchicalBayesianMultinomialLogit)
so this harness reproduces the study without depending on that repo. See
[`src/README.md`](src/README.md) for what is vendored and what is harness glue.

## The two studies

Both use the same DGP family (300 decision units, 30 choice tasks each, 4 alternatives,
2 demographic covariates), 100 replicate datasets per condition, and 2 chains per fit.

**Mixture study** (jobs `100`-`103`). A 5-component mixture is fitted (`K_MODEL = 5`)
to data generated with `k_true` in {1, 2, 3, 5} (scenarios `1comp`, `2comp_equal`,
`3comp_equal`, `5comp_equal`), so the model is correctly specified in one cell and
overspecified in the others. Four samplers, 400 runs each:

| Job | Sampler |
|-----|---------|
| `100_mixture_bayesm` | `bayesm::rhierMnlRwMixture` (R, the reference implementation) |
| `101_mixture_replicate` | a line-faithful JAX/Goose port of that Gibbs sampler ("Replication") |
| `102_mixture_hmc` | HMC |
| `103_mixture_nuts` | NUTS |

**Standard study** (jobs `200`-`202`). A single normal heterogeneity component
(`k_true = K_MODEL = 1`, Rossi 2006 section 5.4), which removes label switching and
makes mu and Sigma directly interpretable. Three samplers (`bayesm`, `hmc`, `nuts`),
100 runs each.

Because the mixture posterior is invariant to component relabeling, the mixture
analysis is built on label-invariant quantities (Delta, the marginal heterogeneity
distribution, ECR-relabeled weights) rather than per-component parameters.

## Repository contents

- `jobs/` - one directory per compute job. Each holds `run.qmd` (the notebook executed
  once per row of `params.csv`), `params.R` (builds `params.csv` from the dataset
  manifest), `resources.json` (SLURM resources) and `hpc/` (job-script template).
- `data/dgp/` - the vendored data-generating process (do not edit, see
  [`data/dgp/VENDORED.md`](data/dgp/VENDORED.md)).
- `data/generate_mixture_data.py`, `data/generate_standard_data.py` - deterministic
  dataset generators; they write `data/in/<experiment>/` plus a `manifest.csv`.
- `data/in/` - generated input datasets. `data/out/` - gathered per-run tables
  (`mixture_c2/`, `standard_model/`).
- `src/` - model, samplers and summary code shared by all jobs.
- `hpc_analysis/` - the analysis pipeline (figures, CSV tables, LaTeX fragments), one
  subfolder per study. See [`hpc_analysis/README.md`](hpc_analysis/README.md).
- `scripts/` - helpers to run, submit, download and gather jobs.
- `guides/` - HPC setup and workflow guides.
- [`Documentation.md`](Documentation.md) - notable data/modelling issues and how they
  are handled.

## Reproducing the analysis

This is the cheap path: it reads the gathered tables in `data/out/` and rewrites every
figure and table. Nothing is refitted.

```shell
# mixture study
.venv/Scripts/python.exe hpc_analysis/mixture_models/make_tables.py
.venv/Scripts/python.exe hpc_analysis/mixture_models/make_plots.py

# standard study
.venv/Scripts/python.exe hpc_analysis/standard_model/make_tables.py
.venv/Scripts/python.exe hpc_analysis/standard_model/make_plots.py

# LaTeX table fragments for the thesis
.venv/Scripts/python.exe hpc_analysis/make_tex_tables.py
```

Figures and CSV tables land in `hpc_analysis/<study>/out/`, LaTeX fragments in
`hpc_analysis/tex_tables/`. On Linux/macOS use `.venv/bin/python` instead.

## Reproducing the experiments

### Environment setup

R and Python dependencies are both managed through
[`renv`](https://rstudio.github.io/renv/). Start an R session in the project directory
(this bootstraps `renv`):

```shell
R
```

and restore both environments:

```r
renv::restore()
```

This installs the R packages from `renv.lock` and the Python packages from
`requirements.txt` into `.venv/`. Activate the Python environment with
`source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\activate` (Windows).

### Generate the input data

The generators are deterministic in the dataset seed, so the datasets can always be
rebuilt instead of downloaded:

```shell
python data/generate_mixture_data.py     # -> data/in/k5model_mixture/
python data/generate_standard_data.py    # -> data/in/standard_model/
```

Each writes a `manifest.csv` that the jobs' `params.R` reads to build `params.csv`.

### Run the jobs

A single run, interactively: open the job's `run.qmd`, point the `JOB_ROW` parameter at
the row of `params.csv` you want, set `JOB_TESTING` to `False`, and render it
(`quarto render run.qmd`).

Selected jobs in sequence, locally (set `JOB_PREFIXES` inside the script first):

```shell
python scripts/run_jobs_locally.py
```

All jobs on an HPC (this is how the study was actually run; 1900 fits in total):

```shell
python scripts/submit_all.py      # after completing guides/hpc_setup.md
python scripts/download_all.py    # once the arrays have finished
```

### Gather the per-run outputs

Each run writes small tidy CSVs; the gather step concatenates them per study:

```shell
# mixture study (these are the script defaults) -> data/out/mixture_c2/
python scripts/gather_summaries.py

# standard study -> data/out/standard_model/
python scripts/gather_summaries.py --glob "jobs/20[0-2]-standard-*" --out-name standard_model
```

After this the analysis scripts above can be run.

## Known data issue

One generated dataset (`kt1_s70`) cannot be fitted by `bayesm`, because only 3 of the 4
alternatives are ever chosen and `rhierMnlRwMixture` requires all of them. It is
screened out at generation time and backfilled from a spare seed, so every sampler is
compared on the same 100 datasets per scenario. The full account is in
[`Documentation.md`](Documentation.md).

## Attribution

This repository was created based on the "Template for Reproducible Experimentation", Johannes Brachem (2026): https://github.com/jobrachem/hpc.
