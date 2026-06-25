# CLAUDE.md

Guidance for working in this repository. Read this first.

## What this repo is

This is **not** the simulation study itself — it is the **HPC execution harness**
(a fork of the "Template for Reproducible Experimentation" by Johannes Brachem,
https://github.com/jobrachem/hpc). Its job is to run many *independent* compute
runs in parallel as SLURM array jobs on the GWDG (GWDG SCC) cluster, then gather
the outputs into tidy CSVs for analysis with minimal post-processing.

The **science** lives in a separate repo:
**HierarchicalBayesianMNL** — https://github.com/Paul-Jarschke/HierarchicalBayesianMNL

**The goal of this project:** port the HierarchicalBayesianMNL simulation study into
this harness as one or more `jobs/`, so all study conditions can be run as
parallelized, reproducible HPC array jobs (and tested locally on Windows first).

### What the study is about

A simulation study comparing two Bayesian approaches to **hierarchical Bayesian
multinomial logit (HBMNL) models with a mixture-of-normals heterogeneity
distribution**:

- **bayesm** (R) — Gibbs sampling with random-walk Metropolis (the established reference).
- **Liesel / Goose** (Python) — gradient-based MCMC: **NUTS**, **HMC**, and an
  experimental **IWLS** sampler.

Identical datasets go through both implementations to isolate differences due to the
*sampler* rather than the *data*. The design deliberately decouples `K_MODEL` (number
of mixture components the model fits) from `K_TRUE` (number in the data-generating
process), so both correctly-specified and overspecified cases are evaluated. Data
mimics the Rossi (2006) margarine example: ~300 decision units × 30 observations,
4 alternatives, 2 demographics. Scenarios: `1comp`, `2comp_equal`, `3comp_equal`,
`5comp_equal`.

## How the harness works (the mental model)

```
jobs/<NNN-name>/
  run.qmd  | run.ipynb   # the per-run notebook; ONE run per invocation
  params.csv             # one ROW = one independent run (the unit of parallelism)
  params.R               # script that generates params.csv (expand_grid of conditions)
  resources.json         # SLURM resources + array batching knobs
  hpc/
    submit.py            # run LOCALLY: ssh to HPC, git pull, renv::restore, render.py, sbatch
    render.py            # runs ON HPC: renders template.sh.j2 -> sbatch.sh for remaining rows
    template.sh.j2       # the SLURM array script (rsync to node-local tmp, quarto render, rsync back)
    download.sh / clear.sh
  log/ out/ finished/    # created at runtime
```

Execution model:
- A run is identified by `(JOB_DIR, JOB_ROW)`. `JOB_ROW` indexes into `params.csv` (0-based).
- The notebook reads its row from `params.csv`, does the work, writes one CSV per output
  type to `out/<type>/results-row<NNNN>.csv`, then `touch`es `finished/<row>` so it is
  never re-run. `JOB_TESTING=True` writes to `out-test/` and skips the finished-marker.
- On the HPC, `template.sh.j2` becomes a SLURM array; each array task renders the notebook
  for one (or `JOBROWS_PER_ARRAY_SUBJOB`) row(s), copying the job dir into node-local
  `$LOCAL_TMPDIR` to avoid temp-file clashes between parallel tasks.
- `resources.json` knobs: `HPC_NUMBER_OF_JOBS_TO_SUBMIT` (rows per submit call),
  `JOBROWS_PER_ARRAY_SUBJOB` (rows per array task), `SBATCH_ARRAY_MAX_CONCURRENT` (`%N`),
  plus standard SBATCH partition/cpus/mem/time.

Existing jobs (use as templates):
- `001-demo-knitr` — Quarto **knitr** engine (R primary, calls Python via `reticulate`/`r`/`py`).
- `002-demo-jupyter` — Quarto **jupyter** engine (Python primary).
- `003-liesel_gam` — **jupyter** engine, real Liesel/Goose model fit. **Closest template
  for the Python HBMNL runs.**

### Dependency management (important)

Dependencies are managed by R's **`renv`** even for Python:
- `renv.lock` pins R packages; `requirements.txt` pins Python packages; `renv::restore()`
  installs both (Python into `.venv/`).
- The study repo upstream uses `uv` + `pyproject.toml`. **In this harness we do NOT use
  `uv`** — every Python dependency the study needs must be added to `requirements.txt`
  (and any R package, e.g. `bayesm`, to `renv` / `renv.lock`).
- Good news: `requirements.txt` already pins `liesel==0.4.3`, `liesel_gam==0.1.0`, `jax`,
  `blackjax`, `arviz`, `plotnine`, `pandas`, `numpy`, etc. — most of the Python stack is
  present. `bayesm` (R) is **not** yet in `renv` and must be added for the R comparison.

## Windows notes (the dev machine is Windows; the HPC is Linux)

Local development/testing happens on **Windows**; the HPC is **Linux + SLURM**. Keep both
working. Adaptations already made for Windows (commit `9fcc60e`), do not regress them:

- `jobs/001-demo-knitr/hpc/submit.py`:
  - calls git-bash explicitly: `["C:/Program Files/Git/bin/bash.exe", ".../check_git_status.sh"]`.
  - normalizes line endings before piping over ssh: `input=submit.replace("\r\n","\n").encode("utf-8")`
    and uses `jobdir.relative_to(basedir).as_posix()` (forward slashes).
  - **The other jobs' `submit.py` (002, 003) still use the un-fixed Linux form** — apply the
    same Windows fixes when you touch them.
- `requirements.txt`: `uvloop==0.22.1; sys_platform != 'win32'` (uvloop has no Windows wheels).
- `.Renviron` was added to recreate the Python env on Windows.
- `.Rprofile` adds an HPC-only `.libPaths()` entry (guarded by `dir.exists`) — harmless on Windows.

Watch out for:
- CRLF in `*.sh` / `*.j2` files breaks bash heredocs on the HPC. Prefer enforcing LF for
  shell/template files (a `.gitattributes` with `*.sh text eol=lf` and `*.j2 text eol=lf`
  would make this robust — not yet present).
- `params.R` and the knitr `run.qmd` use `rstudioapi::getActiveDocumentContext()`, which only
  works inside RStudio. In VS Code/Positron, regenerate `params.csv` another way or run the
  R block in RStudio. The `run.qmd` has a fallback for `JOB_DIR`; `params.R` does not.
- Local single-run testing is the fast path: open a job's `run.qmd`/`run.ipynb`, set
  `JOB_ROW` and `JOB_TESTING=False`, then `quarto render`. Or batch locally via
  `scripts/run_jobs_locally.py` (edit `JOB_PREFIXES`).

## Mapping the study onto the harness (the porting plan)

**Decisions made for this port (2026-06-25):**
- **Vendor study `src/` by copying it into each job dir** (self-contained, reproducible per
  job — accept the duplication; not a shared root package, not a submodule).
- **Pre-generate datasets** with `generate_data.py` into **`data/in/`** and ship them (NOT
  on-the-fly), so every sampler sees byte-identical data.
- **Python samplers (NUTS/HMC/IWLS) first**; the **bayesm (R) comparison is out of scope for
  now** — add it later as a separate job.

| Study repo concept | Harness equivalent |
|---|---|
| `experiment_configs.py` scenarios × samplers × `K_MODEL` strategy × chains × replicate seeds | rows of `params.csv` (each row = one array task). Generate via `params.R`. |
| `generate_data.py` (→ `data/simulated/mixture/<scenario>.json`) | **pre-generate once into `data/in/`** and commit/ship the JSON datasets; the notebook *loads* the dataset for its row (keyed by scenario/seed). |
| `run_single_experiment.py` (per-run entry point) | the **body** of `jobs/<NNN-hbmnl>/run.ipynb` (jupyter engine, like `003`). Its argparse args (`--scenario --k-model --sampler --chains --warmup --posterior --seed --a-mu --a-delta --dirichlet-a --num-integration-steps`) become **columns in `params.csv`**. |
| `src/` modules: `dgp.py`, `mixturemodel.py` (`build_mixture_hbmnl_model`), `analysis.py` (`export_posterior_to_pickle`), `inference/{nuts,hmc,iwls}.py` | **copy `src/` into the job dir** so the notebook can `import` them locally. Add their deps to `requirements.txt`. |
| bayesm (R) comparison | **deferred** — later a separate job (e.g. `0NN-hbmnl-bayesm`, knitr engine) with `bayesm` added to `renv`. |
| outputs `mcmc_results.pkl`, `posterior_raw.pkl`, `meta.json`, `status.json`, `summary.txt` | write a flat **`out/results/results-row<NNNN>.csv`** of summary metrics (what `gather_out_*.R` concatenates) **plus** keep heavy artifacts as extra `out/<type>/` dirs (pickles) if needed. |
| analysis notebooks (`analysis_template.ipynb`, `execute_analysis_notebooks.py`) | R/Python scripts in `analysis/`, reading the gathered `data/out/jobs/results.csv`. |

### Status & remaining steps (updated 2026-06-26)

**Done:**
- **Dependencies** reconciled — the study's Python deps are already in `requirements.txt`
  (liesel / jax / arviz / tfp / ...). No `bayesm`/renv work yet (deferred with the R arm).
- **Data generated.** `data/generate_mixture_data.py` + vendored DGP at `data/dgp/` (copied
  verbatim from the study `@12ca13b`) produced **200 datasets** at
  `data/in/k5model_mixture/kt{K}_s{NN}.json` (K_true ∈ {1,2,3,5} × seeds **1–50**, paired
  design) plus `manifest.csv`. ~1 GB, gitignored, regenerable — on the HPC, regenerate after
  `git pull` rather than uploading.
- **Fit job `jobs/001-k5model-liesel/` built and locally validated** (NUTS + HMC under
  `JOB_TESTING`). Anatomy:
  - `run.qmd` (jupyter engine, `execute: timeout: 7200`) ports `run_single_experiment.py`:
    load dataset by `dataset_key` → `build_mixture_hbmnl_model(K=k_model)` → dispatch
    `run_{nuts,hmc}_inference_mixture_hbmnl` → `src/summaries.py` builds the tidy tables.
  - `src/` = vendored `mixturemodel.py`, `analysis.py`, `inference/{nuts,hmc}.py` + the
    harness-glue `summaries.py` (the only non-vendored module; reuses analysis.py's
    label-switching-aware helpers).
  - **5 tidy output tables**, each in its own `out/<type>/` and carrying all condition
    columns: `diagnostics` (1 row: runtime_s, rhat_max, ess_min, n_sampling_errors),
    `pvec_summary` (K rows, rank-sorted vs truth), `mu_summary` / `sigma_summary` (K×P,
    model→true matched by `linear_sum_assignment`), `recovery` (P rows: bias/rmse/coverage of
    `beta_i`). Gather → `data/out/k5model_mixture/<type>.csv`.
  - Heavy posterior → `out/posterior/<key>.pkl` (component-level mu/sigma/std/pvec only,
    ~58 KB; **kept on the HPC**, not gathered).
  - `params.R` builds the **800-row** grid (200 datasets × {nuts,hmc} × {1,2} chains,
    K_MODEL=5, warmup 2000 / posterior 10000). A 2-row test `params.csv` is in place for
    local validation; running `params.R` replaces it with the full grid.

**Gotchas discovered (important):**
- **`execute: timeout` is mandatory** in the notebook YAML. Without it, nbclient kills the
  kernel ("Kernel died") on any fit longer than its short default — this bit us at ~95 s.
  Applies locally *and* on the HPC (`template.sh.j2` runs `quarto render`). Must exceed the
  slowest single fit; SLURM `SBATCH_TIME` is the real wall-clock cap.
- **`rhat` needs ≥2 chains** — `n_chains=1` rows have `rhat_max` empty (honest); `ess_min`
  is still computed. The `n_chains=2` grid rows populate rhat.
- Data path in `run.qmd` is `JOB_PATH.parent.parent / "data" / "in" / ...` so it resolves
  both locally (cwd = project root) and on the HPC (cwd = node-local rundir, where
  `template.sh.j2` symlinks `data` two levels up).
- Local `quarto render` needs R off-PATH handled + `QUARTO_PYTHON` → the `.venv` python.

**Remaining:**
1. **HPC timing probe** — submit a 1-row *production* fit (one NUTS + one HMC) to measure
   real runtime/memory on the cluster, then finalize `resources.json` (`SBATCH_TIME/MEM`,
   `JOBROWS_PER_ARRAY_SUBJOB`, `SBATCH_ARRAY_MAX_CONCURRENT`). Current values are drafts.
2. **Generate the full grid** — run `params.R` → 800-row `params.csv` (replaces the test grid).
3. **Submit 800** → `squeue --me` → `hpc/download.sh` → `scripts/gather_out_*.R` →
   `data/out/k5model_mixture/`.
4. **Analysis** scripts in `analysis/` over the gathered tables.
5. **(Deferred)** bayesm (R) job + the IWLS sampler.

Also still open: apply the `submit.py` Windows fixes to `002`/`003` if used; add a
`.gitattributes` LF rule for `*.sh`/`*.j2`; fix the UTF-16 `.Renviron` (see WINDOWSSETUP.md).

## Common commands

```powershell
# Restore R + Python envs (run inside an R session in the project root)
R            # bootstraps renv
# then in R:  renv::restore()

# Activate the Python venv (Windows)
.\.venv\Scripts\Activate.ps1

# Render a single run locally for testing (set JOB_ROW / JOB_TESTING in the notebook first)
quarto render jobs/003-liesel_gam/run.qmd --to gfm --execute

# Run all rows of selected jobs locally (edit JOB_PREFIXES in the script)
python scripts/run_jobs_locally.py

# Submit one job to the HPC (run locally; needs .env with HPC_SSH_ALIAS etc.)
python jobs/001-demo-knitr/hpc/submit.py

# Gather outputs into data/out/ (run in an interactive R session)
#   scripts/gather_out_greedy.R   (fewer files) or scripts/gather_out_lazy.R
```

HPC setup (SSH, micromamba `r-4.5`, Quarto, `.env`/`.dotenv`) is documented in
`guides/hpc_setup.md`. `guides/hpc_workflow.md` and `guides/before_publication.md` cover the
day-to-day workflow and release checklist.

## Conventions

- ALL-CAPS notebook variables (`JOB_DIR`, `JOB_ROW`, `JOB_TESTING`, `PARAMS`, `DIR_OUT`, …)
  are harness machinery — don't rename them; only touch the "Model Code" / "Save Results"
  sections per job.
- Every output dataframe must carry enough of `PARAMS` (job, run id, row, condition columns)
  to identify the exact run that produced it — the gather scripts concatenate rows blindly.
- One run = one row of `params.csv`. Keep runs independent (no shared mutable state) so they
  parallelize cleanly.
- Reproducibility: seed every RNG from a `params.csv` column (the demos use `JOB_ROW`; the
  study should use an explicit `data_seed`/`seed`).
