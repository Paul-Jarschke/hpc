# standard_model

Analysis home for the STANDARD HBMNL runs: a single normal heterogeneity component
(no mixture), Rossi (2006) section 5.4 - jobs `200-standard-bayesm`,
`201-standard-hmc`, `202-standard-nuts` (2 chains, 100 replicate seeds, datasets in
`data/in/standard_model/`).

The jobs write per-run tidy tables via `src/summaries_standard.py`
(mu / posterior Sigma incl. empirical-covariance overlay / Delta / beta recovery,
per-parameter R-hat/ESS convergence, moments, marginal distances + density
diagnostics on the two grid scenarios full / chebyshev, sampler diagnostics).

## Pipeline

1. **Gather** the per-run tables into `data/out/standard_model/*.csv`:

       .venv/Scripts/python.exe scripts/gather_summaries.py --glob "jobs/20[0-2]-standard-*" --out-name standard_model

   (the mixture-only tables - ecr_report, weights, pvec_means - simply report
   "(no files)"). Alternatively regenerate everything from the raw Tier-2
   posteriors with `hpc_analysis/standard_model/post_process.py` (imports
   `src.summaries_standard`, reads `data/in/standard_model/` truths).

2. **Figures**: `.venv/Scripts/python.exe hpc_analysis/standard_model/make_plots.py`
   regenerates the full figure set under `hpc_analysis/standard_model/out/`
   (delta, mu, sigma, runtime, marginal comparison incl. per-grid
   marginal-series ESS/R-hat diagnostics).

3. **Tables**: `.venv/Scripts/python.exe hpc_analysis/standard_model/make_tables.py`
   writes summary CSVs under `hpc_analysis/standard_model/out/<topic>/tables/`
   (runtime, delta bias/MSE + MCSEs + SD,
   mu recovery, sigma recovery incl. the `empirical` reference column, marginal
   distance summaries per grid, marginal ESS/R-hat summaries).

`marginal_diag.py` is also runnable standalone.

## Scripts

- `plot_recovery.py` - data loading (`load_recovery`, `RECOVERY_FILES`), the general
  `recovery_boxplot` core, sampler palette (`SAMPLER_ORDER/LABELS/COLORS`), all
  Delta plot functions ported from `../mixture_models/`, plus the
  standard-model-specific `mu_bias_by_param`/`mu_mse_by_param` and
  `sigma_bias_faceted_by_element`/`sigma_mse_faceted_by_element` analyses, runtime plots.
- `make_plots.py` / `make_tables.py` - the two entry points (see above).
- `marginal_diag.py` - ESS/R-hat of the marginal-density and moment series, per grid.
- `post_process.py` - offline re-derivation of the gathered CSVs from
  `posterior_raw/*.pkl` (Tier-2, data_seed <= 5 only on the real grid).

## Differences vs `../mixture_models/`

- Three samplers (bayesm, nuts, hmc) - no bayesm_gibbs replication arm.
- No label switching (K=1): no ECR, weights, pvec or component-count analyses.
- k_true == k_model == 1 is a single condition cell, so every mixture-side
  "per k_true" figure/table variant collapses to one overall version here.
- `convergence.csv` has directly meaningful per-parameter R-hat/ESS (quantity column:
  `mu:<param>`, `Sigma:<row>,<col>`, `tr(Sigma)`, `Delta:<demo>:<param>`,
  `beta[unit0/N-1]:<param>`).
- `sigma_recovery.csv` includes an `empirical` column (covariance of the true unit
  betas) matching the notebook's overlay; it is reported in the sigma summary table.
- Marginal tables carry the same `grid` column (`full` / `chebyshev` / `moments`).
- The study's standard model-comparison notebook additionally compares the mu
  posteriors across samplers DIRECTLY (valid with K=1) - a candidate future plot here,
  using the Tier-2 posteriors (data_seed <= 5).
