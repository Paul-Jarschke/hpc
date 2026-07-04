# standard_model

Analysis home for the STANDARD HBMNL runs: a single normal heterogeneity component
(no mixture), Rossi (2006) section 5.4 - jobs `200-standard-bayesm`,
`201-standard-hmc`, `202-standard-nuts` (2 chains, 100 replicate seeds, datasets in
`data/in/standard_model/`).

Status: the JOBS exist and write per-run tidy tables via `src/summaries_standard.py`
(mu / posterior Sigma incl. empirical-covariance overlay / Delta / beta recovery,
per-parameter R-hat/ESS convergence, moments, marginal distances + density
diagnostics on the two grid scenarios full / chebyshev, sampler diagnostics).
Gather them with:

    .venv/Scripts/python.exe scripts/gather_summaries.py --glob "jobs/20[0-2]-standard-*" --out-name standard_model

which writes `data/out/standard_model/*.csv` (the mixture-only tables - ecr_report,
weights, pvec_means - simply report "(no files)").

The plotting/table scripts for this folder are NOT yet built. When porting them,
mirror `../mixture_models/` with these differences:
- Three samplers (bayesm, nuts, hmc) - no bayesm_gibbs replication arm.
- No label switching (K=1): no ECR, weights, pvec or component-count analyses.
- `convergence.csv` has directly meaningful per-parameter R-hat/ESS (quantity column:
  `mu:<param>`, `Sigma:<row>,<col>`, `tr(Sigma)`, `Delta:<demo>:<param>`,
  `beta[unit0/N-1]:<param>`).
- `sigma_recovery.csv` includes an `empirical` column (covariance of the true unit
  betas) matching the notebook's overlay.
- Marginal tables carry the same `grid` column (`full` / `chebyshev` / `moments`).
- The study's standard model-comparison notebook additionally compares the mu
  posteriors across samplers DIRECTLY (valid with K=1) - a candidate plot here,
  using the Tier-2 posteriors (data_seed <= 5).
