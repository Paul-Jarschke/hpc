# standard_model (placeholder)

Analysis for the upcoming STANDARD HBMNL runs: a single normal heterogeneity
component (no mixture), Rossi (2006) section 5.4. Not yet populated - the jobs do not
exist yet.

When those jobs are built, this folder gets the analogue of `../mixture_models/`:
scripts reading the gathered tables from `data/out/<standard-out-name>/` and writing
figures/tables to `standard_model/out/`. Relevant upstream support already vendored in
`src/`: `marginal_comparison.load_sampler_standard` / `true_dgp_standard` (K=1
packaging with a size-1 component axis and pvec == 1), plus the study repo's
`standardmodel.py` / `standard_analysis_template.ipynb` /
`standard_model_comparison_template.ipynb` (not yet vendored - vendor when porting).

Notes for the port:
- The standard model has no label switching (K=1), so no ECR / relabeling tables.
- Posterior keys differ: `mu`, `sigma_inv_chol_latent` (no `_k` suffix, no pvec).
- The two marginal grid scenarios (`full` / `chebyshev`) apply unchanged.
