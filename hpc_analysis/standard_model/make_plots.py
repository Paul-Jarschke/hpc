"""
Regenerate the full standard_model figure set from data/out/standard_model/*.csv.

Run after the gather step whenever new runs are gathered. Writes PNGs to
hpc_analysis/standard_model/out/{delta,mu,sigma,beta,runtime,marginal_comparison}/.
All figures cover the 2-chain (c2) jobs 200-202 with the three samplers
bayesm / nuts / hmc; the standard model is a single condition cell
(k_true == k_model == 1), so there are no per-k_true figure variants.
Marginal-distance figures are produced once per evaluation grid ('full' and
'chebyshev'; full/trimmed output subfolder), including the head-to-head win-rate
figures + tables (marginal_winrate) and the marginal-series ESS/R-hat diagnostics
(marginal_diag).

    .venv/Scripts/python.exe hpc_analysis/standard_model/make_plots.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_recovery import (  # noqa: E402
    save,
    load_recovery,
    compute_beta_correlation,
    compute_beta_post_std,
    delta_bias_faceted_by_element,
    delta_sd_faceted_by_element,
    delta_rmse_faceted_by_element,
    delta_coverage_faceted_by_element,
    mu_bias_by_param,
    mu_coverage_by_param,
    sigma_bias_faceted_by_element,
    beta_bias_by_param,
    beta_sd_by_param,
    beta_rmse_by_param,
    beta_correlation_by_param,
    beta_coverage_by_param,
    marginal_metric_boxplot,
    marginal_distances_faceted_by_metric,
    retained_mass_boxplot,
    kl_inf_count_plot,
    MARGINAL_METRICS,
    runtime_by_sampler,
    consolidated_rmse_boxplot,
)
import marginal_diag  # noqa: E402
import marginal_winrate  # noqa: E402

CHAINS = 2
SAMPLERS = ["bayesm", "nuts", "hmc"]
GRIDS = ["full", "chebyshev"]  # marginal-distance evaluation-grid scenarios
GRID_FOLDER = {"full": "full", "chebyshev": "trimmed"}  # output subfolder per grid


def main():
    # Delta bias: 4x2 element grid, free y-scale per panel, transparent boxes, points by sampler.
    save(delta_bias_faceted_by_element(CHAINS), f"delta/bias/plots/delta_bias_elements.png")

    # Delta posterior SD: same layout as bias grid.
    save(delta_sd_faceted_by_element(CHAINS), f"delta/sd/plots/delta_sd_elements.png")

    # Delta absolute error: same layout, y = |post_mean - true_value| per seed.
    save(delta_rmse_faceted_by_element(CHAINS), f"delta/rmse/plots/delta_rmse_elements.png")

    # Delta coverage: bar chart, one bar per sampler per element, 95% reference line.
    save(delta_coverage_faceted_by_element(CHAINS), f"delta/coverage/plots/delta_coverage.png")

    # Mu recovery (standard-model specific): bias boxplot + coverage bars per parameter.
    save(mu_bias_by_param(CHAINS), f"mu/plots/mu_bias.png")
    save(mu_coverage_by_param(CHAINS), f"mu/plots/mu_coverage.png")

    # Posterior Sigma recovery (standard-model specific): signed element errors,
    # one panel per lower-triangle element.
    save(sigma_bias_faceted_by_element(CHAINS), f"sigma/plots/sigma_bias_elements.png")

    # Runtime: all samplers in one figure (log scale).
    save(runtime_by_sampler(CHAINS), f"runtime/plots/runtime_by_sampler.png")

    # Beta bias: 1x4 parameter grid, signed error distribution over seeds, 0 reference.
    save(beta_bias_by_param(CHAINS), f"beta/bias/plots/beta_bias.png")

    # Beta RMSE: 1x4 parameter grid, distribution over seeds.
    save(beta_rmse_by_param(CHAINS), f"beta/rmse/plots/beta_rmse.png")

    # Beta posterior SD and correlation both derive from beta_summary.csv (large),
    # so load it once and feed both.
    print("loading beta_summary.csv for posterior SD + correlation plots ...")
    df_summary = load_recovery("beta_summary")

    # Beta posterior SD: mean over units of post_std, 1x4 parameter grid over seeds.
    sd_df = compute_beta_post_std(df_summary)
    save(beta_sd_by_param(CHAINS, sd_df=sd_df), f"beta/sd/plots/beta_sd.png")

    # Beta correlation.
    corr_df = compute_beta_correlation(df_summary)
    save(beta_correlation_by_param(CHAINS, corr_df=corr_df),
         f"beta/correlation/plots/beta_correlation.png")

    # Beta coverage: bar chart per parameter.
    save(beta_coverage_by_param(CHAINS), f"beta/coverage/plots/beta_coverage.png")

    # Marginal comparison: all output under marginal_comparison/, once PER GRID scenario.
    # Per metric a sampler boxplot (x=sampler, 1x4 param facets) plus the all-metric grid.
    # Guarded so a partial-data run (e.g. one job not finished yet) plots what's
    # available, not aborts.
    for grid in GRIDS:
        for metric in MARGINAL_METRICS:
            slug = metric.lower().replace("-", "").replace(" ", "_")
            try:
                save(marginal_metric_boxplot(metric, CHAINS, grid=grid),
                     f"marginal_comparison/{GRID_FOLDER[grid]}/plots/{slug}_boxplot.png")
            except ValueError as e:
                print(f"  skip {slug}_boxplot ({grid}): {e}")
        try:
            save(marginal_distances_faceted_by_metric(CHAINS, grid=grid),
                 f"marginal_comparison/{GRID_FOLDER[grid]}/plots/all_metrics.png")
        except ValueError as e:
            print(f"  skip all_metrics ({grid}): {e}")

    # Retained probability mass vs the theoretical Chebyshev guarantee (chebyshev grid
    # only - the full grid trivially retains ~100%).
    try:
        save(retained_mass_boxplot(CHAINS, grid="chebyshev"),
             f"marginal_comparison/{GRID_FOLDER['chebyshev']}/plots/retained_mass_boxplot.png")
    except ValueError as e:
        print(f"  skip retained_mass_boxplot: {e}")

    # KL = inf counts (one per evaluation grid - 'full' is far more prone to this).
    for grid in GRIDS:
        try:
            save(kl_inf_count_plot(CHAINS, grid=grid),
                 f"marginal_comparison/{GRID_FOLDER[grid]}/plots/kl_inf_count.png")
        except ValueError as e:
            print(f"  skip kl_inf_count ({grid}): {e}")

    # Marginal-series convergence: ESS/R-hat grids (density x mean x variance) + density-only
    # figures, under marginal_comparison/ (density series once per grid).
    marginal_diag.make_plots(CHAINS)

    # Head-to-head win rates on the marginal distances (tables + one figure per pairwise
    # sampler comparison, per grid).
    marginal_winrate.main()

    # Consolidated RMSE: one pooled per-run number per parameter block, saved into
    # each block's own rmse/plots folder.
    save(consolidated_rmse_boxplot("beta", CHAINS),
         f"beta/rmse/plots/beta_consolidated_rmse.png")
    save(consolidated_rmse_boxplot("delta", CHAINS),
         f"delta/rmse/plots/delta_consolidated_rmse.png")

    print("regenerated all figures -> hpc_analysis/standard_model/out/")


if __name__ == "__main__":
    main()
