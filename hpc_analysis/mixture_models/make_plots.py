"""
Regenerate the full mixture_c2 figure set from data/out/mixture_c2/*.csv.

Run after scripts/gather_summaries.py whenever new runs are gathered. Writes PNGs to
hpc_analysis/mixture_models/out/{delta,runtime,beta,marginal_comparison,components}/. All figures cover
the 2-chain (c2) jobs 100-103 with the four samplers bayesm / bayesm_gibbs / nuts / hmc.
Marginal-distance figures are produced once per evaluation grid ('full' and 'chebyshev';
filename suffix _<grid>).

    .venv/Scripts/python.exe hpc_analysis/make_plots.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_recovery import (  # noqa: E402
    save,
    compute_beta_correlation,
    delta_bias_faceted_by_element,
    delta_sd_faceted_by_element,
    delta_rmse_faceted_by_element,
    delta_coverage_faceted_by_element,
    delta_coverage_by_ktrue,
    beta_rmse_by_param,
    beta_correlation_by_param,
    beta_correlation_by_ktrue,
    beta_coverage_by_param,
    beta_coverage_by_ktrue,
    marginal_metric_boxplot,
    marginal_distance_by_ktrue,
    marginal_distances_faceted_by_metric,
    retained_mass_boxplot,
    MARGINAL_METRICS,
    runtime_by_ktrue,
    runtime_samplers_by_ktrue,
)
import component_count  # noqa: E402
import marginal_diag  # noqa: E402

CHAINS = 2
KTRUE = [1, 2, 3, 5]
SAMPLERS = ["bayesm", "bayesm_gibbs", "nuts", "hmc"]
GRIDS = ["full", "chebyshev"]  # marginal-distance evaluation-grid scenarios
GRID_FOLDER = {"full": "full", "chebyshev": "trimmed"}  # output subfolder per grid


def main():
    # Delta bias: 4x2 element grid, free y-scale per panel, transparent boxes, points by sampler.
    for kt in KTRUE:
        save(delta_bias_faceted_by_element(CHAINS, kt), f"delta/bias/plots/delta_bias_elements_kt{kt}.png")

    # Delta posterior SD: same layout as bias grid.
    for kt in KTRUE:
        save(delta_sd_faceted_by_element(CHAINS, kt), f"delta/sd/plots/delta_sd_elements_kt{kt}.png")

    # Delta absolute error: same layout, y = |post_mean - true_value| per seed.
    for kt in KTRUE:
        save(delta_rmse_faceted_by_element(CHAINS, kt), f"delta/rmse/plots/delta_rmse_elements_kt{kt}.png")

    # Delta coverage: bar chart, one bar per sampler per element, 95% reference line.
    for kt in KTRUE:
        save(delta_coverage_faceted_by_element(CHAINS, kt), f"delta/coverage/plots/delta_coverage_kt{kt}.png")

    # Delta coverage by k_true: dodged bars (sampler) on x=k_true, all k_true in one figure.
    save(delta_coverage_by_ktrue(CHAINS), f"delta/coverage/plots/delta_coverage_by_ktrue.png")

    # Runtime: per sampler by k_true (nuts in hours; bayesm/bayesm_gibbs/hmc in minutes, linear).
    for s in SAMPLERS:
        save(runtime_by_ktrue(s, CHAINS), f"runtime/plots/runtime_{s}_by_ktrue.png")

    # Runtime: all samplers in one figure (log scale).
    save(runtime_samplers_by_ktrue(CHAINS), "runtime/plots/runtime_samplers_by_ktrue.png")

    # Beta RMSE: 1x4 parameter grid, distribution over seeds.
    for kt in KTRUE:
        save(beta_rmse_by_param(CHAINS, kt), f"beta/rmse/plots/beta_rmse_kt{kt}.png")

    # Beta correlation: load beta_summary once (large) then reuse across all k_true.
    print("computing beta correlations from beta_summary.csv ...")
    corr_df = compute_beta_correlation()
    for kt in KTRUE:
        save(beta_correlation_by_param(CHAINS, kt, corr_df=corr_df),
             f"beta/correlation/plots/beta_correlation_kt{kt}.png")
    save(beta_correlation_by_ktrue(CHAINS, corr_df=corr_df),
         f"beta/correlation/plots/beta_correlation_by_ktrue.png")

    # Beta coverage: bar chart per k_true + one combined by-k_true figure.
    for kt in KTRUE:
        save(beta_coverage_by_param(CHAINS, kt), f"beta/coverage/plots/beta_coverage_kt{kt}.png")
    save(beta_coverage_by_ktrue(CHAINS), f"beta/coverage/plots/beta_coverage_by_ktrue.png")

    # Marginal comparison: all output under marginal_comparison/, once PER GRID scenario.
    # Per metric a sampler boxplot (x=sampler, k_true x param grid) + the by-k_true view;
    # plus the per-k_true all-metric grid. Guarded so a partial-data run (e.g. one job not
    # finished yet) plots what's available, not aborts.
    for grid in GRIDS:
        for metric in MARGINAL_METRICS:
            slug = metric.lower().replace("-", "").replace(" ", "_")
            try:
                save(marginal_metric_boxplot(metric, CHAINS, grid=grid),
                     f"marginal_comparison/{GRID_FOLDER[grid]}/plots/{slug}_boxplot.png")
            except ValueError as e:
                print(f"  skip {slug}_boxplot ({grid}): {e}")
            try:
                save(marginal_distance_by_ktrue(CHAINS, metric, grid=grid),
                     f"marginal_comparison/{GRID_FOLDER[grid]}/plots/{slug}_by_ktrue.png")
            except ValueError as e:
                print(f"  skip {slug}_by_ktrue ({grid}): {e}")
        for kt in KTRUE:
            try:
                save(marginal_distances_faceted_by_metric(CHAINS, kt, grid=grid),
                     f"marginal_comparison/{GRID_FOLDER[grid]}/plots/all_metrics_kt{kt}.png")
            except ValueError as e:
                print(f"  skip all_metrics_kt{kt} ({grid}): {e}")

    # Retained probability mass vs the theoretical Chebyshev guarantee (chebyshev grid
    # only - the full grid trivially retains ~100%).
    try:
        save(retained_mass_boxplot(CHAINS, grid="chebyshev"),
             f"marginal_comparison/{GRID_FOLDER['chebyshev']}/plots/retained_mass_boxplot.png")
    except ValueError as e:
        print(f"  skip retained_mass_boxplot: {e}")

    # Component-count: effective-K boxplot, est_k confusion by k_true, and the sorted
    # weight profiles (after-ECR + before/after) under components/.
    component_count.make_plots(CHAINS)

    # Marginal-series convergence: ESS/R-hat grids (density x mean x variance) + density-only
    # figures, under marginal_comparison/ (density series once per grid).
    marginal_diag.make_plots(CHAINS)

    # Consolidated RMSE: one pooled per-run number per parameter block, saved into
    # each block's own rmse/plots folder.
    from plot_recovery import consolidated_rmse_boxplot
    save(consolidated_rmse_boxplot("beta", CHAINS),
         f"beta/rmse/plots/beta_consolidated_rmse.png")
    save(consolidated_rmse_boxplot("delta", CHAINS),
         f"delta/rmse/plots/delta_consolidated_rmse.png")

    print("regenerated all figures -> hpc_analysis/mixture_models/out/")


if __name__ == "__main__":
    main()
