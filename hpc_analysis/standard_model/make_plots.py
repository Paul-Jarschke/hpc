# Regenerate the full standard_model figure set (jobs 200-202, c2 only).
# Reads data/out/standard_model/*.csv, writes PNGs under out/<topic>/.
# Marginal figures once per grid ('full' -> full/, 'chebyshev' -> trimmed/),
# incl. the ESS/R-hat diagnostics from marginal_diag. Run after gathering:
# .venv/Scripts/python.exe hpc_analysis/standard_model/make_plots.py

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_recovery import (  # noqa: E402
    save,
    delta_bias_faceted_by_element,
    delta_sd_faceted_by_element,
    delta_mse_faceted_by_element,
    mu_bias_by_param,
    mu_mse_by_param,
    sigma_bias_faceted_by_element,
    sigma_mse_faceted_by_element,
    marginal_metric_boxplot,
    marginal_distances_faceted_by_metric,
    retained_mass_boxplot,
    kl_inf_count_plot,
    MARGINAL_METRICS,
    runtime_by_sampler,
)
import marginal_diag  # noqa: E402

CHAINS = 2
SAMPLERS = ["bayesm", "nuts", "hmc"]
GRIDS = ["full", "chebyshev"]  # marginal-distance evaluation-grid scenarios
GRID_FOLDER = {"full": "full", "chebyshev": "trimmed"}  # output subfolder per grid


def main():
    # Delta bias: 4x2 element grid, free y-scale per panel, transparent boxes, points by sampler.
    save(delta_bias_faceted_by_element(CHAINS), f"delta/bias/plots/delta_bias_elements.png")

    # Delta posterior SD: same layout as bias grid.
    save(delta_sd_faceted_by_element(CHAINS), f"delta/sd/plots/delta_sd_elements.png")

    # Delta squared error: same layout, y = (post_mean - true_value)^2 per seed (box mean = MSE).
    save(delta_mse_faceted_by_element(CHAINS), f"delta/mse/plots/delta_mse_elements.png")

    # Mu recovery (standard-model specific): bias + squared-error boxplots per parameter.
    save(mu_bias_by_param(CHAINS), f"mu/plots/mu_bias.png")
    save(mu_mse_by_param(CHAINS), f"mu/plots/mu_mse.png")

    # Posterior Sigma recovery (standard-model specific): signed error + squared error,
    # one panel per lower-triangle element.
    save(sigma_bias_faceted_by_element(CHAINS), f"sigma/plots/sigma_bias_elements.png")
    save(sigma_mse_faceted_by_element(CHAINS), f"sigma/plots/sigma_mse_elements.png")

    # Runtime: all samplers in one figure (log scale).
    save(runtime_by_sampler(CHAINS), f"runtime/plots/runtime_by_sampler.png")

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

    print("regenerated all figures -> hpc_analysis/standard_model/out/")


if __name__ == "__main__":
    main()
