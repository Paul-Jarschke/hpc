"""
Regenerate the full k5model_mixture figure set from data/out/k5model_mixture/*.csv.

Run after scripts/gather_summaries.py whenever new runs are gathered. Writes PNGs to
analysis/out/k5_results/{delta,runtime,beta}/. All figures cover the 2-chain (c2) arm,
where nuts/hmc/bayesm are complete.

    .venv/Scripts/python.exe analysis/make_plots.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_recovery import (  # noqa: E402
    save,
    delta_bias_faceted_by_element,
    delta_sd_faceted_by_element,
    delta_rmse_faceted_by_element,
    runtime_by_ktrue,
    runtime_samplers_by_ktrue,
)

CHAINS = 2
KTRUE = [1, 2, 3, 5]
SAMPLERS = ["nuts", "hmc", "bayesm"]


def main():
    # Delta bias: 4x2 element grid, free y-scale per panel, transparent boxes, points by sampler.
    for kt in KTRUE:
        save(delta_bias_faceted_by_element(CHAINS, kt), f"delta/bias/plots/delta_bias_elements_c2_kt{kt}.png")

    # Delta posterior SD: same layout as bias grid.
    for kt in KTRUE:
        save(delta_sd_faceted_by_element(CHAINS, kt), f"delta/sd/plots/delta_sd_elements_c2_kt{kt}.png")

    # Delta absolute error: same layout, y = |post_mean - true_value| per seed.
    for kt in KTRUE:
        save(delta_rmse_faceted_by_element(CHAINS, kt), f"delta/rmse/plots/delta_rmse_elements_c2_kt{kt}.png")

    # Runtime: per sampler by k_true (nuts in hours, hmc/bayesm in minutes, linear).
    for s in SAMPLERS:
        save(runtime_by_ktrue(s, CHAINS), f"runtime/plots/runtime_{s}_c2_by_ktrue.png")

    # Runtime: all samplers in one figure (log scale).
    save(runtime_samplers_by_ktrue(CHAINS), "runtime/plots/runtime_samplers_c2_by_ktrue.png")

    # Beta recovery plots go to beta/ (not yet implemented).

    print("regenerated all figures -> analysis/out/k5_results/{delta/bias,delta/sd,runtime}/plots/")


if __name__ == "__main__":
    main()
