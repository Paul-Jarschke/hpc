"""
Regenerate the full k5model_mixture figure set from data/out/k5model_mixture/*.csv.

Run after scripts/gather_summaries.py whenever new runs are gathered. Writes PNGs to
analysis/out/. All figures cover the 2-chain (c2) arm, where nuts/hmc/bayesm are complete.

    .venv/Scripts/python.exe analysis/make_plots.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_recovery import (  # noqa: E402
    load_recovery,
    recovery_boxplot,
    save,
    delta_bias_across_seeds,
    delta_bias_samplers_by_element,
    runtime_by_ktrue,
    runtime_samplers_by_ktrue,
)

CHAINS = 2
KTRUE = [1, 2, 3, 5]
SAMPLERS = ["nuts", "hmc", "bayesm"]


def main():
    # Delta bias: all three samplers side by side, per element - one figure per scenario.
    for kt in KTRUE:
        save(delta_bias_samplers_by_element(CHAINS, kt), f"delta_bias_samplers_c2_kt{kt}.png")

    # Delta bias: single sampler, per element, across all seeds (kt1).
    for s in SAMPLERS:
        save(delta_bias_across_seeds(s, CHAINS, 1), f"delta_bias_{s}_c2_kt1.png")

    # Runtime: per sampler by k_true (nuts in hours, hmc/bayesm in minutes, linear).
    for s in SAMPLERS:
        save(runtime_by_ktrue(s, CHAINS), f"runtime_{s}_c2_by_ktrue.png")

    # Runtime: all samplers in one figure (log scale).
    save(runtime_samplers_by_ktrue(CHAINS), "runtime_samplers_c2_by_ktrue.png")

    # Cross-sampler delta bias, faceted by k_true.
    df = load_recovery("delta")
    df["element"] = df["demo"].astype(str) + ":" + df["param"].astype(str)
    p = recovery_boxplot(
        df, value="bias", x="sampler", filters={"n_chains": CHAINS},
        facet_wrap_by="k_true", facet_scales="fixed", hline=0.0, jitter=False,
        title="Δ bias by sampler (2 chains), per true-component count",
        ylab="Bias (post_mean - true_value)", xlab="Sampler", figure_size=(8.5, 6.0),
    )
    save(p, "delta_bias_c2_by_sampler.png")

    print("regenerated all figures -> analysis/out/")


if __name__ == "__main__":
    main()
