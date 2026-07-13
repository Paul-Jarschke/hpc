"""
Single-run diagnostic plots, read straight from a run's saved per-run summary CSVs
(the out[-test]/delta_summary/<run_key>.csv and beta_summary/<run_key>.csv that each
fit writes). Where plot_recovery.py pools many runs into cross-sampler boxplots, this
module zooms into ONE run to eyeball whether a single fit recovered its parameters -
the natural "did the per-run save work?" check.

`plot_delta_bias_single_run` plots, for one run, the bias (post_mean - true_value) of
every Delta element with its 95% credible interval drawn relative to truth: the dashed
line at 0 is the true value, so an interval crossing 0 means the true Delta element is
inside the 95% CI (recovered), and the point shows the signed bias and its direction.

Run from the repo root with the project venv:
    .venv/Scripts/python.exe hpc_analysis/plot_single_run.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless: save figures without a display (HPC + headless Windows)

import pandas as pd
from plotnine import (
    aes,
    element_text,
    geom_hline,
    geom_point,
    ggplot,
    labs,
    theme,
    theme_bw,
)

REPO = Path(__file__).resolve().parents[2]
DIR_FIG = Path(__file__).resolve().parent / "out"


def find_summary_csv(run_key: str, kind: str = "delta", testing: Optional[bool] = None,
                     repo: Path = REPO) -> Path:
    """Locate a run's per-run summary CSV: jobs/*/{out|out-test}/<kind>_summary/<run_key>.csv.

    testing=None searches real `out/` first then `out-test/`; True/False forces one.
    Returns the first match (a run_key is unique to one job)."""
    outs = ["out", "out-test"] if testing is None else (["out-test"] if testing else ["out"])
    for out in outs:
        hits = sorted(repo.glob(f"jobs/*/{out}/{kind}_summary/{run_key}.csv"))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"No {kind}_summary CSV for run_key='{run_key}' (searched jobs/*/{{{','.join(outs)}}}/"
        f"{kind}_summary/). Has the run been rendered yet?"
    )


def plot_delta_bias_single_run(run_key: str = "kt1_s01__k5_nuts_c1", *,
                               testing: Optional[bool] = None,
                               save_fig: bool = True) -> ggplot:
    """Plot per-element Delta bias for ONE run.

    Reads that run's saved delta_summary/<run_key>.csv (8 rows = D*P elements) and plots
    bias = post_mean - true_value for each Delta element, one point per element. The
    dashed line at 0 marks zero bias (the element's posterior mean equals truth); points
    above it overestimate the true value, points below underestimate it. No credible-
    interval information is shown - just the bias per parameter.
    """
    csv = find_summary_csv(run_key, "delta", testing)
    df = pd.read_csv(csv)
    if df.empty:
        raise ValueError(f"{csv} has no rows (no Delta / no demographics for this run?).")

    # x-axis: the D*P Delta elements, kept in their saved (demo-major, param-minor) order.
    df["element"] = df["demo"].astype(str) + " : " + df["param"].astype(str)
    df["element"] = pd.Categorical(df["element"], categories=df["element"].tolist(), ordered=True)

    r0 = df.iloc[0]
    title = (f"Δ bias per element - {r0['sampler']}, {r0['scenario']} "
             f"(k_true={r0['k_true']}, seed={r0['data_seed']}, n_chains={r0['n_chains']})\n"
             f"run_key: {run_key}")

    p = (
        ggplot(df, aes("element", "bias"))
        + geom_hline(yintercept=0, linetype="dashed", color="#7f7f7f")
        + geom_point(color="#1f77b4", size=3.0)
        + labs(x="Δ element",
               y="Bias (Δ̂−Δ)",
               title=title)
        + theme_bw()
        + theme(figure_size=(8.5, 5.0),
                axis_text_x=element_text(rotation=30),
                plot_title=element_text(size=10))
    )

    if save_fig:
        DIR_FIG.mkdir(parents=True, exist_ok=True)
        out = DIR_FIG / f"delta_bias_{run_key}.png"
        p.save(out, dpi=150, verbose=False)
        print(f"read   {csv.relative_to(REPO)}")
        print(f"wrote  {out.relative_to(REPO)}")
    return p


def main() -> None:
    # The requested check: NUTS, 1-component dataset (k_true=1), seed 1 -> dataset kt1_s01.
    plot_delta_bias_single_run("kt1_s01__k5_nuts_c1")


if __name__ == "__main__":
    main()
