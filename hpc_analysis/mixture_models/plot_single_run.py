# One-run sanity plots straight from a run's saved summary CSVs
# (out[-test]/delta_summary/<run_key>.csv). plot_recovery.py pools
# many runs; this zooms into ONE fit - "did the per-run save work?".
# run: .venv/Scripts/python.exe hpc_analysis/mixture_models/plot_single_run.py
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


# testing=None tries out/ then out-test/; True/False forces one.
# first match wins - a run_key is unique to one job.
def find_summary_csv(run_key: str, kind: str = "delta", testing: Optional[bool] = None,
                     repo: Path = REPO) -> Path:
    outs = ["out", "out-test"] if testing is None else (["out-test"] if testing else ["out"])
    for out in outs:
        hits = sorted(repo.glob(f"jobs/*/{out}/{kind}_summary/{run_key}.csv"))
        if hits:
            return hits[0]
    raise FileNotFoundError(
        f"No {kind}_summary CSV for run_key='{run_key}' (searched jobs/*/{{{','.join(outs)}}}/"
        f"{kind}_summary/). Has the run been rendered yet?"
    )


# bias = post_mean - true_value per Delta element (D*P rows).
# dashed 0 = zero bias; above over-, below under-estimates. no CIs.
def plot_delta_bias_single_run(run_key: str = "kt1_s01__k5_nuts_c1", *,
                               testing: Optional[bool] = None,
                               save_fig: bool = True) -> ggplot:
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
