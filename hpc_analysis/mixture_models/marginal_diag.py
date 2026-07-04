"""
Convergence diagnostics (ESS and R-hat) of the marginal-comparison series, mixture_c2.

Source: data/out/mixture_c2/marginal_diagnostics.csv (src.summaries ->
marginal_comparison.{density_series_diagnostics, moment_series_diagnostics}). Every quantity
here is LABEL-INVARIANT, so relabeling never touches it - these are the diagnostics of the
functionals the study actually reports (the marginal density, Eq. 5.5.19, and the mixture
mean/variance, Eq. 5.5.2), not of the permutation-ambiguous component parameters.

The table mixes two row shapes; this module unifies them to one (ESS, Rhat) per row:

  * kind == 'density'  - the per-draw marginal density f(x) is a SERIES over draws evaluated
    at each grid point in the high-density region; ESS/R-hat are computed per point and stored
    as min_ESS/mean_ESS and max_Rhat/mean_Rhat. We take the WORST point by default
    (ESS = min_ESS, Rhat = max_Rhat) - the conservative "did the whole density converge" read.
    density_agg='mean' switches to the average-over-points summary instead. Density rows
    exist once per evaluation-grid scenario (grid == 'full' | 'chebyshev'); load_diag keeps
    ONE grid at a time and every output is produced per grid (filename suffix _<grid>).
  * kind == 'moment_Mean' / 'moment_Var' - a single scalar series per draw (the mixture mean /
    variance), so ESS and Rhat are already single values. These rows are grid-independent
    (grid == 'moments') and are kept regardless of the selected density grid.

Thresholds: R-hat <= 1.1 is the convergence gate used study-wide (label_switching.classify_
outcome). ESS >= 400 is a rule-of-thumb target for a stable functional estimate; both are drawn
as reference lines and reported as pass-rates. Only c2 (2-chain) runs carry these (R-hat needs
>= 2 chains).

Run from the repo root with the project venv:
    .venv/Scripts/python.exe hpc_analysis/marginal_diag.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    element_text,
    facet_grid,
    facet_wrap,
    geom_boxplot,
    geom_hline,
    ggplot,
    labs,
    position_dodge,
    scale_color_manual,
    scale_x_discrete,
    scale_y_log10,
    theme,
    theme_bw,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_recovery import (  # noqa: E402
    DIR_FIG,
    SAMPLER_COLORS,
    SAMPLER_LABELS,
    SAMPLER_ORDER,
    load_recovery,
    save,
)

# --------------------------------------------------------------------------------- #
# Conventions for this analysis.
# --------------------------------------------------------------------------------- #
RHAT_THRESH = 1.1        # study-wide convergence gate (classify_outcome)
ESS_MIN = 400            # rule-of-thumb ESS target for a stable functional estimate
KTRUE_ORDER = [1, 2, 3, 5]
_PARAM_ORDER = ["Alt1", "Alt2", "Alt3", "Price"]

# Series (the invariant functional whose chain is diagnosed) and readable labels.
SERIES_ORDER = ["density", "Mean", "Var"]
SERIES_LABELS = {"density": "Marginal density", "Mean": "Mixture mean", "Var": "Mixture variance"}

DIR_OUT_BASE = DIR_FIG / "marginal_comparison"   # per-grid subfolder added per output
GRIDS = ["full", "chebyshev"]  # density-series evaluation-grid scenarios
GRID_FOLDER = {"full": "full", "chebyshev": "trimmed"}  # output subfolder per grid


# --------------------------------------------------------------------------------- #
# Load + unify the two row shapes to one (ESS, Rhat) per row.
# --------------------------------------------------------------------------------- #
def load_diag(n_chains: int = 2, *, grid: str = "chebyshev", density_agg: str = "worst",
              df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Return one tidy frame with a single ESS and Rhat per row plus a 'series' column
    ('density' | 'Mean' | 'Var'). density_agg: 'worst' (min_ESS/max_Rhat) or 'mean'.
    `grid` selects the density-series evaluation grid ('full' or 'chebyshev'); the
    grid-independent moment rows (grid == 'moments') are always kept."""
    d = load_recovery("marginal_diagnostics") if df is None else df.copy()
    if "grid" in d.columns:
        d = d[d["grid"].isin([grid, "moments"])]
    d = d[d["n_chains"] == n_chains].copy()
    if d.empty:
        raise ValueError(f"No marginal_diagnostics rows for n_chains={n_chains}.")
    is_density = d["kind"] == "density"
    d["series"] = np.where(is_density, "density", d["kind"].str.replace("moment_", "", regex=False))
    ess_src, rhat_src = ("min_ESS", "max_Rhat") if density_agg == "worst" else ("mean_ESS", "mean_Rhat")
    d.loc[is_density, "ESS"] = d.loc[is_density, ess_src]
    d.loc[is_density, "Rhat"] = d.loc[is_density, rhat_src]
    return d.dropna(subset=["ESS", "Rhat"])


def _prep(d: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Apply canonical ordered categoricals (sampler, k_true, series, param)."""
    sampler_order = [s for s in SAMPLER_ORDER if s in set(d["sampler"])]
    series_order = [s for s in SERIES_ORDER if s in set(d["series"])]
    param_order = [p for p in _PARAM_ORDER if p in set(d["param"])]
    ktrue_order = [str(k) for k in KTRUE_ORDER if k in set(d["k_true"])]
    d = d.copy()
    d["sampler"] = pd.Categorical(d["sampler"], categories=sampler_order, ordered=True)
    d["series"] = pd.Categorical(d["series"].map(SERIES_LABELS),
                                 categories=[SERIES_LABELS[s] for s in series_order], ordered=True)
    d["param"] = pd.Categorical(d["param"], categories=param_order, ordered=True)
    d["k_true"] = pd.Categorical(d["k_true"].astype(str), categories=ktrue_order, ordered=True)
    return d, sampler_order


# --------------------------------------------------------------------------------- #
# Plots.
# --------------------------------------------------------------------------------- #
def _diag_grid(d: pd.DataFrame, metric: str, n_chains: int,
               grid: Optional[str] = None) -> ggplot:
    """metric in {'Rhat','ESS'}: x = k_true, dodged box per sampler, facet_grid(series x param).
    `grid` only annotates the title (the frame is already filtered by load_diag)."""
    d, sampler_order = _prep(d)
    counts = d.groupby(["k_true", "sampler"], observed=True)["data_seed"].nunique().to_dict()
    print(f"[marginal_diag:{metric}] c{n_chains}: seeds/box={counts}")
    color_vals = [SAMPLER_COLORS[s] for s in sampler_order]
    dodge = position_dodge(width=0.9)
    gtag = f", {grid} grid" if grid else ""

    p = (
        ggplot(d, aes(x="k_true", y=metric, color="sampler"))
        + geom_boxplot(width=0.6, fill="#FFFFFF00", outlier_alpha=0.25, position=dodge)
    )
    if metric == "Rhat":
        p = p + geom_hline(yintercept=RHAT_THRESH, linetype="dashed", color="#555555", size=0.7)
        ylab, title = ("R-hat  (worst grid point for density)",
                       f"Marginal-Series R-hat by True Components  (c{n_chains}{gtag})")
    else:
        p = (p + scale_y_log10()
             + geom_hline(yintercept=ESS_MIN, linetype="dashed", color="#555555", size=0.7))
        ylab, title = ("ESS  (log scale; min over grid for density)",
                       f"Marginal-Series ESS by True Components  (c{n_chains}{gtag})")
    return (
        p + facet_grid(rows="series", cols="param", scales="free_y", labeller="label_value")
        + scale_color_manual(values=color_vals,
                             labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
        + labs(x="True Number of components", y=ylab, color="Sampler", title=title)
        + theme_bw()
        + theme(figure_size=(13, 8), axis_text_x=element_text(size=8),
                plot_title=element_text(size=11))
    )


def plot_rhat_grid(n_chains: int = 2, df: Optional[pd.DataFrame] = None,
                   grid: str = "chebyshev") -> ggplot:
    """R-hat of all three invariant series (density / mean / variance) x 4 params, by k_true."""
    return _diag_grid(load_diag(n_chains, grid=grid, df=df), "Rhat", n_chains, grid)


def plot_ess_grid(n_chains: int = 2, df: Optional[pd.DataFrame] = None,
                  grid: str = "chebyshev") -> ggplot:
    """ESS (log scale) of all three invariant series x 4 params, by k_true."""
    return _diag_grid(load_diag(n_chains, grid=grid, df=df), "ESS", n_chains, grid)


def _diag_series(d: pd.DataFrame, metric: str, series_key: str, n_chains: int,
                 grid: Optional[str] = None) -> ggplot:
    """Single-series view: facet_wrap by param (1x4), x = k_true, dodged by sampler.

    `series_key` is the RAW series value ('density' | 'Mean' | 'Var'); _prep then maps it to
    its display label."""
    d = d[d["series"] == series_key].copy()
    if d.empty:
        raise ValueError(f"No rows for series={series_key!r}.")
    d, sampler_order = _prep(d)
    color_vals = [SAMPLER_COLORS[s] for s in sampler_order]
    dodge = position_dodge(width=0.9)
    slabel = SERIES_LABELS.get(series_key, series_key)
    gtag = f", {grid} grid" if grid else ""

    p = (
        ggplot(d, aes(x="k_true", y=metric, color="sampler"))
        + geom_boxplot(width=0.55, fill="#FFFFFF00", outlier_alpha=0.3, position=dodge)
    )
    if metric == "Rhat":
        p = p + geom_hline(yintercept=RHAT_THRESH, linetype="dashed", color="#555555", size=0.7)
        ylab, title = "R-hat", f"{slabel} R-hat by True Components  (c{n_chains}{gtag})"
    else:
        p = (p + scale_y_log10()
             + geom_hline(yintercept=ESS_MIN, linetype="dashed", color="#555555", size=0.7))
        ylab, title = "ESS  (log scale)", f"{slabel} ESS by True Components  (c{n_chains}{gtag})"
    return (
        p + facet_wrap("param", ncol=4, scales="free_y", labeller="label_value")
        + scale_color_manual(values=color_vals,
                             labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
        + labs(x="True Number of components", y=ylab, color="Sampler", title=title)
        + theme_bw()
        + theme(figure_size=(13, 4.5), axis_text_x=element_text(size=9),
                plot_title=element_text(size=11))
    )


def plot_density_rhat(n_chains: int = 2, df: Optional[pd.DataFrame] = None,
                      grid: str = "chebyshev") -> ggplot:
    """R-hat of the marginal density (the primary functional, Eq. 5.5.19), by param and k_true."""
    return _diag_series(load_diag(n_chains, grid=grid, df=df), "Rhat", "density", n_chains, grid)


def plot_density_ess(n_chains: int = 2, df: Optional[pd.DataFrame] = None,
                     grid: str = "chebyshev") -> ggplot:
    """ESS of the marginal density (log scale), by param and k_true."""
    return _diag_series(load_diag(n_chains, grid=grid, df=df), "ESS", "density", n_chains, grid)


# --------------------------------------------------------------------------------- #
# Tables.
# --------------------------------------------------------------------------------- #
def rhat_summary_table(n_chains: int = 2, d: Optional[pd.DataFrame] = None,
                       grid: str = "chebyshev") -> pd.DataFrame:
    """Per (k_true, series, param, sampler): median / q75 / max R-hat and the convergence
    pass-rate frac(R-hat <= 1.1). Higher frac_converged = more replicate seeds mixed."""
    d = load_diag(n_chains, grid=grid) if d is None else d
    g = d.groupby(["k_true", "series", "param", "sampler"], observed=True)["Rhat"]
    out = g.agg(median_rhat="median", q75_rhat=lambda s: s.quantile(0.75), max_rhat="max",
                frac_converged=lambda s: (s <= RHAT_THRESH).mean(), n_sim="size").reset_index()
    return _finish(out, ["median_rhat", "q75_rhat", "max_rhat", "frac_converged"])


def ess_summary_table(n_chains: int = 2, d: Optional[pd.DataFrame] = None,
                      grid: str = "chebyshev") -> pd.DataFrame:
    """Per (k_true, series, param, sampler): ESS quartiles and the frac(ESS >= 400) pass-rate."""
    d = load_diag(n_chains, grid=grid) if d is None else d
    g = d.groupby(["k_true", "series", "param", "sampler"], observed=True)["ESS"]
    out = g.agg(min_ess="min", q25_ess=lambda s: s.quantile(0.25), median_ess="median",
                q75_ess=lambda s: s.quantile(0.75),
                frac_ess_ge_400=lambda s: (s >= ESS_MIN).mean(), n_sim="size").reset_index()
    return _finish(out, ["min_ess", "q25_ess", "median_ess", "q75_ess", "frac_ess_ge_400"])


def _finish(out: pd.DataFrame, round_cols: list) -> pd.DataFrame:
    """Canonical sampler/series ordering + rounding for a summary table."""
    sampler_order = [s for s in SAMPLER_ORDER if s in set(out["sampler"])]
    series_order = [s for s in SERIES_ORDER if s in set(out["series"])]
    param_order = [p for p in _PARAM_ORDER if p in set(out["param"])]
    out["sampler"] = pd.Categorical(out["sampler"], categories=sampler_order, ordered=True)
    out["series"] = pd.Categorical(out["series"], categories=series_order, ordered=True)
    out["param"] = pd.Categorical(out["param"], categories=param_order, ordered=True)
    out[round_cols] = out[round_cols].round(3)
    return out.sort_values(["k_true", "series", "param", "sampler"]).reset_index(drop=True)


# --------------------------------------------------------------------------------- #
# Entry points.
# --------------------------------------------------------------------------------- #
def write_tables(n_chains: int = 2) -> None:
    for grid in GRIDS:
        out_dir = DIR_OUT_BASE / GRID_FOLDER[grid] / "tables"
        out_dir.mkdir(parents=True, exist_ok=True)
        d = load_diag(n_chains, grid=grid)
        for name, tbl in {
            f"marginal_rhat_summary_c{n_chains}.csv": rhat_summary_table(n_chains, d),
            f"marginal_ess_summary_c{n_chains}.csv":  ess_summary_table(n_chains, d),
        }.items():
            path = out_dir / name
            tbl.to_csv(path, index=False)
            print(f"wrote {len(tbl)} rows -> {path}")


def make_plots(n_chains: int = 2) -> None:
    for grid in GRIDS:
        plots = f"marginal_comparison/{GRID_FOLDER[grid]}/plots"
        d = load_diag(n_chains, grid=grid)
        print("wrote", save(_diag_grid(d, "Rhat", n_chains, grid),
                            f"{plots}/marginal_rhat_grid_c{n_chains}.png"))
        print("wrote", save(_diag_grid(d, "ESS", n_chains, grid),
                            f"{plots}/marginal_ess_grid_c{n_chains}.png"))
        print("wrote", save(_diag_series(d, "Rhat", "density", n_chains, grid),
                            f"{plots}/marginal_density_rhat_c{n_chains}.png"))
        print("wrote", save(_diag_series(d, "ESS", "density", n_chains, grid),
                            f"{plots}/marginal_density_ess_c{n_chains}.png"))


def main() -> None:
    write_tables()
    make_plots()
    print("marginal-diagnostics tables + figures -> hpc_analysis/mixture_models/out/marginal_comparison/")


if __name__ == "__main__":
    main()
