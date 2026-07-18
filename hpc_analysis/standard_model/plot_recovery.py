"""
Reusable parameter-recovery plotting for the standard_model sampler comparison (jobs 200-202).

The gather step (scripts/gather_summaries.py --glob "jobs/20[0-2]-standard-*"
--out-name standard_model, or hpc_analysis/standard_model/post_process.py) writes tidy
per-element recovery tables to data/out/standard_model/{delta,mu,sigma,beta}_recovery.csv.
Every row is ONE estimated element of ONE run, carrying all condition columns
(sampler, n_chains, scenario, k_true, k_model, data_seed, param, ...). The standard
model is the SINGLE-normal-component HBMNL (Rossi section 5.4): k_true == k_model == 1
throughout, so there is no k_true dimension to facet or loop over - every mixture-side
"per k_true" plot collapses to one overall figure here.

This module exposes ONE general boxplot core, `recovery_boxplot`, that draws a
boxplot of any chosen value column grouped on an x-axis (default: sampler), with
optional faceting and row filtering. On top of the Delta/beta plots ported from the
mixture pipeline it adds the two standard-model-specific analyses: mu recovery
(population mean vs TRUE_MU) and POSTERIOR SIGMA recovery (every lower-triangle
element vs TRUE_SIGMA, with the empirical covariance of the true unit betas as a
reference in the tables).

plotnine is used deliberately (see module-level note in the docstring of
`recovery_boxplot`): the study already uses plotnine, and its grammar-of-graphics
API gives a single faceting/aesthetic vocabulary that scales to "much more of those
kinds of plots" without bespoke matplotlib axis wiring per plot.

Run from the repo root with the project venv:
    .venv/Scripts/python.exe hpc_analysis/standard_model/plot_recovery.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import matplotlib
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # non-interactive: save figures without a display/Tk (HPC + headless Windows)

import pandas as pd
from plotnine import (
    aes,
    element_text,
    facet_grid,
    facet_wrap,
    coord_cartesian,
    geom_boxplot,
    geom_col,
    geom_hline,
    geom_jitter,
    ggplot,
    labs,
    scale_color_manual,
    scale_fill_manual,
    scale_x_discrete,
    scale_y_continuous,
    scale_y_log10,
    theme,
    theme_bw,
)

# ----------------------------------------------------------------------------- #
# Locations & study-wide conventions (single source of truth for every plot).
# NOTE: DIR_RECOVERY / DIR_FIG are resolved at CALL time (defaults are None in the
# functions below), so tests can monkeypatch plot_recovery.DIR_RECOVERY / DIR_FIG.
# ----------------------------------------------------------------------------- #
REPO = Path(__file__).resolve().parents[2]
DIR_RECOVERY = REPO / "data" / "out" / "standard_model"
DIR_FIG = Path(__file__).resolve().parent / "out"

# Fixed sampler order/labels/colors so EVERY figure in the study is consistent.
# Same palette entries as the mixture pipeline for the three samplers the standard
# family runs (no bayesm_gibbs replication arm here): bayesm = red, nuts/hmc = two
# shades of blue (matching the marginal-density reference plot).
SAMPLER_ORDER = ["bayesm", "nuts", "hmc"]
SAMPLER_LABELS = {"bayesm": "bayesm", "nuts": "NUTS", "hmc": "HMC"}
SAMPLER_COLORS = {"nuts": "#08519c", "hmc": "#4292c6", "iwls": "#9ecae1", "bayesm": "#d62728"}
TRUE_COLOR = "#000000"

# Element label helpers (single source of truth for the Δ_{d,p} and Σ_{i,j} notation).
_DEMO_ORDER = ["z1", "z2"]
_PARAM_ORDER = ["Alt1", "Alt2", "Alt3", "Price"]
_SUBSCRIPTS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def delta_element_label(demo: str, param: str) -> str:
    """Return 'Δ₁,₁ (z1:Alt1)' style label for one Delta matrix element.

    Indices are 1-based row (demo) and column (param) positions in the D x P matrix.
    Falls back to '?' for names not in the known orders so new datasets don't crash.
    """
    d = (_DEMO_ORDER.index(demo) + 1) if demo in _DEMO_ORDER else "?"
    p = (_PARAM_ORDER.index(param) + 1) if param in _PARAM_ORDER else "?"
    return f"Δ{str(d).translate(_SUBSCRIPTS)},{str(p).translate(_SUBSCRIPTS)} ({demo}:{param})"


def sigma_element_label(row: str, col: str) -> str:
    """Return 'Σ₁,₁ (Alt1:Alt1)' style label for one lower-triangle Sigma element.

    Indices are 1-based row/col positions in the P x P covariance matrix.
    Falls back to '?' for names not in the known parameter order.
    """
    i = (_PARAM_ORDER.index(row) + 1) if row in _PARAM_ORDER else "?"
    j = (_PARAM_ORDER.index(col) + 1) if col in _PARAM_ORDER else "?"
    return f"Σ{str(i).translate(_SUBSCRIPTS)},{str(j).translate(_SUBSCRIPTS)} ({row}:{col})"


# Crisp axis labels for the value columns we commonly plot. theta (θ) stands in for
# whichever generic column (Delta/mu/Sigma/beta) is loaded; theta-hat (θ̂) = post_mean.
VALUE_LABELS = {
    "bias": "Bias (θ̂−θ)",
    "abs_diff": "|θ̂−θ|",
    "post_std": "SD(θ̂)",
    "post_mean": "Mean(θ̂)",
    "rmse": "RMSE",
    "mean_abs_err": "MAE",
    "coverage95": "Coverage (%)",
    "runtime_s": "Runtime (s)",
    "n_divergent": "Divergences",
}

# Tables keyed by short name; what `load_recovery` reads. The first five are per-element
# recovery tables; the rest are per-run / per-kernel tables for direct-column boxplots.
# No ecr_report / weights / pvec_means here: the standard model has K = 1, so there is
# no label switching and no mixture-weight bookkeeping.
RECOVERY_FILES = {
    "delta": "delta_recovery.csv",
    "mu": "mu_recovery.csv",
    "sigma": "sigma_recovery.csv",
    "beta": "beta_recovery.csv",
    "beta_summary": "beta_summary.csv",
    "runs": "runs.csv",
    "convergence": "convergence.csv",
    "moments": "moments.csv",
    "diagnostics": "diagnostics.csv",
    "marginal_distances": "marginal_distances.csv",
    "marginal_diagnostics": "marginal_diagnostics.csv",
}

MARGINAL_METRICS = ["Hellinger", "KL", "JSD", "TVD"]
MARGINAL_METRIC_LABELS = {
    "Hellinger": "Hellinger",
    "KL": "KL divergence",
    "JSD": "JS divergence",
    "TVD": "TVD",
}


# ----------------------------------------------------------------------------- #
# Data loading + the only derived column the schema does not already store.
# ----------------------------------------------------------------------------- #
def load_recovery(table: str, dir_recovery: Optional[Path] = None) -> pd.DataFrame:
    """Load a recovery table by short name ('delta'|'mu'|'sigma'|'beta'|...) and add
    a signed `bias` column where the table stores post_mean + true_value.

    delta/mu/sigma carry post_mean & true_value -> bias = post_mean - true_value
    (the per-element signed deviation). beta_recovery is already aggregated and
    ships its own `bias`/`rmse`/`coverage95`; it is returned unchanged.
    """
    if dir_recovery is None:
        dir_recovery = DIR_RECOVERY
    df = pd.read_csv(dir_recovery / RECOVERY_FILES[table])
    if "bias" not in df.columns and {"post_mean", "true_value"}.issubset(df.columns):
        df["bias"] = df["post_mean"] - df["true_value"]
    return df


def _apply_filters(df: pd.DataFrame, filters: Optional[dict]) -> pd.DataFrame:
    """Keep rows matching every {column: value|list-of-values} in `filters`."""
    if not filters:
        return df
    mask = pd.Series(True, index=df.index)
    for col, val in filters.items():
        wanted = val if isinstance(val, (list, tuple, set)) else [val]
        mask &= df[col].isin(list(wanted))
    return df.loc[mask]


def paired_complete(df: pd.DataFrame, key=("dataset_key", "demo", "param"), by="sampler"):
    """Keep only key-cells covered by EVERY `by` group, so each box compares the SAME
    datasets/elements across samplers (the paired design). Returns (kept_df, report)."""
    key = [k for k in key if k in df.columns]
    nby = df[by].nunique()
    keep = df.groupby(key)[by].transform("nunique") == nby
    return df[keep], {"samplers": nby, "kept_rows": int(keep.sum()), "dropped_rows": int((~keep).sum())}


def _maybe_paired(df: pd.DataFrame, paired: bool, tag: str) -> pd.DataFrame:
    """Apply paired_complete, but fall back (loudly) to unpaired data if it would empty
    the frame - e.g. the sparse local TEST data where samplers cover different datasets.
    On the real grid every sampler covers every dataset, so nothing is dropped."""
    if not paired:
        return df
    pc, rep = paired_complete(df)
    if len(pc):
        print(f"[{tag}] paired: kept {rep['kept_rows']} rows across {rep['samplers']} samplers "
              f"(dropped {rep['dropped_rows']} unpaired)")
        return pc
    print(f"[{tag}] WARNING: no (dataset,element) cell is covered by all samplers - sparse/test "
          f"data; falling back to UNPAIRED (boxes are NOT like-for-like across samplers).")
    return df


def _print_box_counts(df: pd.DataFrame, facet: Optional[str] = None, by: str = "sampler"):
    """Print n (element-rows) per box so the figure's sample sizes are explicit."""
    cols = [c for c in (facet, by) if c is not None and c in df.columns]
    if cols:
        tab = df.groupby(cols).size().rename("n_rows").reset_index()
        print("box counts (n element-rows):\n" + tab.to_string(index=False))


# ----------------------------------------------------------------------------- #
# THE reusable core. Every recovery boxplot in the study is one call to this.
# ----------------------------------------------------------------------------- #
def recovery_boxplot(
    df: pd.DataFrame,
    value: str = "bias",
    x: str = "sampler",
    *,
    filters: Optional[dict] = None,
    facet_row: Optional[str] = None,
    facet_col: Optional[str] = None,
    facet_wrap_by: Optional[Union[str, Sequence[str]]] = None,
    facet_scales: str = "fixed",
    color: Optional[str] = None,
    hline: Optional[float] = None,
    x_order: Optional[Sequence[str]] = None,
    x_labels: Optional[dict] = None,
    title: Optional[str] = None,
    xlab: Optional[str] = None,
    ylab: Optional[str] = None,
    jitter: bool = False,
    logy: bool = False,
    figure_size: tuple = (7.0, 4.5),
) -> ggplot:
    """Boxplot of `value` grouped by `x`, over a tidy recovery table.

    This is the single general entry point: pick the value column to summarize
    (bias / post_std / abs_diff / any numeric column present), the grouping for
    the x-axis (default the sampler), optional facets, and optional row filters.
    Each box aggregates every element-row that survives `filters` for that x
    group/facet cell (e.g. all D*P Delta elements * all datasets for a sampler).

    Why plotnine: the study already standardizes on it, and the grammar-of-
    graphics call signature means new variants (different value, facet by
    scenario/param, color by n_chains, etc.) are argument changes rather than
    new plotting code -- exactly what "much more of those kinds of plots"
    needs. Returns the ggplot so callers can further adjust or `.save()` it.

    Parameters
    ----------
    df : tidy recovery frame (from `load_recovery`).
    value : y-axis column to summarize (default 'bias'; 'post_std' for the SD plot).
    x : grouping column for the x-axis (default 'sampler').
    filters : {col: value | [values]} row filter, e.g. {'n_chains': 2}.
    facet_row, facet_col : columns for a facet_grid (rows x cols).
    facet_wrap_by : column(s) for facet_wrap (use instead of grid for one dim).
    facet_scales : 'fixed' | 'free' | 'free_x' | 'free_y' (passed to the facet).
    color : optional outline grouping (e.g. 'n_chains') drawn as dodged boxes.
    hline : optional horizontal reference line (e.g. 0.0 for a bias plot).
    x_order : category order for x; defaults to SAMPLER_ORDER when x == 'sampler'.
    x_labels : x tick relabeling; defaults to SAMPLER_LABELS when x == 'sampler'.
    title, xlab, ylab : labels (sensible defaults derived from value/x).
    figure_size : (width, height) inches.
    """
    data = _apply_filters(df, filters).copy()

    # Stable, study-consistent ordering/labels for the sampler axis (or any x).
    if x_order is None and x == "sampler":
        x_order = [s for s in SAMPLER_ORDER if s in set(data[x])]
    if x_order is not None:
        data[x] = pd.Categorical(data[x], categories=list(x_order), ordered=True)
    if x_labels is None and x == "sampler":
        x_labels = SAMPLER_LABELS

    mapping = aes(x=x, y=value)
    if color is not None:
        data[color] = data[color].astype(str)
        mapping = aes(x=x, y=value, color=color)
    elif x == "sampler":
        # Outline boxes by sampler when sampler is on the x-axis (no explicit grouping).
        # House style: boxes are NEVER filled - colored outline + transparent body.
        mapping = aes(x=x, y=value, color=x)

    p = ggplot(data, mapping)
    if hline is not None:
        p = p + geom_hline(yintercept=hline, linetype="dashed", color="#7f7f7f")
    p = p + geom_boxplot(fill="#FFFFFF00", outlier_size=0.6, outlier_alpha=(0.0 if jitter else 0.4))
    if jitter:  # show the raw element-points (recommended at modest n per box)
        if color is not None or x == "sampler":
            # House style: points inherit the color aesthetic, matching their box outline.
            p = p + geom_jitter(width=0.18, height=0.0, size=0.8, alpha=0.5)
        else:
            p = p + geom_jitter(width=0.18, height=0.0, size=0.8, alpha=0.5, color="#444444")
    if logy:  # runtime / ESS etc. span orders of magnitude
        p = p + scale_y_log10()

    # Apply the study-wide sampler color palette whenever sampler drives fill.
    if x == "sampler" or color == "sampler":
        s_order = (
            x_order  # already resolved above for x == "sampler"
            if x == "sampler"
            else [s for s in SAMPLER_ORDER if s in set(data["sampler"])]
        )
        s_vals = [SAMPLER_COLORS.get(str(s), "#888888") for s in s_order]
        s_labs = [SAMPLER_LABELS.get(str(s), str(s)) for s in s_order]
        p = p + scale_color_manual(values=s_vals, breaks=list(s_order), labels=s_labs)

    # Faceting: grid takes priority if either dim given, else optional wrap.
    if facet_row is not None or facet_col is not None:
        p = p + facet_grid(
            rows=facet_row, cols=facet_col, scales=facet_scales, labeller="label_both"
        )
    elif facet_wrap_by is not None:
        p = p + facet_wrap(facet_wrap_by, scales=facet_scales, labeller="label_both")

    p = p + labs(
        x=xlab if xlab is not None else x,
        y=ylab if ylab is not None else VALUE_LABELS.get(value, value),
        title=title,
    )
    if x_labels is not None:
        # Relabel x ticks via the categorical's display labels.
        cats = list(data[x].cat.categories) if hasattr(data[x], "cat") else x_order
        if cats is not None:
            p = p + scale_x_discrete(labels=[x_labels.get(c, c) for c in cats])

    p = p + theme_bw() + theme(
        figure_size=figure_size,
        axis_text_x=element_text(rotation=0),
        plot_title=element_text(size=11),
    )
    return p


def save(plot, filename: str, dir_fig: Optional[Path] = None, dpi: int = 150) -> Path:
    """Save a ggplot or matplotlib Figure under dir_fig (created if missing). Returns the path."""
    if dir_fig is None:
        dir_fig = DIR_FIG
    out = dir_fig / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(plot, matplotlib.figure.Figure):
        plot.savefig(str(out), dpi=dpi, bbox_inches="tight")
        plt.close(plot)
    else:
        plot.save(out, dpi=dpi, verbose=False)
    return out


# ----------------------------------------------------------------------------- #
# Delta recovery (same schema as the mixture pipeline; single k_true == 1 here,
# so the mixture's per-k_true loops collapse to one overall figure each).
# ----------------------------------------------------------------------------- #
def delta_bias_faceted_by_element(n_chains: int = 2, df=None, *, jitter: bool = True) -> ggplot:
    """Delta bias with one panel per Δ element in a 4x2 grid, free y-scale per panel.

    Each element gets its own y-axis so outliers in one element do not compress others.
    Boxplots are transparent (outline only); jittered points are colored by sampler.
    One combined sampler legend on the right.
    """
    df = load_recovery("delta") if df is None else df
    df = df.copy()
    df["element"] = df.apply(lambda r: delta_element_label(r["demo"], r["param"]), axis=1)
    sub = _apply_filters(df, {"n_chains": n_chains}).copy()
    if sub.empty:
        raise ValueError(f"No delta_recovery rows for n_chains={n_chains}.")

    sampler_order = [s for s in SAMPLER_ORDER if s in set(sub["sampler"])]
    sub["sampler"] = pd.Categorical(sub["sampler"], categories=sampler_order, ordered=True)
    element_order = (
        sub.sort_values(["demo", "param"])["element"].drop_duplicates().tolist()
    )
    sub["element"] = pd.Categorical(sub["element"], categories=element_order, ordered=True)

    counts = sub.groupby("sampler", observed=True)["data_seed"].nunique().to_dict()
    print(f"[delta_bias_faceted_by_element] n_chains={n_chains}: seeds/sampler={counts}")

    color_vals = [SAMPLER_COLORS[s] for s in sampler_order]

    # Draw jitter first (behind box outline), then transparent box on top so both are visible.
    p = ggplot(sub, aes(x="sampler", y="bias", color="sampler"))
    p = p + geom_hline(yintercept=0, linetype="dashed", color="#aaaaaa")
    if jitter:
        p = p + geom_jitter(width=0.2, height=0, size=0.8, alpha=0.45)
    p = (p
         + geom_boxplot(fill="#FFFFFF00", outlier_alpha=0)
         + facet_wrap("element", ncol=4, scales="free_y", labeller="label_value")
         + scale_color_manual(values=color_vals, labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
         + scale_x_discrete(labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
         + labs(x="Sampler", y="Bias (Δ̂−Δ)", color="Sampler",
                title="Bias of Δ - Standard Model")
         + theme_bw()
         + theme(figure_size=(14, 7), axis_text_x=element_text(size=8),
                 plot_title=element_text(size=11))
    )
    return p


def delta_sd_faceted_by_element(n_chains: int = 2, df=None, *, jitter: bool = True) -> ggplot:
    """Posterior SD of Delta with one panel per element in a 4x2 grid, free y-scale per panel.

    Same layout as delta_bias_faceted_by_element: transparent boxes, jitter colored by sampler,
    one legend on the right. y-axis shows the posterior standard deviation (post_std), which
    reflects how precisely each sampler pins down the corresponding Delta element.
    """
    df = load_recovery("delta") if df is None else df
    df = df.copy()
    df["element"] = df.apply(lambda r: delta_element_label(r["demo"], r["param"]), axis=1)
    sub = _apply_filters(df, {"n_chains": n_chains}).copy()
    if sub.empty:
        raise ValueError(f"No delta_recovery rows for n_chains={n_chains}.")

    sampler_order = [s for s in SAMPLER_ORDER if s in set(sub["sampler"])]
    sub["sampler"] = pd.Categorical(sub["sampler"], categories=sampler_order, ordered=True)
    element_order = (
        sub.sort_values(["demo", "param"])["element"].drop_duplicates().tolist()
    )
    sub["element"] = pd.Categorical(sub["element"], categories=element_order, ordered=True)

    counts = sub.groupby("sampler", observed=True)["data_seed"].nunique().to_dict()
    print(f"[delta_sd_faceted_by_element] n_chains={n_chains}: seeds/sampler={counts}")

    color_vals = [SAMPLER_COLORS[s] for s in sampler_order]

    p = ggplot(sub, aes(x="sampler", y="post_std", color="sampler"))
    if jitter:
        p = p + geom_jitter(width=0.2, height=0, size=0.8, alpha=0.45)
    p = (p
         + geom_boxplot(fill="#FFFFFF00", outlier_alpha=0)
         + facet_wrap("element", ncol=4, scales="free_y", labeller="label_value")
         + scale_color_manual(values=color_vals, labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
         + scale_x_discrete(labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
         + labs(x="Sampler", y="SD(Δ̂)", color="Sampler",
                title="SD of Δ - Standard Model")
         + theme_bw()
         + theme(figure_size=(14, 7), axis_text_x=element_text(size=8),
                 plot_title=element_text(size=11))
    )
    return p


def delta_rmse_faceted_by_element(n_chains: int = 2, df=None) -> ggplot:
    """RMSE of each Delta element across the replicate seeds, per sampler, in a 4x2 grid.

    RMSE = sqrt(mean over seeds of (post_mean - true_value)^2). Delta is a population
    parameter with a SINGLE estimate per run, so - unlike beta (RMSE over the 300 units,
    which yields a per-run distribution) - the mean square is taken across the replicate
    datasets: the Monte-Carlo RMSE of the estimator. That is one number per (element,
    sampler), so this is a bar chart (one bar per sampler per element), not a boxplot.
    Free y-scale per panel so small-RMSE elements are not compressed by larger ones.
    """
    df = load_recovery("delta") if df is None else df
    df = df.copy()
    df["element"] = df.apply(lambda r: delta_element_label(r["demo"], r["param"]), axis=1)
    sub = _apply_filters(df, {"n_chains": n_chains}).copy()
    if sub.empty:
        raise ValueError(f"No delta_recovery rows for n_chains={n_chains}.")

    # Collapse each (element, sampler) cell to its Monte-Carlo RMSE over the seeds.
    rmse = (
        sub.groupby(["element", "demo", "param", "sampler"], observed=True)["bias"]
        .apply(lambda e: float(np.sqrt((e ** 2).mean())))
        .reset_index()
        .rename(columns={"bias": "rmse"})
    )

    sampler_order = [s for s in SAMPLER_ORDER if s in set(rmse["sampler"])]
    rmse["sampler"] = pd.Categorical(rmse["sampler"], categories=sampler_order, ordered=True)
    element_order = (
        rmse.sort_values(["demo", "param"])["element"].drop_duplicates().tolist()
    )
    rmse["element"] = pd.Categorical(rmse["element"], categories=element_order, ordered=True)

    counts = sub.groupby("sampler", observed=True)["data_seed"].nunique().to_dict()
    print(f"[delta_rmse_faceted_by_element] n_chains={n_chains}: seeds/sampler={counts}")

    fill_vals = [SAMPLER_COLORS[s] for s in sampler_order]

    p = (
        ggplot(rmse, aes(x="sampler", y="rmse", fill="sampler"))
        + geom_col(width=0.6)
        + facet_wrap("element", ncol=4, scales="free_y", labeller="label_value")
        + scale_fill_manual(values=fill_vals,
                            labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
        + scale_x_discrete(labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
        + labs(x="Sampler", y="RMSE (across seeds)", fill="Sampler",
               title="RMSE of Δ - Standard Model")
        + theme_bw()
        + theme(figure_size=(14, 7), axis_text_x=element_text(size=8),
                plot_title=element_text(size=11))
    )
    return p


def delta_coverage_faceted_by_element(n_chains: int = 2, df=None) -> ggplot:
    """Empirical 95% CI coverage rate for each Delta element, per sampler.

    Coverage = (number of seeds where true_value falls in the 95% credible interval)
               / n_sim * 100.  One bar per sampler per element; dashed reference at 95%.
    A well-calibrated sampler should land near 95% for every element.
    """
    df = load_recovery("delta") if df is None else df
    df = df.copy()
    df["element"] = df.apply(lambda r: delta_element_label(r["demo"], r["param"]), axis=1)
    df["in_ci"] = df["in_ci"].astype(bool)
    sub = _apply_filters(df, {"n_chains": n_chains}).copy()
    if sub.empty:
        raise ValueError(f"No delta_recovery rows for n_chains={n_chains}.")

    # Aggregate to one coverage value per (element, sampler).
    cov = (
        sub.groupby(["element", "demo", "param", "sampler"], observed=True)["in_ci"]
        .mean()
        .mul(100)
        .reset_index()
        .rename(columns={"in_ci": "coverage_pct"})
    )

    sampler_order = [s for s in SAMPLER_ORDER if s in set(cov["sampler"])]
    cov["sampler"] = pd.Categorical(cov["sampler"], categories=sampler_order, ordered=True)
    element_order = (
        cov.sort_values(["demo", "param"])["element"].drop_duplicates().tolist()
    )
    cov["element"] = pd.Categorical(cov["element"], categories=element_order, ordered=True)

    fill_vals = [SAMPLER_COLORS[s] for s in sampler_order]

    p = (
        ggplot(cov, aes(x="sampler", y="coverage_pct", fill="sampler"))
        + geom_col(width=0.6)
        + geom_hline(yintercept=95, linetype="dashed", color="#555555", size=0.8)
        + facet_wrap("element", ncol=4, labeller="label_value")
        + scale_fill_manual(values=fill_vals,
                            labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
        + scale_x_discrete(labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
        + scale_y_continuous(breaks=[80, 85, 90, 95, 100])
        + coord_cartesian(ylim=(80, 100))
        + labs(x="Sampler", y="Coverage (%)", fill="Sampler",
               title="95% CI Coverage of Δ - Standard Model")
        + theme_bw()
        + theme(figure_size=(14, 7), axis_text_x=element_text(size=8),
                plot_title=element_text(size=11))
    )
    return p


# ----------------------------------------------------------------------------- #
# Shared per-parameter boxplot core (beta AND mu use the same 1x4 layout).
# ----------------------------------------------------------------------------- #
def _param_boxplot(df: pd.DataFrame, y_col: str, y_label: str,
                   title: str, sampler_order: list, *, jitter: bool,
                   hline: Optional[float] = None) -> ggplot:
    """Shared core for per-param boxplots (beta bias / rmse / mae and mu bias).

    `hline` draws a dashed reference line at that y value (e.g. 0.0 for a bias plot),
    added behind the jitter/box so both stay visible.
    """
    param_order = [p for p in _PARAM_ORDER if p in set(df["param"])]
    df = df.copy()
    df["param"] = pd.Categorical(df["param"], categories=param_order, ordered=True)
    color_vals = [SAMPLER_COLORS[s] for s in sampler_order]

    p = ggplot(df, aes(x="sampler", y=y_col, color="sampler"))
    if hline is not None:
        p = p + geom_hline(yintercept=hline, linetype="dashed", color="#aaaaaa")
    if jitter:
        p = p + geom_jitter(width=0.2, height=0, size=0.8, alpha=0.45)
    p = (p
         + geom_boxplot(fill="#FFFFFF00", outlier_alpha=0)
         + facet_wrap("param", ncol=4, scales="free_y", labeller="label_value")
         + scale_color_manual(values=color_vals,
                              labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
         + scale_x_discrete(labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
         + labs(x="Sampler", y=y_label, color="Sampler", title=title)
         + theme_bw()
         + theme(figure_size=(12, 5), axis_text_x=element_text(size=8),
                 plot_title=element_text(size=11))
    )
    return p


# ----------------------------------------------------------------------------- #
# Beta recovery (unit-level coefficients; schema identical to the mixture pipeline).
# ----------------------------------------------------------------------------- #
def compute_beta_correlation(df_summary=None) -> pd.DataFrame:
    """Compute per-run per-param Pearson correlation between post_mean and true_value.

    Loads beta_summary.csv (large) and returns one row per
    (dataset_key, scenario, k_true, data_seed, k_model, sampler, n_chains, param)
    with a `correlation` column: corr_i(post_mean_i, true_value_i) across the units.
    """
    if df_summary is None:
        df_summary = load_recovery("beta_summary")
    group_cols = [
        "dataset_key", "scenario", "k_true", "data_seed",
        "k_model", "sampler", "n_chains", "param",
    ]
    corr = (
        df_summary.groupby(group_cols, observed=True)
        .apply(lambda g: g["post_mean"].corr(g["true_value"]), include_groups=False)
        .reset_index()
        .rename(columns={0: "correlation"})
    )
    return corr


def compute_beta_post_std(df_summary=None) -> pd.DataFrame:
    """Mean posterior SD of the unit-level coefficients beta_i, per run and parameter.

    The aggregated beta_recovery table drops the posterior SD (it keeps only bias / rmse /
    coverage), so - like compute_beta_correlation - this reads beta_summary.csv (large,
    one row per unit x param) and averages the per-unit `post_std` over the N decision
    units. Returns one row per
    (dataset_key, scenario, k_true, data_seed, k_model, sampler, n_chains, param)
    with a `mean_post_std` column: how tightly, on average, the sampler pins down an
    individual's coefficient for that parameter.
    """
    if df_summary is None:
        df_summary = load_recovery("beta_summary")
    group_cols = [
        "dataset_key", "scenario", "k_true", "data_seed",
        "k_model", "sampler", "n_chains", "param",
    ]
    out = (
        df_summary.groupby(group_cols, observed=True)["post_std"]
        .mean()
        .reset_index()
        .rename(columns={"post_std": "mean_post_std"})
    )
    return out


def beta_bias_by_param(n_chains: int = 2, df=None, *, jitter: bool = True) -> ggplot:
    """Bias (mean post_mean - true_value over all units) per beta parameter.

    One boxplot per (sampler, param), distribution over replicate seeds.
    Faceted by parameter (Alt1/Alt2/Alt3/Price) in a single row of 4 panels.
    """
    df = load_recovery("beta") if df is None else df
    sub = _apply_filters(df, {"n_chains": n_chains}).copy()
    if sub.empty:
        raise ValueError(f"No beta_recovery rows for n_chains={n_chains}.")
    sampler_order = [s for s in SAMPLER_ORDER if s in set(sub["sampler"])]
    sub["sampler"] = pd.Categorical(sub["sampler"], categories=sampler_order, ordered=True)
    counts = sub.groupby("sampler", observed=True)["data_seed"].nunique().to_dict()
    print(f"[beta_bias_by_param] n_chains={n_chains}: seeds/sampler={counts}")
    return _param_boxplot(sub, "bias", "Bias (β̂−β)",
                          "Bias of β - Standard Model", sampler_order,
                          jitter=jitter, hline=0.0)


def beta_rmse_by_param(n_chains: int = 2, df=None, *, jitter: bool = True) -> ggplot:
    """RMSE of beta_i posteriors per parameter (aggregated over the decision units).

    RMSE = sqrt(mean((post_mean_i - true_i)^2)) across units, one value per run.
    Distribution over replicate seeds shown as a boxplot per sampler per parameter.
    """
    df = load_recovery("beta") if df is None else df
    sub = _apply_filters(df, {"n_chains": n_chains}).copy()
    if sub.empty:
        raise ValueError(f"No beta_recovery rows for n_chains={n_chains}.")
    sampler_order = [s for s in SAMPLER_ORDER if s in set(sub["sampler"])]
    sub["sampler"] = pd.Categorical(sub["sampler"], categories=sampler_order, ordered=True)
    counts = sub.groupby("sampler", observed=True)["data_seed"].nunique().to_dict()
    print(f"[beta_rmse_by_param] n_chains={n_chains}: seeds/sampler={counts}")
    return _param_boxplot(sub, "rmse", "RMSE",
                          "RMSE of β by Parameter - Standard Model", sampler_order, jitter=jitter)


def beta_sd_by_param(n_chains: int = 2, sd_df=None, *, jitter: bool = True) -> ggplot:
    """Mean posterior SD of beta_i (averaged over the decision units) per parameter.

    Each point is one seed: the mean over units of post_std for that parameter. The box
    summarizes the distribution across replicate seeds - the beta counterpart of
    delta_sd_faceted_by_element, showing how tightly each sampler pins down individual
    coefficients. Faceted by parameter (Alt1/Alt2/Alt3/Price) in a single row of 4 panels.
    """
    if sd_df is None:
        print("[beta_sd_by_param] loading beta_summary to compute posterior SDs ...")
        sd_df = compute_beta_post_std()
    sub = _apply_filters(sd_df, {"n_chains": n_chains}).copy()
    if sub.empty:
        raise ValueError(f"No beta posterior-SD rows for n_chains={n_chains}.")
    sampler_order = [s for s in SAMPLER_ORDER if s in set(sub["sampler"])]
    sub["sampler"] = pd.Categorical(sub["sampler"], categories=sampler_order, ordered=True)
    counts = sub.groupby("sampler", observed=True)["data_seed"].nunique().to_dict()
    print(f"[beta_sd_by_param] n_chains={n_chains}: seeds/sampler={counts}")
    return _param_boxplot(sub, "mean_post_std", "Mean SD(β̂) (over units)",
                          "SD of β - Standard Model", sampler_order, jitter=jitter)


def beta_correlation_by_param(n_chains: int = 2, corr_df=None, *, jitter: bool = True) -> ggplot:
    """Pearson correlation between posterior mean and true beta_i across units, per parameter.

    Each point is one seed: corr_i(post_mean_i, true_value_i) over the decision units.
    Values near 1 mean the sampler correctly ranks individuals by their true preferences.
    Faceted by parameter (Alt1/Alt2/Alt3/Price) in a single row of 4 panels.
    """
    if corr_df is None:
        print("[beta_correlation_by_param] loading beta_summary to compute correlations ...")
        corr_df = compute_beta_correlation()
    sub = _apply_filters(corr_df, {"n_chains": n_chains}).copy()
    if sub.empty:
        raise ValueError(f"No beta correlation rows for n_chains={n_chains}.")
    sampler_order = [s for s in SAMPLER_ORDER if s in set(sub["sampler"])]
    sub["sampler"] = pd.Categorical(sub["sampler"], categories=sampler_order, ordered=True)
    counts = sub.groupby("sampler", observed=True)["data_seed"].nunique().to_dict()
    print(f"[beta_correlation_by_param] n_chains={n_chains}: seeds/sampler={counts}")
    return _param_boxplot(sub, "correlation", "r(β̂, β)",
                          "r(β̂, β) by Parameter - Standard Model",
                          sampler_order, jitter=jitter)


def beta_coverage_by_param(n_chains: int = 2, df=None) -> ggplot:
    """Empirical 95% CI coverage of individual beta_i posteriors, per parameter.

    coverage95 = fraction of units whose true beta_i falls inside the posterior 95% CI,
    averaged over replicate seeds. Dashed reference line at 95%.
    """
    df = load_recovery("beta") if df is None else df
    sub = _apply_filters(df, {"n_chains": n_chains}).copy()
    if sub.empty:
        raise ValueError(f"No beta_recovery rows for n_chains={n_chains}.")

    cov = (
        sub.groupby(["param", "sampler"], observed=True)["coverage95"]
        .mean().mul(100).reset_index()
        .rename(columns={"coverage95": "coverage_pct"})
    )
    sampler_order = [s for s in SAMPLER_ORDER if s in set(cov["sampler"])]
    param_order = [p for p in _PARAM_ORDER if p in set(cov["param"])]
    cov["sampler"] = pd.Categorical(cov["sampler"], categories=sampler_order, ordered=True)
    cov["param"] = pd.Categorical(cov["param"], categories=param_order, ordered=True)
    fill_vals = [SAMPLER_COLORS[s] for s in sampler_order]

    p = (
        ggplot(cov, aes(x="sampler", y="coverage_pct", fill="sampler"))
        + geom_col(width=0.6)
        + geom_hline(yintercept=95, linetype="dashed", color="#555555", size=0.8)
        + facet_wrap("param", ncol=4, labeller="label_value")
        + scale_fill_manual(values=fill_vals,
                            labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
        + scale_x_discrete(labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
        + scale_y_continuous(breaks=[80, 85, 90, 95, 100])
        + coord_cartesian(ylim=(80, 100))
        + labs(x="Sampler", y="Coverage (%)", fill="Sampler",
               title="95% CI Coverage of β - Standard Model")
        + theme_bw()
        + theme(figure_size=(12, 5), axis_text_x=element_text(size=8),
                plot_title=element_text(size=11))
    )
    return p


# ----------------------------------------------------------------------------- #
# Mu recovery (NEW for the standard model: the population mean vs TRUE_MU is a
# plain 4-vector at K = 1, so per-parameter bias/coverage is directly meaningful).
# ----------------------------------------------------------------------------- #
def mu_bias_by_param(n_chains: int = 2, df=None, *, jitter: bool = True) -> ggplot:
    """Bias of the population mean mu (post_mean - true_value) per parameter.

    One boxplot per (sampler, param), distribution over replicate seeds - the same
    1x4 layout as the beta plots. This is the numeric form of the standard analysis
    notebook's summarize_mu comparison.
    """
    df = load_recovery("mu") if df is None else df
    sub = _apply_filters(df, {"n_chains": n_chains}).copy()
    if sub.empty:
        raise ValueError(f"No mu_recovery rows for n_chains={n_chains}.")
    sampler_order = [s for s in SAMPLER_ORDER if s in set(sub["sampler"])]
    sub["sampler"] = pd.Categorical(sub["sampler"], categories=sampler_order, ordered=True)
    counts = sub.groupby("sampler", observed=True)["data_seed"].nunique().to_dict()
    print(f"[mu_bias_by_param] n_chains={n_chains}: seeds/sampler={counts}")
    return _param_boxplot(sub, "bias", "Bias (μ̂−μ)",
                          "Bias of μ - Standard Model", sampler_order, jitter=jitter)


def mu_coverage_by_param(n_chains: int = 2, df=None) -> ggplot:
    """Empirical 95% CI coverage rate for each mu parameter, per sampler.

    Coverage = (number of seeds where TRUE_MU[p] falls in the 95% credible interval)
               / n_sim * 100. One bar per sampler per parameter; dashed reference at 95%.
    """
    df = load_recovery("mu") if df is None else df
    df = df.copy()
    df["in_ci"] = df["in_ci"].astype(bool)
    sub = _apply_filters(df, {"n_chains": n_chains}).copy()
    if sub.empty:
        raise ValueError(f"No mu_recovery rows for n_chains={n_chains}.")

    cov = (
        sub.groupby(["param", "sampler"], observed=True)["in_ci"]
        .mean().mul(100).reset_index()
        .rename(columns={"in_ci": "coverage_pct"})
    )
    sampler_order = [s for s in SAMPLER_ORDER if s in set(cov["sampler"])]
    param_order = [p for p in _PARAM_ORDER if p in set(cov["param"])]
    cov["sampler"] = pd.Categorical(cov["sampler"], categories=sampler_order, ordered=True)
    cov["param"] = pd.Categorical(cov["param"], categories=param_order, ordered=True)
    fill_vals = [SAMPLER_COLORS[s] for s in sampler_order]

    p = (
        ggplot(cov, aes(x="sampler", y="coverage_pct", fill="sampler"))
        + geom_col(width=0.6)
        + geom_hline(yintercept=95, linetype="dashed", color="#555555", size=0.8)
        + facet_wrap("param", ncol=4, labeller="label_value")
        + scale_fill_manual(values=fill_vals,
                            labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
        + scale_x_discrete(labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
        + scale_y_continuous(breaks=[80, 85, 90, 95, 100])
        + coord_cartesian(ylim=(80, 100))
        + labs(x="Sampler", y="Coverage (%)", fill="Sampler",
               title="95% CI Coverage of μ - Standard Model")
        + theme_bw()
        + theme(figure_size=(12, 5), axis_text_x=element_text(size=8),
                plot_title=element_text(size=11))
    )
    return p


# ----------------------------------------------------------------------------- #
# Sigma recovery (NEW for the standard model: the POSTERIOR SIGMA analysis - every
# lower-triangle element of the heterogeneity covariance vs TRUE_SIGMA; the
# `empirical` reference column - cov of the TRUE unit betas - lives in the
# make_tables summary, matching the notebook's plot_final_covariance_complete).
# ----------------------------------------------------------------------------- #
def sigma_bias_faceted_by_element(n_chains: int = 2, df=None, *, jitter: bool = True) -> ggplot:
    """Signed error of each posterior Sigma element (post_mean - true_value), per sampler.

    One panel per lower-triangle element (incl. the diagonal; 10 panels for P = 4)
    in a 4-column grid with free y-scale per panel - the same transparent-box +
    jitter layout as the Delta element plots.
    """
    df = load_recovery("sigma") if df is None else df
    df = df.copy()
    df["element"] = df.apply(lambda r: sigma_element_label(r["row"], r["col"]), axis=1)
    sub = _apply_filters(df, {"n_chains": n_chains}).copy()
    if sub.empty:
        raise ValueError(f"No sigma_recovery rows for n_chains={n_chains}.")

    sampler_order = [s for s in SAMPLER_ORDER if s in set(sub["sampler"])]
    sub["sampler"] = pd.Categorical(sub["sampler"], categories=sampler_order, ordered=True)
    element_order = (
        sub.sort_values(["row", "col"])["element"].drop_duplicates().tolist()
    )
    sub["element"] = pd.Categorical(sub["element"], categories=element_order, ordered=True)

    counts = sub.groupby("sampler", observed=True)["data_seed"].nunique().to_dict()
    print(f"[sigma_bias_faceted_by_element] n_chains={n_chains}: seeds/sampler={counts}")

    color_vals = [SAMPLER_COLORS[s] for s in sampler_order]

    p = ggplot(sub, aes(x="sampler", y="bias", color="sampler"))
    p = p + geom_hline(yintercept=0, linetype="dashed", color="#aaaaaa")
    if jitter:
        p = p + geom_jitter(width=0.2, height=0, size=0.8, alpha=0.45)
    p = (p
         + geom_boxplot(fill="#FFFFFF00", outlier_alpha=0)
         + facet_wrap("element", ncol=4, scales="free_y", labeller="label_value")
         + scale_color_manual(values=color_vals, labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
         + scale_x_discrete(labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
         + labs(x="Sampler", y="Error (Σ̂−Σ)", color="Sampler",
                title="Error of Σ Elements - Standard Model")
         + theme_bw()
         + theme(figure_size=(14, 8.5), axis_text_x=element_text(size=8),
                 plot_title=element_text(size=11))
    )
    return p


# ----------------------------------------------------------------------------- #
# Marginal-density distances vs the true DGP.
# ----------------------------------------------------------------------------- #
def marginal_metric_boxplot(metric: str = "Hellinger", n_chains: int = 2,
                            df=None, *, grid: str = "chebyshev",
                            jitter: bool = True) -> ggplot:
    """Per-metric marginal-distance boxplot comparing the samplers against the true DGP.

    x-axis = sampler, faceted by parameter (1x4), free y per panel. Each box pools the
    replicate seeds for that (param, sampler) cell; each point is one fit. Directly
    answers: under `metric`, which sampler's fitted marginal sits closest to the true
    DGP marginal, per parameter.

    Non-finite values are dropped (e.g. KL = inf where the model has mass in the true
    marginal's deep tail), so the boxplot renders cleanly; a note prints how many were
    cut. `metric` is one of MARGINAL_METRICS. `grid` selects the evaluation-grid
    scenario the distances were computed on ('full' or 'chebyshev').
    """
    df = load_recovery("marginal_distances") if df is None else df
    if "grid" in df.columns:
        df = df[df["grid"] == grid]
    sub = df[df["n_chains"] == n_chains].copy()
    if sub.empty:
        raise ValueError(f"No marginal_distances rows for n_chains={n_chains}, grid={grid!r}.")
    if metric not in sub.columns:
        raise ValueError(f"Metric '{metric}' not in marginal_distances columns. "
                         f"Available: {[c for c in MARGINAL_METRICS if c in sub.columns]}")

    n_total = len(sub)
    sub = sub[np.isfinite(sub[metric])]
    n_cut = n_total - len(sub)
    if sub.empty:
        raise ValueError(f"All {metric} values non-finite for n_chains={n_chains}.")

    sampler_order = [s for s in SAMPLER_ORDER if s in set(sub["sampler"])]
    param_order = [p for p in _PARAM_ORDER if p in set(sub["param"])]
    sub["sampler"] = pd.Categorical(sub["sampler"], categories=sampler_order, ordered=True)
    sub["param"] = pd.Categorical(sub["param"], categories=param_order, ordered=True)

    counts = sub.groupby("sampler", observed=True)["data_seed"].nunique().to_dict()
    print(f"[marginal_metric_boxplot] {metric} c{n_chains} {grid}: seeds/box={counts}"
          + (f"  (dropped {n_cut} non-finite)" if n_cut else ""))

    color_vals = [SAMPLER_COLORS[s] for s in sampler_order]
    ylabel = MARGINAL_METRIC_LABELS.get(metric, metric)

    p = ggplot(sub, aes(x="sampler", y=metric, color="sampler"))
    if jitter:
        p = p + geom_jitter(width=0.2, height=0, size=0.7, alpha=0.4)
    p = (p
         + geom_boxplot(fill="#FFFFFF00", outlier_alpha=0)
         + facet_wrap("param", ncol=4, scales="free_y", labeller="label_value")
         + scale_color_manual(values=color_vals,
                              labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
         + scale_x_discrete(labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
         + labs(x="Sampler", y=ylabel, color="Sampler",
                title=f"{ylabel} vs True DGP ({grid} grid)")
         + theme_bw()
         + theme(figure_size=(12, 4.5), axis_text_x=element_text(size=7),
                 plot_title=element_text(size=11))
    )
    return p


def retained_mass_boxplot(n_chains: int = 2, df=None, *, grid: str = "chebyshev",
                          jitter: bool = True) -> ggplot:
    """Realised probability mass of each model's own marginal retained inside the
    evaluation-grid window (mc.retained_mass), vs the theoretical Chebyshev guarantee.

    x-axis = sampler, faceted by parameter (1x4), free y per panel. Each box pools the
    replicate seeds for that (param, sampler) cell; each point is one fit. Dashed line at
    0.96 = the theoretical minimum guaranteed by Chebyshev's inequality at k=5
    (1 - 1/5**2); values should sit at or above it. `grid` selects the evaluation-grid
    scenario ('full' trivially retains ~100%; 'chebyshev' is the meaningful case).
    """
    df = load_recovery("marginal_distances") if df is None else df
    if "grid" in df.columns:
        df = df[df["grid"] == grid]
    sub = df[df["n_chains"] == n_chains].copy()
    if sub.empty:
        raise ValueError(f"No marginal_distances rows for n_chains={n_chains}, grid={grid!r}.")
    if "retained_mass_model" not in sub.columns:
        raise ValueError("Column 'retained_mass_model' not in marginal_distances - "
                         "re-gather data/out after the Chebyshev mass-guarantee fix.")

    sampler_order = [s for s in SAMPLER_ORDER if s in set(sub["sampler"])]
    param_order = [p for p in _PARAM_ORDER if p in set(sub["param"])]
    sub["sampler"] = pd.Categorical(sub["sampler"], categories=sampler_order, ordered=True)
    sub["param"] = pd.Categorical(sub["param"], categories=param_order, ordered=True)

    counts = sub.groupby("sampler", observed=True)["data_seed"].nunique().to_dict()
    print(f"[retained_mass_boxplot] c{n_chains} {grid}: seeds/box={counts}")

    color_vals = [SAMPLER_COLORS[s] for s in sampler_order]

    p = ggplot(sub, aes(x="sampler", y="retained_mass_model", color="sampler"))
    if jitter:
        p = p + geom_jitter(width=0.2, height=0, size=0.7, alpha=0.4)
    p = (p
         + geom_boxplot(fill="#FFFFFF00", outlier_alpha=0)
         + geom_hline(yintercept=0.96, linetype="dashed", color="#555555", size=0.7)
         + facet_wrap("param", ncol=4, scales="free_y", labeller="label_value")
         + scale_color_manual(values=color_vals,
                              labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
         + scale_x_discrete(labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
         + labs(x="Sampler", y="Retained mass", color="Sampler",
                title=f"Retained Mass vs Chebyshev Guarantee ({grid} grid)")
         + theme_bw()
         + theme(figure_size=(12, 4.5), axis_text_x=element_text(size=7),
                 plot_title=element_text(size=11))
    )
    return p


def kl_inf_count_plot(n_chains: int = 2, df=None, *, grid: str = "chebyshev") -> ggplot:
    """Number of seeds where KL(model||true) came back +inf (catastrophic tail
    mismatch: the fitted marginal puts mass where the true DGP density is ~0).

    x-axis = sampler, faceted by parameter (4 panels). y = raw count of +inf seeds out
    of ~100. `grid` selects the evaluation-grid scenario the distances were computed on -
    the 'full' envelope is far more prone to this than the 'chebyshev'-trimmed grid.
    """
    df = load_recovery("marginal_distances") if df is None else df
    if "grid" in df.columns:
        df = df[df["grid"] == grid]
    sub = df[df["n_chains"] == n_chains].copy()
    if sub.empty:
        raise ValueError(f"No marginal_distances rows for n_chains={n_chains}, grid={grid!r}.")
    if "KL" not in sub.columns:
        raise ValueError("Column 'KL' not in marginal_distances.")

    sub["is_inf"] = ~np.isfinite(sub["KL"])
    cnt = (
        sub.groupby(["param", "sampler"], observed=True)["is_inf"]
        .sum().reset_index().rename(columns={"is_inf": "n_inf"})
    )

    sampler_order = [s for s in SAMPLER_ORDER if s in set(cnt["sampler"])]
    param_order = [p for p in _PARAM_ORDER if p in set(cnt["param"])]
    cnt["sampler"] = pd.Categorical(cnt["sampler"], categories=sampler_order, ordered=True)
    cnt["param"] = pd.Categorical(cnt["param"], categories=param_order, ordered=True)

    fill_vals = [SAMPLER_COLORS[s] for s in sampler_order]

    p = (
        ggplot(cnt, aes(x="sampler", y="n_inf", fill="sampler"))
        + geom_col(width=0.6)
        + facet_wrap("param", ncol=4, labeller="label_value")
        + scale_fill_manual(values=fill_vals,
                            labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
        + scale_x_discrete(labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
        + labs(x="Sampler", y="# seeds with KL = inf", fill="Sampler",
               title=f"KL Divergence: Infinite-Value Count ({grid} grid)")
        + theme_bw()
        + theme(figure_size=(12, 4.5), axis_text_x=element_text(size=8),
                plot_title=element_text(size=11))
    )
    return p


def marginal_distances_faceted_by_metric(n_chains: int = 2, df=None, *,
                                          grid: str = "chebyshev") -> ggplot:
    """All five distance metrics in one figure.

    x-axis = sampler, faceted by metric (5 rows) and param (4 cols).
    Useful for a quick side-by-side sanity check that the metrics agree on ordering.
    `grid` selects the evaluation-grid scenario ('full' or 'chebyshev').
    """
    df = load_recovery("marginal_distances") if df is None else df
    if "grid" in df.columns:
        df = df[df["grid"] == grid]
    sub = df[df["n_chains"] == n_chains].copy()
    if sub.empty:
        raise ValueError(f"No marginal_distances rows for n_chains={n_chains}, grid={grid!r}.")

    # Melt to long format: one row per (run, param, metric).
    id_cols = ["dataset_key", "scenario", "k_true", "data_seed", "k_model", "sampler", "n_chains", "param"]
    id_cols = [c for c in id_cols if c in sub.columns]
    metrics_present = [m for m in MARGINAL_METRICS if m in sub.columns]
    long = sub.melt(id_vars=id_cols, value_vars=metrics_present,
                    var_name="metric", value_name="distance")

    sampler_order = [s for s in SAMPLER_ORDER if s in set(long["sampler"])]
    param_order = [p for p in _PARAM_ORDER if p in set(long["param"])]
    long["sampler"] = pd.Categorical(long["sampler"], categories=sampler_order, ordered=True)
    long["param"] = pd.Categorical(long["param"], categories=param_order, ordered=True)
    long["metric"] = pd.Categorical(long["metric"], categories=metrics_present, ordered=True)

    color_vals = [SAMPLER_COLORS[s] for s in sampler_order]

    p = (
        ggplot(long, aes(x="sampler", y="distance", color="sampler"))
        + geom_boxplot(fill="#FFFFFF00", outlier_alpha=0.3)
        + facet_grid(rows="metric", cols="param", scales="free_y")
        + scale_color_manual(values=color_vals,
                             labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
        + scale_x_discrete(labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
        + labs(x="Sampler", y="Distance", color="Sampler",
               title=f"Marginal Distances vs True DGP - Standard Model ({grid} grid)")
        + theme_bw()
        + theme(figure_size=(14, 12), axis_text_x=element_text(size=7),
                plot_title=element_text(size=11))
    )
    return p


# ----------------------------------------------------------------------------- #
# Runtime.
# ----------------------------------------------------------------------------- #
def runtime_by_sampler(n_chains: int = 2, df: Optional[pd.DataFrame] = None, *,
                       logy: bool = False, jitter: bool = True) -> ggplot:
    """Wall-clock runtime per fit (in MINUTES), by sampler. Each box pools the per-dataset
    runtimes of one sampler over all replicate seeds. Linear y-axis by default (the three
    standard-model samplers run on comparable time scales); pass logy=True to fall back to
    a log axis if a sampler's cost spans orders of magnitude. Source: runs.csv
    (runtime_s = the timed sampler call, converted to minutes here)."""
    df = load_recovery("runs") if df is None else df
    df = df.copy()
    df["runtime_min"] = df["runtime_s"] / 60.0
    filters = {"n_chains": n_chains}
    sub = _apply_filters(df, filters)
    if sub.empty:
        raise ValueError(f"No runs for n_chains={n_chains}.")
    print(f"[runtime_by_sampler] n_chains={n_chains}: runs/sampler="
          f"{sub.groupby('sampler', observed=True)['runtime_min'].count().to_dict()}")
    return recovery_boxplot(
        df, value="runtime_min", x="sampler", filters=filters,
        hline=None, jitter=jitter, logy=logy,
        title=f"Runtime by sampler - Standard Model (n_chains={n_chains})",
        xlab="Sampler",
        ylab="Runtime (min, log)" if logy else "Runtime (min)", figure_size=(7.5, 5.0),
    )


def main() -> None:
    print("== Runtime by sampler ==")
    p_rt = runtime_by_sampler()
    print("wrote", save(p_rt, "runtime/plots/runtime_by_sampler.png"))


# ----------------------------------------------------------------------------- #
# Consolidated RMSE: ONE per-run number per parameter block (beta / Delta), pooling
# every element of the block. Definitions per run (= one dataset x sampler fit):
#   beta : sqrt(mean over params of rmse_p^2). rmse_p in beta_recovery.csv is the
#          unit-level RMSE over the N units, and every param covers the same N,
#          so this equals the RMSE pooled over all N*P unit-level errors.
#   delta: sqrt(mean over the D*P elements of (post_mean - true)^2), from the
#          per-element signed `bias` column of delta_recovery.csv.
# ----------------------------------------------------------------------------- #
def consolidated_rmse_by_run(n_chains: int = 2) -> pd.DataFrame:
    """Per-run consolidated RMSE for both blocks. Long frame:
    [block ('beta'|'delta'), sampler, dataset_key, rmse]."""
    beta = load_recovery("beta")
    beta = beta[beta["n_chains"] == n_chains]
    b = (beta.assign(sq=beta["rmse"] ** 2)
         .groupby(["sampler", "dataset_key"], as_index=False)["sq"].mean())
    b["rmse"] = np.sqrt(b.pop("sq"))
    b["block"] = "beta"

    delta = load_recovery("delta")
    delta = delta[delta["n_chains"] == n_chains]
    d = (delta.assign(sq=delta["bias"] ** 2)
         .groupby(["sampler", "dataset_key"], as_index=False)["sq"].mean())
    d["rmse"] = np.sqrt(d.pop("sq"))
    d["block"] = "delta"

    return pd.concat([b, d], ignore_index=True)


def consolidated_rmse_boxplot(block: str, n_chains: int = 2, logy: bool = None) -> ggplot:
    """Distribution of the per-run consolidated RMSE of ONE parameter block
    ('beta' or 'delta'): sampler on the x-axis, one box per sampler pooling all
    replicate seeds - the same layout as the element-wise recovery plots, so boxes
    and their jitter points line up. beta defaults to a log y-scale."""
    df = consolidated_rmse_by_run(n_chains)
    df = df[df["block"] == block].copy()
    label = {"beta": "β", "delta": "Δ"}[block]
    if logy is None:
        logy = block == "beta"
    return recovery_boxplot(
        df, value="rmse", x="sampler", jitter=True, logy=logy,
        title=f"Consolidated {label} RMSE (all elements pooled per run) - {n_chains} chain(s)",
        xlab="", ylab="RMSE (per-run, pooled)" + (" [log]" if logy else ""),
        figure_size=(7.5, 5.0),
    )


if __name__ == "__main__":
    main()
