"""
Reusable parameter-recovery plotting for the mixture_c2 sampler comparison (jobs 100-103).

The post-processing (hpc_analysis/post_process.py) writes tidy per-element recovery
tables to data/out/mixture_c2/{delta,mu,sigma,beta}_recovery.csv. Every row is
ONE estimated element of ONE run, carrying all condition columns
(sampler, n_chains, scenario, k_true, k_model, data_seed, param, ...).

This module exposes ONE general boxplot core, `recovery_boxplot`, that draws a
boxplot of any chosen value column grouped on an x-axis (default: sampler), with
optional faceting and row filtering. The two plots the user asked for (signed bias
and posterior SD of the Delta = Z-covariate estimates, n_chains==1, samplers
bayesm/nuts/hmc) are each a single call; the many future variants (mu/sigma/beta,
other value columns, faceting by scenario or param, etc.) are also one call each.

plotnine is used deliberately (see module-level note in the docstring of
`recovery_boxplot`): the study already uses plotnine, and its grammar-of-graphics
API gives a single faceting/aesthetic vocabulary that scales to "much more of those
kinds of plots" without bespoke matplotlib axis wiring per plot.

Run from the repo root with the project venv:
    .venv/Scripts/python.exe hpc_analysis/plot_recovery.py
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
    geom_point,
    ggplot,
    labs,
    position_dodge,
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
# ----------------------------------------------------------------------------- #
REPO = Path(__file__).resolve().parents[2]
DIR_RECOVERY = REPO / "data" / "out" / "mixture_c2"
DIR_FIG = Path(__file__).resolve().parent / "out"

# Fixed sampler order/labels/colors so EVERY figure in the study is consistent.
# bayesm = red, bayesm_gibbs (the Python Gibbs replication) = light red so the two Gibbs
# arms group visually; nuts/hmc = two shades of blue (matching the marginal-density
# reference plot).
SAMPLER_ORDER = ["bayesm", "bayesm_gibbs", "nuts", "hmc"]
SAMPLER_LABELS = {"bayesm": "bayesm", "bayesm_gibbs": "Replication", "nuts": "NUTS", "hmc": "HMC"}
SAMPLER_COLORS = {"nuts": "#08519c", "hmc": "#4292c6", "iwls": "#9ecae1",
                  "bayesm": "#d62728", "bayesm_gibbs": "#fb6a4a"}
TRUE_COLOR = "#000000"

# Delta element label helpers (single source of truth for Δ_{d,p} (demo:param) notation).
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
    "invariant_ess_min": "min ESS",
    "n_divergent": "Divergences",
}

# Tables keyed by short name; what `load_recovery` reads. delta/beta are per-element
# recovery tables; the rest are per-run / per-kernel tables for direct-column boxplots.
# No mu/sigma entries: upstream only ECR-relabels pvec now, so per-component mu_k/Sigma
# recovery vs ground truth is no longer computed (see src/summaries.py).
RECOVERY_FILES = {
    "delta": "delta_recovery.csv",
    "beta": "beta_recovery.csv",
    "beta_summary": "beta_summary.csv",
    "runs": "runs.csv",
    "ecr": "ecr_report.csv",
    "diagnostics": "diagnostics.csv",
    "marginal_distances": "marginal_distances.csv",
    "marginal_diagnostics": "marginal_diagnostics.csv",
    "weights": "weights.csv",
    "pvec_means": "pvec_means.csv",
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
def load_recovery(table: str, dir_recovery: Path = DIR_RECOVERY) -> pd.DataFrame:
    """Load a recovery table by short name ('delta'|'mu'|'sigma'|'beta') and add
    a signed `bias` column where the table stores post_mean + true_value.

    delta/mu/sigma carry post_mean & true_value -> bias = post_mean - true_value
    (the per-element signed deviation). beta_recovery is already aggregated and
    ships its own `bias`/`rmse`/`coverage95`; it is returned unchanged.
    """
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


def _print_box_counts(df: pd.DataFrame, facet: Optional[str] = "k_true", by: str = "sampler"):
    """Print n (element-rows) per box so the figure's sample sizes are explicit."""
    cols = [c for c in (facet, by) if c in df.columns]
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
    scenario/param/k_true, color by n_chains, etc.) are argument changes rather
    than new plotting code -- exactly what "much more of those kinds of plots"
    needs. Returns the ggplot so callers can further adjust or `.save()` it.

    Parameters
    ----------
    df : tidy recovery frame (from `load_recovery`).
    value : y-axis column to summarize (default 'bias'; 'post_std' for the SD plot).
    x : grouping column for the x-axis (default 'sampler').
    filters : {col: value | [values]} row filter, e.g. {'n_chains': 1}.
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


def save(plot, filename: str, dir_fig: Path = DIR_FIG, dpi: int = 150) -> Path:
    """Save a ggplot or matplotlib Figure under dir_fig (created if missing). Returns the path."""
    out = dir_fig / filename
    out.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(plot, matplotlib.figure.Figure):
        plot.savefig(str(out), dpi=dpi, bbox_inches="tight")
        plt.close(plot)
    else:
        plot.save(out, dpi=dpi, verbose=False)
    return out


def delta_bias_faceted_by_element(n_chains: int = 2, k_true: int = 1,
                                   df=None, *, jitter: bool = True) -> ggplot:
    """Delta bias with one panel per Δ element in a 4x2 grid, free y-scale per panel.

    Each element gets its own y-axis so outliers in one element do not compress others.
    Boxplots are transparent (outline only); jittered points are colored by sampler.
    One combined sampler legend on the right.
    """
    df = load_recovery("delta") if df is None else df
    df = df.copy()
    df["element"] = df.apply(lambda r: delta_element_label(r["demo"], r["param"]), axis=1)
    filters = {"n_chains": n_chains, "k_true": k_true}
    sub = _apply_filters(df, filters).copy()
    if sub.empty:
        raise ValueError(f"No delta_recovery rows for n_chains={n_chains}, k_true={k_true}.")

    sampler_order = [s for s in SAMPLER_ORDER if s in set(sub["sampler"])]
    sub["sampler"] = pd.Categorical(sub["sampler"], categories=sampler_order, ordered=True)
    element_order = (
        sub.sort_values(["demo", "param"])["element"].drop_duplicates().tolist()
    )
    sub["element"] = pd.Categorical(sub["element"], categories=element_order, ordered=True)

    counts = sub.groupby("sampler", observed=True)["data_seed"].nunique().to_dict()
    print(f"[delta_bias_faceted_by_element] n_chains={n_chains} k_true={k_true}: "
          f"seeds/sampler={counts}")

    color_vals = [SAMPLER_COLORS[s] for s in sampler_order]
    n_comp = f"{k_true} True Component" + ("s" if k_true != 1 else "")

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
                title=f"Bias of Δ - {n_comp}")
         + theme_bw()
         + theme(figure_size=(14, 7), axis_text_x=element_text(size=8),
                 plot_title=element_text(size=11))
    )
    return p


def delta_sd_faceted_by_element(n_chains: int = 2, k_true: int = 1,
                                df=None, *, jitter: bool = True) -> ggplot:
    """Posterior SD of Delta with one panel per element in a 4x2 grid, free y-scale per panel.

    Same layout as delta_bias_faceted_by_element: transparent boxes, jitter colored by sampler,
    one legend on the right. y-axis shows the posterior standard deviation (post_std), which
    reflects how precisely each sampler pins down the corresponding Delta element.
    """
    df = load_recovery("delta") if df is None else df
    df = df.copy()
    df["element"] = df.apply(lambda r: delta_element_label(r["demo"], r["param"]), axis=1)
    filters = {"n_chains": n_chains, "k_true": k_true}
    sub = _apply_filters(df, filters).copy()
    if sub.empty:
        raise ValueError(f"No delta_recovery rows for n_chains={n_chains}, k_true={k_true}.")

    sampler_order = [s for s in SAMPLER_ORDER if s in set(sub["sampler"])]
    sub["sampler"] = pd.Categorical(sub["sampler"], categories=sampler_order, ordered=True)
    element_order = (
        sub.sort_values(["demo", "param"])["element"].drop_duplicates().tolist()
    )
    sub["element"] = pd.Categorical(sub["element"], categories=element_order, ordered=True)

    counts = sub.groupby("sampler", observed=True)["data_seed"].nunique().to_dict()
    print(f"[delta_sd_faceted_by_element] n_chains={n_chains} k_true={k_true}: "
          f"seeds/sampler={counts}")

    color_vals = [SAMPLER_COLORS[s] for s in sampler_order]
    n_comp = f"{k_true} True Component" + ("s" if k_true != 1 else "")

    p = ggplot(sub, aes(x="sampler", y="post_std", color="sampler"))
    if jitter:
        p = p + geom_jitter(width=0.2, height=0, size=0.8, alpha=0.45)
    p = (p
         + geom_boxplot(fill="#FFFFFF00", outlier_alpha=0)
         + facet_wrap("element", ncol=4, scales="free_y", labeller="label_value")
         + scale_color_manual(values=color_vals, labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
         + scale_x_discrete(labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
         + labs(x="Sampler", y="SD(Δ̂)", color="Sampler",
                title=f"SD of Δ - {n_comp}")
         + theme_bw()
         + theme(figure_size=(14, 7), axis_text_x=element_text(size=8),
                 plot_title=element_text(size=11))
    )
    return p


def delta_rmse_faceted_by_element(n_chains: int = 2, k_true: int = 1,
                                   df=None, *, jitter: bool = True) -> ggplot:
    """Absolute error |post_mean - true_value| of Delta per element in a 4x2 grid.

    Same layout as delta_bias_faceted_by_element. Each jittered point is one replicate
    seed; the box summarizes the distribution of absolute errors across seeds.
    Unlike bias, absolute error cannot cancel across seeds, so this directly shows
    how large the typical estimation error is per element per sampler.
    """
    df = load_recovery("delta") if df is None else df
    df = df.copy()
    df["element"] = df.apply(lambda r: delta_element_label(r["demo"], r["param"]), axis=1)
    df["abs_error"] = df["bias"].abs()
    filters = {"n_chains": n_chains, "k_true": k_true}
    sub = _apply_filters(df, filters).copy()
    if sub.empty:
        raise ValueError(f"No delta_recovery rows for n_chains={n_chains}, k_true={k_true}.")

    sampler_order = [s for s in SAMPLER_ORDER if s in set(sub["sampler"])]
    sub["sampler"] = pd.Categorical(sub["sampler"], categories=sampler_order, ordered=True)
    element_order = (
        sub.sort_values(["demo", "param"])["element"].drop_duplicates().tolist()
    )
    sub["element"] = pd.Categorical(sub["element"], categories=element_order, ordered=True)

    counts = sub.groupby("sampler", observed=True)["data_seed"].nunique().to_dict()
    print(f"[delta_rmse_faceted_by_element] n_chains={n_chains} k_true={k_true}: "
          f"seeds/sampler={counts}")

    color_vals = [SAMPLER_COLORS[s] for s in sampler_order]
    n_comp = f"{k_true} True Component" + ("s" if k_true != 1 else "")

    p = ggplot(sub, aes(x="sampler", y="abs_error", color="sampler"))
    if jitter:
        p = p + geom_jitter(width=0.2, height=0, size=0.8, alpha=0.45)
    p = (p
         + geom_boxplot(fill="#FFFFFF00", outlier_alpha=0)
         + facet_wrap("element", ncol=4, scales="free_y", labeller="label_value")
         + scale_color_manual(values=color_vals, labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
         + scale_x_discrete(labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
         + labs(x="Sampler", y="|Δ̂−Δ|", color="Sampler",
                title=f"|Δ̂−Δ| - {n_comp}")
         + theme_bw()
         + theme(figure_size=(14, 7), axis_text_x=element_text(size=8),
                 plot_title=element_text(size=11))
    )
    return p


def delta_coverage_faceted_by_element(n_chains: int = 2, k_true: int = 1,
                                       df=None) -> ggplot:
    """Empirical 95% CI coverage rate for each Delta element, per sampler.

    Coverage = (number of seeds where true_value falls in the 95% credible interval)
               / n_sim * 100.  One bar per sampler per element; dashed reference at 95%.
    A well-calibrated sampler should land near 95% for every element.
    """
    df = load_recovery("delta") if df is None else df
    df = df.copy()
    df["element"] = df.apply(lambda r: delta_element_label(r["demo"], r["param"]), axis=1)
    df["in_ci"] = df["in_ci"].astype(bool)
    filters = {"n_chains": n_chains, "k_true": k_true}
    sub = _apply_filters(df, filters).copy()
    if sub.empty:
        raise ValueError(f"No delta_recovery rows for n_chains={n_chains}, k_true={k_true}.")

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
    n_comp = f"{k_true} True Component" + ("s" if k_true != 1 else "")

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
               title=f"95% CI Coverage of Δ - {n_comp}")
        + theme_bw()
        + theme(figure_size=(14, 7), axis_text_x=element_text(size=8),
                plot_title=element_text(size=11))
    )
    return p


def delta_coverage_by_ktrue(n_chains: int = 2, df=None) -> ggplot:
    """Empirical 95% CI coverage of Delta elements across true component counts.

    x-axis = k_true, bars dodged by sampler, faceted 4x2 by element.
    Shows whether coverage degrades as the model is overspecified (k_model > k_true).
    """
    df = load_recovery("delta") if df is None else df
    df = df[df["n_chains"] == n_chains].copy()
    if df.empty:
        raise ValueError(f"No delta_recovery rows for n_chains={n_chains}.")

    df["element"] = df.apply(lambda r: delta_element_label(r["demo"], r["param"]), axis=1)
    df["in_ci"] = df["in_ci"].astype(bool)

    cov = (
        df.groupby(["element", "demo", "param", "sampler", "k_true"], observed=True)["in_ci"]
        .mean().mul(100).reset_index()
        .rename(columns={"in_ci": "coverage_pct"})
    )

    sampler_order = [s for s in SAMPLER_ORDER if s in set(cov["sampler"])]
    ktrue_order = [str(k) for k in sorted(cov["k_true"].unique())]
    cov["sampler"] = pd.Categorical(cov["sampler"], categories=sampler_order, ordered=True)
    cov["k_true"] = pd.Categorical(cov["k_true"].astype(str), categories=ktrue_order, ordered=True)
    element_order = cov.sort_values(["demo", "param"])["element"].drop_duplicates().tolist()
    cov["element"] = pd.Categorical(cov["element"], categories=element_order, ordered=True)

    fill_vals = [SAMPLER_COLORS[s] for s in sampler_order]

    p = (
        ggplot(cov, aes(x="k_true", y="coverage_pct", fill="sampler"))
        + geom_col(position="dodge", width=0.7)
        + geom_hline(yintercept=95, linetype="dashed", color="#555555", size=0.8)
        + facet_wrap("element", ncol=4, labeller="label_value")
        + scale_fill_manual(values=fill_vals,
                            labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
        + scale_y_continuous(breaks=[80, 85, 90, 95, 100])
        + coord_cartesian(ylim=(80, 100))
        + labs(x="k_true", y="Coverage (%)", fill="Sampler",
               title="95% CI Coverage of Δ vs k_true")
        + theme_bw()
        + theme(figure_size=(14, 7), axis_text_x=element_text(size=9),
                plot_title=element_text(size=11))
    )
    return p


def _beta_boxplot(df: pd.DataFrame, y_col: str, y_label: str,
                  title: str, sampler_order: list, *, jitter: bool) -> ggplot:
    """Shared core for beta per-param boxplots (bias / rmse / mae)."""
    param_order = [p for p in _PARAM_ORDER if p in set(df["param"])]
    df = df.copy()
    df["param"] = pd.Categorical(df["param"], categories=param_order, ordered=True)
    color_vals = [SAMPLER_COLORS[s] for s in sampler_order]

    p = ggplot(df, aes(x="sampler", y=y_col, color="sampler"))
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


def compute_beta_correlation(df_summary=None) -> pd.DataFrame:
    """Compute per-run per-param Pearson correlation between post_mean and true_value.

    Loads beta_summary.csv (1.68M rows) and returns one row per
    (dataset_key, scenario, k_true, data_seed, k_model, sampler, n_chains, param)
    with a `correlation` column: corr_i(post_mean_i, true_value_i) across 330 units.
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


def beta_bias_by_param(n_chains: int = 2, k_true: int = 1,
                       df=None, *, jitter: bool = True) -> ggplot:
    """Bias (mean post_mean - true_value over 330 units) per beta parameter.

    One boxplot per (sampler, param), distribution over replicate seeds.
    Faceted by parameter (Alt1/Alt2/Alt3/Price) in a single row of 4 panels.
    """
    df = load_recovery("beta") if df is None else df
    sub = _apply_filters(df, {"n_chains": n_chains, "k_true": k_true}).copy()
    if sub.empty:
        raise ValueError(f"No beta_recovery rows for n_chains={n_chains}, k_true={k_true}.")
    sampler_order = [s for s in SAMPLER_ORDER if s in set(sub["sampler"])]
    sub["sampler"] = pd.Categorical(sub["sampler"], categories=sampler_order, ordered=True)
    n_comp = f"{k_true} True Component" + ("s" if k_true != 1 else "")
    counts = sub.groupby("sampler", observed=True)["data_seed"].nunique().to_dict()
    print(f"[beta_bias_by_param] n_chains={n_chains} k_true={k_true}: seeds/sampler={counts}")
    return _beta_boxplot(sub, "bias", "Bias (β̂−β)",
                         f"β Bias by Parameter - {n_comp}", sampler_order, jitter=jitter)


def beta_rmse_by_param(n_chains: int = 2, k_true: int = 1,
                       df=None, *, jitter: bool = True) -> ggplot:
    """RMSE of beta_i posteriors per parameter (aggregated over 330 decision units).

    RMSE = sqrt(mean((post_mean_i - true_i)^2)) across units, one value per run.
    Distribution over replicate seeds shown as a boxplot per sampler per parameter.
    """
    df = load_recovery("beta") if df is None else df
    sub = _apply_filters(df, {"n_chains": n_chains, "k_true": k_true}).copy()
    if sub.empty:
        raise ValueError(f"No beta_recovery rows for n_chains={n_chains}, k_true={k_true}.")
    sampler_order = [s for s in SAMPLER_ORDER if s in set(sub["sampler"])]
    sub["sampler"] = pd.Categorical(sub["sampler"], categories=sampler_order, ordered=True)
    n_comp = f"{k_true} True Component" + ("s" if k_true != 1 else "")
    counts = sub.groupby("sampler", observed=True)["data_seed"].nunique().to_dict()
    print(f"[beta_rmse_by_param] n_chains={n_chains} k_true={k_true}: seeds/sampler={counts}")
    return _beta_boxplot(sub, "rmse", "RMSE",
                         f"β RMSE by Parameter - {n_comp}", sampler_order, jitter=jitter)


def beta_correlation_by_param(n_chains: int = 2, k_true: int = 1,
                              corr_df=None, *, jitter: bool = True) -> ggplot:
    """Pearson correlation between posterior mean and true beta_i across 330 units, per parameter.

    Each point is one seed: corr_i(post_mean_i, true_value_i) over the 330 decision units.
    Values near 1 mean the sampler correctly ranks individuals by their true preferences.
    Faceted by parameter (Alt1/Alt2/Alt3/Price) in a single row of 4 panels.
    """
    if corr_df is None:
        print(f"[beta_correlation_by_param] loading beta_summary to compute correlations ...")
        corr_df = compute_beta_correlation()
    sub = _apply_filters(corr_df, {"n_chains": n_chains, "k_true": k_true}).copy()
    if sub.empty:
        raise ValueError(f"No beta correlation rows for n_chains={n_chains}, k_true={k_true}.")
    sampler_order = [s for s in SAMPLER_ORDER if s in set(sub["sampler"])]
    sub["sampler"] = pd.Categorical(sub["sampler"], categories=sampler_order, ordered=True)
    n_comp = f"{k_true} True Component" + ("s" if k_true != 1 else "")
    counts = sub.groupby("sampler", observed=True)["data_seed"].nunique().to_dict()
    print(f"[beta_correlation_by_param] n_chains={n_chains} k_true={k_true}: seeds/sampler={counts}")
    return _beta_boxplot(sub, "correlation", "r(β̂, β)",
                         f"β Correlation by Parameter - {n_comp}", sampler_order, jitter=jitter)


def beta_correlation_by_ktrue(n_chains: int = 2, corr_df=None) -> ggplot:
    """Beta correlation across all true component counts in one figure.

    x-axis = k_true, boxplots dodged by sampler, faceted by parameter.
    Shows whether individual ranking degrades as the model becomes more overspecified.
    Jitter is omitted here because 3 samplers x 4 k_true groups per panel is too crowded.
    """
    if corr_df is None:
        print("[beta_correlation_by_ktrue] loading beta_summary to compute correlations ...")
        corr_df = compute_beta_correlation()
    df = corr_df[corr_df["n_chains"] == n_chains].copy()
    if df.empty:
        raise ValueError(f"No beta correlation rows for n_chains={n_chains}.")
    sampler_order = [s for s in SAMPLER_ORDER if s in set(df["sampler"])]
    param_order = [p for p in _PARAM_ORDER if p in set(df["param"])]
    ktrue_order = [str(k) for k in sorted(df["k_true"].unique())]
    df["sampler"] = pd.Categorical(df["sampler"], categories=sampler_order, ordered=True)
    df["param"] = pd.Categorical(df["param"], categories=param_order, ordered=True)
    df["k_true"] = pd.Categorical(df["k_true"].astype(str), categories=ktrue_order, ordered=True)
    color_vals = [SAMPLER_COLORS[s] for s in sampler_order]

    dodge = position_dodge(width=0.9)
    p = (
        ggplot(df, aes(x="k_true", y="correlation", color="sampler"))
        + geom_boxplot(width=0.28, fill="#FFFFFF00", outlier_alpha=0, position=dodge)
        + facet_wrap("param", ncol=4, scales="free_y", labeller="label_value")
        + scale_color_manual(values=color_vals,
                             labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
        + scale_x_discrete(expand=(0, 0.4))
        + labs(x="k_true", y="r(β̂, β)", color="Sampler",
               title="β Correlation vs k_true")
        + theme_bw()
        + theme(figure_size=(12, 5), axis_text_x=element_text(size=9),
                plot_title=element_text(size=11))
    )
    return p


def beta_coverage_by_param(n_chains: int = 2, k_true: int = 1, df=None) -> ggplot:
    """Empirical 95% CI coverage of individual beta_i posteriors, per parameter.

    coverage95 = fraction of 330 units whose true beta_i falls inside the posterior 95% CI,
    averaged over replicate seeds. Dashed reference line at 95%.
    """
    df = load_recovery("beta") if df is None else df
    sub = _apply_filters(df, {"n_chains": n_chains, "k_true": k_true}).copy()
    if sub.empty:
        raise ValueError(f"No beta_recovery rows for n_chains={n_chains}, k_true={k_true}.")

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
    n_comp = f"{k_true} True Component" + ("s" if k_true != 1 else "")

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
               title=f"95% CI Coverage of β - {n_comp}")
        + theme_bw()
        + theme(figure_size=(12, 5), axis_text_x=element_text(size=8),
                plot_title=element_text(size=11))
    )
    return p


def beta_coverage_by_ktrue(n_chains: int = 2, df=None) -> ggplot:
    """Beta 95% CI coverage across true component counts.

    x-axis = k_true, bars dodged by sampler, faceted by parameter.
    Shows whether individual-level beta coverage degrades with overspecification.
    """
    df = load_recovery("beta") if df is None else df
    df = df[df["n_chains"] == n_chains].copy()
    if df.empty:
        raise ValueError(f"No beta_recovery rows for n_chains={n_chains}.")

    cov = (
        df.groupby(["param", "sampler", "k_true"], observed=True)["coverage95"]
        .mean().mul(100).reset_index()
        .rename(columns={"coverage95": "coverage_pct"})
    )
    sampler_order = [s for s in SAMPLER_ORDER if s in set(cov["sampler"])]
    param_order = [p for p in _PARAM_ORDER if p in set(cov["param"])]
    ktrue_order = [str(k) for k in sorted(cov["k_true"].unique())]
    cov["sampler"] = pd.Categorical(cov["sampler"], categories=sampler_order, ordered=True)
    cov["param"] = pd.Categorical(cov["param"], categories=param_order, ordered=True)
    cov["k_true"] = pd.Categorical(cov["k_true"].astype(str), categories=ktrue_order, ordered=True)
    fill_vals = [SAMPLER_COLORS[s] for s in sampler_order]

    p = (
        ggplot(cov, aes(x="k_true", y="coverage_pct", fill="sampler"))
        + geom_col(position="dodge", width=0.7)
        + geom_hline(yintercept=95, linetype="dashed", color="#555555", size=0.8)
        + facet_wrap("param", ncol=4, labeller="label_value")
        + scale_fill_manual(values=fill_vals,
                            labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
        + scale_y_continuous(breaks=[80, 85, 90, 95, 100])
        + coord_cartesian(ylim=(80, 100))
        + labs(x="k_true", y="Coverage (%)", fill="Sampler",
               title="95% CI Coverage of β vs k_true")
        + theme_bw()
        + theme(figure_size=(12, 5), axis_text_x=element_text(size=9),
                plot_title=element_text(size=11))
    )
    return p


def marginal_distance_by_ktrue(n_chains: int = 2, metric: str = "Hellinger",
                               df=None, *, grid: str = "chebyshev") -> ggplot:
    """Distance between fitted marginal density and true DGP marginal, by true component count.

    x-axis = k_true, boxplots dodged by sampler, faceted by parameter (4 panels).
    Each box pools ~100 replicate seeds; each data point is one fit.
    The `metric` argument selects which distance to plot; see MARGINAL_METRICS for options.
    Hellinger is the primary metric (bounded in [0,1] and symmetric); KL/JSD/TVD
    are supplementary. `grid` selects the evaluation-grid scenario the distances were
    computed on ('full' or 'chebyshev'; every metric is stored for both).
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

    sampler_order = [s for s in SAMPLER_ORDER if s in set(sub["sampler"])]
    param_order = [p for p in _PARAM_ORDER if p in set(sub["param"])]
    ktrue_order = [str(k) for k in sorted(sub["k_true"].unique())]
    sub["sampler"] = pd.Categorical(sub["sampler"], categories=sampler_order, ordered=True)
    sub["param"] = pd.Categorical(sub["param"], categories=param_order, ordered=True)
    sub["k_true"] = pd.Categorical(sub["k_true"].astype(str), categories=ktrue_order, ordered=True)

    counts = sub.groupby(["sampler", "k_true"], observed=True)["data_seed"].nunique().to_dict()
    print(f"[marginal_distance_by_ktrue] {metric} c{n_chains} {grid}: seeds/box={counts}")

    color_vals = [SAMPLER_COLORS[s] for s in sampler_order]
    dodge = position_dodge(width=0.9)
    ylabel = MARGINAL_METRIC_LABELS.get(metric, metric)

    p = (
        ggplot(sub, aes(x="k_true", y=metric, color="sampler"))
        + geom_boxplot(width=0.28, fill="#FFFFFF00", outlier_alpha=0.3, position=dodge)
        + facet_wrap("param", ncol=4, scales="free_y", labeller="label_value")
        + scale_color_manual(values=color_vals,
                             labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
        + scale_x_discrete(expand=(0, 0.4))
        + labs(x="k_true", y=ylabel, color="Sampler",
               title=f"{ylabel} vs k_true ({grid} grid)")
        + theme_bw()
        + theme(figure_size=(12, 5), axis_text_x=element_text(size=9),
                plot_title=element_text(size=11))
    )
    return p


def marginal_distances_faceted_by_metric(n_chains: int = 2, k_true: int = 1,
                                          df=None, *, grid: str = "chebyshev") -> ggplot:
    """All five distance metrics in one figure for a single k_true value.

    x-axis = sampler, faceted by metric (5 rows) and param (4 cols).
    Useful for a quick side-by-side sanity check that the metrics agree on ordering.
    `grid` selects the evaluation-grid scenario ('full' or 'chebyshev').
    """
    df = load_recovery("marginal_distances") if df is None else df
    if "grid" in df.columns:
        df = df[df["grid"] == grid]
    sub = df[(df["n_chains"] == n_chains) & (df["k_true"] == k_true)].copy()
    if sub.empty:
        raise ValueError(f"No marginal_distances rows for n_chains={n_chains}, "
                         f"k_true={k_true}, grid={grid!r}.")

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
    n_comp = f"{k_true} True Component" + ("s" if k_true != 1 else "")

    p = (
        ggplot(long, aes(x="sampler", y="distance", color="sampler"))
        + geom_boxplot(fill="#FFFFFF00", outlier_alpha=0.3)
        + facet_grid(rows="metric", cols="param", scales="free_y")
        + scale_color_manual(values=color_vals,
                             labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
        + scale_x_discrete(labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
        + labs(x="Sampler", y="Distance", color="Sampler",
               title=f"Distances vs True DGP - {n_comp} ({grid} grid)")
        + theme_bw()
        + theme(figure_size=(14, 12), axis_text_x=element_text(size=7),
                plot_title=element_text(size=11))
    )
    return p


def marginal_metric_boxplot(metric: str = "Hellinger", n_chains: int = 2,
                            df=None, *, grid: str = "chebyshev",
                            jitter: bool = True) -> ggplot:
    """Per-metric marginal-distance boxplot comparing the samplers against the true DGP.

    x-axis = sampler, faceted as a k_true (rows) x param (cols) grid, free y per panel.
    Each box pools the replicate seeds for that (k_true, param, sampler) cell; each point is
    one fit. Directly answers: under `metric`, which sampler's fitted marginal sits closest
    to the true DGP marginal, per parameter and per overspecification level.

    Non-finite values are dropped (e.g. KL = inf where the model has mass in the true
    marginal's deep tail), so the boxplot renders cleanly; a note prints how many were cut.
    `metric` is one of MARGINAL_METRICS. `grid` selects the evaluation-grid scenario the
    distances were computed on ('full' or 'chebyshev').
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
    ktrue_order = [str(k) for k in sorted(sub["k_true"].unique())]
    sub["sampler"] = pd.Categorical(sub["sampler"], categories=sampler_order, ordered=True)
    sub["param"] = pd.Categorical(sub["param"], categories=param_order, ordered=True)
    sub["k_true"] = pd.Categorical(sub["k_true"].astype(str), categories=ktrue_order, ordered=True)

    counts = sub.groupby(["sampler", "k_true"], observed=True)["data_seed"].nunique().to_dict()
    print(f"[marginal_metric_boxplot] {metric} c{n_chains} {grid}: seeds/box={counts}"
          + (f"  (dropped {n_cut} non-finite)" if n_cut else ""))

    color_vals = [SAMPLER_COLORS[s] for s in sampler_order]
    ylabel = MARGINAL_METRIC_LABELS.get(metric, metric)

    p = ggplot(sub, aes(x="sampler", y=metric, color="sampler"))
    if jitter:
        p = p + geom_jitter(width=0.2, height=0, size=0.7, alpha=0.4)
    p = (p
         + geom_boxplot(fill="#FFFFFF00", outlier_alpha=0)
         + facet_grid(rows="k_true", cols="param", scales="free_y", labeller="label_both")
         + scale_color_manual(values=color_vals,
                              labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
         + scale_x_discrete(labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
         + labs(x="Sampler", y=ylabel, color="Sampler",
                title=f"{ylabel} vs True DGP ({grid} grid)")
         + theme_bw()
         + theme(figure_size=(12, 9), axis_text_x=element_text(size=7),
                 plot_title=element_text(size=11))
    )
    return p


def retained_mass_boxplot(n_chains: int = 2, df=None, *, grid: str = "chebyshev",
                          jitter: bool = True) -> ggplot:
    """Realised probability mass of each model's own marginal retained inside the
    evaluation-grid window (mc.retained_mass), vs the theoretical Chebyshev guarantee.

    x-axis = sampler, faceted as a k_true (rows) x param (cols) grid, free y per panel.
    Each box pools the replicate seeds for that (k_true, param, sampler) cell; each point
    is one fit. Dashed line at 0.96 = the theoretical minimum guaranteed by Chebyshev's
    inequality at k=5 (1 - 1/5**2); values should sit at or above it. `grid` selects the
    evaluation-grid scenario ('full' trivially retains ~100%; 'chebyshev' is the
    meaningful case).
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
    ktrue_order = [str(k) for k in sorted(sub["k_true"].unique())]
    sub["sampler"] = pd.Categorical(sub["sampler"], categories=sampler_order, ordered=True)
    sub["param"] = pd.Categorical(sub["param"], categories=param_order, ordered=True)
    sub["k_true"] = pd.Categorical(sub["k_true"].astype(str), categories=ktrue_order, ordered=True)

    counts = sub.groupby(["sampler", "k_true"], observed=True)["data_seed"].nunique().to_dict()
    print(f"[retained_mass_boxplot] c{n_chains} {grid}: seeds/box={counts}")

    color_vals = [SAMPLER_COLORS[s] for s in sampler_order]

    p = ggplot(sub, aes(x="sampler", y="retained_mass_model", color="sampler"))
    if jitter:
        p = p + geom_jitter(width=0.2, height=0, size=0.7, alpha=0.4)
    p = (p
         + geom_boxplot(fill="#FFFFFF00", outlier_alpha=0)
         + geom_hline(yintercept=0.96, linetype="dashed", color="#555555", size=0.7)
         + facet_grid(rows="k_true", cols="param", scales="free_y", labeller="label_both")
         + scale_color_manual(values=color_vals,
                              labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
         + scale_x_discrete(labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
         + labs(x="Sampler", y="Retained mass", color="Sampler",
                title=f"Retained Mass vs Chebyshev Guarantee ({grid} grid)")
         + theme_bw()
         + theme(figure_size=(12, 9), axis_text_x=element_text(size=7),
                 plot_title=element_text(size=11))
    )
    return p


def kl_inf_count_plot(n_chains: int = 2, df=None, *, grid: str = "chebyshev") -> ggplot:
    """Number of seeds where KL(model||true) came back +inf (catastrophic tail
    mismatch: the fitted marginal puts mass where the true DGP density is ~0).

    x-axis = k_true, bars dodged by sampler, faceted by parameter (4 panels). y = raw
    count of +inf seeds out of ~100. `grid` selects the evaluation-grid scenario the
    distances were computed on - the 'full' envelope is far more prone to this than the
    'chebyshev'-trimmed grid, since it stretches into the deep tails of surplus components.
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
        sub.groupby(["k_true", "param", "sampler"], observed=True)["is_inf"]
        .sum().reset_index().rename(columns={"is_inf": "n_inf"})
    )

    sampler_order = [s for s in SAMPLER_ORDER if s in set(cnt["sampler"])]
    param_order = [p for p in _PARAM_ORDER if p in set(cnt["param"])]
    ktrue_order = [str(k) for k in sorted(cnt["k_true"].unique())]
    cnt["sampler"] = pd.Categorical(cnt["sampler"], categories=sampler_order, ordered=True)
    cnt["param"] = pd.Categorical(cnt["param"], categories=param_order, ordered=True)
    cnt["k_true"] = pd.Categorical(cnt["k_true"].astype(str), categories=ktrue_order, ordered=True)

    fill_vals = [SAMPLER_COLORS[s] for s in sampler_order]

    p = (
        ggplot(cnt, aes(x="k_true", y="n_inf", fill="sampler"))
        + geom_col(position="dodge", width=0.7)
        + facet_wrap("param", ncol=4, labeller="label_value")
        + scale_fill_manual(values=fill_vals,
                            labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
        + labs(x="k_true", y="# seeds with KL = inf", fill="Sampler",
               title=f"KL Divergence: Infinite-Value Count vs k_true ({grid} grid)")
        + theme_bw()
        + theme(figure_size=(14, 5), axis_text_x=element_text(size=9),
                plot_title=element_text(size=11))
    )
    return p


def runtime_samplers_by_ktrue(n_chains: int = 2, df: Optional[pd.DataFrame] = None, *,
                              logy: bool = True, jitter: bool = False) -> ggplot:
    """Runtime by true-component count, with the SAMPLERS side by side (same layout as
    delta_bias_samplers_by_element). x-axis = k_true (1/2/3/5); one dodged box per sampler,
    pooling the per-fit sampling time over all seeds. Log y by default, since NUTS runtime
    blows up on overspecified mixtures (k_true < k_model) while HMC/bayesm stay flat - the
    headline cost finding. Source: runs.csv (runtime_s = the timed sampler call)."""
    df = load_recovery("runs") if df is None else df
    df = df.copy()
    filters = {"n_chains": n_chains}
    sub = _apply_filters(df, filters)
    if sub.empty:
        raise ValueError(f"No runs for n_chains={n_chains}.")
    print(f"[runtime_samplers_by_ktrue] n_chains={n_chains}: runs/sampler="
          f"{sub.groupby('sampler', observed=True)['runtime_s'].count().to_dict()}")
    return recovery_boxplot(
        df, value="runtime_s", x="k_true", color="sampler", filters=filters,
        x_order=[1, 2, 3, 5], hline=None, jitter=jitter, logy=logy,
        title="Runtime vs k_true, by sampler",
        xlab="k_true",
        ylab="Runtime (s, log)" if logy else "Runtime (s)", figure_size=(9.0, 5.5),
    )


def runtime_by_ktrue(sampler: str = "nuts", n_chains: int = 2,
                     df: Optional[pd.DataFrame] = None, *,
                     unit: Optional[str] = None, jitter: bool = True):
    """Runtime by k_true for ONE sampler - transparent boxes, colored outline and jitter points.

    unit: 'h' (hours) or 'min' (minutes); default = hours for nuts, minutes for hmc/bayesm.
    For NUTS (unit='h'): data stored in minutes, left axis = min., right axis = h. every 0.5h.
    Returns a matplotlib Figure for NUTS (dual axis) or a ggplot for the others.
    """
    df = load_recovery("runs") if df is None else df
    df = df.copy()
    if unit is None:
        unit = "h" if sampler == "nuts" else "min"
    # Always store in minutes so the primary (left) axis is consistent.
    df["runtime"] = df["runtime_s"] / 60.0
    filters = {"sampler": sampler, "n_chains": n_chains}
    sub = _apply_filters(df, filters).copy()
    if sub.empty:
        raise ValueError(f"No runs for sampler={sampler}, n_chains={n_chains}.")
    sub["k_true"] = pd.Categorical(sub["k_true"], categories=[1, 2, 3, 5], ordered=True)
    print(f"[runtime_by_ktrue] {sampler} c{n_chains}: runs/k_true="
          f"{sub.groupby('k_true', observed=True)['runtime_s'].count().to_dict()}")

    color = SAMPLER_COLORS.get(sampler, "#888888")
    label = SAMPLER_LABELS.get(sampler, sampler)

    p = ggplot(sub, aes(x="k_true", y="runtime"))
    if jitter:
        p = p + geom_jitter(width=0.18, height=0, size=0.8, alpha=0.5, color=color)
    p = (p
         + geom_boxplot(fill="#FFFFFF00", color=color, outlier_alpha=0)
         + labs(
             x="k_true",
             y="Runtime (min)",
             title=f"Runtime vs k_true - {label}",
         )
         + theme_bw()
         + theme(figure_size=(7.5, 4.8), plot_title=element_text(size=11))
    )

    if unit != "h":
        return p

    # NUTS: draw the plotnine figure and add a secondary right-hand axis in hours.
    fig = p.draw()
    ax = fig.axes[0]
    ymin, ymax = ax.get_ylim()

    # Read the grey tick color from plotnine's themed x-axis labels (untouched by our code).
    x_labels = ax.get_xticklabels()
    tick_color = x_labels[0].get_color() if x_labels else "#4C4C4C"

    # Left axis: minutes, ticks every 30 - grey tick numbers to match plotnine theme.
    min_ticks = np.arange(np.ceil(ymin / 30) * 30, ymax, 30)
    ax.set_yticks(min_ticks)
    ax.set_yticklabels([str(int(m)) for m in min_ticks], color=tick_color)

    # Right axis: hours, ticks every 1 - grey tick numbers, black axis title.
    ax2 = ax.twinx()
    ax2.set_ylim(ymin / 60, ymax / 60)
    hour_ticks = np.arange(0, int(ymax / 60) + 1, 1)
    ax2.set_yticks(hour_ticks)
    ax2.set_yticklabels([str(int(h)) for h in hour_ticks], color=tick_color)
    ax2.tick_params(axis="y", colors=tick_color)
    ax2.set_ylabel("Runtime (h)", color="black")
    ax2.grid(False)

    return fig


def runtime_plot(df: Optional[pd.DataFrame] = None, *, logy: bool = True,
                 jitter: bool = True, paired: bool = True) -> ggplot:
    """Wall-clock runtime per fit, by sampler, one panel per true-component count k_true
    (n_chains==1). Each box = the per-dataset runtimes for that (k_true, sampler). Log y by
    default since sampler cost spans orders of magnitude (NUTS trajectory cost blows up on
    overspecified mixtures while HMC/bayesm stay flat). Source: runs.csv (one row per fit)."""
    df = load_recovery("runs") if df is None else df
    d = _maybe_paired(_apply_filters(df, {"n_chains": 1}), paired, "runtime")
    _print_box_counts(d)
    return recovery_boxplot(
        d, value="runtime_s", x="sampler", facet_wrap_by="k_true", facet_scales="fixed",
        jitter=jitter, logy=logy,
        title="Runtime by sampler vs k_true",
        ylab="Runtime (s, log)" if logy else "Runtime (s)", xlab="Sampler",
        figure_size=(8.5, 6.0),
    )


def main() -> None:
    print("== Runtime by sampler, per k_true ==")
    p_rt = runtime_plot()
    print("wrote", save(p_rt, "runtime_by_sampler_ktrue.png"))


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
    [block ('beta'|'delta'), sampler, k_true, dataset_key, rmse]."""
    beta = load_recovery("beta")
    beta = beta[beta["n_chains"] == n_chains]
    b = (beta.assign(sq=beta["rmse"] ** 2)
         .groupby(["sampler", "k_true", "dataset_key"], as_index=False)["sq"].mean())
    b["rmse"] = np.sqrt(b.pop("sq"))
    b["block"] = "beta"

    delta = load_recovery("delta")
    delta = delta[delta["n_chains"] == n_chains]
    d = (delta.assign(sq=delta["bias"] ** 2)
         .groupby(["sampler", "k_true", "dataset_key"], as_index=False)["sq"].mean())
    d["rmse"] = np.sqrt(d.pop("sq"))
    d["block"] = "delta"

    return pd.concat([b, d], ignore_index=True)


def consolidated_rmse_boxplot(block: str, n_chains: int = 2, logy: bool = None) -> ggplot:
    """Distribution of the per-run consolidated RMSE of ONE parameter block
    ('beta' or 'delta'): sampler on the x-axis, one facet per k_true - the same
    layout as the element-wise recovery plots, so boxes and their jitter points
    line up. beta defaults to a log y-scale (its NUTS tail spans a decade)."""
    df = consolidated_rmse_by_run(n_chains)
    df = df[df["block"] == block].copy()
    df["K_true"] = df["k_true"].astype(str)
    label = {"beta": "β", "delta": "Δ"}[block]
    if logy is None:
        logy = block == "beta"
    return recovery_boxplot(
        df, value="rmse", x="sampler", jitter=True, logy=logy,
        facet_wrap_by="K_true",
        title=f"Consolidated {label} RMSE (pooled per run) - c{n_chains}",
        xlab="", ylab="RMSE (per-run, pooled)" + (" [log]" if logy else ""),
        figure_size=(10.0, 6.0),
    )


if __name__ == "__main__":
    main()
