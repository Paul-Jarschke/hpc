"""
Reusable parameter-recovery plotting for the k5model_mixture sampler comparison.

The post-processing (analysis/post_process.py) writes tidy per-element recovery
tables to data/out/k5model_mixture/{delta,mu,sigma,beta}_recovery.csv. Every row is
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
    .venv/Scripts/python.exe analysis/plot_recovery.py
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
    geom_boxplot,
    geom_hline,
    geom_jitter,
    ggplot,
    labs,
    scale_color_manual,
    scale_fill_manual,
    scale_x_discrete,
    scale_y_log10,
    theme,
    theme_bw,
)

# ----------------------------------------------------------------------------- #
# Locations & study-wide conventions (single source of truth for every plot).
# ----------------------------------------------------------------------------- #
REPO = Path(__file__).resolve().parent.parent
DIR_RECOVERY = REPO / "data" / "out" / "k5model_mixture"
DIR_FIG = REPO / "analysis" / "out" / "k5_results"

# Fixed sampler order/labels/colors so EVERY figure in the study is consistent.
# bayesm = red; nuts/hmc = two shades of blue (matching the marginal-density reference plot).
SAMPLER_ORDER = ["bayesm", "nuts", "hmc"]
SAMPLER_LABELS = {"bayesm": "bayesm", "nuts": "NUTS", "hmc": "HMC"}
SAMPLER_COLORS = {"nuts": "#08519c", "hmc": "#4292c6", "iwls": "#9ecae1", "bayesm": "#d62728"}
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


# Human-readable axis labels for the value columns we commonly plot.
VALUE_LABELS = {
    "bias": "Bias  (post_mean - true_value)",
    "abs_diff": "Absolute error  |post_mean - true_value|",
    "post_std": "Posterior SD  (post_std)",
    "post_mean": "Posterior mean",
    "rmse": "RMSE",
    "mean_abs_err": "Mean absolute error",
    "coverage95": "95% coverage",
    "runtime_s": "Runtime (s)",
    "invariant_ess_min": "min ESS (label-invariant)",
    "n_divergent": "divergent transitions",
}

# Tables keyed by short name; what `load_recovery` reads. The first four are per-element
# recovery tables; the rest are per-run / per-kernel tables for direct-column boxplots.
RECOVERY_FILES = {
    "delta": "delta_recovery.csv",
    "mu": "mu_recovery.csv",
    "sigma": "sigma_recovery.csv",
    "beta": "beta_recovery.csv",
    "runs": "runs.csv",
    "ecr": "ecr_report.csv",
    "diagnostics": "diagnostics.csv",
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
    the frame — e.g. the sparse local TEST data where samplers cover different datasets.
    On the real grid every sampler covers every dataset, so nothing is dropped."""
    if not paired:
        return df
    pc, rep = paired_complete(df)
    if len(pc):
        print(f"[{tag}] paired: kept {rep['kept_rows']} rows across {rep['samplers']} samplers "
              f"(dropped {rep['dropped_rows']} unpaired)")
        return pc
    print(f"[{tag}] WARNING: no (dataset,element) cell is covered by all samplers — sparse/test "
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
    color : optional fill grouping (e.g. 'n_chains') drawn as dodged boxes.
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
        mapping = aes(x=x, y=value, fill=color)
    elif x == "sampler":
        # Fill boxes by sampler when sampler is on the x-axis (no explicit color grouping).
        mapping = aes(x=x, y=value, fill=x)

    p = ggplot(data, mapping)
    if hline is not None:
        p = p + geom_hline(yintercept=hline, linetype="dashed", color="#7f7f7f")
    p = p + geom_boxplot(outlier_size=0.6, outlier_alpha=(0.0 if jitter else 0.4))
    if jitter:  # show the raw element-points (recommended at modest n per box)
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
        p = p + scale_fill_manual(values=s_vals, breaks=list(s_order), labels=s_labs)

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

    counts = sub.groupby("sampler")["data_seed"].nunique().to_dict()
    print(f"[delta_bias_faceted_by_element] n_chains={n_chains} k_true={k_true}: "
          f"seeds/sampler={counts}")

    color_vals = [SAMPLER_COLORS[s] for s in sampler_order]
    n_comp = f"{k_true} True Component" + ("s" if k_true != 1 else "")

    # Draw jitter first (behind box outline), then transparent box on top so both are visible.
    p = (
        ggplot(sub, aes(x="sampler", y="bias", color="sampler"))
        + geom_hline(yintercept=0, linetype="dashed", color="#aaaaaa")
        + geom_jitter(width=0.2, height=0, size=0.8, alpha=0.45)
        + geom_boxplot(fill="#FFFFFF00", outlier_alpha=0)
        + facet_wrap("element", ncol=4, scales="free_y", labeller="label_value")
        + scale_color_manual(values=color_vals, labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
        + scale_x_discrete(labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
        + labs(
            x="Sampler",
            y="Empirical Bias",
            color="Sampler",
            title=f"Empirical Bias of Δ - {n_comp}",
        )
        + theme_bw()
        + theme(
            figure_size=(14, 7),
            axis_text_x=element_text(size=8),
            plot_title=element_text(size=11),
        )
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

    counts = sub.groupby("sampler")["data_seed"].nunique().to_dict()
    print(f"[delta_sd_faceted_by_element] n_chains={n_chains} k_true={k_true}: "
          f"seeds/sampler={counts}")

    color_vals = [SAMPLER_COLORS[s] for s in sampler_order]
    n_comp = f"{k_true} True Component" + ("s" if k_true != 1 else "")

    p = (
        ggplot(sub, aes(x="sampler", y="post_std", color="sampler"))
        + geom_jitter(width=0.2, height=0, size=0.8, alpha=0.45)
        + geom_boxplot(fill="#FFFFFF00", outlier_alpha=0)
        + facet_wrap("element", ncol=4, scales="free_y", labeller="label_value")
        + scale_color_manual(values=color_vals, labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
        + scale_x_discrete(labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
        + labs(
            x="Sampler",
            y="Posterior SD",
            color="Sampler",
            title=f"Posterior SD of Δ - {n_comp}",
        )
        + theme_bw()
        + theme(
            figure_size=(14, 7),
            axis_text_x=element_text(size=8),
            plot_title=element_text(size=11),
        )
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

    counts = sub.groupby("sampler")["data_seed"].nunique().to_dict()
    print(f"[delta_rmse_faceted_by_element] n_chains={n_chains} k_true={k_true}: "
          f"seeds/sampler={counts}")

    color_vals = [SAMPLER_COLORS[s] for s in sampler_order]
    n_comp = f"{k_true} True Component" + ("s" if k_true != 1 else "")

    p = (
        ggplot(sub, aes(x="sampler", y="abs_error", color="sampler"))
        + geom_jitter(width=0.2, height=0, size=0.8, alpha=0.45)
        + geom_boxplot(fill="#FFFFFF00", outlier_alpha=0)
        + facet_wrap("element", ncol=4, scales="free_y", labeller="label_value")
        + scale_color_manual(values=color_vals, labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
        + scale_x_discrete(labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
        + labs(
            x="Sampler",
            y="|post_mean - true_value|",
            color="Sampler",
            title=f"Absolute Error of Δ - {n_comp}",
        )
        + theme_bw()
        + theme(
            figure_size=(14, 7),
            axis_text_x=element_text(size=8),
            plot_title=element_text(size=11),
        )
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
          f"{sub.groupby('sampler')['runtime_s'].count().to_dict()}")
    return recovery_boxplot(
        df, value="runtime_s", x="k_true", color="sampler", filters=filters,
        x_order=[1, 2, 3, 5], hline=None, jitter=jitter, logy=logy,
        title=f"Runtime by true-component count, by sampler (n_chains={n_chains})",
        xlab="k_true  (true number of mixture components)",
        ylab="Runtime (s, log scale)" if logy else "Runtime (s)", figure_size=(9.0, 5.5),
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
          f"{sub.groupby('k_true')['runtime_s'].count().to_dict()}")

    color = SAMPLER_COLORS.get(sampler, "#888888")
    label = SAMPLER_LABELS.get(sampler, sampler)

    p = ggplot(sub, aes(x="k_true", y="runtime"))
    if jitter:
        p = p + geom_jitter(width=0.18, height=0, size=0.8, alpha=0.5, color=color)
    p = (p
         + geom_boxplot(fill="#FFFFFF00", color=color, outlier_alpha=0)
         + labs(
             x="Number of mixture components",
             y="Runtime (min.)",
             title=f"Runtime by Number of mixture components - {label}",
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
    ax2.set_ylabel("Runtime (h.)", color="black")
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
        title="Runtime by sampler, per true-component count  (n_chains = 1)",
        ylab="Runtime (s, log scale)" if logy else "Runtime (s)", xlab="Sampler",
        figure_size=(8.5, 6.0),
    )


def main() -> None:
    print("== Runtime by sampler, per k_true ==")
    p_rt = runtime_plot()
    print("wrote", save(p_rt, "runtime_by_sampler_ktrue.png"))


if __name__ == "__main__":
    main()
