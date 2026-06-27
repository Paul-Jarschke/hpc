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
    scale_y_log10,
    theme,
    theme_bw,
)

# ----------------------------------------------------------------------------- #
# Locations & study-wide conventions (single source of truth for every plot).
# ----------------------------------------------------------------------------- #
REPO = Path(__file__).resolve().parent.parent
DIR_RECOVERY = REPO / "data" / "out" / "k5model_mixture"
DIR_FIG = REPO / "analysis" / "out"

# Fixed sampler order/labels so EVERY figure in the study is consistent.
SAMPLER_ORDER = ["bayesm", "nuts", "hmc"]
SAMPLER_LABELS = {"bayesm": "bayesm", "nuts": "NUTS", "hmc": "HMC"}

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

    p = ggplot(data, mapping)
    if hline is not None:
        p = p + geom_hline(yintercept=hline, linetype="dashed", color="#7f7f7f")
    p = p + geom_boxplot(outlier_size=0.6, outlier_alpha=(0.0 if jitter else 0.4))
    if jitter:  # show the raw element-points (recommended at modest n per box)
        p = p + geom_jitter(width=0.18, height=0.0, size=0.8, alpha=0.5, color="#444444")
    if logy:  # runtime / ESS etc. span orders of magnitude
        p = p + scale_y_log10()

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
            from plotnine import scale_x_discrete

            p = p + scale_x_discrete(labels=[x_labels.get(c, c) for c in cats])

    p = p + theme_bw() + theme(
        figure_size=figure_size,
        axis_text_x=element_text(rotation=0),
        plot_title=element_text(size=11),
    )
    return p


def save(plot: ggplot, filename: str, dir_fig: Path = DIR_FIG, dpi: int = 150) -> Path:
    """Save a plot to analysis/out/ (created if missing). Returns the path."""
    dir_fig.mkdir(parents=True, exist_ok=True)
    out = dir_fig / filename
    plot.save(out, dpi=dpi, verbose=False)
    return out


# ----------------------------------------------------------------------------- #
# The two figures the user asked for, each a single call to the core.
# ----------------------------------------------------------------------------- #
def delta_bias_plot(df: Optional[pd.DataFrame] = None, *, jitter: bool = True,
                    paired: bool = True) -> ggplot:
    """(a) BIAS of the Z-covariate (Delta) estimates, n_chains==1, by sampler, with ONE
    panel per true-component count k_true (1/2/3/5). Dashed line at 0 = unbiased; each box
    pools the D*P=8 Delta elements over that k_true's datasets (paired across samplers)."""
    df = load_recovery("delta") if df is None else df
    d = _maybe_paired(_apply_filters(df, {"n_chains": 1}), paired, "delta_bias")
    _print_box_counts(d)
    return recovery_boxplot(
        d, value="bias", x="sampler", facet_wrap_by="k_true", facet_scales="fixed",
        hline=0.0, jitter=jitter,
        title="Δ (Z-covariate) bias by sampler, per true-component count  (n_chains = 1)",
        ylab="Bias  (post_mean − true_value)", xlab="Sampler", figure_size=(8.5, 6.0),
    )


def delta_sd_plot(df: Optional[pd.DataFrame] = None, *, jitter: bool = True,
                  paired: bool = True) -> ggplot:
    """(b) Posterior SD (post_std) of the Z-covariate (Delta) estimates, n_chains==1, by
    sampler, one panel per true-component count k_true (1/2/3/5)."""
    df = load_recovery("delta") if df is None else df
    d = _maybe_paired(_apply_filters(df, {"n_chains": 1}), paired, "delta_sd")
    _print_box_counts(d)
    return recovery_boxplot(
        d, value="post_std", x="sampler", facet_wrap_by="k_true", facet_scales="fixed",
        jitter=jitter,
        title="Δ (Z-covariate) posterior SD by sampler, per true-component count  (n_chains = 1)",
        ylab="Posterior SD  (post_std)", xlab="Sampler", figure_size=(8.5, 6.0),
    )


def delta_bias_across_seeds(sampler: str = "nuts", n_chains: int = 1, k_true: int = 1,
                            df: Optional[pd.DataFrame] = None, *, jitter: bool = True) -> ggplot:
    """Boxplot of Delta bias per element ACROSS SEEDS for ONE model and one k_true.

    Filters delta_recovery.csv to a single (sampler, n_chains, k_true) - e.g. NUTS, 1 chain,
    1 true component - and draws one box per Delta element (demographic : parameter). Each
    box pools that element's bias = post_mean - true_value over every replicate dataset
    (data_seed), so it is the sampling distribution of the bias for that element across
    datasets. Dashed line at 0 = unbiased. Needs the real multi-seed runs gathered by
    post_process; with only one seed present each "box" is a single point.
    """
    df = load_recovery("delta") if df is None else df
    df = df.copy()
    df["element"] = df["demo"].astype(str) + " : " + df["param"].astype(str)
    filters = {"sampler": sampler, "n_chains": n_chains, "k_true": k_true}
    sub = _apply_filters(df, filters)
    if sub.empty:
        raise ValueError(f"No delta_recovery rows for sampler={sampler}, n_chains={n_chains}, "
                         f"k_true={k_true}. Have the runs been downloaded + gathered?")
    n_seeds = sub["data_seed"].nunique()
    order = sub.sort_values(["demo", "param"])["element"].drop_duplicates().tolist()
    print(f"[delta_bias_across_seeds] {sampler} c{n_chains} k_true={k_true}: "
          f"{n_seeds} seed(s), {len(sub)} element-rows ({len(order)} elements)")
    return recovery_boxplot(
        df, value="bias", x="element", filters=filters, x_order=order,
        hline=0.0, jitter=jitter,
        title=f"Δ bias across {n_seeds} seed(s) - {SAMPLER_LABELS.get(sampler, sampler)}, "
              f"k_true={k_true} (n_chains={n_chains})",
        xlab="Δ element  (demographic : parameter)",
        ylab="Bias  (post_mean - true_value)", figure_size=(9.0, 5.0),
    )


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
    delta = load_recovery("delta")
    print("== Delta bias by sampler, per k_true ==")
    p_bias = delta_bias_plot(delta)
    print("== Delta posterior SD by sampler, per k_true ==")
    p_sd = delta_sd_plot(delta)
    print("== Runtime by sampler, per k_true ==")
    p_rt = runtime_plot()
    print("wrote", save(p_bias, "delta_bias_by_sampler_ktrue.png"))
    print("wrote", save(p_sd, "delta_sd_by_sampler_ktrue.png"))
    print("wrote", save(p_rt, "runtime_by_sampler_ktrue.png"))


if __name__ == "__main__":
    main()
