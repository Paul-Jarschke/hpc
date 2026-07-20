# Recovery boxplots for the standard model (jobs 200-202). Reads the tidy
# tables in data/out/standard_model/ (gather_summaries or post_process).
# k_true == k_model == 1 here, so the mixture pipeline's per-k_true loops
# all collapse to one figure each.
# run: .venv/Scripts/python.exe hpc_analysis/standard_model/plot_recovery.py
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

# Canonical sampler order/labels/colors - same palette as the mixture
# pipeline so every figure in the study matches (bayesm red, nuts/hmc blues).
SAMPLER_ORDER = ["bayesm", "nuts", "hmc"]
SAMPLER_LABELS = {"bayesm": "bayesm", "nuts": "NUTS", "hmc": "HMC"}
SAMPLER_COLORS = {"nuts": "#08519c", "hmc": "#4292c6", "iwls": "#9ecae1", "bayesm": "#d62728"}
TRUE_COLOR = "#000000"

# Element label helpers (single source of truth for the Δ_{d,p} and Σ_{i,j} notation).
_DEMO_ORDER = ["z1", "z2"]
_PARAM_ORDER = ["Alt1", "Alt2", "Alt3", "Price"]
_SUBSCRIPTS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


# 'Δ₁,₁ (z1:Alt1)' style; 1-based (demo, param) index, '?' if unknown
def delta_element_label(demo: str, param: str) -> str:
    d = (_DEMO_ORDER.index(demo) + 1) if demo in _DEMO_ORDER else "?"
    p = (_PARAM_ORDER.index(param) + 1) if param in _PARAM_ORDER else "?"
    return f"Δ{str(d).translate(_SUBSCRIPTS)},{str(p).translate(_SUBSCRIPTS)} ({demo}:{param})"


# 'Σ₁,₁ (Alt1:Alt1)' style; 1-based (row, col) index, '?' if unknown
def sigma_element_label(row: str, col: str) -> str:
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
# adds bias = post_mean - true_value where the table stores both cols
# (beta_recovery already ships its own bias/rmse, returned unchanged)
def load_recovery(table: str, dir_recovery: Optional[Path] = None) -> pd.DataFrame:
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


# paired design: keep cells covered by every sampler -> (kept_df, report)
def paired_complete(df: pd.DataFrame, key=("dataset_key", "demo", "param"), by="sampler"):
    key = [k for k in key if k in df.columns]
    nby = df[by].nunique()
    keep = df.groupby(key)[by].transform("nunique") == nby
    return df[keep], {"samplers": nby, "kept_rows": int(keep.sum()), "dropped_rows": int((~keep).sum())}


# falls back to unpaired (loud print) if pairing would empty the frame -
# only happens on sparse local test data, never on the real grid
def _maybe_paired(df: pd.DataFrame, paired: bool, tag: str) -> pd.DataFrame:
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
# plotnine on purpose (study standard): new variants = arg changes, not code.
# each box pools every element-row surviving `filters` for its x/facet cell.
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
# 4x2 element grid, free y per panel so one wild element can't compress
# the rest; transparent boxes + jitter colored by sampler
def delta_bias_faceted_by_element(n_chains: int = 2, df=None, *, jitter: bool = True) -> ggplot:
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
         + labs(x="Sampler", y="Empirical bias", color="Sampler",
                title="Bias of Δ - Standard Model")
         + theme_bw()
         + theme(figure_size=(14, 7), axis_text_x=element_text(size=8),
                 plot_title=element_text(size=11))
    )
    return p


# same layout as the bias grid, y = post_std (how tightly each sampler
# pins down each element)
def delta_sd_faceted_by_element(n_chains: int = 2, df=None, *, jitter: bool = True) -> ggplot:
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


# y = squared error per seed; box MEAN over seeds = the MSE reported
# in delta_bias_mse_table
def delta_mse_faceted_by_element(n_chains: int = 2, df=None, *, jitter: bool = True) -> ggplot:
    df = load_recovery("delta") if df is None else df
    df = df.copy()
    df["element"] = df.apply(lambda r: delta_element_label(r["demo"], r["param"]), axis=1)
    df["sq_error"] = df["bias"] ** 2
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
    print(f"[delta_mse_faceted_by_element] n_chains={n_chains}: seeds/sampler={counts}")

    color_vals = [SAMPLER_COLORS[s] for s in sampler_order]

    p = ggplot(sub, aes(x="sampler", y="sq_error", color="sampler"))
    if jitter:
        p = p + geom_jitter(width=0.2, height=0, size=0.8, alpha=0.45)
    p = (p
         + geom_boxplot(fill="#FFFFFF00", outlier_alpha=0)
         + facet_wrap("element", ncol=4, scales="free_y", labeller="label_value")
         + scale_color_manual(values=color_vals,
                              labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
         + scale_x_discrete(labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
         + labs(x="Sampler", y="Mean Squared Error (MSE)", color="Sampler",
                title="Squared error of Δ - Standard Model")
         + theme_bw()
         + theme(figure_size=(14, 7), axis_text_x=element_text(size=8),
                 plot_title=element_text(size=11))
    )
    return p


# ----------------------------------------------------------------------------- #
# Shared per-parameter boxplot core (mu uses the 1x4 layout).
# ----------------------------------------------------------------------------- #
# hline (e.g. 0 for bias) drawn behind the jitter/box so both stay visible
def _param_boxplot(df: pd.DataFrame, y_col: str, y_label: str,
                   title: str, sampler_order: list, *, jitter: bool,
                   hline: Optional[float] = None) -> ggplot:
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
# Mu recovery (NEW for the standard model: the population mean vs TRUE_MU is a
# plain 4-vector at K = 1, so per-parameter bias is directly meaningful).
# ----------------------------------------------------------------------------- #
# numeric form of the analysis notebook's summarize_mu comparison
def mu_bias_by_param(n_chains: int = 2, df=None, *, jitter: bool = True) -> ggplot:
    df = load_recovery("mu") if df is None else df
    sub = _apply_filters(df, {"n_chains": n_chains}).copy()
    if sub.empty:
        raise ValueError(f"No mu_recovery rows for n_chains={n_chains}.")
    sampler_order = [s for s in SAMPLER_ORDER if s in set(sub["sampler"])]
    sub["sampler"] = pd.Categorical(sub["sampler"], categories=sampler_order, ordered=True)
    counts = sub.groupby("sampler", observed=True)["data_seed"].nunique().to_dict()
    print(f"[mu_bias_by_param] n_chains={n_chains}: seeds/sampler={counts}")
    return _param_boxplot(sub, "bias", "Empirical bias",
                          "Bias of μ - Standard Model", sampler_order, jitter=jitter)


# box MEAN over seeds = the MSE reported in mu_recovery_summary_table
def mu_mse_by_param(n_chains: int = 2, df=None, *, jitter: bool = True) -> ggplot:
    df = load_recovery("mu") if df is None else df
    sub = _apply_filters(df, {"n_chains": n_chains}).copy()
    if sub.empty:
        raise ValueError(f"No mu_recovery rows for n_chains={n_chains}.")
    sub["sq_error"] = sub["bias"] ** 2
    sampler_order = [s for s in SAMPLER_ORDER if s in set(sub["sampler"])]
    sub["sampler"] = pd.Categorical(sub["sampler"], categories=sampler_order, ordered=True)
    counts = sub.groupby("sampler", observed=True)["data_seed"].nunique().to_dict()
    print(f"[mu_mse_by_param] n_chains={n_chains}: seeds/sampler={counts}")
    return _param_boxplot(sub, "sq_error", "Mean Squared Error (MSE)",
                          "Squared error of μ - Standard Model", sampler_order, jitter=jitter)


# ----------------------------------------------------------------------------- #
# Sigma recovery (NEW for the standard model: the POSTERIOR SIGMA analysis - every
# lower-triangle element of the heterogeneity covariance vs TRUE_SIGMA; the
# `empirical` reference column - cov of the TRUE unit betas - lives in the
# make_tables summary, matching the notebook's plot_final_covariance_complete).
# ----------------------------------------------------------------------------- #
# facet_grid(row, col) lays panels out as the LOWER TRIANGLE of the PxP
# covariance matrix; upper triangle stays blank
def sigma_bias_faceted_by_element(n_chains: int = 2, df=None, *, jitter: bool = True) -> ggplot:
    df = load_recovery("sigma") if df is None else df
    df = df.copy()
    sub = _apply_filters(df, {"n_chains": n_chains}).copy()
    if sub.empty:
        raise ValueError(f"No sigma_recovery rows for n_chains={n_chains}.")

    sampler_order = [s for s in SAMPLER_ORDER if s in set(sub["sampler"])]
    sub["sampler"] = pd.Categorical(sub["sampler"], categories=sampler_order, ordered=True)
    sub["row"] = pd.Categorical(sub["row"], categories=_PARAM_ORDER, ordered=True)
    sub["col"] = pd.Categorical(sub["col"], categories=_PARAM_ORDER, ordered=True)

    counts = sub.groupby("sampler", observed=True)["data_seed"].nunique().to_dict()
    print(f"[sigma_bias_faceted_by_element] n_chains={n_chains}: seeds/sampler={counts}")

    color_vals = [SAMPLER_COLORS[s] for s in sampler_order]

    p = ggplot(sub, aes(x="sampler", y="bias", color="sampler"))
    p = p + geom_hline(yintercept=0, linetype="dashed", color="#aaaaaa")
    if jitter:
        p = p + geom_jitter(width=0.2, height=0, size=0.8, alpha=0.45)
    p = (p
         + geom_boxplot(fill="#FFFFFF00", outlier_alpha=0)
         + facet_grid(rows="row", cols="col", scales="free_y", labeller="label_value")
         + scale_color_manual(values=color_vals, labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
         + scale_x_discrete(labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
         + labs(x="Sampler", y="Empirical bias", color="Sampler",
                title="Error of Σ Elements (lower triangle) - Standard Model")
         + theme_bw()
         + theme(figure_size=(12, 10), axis_text_x=element_text(size=8),
                 plot_title=element_text(size=11))
    )
    return p


# same lower-triangle layout; box MEAN over seeds = the MSE reported
# in sigma_recovery_summary_table
def sigma_mse_faceted_by_element(n_chains: int = 2, df=None, *, jitter: bool = True) -> ggplot:
    df = load_recovery("sigma") if df is None else df
    df = df.copy()
    df["sq_error"] = df["bias"] ** 2
    sub = _apply_filters(df, {"n_chains": n_chains}).copy()
    if sub.empty:
        raise ValueError(f"No sigma_recovery rows for n_chains={n_chains}.")

    sampler_order = [s for s in SAMPLER_ORDER if s in set(sub["sampler"])]
    sub["sampler"] = pd.Categorical(sub["sampler"], categories=sampler_order, ordered=True)
    sub["row"] = pd.Categorical(sub["row"], categories=_PARAM_ORDER, ordered=True)
    sub["col"] = pd.Categorical(sub["col"], categories=_PARAM_ORDER, ordered=True)

    counts = sub.groupby("sampler", observed=True)["data_seed"].nunique().to_dict()
    print(f"[sigma_mse_faceted_by_element] n_chains={n_chains}: seeds/sampler={counts}")

    color_vals = [SAMPLER_COLORS[s] for s in sampler_order]

    p = ggplot(sub, aes(x="sampler", y="sq_error", color="sampler"))
    if jitter:
        p = p + geom_jitter(width=0.2, height=0, size=0.8, alpha=0.45)
    p = (p
         + geom_boxplot(fill="#FFFFFF00", outlier_alpha=0)
         + facet_grid(rows="row", cols="col", scales="free_y", labeller="label_value")
         + scale_color_manual(values=color_vals, labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
         + scale_x_discrete(labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
         + labs(x="Sampler", y="Mean Squared Error (MSE)", color="Sampler",
                title="Squared error of Σ Elements (lower triangle) - Standard Model")
         + theme_bw()
         + theme(figure_size=(12, 10), axis_text_x=element_text(size=8),
                 plot_title=element_text(size=11))
    )
    return p


# ----------------------------------------------------------------------------- #
# Marginal-density distances vs the true DGP.
# ----------------------------------------------------------------------------- #
# which sampler's fitted marginal sits closest to the true DGP, per param;
# non-finite dropped (KL=inf in deep tails), count printed
def marginal_metric_boxplot(metric: str = "Hellinger", n_chains: int = 2,
                            df=None, *, grid: str = "chebyshev",
                            jitter: bool = True) -> ggplot:
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


# realised mass inside the grid window (mc.retained_mass); dashed 0.96 =
# Chebyshev bound at k=5 (1 - 1/5**2). 'full' grid trivially retains ~1
def retained_mass_boxplot(n_chains: int = 2, df=None, *, grid: str = "chebyshev",
                          jitter: bool = True) -> ggplot:
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


# seeds where KL(model||true) = +inf: model mass where true density ~0;
# the 'full' grid is far more prone to this than 'chebyshev'
def kl_inf_count_plot(n_chains: int = 2, df=None, *, grid: str = "chebyshev") -> ggplot:
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


# all metrics x params in one grid - quick check the metrics agree on order
def marginal_distances_faceted_by_metric(n_chains: int = 2, df=None, *,
                                          grid: str = "chebyshev") -> ggplot:
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
# runtime_s from runs.csv -> minutes; linear y by default (samplers run
# on comparable scales here), logy=True if one blows up
def runtime_by_sampler(n_chains: int = 2, df: Optional[pd.DataFrame] = None, *,
                       logy: bool = False, jitter: bool = True) -> ggplot:
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


if __name__ == "__main__":
    main()
