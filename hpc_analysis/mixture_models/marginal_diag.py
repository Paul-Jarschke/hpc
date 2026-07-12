"""
Convergence diagnostics of the marginal comparison, mixture_c2.

Source: data/out/mixture_c2/marginal_diagnostics.csv (src.summaries ->
marginal_comparison.functional_diagnostics). Every quantity here is LABEL-INVARIANT, so
relabeling never touches it - these are Goose-identical arviz diagnostics (the exact
az.rhat / az.ess calls liesel.goose.summary_m makes) of grid-free scalar FUNCTIONALS of
each per-draw marginal (Rossi Eq. 5.5.19): the marginal mean, sd and the q05/q50/q95
quantiles. There is one row per (param, functional):

  * Rhat            - rank-normalised split-R-hat (>= 2 chains; c2 only).
  * ESS_bulk        - bulk effective sample size (central mixing).
  * ESS_tail        - tail effective sample size (5%/95% quantile mixing).
  * ESS_bulk/s,
    ESS_tail/s      - effective draws per fit-second (ESS / total wall-clock incl. warmup,
                      from meta.json runtime_s) - the cross-sampler EFFICIENCY metric.
                      Renamed ESS_bulk_per_s / ESS_tail_per_s on load ('/' would parse as
                      a division inside plotnine aes()).

The old grid-based density-series / moment-series ESS/R-hat (min_ESS, max_Rhat, kind,
grid columns) were replaced upstream by these functional diagnostics; this module reads
the new schema only. Read ESS only where R-hat ~ 1.

Thresholds: R-hat <= 1.1 is the study-wide convergence gate (label_switching.classify_
outcome); ESS >= 400 is the rule-of-thumb target for a stable functional estimate. Both
are drawn as reference lines and reported as pass-rates. c2 (2-chain) runs only.

Run from the repo root with the project venv:
    .venv/Scripts/python.exe hpc_analysis/mixture_models/marginal_diag.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from plotnine import (
    aes,
    element_text,
    facet_grid,
    geom_boxplot,
    geom_hline,
    ggplot,
    labs,
    position_dodge,
    scale_color_manual,
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

# The scalar functionals of each per-draw marginal, and readable labels.
FUNC_ORDER = ["mean", "sd", "q05", "q50", "q95"]
FUNC_LABELS = {"mean": "Mean", "sd": "SD", "q05": "Q05", "q50": "Median", "q95": "Q95"}

# Metric -> (axis label, log scale?, reference line). ESS/s carries no fixed target
# (its scale is sampler-dependent), so no reference line.
METRIC_CFG = {
    "Rhat":           ("R-hat", False, RHAT_THRESH, "R-hat"),
    "ESS_bulk":       ("ESS (bulk, log scale)", True, ESS_MIN, "Bulk ESS"),
    "ESS_tail":       ("ESS (tail, log scale)", True, ESS_MIN, "Tail ESS"),
    "ESS_bulk_per_s": ("ESS/s (bulk, log scale)", True, None, "Bulk ESS per second"),
    "ESS_tail_per_s": ("ESS/s (tail, log scale)", True, None, "Tail ESS per second"),
}

DIR_OUT_BASE = DIR_FIG / "marginal_comparison"


# --------------------------------------------------------------------------------- #
# Load.
# --------------------------------------------------------------------------------- #
def load_diag(n_chains: int = 2, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Return the functional-diagnostics frame for the given chain count, with the
    ESS-per-second columns renamed to plotnine-safe identifiers ('/' parses as a
    division inside aes())."""
    d = load_recovery("marginal_diagnostics") if df is None else df.copy()
    d = d.rename(columns={"ESS_bulk/s": "ESS_bulk_per_s", "ESS_tail/s": "ESS_tail_per_s"})
    d = d[d["n_chains"] == n_chains].copy()
    if d.empty:
        raise ValueError(f"No marginal_diagnostics rows for n_chains={n_chains}.")
    return d


def _prep(d: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Apply canonical ordered categoricals (sampler, k_true, functional, param)."""
    sampler_order = [s for s in SAMPLER_ORDER if s in set(d["sampler"])]
    func_order = [f for f in FUNC_ORDER if f in set(d["functional"])]
    param_order = [p for p in _PARAM_ORDER if p in set(d["param"])]
    ktrue_order = [str(k) for k in KTRUE_ORDER if k in set(d["k_true"])]
    d = d.copy()
    d["sampler"] = pd.Categorical(d["sampler"], categories=sampler_order, ordered=True)
    d["functional"] = pd.Categorical(d["functional"].map(FUNC_LABELS),
                                     categories=[FUNC_LABELS[f] for f in func_order], ordered=True)
    d["param"] = pd.Categorical(d["param"], categories=param_order, ordered=True)
    d["k_true"] = pd.Categorical(d["k_true"].astype(str), categories=ktrue_order, ordered=True)
    return d, sampler_order


# --------------------------------------------------------------------------------- #
# Plots: x = k_true, dodged box per sampler, facet_grid(functional x param).
# --------------------------------------------------------------------------------- #
def _diag_grid(d: pd.DataFrame, metric: str, n_chains: int) -> ggplot:
    ylab, logscale, hline, mtitle = METRIC_CFG[metric]
    d = d.dropna(subset=[metric])
    d, sampler_order = _prep(d)
    counts = d.groupby(["k_true", "sampler"], observed=True)["data_seed"].nunique().to_dict()
    print(f"[marginal_diag:{metric}] c{n_chains}: seeds/box={counts}")
    color_vals = [SAMPLER_COLORS[s] for s in sampler_order]
    dodge = position_dodge(width=0.9)

    p = (
        ggplot(d, aes(x="k_true", y=metric, color="sampler"))
        + geom_boxplot(width=0.6, fill="#FFFFFF00", outlier_alpha=0.25, position=dodge)
    )
    if logscale:
        p = p + scale_y_log10()
    if hline is not None:
        p = p + geom_hline(yintercept=hline, linetype="dashed", color="#555555", size=0.7)
    return (
        p + facet_grid(rows="functional", cols="param", scales="free_y", labeller="label_value")
        + scale_color_manual(values=color_vals,
                             labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
        + labs(x="True Number of components", y=ylab, color="Sampler",
               title=f"Marginal {mtitle} by True Components  (c{n_chains})")
        + theme_bw()
        + theme(figure_size=(13, 11), axis_text_x=element_text(size=8),
                plot_title=element_text(size=11))
    )


def plot_rhat_grid(n_chains: int = 2, df: Optional[pd.DataFrame] = None) -> ggplot:
    """R-hat of every marginal functional (mean/sd/q05/q50/q95) x 4 params, by k_true."""
    return _diag_grid(load_diag(n_chains, df=df), "Rhat", n_chains)


def plot_ess_bulk_grid(n_chains: int = 2, df: Optional[pd.DataFrame] = None) -> ggplot:
    """Bulk ESS (log scale) of every marginal functional x 4 params, by k_true."""
    return _diag_grid(load_diag(n_chains, df=df), "ESS_bulk", n_chains)


def plot_ess_tail_grid(n_chains: int = 2, df: Optional[pd.DataFrame] = None) -> ggplot:
    """Tail ESS (log scale) of every marginal functional x 4 params, by k_true."""
    return _diag_grid(load_diag(n_chains, df=df), "ESS_tail", n_chains)


def plot_ess_bulk_per_s_grid(n_chains: int = 2, df: Optional[pd.DataFrame] = None) -> ggplot:
    """Bulk ESS per fit-second (log scale) - the cross-sampler efficiency metric."""
    return _diag_grid(load_diag(n_chains, df=df), "ESS_bulk_per_s", n_chains)


def plot_ess_tail_per_s_grid(n_chains: int = 2, df: Optional[pd.DataFrame] = None) -> ggplot:
    """Tail ESS per fit-second (log scale) - the cross-sampler efficiency metric."""
    return _diag_grid(load_diag(n_chains, df=df), "ESS_tail_per_s", n_chains)


# --------------------------------------------------------------------------------- #
# Tables.
# --------------------------------------------------------------------------------- #
def rhat_summary_table(n_chains: int = 2, d: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Per (k_true, functional, param, sampler): median / q75 / max R-hat and the
    convergence pass-rate frac(R-hat <= 1.1)."""
    d = load_diag(n_chains) if d is None else d
    g = d.dropna(subset=["Rhat"]).groupby(
        ["k_true", "functional", "param", "sampler"], observed=True)["Rhat"]
    out = g.agg(median_rhat="median", q75_rhat=lambda s: s.quantile(0.75), max_rhat="max",
                frac_converged=lambda s: (s <= RHAT_THRESH).mean(), n_sim="size").reset_index()
    return _finish(out, ["median_rhat", "q75_rhat", "max_rhat", "frac_converged"])


def ess_summary_table(n_chains: int = 2, d: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Per (k_true, functional, param, sampler): bulk/tail ESS medians and the frac(ESS_bulk
    >= 400) pass-rate, plus the median bulk/tail ESS per fit-second (efficiency)."""
    d = load_diag(n_chains) if d is None else d
    g = d.groupby(["k_true", "functional", "param", "sampler"], observed=True)
    out = g.agg(median_ess_bulk=("ESS_bulk", "median"),
                median_ess_tail=("ESS_tail", "median"),
                frac_ess_bulk_ge_400=("ESS_bulk", lambda s: (s >= ESS_MIN).mean()),
                median_ess_bulk_per_s=("ESS_bulk_per_s", "median"),
                median_ess_tail_per_s=("ESS_tail_per_s", "median"),
                n_sim=("ESS_bulk", "size")).reset_index()
    return _finish(out, ["median_ess_bulk", "median_ess_tail", "frac_ess_bulk_ge_400",
                         "median_ess_bulk_per_s", "median_ess_tail_per_s"])


def _finish(out: pd.DataFrame, round_cols: list) -> pd.DataFrame:
    """Canonical sampler/functional/param ordering + rounding for a summary table."""
    sampler_order = [s for s in SAMPLER_ORDER if s in set(out["sampler"])]
    func_order = [f for f in FUNC_ORDER if f in set(out["functional"])]
    param_order = [p for p in _PARAM_ORDER if p in set(out["param"])]
    out["sampler"] = pd.Categorical(out["sampler"], categories=sampler_order, ordered=True)
    out["functional"] = pd.Categorical(out["functional"], categories=func_order, ordered=True)
    out["param"] = pd.Categorical(out["param"], categories=param_order, ordered=True)
    out[round_cols] = out[round_cols].round(3)
    return out.sort_values(["k_true", "functional", "param", "sampler"]).reset_index(drop=True)


# --------------------------------------------------------------------------------- #
# Entry points (public API used by make_plots.py / make_tables.py).
# --------------------------------------------------------------------------------- #
def write_tables(n_chains: int = 2) -> None:
    out_dir = DIR_OUT_BASE / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    d = load_diag(n_chains)
    for name, tbl in {
        f"marginal_rhat_summary_c{n_chains}.csv": rhat_summary_table(n_chains, d),
        f"marginal_ess_summary_c{n_chains}.csv":  ess_summary_table(n_chains, d),
    }.items():
        path = out_dir / name
        tbl.to_csv(path, index=False)
        print(f"wrote {len(tbl)} rows -> {path}")


def make_plots(n_chains: int = 2) -> None:
    plots = "marginal_comparison/plots"
    d = load_diag(n_chains)
    for metric in METRIC_CFG:
        print("wrote", save(_diag_grid(d, metric, n_chains),
                            f"{plots}/marginal_{metric.lower()}_grid_c{n_chains}.png"))


def main() -> None:
    write_tables()
    make_plots()
    print("marginal-diagnostics tables + figures -> hpc_analysis/mixture_models/out/marginal_comparison/")


if __name__ == "__main__":
    main()
