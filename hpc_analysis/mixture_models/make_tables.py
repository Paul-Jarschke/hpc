"""
Generate summary tables for the mixture_c2 sampler comparison (jobs 100-103).

Run after scripts/gather_summaries.py has refreshed data/out/mixture_c2/:
    .venv/Scripts/python.exe hpc_analysis/make_tables.py

Writes CSVs to hpc_analysis/mixture_models/out/<topic>/tables/. Marginal-distance tables are produced
once per evaluation grid ('full' and 'chebyshev'; filename suffix _<grid>).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_recovery import (  # noqa: E402
    load_recovery, DIR_FIG, delta_element_label, MARGINAL_METRICS,
)
import component_count  # noqa: E402
import marginal_diag  # noqa: E402

OUT_DIR_DELTA_BIAS     = DIR_FIG / "delta" / "bias"     / "tables"
OUT_DIR_DELTA_SD       = DIR_FIG / "delta" / "sd"       / "tables"
OUT_DIR_RUNTIME        = DIR_FIG / "runtime" / "tables"
OUT_DIR_MARGINAL         = DIR_FIG / "marginal_comparison"   # per-grid subfolder added below

_PARAM_ORDER = ["Alt1", "Alt2", "Alt3", "Price"]

CHAINS = 2
GRIDS = ["full", "chebyshev"]  # marginal-distance evaluation-grid scenarios
GRID_FOLDER = {"full": "full", "chebyshev": "trimmed"}  # output subfolder per grid
SAMPLER_ORDER = ["bayesm", "bayesm_gibbs", "nuts", "hmc"]
SAMPLER_LABELS = {"bayesm": "bayesm", "bayesm_gibbs": "Replication", "nuts": "NUTS", "hmc": "HMC"}


def _bias_mse_stats(d: pd.Series) -> dict:
    """Point estimates and Monte Carlo SEs of bias and MSE (Morris et al. 2019, Table 6),
    from the per-replication differences d_i = post_mean_i - true_value_i (each evaluated
    against the per-replication truth theta_i):

        bias = mean(d),    mcse_bias = std(d,   ddof=1) / sqrt(n_sim)
        mse  = mean(d^2),  mcse_mse  = std(d^2, ddof=1) / sqrt(n_sim)
    """
    d = np.asarray(d, dtype=float)
    n = d.size
    sq = d ** 2
    return {
        "bias":      d.mean(),
        "mcse_bias": d.std(ddof=1) / np.sqrt(n),
        "mse":       sq.mean(),
        "mcse_mse":  sq.std(ddof=1) / np.sqrt(n),
        "n_sim":     n,
    }


def runtime_summary_table(n_chains: int = CHAINS) -> pd.DataFrame:
    """Runtime summary statistics by sampler and k_true (in minutes).

    Returns one row per (k_true, sampler) with columns:
        min, q1 (25th pct), median, q3 (75th pct), max, n_runs
    """
    df = load_recovery("runs")
    df = df[df["n_chains"] == n_chains].copy()
    df["runtime_min"] = df["runtime_s"] / 60.0

    def _agg(g):
        r = g["runtime_min"]
        return pd.Series({
            "min":    r.min(),
            "q1":     r.quantile(0.25),
            "mean":   r.mean(),
            "median": r.median(),
            "q3":     r.quantile(0.75),
            "max":    r.max(),
            "n_runs": len(r),
        })

    agg = (
        df.groupby(["k_true", "sampler"])
        .apply(_agg, include_groups=False)
        .reset_index()
    )
    # Canonical sampler order and readable labels
    samplers_present = [s for s in SAMPLER_ORDER if s in agg["sampler"].unique()]
    agg["sampler"] = pd.Categorical(agg["sampler"], categories=samplers_present, ordered=True)
    agg["sampler_label"] = agg["sampler"].map(SAMPLER_LABELS)
    agg = agg.sort_values(["k_true", "sampler"])
    stat_cols = ["min", "q1", "mean", "median", "q3", "max"]
    agg[stat_cols] = agg[stat_cols].round(2)
    return agg[["k_true", "sampler_label", *stat_cols, "n_runs"]].rename(
        columns={"sampler_label": "sampler"}
    )


def delta_sd_summary_table(n_chains: int = CHAINS) -> pd.DataFrame:
    """Distribution of posterior SD for every Delta element, by sampler and k_true.

    For each (k_true, element, sampler) cell the statistics summarize the post_std
    values across all replicate seeds, i.e. how tightly (and consistently) each
    sampler pins down each element over repeated datasets.

    Returns a tidy long DataFrame:
        rows    = (k_true, element, sampler)
        columns = min, q1, mean, median, q3, max, n_sim
    """
    df = load_recovery("delta")
    df = df[df["n_chains"] == n_chains].copy()
    df["element"] = df.apply(lambda r: delta_element_label(r["demo"], r["param"]), axis=1)

    def _agg(g):
        s = g["post_std"]
        return pd.Series({
            "min":    s.min(),
            "q1":     s.quantile(0.25),
            "mean":   s.mean(),
            "median": s.median(),
            "q3":     s.quantile(0.75),
            "max":    s.max(),
            "n_sim":  len(s),
        })

    agg = (
        df.groupby(["k_true", "element", "sampler"])
        .apply(_agg, include_groups=False)
        .reset_index()
    )
    # Canonical sampler order and readable labels
    samplers_present = [s for s in SAMPLER_ORDER if s in agg["sampler"].unique()]
    agg["sampler"] = pd.Categorical(agg["sampler"], categories=samplers_present, ordered=True)
    agg["sampler"] = agg["sampler"].map(SAMPLER_LABELS)
    agg = agg.sort_values(["k_true", "element", "sampler"])
    stat_cols = ["min", "q1", "mean", "median", "q3", "max"]
    agg[stat_cols] = agg[stat_cols].round(4)
    return agg[["k_true", "element", "sampler", *stat_cols, "n_sim"]].reset_index(drop=True)


def delta_bias_mse_table(n_chains: int = CHAINS) -> pd.DataFrame:
    """Bias, MSE and their Monte Carlo SEs for every Delta element, by sampler and k_true.

    For each (sampler, k_true, element) cell across n_sim replicate seeds, from the
    per-seed differences d_i = post_mean_i - true_value_i (Morris et al. 2019, Table 6):
        bias      = mean(d_i)
        mcse_bias = std(d_i,   ddof=1) / sqrt(n_sim)
        mse       = mean(d_i^2)
        mcse_mse  = std(d_i^2, ddof=1) / sqrt(n_sim)

    Returns a wide DataFrame:
        rows    = (k_true, element)
        columns = {bias, mcse_bias, mse, mcse_mse, n_sim}_{sampler}
    """
    df = load_recovery("delta")
    df = df[df["n_chains"] == n_chains].copy()
    df["element"] = df.apply(lambda r: delta_element_label(r["demo"], r["param"]), axis=1)

    agg = (
        df.groupby(["k_true", "element", "sampler"])
        .apply(lambda g: pd.Series(_bias_mse_stats(g["bias"])), include_groups=False)
        .reset_index()
    )

    metrics = ["bias", "mcse_bias", "mse", "mcse_mse", "n_sim"]
    wide = agg.pivot_table(index=["k_true", "element"], columns="sampler", values=metrics)
    samplers = [s for s in SAMPLER_ORDER if s in agg["sampler"].unique()]
    wide = wide.reindex(columns=pd.MultiIndex.from_product([metrics, samplers]))
    wide = wide.round(6)
    wide.columns = [f"{metric}_{SAMPLER_LABELS.get(s, s)}" for metric, s in wide.columns]
    return wide.reset_index()


def marginal_distance_summary_table(n_chains: int = CHAINS,
                                    grid: str = "chebyshev") -> pd.DataFrame:
    """Distribution of all five marginal distance metrics, by sampler, k_true, and parameter.

    For each (k_true, param, sampler, metric) cell across n_sim replicate seeds, summarises
    the distribution of KL/TVD/Hellinger/JSD values: lower is better (fitted
    marginal closer to the true DGP marginal). Hellinger is bounded in [0,1]; the others are
    unbounded but comparable within a metric. `grid` selects the evaluation-grid scenario
    ('full' or 'chebyshev') the distances were computed on.

    Returns a tidy long DataFrame:
        rows    = (k_true, param, sampler, metric)
        columns = min, q1, mean, median, q3, max, n_sim
    """
    df = load_recovery("marginal_distances")
    if "grid" in df.columns:
        df = df[df["grid"] == grid]
    df = df[df["n_chains"] == n_chains].copy()
    metrics_present = [m for m in MARGINAL_METRICS if m in df.columns]
    id_cols = [c for c in ("k_true", "param", "sampler") if c in df.columns]
    long = df.melt(id_vars=id_cols, value_vars=metrics_present,
                   var_name="metric", value_name="distance")

    def _agg(g):
        d = g["distance"]
        return pd.Series({
            "min":    d.min(),
            "q1":     d.quantile(0.25),
            "mean":   d.mean(),
            "median": d.median(),
            "q3":     d.quantile(0.75),
            "max":    d.max(),
            "n_sim":  len(d),
        })

    agg = (
        long.groupby(["k_true", "param", "sampler", "metric"], observed=True)
        .apply(_agg, include_groups=False)
        .reset_index()
    )
    samplers_present = [s for s in SAMPLER_ORDER if s in agg["sampler"].unique()]
    agg["sampler"] = pd.Categorical(agg["sampler"], categories=samplers_present, ordered=True)
    agg["sampler_label"] = agg["sampler"].map(SAMPLER_LABELS)
    agg["metric"] = pd.Categorical(agg["metric"], categories=metrics_present, ordered=True)
    agg = agg.sort_values(["k_true", "param", "metric", "sampler"])
    stat_cols = ["min", "q1", "mean", "median", "q3", "max"]
    agg[stat_cols] = agg[stat_cols].round(5)
    return agg[["k_true", "param", "sampler_label", "metric", *stat_cols, "n_sim"]].rename(
        columns={"sampler_label": "sampler"}
    ).reset_index(drop=True)


def retained_mass_summary_table(n_chains: int = CHAINS, grid: str = "chebyshev") -> pd.DataFrame:
    """Distribution of retained_mass_model (mc.retained_mass), by sampler, k_true and
    parameter - the realised counterpart to the theoretical Chebyshev mass guarantee.

    For each (k_true, param, sampler) cell across n_sim replicate seeds, summarises the
    fraction of each fitted model's own marginal mass retained inside the evaluation-grid
    window. frac_below_guarantee = fraction of seeds where retained mass fell BELOW the
    theoretical minimum (k=5 -> 1 - 1/5**2 = 0.96) - should be ~0 if the Chebyshev fix is
    working as intended. `grid` selects the evaluation-grid scenario ('full' trivially
    retains ~100%; 'chebyshev' is the meaningful case).

    Returns a tidy long DataFrame:
        rows    = (k_true, param, sampler)
        columns = min, q1, mean, median, q3, max, frac_below_guarantee, n_sim
    """
    df = load_recovery("marginal_distances")
    if "grid" in df.columns:
        df = df[df["grid"] == grid]
    df = df[df["n_chains"] == n_chains].copy()
    if "retained_mass_model" not in df.columns:
        raise ValueError("Column 'retained_mass_model' not in marginal_distances - "
                         "re-gather data/out after the Chebyshev mass-guarantee fix.")

    def _agg(g):
        r = g["retained_mass_model"]
        return pd.Series({
            "min":    r.min(),
            "q1":     r.quantile(0.25),
            "mean":   r.mean(),
            "median": r.median(),
            "q3":     r.quantile(0.75),
            "max":    r.max(),
            "frac_below_guarantee": (r < 0.96).mean(),
            "n_sim":  len(r),
        })

    agg = (
        df.groupby(["k_true", "param", "sampler"], observed=True)
        .apply(_agg, include_groups=False)
        .reset_index()
    )
    samplers_present = [s for s in SAMPLER_ORDER if s in agg["sampler"].unique()]
    agg["sampler"] = pd.Categorical(agg["sampler"], categories=samplers_present, ordered=True)
    agg["sampler_label"] = agg["sampler"].map(SAMPLER_LABELS)
    agg = agg.sort_values(["k_true", "param", "sampler"])
    stat_cols = ["min", "q1", "mean", "median", "q3", "max", "frac_below_guarantee"]
    agg[stat_cols] = agg[stat_cols].round(5)
    return agg[["k_true", "param", "sampler_label", *stat_cols, "n_sim"]].rename(
        columns={"sampler_label": "sampler"}
    ).reset_index(drop=True)


def kl_inf_summary_table(n_chains: int = CHAINS, grid: str = "chebyshev") -> pd.DataFrame:
    """Count of seeds where KL(model||true) came back +inf, by sampler, k_true and
    parameter - a direct measure of catastrophic tail mismatch (the fitted marginal
    puts mass where the true DGP density is ~0). `grid` selects the evaluation-grid
    scenario ('full' is far more prone to this than the 'chebyshev'-trimmed grid).

    Returns a tidy long DataFrame:
        rows    = (k_true, param, sampler)
        columns = n_inf, n_total, inf_rate
    """
    df = load_recovery("marginal_distances")
    if "grid" in df.columns:
        df = df[df["grid"] == grid]
    df = df[df["n_chains"] == n_chains].copy()
    if "KL" not in df.columns:
        raise ValueError("Column 'KL' not in marginal_distances.")
    df["is_inf"] = ~np.isfinite(df["KL"])

    agg = (
        df.groupby(["k_true", "param", "sampler"], observed=True)["is_inf"]
        .agg(n_inf="sum", n_total="count").reset_index()
    )
    agg["inf_rate"] = (agg["n_inf"] / agg["n_total"]).round(4)
    samplers_present = [s for s in SAMPLER_ORDER if s in agg["sampler"].unique()]
    agg["sampler"] = pd.Categorical(agg["sampler"], categories=samplers_present, ordered=True)
    agg["sampler_label"] = agg["sampler"].map(SAMPLER_LABELS)
    agg = agg.sort_values(["k_true", "param", "sampler"])
    return agg[["k_true", "param", "sampler_label", "n_inf", "n_total", "inf_rate"]].rename(
        columns={"sampler_label": "sampler"}
    ).reset_index(drop=True)


def main():
    OUT_DIR_DELTA_BIAS.mkdir(parents=True, exist_ok=True)
    OUT_DIR_DELTA_SD.mkdir(parents=True, exist_ok=True)
    OUT_DIR_RUNTIME.mkdir(parents=True, exist_ok=True)

    # --- Delta bias + MSE tables (bias, MCSE(bias), MSE, MCSE(MSE) per sampler) ---
    tbl = delta_bias_mse_table()

    path = OUT_DIR_DELTA_BIAS / f"delta_bias_mse_c{CHAINS}_all.csv"
    tbl.to_csv(path, index=False)
    print(f"wrote {len(tbl)} rows -> {path}")

    for kt in sorted(tbl["k_true"].unique()):
        sub = tbl[tbl["k_true"] == kt].drop(columns="k_true")
        path = OUT_DIR_DELTA_BIAS / f"delta_bias_mse_c{CHAINS}_kt{int(kt)}.csv"
        sub.to_csv(path, index=False)
        print(f"wrote {len(sub)} rows -> {path}")

    # --- Delta posterior SD tables ---
    sd = delta_sd_summary_table()

    path = OUT_DIR_DELTA_SD / f"delta_sd_summary_c{CHAINS}_all.csv"
    sd.to_csv(path, index=False)
    print(f"wrote {len(sd)} rows -> {path}")

    for kt in sorted(sd["k_true"].unique()):
        sub = sd[sd["k_true"] == kt].drop(columns="k_true")
        path = OUT_DIR_DELTA_SD / f"delta_sd_summary_c{CHAINS}_kt{int(kt)}.csv"
        sub.to_csv(path, index=False)
        print(f"wrote {len(sub)} rows -> {path}")

    # --- Runtime summary table ---
    rt = runtime_summary_table()
    path = OUT_DIR_RUNTIME / f"runtime_summary_c{CHAINS}.csv"
    rt.to_csv(path, index=False)
    print(f"wrote {len(rt)} rows -> {path}")

    # --- Component-count tables (recovery summary, confusion, threshold sensitivity) ---
    component_count.write_tables(CHAINS)

    # --- Marginal-series convergence tables (R-hat + ESS summaries) ---
    marginal_diag.write_tables(CHAINS)

    # --- Marginal distance summary tables (one set per evaluation grid) ---
    for grid in GRIDS:
        out_dir = OUT_DIR_MARGINAL / GRID_FOLDER[grid] / "tables"
        out_dir.mkdir(parents=True, exist_ok=True)
        mdist = marginal_distance_summary_table(grid=grid)
        path = out_dir / f"marginal_distance_summary_c{CHAINS}_all.csv"
        mdist.to_csv(path, index=False)
        print(f"wrote {len(mdist)} rows -> {path}")
        for kt in sorted(mdist["k_true"].unique()):
            sub = mdist[mdist["k_true"] == kt].drop(columns="k_true")
            path = out_dir / f"marginal_distance_summary_c{CHAINS}_kt{int(kt)}.csv"
            sub.to_csv(path, index=False)
            print(f"wrote {len(sub)} rows -> {path}")

    # --- Retained-mass summary table (chebyshev grid only - full trivially retains ~100%) ---
    out_dir = OUT_DIR_MARGINAL / GRID_FOLDER["chebyshev"] / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    rmass = retained_mass_summary_table(grid="chebyshev")
    path = out_dir / f"retained_mass_summary_c{CHAINS}.csv"
    rmass.to_csv(path, index=False)
    print(f"wrote {len(rmass)} rows -> {path}")

    # --- KL = inf count tables (one per evaluation grid - 'full' is far more prone to this) ---
    for grid in GRIDS:
        out_dir = OUT_DIR_MARGINAL / GRID_FOLDER[grid] / "tables"
        out_dir.mkdir(parents=True, exist_ok=True)
        kinf = kl_inf_summary_table(grid=grid)
        path = out_dir / f"kl_inf_summary_c{CHAINS}.csv"
        kinf.to_csv(path, index=False)
        print(f"wrote {len(kinf)} rows -> {path}")


if __name__ == "__main__":
    main()
