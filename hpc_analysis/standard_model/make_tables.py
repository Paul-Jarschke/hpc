"""
Generate summary tables for the standard_model sampler comparison (jobs 200-202).

Run after the gather step has refreshed data/out/standard_model/:
    .venv/Scripts/python.exe hpc_analysis/standard_model/make_tables.py

Writes CSVs to hpc_analysis/standard_model/out/<topic>/tables/. Marginal-distance tables
are produced once per evaluation grid ('full' and 'chebyshev'; full/trimmed subfolder).
The standard model has a single condition cell (k_true == k_model == 1), so there are no
per-k_true table variants - each topic gets ONE table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import plot_recovery as pr  # noqa: E402  (module ref so DIR_FIG stays patchable)
from plot_recovery import (  # noqa: E402
    load_recovery, delta_element_label, sigma_element_label, MARGINAL_METRICS,
)
import marginal_diag  # noqa: E402

_PARAM_ORDER = ["Alt1", "Alt2", "Alt3", "Price"]

CHAINS = 2
GRIDS = ["full", "chebyshev"]  # marginal-distance evaluation-grid scenarios
GRID_FOLDER = {"full": "full", "chebyshev": "trimmed"}  # output subfolder per grid
SAMPLER_ORDER = ["bayesm", "nuts", "hmc"]
SAMPLER_LABELS = {"bayesm": "bayesm", "nuts": "NUTS", "hmc": "HMC"}


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
    """Runtime summary statistics by sampler (in minutes).

    Returns one row per sampler with columns:
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
        df.groupby("sampler")
        .apply(_agg, include_groups=False)
        .reset_index()
    )
    # Canonical sampler order and readable labels
    samplers_present = [s for s in SAMPLER_ORDER if s in agg["sampler"].unique()]
    agg["sampler"] = pd.Categorical(agg["sampler"], categories=samplers_present, ordered=True)
    agg["sampler_label"] = agg["sampler"].map(SAMPLER_LABELS)
    agg = agg.sort_values("sampler")
    stat_cols = ["min", "q1", "mean", "median", "q3", "max"]
    agg[stat_cols] = agg[stat_cols].round(2)
    return agg[["sampler_label", *stat_cols, "n_runs"]].rename(
        columns={"sampler_label": "sampler"}
    )


def delta_sd_summary_table(n_chains: int = CHAINS) -> pd.DataFrame:
    """Distribution of posterior SD for every Delta element, by sampler.

    For each (element, sampler) cell the statistics summarize the post_std
    values across all replicate seeds, i.e. how tightly (and consistently) each
    sampler pins down each element over repeated datasets.

    Returns a tidy long DataFrame:
        rows    = (element, sampler)
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
        df.groupby(["element", "sampler"])
        .apply(_agg, include_groups=False)
        .reset_index()
    )
    # Canonical sampler order and readable labels
    samplers_present = [s for s in SAMPLER_ORDER if s in agg["sampler"].unique()]
    agg["sampler"] = pd.Categorical(agg["sampler"], categories=samplers_present, ordered=True)
    agg["sampler"] = agg["sampler"].map(SAMPLER_LABELS)
    agg = agg.sort_values(["element", "sampler"])
    stat_cols = ["min", "q1", "mean", "median", "q3", "max"]
    agg[stat_cols] = agg[stat_cols].round(4)
    return agg[["element", "sampler", *stat_cols, "n_sim"]].reset_index(drop=True)


def delta_bias_mse_table(n_chains: int = CHAINS) -> pd.DataFrame:
    """Bias, MSE and their Monte Carlo SEs for every Delta element, by sampler.

    For each (sampler, element) cell across n_sim replicate seeds, from the per-seed
    differences d_i = post_mean_i - true_value_i (Morris et al. 2019, Table 6):
        bias      = mean(d_i)
        mcse_bias = std(d_i,   ddof=1) / sqrt(n_sim)
        mse       = mean(d_i^2)
        mcse_mse  = std(d_i^2, ddof=1) / sqrt(n_sim)

    Returns a wide DataFrame:
        rows    = element
        columns = {bias, mcse_bias, mse, mcse_mse, n_sim}_{sampler}
    """
    df = load_recovery("delta")
    df = df[df["n_chains"] == n_chains].copy()
    df["element"] = df.apply(lambda r: delta_element_label(r["demo"], r["param"]), axis=1)

    agg = (
        df.groupby(["element", "sampler"])
        .apply(lambda g: pd.Series(_bias_mse_stats(g["bias"])), include_groups=False)
        .reset_index()
    )

    metrics = ["bias", "mcse_bias", "mse", "mcse_mse", "n_sim"]
    wide = agg.pivot_table(index="element", columns="sampler", values=metrics)
    samplers = [s for s in SAMPLER_ORDER if s in agg["sampler"].unique()]
    wide = wide.reindex(columns=pd.MultiIndex.from_product([metrics, samplers]))
    wide = wide.round(6)
    wide.columns = [f"{metric}_{SAMPLER_LABELS.get(s, s)}" for metric, s in wide.columns]
    return wide.reset_index()


def mu_recovery_summary_table(n_chains: int = CHAINS) -> pd.DataFrame:
    """Bias, MSE and their Monte Carlo SEs of the population mean mu, per parameter and
    sampler (NEW for the standard model - directly meaningful at K = 1).

    For each (param, sampler) cell across n_sim replicate seeds, from the per-seed
    differences d_i = post_mean_i - true_value_i (Morris et al. 2019, Table 6):
        bias         = mean(d_i)
        mcse_bias    = std(d_i,   ddof=1) / sqrt(n_sim)
        mse          = mean(d_i^2)
        mcse_mse     = std(d_i^2, ddof=1) / sqrt(n_sim)
        mean_post_std= mean posterior SD (how tight the mu posterior typically is)

    Returns a tidy long DataFrame:
        rows    = (param, sampler)
        columns = bias, mcse_bias, mse, mcse_mse, mean_post_std, n_sim
    """
    df = load_recovery("mu")
    df = df[df["n_chains"] == n_chains].copy()

    def _agg(g):
        s = _bias_mse_stats(g["bias"])
        s["mean_post_std"] = g["post_std"].mean()
        return pd.Series(s)

    agg = (
        df.groupby(["param", "sampler"], observed=True)
        .apply(_agg, include_groups=False)
        .reset_index()
    )
    samplers_present = [s for s in SAMPLER_ORDER if s in agg["sampler"].unique()]
    params_present = [p for p in _PARAM_ORDER if p in agg["param"].unique()]
    agg["sampler"] = pd.Categorical(agg["sampler"], categories=samplers_present, ordered=True)
    agg["param"] = pd.Categorical(agg["param"], categories=params_present, ordered=True)
    agg["sampler"] = agg["sampler"].map(SAMPLER_LABELS)
    agg = agg.sort_values(["param", "sampler"])
    val_cols = ["bias", "mcse_bias", "mse", "mcse_mse", "mean_post_std"]
    agg[val_cols] = agg[val_cols].round(6)
    return agg[["param", "sampler", *val_cols, "n_sim"]].reset_index(drop=True)


def sigma_recovery_summary_table(n_chains: int = CHAINS) -> pd.DataFrame:
    """POSTERIOR SIGMA recovery summary, per lower-triangle element and sampler (NEW for
    the standard model - the numeric form of the notebook's posterior-covariance analysis).

    For each (element, sampler) cell across n_sim replicate seeds, from the per-seed
    differences d_i = post_mean_i - true_value_i (Morris et al. 2019, Table 6):
        bias           = mean(d_i)
        mcse_bias      = std(d_i,   ddof=1) / sqrt(n_sim)
        mse            = mean(d_i^2)
        mcse_mse       = std(d_i^2, ddof=1) / sqrt(n_sim)
    plus descriptive references:
        mean_post_mean = mean posterior mean of the element
        mean_true      = mean TRUE_SIGMA value (drawn per seed, so this is the seed average)
        mean_empirical = mean empirical covariance of the TRUE unit betas (what the finite
                         N units actually realise)
        mean_abs_diff  = mean |post_mean - true_value|

    Returns a tidy long DataFrame:
        rows    = (element, sampler)
        columns = bias, mcse_bias, mse, mcse_mse, mean_post_mean, mean_true,
                  mean_empirical, mean_abs_diff, n_sim
    """
    df = load_recovery("sigma")
    df = df[df["n_chains"] == n_chains].copy()
    df["element"] = df.apply(lambda r: sigma_element_label(r["row"], r["col"]), axis=1)

    def _agg(g):
        s = _bias_mse_stats(g["bias"])
        s["mean_post_mean"] = g["post_mean"].mean()
        s["mean_true"]      = g["true_value"].mean()
        s["mean_empirical"] = g["empirical"].mean()
        s["mean_abs_diff"]  = g["abs_diff"].mean()
        return pd.Series(s)

    agg = (
        df.groupby(["element", "sampler"], observed=True)
        .apply(_agg, include_groups=False)
        .reset_index()
    )
    samplers_present = [s for s in SAMPLER_ORDER if s in agg["sampler"].unique()]
    agg["sampler"] = pd.Categorical(agg["sampler"], categories=samplers_present, ordered=True)
    agg["sampler"] = agg["sampler"].map(SAMPLER_LABELS)
    agg = agg.sort_values(["element", "sampler"])
    val_cols = ["bias", "mcse_bias", "mse", "mcse_mse",
                "mean_post_mean", "mean_true", "mean_empirical", "mean_abs_diff"]
    agg[val_cols] = agg[val_cols].round(6)
    return agg[["element", "sampler", *val_cols, "n_sim"]].reset_index(drop=True)


def marginal_distance_summary_table(n_chains: int = CHAINS,
                                    grid: str = "chebyshev") -> pd.DataFrame:
    """Distribution of all five marginal distance metrics, by sampler and parameter.

    For each (param, sampler, metric) cell across n_sim replicate seeds, summarises
    the distribution of KL/TVD/Hellinger/JSD values: lower is better (fitted
    marginal closer to the true DGP marginal). Hellinger is bounded in [0,1]; the others are
    unbounded but comparable within a metric. `grid` selects the evaluation-grid scenario
    ('full' or 'chebyshev') the distances were computed on.

    Returns a tidy long DataFrame:
        rows    = (param, sampler, metric)
        columns = min, q1, mean, median, q3, max, n_sim
    """
    df = load_recovery("marginal_distances")
    if "grid" in df.columns:
        df = df[df["grid"] == grid]
    df = df[df["n_chains"] == n_chains].copy()
    metrics_present = [m for m in MARGINAL_METRICS if m in df.columns]
    id_cols = [c for c in ("param", "sampler") if c in df.columns]
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
        long.groupby(["param", "sampler", "metric"], observed=True)
        .apply(_agg, include_groups=False)
        .reset_index()
    )
    samplers_present = [s for s in SAMPLER_ORDER if s in agg["sampler"].unique()]
    agg["sampler"] = pd.Categorical(agg["sampler"], categories=samplers_present, ordered=True)
    agg["sampler_label"] = agg["sampler"].map(SAMPLER_LABELS)
    agg["metric"] = pd.Categorical(agg["metric"], categories=metrics_present, ordered=True)
    agg = agg.sort_values(["param", "metric", "sampler"])
    stat_cols = ["min", "q1", "mean", "median", "q3", "max"]
    agg[stat_cols] = agg[stat_cols].round(5)
    return agg[["param", "sampler_label", "metric", *stat_cols, "n_sim"]].rename(
        columns={"sampler_label": "sampler"}
    ).reset_index(drop=True)


def retained_mass_summary_table(n_chains: int = CHAINS, grid: str = "chebyshev") -> pd.DataFrame:
    """Distribution of retained_mass_model (mc.retained_mass), by sampler and parameter -
    the realised counterpart to the theoretical Chebyshev mass guarantee.

    For each (param, sampler) cell across n_sim replicate seeds, summarises the fraction
    of each fitted model's own marginal mass retained inside the evaluation-grid window.
    frac_below_guarantee = fraction of seeds where retained mass fell BELOW the theoretical
    minimum (k=5 -> 1 - 1/5**2 = 0.96) - should be ~0 if the Chebyshev fix is working as
    intended. `grid` selects the evaluation-grid scenario ('full' trivially retains ~100%;
    'chebyshev' is the meaningful case).

    Returns a tidy long DataFrame:
        rows    = (param, sampler)
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
        df.groupby(["param", "sampler"], observed=True)
        .apply(_agg, include_groups=False)
        .reset_index()
    )
    samplers_present = [s for s in SAMPLER_ORDER if s in agg["sampler"].unique()]
    agg["sampler"] = pd.Categorical(agg["sampler"], categories=samplers_present, ordered=True)
    agg["sampler_label"] = agg["sampler"].map(SAMPLER_LABELS)
    agg = agg.sort_values(["param", "sampler"])
    stat_cols = ["min", "q1", "mean", "median", "q3", "max", "frac_below_guarantee"]
    agg[stat_cols] = agg[stat_cols].round(5)
    return agg[["param", "sampler_label", *stat_cols, "n_sim"]].rename(
        columns={"sampler_label": "sampler"}
    ).reset_index(drop=True)


def kl_inf_summary_table(n_chains: int = CHAINS, grid: str = "chebyshev") -> pd.DataFrame:
    """Count of seeds where KL(model||true) came back +inf, by sampler and parameter -
    a direct measure of catastrophic tail mismatch (the fitted marginal puts mass where
    the true DGP density is ~0). `grid` selects the evaluation-grid scenario ('full' is
    far more prone to this than the 'chebyshev'-trimmed grid).

    Returns a tidy long DataFrame:
        rows    = (param, sampler)
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
        df.groupby(["param", "sampler"], observed=True)["is_inf"]
        .agg(n_inf="sum", n_total="count").reset_index()
    )
    agg["inf_rate"] = (agg["n_inf"] / agg["n_total"]).round(4)
    samplers_present = [s for s in SAMPLER_ORDER if s in agg["sampler"].unique()]
    agg["sampler"] = pd.Categorical(agg["sampler"], categories=samplers_present, ordered=True)
    agg["sampler_label"] = agg["sampler"].map(SAMPLER_LABELS)
    agg = agg.sort_values(["param", "sampler"])
    return agg[["param", "sampler_label", "n_inf", "n_total", "inf_rate"]].rename(
        columns={"sampler_label": "sampler"}
    ).reset_index(drop=True)


def main():
    # Output dirs resolved at call time so tests can monkeypatch plot_recovery.DIR_FIG.
    out_delta_bias     = pr.DIR_FIG / "delta" / "bias"     / "tables"
    out_delta_sd       = pr.DIR_FIG / "delta" / "sd"       / "tables"
    out_runtime        = pr.DIR_FIG / "runtime" / "tables"
    out_mu    = pr.DIR_FIG / "mu"    / "tables"
    out_sigma = pr.DIR_FIG / "sigma" / "tables"
    out_marginal = pr.DIR_FIG / "marginal_comparison"   # per-grid subfolder added below

    for d in (out_delta_bias, out_delta_sd,
              out_runtime, out_mu, out_sigma):
        d.mkdir(parents=True, exist_ok=True)

    # --- Delta bias + MSE table (bias, MCSE(bias), MSE, MCSE(MSE) per sampler) ---
    tbl = delta_bias_mse_table()
    path = out_delta_bias / f"delta_bias_mse_c{CHAINS}.csv"
    tbl.to_csv(path, index=False)
    print(f"wrote {len(tbl)} rows -> {path}")

    # --- Delta posterior SD table ---
    sd = delta_sd_summary_table()
    path = out_delta_sd / f"delta_sd_summary_c{CHAINS}.csv"
    sd.to_csv(path, index=False)
    print(f"wrote {len(sd)} rows -> {path}")

    # --- Runtime summary table ---
    rt = runtime_summary_table()
    path = out_runtime / f"runtime_summary_c{CHAINS}.csv"
    rt.to_csv(path, index=False)
    print(f"wrote {len(rt)} rows -> {path}")

    # --- Mu recovery summary (bias / MSE + MCSEs per param x sampler) ---
    mu = mu_recovery_summary_table()
    path = out_mu / f"mu_recovery_summary_c{CHAINS}.csv"
    mu.to_csv(path, index=False)
    print(f"wrote {len(mu)} rows -> {path}")

    # --- Sigma recovery summary (bias / MSE + MCSEs, plus posterior/true/empirical refs) ---
    sig = sigma_recovery_summary_table()
    path = out_sigma / f"sigma_recovery_summary_c{CHAINS}.csv"
    sig.to_csv(path, index=False)
    print(f"wrote {len(sig)} rows -> {path}")

    # --- Marginal-series convergence tables (R-hat + ESS summaries, one set per grid) ---
    marginal_diag.write_tables(CHAINS)

    # --- Marginal distance summary tables (one per evaluation grid) ---
    for grid in GRIDS:
        out_dir = out_marginal / GRID_FOLDER[grid] / "tables"
        out_dir.mkdir(parents=True, exist_ok=True)
        mdist = marginal_distance_summary_table(grid=grid)
        path = out_dir / f"marginal_distance_summary_c{CHAINS}.csv"
        mdist.to_csv(path, index=False)
        print(f"wrote {len(mdist)} rows -> {path}")

    # --- Retained-mass summary table (chebyshev grid only - full trivially retains ~100%) ---
    out_dir = out_marginal / GRID_FOLDER["chebyshev"] / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    rmass = retained_mass_summary_table(grid="chebyshev")
    path = out_dir / f"retained_mass_summary_c{CHAINS}.csv"
    rmass.to_csv(path, index=False)
    print(f"wrote {len(rmass)} rows -> {path}")

    # --- KL = inf count tables (one per evaluation grid - 'full' is far more prone to this) ---
    for grid in GRIDS:
        out_dir = out_marginal / GRID_FOLDER[grid] / "tables"
        out_dir.mkdir(parents=True, exist_ok=True)
        kinf = kl_inf_summary_table(grid=grid)
        path = out_dir / f"kl_inf_summary_c{CHAINS}.csv"
        kinf.to_csv(path, index=False)
        print(f"wrote {len(kinf)} rows -> {path}")


if __name__ == "__main__":
    main()
