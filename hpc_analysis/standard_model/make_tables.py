# Summary tables for the standard model (jobs 200-202). Single condition
# cell (k_true == k_model == 1), so one table per topic - no k_true variants.
# Reads data/out/standard_model/, writes out/<topic>/tables/*.csv;
# marginal tables once per grid ('full' -> full/, 'chebyshev' -> trimmed/).
# run: .venv/Scripts/python.exe hpc_analysis/standard_model/make_tables.py

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


# bias/MSE + Monte Carlo SEs per Morris et al. 2019 Table 6;
# d_i = post_mean_i - true_value_i, mcse = std(., ddof=1)/sqrt(n_sim)
def _bias_mse_stats(d: pd.Series) -> dict:
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


# runtime five-number summary (+ mean) per sampler, in minutes
def runtime_summary_table(n_chains: int = CHAINS) -> pd.DataFrame:
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


# spread of post_std across replicate seeds per (element, sampler)
def delta_sd_summary_table(n_chains: int = CHAINS) -> pd.DataFrame:
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


# wide: element rows x {bias,mcse_bias,mse,mcse_mse,n_sim}_<sampler>
def delta_bias_mse_table(n_chains: int = CHAINS) -> pd.DataFrame:
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


# mu bias/MSE + MCSEs per (param, sampler); mean_post_std = typical
# posterior SD. directly meaningful at K = 1
def mu_recovery_summary_table(n_chains: int = CHAINS) -> pd.DataFrame:
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


# Sigma lower-triangle bias/MSE + MCSEs per (element, sampler);
# mean_empirical = cov of the TRUE unit betas (finite-N reference)
def sigma_recovery_summary_table(n_chains: int = CHAINS) -> pd.DataFrame:
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


# KL/TVD/Hellinger/JSD spread per (param, sampler, metric); lower =
# closer to the true DGP marginal. Hellinger bounded in [0,1]
def marginal_distance_summary_table(n_chains: int = CHAINS,
                                    grid: str = "chebyshev") -> pd.DataFrame:
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


# retained_mass_model spread per (param, sampler); frac_below_guarantee
# vs 0.96 = 1 - 1/5**2 (Chebyshev, k=5) - should be ~0 after the fix
def retained_mass_summary_table(n_chains: int = CHAINS, grid: str = "chebyshev") -> pd.DataFrame:
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


# seeds with KL = +inf per (param, sampler): catastrophic tail mismatch
# (model mass where true density ~0); 'full' grid far more prone
def kl_inf_summary_table(n_chains: int = CHAINS, grid: str = "chebyshev") -> pd.DataFrame:
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
