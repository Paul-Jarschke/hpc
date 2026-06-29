"""
Generate summary tables for the k5model_mixture sampler comparison.

Run after scripts/gather_summaries.py:
    .venv/Scripts/python.exe analysis/make_tables.py

Writes CSVs to analysis/out/k5_results/tables/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plot_recovery import load_recovery, DIR_FIG, delta_element_label  # noqa: E402

OUT_DIR_DELTA_BIAS = DIR_FIG / "delta" / "bias" / "tables"
OUT_DIR_DELTA_SD   = DIR_FIG / "delta" / "sd"   / "tables"
OUT_DIR_DELTA_RMSE = DIR_FIG / "delta" / "rmse" / "tables"
OUT_DIR_RUNTIME    = DIR_FIG / "runtime" / "tables"

CHAINS = 2
SAMPLER_ORDER = ["bayesm", "nuts", "hmc"]
SAMPLER_LABELS = {"bayesm": "bayesm", "nuts": "NUTS", "hmc": "HMC"}


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
    stat_cols = ["min", "q1", "median", "q3", "max"]
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


def delta_rmse_summary_table(n_chains: int = CHAINS) -> pd.DataFrame:
    """Distribution of absolute error |post_mean - true_value| for every Delta element.

    For each (k_true, element, sampler) cell the statistics summarize |bias| values
    across all replicate seeds. Unlike the bias table (which reports the mean signed
    error and its MCSE), this captures the typical magnitude of error regardless of
    direction, matching what is shown in the RMSE boxplots.

    Returns a tidy long DataFrame:
        rows    = (k_true, element, sampler)
        columns = min, q1, mean, median, q3, max, n_sim
    """
    df = load_recovery("delta")
    df = df[df["n_chains"] == n_chains].copy()
    df["element"] = df.apply(lambda r: delta_element_label(r["demo"], r["param"]), axis=1)
    df["abs_error"] = df["bias"].abs()

    def _agg(g):
        a = g["abs_error"]
        return pd.Series({
            "min":    a.min(),
            "q1":     a.quantile(0.25),
            "mean":   a.mean(),
            "median": a.median(),
            "q3":     a.quantile(0.75),
            "max":    a.max(),
            "n_sim":  len(a),
        })

    agg = (
        df.groupby(["k_true", "element", "sampler"])
        .apply(_agg, include_groups=False)
        .reset_index()
    )
    samplers_present = [s for s in SAMPLER_ORDER if s in agg["sampler"].unique()]
    agg["sampler"] = pd.Categorical(agg["sampler"], categories=samplers_present, ordered=True)
    agg["sampler"] = agg["sampler"].map(SAMPLER_LABELS)
    agg = agg.sort_values(["k_true", "element", "sampler"])
    stat_cols = ["min", "q1", "mean", "median", "q3", "max"]
    agg[stat_cols] = agg[stat_cols].round(4)
    return agg[["k_true", "element", "sampler", *stat_cols, "n_sim"]].reset_index(drop=True)


def delta_bias_mcse_table(n_chains: int = CHAINS) -> pd.DataFrame:
    """Bias and Monte Carlo SE for every Delta element, by sampler and k_true.

    For each (sampler, k_true, element) cell across n_sim replicate seeds:
        bias = mean(post_mean - true_value)
        mcse = std(post_mean - true_value) / sqrt(n_sim)   [SE of the bias estimate]

    Returns a wide DataFrame:
        rows    = (k_true, element)
        columns = MultiIndex (sampler, metric) with metric in {bias, mcse, n_sim}
    """
    df = load_recovery("delta")
    df = df[df["n_chains"] == n_chains].copy()
    df["element"] = df.apply(lambda r: delta_element_label(r["demo"], r["param"]), axis=1)

    def _agg(g):
        b = g["bias"]
        n = len(b)
        return pd.Series({
            "bias":  b.mean(),
            "mcse":  b.std(ddof=1) / np.sqrt(n),
            "n_sim": n,
        })

    agg = (
        df.groupby(["k_true", "element", "sampler"])
        .apply(_agg, include_groups=False)
        .reset_index()
    )

    # Wide format: one (bias, mcse, n_sim) triple per sampler
    wide = agg.pivot_table(
        index=["k_true", "element"],
        columns="sampler",
        values=["bias", "mcse", "n_sim"],
    )
    # Reorder columns: sampler-major, metric-minor
    samplers = [s for s in SAMPLER_ORDER if s in agg["sampler"].unique()]
    wide = wide.reindex(columns=pd.MultiIndex.from_product([["bias", "mcse", "n_sim"], samplers]))
    wide = wide.round(4)
    wide.columns = [f"{metric}_{SAMPLER_LABELS.get(s, s)}" for metric, s in wide.columns]
    return wide.reset_index()


def main():
    OUT_DIR_DELTA_BIAS.mkdir(parents=True, exist_ok=True)
    OUT_DIR_DELTA_SD.mkdir(parents=True, exist_ok=True)
    OUT_DIR_DELTA_RMSE.mkdir(parents=True, exist_ok=True)
    OUT_DIR_RUNTIME.mkdir(parents=True, exist_ok=True)

    # --- Delta bias / MCSE tables ---
    tbl = delta_bias_mcse_table()

    path = OUT_DIR_DELTA_BIAS / f"delta_bias_mcse_c{CHAINS}_all.csv"
    tbl.to_csv(path, index=False)
    print(f"wrote {len(tbl)} rows -> {path}")

    for kt in sorted(tbl["k_true"].unique()):
        sub = tbl[tbl["k_true"] == kt].drop(columns="k_true")
        path = OUT_DIR_DELTA_BIAS / f"delta_bias_mcse_c{CHAINS}_kt{int(kt)}.csv"
        sub.to_csv(path, index=False)
        print(f"wrote {len(sub)} rows -> {path}")

    # --- Delta absolute error / RMSE tables ---
    rmse = delta_rmse_summary_table()

    path = OUT_DIR_DELTA_RMSE / f"delta_rmse_summary_c{CHAINS}_all.csv"
    rmse.to_csv(path, index=False)
    print(f"wrote {len(rmse)} rows -> {path}")

    for kt in sorted(rmse["k_true"].unique()):
        sub = rmse[rmse["k_true"] == kt].drop(columns="k_true")
        path = OUT_DIR_DELTA_RMSE / f"delta_rmse_summary_c{CHAINS}_kt{int(kt)}.csv"
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


if __name__ == "__main__":
    main()
