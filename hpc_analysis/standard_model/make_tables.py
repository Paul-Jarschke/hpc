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
    load_recovery, delta_element_label, sigma_element_label, compute_beta_correlation,
    compute_beta_post_std, MARGINAL_METRICS,
)
import marginal_diag  # noqa: E402

_PARAM_ORDER = ["Alt1", "Alt2", "Alt3", "Price"]

CHAINS = 2
GRIDS = ["full", "chebyshev"]  # marginal-distance evaluation-grid scenarios
GRID_FOLDER = {"full": "full", "chebyshev": "trimmed"}  # output subfolder per grid
SAMPLER_ORDER = ["bayesm", "nuts", "hmc"]
SAMPLER_LABELS = {"bayesm": "bayesm", "nuts": "NUTS", "hmc": "HMC"}


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
    stat_cols = ["min", "q1", "median", "q3", "max"]
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


def delta_coverage_table(n_chains: int = CHAINS) -> pd.DataFrame:
    """Empirical 95% CI coverage rate for every Delta element, by sampler.

    Coverage (%) = count(in_ci == True) / n_sim * 100.
    A well-calibrated sampler should be near 95% for every element.

    Returns a wide DataFrame:
        rows    = element
        columns = coverage_pct_{sampler} and n_sim_{sampler}
    """
    df = load_recovery("delta")
    df = df[df["n_chains"] == n_chains].copy()
    df["element"] = df.apply(lambda r: delta_element_label(r["demo"], r["param"]), axis=1)
    df["in_ci"] = df["in_ci"].astype(bool)

    agg = (
        df.groupby(["element", "sampler"], observed=True)["in_ci"]
        .agg(coverage_pct=lambda x: round(x.mean() * 100, 1), n_sim="count")
        .reset_index()
    )

    samplers = [s for s in SAMPLER_ORDER if s in agg["sampler"].unique()]
    wide = agg.pivot_table(
        index="element",
        columns="sampler",
        values=["coverage_pct", "n_sim"],
    )
    wide = wide.reindex(
        columns=pd.MultiIndex.from_product([["coverage_pct", "n_sim"], samplers])
    )
    wide.columns = [f"{metric}_{SAMPLER_LABELS.get(s, s)}" for metric, s in wide.columns]
    return wide.reset_index()


def delta_rmse_summary_table(n_chains: int = CHAINS) -> pd.DataFrame:
    """RMSE of every Delta element across the replicate seeds, by sampler.

    Delta is a POPULATION parameter: each element has a single point estimate per run,
    so - unlike beta (RMSE over the 300 units) - the error cannot be root-mean-squared
    within a run. The RMSE here is the Monte-Carlo RMSE of the estimator, taken across
    the n_sim replicate datasets:

        rmse = sqrt(mean_seeds((post_mean - true_value)^2))

    which decomposes exactly as rmse^2 = bias^2 + error_sd^2 (both reported; error_sd is
    the population SD of the signed error over seeds, ddof=0). mean_abs_error is kept for
    reference (it is what the previous version of this table reported).

    Returns a tidy long DataFrame:
        rows    = (element, sampler)
        columns = rmse, bias, error_sd, mean_abs_error, n_sim
    """
    df = load_recovery("delta")
    df = df[df["n_chains"] == n_chains].copy()
    df["element"] = df.apply(lambda r: delta_element_label(r["demo"], r["param"]), axis=1)

    def _agg(g):
        e = g["bias"]  # signed error post_mean - true_value, one per seed
        return pd.Series({
            "rmse":           np.sqrt((e ** 2).mean()),
            "bias":           e.mean(),
            "error_sd":       e.std(ddof=0),
            "mean_abs_error": e.abs().mean(),
            "n_sim":          len(e),
        })

    agg = (
        df.groupby(["element", "sampler"])
        .apply(_agg, include_groups=False)
        .reset_index()
    )
    samplers_present = [s for s in SAMPLER_ORDER if s in agg["sampler"].unique()]
    agg["sampler"] = pd.Categorical(agg["sampler"], categories=samplers_present, ordered=True)
    agg["sampler"] = agg["sampler"].map(SAMPLER_LABELS)
    agg = agg.sort_values(["element", "sampler"])
    val_cols = ["rmse", "bias", "error_sd", "mean_abs_error"]
    agg[val_cols] = agg[val_cols].round(4)
    return agg[["element", "sampler", *val_cols, "n_sim"]].reset_index(drop=True)


def delta_bias_mcse_table(n_chains: int = CHAINS) -> pd.DataFrame:
    """Bias and Monte Carlo SE for every Delta element, by sampler.

    For each (sampler, element) cell across n_sim replicate seeds:
        bias = mean(post_mean - true_value)
        mcse = std(post_mean - true_value) / sqrt(n_sim)   [SE of the bias estimate]

    Returns a wide DataFrame:
        rows    = element
        columns = one (bias, mcse, n_sim) triple per sampler
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
        df.groupby(["element", "sampler"])
        .apply(_agg, include_groups=False)
        .reset_index()
    )

    # Wide format: one (bias, mcse, n_sim) triple per sampler
    wide = agg.pivot_table(
        index="element",
        columns="sampler",
        values=["bias", "mcse", "n_sim"],
    )
    # Reorder columns: sampler-major, metric-minor
    samplers = [s for s in SAMPLER_ORDER if s in agg["sampler"].unique()]
    wide = wide.reindex(columns=pd.MultiIndex.from_product([["bias", "mcse", "n_sim"], samplers]))
    wide = wide.round(4)
    wide.columns = [f"{metric}_{SAMPLER_LABELS.get(s, s)}" for metric, s in wide.columns]
    return wide.reset_index()


def beta_bias_mcse_table(n_chains: int = CHAINS) -> pd.DataFrame:
    """Bias and Monte Carlo SE for each beta parameter, by sampler.

    beta_recovery.csv already stores, per run, the signed `bias` = mean over the N
    decision units of (post_mean_i - true_i) for each parameter. For each
    (param, sampler) cell across n_sim replicate seeds this reports:
        bias = mean(per-run bias)
        mcse = std(per-run bias) / sqrt(n_sim)   [SE of the bias estimate]
    the beta counterpart of delta_bias_mcse_table (per-param rather than per-element).

    Returns a wide DataFrame:
        rows    = param
        columns = one (bias, mcse, n_sim) triple per sampler
    """
    df = load_recovery("beta")
    df = df[df["n_chains"] == n_chains].copy()

    def _agg(g):
        b = g["bias"]
        n = len(b)
        return pd.Series({
            "bias":  b.mean(),
            "mcse":  b.std(ddof=1) / np.sqrt(n),
            "n_sim": n,
        })

    agg = (
        df.groupby(["param", "sampler"], observed=True)
        .apply(_agg, include_groups=False)
        .reset_index()
    )

    # Wide format: one (bias, mcse, n_sim) triple per sampler, params in canonical order.
    wide = agg.pivot_table(index="param", columns="sampler",
                           values=["bias", "mcse", "n_sim"])
    samplers = [s for s in SAMPLER_ORDER if s in agg["sampler"].unique()]
    wide = wide.reindex(columns=pd.MultiIndex.from_product([["bias", "mcse", "n_sim"], samplers]))
    wide = wide.round(4)
    wide.columns = [f"{metric}_{SAMPLER_LABELS.get(s, s)}" for metric, s in wide.columns]
    params_present = [p for p in _PARAM_ORDER if p in wide.index]
    wide = wide.reindex(params_present)
    return wide.reset_index()


def beta_sd_summary_table(n_chains: int = CHAINS, sd_df=None) -> pd.DataFrame:
    """Distribution of the mean posterior SD of beta_i, per parameter and sampler.

    From compute_beta_post_std: per run, the mean over the N units of post_std for each
    parameter. For each (param, sampler) cell the statistics summarize those per-run mean
    SDs across replicate seeds - how tightly (and how consistently) each sampler pins down
    individual coefficients. The beta counterpart of delta_sd_summary_table.

    Returns a tidy long DataFrame:
        rows    = (param, sampler)
        columns = min, q1, mean, median, q3, max, n_sim
    """
    if sd_df is None:
        print("loading beta_summary.csv to compute posterior SDs ...")
        sd_df = compute_beta_post_std()
    df = sd_df[sd_df["n_chains"] == n_chains].copy()

    def _agg(g):
        s = g["mean_post_std"]
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
    stat_cols = ["min", "q1", "mean", "median", "q3", "max"]
    agg[stat_cols] = agg[stat_cols].round(4)
    return agg[["param", "sampler", *stat_cols, "n_sim"]].reset_index(drop=True)


def beta_rmse_summary_table(n_chains: int = CHAINS) -> pd.DataFrame:
    """Distribution of per-run RMSE for each beta parameter, by sampler.

    RMSE is pre-aggregated over the units in beta_recovery.csv. This table
    summarises those per-run values across replicate seeds.

    Returns a tidy long DataFrame:
        rows    = (param, sampler)
        columns = min, q1, mean, median, q3, max, n_sim
    """
    df = load_recovery("beta")
    df = df[df["n_chains"] == n_chains].copy()

    def _agg(g):
        r = g["rmse"]
        return pd.Series({
            "min":    r.min(),
            "q1":     r.quantile(0.25),
            "mean":   r.mean(),
            "median": r.median(),
            "q3":     r.quantile(0.75),
            "max":    r.max(),
            "n_sim":  len(r),
        })

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
    stat_cols = ["min", "q1", "mean", "median", "q3", "max"]
    agg[stat_cols] = agg[stat_cols].round(4)
    return agg[["param", "sampler", *stat_cols, "n_sim"]].reset_index(drop=True)


def beta_correlation_summary_table(n_chains: int = CHAINS,
                                   corr_df=None) -> pd.DataFrame:
    """Distribution of per-run Pearson correlation for each beta parameter.

    Correlation = corr_i(post_mean_i, true_value_i) across the units, one value per run.
    This table summarises those values across replicate seeds.

    Returns a tidy long DataFrame:
        rows    = (param, sampler)
        columns = min, q1, mean, median, q3, max, n_sim
    """
    if corr_df is None:
        print("loading beta_summary.csv to compute correlations ...")
        corr_df = compute_beta_correlation()
    df = corr_df[corr_df["n_chains"] == n_chains].copy()

    def _agg(g):
        r = g["correlation"]
        return pd.Series({
            "min":    r.min(),
            "q1":     r.quantile(0.25),
            "mean":   r.mean(),
            "median": r.median(),
            "q3":     r.quantile(0.75),
            "max":    r.max(),
            "n_sim":  len(r),
        })

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
    stat_cols = ["min", "q1", "mean", "median", "q3", "max"]
    agg[stat_cols] = agg[stat_cols].round(4)
    return agg[["param", "sampler", *stat_cols, "n_sim"]].reset_index(drop=True)


def beta_coverage_table(n_chains: int = CHAINS) -> pd.DataFrame:
    """Mean 95% CI coverage of individual betas, by parameter and sampler.

    coverage95 in beta_recovery is the fraction of units whose true beta_i
    falls in the posterior 95% CI for that run. This table averages that fraction
    across replicate seeds and expresses it as a percentage.

    Returns a wide DataFrame:
        rows    = param
        columns = coverage_pct_{sampler} and n_sim_{sampler}
    """
    df = load_recovery("beta")
    df = df[df["n_chains"] == n_chains].copy()

    agg = (
        df.groupby(["param", "sampler"], observed=True)["coverage95"]
        .agg(coverage_pct=lambda x: round(x.mean() * 100, 1), n_sim="count")
        .reset_index()
    )
    samplers = [s for s in SAMPLER_ORDER if s in agg["sampler"].unique()]
    wide = agg.pivot_table(
        index="param",
        columns="sampler",
        values=["coverage_pct", "n_sim"],
    )
    wide = wide.reindex(
        columns=pd.MultiIndex.from_product([["coverage_pct", "n_sim"], samplers])
    )
    wide.columns = [f"{metric}_{SAMPLER_LABELS.get(s, s)}" for metric, s in wide.columns]
    return wide.reset_index()


def mu_recovery_summary_table(n_chains: int = CHAINS) -> pd.DataFrame:
    """Bias, Monte Carlo SE and 95% CI coverage of the population mean mu, per
    parameter and sampler (NEW for the standard model - directly meaningful at K = 1).

    For each (param, sampler) cell across n_sim replicate seeds:
        bias         = mean(post_mean - true_value)
        mcse         = std(post_mean - true_value) / sqrt(n_sim)
        mean_post_std= mean posterior SD (how tight the mu posterior typically is)
        coverage_pct = share of seeds with TRUE_MU[p] inside the 95% CI, in percent

    Returns a tidy long DataFrame:
        rows    = (param, sampler)
        columns = bias, mcse, mean_post_std, coverage_pct, n_sim
    """
    df = load_recovery("mu")
    df = df[df["n_chains"] == n_chains].copy()
    df["in_ci"] = df["in_ci"].astype(bool)

    def _agg(g):
        b = g["bias"]
        n = len(b)
        return pd.Series({
            "bias":          b.mean(),
            "mcse":          b.std(ddof=1) / np.sqrt(n),
            "mean_post_std": g["post_std"].mean(),
            "coverage_pct":  g["in_ci"].mean() * 100,
            "n_sim":         n,
        })

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
    agg[["bias", "mcse", "mean_post_std"]] = agg[["bias", "mcse", "mean_post_std"]].round(4)
    agg["coverage_pct"] = agg["coverage_pct"].round(1)
    return agg[["param", "sampler", "bias", "mcse", "mean_post_std",
                "coverage_pct", "n_sim"]].reset_index(drop=True)


def sigma_recovery_summary_table(n_chains: int = CHAINS) -> pd.DataFrame:
    """POSTERIOR SIGMA recovery summary, per lower-triangle element and sampler (NEW for
    the standard model - the numeric form of the notebook's posterior-covariance analysis).

    For each (element, sampler) cell across n_sim replicate seeds:
        mean_post_mean = mean posterior mean of the element
        mean_true      = mean TRUE_SIGMA value (constant per dataset family in principle,
                         but TRUE_SIGMA is drawn per seed, so this is the seed average)
        mean_empirical = mean empirical covariance of the TRUE unit betas (reference:
                         what the finite N units actually realise)
        mean_abs_diff  = mean |post_mean - true_value|
        coverage_pct   = share of seeds with the true value inside the 95% CI, in percent

    Returns a tidy long DataFrame:
        rows    = (element, sampler)
        columns = mean_post_mean, mean_true, mean_empirical, mean_abs_diff,
                  coverage_pct, n_sim
    """
    df = load_recovery("sigma")
    df = df[df["n_chains"] == n_chains].copy()
    df["element"] = df.apply(lambda r: sigma_element_label(r["row"], r["col"]), axis=1)
    df["in_ci"] = df["in_ci"].astype(bool)

    def _agg(g):
        return pd.Series({
            "mean_post_mean": g["post_mean"].mean(),
            "mean_true":      g["true_value"].mean(),
            "mean_empirical": g["empirical"].mean(),
            "mean_abs_diff":  g["abs_diff"].mean(),
            "coverage_pct":   g["in_ci"].mean() * 100,
            "n_sim":          len(g),
        })

    agg = (
        df.groupby(["element", "sampler"], observed=True)
        .apply(_agg, include_groups=False)
        .reset_index()
    )
    samplers_present = [s for s in SAMPLER_ORDER if s in agg["sampler"].unique()]
    agg["sampler"] = pd.Categorical(agg["sampler"], categories=samplers_present, ordered=True)
    agg["sampler"] = agg["sampler"].map(SAMPLER_LABELS)
    agg = agg.sort_values(["element", "sampler"])
    val_cols = ["mean_post_mean", "mean_true", "mean_empirical", "mean_abs_diff"]
    agg[val_cols] = agg[val_cols].round(4)
    agg["coverage_pct"] = agg["coverage_pct"].round(1)
    return agg[["element", "sampler", *val_cols, "coverage_pct", "n_sim"]].reset_index(drop=True)


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


def consolidated_rmse_table(n_chains: int = CHAINS) -> pd.DataFrame:
    """Consolidated RMSE across ALL elements of a parameter block (beta / Delta),
    by sampler.

    Per run the block's elements are pooled into one RMSE (see
    plot_recovery.consolidated_rmse_by_run); across the replicate runs of a cell
    this reports the distribution of that per-run RMSE plus `rmse_pooled` =
    sqrt(mean of squared per-run RMSEs) - the single-number grand RMSE of the cell.

    Returns a tidy long DataFrame:
        rows    = (block, sampler)
        columns = rmse_pooled, min, q1, mean, median, q3, max, n_sim
    """
    from plot_recovery import consolidated_rmse_by_run

    df = consolidated_rmse_by_run(n_chains)

    def _agg(g):
        a = g["rmse"]
        return pd.Series({
            "rmse_pooled": float(np.sqrt((a ** 2).mean())),
            "min":    a.min(),
            "q1":     a.quantile(0.25),
            "mean":   a.mean(),
            "median": a.median(),
            "q3":     a.quantile(0.75),
            "max":    a.max(),
            "n_sim":  len(a),
        })

    agg = (df.groupby(["block", "sampler"])
           .apply(_agg, include_groups=False)
           .reset_index())
    samplers_present = [s for s in SAMPLER_ORDER if s in agg["sampler"].unique()]
    agg["sampler"] = pd.Categorical(agg["sampler"], categories=samplers_present, ordered=True)
    agg["sampler"] = agg["sampler"].map(SAMPLER_LABELS)
    agg = agg.sort_values(["block", "sampler"])
    stat_cols = ["rmse_pooled", "min", "q1", "mean", "median", "q3", "max"]
    agg[stat_cols] = agg[stat_cols].round(4)
    return agg[["block", "sampler", *stat_cols, "n_sim"]].reset_index(drop=True)


def main():
    # Output dirs resolved at call time so tests can monkeypatch plot_recovery.DIR_FIG.
    out_delta_bias     = pr.DIR_FIG / "delta" / "bias"     / "tables"
    out_delta_sd       = pr.DIR_FIG / "delta" / "sd"       / "tables"
    out_delta_rmse     = pr.DIR_FIG / "delta" / "rmse"     / "tables"
    out_delta_coverage = pr.DIR_FIG / "delta" / "coverage" / "tables"
    out_runtime        = pr.DIR_FIG / "runtime" / "tables"
    out_beta_bias        = pr.DIR_FIG / "beta" / "bias"        / "tables"
    out_beta_sd          = pr.DIR_FIG / "beta" / "sd"          / "tables"
    out_beta_rmse        = pr.DIR_FIG / "beta" / "rmse"        / "tables"
    out_beta_correlation = pr.DIR_FIG / "beta" / "correlation" / "tables"
    out_beta_coverage    = pr.DIR_FIG / "beta" / "coverage"    / "tables"
    out_mu    = pr.DIR_FIG / "mu"    / "tables"
    out_sigma = pr.DIR_FIG / "sigma" / "tables"
    out_marginal = pr.DIR_FIG / "marginal_comparison"   # per-grid subfolder added below

    for d in (out_delta_bias, out_delta_sd, out_delta_rmse, out_delta_coverage,
              out_runtime, out_beta_bias, out_beta_sd, out_beta_rmse,
              out_beta_correlation, out_beta_coverage, out_mu, out_sigma):
        d.mkdir(parents=True, exist_ok=True)

    # --- Delta bias / MCSE table ---
    tbl = delta_bias_mcse_table()
    path = out_delta_bias / f"delta_bias_mcse_c{CHAINS}.csv"
    tbl.to_csv(path, index=False)
    print(f"wrote {len(tbl)} rows -> {path}")

    # --- Delta coverage table ---
    cov = delta_coverage_table()
    path = out_delta_coverage / f"delta_coverage_c{CHAINS}.csv"
    cov.to_csv(path, index=False)
    print(f"wrote {len(cov)} rows -> {path}")

    # --- Delta absolute error / RMSE table ---
    rmse = delta_rmse_summary_table()
    path = out_delta_rmse / f"delta_rmse_summary_c{CHAINS}.csv"
    rmse.to_csv(path, index=False)
    print(f"wrote {len(rmse)} rows -> {path}")

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

    # --- Beta bias / MCSE table ---
    bbias = beta_bias_mcse_table()
    path = out_beta_bias / f"beta_bias_mcse_c{CHAINS}.csv"
    bbias.to_csv(path, index=False)
    print(f"wrote {len(bbias)} rows -> {path}")

    # --- Beta posterior SD table (from beta_summary; the frame is reused for the
    # correlation table below so the large CSV is loaded only once). ---
    print("loading beta_summary.csv to compute posterior SDs + correlations ...")
    df_summary = load_recovery("beta_summary")
    sd_df = compute_beta_post_std(df_summary)
    bsd = beta_sd_summary_table(sd_df=sd_df)
    path = out_beta_sd / f"beta_sd_summary_c{CHAINS}.csv"
    bsd.to_csv(path, index=False)
    print(f"wrote {len(bsd)} rows -> {path}")

    # --- Beta RMSE table ---
    brmse = beta_rmse_summary_table()
    path = out_beta_rmse / f"beta_rmse_summary_c{CHAINS}.csv"
    brmse.to_csv(path, index=False)
    print(f"wrote {len(brmse)} rows -> {path}")

    # --- Beta correlation table (reuses the beta_summary loaded above) ---
    corr_df = compute_beta_correlation(df_summary)
    bcorr = beta_correlation_summary_table(corr_df=corr_df)
    path = out_beta_correlation / f"beta_correlation_summary_c{CHAINS}.csv"
    bcorr.to_csv(path, index=False)
    print(f"wrote {len(bcorr)} rows -> {path}")

    # --- Beta coverage table ---
    bcov = beta_coverage_table()
    path = out_beta_coverage / f"beta_coverage_c{CHAINS}.csv"
    bcov.to_csv(path, index=False)
    print(f"wrote {len(bcov)} rows -> {path}")

    # --- Mu recovery summary (bias / MCSE / coverage per param x sampler) ---
    mu = mu_recovery_summary_table()
    path = out_mu / f"mu_recovery_summary_c{CHAINS}.csv"
    mu.to_csv(path, index=False)
    print(f"wrote {len(mu)} rows -> {path}")

    # --- Sigma recovery summary (posterior vs true vs empirical, per element x sampler) ---
    sig = sigma_recovery_summary_table()
    path = out_sigma / f"sigma_recovery_summary_c{CHAINS}.csv"
    sig.to_csv(path, index=False)
    print(f"wrote {len(sig)} rows -> {path}")

    # --- Marginal-series convergence tables (R-hat + ESS summaries, one set per grid) ---
    marginal_diag.write_tables(CHAINS)

    # --- Consolidated RMSE (all elements of a block pooled per run), one CSV per
    # block in that block's own rmse/tables folder ---
    tbl = consolidated_rmse_table()
    for block, out_dir in [("beta", out_beta_rmse), ("delta", out_delta_rmse)]:
        sub = tbl[tbl["block"] == block].drop(columns="block")
        path = out_dir / f"{block}_consolidated_rmse_c{CHAINS}.csv"
        sub.to_csv(path, index=False)
        print(f"wrote {len(sub)} rows -> {path}")

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


if __name__ == "__main__":
    main()
