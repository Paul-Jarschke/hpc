# How many of the k_model=5 slots are real? jobs 100-103, from the
# after-ECR weights.csv; pre-ECR means are label-smeared (overcount).
# K_eff = 1/sum(w^2); est_k = #slots with mean weight >= tau.
# NB a=1 Dirichlet won't empty extras (Rousseau-Mengersen a<d/2).
# run: .venv/Scripts/python.exe hpc_analysis/mixture_models/component_count.py
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from plotnine import (
    aes,
    element_text,
    facet_grid,
    facet_wrap,
    geom_boxplot,
    geom_col,
    geom_line,
    geom_point,
    geom_ribbon,
    geom_vline,
    ggplot,
    labeller,
    labs,
    position_dodge,
    scale_color_manual,
    scale_fill_manual,
    scale_x_continuous,
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
# Study-wide conventions for this analysis.
# --------------------------------------------------------------------------------- #
THRESHOLDS: tuple[float, ...] = (0.01, 0.025, 0.05)  # weight cutoffs for the integer count
PRIMARY_THRESHOLD: float = 0.05                       # the headline tau
KTRUE_ORDER = [1, 2, 3, 5]
K_MODEL = 5                                            # every run is overspecified to 5 slots

COND_COLS = ["dataset_key", "scenario", "k_true", "data_seed", "k_model", "sampler", "n_chains"]

DIR_OUT_PLOTS = "components/plots"
DIR_OUT_TABLES = DIR_FIG / "components" / "tables"


# per-mille suffix so nearby taus stay distinct: 0.01 -> 'est_k_t010'
def _thr_col(t: float) -> str:
    return f"est_k_t{int(round(t * 1000)):03d}"


PRIMARY_COL = _thr_col(PRIMARY_THRESHOLD)
CI_COL = _thr_col(PRIMARY_THRESHOLD) + "_ci"


# --------------------------------------------------------------------------------- #
# Per-run derivation: one row per fit with both lenses.
# --------------------------------------------------------------------------------- #
# one run = one 5-row group of weights.csv (after-ECR slot weights)
def _run_metrics(g: pd.DataFrame, thresholds: Sequence[float], primary: float) -> pd.Series:
    w = np.clip(g["post_mean"].to_numpy(dtype=float), 0.0, None)
    s = w.sum()
    wn = w / s if s > 0 else w                                  # normalize defensively (sums ~1)
    p = wn[wn > 0]
    out = {
        "k_eff":         float(1.0 / np.sum(wn ** 2)) if s > 0 else np.nan,      # inverse Simpson
        "k_eff_shannon": float(np.exp(-np.sum(p * np.log(p)))) if s > 0 else np.nan,
    }
    for t in thresholds:
        out[_thr_col(t)] = int((w >= t).sum())
    clow = g["ci_low"].to_numpy(dtype=float)
    out[_thr_col(primary) + "_ci"] = int((clow >= primary).sum())               # confidently non-empty
    return pd.Series(out)


# one row per run: k_eff (inv Simpson + Shannon) + est_k per tau.
# must use the ECR-relabeled weights; raw slot means over-count.
def per_run_counts(df: Optional[pd.DataFrame] = None, *, n_chains: Optional[int] = None,
                   thresholds: Sequence[float] = THRESHOLDS,
                   primary: float = PRIMARY_THRESHOLD) -> pd.DataFrame:
    df = load_recovery("weights") if df is None else df
    if n_chains is not None:
        df = df[df["n_chains"] == n_chains]
    if df.empty:
        raise ValueError(f"No weights rows (n_chains={n_chains}).")
    cond = [c for c in COND_COLS if c in df.columns]
    agg = (
        df.groupby(cond, observed=True, sort=False)
        .apply(_run_metrics, thresholds, primary, include_groups=False)
        .reset_index()
    )
    # est_k columns come back as float from the Series; restore integer dtype.
    for c in agg.columns:
        if c.startswith("est_k_"):
            agg[c] = agg[c].astype(int)
    return agg


# --------------------------------------------------------------------------------- #
# Aggregation tables (per sampler x k_true).
# --------------------------------------------------------------------------------- #
# headline: K_eff stats + est_k accuracy per (k_true, sampler);
# 'correct' = estimated count == k_true (at the primary tau).
def recovery_summary_table(n_chains: int = 2, runs: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    runs = per_run_counts(n_chains=n_chains) if runs is None else runs.copy()
    kt = runs["k_true"].to_numpy()
    est = runs[PRIMARY_COL].to_numpy()
    runs = runs.assign(
        correct=(est == kt), over=(est > kt), under=(est < kt),
        keff_correct=(np.rint(runs["k_eff"]).astype("Int64") == runs["k_true"]),
    )

    def _agg(gp: pd.DataFrame) -> pd.Series:
        return pd.Series({
            "n_sim":            len(gp),
            "mean_k_eff":       gp["k_eff"].mean(),
            "median_k_eff":     gp["k_eff"].median(),
            "sd_k_eff":         gp["k_eff"].std(ddof=1),
            "mean_est_k":       gp[PRIMARY_COL].mean(),
            "frac_correct":     gp["correct"].mean(),
            "frac_over":        gp["over"].mean(),
            "frac_under":       gp["under"].mean(),
            "frac_correct_keff": gp["keff_correct"].mean(),
            "mean_est_k_ci":    gp[CI_COL].mean(),
        })

    out = (runs.groupby(["k_true", "sampler"], observed=True).apply(_agg, include_groups=False)
           .reset_index())
    out = _order_samplers(out)
    num = ["mean_k_eff", "median_k_eff", "sd_k_eff", "mean_est_k",
           "frac_correct", "frac_over", "frac_under", "frac_correct_keff", "mean_est_k_ci"]
    out[num] = out[num].round(3)
    return out.sort_values(["k_true", "sampler"]).reset_index(drop=True)


# p_estJ = fraction of seeds with est_k == J; rows sum to 1.
def confusion_table(n_chains: int = 2, runs: Optional[pd.DataFrame] = None,
                    est_col: str = PRIMARY_COL) -> pd.DataFrame:
    runs = per_run_counts(n_chains=n_chains) if runs is None else runs.copy()
    frac = (
        runs.groupby(["k_true", "sampler"], observed=True)[est_col]
        .value_counts(normalize=True).rename("frac").reset_index()
    )
    wide = (frac.pivot_table(index=["k_true", "sampler"], columns=est_col, values="frac")
            .reindex(columns=range(1, K_MODEL + 1)).fillna(0.0))
    wide.columns = [f"p_est{int(c)}" for c in wide.columns]
    wide = _order_samplers(wide.reset_index())
    val = [c for c in wide.columns if c.startswith("p_est")]
    wide[val] = wide[val].round(3)
    return wide.sort_values(["k_true", "sampler"]).reset_index(drop=True)


# mean_est_k / frac_correct vs tau, long over swept thresholds.
def threshold_sensitivity_table(n_chains: int = 2,
                                runs: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    runs = per_run_counts(n_chains=n_chains) if runs is None else runs.copy()
    rows = []
    for t in THRESHOLDS:
        col = _thr_col(t)
        tmp = runs.assign(threshold=t, est_k=runs[col], correct=(runs[col] == runs["k_true"]))
        rows.append(tmp[["k_true", "sampler", "threshold", "est_k", "correct"]])
    long = pd.concat(rows, ignore_index=True)
    out = (long.groupby(["k_true", "sampler", "threshold"], observed=True)
           .agg(mean_est_k=("est_k", "mean"), frac_correct=("correct", "mean"),
                n_sim=("est_k", "size")).reset_index())
    out = _order_samplers(out)
    out[["mean_est_k", "frac_correct"]] = out[["mean_est_k", "frac_correct"]].round(3)
    return out.sort_values(["k_true", "sampler", "threshold"]).reset_index(drop=True)


def _order_samplers(df: pd.DataFrame) -> pd.DataFrame:
    order = [s for s in SAMPLER_ORDER if s in set(df["sampler"])]
    df = df.copy()
    df["sampler"] = pd.Categorical(df["sampler"], categories=order, ordered=True)
    return df


# --------------------------------------------------------------------------------- #
# Plots.
# --------------------------------------------------------------------------------- #
# K_eff by k_true; one dodged box per sampler over replicate seeds.
def plot_k_eff_by_ktrue(n_chains: int = 2, runs: Optional[pd.DataFrame] = None) -> ggplot:
    runs = per_run_counts(n_chains=n_chains) if runs is None else runs.copy()
    sampler_order = [s for s in SAMPLER_ORDER if s in set(runs["sampler"])]
    ktrue_order = [str(k) for k in KTRUE_ORDER if k in set(runs["k_true"])]
    runs["sampler"] = pd.Categorical(runs["sampler"], categories=sampler_order, ordered=True)
    runs["k_true"] = pd.Categorical(runs["k_true"].astype(str), categories=ktrue_order, ordered=True)

    counts = runs.groupby(["k_true", "sampler"], observed=True)["data_seed"].nunique().to_dict()
    print(f"[plot_k_eff_by_ktrue] c{n_chains}: seeds/box={counts}")

    color_vals = [SAMPLER_COLORS[s] for s in sampler_order]
    dodge = position_dodge(width=0.8)

    p = (
        ggplot(runs, aes(x="k_true", y="k_eff", color="sampler"))
        + geom_boxplot(width=0.6, fill="#FFFFFF00", outlier_alpha=0.35, position=dodge)
        + scale_color_manual(values=color_vals,
                             labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
        + labs(x="k_true",
               y="K_eff (1/Σwₖ²)", color="Sampler",
               title="K_eff vs k_true")
        + theme_bw()
        + theme(figure_size=(9, 5), axis_text_x=element_text(size=9),
                plot_title=element_text(size=11))
    )
    return p


# long P(est_k=j) frame, missing cells filled 0 so bars align.
def _confusion_frac(runs: pd.DataFrame, thresholds: Sequence[float]) -> pd.DataFrame:
    kts, samps = sorted(runs["k_true"].unique()), sorted(runs["sampler"].unique())
    parts = []
    for t in thresholds:
        col = _thr_col(t)
        f = (runs.groupby(["k_true", "sampler"], observed=True)[col]
             .value_counts(normalize=True).rename("frac").reset_index()
             .rename(columns={col: "est_k"}))
        f["threshold"] = t
        parts.append(f)
    long = pd.concat(parts, ignore_index=True)
    full = pd.MultiIndex.from_product(
        [list(thresholds), kts, samps, range(1, K_MODEL + 1)],
        names=["threshold", "k_true", "sampler", "est_k"],
    ).to_frame(index=False)
    return full.merge(long, on=["threshold", "k_true", "sampler", "est_k"], how="left").fillna(
        {"frac": 0.0})


# est_k distribution per k_true panel, single tau. dashed line =
# correct count; mass to its right = over-counting.
def plot_est_k_confusion(n_chains: int = 2, threshold: float = PRIMARY_THRESHOLD,
                         runs: Optional[pd.DataFrame] = None) -> ggplot:
    runs = per_run_counts(n_chains=n_chains) if runs is None else runs.copy()
    frac = _confusion_frac(runs, [threshold])
    sampler_order = [s for s in SAMPLER_ORDER if s in set(frac["sampler"])]
    frac["sampler"] = pd.Categorical(frac["sampler"], categories=sampler_order, ordered=True)
    ref = frac[["k_true"]].drop_duplicates().assign(x=lambda d: d["k_true"])
    color_vals = [SAMPLER_COLORS[s] for s in sampler_order]

    p = (
        ggplot(frac, aes(x="est_k", y="frac", fill="sampler"))
        + geom_vline(ref, aes(xintercept="x"), linetype="dashed", color="#555555",
                     size=0.7, inherit_aes=False)
        + geom_col(position=position_dodge(width=0.8), width=0.75)
        + facet_wrap("k_true", ncol=4, labeller="label_both")
        + scale_x_continuous(breaks=list(range(1, K_MODEL + 1)))
        + scale_fill_manual(values=color_vals,
                            labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
        + labs(x=f"Est. # components (w ≥ {threshold:g})",
               y="Fraction of seeds", fill="Sampler",
               title=f"Recovered Count vs k_true (τ={threshold:g})")
        + theme_bw()
        + theme(figure_size=(14, 4.5), axis_text_x=element_text(size=9),
                plot_title=element_text(size=11))
    )
    return p


# tau (rows, shrinking downward) x k_true (cols) confusion grid;
# lower tau keeps more spurious comps -> mass shifts right of truth.
def plot_est_k_confusion_all_tau(n_chains: int = 2, thresholds: Sequence[float] = THRESHOLDS,
                                 runs: Optional[pd.DataFrame] = None) -> ggplot:
    runs = per_run_counts(n_chains=n_chains) if runs is None else runs.copy()
    frac = _confusion_frac(runs, list(thresholds))
    sampler_order = [s for s in SAMPLER_ORDER if s in set(frac["sampler"])]
    frac["sampler"] = pd.Categorical(frac["sampler"], categories=sampler_order, ordered=True)
    # Label rows "τ = 0.05" and order them so tau SHRINKS top->bottom (over-counting grows
    # as you read down). label_value avoids double-printing the pre-formatted label.
    tau_labels = [f"τ = {t:g}" for t in sorted(thresholds, reverse=True)]
    frac["threshold"] = pd.Categorical(
        frac["threshold"].map(lambda t: f"τ = {t:g}"), categories=tau_labels, ordered=True)
    ref = frac[["k_true"]].drop_duplicates().assign(x=lambda d: d["k_true"])
    color_vals = [SAMPLER_COLORS[s] for s in sampler_order]

    p = (
        ggplot(frac, aes(x="est_k", y="frac", fill="sampler"))
        + geom_vline(ref, aes(xintercept="x"), linetype="dashed", color="#555555",
                     size=0.7, inherit_aes=False)
        + geom_col(position=position_dodge(width=0.8), width=0.75)
        + facet_grid(rows="threshold", cols="k_true", labeller="label_value")
        + scale_x_continuous(breaks=list(range(1, K_MODEL + 1)))
        + scale_fill_manual(values=color_vals,
                            labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
        + labs(x="Est. # components (w ≥ τ)", y="Fraction of seeds", fill="Sampler",
               title="Recovered Count vs k_true and τ")
        + theme_bw()
        + theme(figure_size=(14, 8), axis_text_x=element_text(size=8),
                plot_title=element_text(size=11))
    )
    return p


# per-slot mean + IQR band across seeds
def _weight_profile_frame(df: pd.DataFrame, group_extra: list) -> pd.DataFrame:
    g = ["k_true", "sampler", "slot", *group_extra]
    return (df.groupby(g, observed=True)["value"]
            .agg(mean="mean",
                 lo=lambda s: s.quantile(0.25),
                 hi=lambda s: s.quantile(0.75))
            .reset_index())


# after-ECR weight per slot, line+IQR per sampler; black dashed =
# true profile (1/k_true then 0). shows where the weights fall off.
def plot_weight_profile_by_ktrue(n_chains: int = 2, df: Optional[pd.DataFrame] = None) -> ggplot:
    df = load_recovery("weights") if df is None else df
    sub = df[df["n_chains"] == n_chains].copy()
    if sub.empty:
        raise ValueError(f"No weights rows for n_chains={n_chains}.")
    sub["value"] = sub["post_mean"]
    prof = _weight_profile_frame(sub, group_extra=[])

    sampler_order = [s for s in SAMPLER_ORDER if s in set(prof["sampler"])]
    prof["sampler"] = pd.Categorical(prof["sampler"], categories=sampler_order, ordered=True)
    color_vals = [SAMPLER_COLORS[s] for s in sampler_order]

    # True weight per slot (constant across seeds; 0 where NaN i.e. spurious slots).
    truth = (sub.groupby(["k_true", "slot"], observed=True)["true_weight"].first()
             .fillna(0.0).reset_index().rename(columns={"true_weight": "mean"}))

    p = (
        ggplot(prof, aes(x="slot", y="mean", color="sampler", fill="sampler"))
        + geom_ribbon(aes(ymin="lo", ymax="hi"), alpha=0.15, color="none")
        + geom_line(size=0.8)
        + geom_point(size=1.6)
        + geom_line(truth, aes(x="slot", y="mean"), color="#000000", linetype="dashed",
                    inherit_aes=False)
        + facet_wrap("k_true", ncol=4, labeller="label_both")
        + scale_x_continuous(breaks=list(range(K_MODEL)))
        + scale_color_manual(values=color_vals,
                             labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
        + scale_fill_manual(values=color_vals,
                            labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
        + labs(x="Component slot (descending weight)", y="Mean weight (π̂)",
               color="Sampler", fill="Sampler",
               title="Weight Profile (after ECR)")
        + theme_bw()
        + theme(figure_size=(14, 4.5), axis_text_x=element_text(size=9),
                plot_title=element_text(size=11))
    )
    return p


# before vs after ECR (source: pvec_means.csv). before relabeling,
# label switching smears slot means toward uniform - all 5 slots
# look active; after, true comps separate from the near-zero ones.
def plot_weight_profile_before_after(n_chains: int = 2,
                                     df: Optional[pd.DataFrame] = None) -> ggplot:
    df = load_recovery("pvec_means") if df is None else df
    sub = df[df["n_chains"] == n_chains].copy()
    if sub.empty:
        raise ValueError(f"No pvec_means rows for n_chains={n_chains}.")
    sub = sub.rename(columns={"rank": "slot", "pvec_mean": "value"})
    prof = _weight_profile_frame(sub, group_extra=["stage"])
    prof["stage"] = prof["stage"].map({"before": "Before", "after": "After"})
    prof["stage"] = pd.Categorical(prof["stage"], categories=["Before", "After"], ordered=True)

    sampler_order = [s for s in SAMPLER_ORDER if s in set(prof["sampler"])]
    prof["sampler"] = pd.Categorical(prof["sampler"], categories=sampler_order, ordered=True)
    color_vals = [SAMPLER_COLORS[s] for s in sampler_order]

    p = (
        ggplot(prof, aes(x="slot", y="mean", color="sampler"))
        + geom_line(size=0.8)
        + geom_point(size=1.4)
        + facet_grid(rows="k_true", cols="stage",
                     labeller=labeller(rows="label_both", cols="label_value"))
        + scale_x_continuous(breaks=list(range(K_MODEL)))
        + scale_color_manual(values=color_vals,
                             labels=[SAMPLER_LABELS.get(s, s) for s in sampler_order])
        + labs(x="Component slot (sorted by descending weight)", y="Mean weight (π̂)",
               color="Sampler",
               title="Weight Profile: Before vs After ECR")
        + theme_bw()
        + theme(figure_size=(10, 10), axis_text_x=element_text(size=8),
                plot_title=element_text(size=11))
    )
    return p


# --------------------------------------------------------------------------------- #
# Entry point: write the tables and figures for this analysis.
# --------------------------------------------------------------------------------- #
def write_tables(n_chains: int = 2) -> None:
    DIR_OUT_TABLES.mkdir(parents=True, exist_ok=True)
    runs = per_run_counts(n_chains=n_chains)

    specs = {
        f"component_recovery_summary_c{n_chains}.csv": recovery_summary_table(n_chains, runs),
        f"component_confusion_c{n_chains}.csv":        confusion_table(n_chains, runs),
        f"component_threshold_sensitivity_c{n_chains}.csv": threshold_sensitivity_table(n_chains, runs),
    }
    for name, tbl in specs.items():
        path = DIR_OUT_TABLES / name
        tbl.to_csv(path, index=False)
        print(f"wrote {len(tbl)} rows -> {path}")


def make_plots(n_chains: int = 2) -> None:
    runs = per_run_counts(n_chains=n_chains)
    print("wrote", save(plot_k_eff_by_ktrue(n_chains, runs),
                        f"{DIR_OUT_PLOTS}/k_eff_by_ktrue.png"))
    # Confusion at the primary tau, plus all swept thresholds in one threshold x k_true grid.
    print("wrote", save(plot_est_k_confusion(n_chains, PRIMARY_THRESHOLD, runs),
                        f"{DIR_OUT_PLOTS}/est_k_confusion.png"))
    print("wrote", save(plot_est_k_confusion_all_tau(n_chains, THRESHOLDS, runs),
                        f"{DIR_OUT_PLOTS}/est_k_confusion_all_tau.png"))
    print("wrote", save(plot_weight_profile_by_ktrue(n_chains),
                        f"{DIR_OUT_PLOTS}/weight_profile.png"))
    print("wrote", save(plot_weight_profile_before_after(n_chains),
                        f"{DIR_OUT_PLOTS}/weight_profile_before_after.png"))


def main() -> None:
    write_tables()
    make_plots()
    print("component-count tables + figures -> hpc_analysis/mixture_models/out/components/")


if __name__ == "__main__":
    main()
