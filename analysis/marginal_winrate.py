"""
Head-to-head win rates on the marginal-distance metrics.

All samplers fit byte-identical data, so distances are PAIRED on (dataset_key, param):
for each dataset and MNL parameter, line up the challenger vs the baseline on the same
run and count how often the challenger's distance is smaller (lower distance = closer to
the true DGP marginal = better).

Unit of comparison: one (dataset_key, param) pair. Each dataset contributes 4 params, so
each k_true has 100 seeds x 4 params = 400 paired comparisons per metric.

For every (comparison, metric, k_true) it reports:
  n_pairs   - paired comparisons with BOTH values finite (KL can be +inf -> dropped)
  n_win     - challenger strictly closer (lower) than baseline
  n_tie     - exactly equal (essentially 0 for continuous metrics)
  win_rate  - n_win / (n_win + n_loss), ties excluded  [0.5 = coin flip]
  median_diff - median(challenger - baseline); negative = challenger better on average
  p_value   - two-sided sign test (binomial) that win_rate != 0.5

    .venv/Scripts/python.exe analysis/marginal_winrate.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from plotnine import (
    aes, element_text, facet_wrap, geom_hline, geom_line, geom_point, ggplot,
    labs, scale_color_manual, scale_y_continuous, theme, theme_bw,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_recovery import load_recovery, save, MARGINAL_METRICS, _PARAM_ORDER  # noqa: E402

CHAINS = 2
# (challenger, baseline): "how often does the challenger beat the baseline?"
COMPARISONS = [("nuts", "bayesm"), ("hmc", "bayesm"), ("nuts", "hmc")]
PAIR_KEYS = ["dataset_key", "param"]
FIG_DIR = Path(__file__).resolve().parent / "out" / "k5_results" / "marginal_comparison"
OUT_DIR = FIG_DIR / "tables"
# Fixed per-parameter palette so every win-rate figure is consistent.
PARAM_COLORS = {"Alt1": "#1b9e77", "Alt2": "#d95f02", "Alt3": "#7570b3", "Price": "#e7298a"}


def _winrate_rows(sub: pd.DataFrame, challenger: str, baseline: str, metric: str) -> list[dict]:
    """One row per k_true (+ an 'all' row) of challenger-vs-baseline win rate on `metric`."""
    wide = sub.pivot_table(index=["k_true", *PAIR_KEYS], columns="sampler",
                           values=metric, aggfunc="first")
    if challenger not in wide.columns or baseline not in wide.columns:
        return []
    pair = wide[[challenger, baseline]].dropna()                       # both finite (drops KL inf)
    pair = pair[np.isfinite(pair[challenger]) & np.isfinite(pair[baseline])]
    pair = pair.reset_index()
    diff = pair[challenger] - pair[baseline]

    rows = []
    for kt, grp in list(pair.groupby("k_true")) + [("all", pair)]:
        d = grp[challenger] - grp[baseline]
        n_win = int((d < 0).sum())
        n_loss = int((d > 0).sum())
        n_tie = int((d == 0).sum())
        n_eff = n_win + n_loss
        win_rate = n_win / n_eff if n_eff else np.nan
        p = binomtest(n_win, n_eff, 0.5).pvalue if n_eff else np.nan
        rows.append({
            "comparison": f"{challenger}_vs_{baseline}", "metric": metric,
            "k_true": kt, "n_pairs": len(grp), "n_win": n_win, "n_loss": n_loss,
            "n_tie": n_tie, "win_rate": round(win_rate, 4) if n_eff else np.nan,
            "median_diff": round(float(d.median()), 6), "p_value": p,
        })
    return rows


def win_rate_table(n_chains: int = CHAINS) -> pd.DataFrame:
    df = load_recovery("marginal_distances")
    df = df[df["n_chains"] == n_chains]
    rows = []
    for metric in MARGINAL_METRICS:
        for challenger, baseline in COMPARISONS:
            rows += _winrate_rows(df, challenger, baseline, metric)
    return pd.DataFrame(rows)


def _param_rows(sub: pd.DataFrame, challenger: str, baseline: str, metric: str) -> list[dict]:
    """RUN-level, per parameter: for each (k_true, param) count on how many datasets the
    challenger's distance is lower than the baseline's. No collapsing across params - one row
    per (k_true, param).

    n_total    = all datasets in the cell (~100).
    n_dropped  = datasets excluded because EITHER sampler's metric is non-finite (KL can be +inf
                 when a fitted marginal puts mass where the true DGP density is ~0 - a genuine
                 catastrophic-tail mismatch). n_inf_challenger / n_inf_baseline attribute those.
    n_datasets = finite pairs actually compared; the win_rate denominator.
    NOTE: dropped cases are almost always the challenger losing badly, so for KL the win_rate is
    optimistic for whichever sampler has the larger n_inf - read it against these columns."""
    wide = sub.pivot_table(index=["k_true", "param", "dataset_key"], columns="sampler",
                           values=metric, aggfunc="first")
    if challenger not in wide.columns or baseline not in wide.columns:
        return []
    pair = wide[[challenger, baseline]].reset_index()

    rows = []
    for (kt, param), grp in pair.groupby(["k_true", "param"]):
        fin_c = np.isfinite(grp[challenger])
        fin_b = np.isfinite(grp[baseline])
        both = fin_c & fin_b
        g = grp[both]
        d = g[challenger] - g[baseline]
        n_win = int((d < 0).sum())
        n_loss = int((d > 0).sum())
        n_tie = int((d == 0).sum())
        n_eff = n_win + n_loss
        win_rate = n_win / n_eff if n_eff else np.nan
        p = binomtest(n_win, n_eff, 0.5).pvalue if n_eff else np.nan
        rows.append({
            "comparison": f"{challenger}_vs_{baseline}", "metric": metric,
            "k_true": kt, "param": param, "n_total": len(grp), "n_datasets": len(g),
            "n_dropped": int((~both).sum()), "n_inf_challenger": int((~fin_c).sum()),
            "n_inf_baseline": int((~fin_b).sum()), "n_win": n_win, "n_loss": n_loss,
            "n_tie": n_tie, "win_rate": round(win_rate, 4) if n_eff else np.nan,
            "median_diff": round(float(d.median()), 6) if n_eff else np.nan, "p_value": p,
        })
    return rows


def nonfinite_count_table(n_chains: int = CHAINS) -> pd.DataFrame:
    """Standalone transparency table: number of non-finite (i.e. +inf) metric values per
    (metric, sampler, k_true, param). Effectively only KL is ever non-finite; the count is a
    direct measure of catastrophic tail mismatch, which is itself a finding."""
    df = load_recovery("marginal_distances")
    df = df[df["n_chains"] == n_chains]
    rows = []
    for metric in MARGINAL_METRICS:
        bad = df[~np.isfinite(df[metric])]
        if bad.empty:
            continue
        g = bad.groupby(["sampler", "k_true", "param"]).size()
        for (sampler, kt, param), n in g.items():
            total = len(df[(df["sampler"] == sampler) & (df["k_true"] == kt) & (df["param"] == param)])
            rows.append({"metric": metric, "sampler": sampler, "k_true": kt, "param": param,
                         "n_inf": int(n), "n_total": total,
                         "inf_rate": round(n / total, 4) if total else np.nan})
    return pd.DataFrame(rows)


def param_win_rate_table(n_chains: int = CHAINS) -> pd.DataFrame:
    df = load_recovery("marginal_distances")
    df = df[df["n_chains"] == n_chains]
    rows = []
    for metric in MARGINAL_METRICS:
        for challenger, baseline in COMPARISONS:
            rows += _param_rows(df, challenger, baseline, metric)
    return pd.DataFrame(rows)


def win_rate_plot(comparison: str, tbl: pd.DataFrame) -> ggplot:
    """Win rate vs k_true, one line per param, faceted by metric, for one comparison.

    y = share of the ~100 datasets where the challenger's distance is lower than the baseline's.
    Dashed line at 0.5 = coin flip; above = challenger wins the majority of datasets.
    """
    chall, base = comparison.split("_vs_")
    sub = tbl[tbl["comparison"] == comparison].copy()
    if sub.empty:
        raise ValueError(f"No rows for comparison {comparison}.")
    sub["k_true"] = sub["k_true"].astype(int)
    metric_order = [m for m in MARGINAL_METRICS if m in set(sub["metric"])]
    param_order = [p for p in _PARAM_ORDER if p in set(sub["param"])]
    sub["metric"] = pd.Categorical(sub["metric"], categories=metric_order, ordered=True)
    sub["param"] = pd.Categorical(sub["param"], categories=param_order, ordered=True)

    return (
        ggplot(sub, aes(x="k_true", y="win_rate", color="param", group="param"))
        + geom_hline(yintercept=0.5, linetype="dashed", color="#7f7f7f")
        + geom_line(size=0.7)
        + geom_point(size=1.8)
        + facet_wrap("metric", ncol=5)
        + scale_color_manual(values=[PARAM_COLORS[p] for p in param_order])
        + scale_y_continuous(limits=[0, 1], breaks=[0, 0.25, 0.5, 0.75, 1.0])
        + labs(x="True Components (k_true)", y=f"Share of datasets where {chall} < {base}",
               color="Parameter",
               title=f"Marginal-distance win rate: {chall} vs {base} (c{CHAINS}), by parameter")
        + theme_bw()
        + theme(figure_size=(15, 3.6), plot_title=element_text(size=11),
                axis_text_x=element_text(size=9))
    )


def write_wide_tables(tbl: pd.DataFrame) -> None:
    """Per-comparison wide CSVs: index (metric, param), columns k_true, one for win counts
    ('n_win/n_datasets') and one for the win_rate fraction."""
    for comp in tbl["comparison"].unique():
        m = tbl[tbl["comparison"] == comp].copy()
        m["cell"] = m["n_win"].astype(str) + "/" + m["n_datasets"].astype(str)
        m["metric"] = pd.Categorical(m["metric"], categories=MARGINAL_METRICS, ordered=True)
        m["param"] = pd.Categorical(m["param"], categories=_PARAM_ORDER, ordered=True)
        for value, tag in [("cell", "counts"), ("win_rate", "rate")]:
            wide = (m.pivot(index=["metric", "param"], columns="k_true", values=value)
                    .sort_index())
            wide.to_csv(OUT_DIR / f"win_rates_by_param_{comp}_{tag}_c{CHAINS}.csv")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Pooled-over-params view (400 comparisons per k_true), kept for reference.
    pooled = win_rate_table(CHAINS)
    pooled.to_csv(OUT_DIR / f"win_rates_c{CHAINS}.csv", index=False)

    # RUN-level, per parameter: n_win out of 100 datasets, one row per (comparison, metric, k_true, param).
    tbl = param_win_rate_table(CHAINS)
    path = OUT_DIR / f"win_rates_by_param_c{CHAINS}.csv"
    tbl.to_csv(path, index=False)
    write_wide_tables(tbl)                                              # per-comparison wide CSVs
    print(f"wrote {path}  ({len(tbl)} rows) + wide tables\n")

    # Non-finite (+inf) transparency table: how many catastrophic-tail cases were dropped.
    nf = nonfinite_count_table(CHAINS)
    nf_path = OUT_DIR / f"nonfinite_counts_c{CHAINS}.csv"
    nf.to_csv(nf_path, index=False)
    if not nf.empty:
        print("=== non-finite (+inf) metric values dropped from win rates ===")
        for metric in nf["metric"].unique():
            piv = (nf[nf["metric"] == metric]
                   .pivot_table(index="param", columns=["sampler", "k_true"],
                                values="n_inf", fill_value=0, aggfunc="sum")
                   .reindex([p for p in ["Alt1", "Alt2", "Alt3", "Price"]]))
            print(f"  -- {metric}: +inf counts (out of ~100 datasets per sampler/k_true) --")
            print(piv.to_string(), "\n")

    # Plots: one figure per comparison (win rate vs k_true, line per param, faceted by metric).
    for challenger, baseline in COMPARISONS:
        comp = f"{challenger}_vs_{baseline}"
        save(win_rate_plot(comp, tbl), f"marginal_comparison/plots/win_rate_{comp}_c{CHAINS}.png")
    print("wrote win-rate plots -> marginal_comparison/plots/\n")

    # Console summary: for each comparison + metric, n_win out of n_datasets per (k_true, param).
    for comp in tbl["comparison"].unique():
        chall = comp.split("_vs_")[0]
        print(f"########## {comp}  (datasets where {chall} is closer, out of ~100 per k_true) ##########")
        for metric in MARGINAL_METRICS:
            m = tbl[(tbl["comparison"] == comp) & (tbl["metric"] == metric)].copy()
            if m.empty:
                continue
            m["cell"] = m["n_win"].astype(str) + "/" + m["n_datasets"].astype(str)
            piv = m.pivot(index="param", columns="k_true", values="cell")
            piv = piv.reindex([p for p in ["Alt1", "Alt2", "Alt3", "Price"] if p in piv.index])
            piv = piv[[c for c in [1, 2, 3, 5] if c in piv.columns]]
            print(f"  -- {metric} --")
            print(piv.to_string(), "\n")


if __name__ == "__main__":
    main()
