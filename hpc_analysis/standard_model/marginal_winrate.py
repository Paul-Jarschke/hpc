"""
Head-to-head win rates on the marginal-distance metrics, standard_model.

All samplers fit byte-identical data, so distances are PAIRED on (dataset_key, param):
for each dataset and MNL parameter, line up the challenger vs the baseline on the same
run and count how often the challenger's distance is smaller (lower distance = closer to
the true DGP marginal = better).

Unit of comparison: one (dataset_key, param) pair. Each dataset contributes 4 params, so
100 seeds x 4 params = 400 paired comparisons per metric (pooled table); the per-param
table keeps one row per param (100 paired datasets each).

Comparisons are ALL pairwise combinations of the three samplers, generated from
SAMPLER_ORDER (bayesm, nuts, hmc): the LATER sampler in the order challenges the
EARLIER one, so bayesm (the reference) is always a baseline.

Every distance metric is stored on TWO evaluation grids ('full' and 'chebyshev', keyed
by the `grid` column); all tables and plots are produced once per grid (full/trimmed
output subfolder).

For every (comparison, metric) it reports:
  n_pairs   - paired comparisons with BOTH values finite (KL can be +inf -> dropped)
  n_win     - challenger strictly closer (lower) than baseline
  n_tie     - exactly equal (essentially 0 for continuous metrics)
  win_rate  - n_win / (n_win + n_loss), ties excluded  [0.5 = coin flip]
  median_diff - median(challenger - baseline); negative = challenger better on average
  p_value   - two-sided sign test (binomial) that win_rate != 0.5

    .venv/Scripts/python.exe hpc_analysis/standard_model/marginal_winrate.py
"""

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from plotnine import (
    aes, element_text, facet_wrap, geom_col, geom_hline, ggplot,
    labs, scale_fill_manual, scale_y_continuous, theme, theme_bw,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import plot_recovery as pr  # noqa: E402  (module ref so DIR_FIG stays patchable)
from plot_recovery import (  # noqa: E402
    load_recovery, save, MARGINAL_METRICS, SAMPLER_ORDER, _PARAM_ORDER,
)

CHAINS = 2
GRIDS = ["full", "chebyshev"]  # marginal-distance evaluation-grid scenarios
GRID_FOLDER = {"full": "full", "chebyshev": "trimmed"}  # output subfolder per grid
# (challenger, baseline): "how often does the challenger beat the baseline?" All pairwise
# combinations of the three samplers; the later sampler in SAMPLER_ORDER challenges the
# earlier one (bayesm, first in the order, is always a baseline):
# nuts_vs_bayesm, hmc_vs_bayesm, hmc_vs_nuts.
COMPARISONS = [(b, a) for a, b in itertools.combinations(SAMPLER_ORDER, 2)]
PAIR_KEYS = ["dataset_key", "param"]


def _out_dir(grid):
    return pr.DIR_FIG / "marginal_comparison" / GRID_FOLDER[grid] / "tables"


# Fixed per-parameter palette so every win-rate figure is consistent.
PARAM_COLORS = {"Alt1": "#1b9e77", "Alt2": "#d95f02", "Alt3": "#7570b3", "Price": "#e7298a"}


def _load_distances(n_chains: int, grid: str) -> pd.DataFrame:
    """marginal_distances filtered to one evaluation-grid scenario + chain count."""
    df = load_recovery("marginal_distances")
    if "grid" in df.columns:
        df = df[df["grid"] == grid]
    return df[df["n_chains"] == n_chains]


def _winrate_rows(sub: pd.DataFrame, challenger: str, baseline: str, metric: str) -> list[dict]:
    """One pooled-over-params row of challenger-vs-baseline win rate on `metric`."""
    wide = sub.pivot_table(index=PAIR_KEYS, columns="sampler",
                           values=metric, aggfunc="first")
    if challenger not in wide.columns or baseline not in wide.columns:
        return []
    pair = wide[[challenger, baseline]].dropna()                       # both finite (drops KL inf)
    pair = pair[np.isfinite(pair[challenger]) & np.isfinite(pair[baseline])]
    pair = pair.reset_index()
    d = pair[challenger] - pair[baseline]

    n_win = int((d < 0).sum())
    n_loss = int((d > 0).sum())
    n_tie = int((d == 0).sum())
    n_eff = n_win + n_loss
    win_rate = n_win / n_eff if n_eff else np.nan
    p = binomtest(n_win, n_eff, 0.5).pvalue if n_eff else np.nan
    return [{
        "comparison": f"{challenger}_vs_{baseline}", "metric": metric,
        "n_pairs": len(pair), "n_win": n_win, "n_loss": n_loss,
        "n_tie": n_tie, "win_rate": round(win_rate, 4) if n_eff else np.nan,
        "median_diff": round(float(d.median()), 6) if n_eff else np.nan, "p_value": p,
    }]


def win_rate_table(n_chains: int = CHAINS, grid: str = "chebyshev") -> pd.DataFrame:
    df = _load_distances(n_chains, grid)
    rows = []
    for metric in MARGINAL_METRICS:
        for challenger, baseline in COMPARISONS:
            rows += _winrate_rows(df, challenger, baseline, metric)
    return pd.DataFrame(rows)


def _param_rows(sub: pd.DataFrame, challenger: str, baseline: str, metric: str) -> list[dict]:
    """RUN-level, per parameter: for each param count on how many datasets the challenger's
    distance is lower than the baseline's. No collapsing across params - one row per param.

    n_total    = all datasets in the cell (~100).
    n_dropped  = datasets excluded because EITHER sampler's metric is non-finite (KL can be +inf
                 when a fitted marginal puts mass where the true DGP density is ~0 - a genuine
                 catastrophic-tail mismatch). n_inf_challenger / n_inf_baseline attribute those.
    n_datasets = finite pairs actually compared; the win_rate denominator.
    NOTE: dropped cases are almost always the challenger losing badly, so for KL the win_rate is
    optimistic for whichever sampler has the larger n_inf - read it against these columns."""
    wide = sub.pivot_table(index=["param", "dataset_key"], columns="sampler",
                           values=metric, aggfunc="first")
    if challenger not in wide.columns or baseline not in wide.columns:
        return []
    pair = wide[[challenger, baseline]].reset_index()

    rows = []
    for param, grp in pair.groupby("param"):
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
            "param": param, "n_total": len(grp), "n_datasets": len(g),
            "n_dropped": int((~both).sum()), "n_inf_challenger": int((~fin_c).sum()),
            "n_inf_baseline": int((~fin_b).sum()), "n_win": n_win, "n_loss": n_loss,
            "n_tie": n_tie, "win_rate": round(win_rate, 4) if n_eff else np.nan,
            "median_diff": round(float(d.median()), 6) if n_eff else np.nan, "p_value": p,
        })
    return rows


def nonfinite_count_table(n_chains: int = CHAINS, grid: str = "chebyshev") -> pd.DataFrame:
    """Standalone transparency table: number of non-finite (i.e. +inf) metric values per
    (metric, sampler, param). Effectively only KL is ever non-finite; the count is a
    direct measure of catastrophic tail mismatch, which is itself a finding."""
    df = _load_distances(n_chains, grid)
    rows = []
    for metric in MARGINAL_METRICS:
        bad = df[~np.isfinite(df[metric])]
        if bad.empty:
            continue
        g = bad.groupby(["sampler", "param"]).size()
        for (sampler, param), n in g.items():
            total = len(df[(df["sampler"] == sampler) & (df["param"] == param)])
            rows.append({"metric": metric, "sampler": sampler, "param": param,
                         "n_inf": int(n), "n_total": total,
                         "inf_rate": round(n / total, 4) if total else np.nan})
    return pd.DataFrame(rows)


def param_win_rate_table(n_chains: int = CHAINS, grid: str = "chebyshev") -> pd.DataFrame:
    df = _load_distances(n_chains, grid)
    rows = []
    for metric in MARGINAL_METRICS:
        for challenger, baseline in COMPARISONS:
            rows += _param_rows(df, challenger, baseline, metric)
    return pd.DataFrame(rows)


def win_rate_plot(comparison: str, tbl: pd.DataFrame, grid: str = "") -> ggplot:
    """Win rate per param, faceted by metric, for one comparison.

    y = share of the ~100 datasets where the challenger's distance is lower than the
    baseline's; one bar per param. Dashed line at 0.5 = coin flip; above = the challenger
    wins the majority of datasets. `grid` only annotates the title (the table is already
    per-grid).
    """
    chall, base = comparison.split("_vs_")
    sub = tbl[tbl["comparison"] == comparison].copy()
    if sub.empty:
        raise ValueError(f"No rows for comparison {comparison}.")
    metric_order = [m for m in MARGINAL_METRICS if m in set(sub["metric"])]
    param_order = [p for p in _PARAM_ORDER if p in set(sub["param"])]
    sub["metric"] = pd.Categorical(sub["metric"], categories=metric_order, ordered=True)
    sub["param"] = pd.Categorical(sub["param"], categories=param_order, ordered=True)
    gtag = f" ({grid} grid)" if grid else ""

    return (
        ggplot(sub, aes(x="param", y="win_rate", fill="param"))
        + geom_col(width=0.7)
        + geom_hline(yintercept=0.5, linetype="dashed", color="#7f7f7f")
        + facet_wrap("metric", ncol=5)
        + scale_fill_manual(values=[PARAM_COLORS[p] for p in param_order])
        + scale_y_continuous(limits=[0, 1], breaks=[0, 0.25, 0.5, 0.75, 1.0])
        + labs(x="Parameter", y=f"Share of datasets where {chall} < {base}",
               fill="Parameter",
               title=f"Marginal-distance win rate: {chall} vs {base}{gtag}, by parameter")
        + theme_bw()
        + theme(figure_size=(15, 3.6), plot_title=element_text(size=11),
                axis_text_x=element_text(size=8, rotation=30))
    )


def write_wide_tables(tbl: pd.DataFrame, grid: str) -> None:
    """Per-comparison wide CSVs: index metric, columns param, one for win counts
    ('n_win/n_datasets') and one for the win_rate fraction."""
    for comp in tbl["comparison"].unique():
        m = tbl[tbl["comparison"] == comp].copy()
        m["cell"] = m["n_win"].astype(str) + "/" + m["n_datasets"].astype(str)
        m["metric"] = pd.Categorical(m["metric"], categories=MARGINAL_METRICS, ordered=True)
        m["param"] = pd.Categorical(m["param"], categories=_PARAM_ORDER, ordered=True)
        for value, tag in [("cell", "counts"), ("win_rate", "rate")]:
            wide = (m.pivot(index="metric", columns="param", values=value)
                    .sort_index())
            wide.to_csv(_out_dir(grid) / f"win_rates_by_param_{comp}_{tag}_c{CHAINS}.csv")


def main():
    for grid in GRIDS:
        out_dir = _out_dir(grid)
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"================ grid: {grid} ================\n")

        # Pooled-over-params view (400 comparisons per metric), kept for reference.
        pooled = win_rate_table(CHAINS, grid)
        pooled.to_csv(out_dir / f"win_rates_c{CHAINS}.csv", index=False)

        # RUN-level, per parameter: n_win out of 100 datasets, one row per
        # (comparison, metric, param).
        tbl = param_win_rate_table(CHAINS, grid)
        path = out_dir / f"win_rates_by_param_c{CHAINS}.csv"
        tbl.to_csv(path, index=False)
        write_wide_tables(tbl, grid)                                    # per-comparison wide CSVs
        print(f"wrote {path}  ({len(tbl)} rows) + wide tables\n")

        # Non-finite (+inf) transparency table: how many catastrophic-tail cases were dropped.
        nf = nonfinite_count_table(CHAINS, grid)
        nf_path = out_dir / f"nonfinite_counts_c{CHAINS}.csv"
        nf.to_csv(nf_path, index=False)
        if not nf.empty:
            print("=== non-finite (+inf) metric values dropped from win rates ===")
            for metric in nf["metric"].unique():
                piv = (nf[nf["metric"] == metric]
                       .pivot_table(index="param", columns="sampler",
                                    values="n_inf", fill_value=0, aggfunc="sum")
                       .reindex([p for p in _PARAM_ORDER]))
                print(f"  -- {metric}: +inf counts (out of ~100 datasets per sampler) --")
                print(piv.to_string(), "\n")

        # Plots: one figure per comparison (win rate per param, faceted by metric).
        # Guarded so a missing sampler skips its comparisons instead of aborting.
        for challenger, baseline in COMPARISONS:
            comp = f"{challenger}_vs_{baseline}"
            try:
                save(win_rate_plot(comp, tbl, grid),
                     f"marginal_comparison/{GRID_FOLDER[grid]}/plots/win_rate_{comp}_c{CHAINS}.png")
            except ValueError as e:
                print(f"  skip win_rate_{comp} ({grid}): {e}")
        print("wrote win-rate plots -> marginal_comparison/plots/\n")

        # Console summary: for each comparison + metric, n_win out of n_datasets per param.
        for comp in tbl["comparison"].unique():
            chall = comp.split("_vs_")[0]
            print(f"########## {comp} [{grid}]  (datasets where {chall} is closer, out of ~100) ##########")
            for metric in MARGINAL_METRICS:
                m = tbl[(tbl["comparison"] == comp) & (tbl["metric"] == metric)].copy()
                if m.empty:
                    continue
                m["cell"] = m["n_win"].astype(str) + "/" + m["n_datasets"].astype(str)
                ser = m.set_index("param")["cell"]
                ser = ser.reindex([p for p in _PARAM_ORDER if p in ser.index])
                print(f"  -- {metric} --")
                print(ser.to_string(), "\n")


if __name__ == "__main__":
    main()
