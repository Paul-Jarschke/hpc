#!/usr/bin/env python3
r"""Turn the gathered summary CSVs into booktabs LaTeX fragments.

Usage (from the hpc repo root, after make_tables.py has run for both studies):

    python hpc_analysis/make_tex_tables.py --out /path/to/thesis/tables/sim

Every fragment is a bare `tabular` environment. Caption, label and the
surrounding `table` float stay in the thesis .tex, e.g.:

    \begin{table}[htb]
    \centering
    \caption{...}
    \label{tab:sim-std-mu}
    \input{tables/sim/std_mu_recovery}
    \end{table}

Fragments written (skipped with a warning if the source CSV is missing):

  standard study (6.4)                 mixture study (6.5)
  ------------------------------       ------------------------------
  std_mu_recovery.tex                  mix_tvd_medians.tex
  std_delta_recovery.tex               mix_keff.tex
  std_convergence.tex                  mix_convergence.tex
  std_runtime.tex                      mix_runtime.tex
  std_tvd.tex

Design choices
  * numbers: 3 decimals for bias/MSE/TVD, 2 for R-hat, 0 for ESS,
    1 for runtime minutes; MCSEs in parentheses behind the estimate.
  * convergence tables: functional == "mean" only, averaged over the
    four coefficients (median R-hat, frac converged, median bulk ESS,
    frac ESS >= 400) - matching Section 6.3.
  * runtime: seconds columns are converted to minutes.
  * column names are matched case-insensitively and defensively, so the
    script survives small schema changes; anything it cannot find it
    reports instead of silently writing wrong numbers.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STD = os.path.join(REPO, "hpc_analysis", "standard_model", "out")
MIX = os.path.join(REPO, "hpc_analysis", "mixture_models", "out")

SAMPLER_ORDER = ["bayesm", "Replication", "bayesm_gibbs", "NUTS", "nuts", "HMC", "hmc"]
SAMPLER_LABEL = {
    "bayesm": "bayesm", "bayesm_gibbs": "Replication", "replication": "Replication",
    "nuts": "NUTS", "hmc": "HMC",
}
KT_ORDER = [1, 2, 3, 5]


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _find(patterns: list[str]) -> str | None:
    for pat in patterns:
        hits = sorted(glob.glob(pat, recursive=True))
        if hits:
            return hits[0]
    return None


def _read(path: str | None, what: str) -> pd.DataFrame | None:
    if path is None:
        print(f"  !! missing: {what} (no matching CSV found)")
        return None
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    print(f"  ok: {what} <- {os.path.relpath(path, REPO)} ({len(df)} rows)")
    return df


def _col(df: pd.DataFrame, *candidates: str) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _label(s: str) -> str:
    return SAMPLER_LABEL.get(str(s).strip().lower(), str(s))


def _order_samplers(df: pd.DataFrame, col: str = "sampler") -> pd.DataFrame:
    df = df.copy()
    df["__lab"] = df[col].map(_label)
    order = ["bayesm", "Replication", "NUTS", "HMC"]
    df["__lab"] = pd.Categorical(df["__lab"], categories=order, ordered=True)
    return df.sort_values("__lab").drop(columns="__lab")


def _fmt(x, nd: int) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "$\\infty$"
    return f"{x:.{nd}f}"


def _write(out_dir: str, name: str, header: str, rows: list[str], colspec: str) -> None:
    body = " \\\\\n".join(rows) + " \\\\"
    tex = (
        f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{header} \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n"
    )
    path = os.path.join(out_dir, name)
    with open(path, "w") as fh:
        fh.write(tex)
    print(f"  -> wrote {name}")


# ----------------------------------------------------------------------
# standard study
# ----------------------------------------------------------------------
def std_mu_recovery(out: str) -> None:
    df = _read(_find([f"{STD}/mu/**/mu_recovery_summary_c2.csv"]), "standard mu recovery")
    if df is None:
        return
    p, s = _col(df, "param"), _col(df, "sampler")
    b, bm = _col(df, "bias"), _col(df, "mcse_bias", "bias_mcse", "mcse(bias)")
    m, mm = _col(df, "mse"), _col(df, "mcse_mse", "mse_mcse", "mcse(mse)")
    if None in (p, s, b, m):
        print(f"  !! std mu: unexpected columns {list(df.columns)}")
        return
    rows = []
    for param in df[p].unique():
        sub = _order_samplers(df[df[p] == param], s)
        for i, (_, r) in enumerate(sub.iterrows()):
            lead = param if i == 0 else ""
            bias = _fmt(r[b], 3) + (f" ({_fmt(r[bm], 3)})" if bm else "")
            mse = _fmt(r[m], 3) + (f" ({_fmt(r[mm], 3)})" if mm else "")
            rows.append(f"{lead} & {_label(r[s])} & {bias} & {mse}")
    _write(out, "std_mu_recovery.tex",
           "Coefficient & Sampler & Bias (MCSE) & MSE (MCSE)", rows, "llcc")


def std_delta_recovery(out: str) -> None:
    # The delta_bias_mse table is WIDE: one row per element, with per-sampler columns
    # bias_<sampler>, mcse_bias_<sampler>, mse_<sampler>, ... (sampler suffix lowercased).
    df = _read(_find([f"{STD}/delta/bias/**/delta_bias_mse_c2.csv"]), "standard delta recovery")
    if df is None:
        return
    bias_cols = [c for c in df.columns if c.startswith("bias_")]
    if not bias_cols:
        print(f"  !! std delta: unexpected columns {list(df.columns)}")
        return
    samplers = [c[len("bias_"):] for c in bias_cols]          # e.g. ['bayesm','nuts','hmc']
    order = ["bayesm", "Replication", "NUTS", "HMC"]
    samplers.sort(key=lambda x: order.index(_label(x)) if _label(x) in order else 99)
    rows = []
    for samp in samplers:
        bcol, mccol, mcol = f"bias_{samp}", f"mcse_bias_{samp}", f"mse_{samp}"
        maxabs = df[bcol].abs().max()
        maxmcse = df[mccol].max() if mccol in df.columns else np.nan
        rows.append(
            f"{_label(samp)} & {_fmt(maxabs, 3)} & {_fmt(maxmcse, 3)} & "
            f"{_fmt(df[mcol].min(), 3)}--{_fmt(df[mcol].max(), 3)}"
        )
    _write(out, "std_delta_recovery.tex",
           "Sampler & max $|$Bias$|$ & max MCSE & MSE range", rows, "lccc")


def _convergence_rows(rhat: pd.DataFrame, ess: pd.DataFrame, by_kt: bool) -> list[str]:
    f = _col(rhat, "functional", "series")
    rhat = rhat[rhat[f].str.lower() == "mean"]
    f2 = _col(ess, "functional", "series")
    ess = ess[ess[f2].str.lower() == "mean"]
    s = _col(rhat, "sampler")
    keys = (["k_true", s] if by_kt else [s])
    r_agg = rhat.groupby(keys, observed=True)[
        [_col(rhat, "median_rhat"), _col(rhat, "frac_converged")]
    ].mean().reset_index()
    e_med = _col(ess, "median_ess_bulk", "median_ess")
    e_frac = _col(ess, "frac_ess_bulk_ge_400", "frac_ess_ge_400")
    e_agg = ess.groupby(keys, observed=True)[[e_med, e_frac]].mean().reset_index()
    merged = r_agg.merge(e_agg, on=keys)
    rows = []
    if by_kt:
        for kt in KT_ORDER:
            sub = _order_samplers(merged[merged["k_true"] == kt], s)
            for i, (_, r) in enumerate(sub.iterrows()):
                lead = str(kt) if i == 0 else ""
                rows.append(
                    f"{lead} & {_label(r[s])} & {_fmt(r[_col(rhat,'median_rhat')], 2)} & "
                    f"{_fmt(r[_col(rhat,'frac_converged')], 2)} & "
                    f"{_fmt(r[e_med], 0)} & {_fmt(r[e_frac], 2)}"
                )
    else:
        for _, r in _order_samplers(merged, s).iterrows():
            rows.append(
                f"{_label(r[s])} & {_fmt(r[_col(rhat,'median_rhat')], 2)} & "
                f"{_fmt(r[_col(rhat,'frac_converged')], 2)} & "
                f"{_fmt(r[e_med], 0)} & {_fmt(r[e_frac], 2)}"
            )
    return rows


def std_convergence(out: str) -> None:
    rhat = _read(_find([f"{STD}/marginal_comparison/**/marginal_rhat_summary_c2.csv"]),
                 "standard marginal R-hat")
    ess = _read(_find([f"{STD}/marginal_comparison/**/marginal_ess_summary_c2.csv"]),
                "standard marginal ESS")
    if rhat is None or ess is None:
        return
    rows = _convergence_rows(rhat, ess, by_kt=False)
    _write(out, "std_convergence.tex",
           "Sampler & median $\\widehat{R}$ & frac.\\ $\\widehat{R} \\leq 1.1$ & "
           "median bulk ESS & frac.\\ ESS $\\geq 400$", rows, "lcccc")


def std_runtime(out: str) -> None:
    df = _read(_find([f"{STD}/runtime/**/runtime_summary_c2.csv"]), "standard runtime")
    if df is None:
        return
    s = _col(df, "sampler")
    med = _col(df, "median_min", "median_runtime_min", "median", "median_s", "median_runtime_s")
    q1 = _col(df, "q1", "q25", "q1_min", "q25_min", "q1_s", "q25_s")
    q3 = _col(df, "q3", "q75", "q3_min", "q75_min", "q3_s", "q75_s")
    mx = _col(df, "max", "max_min", "max_s")
    if None in (s, med):
        print(f"  !! std runtime: unexpected columns {list(df.columns)}")
        return
    to_min = 1 / 60.0 if med.endswith("_s") or df[med].max() > 500 else 1.0
    rows = []
    for _, r in _order_samplers(df, s).iterrows():
        iqr = (f"{_fmt(r[q1]*to_min,1)}--{_fmt(r[q3]*to_min,1)}" if q1 and q3 else "--")
        rows.append(f"{_label(r[s])} & {_fmt(r[med]*to_min,1)} & {iqr} & "
                    f"{_fmt(r[mx]*to_min,1) if mx else '--'}")
    _write(out, "std_runtime.tex", "Sampler & Median & IQR & Max", rows, "lccc")


def std_tvd(out: str) -> None:
    df = _read(_find([f"{STD}/marginal_comparison/trimmed/tables/marginal_distance_summary_c2.csv"]),
               "standard TVD (chebyshev grid)")
    if df is None:
        return
    met, s, p = _col(df, "metric"), _col(df, "sampler"), _col(df, "param")
    md = _col(df, "median")
    sub = df[df[met].str.upper() == "TVD"]
    rows = []
    for param in sub[p].unique():
        block = _order_samplers(sub[sub[p] == param], s)
        cells = " & ".join(_fmt(r[md], 3) for _, r in block.iterrows())
        rows.append(f"{param} & {cells}")
    _write(out, "std_tvd.tex", "Coefficient & bayesm & NUTS & HMC", rows, "lccc")


# ----------------------------------------------------------------------
# mixture study
# ----------------------------------------------------------------------
def mix_tvd_medians(out: str) -> None:
    rows = []
    for kt in KT_ORDER:
        df = _read(_find([f"{MIX}/marginal_comparison/trimmed/tables/"
                          f"marginal_distance_summary_c2_kt{kt}.csv"]),
                   f"mixture TVD kt{kt}")
        if df is None:
            return
        met, s, md = _col(df, "metric"), _col(df, "sampler"), _col(df, "median")
        sub = df[df[met].str.upper() == "TVD"].groupby(s, observed=True)[md].mean().reset_index()
        sub = _order_samplers(sub, s)
        cells = " & ".join(_fmt(r[md], 3) for _, r in sub.iterrows())
        rows.append(f"{kt} & {cells}")
    _write(out, "mix_tvd_medians.tex",
           "$K_{\\text{true}}$ & bayesm & Replication & NUTS & HMC", rows, "ccccc")


def mix_keff(out: str) -> None:
    df = _read(_find([f"{MIX}/**/recovery_summary*.csv", f"{MIX}/**/component*summary*.csv",
                      f"{MIX}/**/*keff*.csv"]), "mixture K_eff summary")
    if df is None:
        return
    s = _col(df, "sampler")
    kt = _col(df, "k_true", "ktrue")
    mean_k = _col(df, "mean_k_eff", "mean_keff", "k_eff_mean")
    med_k = _col(df, "median_k_eff", "median_keff")
    sd_k = _col(df, "sd_k_eff", "sd_keff", "k_eff_sd")
    if None in (s, kt, mean_k):
        print(f"  !! mixture K_eff: unexpected columns {list(df.columns)}")
        return
    rows = []
    for k in KT_ORDER:
        sub = _order_samplers(df[df[kt] == k], s)
        for i, (_, r) in enumerate(sub.iterrows()):
            lead = str(k) if i == 0 else ""
            extra = ""
            if med_k:
                extra += f" & {_fmt(r[med_k], 2)}"
            if sd_k:
                extra += f" & {_fmt(r[sd_k], 2)}"
            rows.append(f"{lead} & {_label(r[s])} & {_fmt(r[mean_k], 2)}{extra}")
    hdr = "$K_{\\text{true}}$ & Sampler & mean $K_{\\text{eff}}$"
    spec = "clc"
    if med_k:
        hdr += " & median $K_{\\text{eff}}$"
        spec += "c"
    if sd_k:
        hdr += " & sd"
        spec += "c"
    _write(out, "mix_keff.tex", hdr, rows, spec)


def mix_convergence(out: str) -> None:
    rhat = _read(_find([f"{MIX}/marginal_comparison/full/tables/marginal_rhat_summary_c2.csv",
                        f"{MIX}/marginal_comparison/**/marginal_rhat_summary_c2.csv"]),
                 "mixture marginal R-hat")
    ess = _read(_find([f"{MIX}/marginal_comparison/full/tables/marginal_ess_summary_c2.csv",
                       f"{MIX}/marginal_comparison/**/marginal_ess_summary_c2.csv"]),
                "mixture marginal ESS")
    if rhat is None or ess is None:
        return
    rows = _convergence_rows(rhat, ess, by_kt=True)
    _write(out, "mix_convergence.tex",
           "$K_{\\text{true}}$ & Sampler & median $\\widehat{R}$ & "
           "frac.\\ $\\widehat{R} \\leq 1.1$ & median bulk ESS & frac.\\ ESS $\\geq 400$",
           rows, "clcccc")


def mix_runtime(out: str) -> None:
    df = _read(_find([f"{MIX}/runtime/**/runtime_summary*.csv", f"{MIX}/**/runtime*summary*.csv"]),
               "mixture runtime")
    if df is None:
        return
    s, kt = _col(df, "sampler"), _col(df, "k_true", "ktrue")
    med = _col(df, "median_min", "median", "median_s")
    q1 = _col(df, "q1", "q25", "q1_min", "q25_min", "q1_s")
    q3 = _col(df, "q3", "q75", "q3_min", "q75_min", "q3_s")
    mx = _col(df, "max", "max_min", "max_s")
    if None in (s, kt, med):
        print(f"  !! mixture runtime: unexpected columns {list(df.columns)}")
        return
    to_min = 1 / 60.0 if med.endswith("_s") or df[med].max() > 500 else 1.0
    rows = []
    for k in KT_ORDER:
        sub = _order_samplers(df[df[kt] == k], s)
        for i, (_, r) in enumerate(sub.iterrows()):
            lead = str(k) if i == 0 else ""
            iqr = (f"{_fmt(r[q1]*to_min,1)}--{_fmt(r[q3]*to_min,1)}" if q1 and q3 else "--")
            rows.append(f"{lead} & {_label(r[s])} & {_fmt(r[med]*to_min,1)} & {iqr} & "
                        f"{_fmt(r[mx]*to_min,1) if mx else '--'}")
    _write(out, "mix_runtime.tex",
           "$K_{\\text{true}}$ & Sampler & Median & IQR & Max", rows, "clccc")


# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "hpc_analysis", "tex_tables"),
                    help="directory for the .tex fragments (e.g. the thesis tables/sim dir)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    print(f"writing fragments to {args.out}")

    print("standard study")
    std_mu_recovery(args.out)
    std_delta_recovery(args.out)
    std_convergence(args.out)
    std_runtime(args.out)
    std_tvd(args.out)

    print("mixture study")
    mix_tvd_medians(args.out)
    mix_keff(args.out)
    mix_convergence(args.out)
    mix_runtime(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())