#!/usr/bin/env python3
r"""Turn the gathered summary CSVs into FULL-detail booktabs LaTeX fragments.

Usage (from the hpc repo root, after make_tables.py has run for both studies):

    python hpc_analysis/make_tex_tables.py --out /path/to/thesis/tables/sim

Fragments are written into two subfolders of --out: `standard/` and `mixture/`.
Each fragment is a bare `tabular` (small tables) or `longtable` (large tables)
environment - caption/label stay in the thesis. Small tables go inside a float:

    \begin{table}[htb]\centering
      \caption{...}\label{tab:sim-std-delta}
      \input{tables/sim/standard/delta_recovery}
    \end{table}

`longtable` fragments (the big per-element / per-metric / convergence tables) are
`\input` directly - NOT inside a `table` float - and page-break on their own.

Thesis preamble needed:

    \usepackage{booktabs}   % \toprule / \midrule / \bottomrule
    \usepackage{amsmath}    % \text{}, \widehat, \Delta, \Sigma
    \usepackage{longtable}  % the big tables (see LONGTABLE list printed at the end)

Two tiers of fragments are produced:

  * FULL tables (every element/param/metric/scenario x sampler, with the
    Monte-Carlo SEs) - the complete evidence behind each plot.
  * AGGREGATE tables (prefix `agg_`) - the same information condensed where the
    full detail is repetitive:
      agg_recovery.tex        (standard)  one row per parameter block (mu/Delta/Sigma)
                                          x sampler, aggregated over the block's
                                          elements: mean/max |bias|, max MCSE,
                                          mean/max MSE.
      agg_delta_recovery.tex  (mixture)   same aggregation over the 8 Delta elements,
                                          per K_true x sampler.
      agg_convergence.tex     (both)      one row per sampler (x K_true), computed on
                                          the marginal's MEAN functional and pooled
                                          over coefficients x seeds: mean/median/max
                                          R-hat + frac <= 1.1, median ESS (bulk)/(tail).
      agg_marginal.tex        (both)      median distance (median over seeds, then
                                          median over the 4 coefficients) per
                                          grid x metric (x K_true), samplers as columns.

Design
  * recovery cells: `bias (mcse)` and `mse (mcse)` (4 dp); Unicode element labels
    Δ₁,₁ / Σ₁,₁ become $\Delta_{1,1}$ / $\Sigma_{1,1}$.
  * convergence (Liesel-summary style, both studies): R-hat and ESS are computed on
    the MEAN functional of each coefficient's heterogeneity marginal (its location
    chain; per-seed values from marginal_diagnostics.csv) - never for individual
    model parameters (mu/Delta/Sigma) and with no per-functional breakdown. R-hat is
    summarised over the seeds as mean / median / max + frac(R-hat <= 1.1); ESS as
    ONE median ESS (bulk) and ONE median ESS (tail) per coefficient.
  * distribution tables: full min / Q1 / mean / median / Q3 / max (+ extra columns).
  * mixture study: one COMBINED table per family with k_true as a grouped leading
    column (blank on repeat); standard study is a single k_true = 1 cell. Exceptions
    (matching the per-k_true plot files): delta recovery, delta posterior SD and the
    marginal distances are emitted as one table PER scenario
    (delta_recovery_kt{1,2,3,5}.tex, delta_sd_kt*.tex, marginal_distances_<grid>_kt*.tex),
    each mirroring the standard study's layout.
  * columns matched case-insensitively/defensively; a table whose source CSV or
    columns are missing is reported and skipped, never written with wrong numbers.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STD = os.path.join(REPO, "hpc_analysis", "standard_model", "out")
MIX = os.path.join(REPO, "hpc_analysis", "mixture_models", "out")
DATA_STD = os.path.join(REPO, "data", "out", "standard_model")   # per-seed marginal diagnostics
DATA_MIX = os.path.join(REPO, "data", "out", "mixture_c2")

SAMPLER_LABEL = {
    "bayesm": "bayesm", "bayesm_gibbs": "Replication", "replication": "Replication",
    "nuts": "NUTS", "hmc": "HMC",
}
SAMP_CAT = ["bayesm", "Replication", "NUTS", "HMC"]     # display order
# Only the metrics reported in the thesis: KLD + TVD (Hellinger/JSD are still in the
# source CSVs but are dropped from the tex tables; Wasserstein was removed upstream).
METRIC_CAT = ["KL", "TVD"]

_LONGTABLE_THRESHOLD = 34          # switch tabular -> longtable above this many body rows
_LONGTABLES: list[str] = []        # collected for the closing note


# ----------------------------------------------------------------------
# IO helpers
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


def _need(df: pd.DataFrame, what: str, *cols: str | None) -> bool:
    if any(c is None for c in cols):
        print(f"  !! {what}: unexpected columns {list(df.columns)}")
        return False
    return True


# ----------------------------------------------------------------------
# formatting helpers
# ----------------------------------------------------------------------
def _label(s) -> str:
    return SAMPLER_LABEL.get(str(s).strip().lower(), str(s))


def _num(x, nd: int) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "$\\infty$" if (isinstance(x, float) and x == np.inf) else "{--}"
    return f"{x:.{nd}f}"


def _pm(v, m, nd: int) -> str:
    """`value (mcse)` (both to nd decimals); just `value` if mcse absent."""
    s = _num(v, nd)
    if m is not None and isinstance(m, (int, float, np.floating)) and np.isfinite(m):
        s += f" ({_num(m, nd)})"
    return s


_SUB_TO_ASCII = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
_GREEK = {"Δ": r"\Delta", "Σ": r"\Sigma"}


def _tex_escape(s: str) -> str:
    return (s.replace("\\", r"\textbackslash{}").replace("_", r"\_")
             .replace("&", r"\&").replace("%", r"\%").replace("#", r"\#"))


def _tex_label(s) -> str:
    """`Δ₁,₁ (z1:Alt1)` -> `$\\Delta_{1,1}$ (z1:Alt1)`; plain strings pass through."""
    s = str(s)
    m = re.match(r"^([ΔΣ])([₀-₉,]+)\s*(.*)$", s)
    if not m:
        return _tex_escape(s)
    sym, subs, rest = m.groups()
    math = f"${_GREEK[sym]}_{{{subs.translate(_SUB_TO_ASCII)}}}$"
    return (math + " " + _tex_escape(rest)).strip()


def _samp_ordered(df: pd.DataFrame, col: str = "sampler") -> pd.DataFrame:
    """Map the sampler column to display labels and make it a canonical categorical."""
    df = df.copy()
    df[col] = df[col].map(_label)
    df[col] = pd.Categorical(df[col], categories=SAMP_CAT, ordered=True)
    return df


def _grouped_rows(df: pd.DataFrame, group_cols: list[str], group_fmt: list, value_fn) -> list[str]:
    """One `&`-joined TeX row per DataFrame row. Leading `group_cols` are printed only
    when they change (blank on repeat); `value_fn(row)` returns the remaining cells."""
    prev = [object()] * len(group_cols)
    rows = []
    for _, r in df.iterrows():
        lead, changed = [], False
        for i, c in enumerate(group_cols):
            v = r[c]
            if changed or v != prev[i]:
                lead.append(group_fmt[i](v))
                changed = True
            else:
                lead.append("")
            prev[i] = v
        rows.append(" & ".join(lead + list(value_fn(r))))
    return rows


def _emit(out: str, name: str, colspec: str, header: str, rows: list[str],
          *, longtable: bool | None = None) -> None:
    if longtable is None:
        longtable = len(rows) > _LONGTABLE_THRESHOLD
    body = " \\\\\n".join(rows) + " \\\\"
    if longtable:
        _LONGTABLES.append(name)
        tex = (
            f"\\begin{{longtable}}{{{colspec}}}\n\\toprule\n{header} \\\\\n\\midrule\n"
            f"\\endfirsthead\n\\toprule\n{header} \\\\\n\\midrule\n\\endhead\n"
            f"\\midrule\n\\multicolumn{{{colspec.count('l') + colspec.count('c') + colspec.count('r')}}}"
            f"{{r}}{{\\emph{{continued on next page}}}} \\\\\n\\endfoot\n"
            f"\\bottomrule\n\\endlastfoot\n{body}\n\\end{{longtable}}\n"
        )
    else:
        tex = (
            f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n{header} \\\\\n\\midrule\n"
            f"{body}\n\\bottomrule\n\\end{{tabular}}\n"
        )
    with open(os.path.join(out, name), "w", encoding="utf-8") as fh:
        fh.write(tex)
    assert tex.isascii(), f"{name}: non-ASCII survived label conversion"
    print(f"  -> wrote {name}{'  [longtable]' if longtable else ''}")


# ----------------------------------------------------------------------
# generic builders
# ----------------------------------------------------------------------
def _recovery_from_wide(df: pd.DataFrame, has_kt: bool) -> pd.DataFrame | None:
    """Melt a wide delta_bias_mse table (bias_<s>, mcse_bias_<s>, mse_<s>, mcse_mse_<s>)
    into long rows [k_true?, element, sampler, bias, mcse_bias, mse, mcse_mse]."""
    bias_cols = [c for c in df.columns if c.startswith("bias_")]
    if not bias_cols or "element" not in df.columns:
        return None
    samplers = [c[len("bias_"):] for c in bias_cols]
    recs = []
    for _, r in df.iterrows():
        for s in samplers:
            rec = {"element": r["element"], "sampler": _label(s),
                   "bias": r.get(f"bias_{s}"), "mcse_bias": r.get(f"mcse_bias_{s}"),
                   "mse": r.get(f"mse_{s}"), "mcse_mse": r.get(f"mcse_mse_{s}")}
            if has_kt:
                rec["k_true"] = r["k_true"]
            recs.append(rec)
    return pd.DataFrame(recs)


def _recovery_table(out: str, name: str, df: pd.DataFrame, elem_col: str, has_kt: bool,
                    *, extra=(), extra_hdr=(), nd: int = 4) -> None:
    """Per-element bias(MCSE)/MSE(MCSE) recovery table [+ extra plain columns]."""
    df = _samp_ordered(df)
    sort_cols = (["k_true"] if has_kt else []) + [elem_col, "sampler"]
    df = df.sort_values(sort_cols, kind="stable")
    group_cols = (["k_true"] if has_kt else []) + [elem_col]
    group_fmt = ([lambda v: str(int(v))] if has_kt else []) + [_tex_label]

    def value_fn(r):
        cells = [str(r["sampler"]), _pm(r["bias"], r.get("mcse_bias"), nd),
                 _pm(r["mse"], r.get("mcse_mse"), nd)]
        cells += [_num(r[c], nd) for c in extra]
        return cells

    lead = ("$K_{\\text{true}}$ & " if has_kt else "")
    elem_hdr = "Coefficient" if "\\Delta" in _tex_label(df[elem_col].iloc[0]) or elem_col == "param" else "Element"
    header = (lead + f"{elem_hdr} & Sampler & Bias (MCSE) & MSE (MCSE)"
              + "".join(f" & {h}" for h in extra_hdr))
    colspec = ("c" if has_kt else "") + "ll" + "c" * (2 + len(extra))
    _emit(out, name, colspec, header, _grouped_rows(df, group_cols, group_fmt, value_fn))


_STATS = ["min", "q1", "mean", "median", "q3", "max"]
_STATS_HDR = "min & Q1 & mean & median & Q3 & max"


def _dist_table(out: str, name: str, df: pd.DataFrame, id_cols: list[str],
                id_fmt: list, id_hdr: str, has_kt: bool, *, nd: int = 4,
                extra=(), extra_hdr="") -> None:
    """Grouped 5-number-summary table (min/Q1/mean/median/Q3/max [+ extra]), sampler
    as the innermost printed column."""
    df = _samp_ordered(df)
    sort_cols = (["k_true"] if has_kt else []) + id_cols + ["sampler"]
    df = df.sort_values(sort_cols, kind="stable")
    group_cols = (["k_true"] if has_kt else []) + id_cols
    group_fmt = ([lambda v: str(int(v))] if has_kt else []) + id_fmt
    stat_cols = [_col(df, s) or s for s in _STATS]

    def value_fn(r):
        cells = [str(r["sampler"])] + [_num(r[c], nd) for c in stat_cols]
        cells += [_num(r[c], nd) for c in extra]
        return cells

    lead = ("$K_{\\text{true}}$ & " if has_kt else "")
    header = lead + f"{id_hdr} & Sampler & {_STATS_HDR}" + (f" & {extra_hdr}" if extra_hdr else "")
    colspec = ("c" if has_kt else "") + "l" * len(id_cols) + "l" + "c" * (6 + len(extra))
    _emit(out, name, colspec, header, _grouped_rows(df, group_cols, group_fmt, value_fn))


# ----------------------------------------------------------------------
# recovery tables
# ----------------------------------------------------------------------
def delta_recovery(out: str, root: str, has_kt: bool) -> None:
    pat = (f"{root}/delta/bias/**/delta_bias_mse_c2_all.csv" if has_kt
           else f"{root}/delta/bias/**/delta_bias_mse_c2.csv")
    df = _read(_find([pat]), "delta recovery")
    if df is None:
        return
    long = _recovery_from_wide(df, has_kt)
    if long is None:
        print(f"  !! delta recovery: unexpected columns {list(df.columns)}")
        return
    if has_kt:
        # One SEPARATE table per scenario, mirroring the standard study's layout
        # (8 elements x samplers, no k_true column; each fits a normal table float).
        for kt in sorted(long["k_true"].unique()):
            sub = long[long["k_true"] == kt].drop(columns="k_true")
            _recovery_table(out, f"delta_recovery_kt{int(kt)}.tex", sub, "element",
                            has_kt=False)
    else:
        _recovery_table(out, "delta_recovery.tex", long, "element", has_kt)


def mu_recovery(out: str) -> None:
    df = _read(_find([f"{STD}/mu/**/mu_recovery_summary_c2.csv"]), "mu recovery")
    if df is None:
        return
    if not _need(df, "mu recovery", _col(df, "param"), _col(df, "sampler"),
                 _col(df, "bias"), _col(df, "mse")):
        return
    _recovery_table(out, "mu_recovery.tex", df, "param", has_kt=False)


def sigma_recovery(out: str) -> None:
    df = _read(_find([f"{STD}/sigma/**/sigma_recovery_summary_c2.csv"]), "sigma recovery")
    if df is None:
        return
    if not _need(df, "sigma recovery", _col(df, "element"), _col(df, "sampler"),
                 _col(df, "bias"), _col(df, "mse")):
        return
    _recovery_table(out, "sigma_recovery.tex", df, "element", has_kt=False)


# ----------------------------------------------------------------------
# distribution tables
# ----------------------------------------------------------------------
def delta_sd(out: str, root: str, has_kt: bool) -> None:
    pat = (f"{root}/delta/sd/**/delta_sd_summary_c2_all.csv" if has_kt
           else f"{root}/delta/sd/**/delta_sd_summary_c2.csv")
    df = _read(_find([pat]), "delta posterior SD")
    if df is None:
        return
    if not _need(df, "delta SD", _col(df, "element"), _col(df, "sampler")):
        return
    if has_kt:
        # One table per scenario, matching the per-k_true delta_sd_elements plots.
        for kt in sorted(df["k_true"].unique()):
            sub = df[df["k_true"] == kt].drop(columns="k_true")
            _dist_table(out, f"delta_sd_kt{int(kt)}.tex", sub, ["element"], [_tex_label],
                        "Coefficient", has_kt=False)
    else:
        _dist_table(out, "delta_sd.tex", df, ["element"], [_tex_label], "Coefficient", has_kt)


def runtime(out: str, root: str, has_kt: bool) -> None:
    df = _read(_find([f"{root}/runtime/**/runtime_summary_c2.csv"]), "runtime (minutes)")
    if df is None:
        return
    if not _need(df, "runtime", _col(df, "sampler")):
        return
    _dist_table(out, "runtime.tex", df, [], [], "", has_kt, nd=2)


def marginal_distances(out: str, root: str, has_kt: bool, grid: str) -> None:
    sub = "trimmed" if grid == "trimmed" else "full"
    fn = (f"marginal_distance_summary_c2_all.csv" if has_kt
          else "marginal_distance_summary_c2.csv")
    df = _read(_find([f"{root}/marginal_comparison/{sub}/tables/{fn}"]),
               f"marginal distances ({grid})")
    if df is None:
        return
    p, m = _col(df, "param"), _col(df, "metric")
    if not _need(df, f"marginal {grid}", p, m, _col(df, "sampler")):
        return
    df = df.copy()
    df[m] = pd.Categorical(df[m], categories=METRIC_CAT, ordered=True)
    df = df[df[m].notna()]          # keep only the reported metrics (KL, TVD)
    if has_kt:
        # One table per scenario, matching the per-k_true all_metrics_kt{k} plots.
        for kt in sorted(df["k_true"].unique()):
            sub = df[df["k_true"] == kt].drop(columns="k_true")
            _dist_table(out, f"marginal_distances_{grid}_kt{int(kt)}.tex", sub, [m, p],
                        [str, _tex_escape], "Metric & Coefficient", has_kt=False, nd=4)
    else:
        _dist_table(out, f"marginal_distances_{grid}.tex", df, [m, p],
                    [str, _tex_escape], "Metric & Coefficient", has_kt, nd=4)


def retained_mass(out: str, root: str, has_kt: bool) -> None:
    df = _read(_find([f"{root}/marginal_comparison/trimmed/tables/retained_mass_summary_c2.csv"]),
               "retained mass")
    if df is None:
        return
    p = _col(df, "param")
    if not _need(df, "retained mass", p, _col(df, "sampler")):
        return
    fb = _col(df, "frac_below_guarantee")
    _dist_table(out, "retained_mass.tex", df, [p], [_tex_escape], "Coefficient", has_kt,
                nd=4, extra=([fb] if fb else ()), extra_hdr=("frac $<0.96$" if fb else ""))


# ----------------------------------------------------------------------
# KL = inf counts
# ----------------------------------------------------------------------
def kl_inf(out: str, root: str, has_kt: bool) -> None:
    frames = []
    for grid, sub in (("full", "full"), ("trimmed", "trimmed")):
        df = _read(_find([f"{root}/marginal_comparison/{sub}/tables/kl_inf_summary_c2.csv"]),
                   f"KL-inf counts ({grid})")
        if df is None:
            return
        df = df.copy()
        df["grid"] = grid
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    p = _col(df, "param")
    ni, nt, ir = _col(df, "n_inf"), _col(df, "n_total"), _col(df, "inf_rate")
    if not _need(df, "KL-inf", p, _col(df, "sampler"), ni, nt, ir):
        return
    df = _samp_ordered(df)
    df["grid"] = pd.Categorical(df["grid"], categories=["full", "trimmed"], ordered=True)
    sort_cols = (["k_true"] if has_kt else []) + ["grid", p, "sampler"]
    df = df.sort_values(sort_cols, kind="stable")
    group_cols = (["k_true"] if has_kt else []) + ["grid", p]
    group_fmt = ([lambda v: str(int(v))] if has_kt else []) + [str, _tex_escape]

    def value_fn(r):
        return [str(r["sampler"]), str(int(r[ni])), str(int(r[nt])), _num(r[ir], 3)]

    lead = ("$K_{\\text{true}}$ & " if has_kt else "")
    header = lead + "Grid & Coefficient & Sampler & $n_\\infty$ & $n$ & rate"
    colspec = ("c" if has_kt else "") + "lll" + "ccc"
    _emit(out, "kl_inf.tex", colspec, header, _grouped_rows(df, group_cols, group_fmt, value_fn))


# ----------------------------------------------------------------------
# convergence (R-hat + ESS)
# ----------------------------------------------------------------------
def convergence_rhat(out: str, data_root: str, has_kt: bool) -> None:
    """R-hat of the marginal's MEAN functional only (the location chain of each
    coefficient's heterogeneity marginal) - Liesel-summary style, matching the ESS
    tables. Per coefficient x sampler, summarised over the seeds: mean / median /
    max R-hat and the fraction of runs with R-hat <= 1.1."""
    df = _read(_find([os.path.join(data_root, "marginal_diagnostics.csv")]),
               "marginal R-hat (per-seed)")
    if df is None:
        return
    f, p, r = _col(df, "functional"), _col(df, "param"), _col(df, "rhat")
    if not _need(df, "marginal R-hat", f, p, r, _col(df, "sampler")):
        return
    if "n_chains" in df.columns:
        df = df[df["n_chains"] == 2]
    df = df[df[f].astype(str).str.lower() == "mean"].dropna(subset=[r])
    df = _samp_ordered(df)
    keys = (["k_true"] if has_kt else []) + [p, "sampler"]
    agg = (df.groupby(keys, observed=True)[r]
             .agg(rmean="mean", rmed="median", rmax="max",
                  rfrac=lambda x: (x <= 1.1).mean()).reset_index()
             .sort_values(keys, kind="stable"))

    def value_fn(r_):
        return [str(r_["sampler"]), _num(r_["rmean"], 3), _num(r_["rmed"], 3),
                _num(r_["rmax"], 3), _num(r_["rfrac"], 2)]

    lead = "$K_{\\text{true}}$ & " if has_kt else ""
    header = (lead + "Coefficient & Sampler & mean $\\widehat{R}$ & median $\\widehat{R}$ & "
              "max $\\widehat{R}$ & frac.\\ $\\widehat{R}\\leq1.1$")
    group_cols = (["k_true"] if has_kt else []) + [p]
    group_fmt = ([lambda v: str(int(v))] if has_kt else []) + [_tex_escape]
    colspec = ("c" if has_kt else "") + "ll" + "cccc"
    _emit(out, "convergence_rhat.tex", colspec, header,
          _grouped_rows(agg, group_cols, group_fmt, value_fn))


def convergence_ess(out: str, root: str, has_kt: bool) -> None:
    """Marginal ESS, Liesel-summary style: ONE median ESS (bulk) and ONE median ESS (tail)
    per coefficient x sampler, computed on the marginal's MEAN functional only (the
    location chain of the heterogeneity marginal - in the standard model exactly the
    mu_p chain, i.e. the per-parameter ESS a Goose/Liesel summary table reports)."""
    df = _read(_find([f"{root}/marginal_comparison/**/marginal_ess_summary_c2.csv"]),
               "convergence ESS")
    if df is None:
        return
    f, p = _col(df, "functional"), _col(df, "param")
    mb, mt = _col(df, "median_ess_bulk"), _col(df, "median_ess_tail")
    if not _need(df, "ESS", f, p, _col(df, "sampler"), mb, mt):
        return
    df = df[df[f].astype(str).str.lower() == "mean"]
    df = _samp_ordered(df)
    sort_cols = (["k_true"] if has_kt else []) + [p, "sampler"]
    df = df.sort_values(sort_cols, kind="stable")
    group_cols = (["k_true"] if has_kt else []) + [p]
    group_fmt = ([lambda v: str(int(v))] if has_kt else []) + [_tex_escape]

    def value_fn(r):
        return [str(r["sampler"]), _num(r[mb], 0), _num(r[mt], 0)]

    lead = ("$K_{\\text{true}}$ & " if has_kt else "")
    header = lead + "Coefficient & Sampler & median ESS (bulk) & median ESS (tail)"
    colspec = ("c" if has_kt else "") + "ll" + "cc"
    _emit(out, "convergence_ess.tex", colspec, header, _grouped_rows(df, group_cols, group_fmt, value_fn))


# ----------------------------------------------------------------------
# mixture component-count
# ----------------------------------------------------------------------
def component_recovery(out: str) -> None:
    df = _read(_find([f"{MIX}/**/component_recovery_summary_c2.csv", f"{MIX}/**/recovery_summary*.csv"]),
               "component recovery")
    if df is None:
        return
    kt, s = _col(df, "k_true"), _col(df, "sampler")
    mk, mek = _col(df, "mean_k_eff"), _col(df, "median_k_eff")
    sk, mest = _col(df, "sd_k_eff"), _col(df, "mean_est_k")
    fco, fov, fun = _col(df, "frac_correct"), _col(df, "frac_over"), _col(df, "frac_under")
    if not _need(df, "component recovery", kt, s, mk, mest, fco):
        return
    df = _samp_ordered(df).sort_values([kt, "sampler"], kind="stable")

    def value_fn(r):
        return [str(r["sampler"]), _num(r[mk], 3), _num(r[mek], 3) if mek else "{--}",
                _num(r[sk], 3) if sk else "{--}", _num(r[mest], 3),
                _num(r[fco], 3), _num(r[fov], 3) if fov else "{--}",
                _num(r[fun], 3) if fun else "{--}"]

    header = ("$K_{\\text{true}}$ & Sampler & mean $K_{\\text{eff}}$ & median $K_{\\text{eff}}$ "
              "& SD $K_{\\text{eff}}$ & mean $\\hat K$ & frac.\\ correct & frac.\\ over & frac.\\ under")
    _emit(out, "component_recovery.tex", "clccccccc", header,
          _grouped_rows(df, [kt], [lambda v: str(int(v))], value_fn))


def component_confusion(out: str) -> None:
    df = _read(_find([f"{MIX}/**/component_confusion_c2.csv"]), "component confusion")
    if df is None:
        return
    kt, s = _col(df, "k_true"), _col(df, "sampler")
    pcols = [c for c in df.columns if re.fullmatch(r"p_est\d", c)]
    if not _need(df, "component confusion", kt, s) or not pcols:
        return
    pcols = sorted(pcols)
    df = _samp_ordered(df).sort_values([kt, "sampler"], kind="stable")

    def value_fn(r):
        return [str(r["sampler"])] + [_num(r[c], 3) for c in pcols]

    header = ("$K_{\\text{true}}$ & Sampler & "
              + " & ".join(f"$\\hat K{{=}}{c[-1]}$" for c in pcols))
    _emit(out, "component_confusion.tex", "cl" + "c" * len(pcols), header,
          _grouped_rows(df, [kt], [lambda v: str(int(v))], value_fn))


def component_thresholds(out: str) -> None:
    df = _read(_find([f"{MIX}/**/component_threshold_sensitivity_c2.csv"]),
               "component threshold sensitivity")
    if df is None:
        return
    kt, s, th = _col(df, "k_true"), _col(df, "sampler"), _col(df, "threshold")
    mest, fco = _col(df, "mean_est_k"), _col(df, "frac_correct")
    if not _need(df, "component thresholds", kt, s, th, mest, fco):
        return
    df = _samp_ordered(df).sort_values([kt, "sampler", th], kind="stable")

    def value_fn(r):
        return [str(r["sampler"]), _num(r[th], 3), _num(r[mest], 3), _num(r[fco], 3)]

    header = "$K_{\\text{true}}$ & Sampler & threshold & mean $\\hat K$ & frac.\\ correct"
    _emit(out, "component_thresholds.tex", "cl" + "ccc", header,
          _grouped_rows(df, [kt, s], [lambda v: str(int(v)), _label], value_fn))


# ----------------------------------------------------------------------
# aggregate (summary) tables
# ----------------------------------------------------------------------
_AGG_HDR = "$n$ & mean $|$Bias$|$ & max $|$Bias$|$ & max MCSE & mean MSE & max MSE"


def _recovery_agg_table(out: str, name: str, long_df: pd.DataFrame, keys: list[str],
                        key_hdr: str, key_spec: str, key_fmt: list) -> None:
    """Aggregate a long recovery frame over its elements: one row per (*keys, sampler)
    with mean/max |bias|, the max MCSE(bias) (conservative uncertainty bound for the
    aggregated biases) and mean/max MSE."""
    df = _samp_ordered(long_df)

    def _a(x):
        return pd.Series({
            "n": len(x),
            "mab": x["bias"].abs().mean(), "xab": x["bias"].abs().max(),
            "xmc": x["mcse_bias"].max() if "mcse_bias" in x.columns else np.nan,
            "mms": x["mse"].mean(), "xms": x["mse"].max(),
        })

    agg = (df.groupby(keys + ["sampler"], observed=True)
             .apply(_a, include_groups=False).reset_index()
             .sort_values(keys + ["sampler"], kind="stable"))

    def value_fn(r):
        return [str(r["sampler"]), str(int(r["n"])), _num(r["mab"], 4), _num(r["xab"], 4),
                _num(r["xmc"], 4), _num(r["mms"], 4), _num(r["xms"], 4)]

    _emit(out, name, key_spec + "l" + "c" * 6, f"{key_hdr} & Sampler & {_AGG_HDR}",
          _grouped_rows(agg, keys, key_fmt, value_fn))


def agg_recovery_standard(out: str) -> None:
    """Standard study: bias/MSE aggregated over the elements of each parameter block
    (mu: 4 params, Delta: 8 elements, Sigma: 10 lower-triangle elements)."""
    frames = []
    d = _read(_find([f"{STD}/delta/bias/**/delta_bias_mse_c2.csv"]), "aggregate: delta recovery")
    if d is not None:
        long = _recovery_from_wide(d, has_kt=False)
        if long is not None:
            frames.append(long.assign(block=r"$\Delta$")[
                ["block", "sampler", "bias", "mcse_bias", "mse", "mcse_mse"]])
    for what, blk in (("mu", r"$\mu$"), ("sigma", r"$\Sigma$")):
        df = _read(_find([f"{STD}/{what}/**/{what}_recovery_summary_c2.csv"]),
                   f"aggregate: {what} recovery")
        if df is None or not all(_col(df, c) for c in ("sampler", "bias", "mse")):
            continue
        keep = [c for c in ("sampler", "bias", "mcse_bias", "mse", "mcse_mse") if c in df.columns]
        frames.append(df[keep].assign(block=blk))
    if not frames:
        return
    allb = pd.concat(frames, ignore_index=True)
    allb["block"] = pd.Categorical(
        allb["block"], categories=[r"$\mu$", r"$\Delta$", r"$\Sigma$"], ordered=True)
    _recovery_agg_table(out, "agg_recovery.tex", allb, ["block"], "Block", "l", [str])


def agg_delta_recovery_mix(out: str) -> None:
    """Mixture study: Delta bias/MSE aggregated over the 8 elements, per K_true."""
    d = _read(_find([f"{MIX}/delta/bias/**/delta_bias_mse_c2_all.csv"]),
              "aggregate: delta recovery")
    if d is None:
        return
    long = _recovery_from_wide(d, has_kt=True)
    if long is None:
        print(f"  !! aggregate delta: unexpected columns {list(d.columns)}")
        return
    _recovery_agg_table(out, "agg_delta_recovery.tex", long, ["k_true"],
                        "$K_{\\text{true}}$", "c", [lambda v: str(int(v))])


def agg_convergence(out: str, data_root: str, has_kt: bool) -> None:
    """One row per sampler (x K_true), all computed on the marginal's MEAN functional
    only and pooled over coefficients x seeds (per-seed marginal_diagnostics.csv):
    mean / median / max R-hat, frac(R-hat <= 1.1), and median ESS (bulk)/(tail)."""
    df = _read(_find([os.path.join(data_root, "marginal_diagnostics.csv")]),
               "aggregate: marginal R-hat/ESS (per-seed)")
    if df is None:
        return
    fu, r = _col(df, "functional"), _col(df, "rhat")
    eb, et = _col(df, "ess_bulk"), _col(df, "ess_tail")
    if not _need(df, "agg marginal R-hat/ESS", fu, r, eb, et, _col(df, "sampler")):
        return
    if "n_chains" in df.columns:
        df = df[df["n_chains"] == 2]
    df = df[df[fu].astype(str).str.lower() == "mean"]
    df = _samp_ordered(df)
    keys = (["k_true"] if has_kt else []) + ["sampler"]
    agg = (df.groupby(keys, observed=True)
             .agg(rmean=(r, "mean"), rmed=(r, "median"), rmax=(r, "max"),
                  rfrac=(r, lambda x: (x.dropna() <= 1.1).mean()),
                  med_b=(eb, "median"), med_t=(et, "median")).reset_index()
             .sort_values(keys, kind="stable"))

    def value_fn(r_):
        return [str(r_["sampler"]), _num(r_["rmean"], 3), _num(r_["rmed"], 3),
                _num(r_["rmax"], 3), _num(r_["rfrac"], 2),
                _num(r_["med_b"], 0), _num(r_["med_t"], 0)]

    lead = "$K_{\\text{true}}$ & " if has_kt else ""
    header = (lead + "Sampler & mean $\\widehat{R}$ & median $\\widehat{R}$ & "
              "max $\\widehat{R}$ & frac.\\ $\\widehat{R}\\leq1.1$ & "
              "median ESS (bulk) & median ESS (tail)")
    group_cols = ["k_true"] if has_kt else []
    group_fmt = [lambda v: str(int(v))] if has_kt else []
    _emit(out, "agg_convergence.tex", ("c" if has_kt else "") + "l" + "c" * 6, header,
          _grouped_rows(agg, group_cols, group_fmt, value_fn))


def agg_marginal(out: str, root: str, has_kt: bool) -> None:
    """Marginal distances condensed: per grid x metric (x K_true), the median distance
    (median across seeds per cell, then median over the 4 coefficients); samplers as
    columns."""
    frames = []
    for grid in ("full", "trimmed"):
        fn = ("marginal_distance_summary_c2_all.csv" if has_kt
              else "marginal_distance_summary_c2.csv")
        df = _read(_find([f"{root}/marginal_comparison/{grid}/tables/{fn}"]),
                   f"aggregate: marginal distances ({grid})")
        if df is not None:
            frames.append(df.assign(grid=grid))
    if not frames:
        return
    df = pd.concat(frames, ignore_index=True)
    met, md = _col(df, "metric"), _col(df, "median")
    if not _need(df, "agg marginal", met, md, _col(df, "sampler")):
        return
    df = _samp_ordered(df)
    df[met] = pd.Categorical(df[met], categories=METRIC_CAT, ordered=True)
    df = df[df[met].notna()]        # keep only the reported metrics (KL, TVD)
    keys = ["grid", met] + (["k_true"] if has_kt else [])
    agg = (df.groupby(keys + ["sampler"], observed=True)[md]
             .median().reset_index(name="med"))
    piv = (agg.pivot_table(index=keys, columns="sampler", values="med", observed=True)
              .reset_index().sort_values(keys, kind="stable"))
    samp_cols = [c for c in SAMP_CAT if c in piv.columns]

    if has_kt:
        group_cols, group_fmt = ["grid", met], [str, str]

        def value_fn(r):
            return [str(int(r["k_true"]))] + [_num(r[c], 3) for c in samp_cols]

        header = "Grid & Metric & $K_{\\text{true}}$ & " + " & ".join(samp_cols)
        colspec = "llc" + "c" * len(samp_cols)
    else:
        group_cols, group_fmt = ["grid"], [str]

        def value_fn(r):
            return [str(r[met])] + [_num(r[c], 3) for c in samp_cols]

        header = "Grid & Metric & " + " & ".join(samp_cols)
        colspec = "ll" + "c" * len(samp_cols)
    _emit(out, "agg_marginal.tex", colspec, header,
          _grouped_rows(piv, group_cols, group_fmt, value_fn))


# ----------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(REPO, "hpc_analysis", "tex_tables"),
                    help="directory for the .tex fragments (e.g. the thesis tables/sim dir); "
                         "'standard/' and 'mixture/' subfolders are created inside it")
    args = ap.parse_args()
    std_out = os.path.join(args.out, "standard")
    mix_out = os.path.join(args.out, "mixture")
    os.makedirs(std_out, exist_ok=True)
    os.makedirs(mix_out, exist_ok=True)
    print(f"writing FULL tables to {args.out} (subfolders: standard/ + mixture/)")

    print("standard study -> standard/")
    delta_recovery(std_out, STD, has_kt=False)
    mu_recovery(std_out)
    sigma_recovery(std_out)
    delta_sd(std_out, STD, has_kt=False)
    runtime(std_out, STD, has_kt=False)
    for grid in ("full", "trimmed"):
        marginal_distances(std_out, STD, has_kt=False, grid=grid)
    retained_mass(std_out, STD, has_kt=False)
    kl_inf(std_out, STD, has_kt=False)
    convergence_rhat(std_out, DATA_STD, has_kt=False)
    convergence_ess(std_out, STD, has_kt=False)
    agg_recovery_standard(std_out)
    agg_convergence(std_out, DATA_STD, has_kt=False)
    agg_marginal(std_out, STD, has_kt=False)

    print("mixture study -> mixture/")
    delta_recovery(mix_out, MIX, has_kt=True)
    delta_sd(mix_out, MIX, has_kt=True)
    runtime(mix_out, MIX, has_kt=True)
    for grid in ("full", "trimmed"):
        marginal_distances(mix_out, MIX, has_kt=True, grid=grid)
    retained_mass(mix_out, MIX, has_kt=True)
    kl_inf(mix_out, MIX, has_kt=True)
    convergence_rhat(mix_out, DATA_MIX, has_kt=True)
    convergence_ess(mix_out, MIX, has_kt=True)
    component_recovery(mix_out)
    component_confusion(mix_out)
    component_thresholds(mix_out)
    agg_delta_recovery_mix(mix_out)
    agg_convergence(mix_out, DATA_MIX, has_kt=True)
    agg_marginal(mix_out, MIX, has_kt=True)

    if _LONGTABLES:
        print("\nlongtable fragments (\\input directly, not inside a table float; "
              "needs \\usepackage{longtable}):")
        for n in dict.fromkeys(_LONGTABLES):          # dedupe, preserve order
            print(f"  - {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
