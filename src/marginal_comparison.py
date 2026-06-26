"""
Marginal-density comparison across samplers for the mixture HBMNL (Rossi et al.
2006, Eq. 5.5.19 and 5.5.2).

One comparison per `<chains>/<k>_comp/` folder: it contrasts the NUTS, HMC and
bayesm runs that sit side by side there against each other and the true DGP, on
LABEL-INVARIANT quantities only - so no relabeling is needed (a per-draw
permutation of the component triples leaves every quantity here unchanged).

Corrections over the earlier prototype
--------------------------------------
1. Grids are anchored to the FITTED models' own support (union over samplers,
   wide), NOT to the true DGP +/-4 sigma - the latter clips the tails/lobes Rossi
   highlights and is circular. Truth is an overlay only.
2. Convergence is assessed with arviz `rhat`/`ess` on the REAL (chains, draws)
   invariant series (rank-normalised split-R-hat across actual chains), not by
   splitting a single flattened chain into pseudo-chains.
3. The density-support mask uses the INVARIANT marginal (per-draw then average),
   never slot-wise posterior means (which mix components under label switching).

The marginal density (Eq. 5.5.19) and mixture moments (Eq. 5.5.2) themselves are
unchanged from the validated prototype.

References
----------
- Rossi, Allenby & McCulloch (2006), Bayesian Statistics and Marketing, Ch. 5:
  Eq. 5.5.19 (marginal density) and Eq. 5.5.2 (mixture moments).
- Vehtari, Gelman, Simpson, Carpenter & Burkner (2021), "Rank-normalization,
  folding, and localization: An improved R-hat for assessing convergence of MCMC",
  Bayesian Analysis 16(2):667-718 - the rank-normalized split-R-hat used here (via
  arviz, which splits each chain in half by default).
- Gelman et al. (2013), BDA3 sec. 11.4; Stan Reference Manual (Potential Scale
  Reduction): a single chain may be split into two halves to compute a valid
  split-R-hat. For 1-chain runs we do exactly that and report it as a WITHIN-chain
  check only - it cannot detect multimodality a lone chain never explored (the
  original Gelman & Rubin 1992 rationale for multiple over-dispersed chains), so
  the load-bearing between-chain R-hat comes from the multi-chain runs.
"""

import json
import pickle
import pathlib

import numpy as np
import pandas as pd
import arviz as az
from scipy.stats import norm, wasserstein_distance

from src import analysis


# --------------------------------------------------------------------------- #
# Loading - every sampler exposes the same arrays, so one path serves all three
# --------------------------------------------------------------------------- #
def load_sampler(results_dir, name):
    """Load one run's draws as (C,S,K,P)/(C,S,K) arrays. is_mcmc=True."""
    rd = pathlib.Path(results_dir)
    with open(rd / "posterior_raw.pkl", "rb") as f:
        post = pickle.load(f)
    mu    = np.asarray(post["mu_k"])                                  # (C,S,K,P)
    pvec  = np.asarray(analysis._recover_pvec(post))                 # (C,S,K)
    Sigma = np.asarray(
        analysis._sigma_from_latent(np.asarray(post["sigma_inv_chol_k_latent"]))
    )                                                                # (C,S,K,P,P)
    std = np.sqrt(np.clip(np.diagonal(Sigma, axis1=3, axis2=4), 0.0, None))  # (C,S,K,P)
    return {"name": name, "mu": mu, "pvec": pvec, "Sigma": Sigma, "std": std, "is_mcmc": True}


def true_dgp_model(raw_data):
    """Ground-truth components as a 1-chain, 1-draw 'model' (for overlay only)."""
    mu    = np.array(raw_data["TRUE_MU_K"])               # (K_true, P)
    pvec  = np.array(raw_data["TRUE_PVEC"]).ravel()       # (K_true,)
    Sigma = np.array(raw_data["TRUE_SIGMA_K"])            # (K_true, P, P)
    std   = np.sqrt(np.clip(np.diagonal(Sigma, axis1=1, axis2=2), 0.0, None))
    return {"name": "True DGP",
            "mu": mu[None, None], "pvec": pvec[None, None],
            "Sigma": Sigma[None, None], "std": std[None, None], "is_mcmc": False}


def _flat(model):
    """Collapse (C,S,...) -> (R,...) for density / moment estimators."""
    mu = model["mu"]; C, S, K, P = mu.shape
    R = C * S
    return (mu.reshape(R, K, P), model["pvec"].reshape(R, K),
            model["std"].reshape(R, K, P), model["Sigma"].reshape(R, K, P, P), P, K)


# --------------------------------------------------------------------------- #
# Grids - fitted-model support (union, wide). Truth is NOT used to set bounds.
# --------------------------------------------------------------------------- #
def build_grids(fitted_models, K_true, n_grid=1000, n_sigma=4, trim_pct=0.5):
    """Per-parameter grid spanning the support of the per-draw TOP-K_true ('live')
    components, unioned over samplers.

    Using only the live components per draw is essential when K_MODEL > K_TRUE:
    the surplus (empty) components have huge prior-driven mu/sigma that would
    otherwise blow the range up to +/-thousands and squash all real mass into a
    pixel. Selecting the top-K_true by each draw's own pvec is label-invariant
    (no slot identity needed) and excludes the empties. trim_pct trims outlier
    draws via a percentile so a single odd draw can't widen the grid. The empty
    components still appear in the marginal density itself (Eq. 5.5.19) but
    contribute ~0 over this live region."""
    P = fitted_models[0]["mu"].shape[-1]
    lo = np.full(P, np.inf)
    hi = np.full(P, -np.inf)
    for m in fitted_models:
        mu, pvec, std, _, _, K = _flat(m)
        k_live = min(K_true, K)
        topk = np.argsort(-pvec, axis=1)[:, :k_live]              # (R, k_live)
        r = np.arange(mu.shape[0])[:, None]
        mu_t, std_t = mu[r, topk], std[r, topk]                  # (R, k_live, P)
        e_lo = (mu_t - n_sigma * std_t).reshape(-1, P)
        e_hi = (mu_t + n_sigma * std_t).reshape(-1, P)
        lo = np.minimum(lo, np.percentile(e_lo, trim_pct, axis=0))
        hi = np.maximum(hi, np.percentile(e_hi, 100 - trim_pct, axis=0))
    return [np.linspace(lo[j], hi[j], n_grid) for j in range(P)]


# --------------------------------------------------------------------------- #
# Marginal density (Eq. 5.5.19) and mixture moments (Eq. 5.5.2)
# --------------------------------------------------------------------------- #
def marginal_density(model, grids):
    """d_bar_j(x) = (1/R) sum_r sum_k pvec_k^r * N(x ; mu_k^r[j], sigma_k^r[j]).
    Per-draw then average -> label-invariant. Returns a list of P arrays."""
    mu, pvec, std, _, P, K = _flat(model)
    out = []
    for j in range(P):
        x = grids[j]
        d = np.zeros(len(x))
        for k in range(K):
            pdf = norm.pdf(x[None, :], loc=mu[:, k, j, None], scale=std[:, k, j, None] + 1e-8)
            d += np.mean(pvec[:, k, None] * pdf, axis=0)
        out.append(d)
    return out


def mixture_moments(model):
    """Rossi Eq. 5.5.2: E[theta] and Var[theta] (within + between), averaged over
    draws. Returns (mean (P,), var (P,P))."""
    mu, pvec, _, Sigma, P, K = _flat(model)
    mean   = np.sum(pvec[:, :, None] * mu, axis=1)                    # (R,P)
    within = np.sum(pvec[:, :, None, None] * Sigma, axis=1)           # (R,P,P)
    diff   = mu - mean[:, None, :]
    betwn  = np.sum(pvec[:, :, None, None] * (diff[:, :, :, None] * diff[:, :, None, :]), axis=1)
    var    = within + betwn
    return mean.mean(0), var.mean(0)


# --------------------------------------------------------------------------- #
# Distances between two marginal densities on a shared grid
# --------------------------------------------------------------------------- #
def _norm_pdf(d, x):
    area = np.trapz(d, x)
    return d / area if area > 0 else d


def _kl_div(a, b, x):
    """KL(a || b) = integral a log(a/b). a, b are normalised densities on grid x."""
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where((a > 0) & (b > 0), a * np.log(a / np.where(b > 0, b, 1.0)), 0.0)
    return float(np.trapz(term, x))


def density_distances(d_model, d_true, x):
    """Distances of a model marginal to the TRUE marginal on shared grid x.
    KL is KL(model || true). Returns Hellinger (primary), KL, JSD, TVD, Wasserstein-1."""
    p = _norm_pdf(d_model, x)   # model
    q = _norm_pdf(d_true, x)    # true DGP
    m = 0.5 * (p + q)
    hell = np.sqrt(max(0.5 * np.trapz((np.sqrt(p) - np.sqrt(q)) ** 2, x), 0.0))
    kl   = _kl_div(p, q, x)                                   # KL(model || true)
    jsd  = 0.5 * _kl_div(p, m, x) + 0.5 * _kl_div(q, m, x)
    tvd  = 0.5 * np.trapz(np.abs(p - q), x)
    w1   = wasserstein_distance(x, x, u_weights=np.clip(p, 0, None), v_weights=np.clip(q, 0, None))
    return {"Hellinger": hell, "KL": kl, "JSD": jsd, "TVD": tvd, "Wasserstein1": w1}


def distance_table(models, true_model, grids, param_names):
    """Distance of every sampler's marginal to the TRUE DGP marginal, per parameter.
    Samplers are never compared against each other. KL is KL(model || true)."""
    d_true = marginal_density(true_model, grids)
    rows = []
    for m in models:
        d = marginal_density(m, grids)
        for j, pj in enumerate(param_names):
            dist = density_distances(d[j], d_true[j], grids[j])
            rows.append({"sampler": m["name"], "param": pj,
                         **{k: round(v, 5) for k, v in dist.items()}})
    return pd.DataFrame(rows).set_index(["sampler", "param"])


# --------------------------------------------------------------------------- #
# Invariant convergence on the marginal series (REAL chains + arviz)
# --------------------------------------------------------------------------- #
def _as_chains(series):
    """arviz rank-normalised R-hat needs >= 2 chains. For a 1-chain run, split the
    single chain into two halves (split-R-hat; Vehtari et al. 2021; Stan; BDA3
    sec. 11.4) so a valid within-chain value is produced instead of a NaN/warning.
    Note: a lone split chain cannot reveal modes it never visited - see module refs."""
    series = np.asarray(series)
    if series.shape[0] >= 2:
        return series
    h = series.shape[1] // 2
    return np.stack([series[0, :h], series[0, h:2 * h]])


def _rhat(series):
    return float(az.rhat(_as_chains(series)))


def _ess(series):
    return float(az.ess(_as_chains(series)))


def density_series_diagnostics(model, grids, param_names, n_eval=40, density_threshold=0.01):
    """ESS / R-hat of the per-draw marginal density f_{c,s}(x) at a grid of x,
    restricted to the high-density region. The mask uses the INVARIANT marginal,
    not slot-wise means. R-hat/ESS via arviz on the real (C,S) chains."""
    mu, std, pvec = model["mu"], model["std"], model["pvec"]
    C, S, K, P = mu.shape
    marg = marginal_density(model, grids)             # invariant, for the mask
    rows = []
    for j, pj in enumerate(param_names):
        x_full = grids[j]
        eval_pts = np.linspace(x_full.min(), x_full.max(), n_eval)
        marg_at = np.interp(eval_pts, x_full, marg[j])
        keep = eval_pts[marg_at >= marg_at.max() * density_threshold]
        ess_vals, rhat_vals = [], []
        for x in keep:
            pdf = norm.pdf(x, loc=mu[:, :, :, j], scale=std[:, :, :, j] + 1e-8)  # (C,S,K)
            f = np.sum(pvec * pdf, axis=2)                                        # (C,S)
            ess_vals.append(_ess(f))
            rhat_vals.append(_rhat(f))
        if not ess_vals:                      # no point cleared the density mask
            rows.append({"param": pj, "n_pts": 0, "min_ESS": np.nan, "mean_ESS": np.nan,
                         "max_Rhat": np.nan, "mean_Rhat": np.nan})
            continue
        rows.append({"param": pj, "n_pts": int(len(keep)),
                     "min_ESS": float(np.min(ess_vals)), "mean_ESS": float(np.mean(ess_vals)),
                     "max_Rhat": float(np.max(rhat_vals)), "mean_Rhat": float(np.mean(rhat_vals))})
    return pd.DataFrame(rows).set_index("param")


def moment_series_diagnostics(model, param_names):
    """ESS / R-hat of the per-draw marginal mean and variance (Eq. 5.5.2),
    label-invariant, on the real (C,S) chains via arviz."""
    mu, Sigma, pvec = model["mu"], model["Sigma"], model["pvec"]
    C, S, K, P = mu.shape
    rows = []
    for j, pj in enumerate(param_names):
        mu_j  = mu[:, :, :, j]                       # (C,S,K)
        var_j = Sigma[:, :, :, j, j]                 # (C,S,K)
        mean_r = np.sum(pvec * mu_j, axis=2)                          # (C,S)
        var_r  = np.sum(pvec * (var_j + mu_j ** 2), axis=2) - mean_r ** 2
        for series, mom in [(mean_r, "Mean"), (var_r, "Var")]:
            rows.append({"param": pj, "moment": mom,
                         "ESS": _ess(series), "Rhat": _rhat(series)})
    return pd.DataFrame(rows).set_index(["param", "moment"])
