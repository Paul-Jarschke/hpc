"""
Marginal-density comparison across samplers for the mixture HBMNL (Rossi et al.
2006, Eq. 5.5.19 and 5.5.2).

One comparison per `<chains>/<k>_comp/` folder: it contrasts the NUTS, HMC and
bayesm runs that sit side by side there against each other and the true DGP, on
LABEL-INVARIANT quantities only - so no relabeling is needed (a per-draw
permutation of the component triples leaves every quantity here unchanged).

Methodology
-----------
1. Grids are anchored to the FITTED models' own support (union over samplers,
   wide); the true DGP is an overlay only, since anchoring the grid to the
   truth would clip the tails/lobes Rossi highlights and be circular.
2. Convergence is assessed on the REAL (chains, draws) label-invariant series:
   arviz rank-normalised split-R-hat/ESS for scalar series, plus two reductions
   of the curve-valued marginal chain itself - `curve_diagnostics` (ONE R-hat
   and ONE ESS per marginal: Brooks-Gelman multivariate PSRF, Vats-Flegal-Jones
   multivariate ESS) and `functional_diagnostics` (Goose-identical arviz calls
   on grid-free scalar functionals of each per-draw marginal).
3. The density-support mask uses the INVARIANT marginal (per-draw then
   average); slot-wise posterior means would mix components under label
   switching.

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
- Brooks & Gelman (1998), "General methods for monitoring convergence of
  iterative simulations", JCGS 7(4):434-455 - the multivariate PSRF (the exact
  maximum split-R-hat over all linear functionals of a vector/curve, via one
  generalized eigenvalue) used by `curve_diagnostics`.
- Vats, Flegal & Jones (2019), "Multivariate output analysis for Markov chain
  Monte Carlo", Biometrika 106(2):321-337 - the multivariate ESS
  N*(det Lambda / det Sigma)^(1/p) with batch-means Sigma, used by
  `curve_diagnostics`.

Standard (single-component) model
----------------------------------
`load_sampler_standard`/`true_dgp_standard` load the plain (no `_k` suffix, no
pvec) posterior/DGP keys of the standard HBMNL and package them with a size-1
component axis and pvec == 1; every other function in this module is written
generically in terms of K and applies as-is. With one component there is no
label-switching, so samplers' posterior distributions of `mu` can also be
compared directly (not just via the derived invariant density); that plot,
like all plotting, lives in the notebook layer.
"""

import json
import pickle
import pathlib

import numpy as np
import pandas as pd
import xarray as xr
import arviz as az
from scipy.special import ndtr
from scipy.stats import norm, wasserstein_distance

from src import analysis


# --------------------------------------------------------------------------- #
# Loading - every sampler exposes the same arrays, so one path serves all three
# --------------------------------------------------------------------------- #
def _run_duration(results_dir):
    """Total fit wall-clock in seconds from the run's meta.json (None if absent)."""
    meta_path = pathlib.Path(results_dir) / "meta.json"
    if not meta_path.exists():
        return None
    d = json.load(open(meta_path)).get("duration_s")
    return float(d) if d else None


def load_sampler(results_dir, name):
    """Load one run's draws as (C,S,K,P)/(C,S,K) arrays. is_mcmc=True;
    duration_s carries the fit's total wall-clock (from meta.json)."""
    rd = pathlib.Path(results_dir)
    with open(rd / "posterior_raw.pkl", "rb") as f:
        post = pickle.load(f)
    mu    = np.asarray(post["mu_k"])                                  # (C,S,K,P)
    pvec  = np.asarray(analysis._recover_pvec(post))                 # (C,S,K)
    Sigma = np.asarray(
        analysis._sigma_from_latent(np.asarray(post["sigma_inv_chol_k_latent"]))
    )                                                                # (C,S,K,P,P)
    std = np.sqrt(np.clip(np.diagonal(Sigma, axis1=3, axis2=4), 0.0, None))  # (C,S,K,P)
    return {"name": name, "mu": mu, "pvec": pvec, "Sigma": Sigma, "std": std,
            "is_mcmc": True, "duration_s": _run_duration(rd)}


def true_dgp_model(raw_data):
    """Ground-truth components as a 1-chain, 1-draw 'model' (for overlay only)."""
    mu    = np.array(raw_data["TRUE_MU_K"])               # (K_true, P)
    pvec  = np.array(raw_data["TRUE_PVEC"]).ravel()       # (K_true,)
    Sigma = np.array(raw_data["TRUE_SIGMA_K"])            # (K_true, P, P)
    std   = np.sqrt(np.clip(np.diagonal(Sigma, axis1=1, axis2=2), 0.0, None))
    return {"name": "True DGP",
            "mu": mu[None, None], "pvec": pvec[None, None],
            "Sigma": Sigma[None, None], "std": std[None, None], "is_mcmc": False}


def load_sampler_standard(results_dir, name):
    """Load one standard-model (K=1) run's draws, packaged with a size-1
    component axis and pvec == 1 so the rest of this module (grids,
    marginal_density, mixture_moments, distances, convergence diagnostics)
    applies completely unchanged. duration_s as in `load_sampler`."""
    rd = pathlib.Path(results_dir)
    with open(rd / "posterior_raw.pkl", "rb") as f:
        post = pickle.load(f)
    mu    = np.asarray(post["mu"])[:, :, None, :]                                # (C,S,1,P)
    Sigma = analysis._sigma_from_latent(
        np.asarray(post["sigma_inv_chol_latent"])
    )[:, :, None, :, :]                                                          # (C,S,1,P,P)
    std   = np.sqrt(np.clip(np.diagonal(Sigma, axis1=3, axis2=4), 0.0, None))    # (C,S,1,P)
    pvec  = np.ones(mu.shape[:3])                                                # (C,S,1)
    return {"name": name, "mu": mu, "pvec": pvec, "Sigma": Sigma, "std": std,
            "is_mcmc": True, "duration_s": _run_duration(rd)}


def true_dgp_standard(raw_data):
    """Ground truth for the standard (K=1) model as a 1-chain, 1-draw, 1-component
    'model' (for overlay only), in the same shape convention as true_dgp_model."""
    mu    = np.array(raw_data["TRUE_MU"])[None, :]           # (1, P)
    Sigma = np.array(raw_data["TRUE_SIGMA"])[None, :, :]     # (1, P, P)
    std   = np.sqrt(np.clip(np.diagonal(Sigma, axis1=1, axis2=2), 0.0, None))
    pvec  = np.ones(1)
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


def build_grids_full(fitted_models, true_model=None, n_grid=4000, n_sigma=6):
    """Alternative to `build_grids` that sets NO clipping bounds.

    Unlike `build_grids` - which anchors the grid to the per-draw TOP-K_true
    ('live') components and trims the extremes with a percentile - this builder
    spans the FULL support of EVERY component of EVERY model (all K, including the
    surplus/empty ones when K_MODEL > K_TRUE) AND of the true DGP, using the raw
    min/max envelope (mu +/- n_sigma*std, no trimming). Nothing is excluded, so
    every distance metric (Hellinger, KL, JSD, TVD, Wasserstein-1) integrates over
    the ENTIRE marginal distribution - the true marginal's full tails plus any
    diffuse model mass - rather than only the fitted live-component region.

    Caveat: when K_MODEL > K_TRUE the surplus components carry huge prior-driven
    mu/sigma, so this range can be very wide and the real mass occupies few grid
    cells; `n_grid` and `n_sigma` are raised accordingly to keep resolution. This
    is the trade-off of the unbounded grid - contrast `build_grids`
    (live-support, trimmed). Truth is included in the envelope but is still an
    overlay in the plots."""
    P = fitted_models[0]["mu"].shape[-1]
    lo = np.full(P, np.inf)
    hi = np.full(P, -np.inf)
    models = list(fitted_models) + ([true_model] if true_model is not None else [])
    for m in models:
        mu, _, std, _, _, _ = _flat(m)
        e_lo = (mu - n_sigma * std).reshape(-1, P)
        e_hi = (mu + n_sigma * std).reshape(-1, P)
        lo = np.minimum(lo, e_lo.min(axis=0))
        hi = np.maximum(hi, e_hi.max(axis=0))
    return [np.linspace(lo[j], hi[j], n_grid) for j in range(P)]


def build_grids_chebyshev(fitted_models, true_model=None, n_grid=2000, k=5.0):
    """Per-parameter grid clipped to each model's own [mean - k*std, mean + k*std],
    unioned over samplers (and the True DGP if given). mean/std are the AGGREGATE
    mixture mean and variance (Rossi Eq. 5.5.2, via `mixture_moments`), not
    per-component - so a diffuse surplus/empty component is already down-weighted
    by its own pvec before the window is set, unlike `build_grids_full`'s raw
    per-component min/max envelope.

    Chebyshev's inequality, P(|X - mean| >= k*std) <= 1/k**2, holds for ANY
    distribution with finite variance - no normality/unimodality assumption,
    which matters here since the invariant marginal is itself a mixture and can
    be skewed or multimodal. k=5 -> at least 1 - 1/5**2 = 96% of each model's own
    marginal mass is guaranteed to lie inside its window before the union.

    Contrast `build_grids_full` (unbounded raw envelope over every component -
    can be arbitrarily wide when K_MODEL > K_TRUE, squashing real mass into a
    few pixels) and `build_grids` (per-draw live-component top-K_true support,
    percentile-trimmed - excludes surplus components entirely)."""
    P = fitted_models[0]["mu"].shape[-1]
    lo = np.full(P, np.inf)
    hi = np.full(P, -np.inf)
    models = list(fitted_models) + ([true_model] if true_model is not None else [])
    for m in models:
        mean, var = mixture_moments(m)
        std = np.sqrt(np.clip(np.diag(var), 0.0, None))
        lo = np.minimum(lo, mean - k * std)
        hi = np.maximum(hi, mean + k * std)
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


def distance_table(models, true_model, grids, param_names, dens=None, dens_true=None):
    """Distance of every sampler's marginal to the TRUE DGP marginal, per parameter.
    Samplers are never compared against each other. KL is KL(model || true).

    `dens` (dict: model name -> marginal_density(...)) and `dens_true` may be
    passed in to reuse densities already computed for this (models, grids) pair
    and skip the O(R*K*n_grid) cost of `marginal_density`."""
    d_true = dens_true if dens_true is not None else marginal_density(true_model, grids)
    rows = []
    for m in models:
        d = dens[m["name"]] if dens is not None else marginal_density(m, grids)
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


def _rhat_ess_batch(series):
    """Vectorized R-hat and ESS over a trailing 'point' dim: series is (C,S,n_points).
    One az.rhat/az.ess call each (via xarray, which vectorizes elementwise over
    extra dims); arviz has a fixed per-call setup cost, so per-point calls would
    dominate the runtime. Returns (rhat (n_points,), ess (n_points,))."""
    chains = _as_chains(series)                                  # (C',S,n_points)
    ds = xr.Dataset({"x": xr.DataArray(chains, dims=("chain", "draw", "point"))})
    return az.rhat(ds)["x"].values, az.ess(ds)["x"].values


def density_series_diagnostics(model, grids, param_names, n_eval=40, density_threshold=0.01, marg=None):
    """ESS / R-hat of the per-draw marginal density f_{c,s}(x) at a grid of x,
    restricted to the high-density region. The mask uses the INVARIANT marginal,
    not slot-wise means. R-hat/ESS via arviz on the real (C,S) chains, batched
    across all surviving grid points in one call (see `_rhat_ess_batch`).

    `marg` (this model's marginal_density(model, grids) result) may be passed in
    to reuse a density already computed for this (model, grids) pair."""
    mu, std, pvec = model["mu"], model["std"], model["pvec"]
    C, S, K, P = mu.shape
    if marg is None:
        marg = marginal_density(model, grids)         # invariant, for the mask
    rows = []
    for j, pj in enumerate(param_names):
        x_full = grids[j]
        eval_pts = np.linspace(x_full.min(), x_full.max(), n_eval)
        marg_at = np.interp(eval_pts, x_full, marg[j])
        keep = eval_pts[marg_at >= marg_at.max() * density_threshold]
        if len(keep):
            pdf = norm.pdf(keep[None, None, None, :],
                            loc=mu[:, :, :, j, None], scale=std[:, :, :, j, None] + 1e-8)  # (C,S,K,n_keep)
            f = np.sum(pvec[:, :, :, None] * pdf, axis=2)                                   # (C,S,n_keep)
            rhat_vals, ess_vals = _rhat_ess_batch(f)
        else:
            rhat_vals, ess_vals = [], []
        if not len(ess_vals):                 # no point cleared the density mask
            rows.append({"param": pj, "n_pts": 0, "min_ESS": np.nan, "mean_ESS": np.nan,
                         "max_Rhat": np.nan, "mean_Rhat": np.nan})
            continue
        rows.append({"param": pj, "n_pts": int(len(keep)),
                     "min_ESS": float(np.min(ess_vals)), "mean_ESS": float(np.mean(ess_vals)),
                     "max_Rhat": float(np.max(rhat_vals)), "mean_Rhat": float(np.mean(rhat_vals))})
    return pd.DataFrame(rows).set_index("param")


# --------------------------------------------------------------------------- #
# Convergence of the marginals themselves - two single-number reductions
# --------------------------------------------------------------------------- #
# Diagnosed object: the PER-DRAW marginal density of parameter j,
#     d_j^{(c,s)}(x) = sum_k pvec_k^{(c,s)} N(x ; mu_kj^{(c,s)}, sd_kj^{(c,s)})
# (the summand of Rossi Eq. 5.5.19) - one label-invariant curve per MCMC draw.
# (A) `curve_diagnostics`  - ONE R-hat and ONE ESS per marginal, via multivariate
#     diagnostics of the discretized curve (Brooks-Gelman 1998 MPSRF; Vats-
#     Flegal-Jones 2019 multivariate ESS). Invariant to nonsingular linear
#     re-encodings of the curve, so only the grid's SPAN matters.
# (B) `functional_diagnostics` - Goose-identical scalars: az.rhat / az.ess (the
#     exact calls in liesel.goose.summary_m) on grid-free closed-form
#     functionals of each draw's marginal (mean, sd, quantiles).
def _split_halves(F):
    """(C, S, m) -> (2C, S//2, m): split each chain in half (Vehtari et al. 2021),
    so 1-chain runs give a within-chain check like the scalar diagnostics."""
    h = F.shape[1] // 2
    return np.concatenate([F[:, :h], F[:, h:2 * h]], axis=0)


def _project_curves(F2, rel_tol):
    """Project split-chain curves (M, n, m) onto the principal subspace of the
    pooled draw covariance -> scores (M, n, r). Nearby grid points are nearly
    collinear, making the raw m-dim covariances numerically singular; the
    diagnostics are invariant under nonsingular linear maps, so dropping only
    directions with eigenvalue <= rel_tol * largest changes conditioning, not
    the result."""
    M, n, m = F2.shape
    X = F2.reshape(M * n, m)
    Xc = X - X.mean(axis=0)
    _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
    lam = (s ** 2) / max(M * n - 1, 1)
    if lam.size == 0 or lam[0] <= 0.0:                 # constant curves
        return Xc.reshape(M, n, m)[:, :, :1]
    r = max(int(np.sum(lam > lam[0] * rel_tol)), 1)
    return (Xc @ Vt[:r].T).reshape(M, n, r)


def _mpsrf_from_scores(Y):
    """Brooks-Gelman multivariate PSRF of score chains (M, n, r), sqrt scale:
    sqrt((n-1)/n + ((M+1)/M) * lambda_max(W^{-1} B/n))."""
    M, n, r = Y.shape
    means = Y.mean(axis=1)                                     # (M, r)
    dev = means - means.mean(axis=0)
    Bn = (dev.T @ dev) / (M - 1)                               # B/n
    Yc = Y - means[:, None, :]
    W = np.einsum("cnp,cnq->pq", Yc, Yc) / (M * (n - 1))
    W = W + np.eye(r) * (np.trace(W) / r) * 1e-12              # conditioning floor
    L = np.linalg.cholesky(W)
    T = np.linalg.solve(L, Bn)
    T = np.linalg.solve(L, T.T)                                # L^-1 (B/n) L^-T
    lam1 = max(float(np.linalg.eigvalsh(0.5 * (T + T.T))[-1]), 0.0)
    return float(np.sqrt((n - 1) / n + (M + 1) / M * lam1))


def _mess_from_scores(Y, batch_size=None):
    """Vats-Flegal-Jones multivariate ESS of score chains (M, n, r):
    mESS = N * (det Lambda / det Sigma)^(1/r) - a geometric mean over the curve's
    principal directions, so one badly-mixing direction is diluted by the others.
    ESS_min = N * lambda_min(pencil(Lambda, Sigma)) is therefore also returned:
    the exact minimum of N * a'Lambda a / a'Sigma a over all linear functionals
    a - the worst-case companion to the MPSRF maximum. Sigma via grand-centred
    replicated batch means (batch size ~ sqrt(n)), so chains sitting in
    different modes inflate Sigma and deflate both numbers. Returns
    (mESS, ESS_min, r_used); r is capped at the batch-mean degrees of freedom
    (scores are PCA-ordered, leading ones kept)."""
    M, n, r = Y.shape
    if batch_size is None:
        batch_size = max(int(np.floor(np.sqrt(n))), 2)
    b = min(int(batch_size), n)
    a = n // b
    if a < 2:
        return float("nan"), float("nan"), r
    r_cap = M * a - 1
    if r > r_cap:
        Y = Y[:, :, :r_cap]
        r = r_cap
    N = M * n
    X = Y.reshape(N, r)
    grand = X.mean(axis=0)
    Xc = X - grand
    Lam = (Xc.T @ Xc) / (N - 1)
    bm = Y[:, :a * b].reshape(M, a, b, r).mean(axis=2) - grand         # (M, a, r)
    Sig = b * np.einsum("cap,caq->pq", bm, bm) / (M * a - 1)
    jit_L = np.eye(r) * (np.trace(Lam) / r) * 1e-12
    jit_S = np.eye(r) * (np.trace(Sig) / r) * 1e-12
    Lam, Sig = Lam + jit_L, Sig + jit_S
    _, ld_L = np.linalg.slogdet(Lam)
    _, ld_S = np.linalg.slogdet(Sig)
    mess = float(N * np.exp((ld_L - ld_S) / r))
    Ls = np.linalg.cholesky(Sig)
    T = np.linalg.solve(Ls, Lam)
    T = np.linalg.solve(Ls, T.T)                                       # Ls^-1 Lam Ls^-T
    ess_min = float(N * max(np.linalg.eigvalsh(0.5 * (T + T.T))[0], 0.0))
    return mess, ess_min, r


def mpsrf(F, rel_tol=1e-8):
    """ONE split-R-hat for a curve-valued chain F (chains, draws, n_points):
    Brooks & Gelman's (1998) multivariate PSRF - the exact maximum split-R-hat
    over ALL linear functionals of the curve (every point evaluation, interval
    probability and weighted mean), via one generalized eigenvalue; NOT an
    aggregation of pointwise R-hats. sqrt scale, reads like arviz R-hat.
    Invariant to nonsingular linear re-encodings of the curve."""
    Y = _project_curves(_split_halves(np.asarray(F, dtype=float)), rel_tol)
    return _mpsrf_from_scores(Y)


def multivariate_ess(F, rel_tol=1e-8, batch_size=None):
    """ONE ESS for a curve-valued chain F (chains, draws, n_points): Vats,
    Flegal & Jones (2019). Interpretation: the number of INDEPENDENT draws of
    the curve whose large-sample confidence ellipsoid for the mean curve has
    the same volume as the chain's. A geometric mean across directions - see
    `curve_diagnostics`'s ESS_min column for the worst-case companion. Only
    meaningful when mpsrf() is ~1."""
    Y = _project_curves(_split_halves(np.asarray(F, dtype=float)), rel_tol)
    ess, _, _ = _mess_from_scores(Y, batch_size)
    return ess


def curve_diagnostics(model, grids, param_names, n_eval=64,
                      rel_tol=1e-8, batch_size=None):
    """(A) ONE R-hat and ONE ESS per marginal: multivariate diagnostics of the
    per-draw marginal density curves. No pointwise aggregation and no density
    mask - low-density grid points carry ~zero variance and are absorbed by the
    principal-subspace projection.

    Columns: Rhat_max (`mpsrf` - max split-R-hat over all linear functionals),
    mESS (`multivariate_ess` - volume-based, a geometric mean across directions),
    ESS_min (worst-case ESS over all linear functionals - the min companion to
    Rhat_max's max), rank (effective dimension the curves occupy), draws (total
    unsplit draws). Values depend on the grid only through its span, not its
    resolution."""
    mu, std, pvec = model["mu"], model["std"], model["pvec"]
    C, S = mu.shape[:2]
    rows = []
    for j, pj in enumerate(param_names):
        x = np.linspace(grids[j][0], grids[j][-1], n_eval)
        pdf = norm.pdf(x[None, None, None, :],
                       loc=mu[:, :, :, j, None], scale=std[:, :, :, j, None] + 1e-8)
        F = np.sum(pvec[:, :, :, None] * pdf, axis=2)                 # (C, S, n_eval)
        Y = _project_curves(_split_halves(F), rel_tol)
        ess, ess_min, _ = _mess_from_scores(Y, batch_size)
        rows.append({"param": pj, "Rhat_max": _mpsrf_from_scores(Y),
                     "mESS": ess, "ESS_min": ess_min,
                     "rank": int(Y.shape[2]), "draws": int(C * S)})
    return pd.DataFrame(rows).set_index("param")


def _mixture_quantiles(mu_j, sd_j, pvec, alphas, iters=50):
    """Per-draw quantiles of the mixture marginal by bisection on its CDF
    G(q) = sum_k pvec_k Phi((q - mu_k)/sd_k). mu_j, sd_j, pvec: (C, S, K).
    Returns (len(alphas), C, S)."""
    lo = (mu_j - 8.0 * sd_j).min(axis=2)
    hi = (mu_j + 8.0 * sd_j).max(axis=2)
    out = np.empty((len(alphas),) + lo.shape)
    for i, alpha in enumerate(alphas):
        a, b = lo.copy(), hi.copy()
        for _ in range(iters):
            mid = 0.5 * (a + b)
            cdf = np.sum(pvec * ndtr((mid[..., None] - mu_j) / sd_j), axis=2)
            hi_side = cdf > alpha
            b = np.where(hi_side, mid, b)
            a = np.where(hi_side, a, mid)
        out[i] = 0.5 * (a + b)
    return out


def functional_diagnostics(model, param_names, quantiles=(0.05, 0.5, 0.95)):
    """(B) Goose-identical convergence diagnostics for scalar functionals of
    each per-draw marginal: mean, sd, and the given quantiles. Rhat is az.rhat
    (rank-normalised split-R-hat) and ESS az.ess bulk/tail - exactly the calls
    liesel.goose.summary_m makes - so values read like any Goose summary. All
    functionals are closed-form (mean, sd) or CDF root-finding (quantiles): no
    grid is involved. 1-chain runs are split into halves (within-chain check).

    When the model carries duration_s (set by the loaders from meta.json - the
    fit's total wall-clock incl. warmup), ESS_bulk/s and ESS_tail/s columns are
    added: effective draws per second, the cross-sampler efficiency metric."""
    mu, std, pvec = model["mu"], model["std"], model["pvec"]
    duration = model.get("duration_s")
    rows = []
    for j, pj in enumerate(param_names):
        mu_j, sd_j = mu[:, :, :, j], std[:, :, :, j]
        mean = np.sum(pvec * mu_j, axis=2)
        var = np.sum(pvec * (sd_j ** 2 + mu_j ** 2), axis=2) - mean ** 2
        series = {"mean": mean, "sd": np.sqrt(np.clip(var, 0.0, None))}
        for alpha, q in zip(quantiles, _mixture_quantiles(mu_j, sd_j, pvec, quantiles)):
            series[f"q{int(round(alpha * 100)):02d}"] = q
        for name, s in series.items():
            row = {"param": pj, "functional": name,
                   "Rhat": _rhat(s), "ESS_bulk": _ess(s),
                   "ESS_tail": float(az.ess(_as_chains(s), method="tail"))}
            if duration:
                row["ESS_bulk/s"] = row["ESS_bulk"] / duration
                row["ESS_tail/s"] = row["ESS_tail"] / duration
            rows.append(row)
    return pd.DataFrame(rows).set_index(["param", "functional"])


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
