"""
bayesm-exact hybrid Gibbs runner for the AUGMENTED mixture HBMNL model.

This is a line-faithful port of bayesm::rhierMnlRwMixture (Rossi, Allenby &
McCulloch 2006, §5.5, Eq. 5.5.5-5.5.18) onto Liesel/Goose, run against the
model built by src.bayesm_mixture_model.build_bayesm_mixture_hbmnl_model
(explicit allocations `ind`). Correspondence to the bayesm C++ sources:

    bayesm (C++/R)                          here
    ------------------------------------   -----------------------------------
    drawCompsFromLabels / rmultireg         _make_draw_comps   (Eq. 5.5.11-12)
    drawLabelsFromComps                     _make_draw_ind     (Eq. 5.5.9)
    drawPFromLabels                         _make_draw_pvec    (Eq. 5.5.10)
    drawDelta                               _make_draw_delta   (Eq. 5.5.13-18)
    mnlMetropOnce + incroot construction    _make_draw_beta    (Eq. 5.5.5)
    llmnlFract candidate-density tuning     _fractional_candidate_prep

Per-iteration sweep order (exactly rhierMnlRwMixture_rcpp_loop):
    1. {mu_k, Sigma_k} | ind, beta - Z@Delta      (conjugate NIW, masked stats)
    2. ind             | comps_new, pvec_old      (multinomial responsibilities)
    3. pvec            | ind_new                  (Dirichlet)
    4. Delta           | beta, ind, comps         (pooled GLS)
    5. beta_i          | ind_i, comps, Delta      (per-unit RW-Metropolis,
                          increment cov s^2 * (H_i + Sigma_{ind_i}^{-1})^{-1},
                          H_i = unit Hessian at the fractional-likelihood MLE)

bayesm defaults reproduced: s = 2.38/sqrt(n_params) (BayesmConstant.RRScaling;
the package's Roberts-Gelman-Gilks constant - the Rossi 2005 book text instead
states 2.93), w = 0.1, beta_i initialised at the fractional MLEs, ind
initialised in equal blocks.

Because component subsetting is dynamic-shape (forbidden under jit), all
per-component conditionals use one-hot-masked sufficient statistics; for
n_k = 0 these reduce exactly to bayesm's empty-component prior draws.

Every update is drawn in the constrained space and injected into the model's
sampling coordinates through the inverse bijectors (SoftmaxCentered for pvec,
FillScaleTriL for the Cholesky factor of Sigma_k^{-1}).
"""

import time

import numpy as np

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve, solve_triangular

import liesel.goose as gs
import tensorflow_probability.substrates.jax.bijectors as tfb


# --------------------------------------------------------------------------- #
# MNL likelihood on padded per-unit arrays
# --------------------------------------------------------------------------- #
def _unit_loglik(beta, X_u, y_u, mask_u):
    """MNL log-likelihood of one unit. X_u: (T, J, P), y_u: (T,), mask_u: (T,)."""
    logits = jnp.einsum("tjp,p->tj", X_u, beta)
    chosen = jnp.take_along_axis(logits, y_u[:, None], axis=1)[:, 0]
    ll_t = chosen - jax.nn.logsumexp(logits, axis=1)
    return jnp.sum(ll_t * mask_u)


def _prepare_unit_data(data_dict):
    """
    Regroup the flat (n_total, J, P) design into per-unit padded arrays.
    Unequal numbers of observations per unit are handled by zero-padding
    plus a mask (padded rows contribute exactly zero to the likelihood).
    """
    X        = np.asarray(data_dict["X"])
    y        = np.asarray(data_dict["y"]).astype(np.int32)
    unit_idx = np.asarray(data_dict["unit_idx"]).astype(np.int64)
    n_units  = int(data_dict["n_units"])

    counts = np.bincount(unit_idx, minlength=n_units)
    T = int(counts.max())
    J, P = int(X.shape[1]), int(X.shape[2])

    order = np.argsort(unit_idx, kind="stable")
    Xs, ys = X[order], y[order]
    offsets = np.concatenate([[0], np.cumsum(counts)])

    X_units = np.zeros((n_units, T, J, P), dtype=X.dtype)
    y_units = np.zeros((n_units, T), dtype=np.int32)
    mask    = np.zeros((n_units, T), dtype=np.float32)
    for i in range(n_units):
        c = counts[i]
        X_units[i, :c] = Xs[offsets[i]:offsets[i + 1]]
        y_units[i, :c] = ys[offsets[i]:offsets[i + 1]]
        mask[i, :c]    = 1.0

    return jnp.asarray(X_units), jnp.asarray(y_units), jnp.asarray(mask)


# --------------------------------------------------------------------------- #
# Metropolis candidate densities (bayesm's llmnlFract preprocessing)
# --------------------------------------------------------------------------- #
def _newton_minimize(f, x0, iters=40, ridge=1e-6, max_step=10.0):
    """
    Damped Newton minimiser for the (convex) MNL objectives. The step-norm cap
    prevents blow-ups for near-separated units; bayesm handles those via its
    optim convergence flag and the same 0/I fallback applied downstream here.
    """
    grad_f, hess_f = jax.grad(f), jax.hessian(f)
    eye = jnp.eye(x0.shape[0])

    def body(_, x):
        H = hess_f(x) + ridge * eye
        L = jnp.linalg.cholesky(H)
        step = cho_solve((L, True), grad_f(x))
        norm = jnp.linalg.norm(step)
        step = step * jnp.minimum(1.0, max_step / jnp.maximum(norm, 1e-12))
        return x - step

    return jax.lax.fori_loop(0, iters, body, x0)


def _fractional_candidate_prep(X_units, y_units, mask, w):
    """
    bayesm's Metropolis tuning: pooled MLE -> per-unit fractional-likelihood
    MLEs (llmnlFract: (1-w)*ll_i(b) + w*(n_i/N)*(-0.5*(b-bp)'H_pool(b-bp)))
    -> unit Hessians at those MLEs. Failures fall back to beta=0, H=I.
    """
    P = X_units.shape[-1]

    def negll_pooled(b):
        lls = jax.vmap(_unit_loglik, in_axes=(None, 0, 0, 0))(b, X_units, y_units, mask)
        return -jnp.sum(lls)

    betapooled = _newton_minimize(negll_pooled, jnp.zeros(P))
    H_pool = jax.hessian(negll_pooled)(betapooled)

    wgt = mask.sum(axis=1) / mask.sum()          # n_i / N, as in bayesm

    def solve_unit(X_u, y_u, m_u, wg):
        def negfrac(b):
            pen = 0.5 * w * wg * (b - betapooled) @ H_pool @ (b - betapooled)
            return -(1.0 - w) * _unit_loglik(b, X_u, y_u, m_u) + pen

        betafmle = _newton_minimize(negfrac, betapooled)
        H_unit = jax.hessian(lambda b: -_unit_loglik(b, X_u, y_u, m_u))(betafmle)
        return betafmle, H_unit

    betafmle, hess = jax.vmap(solve_unit)(X_units, y_units, mask, wgt)

    bad = ~(jnp.isfinite(betafmle).all(axis=1) & jnp.isfinite(hess).all(axis=(1, 2)))
    betafmle = jnp.where(bad[:, None], 0.0, betafmle)
    hess = jnp.where(bad[:, None, None], jnp.eye(P), hess)
    return betapooled, betafmle, hess, bad


# --------------------------------------------------------------------------- #
# {mu_k, Sigma_k} | ind, theta  (drawCompsFromLabels/rmultireg, Eq. 5.5.11-12)
# --------------------------------------------------------------------------- #
def _niw_conjugate_draw(prng_key, theta, ind, K_comp, n_params, a_mu, nu, V,
                         eye_P, fill_tril):
    """
    One draw of {mu_k, Sigma_k} | ind, theta from their exact bayesm conjugate
    NIW posterior, via masked one-hot sufficient statistics. The single
    implementation of this formula, shared by the Gibbs sweep's draw_comps
    step (this file) and by src.inference.init's NUTS/HMC initial-value
    construction (which reproduces Rossi's first Gibbs draw per chain).
    """
    w1 = jax.nn.one_hot(ind, K_comp, dtype=theta.dtype)               # (n, K)

    n_k   = w1.sum(axis=0)                                            # (K,)
    sum_k = w1.T @ theta                                              # (K, P)
    Syy   = jnp.einsum("nk,np,nq->kpq", w1, theta, theta)             # (K, P, P)

    denom  = n_k + a_mu
    btilde = sum_k / denom[:, None]
    S = Syy - jnp.einsum("kp,kq->kpq", sum_k, sum_k) / denom[:, None, None]
    Vpost = V[None] + S                                               # (K, P, P)
    Vpost = 0.5 * (Vpost + jnp.swapaxes(Vpost, -1, -2))

    # Sigma_k^{-1} ~ Wishart(nu + n_k, Vpost^{-1}) via Bartlett; the draw
    # is produced directly as its lower Cholesky factor M (= model coords).
    Lv = jnp.linalg.cholesky(Vpost)
    Vpost_inv = cho_solve((Lv, True), jnp.broadcast_to(eye_P, Vpost.shape))
    Vpost_inv = 0.5 * (Vpost_inv + jnp.swapaxes(Vpost_inv, -1, -2))
    Ls = jnp.linalg.cholesky(Vpost_inv)                               # (K, P, P)

    k_chi, k_off, k_mu = jax.random.split(prng_key, 3)
    df = nu + n_k                                                     # (K,)
    j = jnp.arange(n_params, dtype=theta.dtype)
    chi2 = 2.0 * jax.random.gamma(k_chi, (df[:, None] - j[None, :]) / 2.0)
    bart = jnp.tril(jax.random.normal(k_off, (K_comp, n_params, n_params)), -1)
    bart = bart + jnp.sqrt(chi2)[:, :, None] * eye_P[None]
    M = Ls @ bart                                                     # (K, P, P)

    # mu_k | Sigma_k ~ N(btilde, Sigma_k / (n_k + a_mu)),  Sigma = (M M')^{-1}
    eps = jax.random.normal(k_mu, (K_comp, n_params))
    dev = solve_triangular(M, eps[..., None], lower=True, trans=1)[..., 0]
    mu_new = btilde + dev / jnp.sqrt(denom)[:, None]

    return {
        "mu_k": mu_new,
        "sigma_inv_chol_k_latent": fill_tril.inverse(M),
    }


# --------------------------------------------------------------------------- #
# Main runner
# --------------------------------------------------------------------------- #
def run_bayesm_gibbs_inference_mixture_hbmnl(
        model,
        data_dict: dict,
        K: int,                      # ── REQUIRED: K_MODEL, for correct reporting
        chains: int = 1,
        r_total: int = 42000,        # total raw Gibbs sweeps (bayesm Mcmc$R)
        burn_in: int = 2000,         # raw sweeps discarded before thinning
        thin: int = 4,               # keep every `thin`-th raw draw after burn-in
        seed: int = 123,
        s: float | None = None,      # RW scale; bayesm default 2.38/sqrt(n_params)
        w: float = 0.1):             # fractional-likelihood weight (bayesm default)
    """
    Configure and run the bayesm-exact hybrid Gibbs sampler (data-augmentation
    Gibbs for the mixture hierarchy + per-unit RW-Metropolis for beta_i) in a
    Liesel/Goose engine.

    The model MUST come from build_bayesm_mixture_hbmnl_model (explicit "ind");
    the prior hyperparameters are read from model.bayesm_prior so the conjugate
    updates provably match the model's prior.

    Iteration scheme matches run_single_bayesm_experiment.R: bayesm is called
    with keep=1 there (every raw draw returned), then burn_in is dropped and
    the remainder thinned by `thin`, both in RAW iteration units. Here the
    Goose posterior epoch runs UNTHINNED for (r_total - burn_in) raw sweeps
    (warmup_duration=burn_in plays the role of bayesm's manual burn-in slice),
    and the thinning is applied afterward with a plain Python stride.
    Posterior sample index j (0-indexed) is raw sweep (burn_in + 1 + j), so
    `[:, ::thin]` keeps j = 0, thin, 2*thin, ... - the same phase as R's
    `seq(burn_in + 1, r_total, by = thin)`.

    Parameters
    ----------
    model     : compiled liesel Model from build_bayesm_mixture_hbmnl_model.
    data_dict : data dictionary with X, y, unit_idx, n_units, n_params, opt. Z.
    K         : number of model components (K_MODEL), for logging only.
    chains    : number of MCMC chains (bayesm itself is single-chain; multiple
                chains differ by RNG stream, as in the R seed-loop convention).
    r_total   : total raw Gibbs sweeps per chain (bayesm default 42000).
    burn_in   : raw sweeps burned via Goose's warmup epoch (bayesm default 2000).
    thin      : post-burn-in thinning interval (bayesm default 4).
    seed      : RNG seed.
    s         : RW-Metropolis scaling; None -> bayesm default 2.38/sqrt(n_params).
    w         : fractional-likelihood weight for candidate-density tuning.

    Returns
    -------
    (results, posterior_samples). `results` is the raw Goose SamplingResults,
    holding all (r_total - burn_in) UNTHINNED posterior sweeps (mirrors what
    NUTS/HMC's mcmc_results.pkl holds). `posterior_samples` is the THINNED dict
    - shape (chains, (r_total-burn_in)//thin, ...) - matching what bayesm's own
    posterior_raw.pkl contains; it additionally has the allocation draws "ind".
    """
    if burn_in >= r_total:
        raise ValueError(f"burn_in ({burn_in}) must be < r_total ({r_total}).")
    prior = getattr(model, "bayesm_prior", None)
    if prior is None:
        raise ValueError(
            "model has no 'bayesm_prior' attribute - this sampler requires the "
            "augmented model from src.bayesm_mixture_model."
            "build_bayesm_mixture_hbmnl_model (explicit 'ind' allocations)."
        )

    n_params = int(data_dict["n_params"])
    n_units  = int(data_dict["n_units"])
    K_comp   = int(prior["K"])
    has_Z    = data_dict.get("Z") is not None

    nu          = float(prior["nu"])
    V           = jnp.asarray(prior["V"])
    a_mu        = float(prior["a_mu"])
    A_delta     = float(prior["A_delta"])
    dirichlet_a = jnp.ones(K_comp) * float(prior["dirichlet_a"])

    if s is None:
        s = 2.38 / np.sqrt(n_params)               # BayesmConstant.RRScaling / sqrt(nvar)
    s = float(s)

    Z = jnp.asarray(data_dict["Z"]) if has_Z else None
    X_units, y_units, mask = _prepare_unit_data(data_dict)

    interface = gs.LieselInterface(model)
    eye_P = jnp.eye(n_params)
    fill_tril = tfb.FillScaleTriL()                # must match the model's bijector
    softmax_c = tfb.SoftmaxCentered()

    print("Starting bayesm-exact Gibbs sampling for mixture HBMNL "
          "(rhierMnlRwMixture port)...")
    print(f" - Demographic covariates (Delta) included : {has_Z}")
    print(f" - Model components (K_MODEL)              : {K}")
    print(f" - RW scale s: {s:.4f} | fractional weight w: {w}")
    print(f" - Prior: nu={nu:.0f}, V=nu*I, a_mu={a_mu}, A_delta={A_delta}, "
          f"dirichlet_a={float(prior['dirichlet_a'])}")
    print(f" - Chains: {chains} | R_total: {r_total} | burn_in: {burn_in} | thin: {thin}")

    print(f"Initializing Metropolis candidate densities for {n_units} units ...")
    t0 = time.time()
    _, betafmle, unit_hess, bad = _fractional_candidate_prep(X_units, y_units, mask, w)
    betafmle.block_until_ready()
    n_bad = int(np.asarray(bad).sum())
    print(f" - done in {time.time() - t0:.1f}s "
          f"({n_bad} unit(s) fell back to beta=0, H=I)" )

    # ── position extraction helpers ────────────────────────────────────────
    hier_keys = ["beta_i", "ind"] + (["Delta"] if has_Z else [])

    def _theta_star(pos):
        """Theta* = beta - Z @ Delta (Rossi Eq. 5.5.8)."""
        if has_Z:
            return pos["beta_i"] - Z @ pos["Delta"]
        return pos["beta_i"]

    # ── 1. {mu_k, Sigma_k} | ind, Theta*  (drawCompsFromLabels/rmultireg) ──
    def draw_comps(prng_key, model_state):
        pos = interface.extract_position(hier_keys, model_state)
        theta = _theta_star(pos)
        return _niw_conjugate_draw(
            prng_key, theta, pos["ind"],
            K_comp, n_params, a_mu, nu, V, eye_P, fill_tril,
        )

    # ── 2. ind | comps, pvec  (drawLabelsFromComps, Eq. 5.5.9) ─────────────
    def draw_ind(prng_key, model_state):
        pos = interface.extract_position(
            hier_keys + ["mu_k", "sigma_inv_chol_k", "pvec"], model_state
        )
        theta = _theta_star(pos)
        L = pos["sigma_inv_chol_k"]                                  # (K, P, P)
        diff = theta[:, None, :] - pos["mu_k"][None, :, :]           # (n, K, P)
        zq = jnp.einsum("kpq,nkp->nkq", L, diff)                     # L'(theta-mu)
        quad = jnp.sum(zq * zq, axis=-1)                             # (n, K)
        logdet = jnp.log(jnp.diagonal(L, axis1=-2, axis2=-1)).sum(-1)  # (K,)
        logits = jnp.log(pos["pvec"])[None, :] + logdet[None, :] - 0.5 * quad
        new_ind = jax.random.categorical(prng_key, logits, axis=-1)
        return {"ind": new_ind.astype(jnp.int32)}

    # ── 3. pvec | ind  (drawPFromLabels, Eq. 5.5.10) ───────────────────────
    def draw_pvec(prng_key, model_state):
        pos = interface.extract_position(["ind"], model_state)
        counts = jax.nn.one_hot(pos["ind"], K_comp).sum(axis=0)
        draw = jax.random.dirichlet(prng_key, dirichlet_a + counts)
        draw = jnp.clip(draw, 1e-10)                 # f32 underflow guard
        draw = draw / draw.sum()
        return {"pvec_latent": softmax_c.inverse(draw)}

    # ── 4. Delta | beta, ind, comps  (drawDelta, Eq. 5.5.16-18) ────────────
    def draw_delta(prng_key, model_state):
        pos = interface.extract_position(
            ["beta_i", "ind", "mu_k", "sigma_inv_chol_k"], model_state
        )
        L = pos["sigma_inv_chol_k"]
        sig_inv = jnp.einsum("kpq,krq->kpr", L, L)                   # (K, P, P)
        w1 = jax.nn.one_hot(pos["ind"], K_comp, dtype=L.dtype)       # (n, K)
        ytil = pos["beta_i"] - pos["mu_k"][pos["ind"]]               # (n, P)

        n_demos = Z.shape[1]
        ZtZ = jnp.einsum("nk,nc,nd->kcd", w1, Z, Z)                  # (K, C, C)
        XtX = jnp.einsum("kcd,kpq->cpdq", ZtZ, sig_inv)
        XtX = XtX.reshape(n_demos * n_params, n_demos * n_params)
        A_cp = jnp.einsum("nk,np,nc->kpc", w1, ytil, Z)              # (K, P, C)
        Xty = jnp.einsum("kpq,kqc->pc", sig_inv, A_cp)               # (P, C)
        Xty = Xty.T.reshape(-1)                                      # vec, demo-major

        prec = XtX + A_delta * jnp.eye(n_demos * n_params)
        Lp = jnp.linalg.cholesky(prec)
        mean = cho_solve((Lp, True), Xty)                            # deltabar = 0
        eps = jax.random.normal(prng_key, mean.shape)
        delta_vec = mean + solve_triangular(Lp, eps, lower=True, trans=1)
        return {"Delta": delta_vec.reshape(n_demos, n_params)}

    # ── 5. beta_i | ind_i, comps, Delta  (mnlMetropOnce, Eq. 5.5.5) ────────
    def draw_beta(prng_key, model_state):
        pos = interface.extract_position(
            hier_keys + ["mu_k", "sigma_inv_chol_k"], model_state
        )
        beta = pos["beta_i"]
        L_i = pos["sigma_inv_chol_k"][pos["ind"]]                    # (n, P, P)
        sig_inv_i = jnp.einsum("npq,nrq->npr", L_i, L_i)
        betabar = pos["mu_k"][pos["ind"]] + (Z @ pos["Delta"] if has_Z else 0.0)

        # increment covariance s^2 * (H_i + Sigma_{ind_i}^{-1})^{-1}
        La = jnp.linalg.cholesky(unit_hess + sig_inv_i)              # (n, P, P)
        k_eps, k_u = jax.random.split(prng_key)
        eps = jax.random.normal(k_eps, beta.shape)
        step = solve_triangular(La, eps[..., None], lower=True, trans=1)[..., 0]
        betac = beta + s * step

        ll_old = jax.vmap(_unit_loglik)(beta,  X_units, y_units, mask)
        ll_new = jax.vmap(_unit_loglik)(betac, X_units, y_units, mask)

        logdet = jnp.log(jnp.diagonal(L_i, axis1=-2, axis2=-1)).sum(-1)

        def log_prior(b):
            d = b - betabar
            zq = jnp.einsum("npq,np->nq", L_i, d)
            return logdet - 0.5 * jnp.sum(zq * zq, axis=-1)

        ldiff = (ll_new + log_prior(betac)) - (ll_old + log_prior(beta))
        accept = jnp.log(jax.random.uniform(k_u, (beta.shape[0],))) < ldiff
        return {"beta_i": jnp.where(accept[:, None], betac, beta)}

    # ── Engine assembly — kernel order is the bayesm sweep order ───────────
    eb = gs.EngineBuilder(seed=seed, num_chains=chains)
    eb.set_model(interface)

    # bayesm initialisation: beta_i at fractional MLEs (ind is already the
    # equal-block split from the model builder).
    state0 = interface.update_state({"beta_i": betafmle}, model.state)
    eb.set_initial_values(state0)

    eb.add_kernel(gs.GibbsKernel(["mu_k", "sigma_inv_chol_k_latent"], draw_comps))
    eb.add_kernel(gs.GibbsKernel(["ind"], draw_ind))
    eb.add_kernel(gs.GibbsKernel(["pvec_latent"], draw_pvec))
    if has_Z:
        eb.add_kernel(gs.GibbsKernel(["Delta"], draw_delta))
    eb.add_kernel(gs.GibbsKernel(["beta_i"], draw_beta))

    # warmup_duration burns the first `burn_in` raw sweeps (bayesm's manual
    # burn-in slice); the posterior epoch runs UNTHINNED for the remaining
    # sweeps - matching bayesm being called with keep=1 - and is thinned
    # afterward in Python (see docstring for the exact index correspondence).
    eb.set_duration(warmup_duration=burn_in, posterior_duration=r_total - burn_in)

    engine = eb.build()
    engine.sample_all_epochs()

    results = engine.get_results()
    posterior_samples_raw = results.get_posterior_samples()

    # Acceptance must be measured on the UNTHINNED chain: between two draws
    # `thin` sweeps apart, beta_i has almost certainly moved at least once,
    # which would make the thinned chain read as ~100% acceptance regardless
    # of the true per-sweep rate.
    acc = beta_acceptance_rates(posterior_samples_raw)
    print(f"RW-Metropolis acceptance (beta_i, per raw sweep): mean {acc.mean():.3f} | "
          f"min {acc.min():.3f} | max {acc.max():.3f} "
          f"(bayesm-typical range ~0.15-0.5)")

    posterior_samples = {
        key: np.asarray(val)[:, ::thin] for key, val in posterior_samples_raw.items()
    }
    n_kept = next(iter(posterior_samples.values())).shape[1]
    print(f"Retained {n_kept} draws/chain from {r_total} raw sweeps "
          f"({burn_in} burned + {r_total - burn_in} sampled, thinned by {thin}).")

    return results, posterior_samples


def beta_acceptance_rates(posterior_samples) -> np.ndarray:
    """
    Per-unit RW-Metropolis acceptance rates, recovered post-hoc as the fraction
    of iterations in which beta_i moved (GibbsKernel does not report MH stats).
    """
    b = np.asarray(posterior_samples["beta_i"])      # (chains, draws, n, P)
    moved = np.any(b[:, 1:] != b[:, :-1], axis=-1)   # (chains, draws-1, n)
    return moved.mean(axis=(0, 1))
