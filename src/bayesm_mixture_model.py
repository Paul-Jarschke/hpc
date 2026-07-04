"""
Bayesian Hierarchical Multinomial Logit (HBMNL) with a mixture-of-normals
heterogeneity distribution — AUGMENTED parameterisation matching bayesm.

This is the model exactly as bayesm::rhierMnlRwMixture samples it: the
component allocation ind_i is an EXPLICIT latent variable (Rossi 2006,
Eq. 5.5.4 DAG), instead of being marginalised out via MixtureSameFamily as
in src.mixturemodel.build_mixture_hbmnl_model:

    ind_i  ~ Categorical(pvec)
    beta_i = Z[i] @ Delta + u_i,   u_i | ind_i ~ N(mu_{ind_i}, Sigma_{ind_i})

Marginally over ind the posterior of (pvec, mu_k, Sigma_k, Delta, beta_i) is
IDENTICAL to the marginalised model, so results are directly comparable.
The explicit allocations exist so that the conjugate data-augmentation Gibbs
sweep of bayesm (Rossi Eq. 5.5.7 / 5.5.9-5.5.18) can be run on the model via
src.inference.bayesm_gibbs.run_bayesm_gibbs_inference_mixture_hbmnl — and so
that allocation draws are available for label.switching post-processing.

Prior structure (identical to the marginalised builder and to bayesm defaults):

    Wishart:   Sigma_k^{-1} ~ W(nu, V^{-1}),  nu = n_params + 3,  V = nu * I
    Normal:    mu_k | Sigma_k ~ N(0, Sigma_k / a_mu)
    Normal:    Delta          ~ N(0, (1/A_delta) * I)
    Dirichlet: pvec           ~ Dir(dirichlet_a)

NOTE: ind is a discrete parameter. It can only be updated by a Gibbs kernel —
gradient-based kernels (NUTS/HMC/IWLS) must never own the "ind" position key.
"""

import numpy as np
import jax.numpy as jnp

import liesel.model as lsl
import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.bijectors as tfb

from src.mixturemodel import _make_mvn_precision, _make_wishart


def bayesm_initial_indicators(n_units: int, K: int) -> np.ndarray:
    """
    bayesm's initial allocation: (K-1) equal blocks of floor(n/K) units,
    remainder assigned to the last component (rhierMnlRwMixture.R).
    """
    if K == 1:
        return np.zeros(n_units, dtype=np.int32)
    ninc = n_units // K
    ind0 = np.concatenate(
        [np.full(ninc, k, dtype=np.int32) for k in range(K - 1)]
        + [np.full(n_units - ninc * (K - 1), K - 1, dtype=np.int32)]
    )
    return ind0


# --------------------------------------------------------------------------- #
# Main model builder
# --------------------------------------------------------------------------- #
def build_bayesm_mixture_hbmnl_model(
        data_dict: dict,
        K: int,                      # ── REQUIRED: number of model components (K_MODEL)
        A_delta: float = 0.01,
        a_mu: float = 0.01,
        dirichlet_a: float = 1.0):
    """
    Build a K-component mixture HBMNL Liesel model with EXPLICIT allocations
    (bayesm-augmented parameterisation).

    Parameters
    ----------
    data_dict   : dict with 'X', 'y', 'unit_idx', 'n_params', 'n_units' and
                  optionally 'Z' (centred, no intercept column).
                  NOTE: 'K' is intentionally ignored if present.
    K           : number of mixture components the MODEL should fit (K_MODEL).
                  This is a modelling choice, decoupled from the data's true K.
    A_delta     : prior precision scaling for Delta (if demographics Z used).
    a_mu        : prior precision scaling for the mixture component means.
    dirichlet_a : Dirichlet concentration on pvec. < 1.0 encourages sparse
                  weights (spurious components collapse toward 0), useful when
                  K_MODEL > K_TRUE; == 1.0 is uniform (uninformative on the
                  simplex - does not assume overspecification); > 1.0 pulls
                  toward equal weights.

    Returns
    -------
    A compiled liesel.model.Model with an extra attribute ``bayesm_prior``
    holding the hyperparameters, read by the bayesm_gibbs runner so that its
    conjugate updates are guaranteed to match the model's prior.
    """
    if K is None:
        raise ValueError(
            "K (number of model components) must be supplied explicitly. "
            "It is decoupled from data_dict['K'] by design."
        )

    n_params = int(data_dict["n_params"])
    n_units  = int(data_dict["n_units"])
    K_comp   = int(K)                                   # K_MODEL - never read from data
    has_Z    = data_dict.get("Z") is not None

    # ── Wishart prior ──────────────────────────────────────────────────────
    nu          = float(n_params + 3)
    V           = nu * jnp.eye(n_params)
    Vinv_chol   = jnp.linalg.cholesky(jnp.linalg.inv(V))
    Vinv_chol_K = jnp.broadcast_to(Vinv_chol[None], (K_comp, n_params, n_params))

    # ── pvec ~ Dirichlet ───────────────────────────────────────────────────
    pvec = lsl.Var.new_param(
        value=jnp.ones(K_comp) / K_comp,
        distribution=lsl.Dist(
            tfd.Dirichlet,
            concentration=jnp.ones(K_comp) * dirichlet_a,
        ),
        name="pvec",
    )
    pvec_latent = pvec.transform(tfb.SoftmaxCentered(), name="pvec_latent")

    # ── Sigma_k^{-1} ~ Wishart via Cholesky ────────────────────────────────
    sigma_inv_chol_k = lsl.Var.new_param(
        value=jnp.broadcast_to(jnp.eye(n_params)[None], (K_comp, n_params, n_params)),
        distribution=lsl.Dist(
            _make_wishart,
            df=jnp.full(K_comp, nu),
            scale_tril=Vinv_chol_K,
        ),
        name="sigma_inv_chol_k",
    )
    sigma_inv_chol_k_latent = sigma_inv_chol_k.transform(
        tfb.FillScaleTriL(), name="sigma_inv_chol_k_latent"
    )

    # ── mu_k | Sigma_k ~ N(0, Sigma_k / a_mu) ──────────────────────────────
    mu_prec_factor_k = lsl.Var.new_calc(
        lambda L: jnp.sqrt(a_mu) * L,
        L=sigma_inv_chol_k,
        name="mu_prec_factor_k",
    )
    mu_k = lsl.Var.new_param(
        value=jnp.zeros((K_comp, n_params)),
        distribution=lsl.Dist(
            _make_mvn_precision,
            loc=jnp.zeros(n_params),
            precision_factor=mu_prec_factor_k,
        ),
        name="mu_k",
    )

    # ── ind_i ~ Categorical(pvec) — the explicit allocation (Eq. 5.5.4) ────
    ind = lsl.Var.new_param(
        value=jnp.asarray(bayesm_initial_indicators(n_units, K_comp)),
        distribution=lsl.Dist(tfd.Categorical, probs=pvec),
        name="ind",
    )

    # ── Delta ~ N(0, (1/A_delta) * I) ──────────────────────────────────────
    if has_Z:
        n_demos           = int(data_dict["Z"].shape[1])
        Z_var             = lsl.Var.new_obs(data_dict["Z"], name="Z_obs")
        Delta_prec_factor = jnp.sqrt(A_delta) * jnp.eye(n_params)

        Delta = lsl.Var.new_param(
            value=jnp.zeros((n_demos, n_params)),
            distribution=lsl.Dist(
                _make_mvn_precision,
                loc=jnp.zeros(n_params),
                precision_factor=Delta_prec_factor,
            ),
            name="Delta",
        )
        z_delta = lsl.Var.new_calc(
            lambda z, d: z @ d, z=Z_var, d=Delta, name="z_delta"
        )

    # ── beta_i | ind_i: unit-specific normal via allocation indexing ───────
    if has_Z:
        beta_loc_i = lsl.Var.new_calc(
            lambda zd, mu, i: zd + mu[i],
            zd=z_delta, mu=mu_k, i=ind,
            name="beta_loc_i",
        )
    else:
        beta_loc_i = lsl.Var.new_calc(
            lambda mu, i: mu[i],
            mu=mu_k, i=ind,
            name="beta_loc_i",
        )
    beta_prec_factor_i = lsl.Var.new_calc(
        lambda L, i: L[i],
        L=sigma_inv_chol_k, i=ind,
        name="beta_prec_factor_i",
    )

    beta_i = lsl.Var.new_param(
        value=jnp.zeros((n_units, n_params)),
        distribution=lsl.Dist(
            _make_mvn_precision,
            loc=beta_loc_i,
            precision_factor=beta_prec_factor_i,
        ),
        name="beta_i",
    )

    # ── Likelihood ─────────────────────────────────────────────────────────
    X_var         = lsl.Var.new_obs(data_dict["X"],        name="X_obs")
    idx_var       = lsl.Var.new_obs(data_dict["unit_idx"], name="idx_obs")
    beta_expanded = lsl.Var.new_calc(
        lambda b, idx: b[idx], b=beta_i, idx=idx_var, name="beta_expanded"
    )
    logits = lsl.Var.new_calc(
        lambda x, b: jnp.einsum("nij,nj->ni", x, b),
        x=X_var, b=beta_expanded,
        name="logits",
    )
    y_var = lsl.Var.new_obs(
        data_dict["y"],
        distribution=lsl.Dist(tfd.Categorical, logits=logits),
        name="y",
    )

    model = lsl.Model([y_var])

    # Hyperparameters travel with the model so the Gibbs runner's conjugate
    # updates provably match the prior actually built here.
    model.bayesm_prior = {
        "K": K_comp,
        "nu": nu,
        "V": np.asarray(V),
        "a_mu": float(a_mu),
        "A_delta": float(A_delta),
        "dirichlet_a": float(dirichlet_a),
        "has_Z": has_Z,
    }
    return model
