"""
Bayesian Hierarchical Multinomial Logit (HBMNL) with a
mixture-of-normals heterogeneity distribution.
 
The number of mixture components fitted by the model (K_MODEL) is supplied
explicitly by the caller and is INDEPENDENT of the number of components in
the data-generating process (K_TRUE). This module never reads "K" from
data_dict — that key is a property of the data, not a modelling decision.
 
Prior structure matches bayesm::rhierMnlRwMixture:
 
    beta_i = Z[i] @ Delta + u_i,   u_i ~ N(mu_k, Sigma_k),  k ~ Categorical(pvec)
 
    Wishart:   Sigma_k^{-1} ~ W(nu, V^{-1}),  nu = n_params + 3,  V = nu * I
    Normal:    mu_k | Sigma_k ~ N(0, Sigma_k / a_mu)
    Normal:    Delta          ~ N(0, (1/A_delta) * I)
    Dirichlet: pvec           ~ Dir(dirichlet_a)
"""
 
import jax.numpy as jnp
 
import liesel.model as lsl
import tensorflow_probability.substrates.jax.distributions as tfd
import tensorflow_probability.substrates.jax.bijectors as tfb
from tensorflow_probability.substrates.jax.experimental import distributions as tfde
from tensorflow_probability.python.internal.backend.jax.compat import v2 as tf


# --------------------------------------------------------------------------- #
# Helper distributions
# --------------------------------------------------------------------------- #
def _make_wishart(df: jnp.ndarray, scale_tril: jnp.ndarray):
    """Wishart distribution parameterised by its Cholesky factor."""
    return tfd.WishartTriL(
        df=df,
        scale_tril=scale_tril,
        input_output_cholesky=True,
        validate_args=False,
    )
 
 
def _make_mvn_precision(loc: jnp.ndarray, precision_factor: jnp.ndarray):
    """Multivariate Normal parameterised by its precision (Cholesky) factor."""
    return tfde.MultivariateNormalPrecisionFactorLinearOperator(
        loc=loc,
        precision_factor=tf.linalg.LinearOperatorLowerTriangular(precision_factor),
        validate_args=False,
    )


# --------------------------------------------------------------------------- #
# Main model builder
# --------------------------------------------------------------------------- #
def build_mixture_hbmnl_model(
        data_dict: dict,
        K: int,                      # ── REQUIRED: number of model components (K_MODEL)
        A_delta: float = 0.01,
        a_mu: float = 0.01,
        dirichlet_a: float = 5.0):
    """
    Build a K-component mixture HBMNL Liesel model.
 
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
                  K_MODEL > K_TRUE; == 1.0 is uniform; > 1.0 pulls toward equal.
 
    Returns
    -------
    A compiled liesel.model.Model.
    """
    if K is None:
        raise ValueError(
            "K (number of model components) must be supplied explicitly. "
            "It is decoupled from data_dict['K'] by design."
        )
 
    n_params = int(data_dict["n_params"])
    n_units  = int(data_dict["n_units"])
    K_comp   = int(K)                                   # K_MODEL - never read from data, needs explicit user definition
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

    # ── beta_i location: Z[i] @ Delta + mu_k   (n_units, K, n_params) ───────
    if has_Z:
        beta_loc = lsl.Var.new_calc(
            lambda zd, mu: zd[:, None, :] + mu[None, :, :],
            zd=z_delta, mu=mu_k,
            name="beta_loc",
        )
    else:
        beta_loc = lsl.Var.new_calc(
            lambda mu: jnp.broadcast_to(mu[None, :, :], (n_units, K_comp, n_params)),
            mu=mu_k,
            name="beta_loc",
        )
 
    def _make_beta_mixture(pvec, locs, precision_factors):
        return tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=pvec),
            components_distribution=tfde.MultivariateNormalPrecisionFactorLinearOperator(
                loc=locs,
                precision_factor=tf.linalg.LinearOperatorLowerTriangular(
                    precision_factors[None]
                ),
            ),
        )
 
    beta_i = lsl.Var.new_param(
        value=jnp.zeros((n_units, n_params)),
        distribution=lsl.Dist(
            _make_beta_mixture,
            pvec=pvec,
            locs=beta_loc,
            precision_factors=sigma_inv_chol_k,
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
 
    return lsl.Model([y_var])