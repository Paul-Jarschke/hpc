"""
Bayesian Hierarchical Multinomial Logit (HBMNL) with a SINGLE multivariate
normal heterogeneity distribution (no mixture) — Rossi (2006) §5.4:

    beta_i = mu + Z[i] @ Delta + u_i,   u_i ~ N(0, Sigma)

Prior structure — EXACTLY the model bayesm::rhierMnlRwMixture fits with
ncomp = 1:

    Wishart:  Sigma^{-1}  ~ W(nu, V^{-1}),  nu = n_params + 3,  V = nu * I
    Normal:   mu | Sigma  ~ N(0, Sigma / a_mu)          (bayesm's Amu)
    Normal:   Delta       ~ N(0, (1/A_delta) * I)       (bayesm's Ad)

Z is centred with no intercept column (the repo-wide convention); mu carries
the population mean. There is one component, so no parameter carries a
component axis or a _k suffix: mu (P,), sigma_inv_chol (P, P) with latent
sigma_inv_chol_latent, Delta (n_demos, P), beta_i (n_units, P).

Sampling functions for NUTS and fixed-step HMC live in this module too. The
Cholesky-of-precision block uses a dense mass matrix (its latent entries are
correlated; a diagonal metric produces divergences on this block).
"""

import jax.numpy as jnp

import liesel.model as lsl
import liesel.goose as gs
import tensorflow_probability.substrates.jax.bijectors as tfb
import tensorflow_probability.substrates.jax.distributions as tfd

from src.mixturemodel import _make_mvn_precision, _make_wishart


# --------------------------------------------------------------------------- #
# Model builder
# --------------------------------------------------------------------------- #
def build_standard_hbmnl_model(
        data_dict: dict,
        A_delta: float = 0.01,
        a_mu: float = 0.01):
    """
    Build the one-component HBMNL Liesel model (Rossi §5.4; bayesm ncomp = 1).

    Parameters
    ----------
    data_dict : dict with 'X', 'y', 'unit_idx', 'n_params', 'n_units' and
                optionally 'Z' (centred, no intercept column).
    A_delta   : prior precision scaling for Delta (bayesm's Ad = A_delta * I).
    a_mu      : prior precision scaling for the population mean
                (mu | Sigma ~ N(0, Sigma / a_mu); bayesm's Amu).

    Returns
    -------
    A compiled liesel.model.Model with an attribute ``prior_hparams`` holding
    the hyperparameters actually built into the model.
    """
    n_params = int(data_dict["n_params"])
    n_units  = int(data_dict["n_units"])
    has_Z    = data_dict.get("Z") is not None

    # ── Wishart prior on the precision matrix ──────────────────────────────
    nu        = float(n_params + 3)
    V         = nu * jnp.eye(n_params)
    Vinv_chol = jnp.linalg.cholesky(jnp.linalg.inv(V))

    sigma_inv_chol = lsl.Var.new_param(
        value=jnp.eye(n_params),
        distribution=lsl.Dist(
            _make_wishart,
            df=nu,
            scale_tril=Vinv_chol,
        ),
        name="sigma_inv_chol",
    )
    sigma_inv_chol_latent = sigma_inv_chol.transform(
        tfb.FillScaleTriL(), name="sigma_inv_chol_latent"
    )

    # ── mu | Sigma ~ N(0, Sigma / a_mu) ────────────────────────────────────
    mu_prec_factor = lsl.Var.new_calc(
        lambda L: jnp.sqrt(a_mu) * L,
        L=sigma_inv_chol,
        name="mu_prec_factor",
    )
    mu = lsl.Var.new_param(
        value=jnp.zeros(n_params),
        distribution=lsl.Dist(
            _make_mvn_precision,
            loc=jnp.zeros(n_params),
            precision_factor=mu_prec_factor,
        ),
        name="mu",
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
        beta_loc = lsl.Var.new_calc(
            lambda zd, m: zd + m[None, :],
            zd=z_delta, m=mu,
            name="beta_loc",
        )
    else:
        beta_loc = lsl.Var.new_calc(
            lambda m: jnp.broadcast_to(m[None, :], (n_units, n_params)),
            m=mu,
            name="beta_loc",
        )

    # ── Unit-level coefficients ────────────────────────────────────────────
    beta_i = lsl.Var.new_param(
        value=jnp.zeros((n_units, n_params)),
        distribution=lsl.Dist(
            _make_mvn_precision,
            loc=beta_loc,
            precision_factor=sigma_inv_chol,
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
    model.prior_hparams = {
        "nu": nu,
        "a_mu": float(a_mu),
        "A_delta": float(A_delta),
        "has_Z": has_Z,
    }
    return model


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def run_nuts_inference_standard_hbmnl(
        model,
        chains: int = 2,
        warmup: int = 1000,
        posterior: int = 5000,
        seed: int = 123):
    """
    Configure and run the Liesel/Goose NUTS engine for the standard HBMNL model.

    Kernel blocks: sigma_inv_chol_latent (dense mass matrix), mu, Delta (if Z
    present), beta_i.

    Returns
    -------
    (results, posterior_samples) from the Goose engine.
    """
    eb = gs.EngineBuilder(seed=seed, num_chains=chains)
    eb.set_model(gs.LieselInterface(model))
    eb.set_initial_values(model.state)

    has_Z = model.prior_hparams["has_Z"]

    eb.add_kernel(gs.NUTSKernel(["sigma_inv_chol_latent"], mm_diag=False))
    eb.add_kernel(gs.NUTSKernel(["mu"]))
    if has_Z:
        eb.add_kernel(gs.NUTSKernel(["Delta"]))
    eb.add_kernel(gs.NUTSKernel(["beta_i"]))

    eb.set_duration(warmup_duration=warmup, posterior_duration=posterior)

    print("Starting NUTS sampling for the standard (one-component) HBMNL...")
    print(f" - Demographic covariates (Delta) included : {has_Z}")
    print(f" - Chains: {chains} | Warmup: {warmup} | Posterior: {posterior}")

    engine = eb.build()
    engine.sample_all_epochs()

    results = engine.get_results()
    return results, results.get_posterior_samples()


def run_hmc_inference_standard_hbmnl(
        model,
        num_integration_steps: int = 10,  # ── leapfrog steps per HMC proposal
        chains: int = 2,
        warmup: int = 1000,
        posterior: int = 5000,
        seed: int = 123):
    """
    Configure and run the Liesel/Goose HMC engine for the standard HBMNL model.

    Mirror of run_nuts_inference_standard_hbmnl with a fixed trajectory length
    (default 10 leapfrog steps) instead of NUTS's dynamic one.

    Returns
    -------
    (results, posterior_samples) from the Goose engine.
    """
    eb = gs.EngineBuilder(seed=seed, num_chains=chains)
    eb.set_model(gs.LieselInterface(model))
    eb.set_initial_values(model.state)

    has_Z = model.prior_hparams["has_Z"]

    eb.add_kernel(gs.HMCKernel(
        ["sigma_inv_chol_latent"],
        num_integration_steps=num_integration_steps,
        mm_diag=False,
    ))
    eb.add_kernel(gs.HMCKernel(
        ["mu"],
        num_integration_steps=num_integration_steps,
    ))
    if has_Z:
        eb.add_kernel(gs.HMCKernel(
            ["Delta"],
            num_integration_steps=num_integration_steps,
        ))
    eb.add_kernel(gs.HMCKernel(
        ["beta_i"],
        num_integration_steps=num_integration_steps,
    ))

    eb.set_duration(warmup_duration=warmup, posterior_duration=posterior)

    print("Starting HMC sampling for the standard (one-component) HBMNL...")
    print(f" - Demographic covariates (Delta) included : {has_Z}")
    print(f" - Integration steps per proposal          : {num_integration_steps}")
    print(f" - Chains: {chains} | Warmup: {warmup} | Posterior: {posterior}")

    engine = eb.build()
    engine.sample_all_epochs()

    results = engine.get_results()
    return results, results.get_posterior_samples()
