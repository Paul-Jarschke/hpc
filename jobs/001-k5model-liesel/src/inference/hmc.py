"""
HMC inference runner for the mixture HBMNL model.

Mirror of nuts.py, but every block is sampled with a fixed-length HMC kernel
instead of NUTS. The number of leapfrog integration steps is fixed (default 10)
rather than chosen dynamically per-step as NUTS does.

K (the number of model components, K_MODEL) is passed in explicitly so the
runner never reports a stale value read from data_dict["K"].
"""

import liesel.goose as gs


def run_hmc_inference_mixture_hbmnl(
        model,
        data_dict: dict,
        K: int,                          # ── REQUIRED: K_MODEL, for correct reporting
        num_integration_steps: int = 10,  # ── leapfrog steps per HMC proposal
        chains: int = 1,
        warmup: int = 1000,
        posterior: int = 5000,
        seed: int = 123):
    """
    Configure and run the Liesel/Goose HMC engine for the mixture HBMNL model.

    The block structure samples pvec, the component covariances, the component
    means, the (optional) demographic shift Delta, and the unit-level betas in
    separate HMC kernels. pvec_latent and sigma_inv_chol_k_latent live in very
    different geometries, so they are split into their own kernels rather than
    blocked together.

    Parameters
    ----------
    model                  : compiled liesel Model from build_mixture_hbmnl_model.
    data_dict              : data dictionary (used only to detect presence of Z).
    K                      : number of model components (K_MODEL), for logging.
    num_integration_steps  : leapfrog steps per HMC proposal (fixed trajectory length).
    chains                 : number of MCMC chains.
    warmup                 : warmup iterations.
    posterior              : posterior iterations.
    seed                   : RNG seed.

    Returns
    -------
    (results, posterior_samples) from the Goose engine.
    """
    eb = gs.EngineBuilder(seed=seed, num_chains=chains)
    eb.set_model(gs.LieselInterface(model))
    eb.set_initial_values(model.state)

    has_Z = data_dict.get("Z") is not None

    # Component weights (simplex geometry, diagonal mass matrix)
    eb.add_kernel(gs.HMCKernel(
        ["pvec_latent"],
        num_integration_steps=num_integration_steps,
        mm_diag=True,
    ))

    # Component covariances (Cholesky-of-precision latent space)
    eb.add_kernel(gs.HMCKernel(
        ["sigma_inv_chol_k_latent"],
        num_integration_steps=num_integration_steps,
    ))

    # Component means
    eb.add_kernel(gs.HMCKernel(
        ["mu_k"],
        num_integration_steps=num_integration_steps,
    ))

    # Global demographic covariates (only if Z present)
    if has_Z:
        eb.add_kernel(gs.HMCKernel(
            ["Delta"],
            num_integration_steps=num_integration_steps,
        ))

    # Unit-level coefficients
    eb.add_kernel(gs.HMCKernel(
        ["beta_i"],
        num_integration_steps=num_integration_steps,
    ))

    eb.set_duration(warmup_duration=warmup, posterior_duration=posterior)

    print("Starting HMC sampling for mixture HBMNL...")
    print(f" - Demographic covariates (Delta) included : {has_Z}")
    print(f" - Model components (K_MODEL)              : {K}")
    print(f" - Integration steps per proposal          : {num_integration_steps}")
    print(f" - Chains: {chains} | Warmup: {warmup} | Posterior: {posterior}")

    engine = eb.build()
    engine.sample_all_epochs()

    results = engine.get_results()
    return results, results.get_posterior_samples()