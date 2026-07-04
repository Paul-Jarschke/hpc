"""
NUTS inference runner for the mixture HBMNL model.

K (the number of model components, K_MODEL) is passed in explicitly so the
runner never reports a stale value read from data_dict["K"].
"""

import liesel.goose as gs

from src.inference.init import build_bayesm_initial_states, set_multichain_initial_values


def run_nuts_inference_mixture_hbmnl(
        model,
        data_dict: dict,
        K: int,                      # ── REQUIRED: K_MODEL, for correct reporting
        chains: int = 1,
        warmup: int = 1000,
        posterior: int = 5000,
        seed: int = 123,
        use_informed_init: bool = True):
    """
    Configure and run the Liesel/Goose NUTS engine for the mixture HBMNL model.

    The block structure samples pvec, the component covariances, the component
    means, the (optional) demographic shift Delta, and the unit-level betas in
    separate NUTS kernels. pvec_latent and sigma_inv_chol_k_latent live in very
    different geometries, so they are split into their own kernels rather than
    blocked together.

    Parameters
    ----------
    model             : compiled liesel Model from build_mixture_hbmnl_model.
    data_dict         : data dictionary (used only to detect presence of Z).
    K                 : number of model components (K_MODEL), for logging only.
    chains            : number of MCMC chains.
    warmup            : warmup iterations.
    posterior         : posterior iterations.
    seed              : RNG seed.
    use_informed_init : if True (default), initialise exactly as Rossi/bayesm's
                         own rhierMnlRwMixture does (fractional-MLE beta_i,
                         equal-block ind, uniform pvec, zero Delta, mu_k/Sigma_k
                         from Rossi's own first-Gibbs-draw formula) instead of
                         the model's naive (0, I, uniform) default. See
                         src.inference.init for the exact correspondence.

    Returns
    -------
    (results, posterior_samples) from the Goose engine.
    """
    eb = gs.EngineBuilder(seed=seed, num_chains=chains)
    eb.set_model(gs.LieselInterface(model))
    if use_informed_init:
        set_multichain_initial_values(
            eb, build_bayesm_initial_states(model, data_dict, K, chains, seed)
        )
    else:
        eb.set_initial_values(model.state)

    has_Z = data_dict.get("Z") is not None

    # Component weights (simplex geometry, diagonal mass matrix)
    eb.add_kernel(gs.NUTSKernel(["pvec_latent"], mm_diag=True))

    # Component covariances (Cholesky-of-precision latent space). Dense mass
    # matrix: this block's failure mode is divergences (not max-tree-depth),
    # the textbook signature of a diagonal metric mismatched to correlated
    # Cholesky-entry geometry. Cheap here (K*P*(P+1)/2 = 50 dims at K=5,P=4).
    eb.add_kernel(gs.NUTSKernel(["sigma_inv_chol_k_latent"], mm_diag=False))

    # Component means
    eb.add_kernel(gs.NUTSKernel(["mu_k"]))

    # Global demographic covariates (only if Z present)
    if has_Z:
        eb.add_kernel(gs.NUTSKernel(["Delta"]))

    # Unit-level coefficients
    eb.add_kernel(gs.NUTSKernel(["beta_i"]))

    eb.set_duration(warmup_duration=warmup, posterior_duration=posterior)

    print("Starting NUTS sampling for mixture HBMNL...")
    print(f" - Demographic covariates (Delta) included : {has_Z}")
    print(f" - Model components (K_MODEL)              : {K}")
    print(f" - Chains: {chains} | Warmup: {warmup} | Posterior: {posterior}")

    engine = eb.build()
    engine.sample_all_epochs()

    results = engine.get_results()
    return results, results.get_posterior_samples()