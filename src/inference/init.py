"""
Initial values for the NUTS/HMC arms (src.mixturemodel's marginalised model)
-- built to be EXACTLY Rossi/bayesm's own initialisation scheme
(rhierMnlRwMixture.R), not an invented alternative. No k-means, no per-chain
jitter/shuffle, no ad hoc regularisation:

    beta_i  <- per-unit fractional-likelihood MLE      (bayesm's oldbetas)
    ind     <- contiguous equal-size blocks by index    (bayesm's ind)
    pvec    <- uniform 1/K                              (bayesm's oldprob)
    Delta   <- zero                                     (bayesm's olddelta;
                                                          already the model's
                                                          own default, left
                                                          untouched here)

beta_i/ind/pvec/Delta are all DETERMINISTIC and IDENTICAL across chains,
exactly matching bayesm's own multi-chain convention (run_single_bayesm_
experiment.R computes oldbetas/ind/oldprob once, shared by every seeded
chain -- only the RNG stream differs downstream).

Rossi's algorithm never separately initialises mu_k/Sigma_k: they are the
OUTPUT of the very first Gibbs draw, conditional on (ind, beta_i). NUTS/HMC
have no Gibbs step, so some starting value is structurally required where
Rossi's algorithm has none. The faithful resolution is to reproduce that
exact first draw once per chain, using each chain's own seed -- precisely
what bayesm's own per-chain RNG stream would produce for iteration 1's
mu_k/Sigma_k under set.seed(cseed). This calls _niw_conjugate_draw from
bayesm_gibbs.py verbatim: the identical formula the Gibbs arm itself runs
every sweep, not a re-derived approximation.
"""

import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.bijectors as tfb
import liesel.goose as gs
from liesel.goose.pytree import stack_leaves
from liesel.option import Option

from src.bayesm_mixture_model import bayesm_initial_indicators
from src.inference.bayesm_gibbs import (
    _prepare_unit_data, _fractional_candidate_prep, _niw_conjugate_draw,
)

_SOFTMAX_C = tfb.SoftmaxCentered()
_FILL_TRIL = tfb.FillScaleTriL()


def set_multichain_initial_values(eb, stacked_state):
    """
    Set a per-chain-distinct initial state on an EngineBuilder.

    EngineBuilder.set_initial_values(state, multiple_chains=True) is not usable
    for this in the installed Goose (liesel==0.4.1): that code path never
    assigns self._model_state. This sets it directly via the same Option
    wrapper Goose uses internally (liesel/goose/builder.py).
    """
    eb._model_state = Option(stacked_state)


def build_bayesm_initial_states(model, data_dict, K, chains, seed, w=0.1):
    """
    Build a stacked initial state for eb.set_initial_values(state,
    multiple_chains=True), following Rossi/bayesm's own initialisation
    exactly (see module docstring).

    Reads (nu, V, a_mu) from model.prior_hparams so this can never silently
    use hyperparameters other than the ones the model was actually built
    with -- model must come from src.mixturemodel.build_mixture_hbmnl_model.

    Parameters
    ----------
    model     : compiled liesel Model from build_mixture_hbmnl_model.
    data_dict : data dictionary with X, y, unit_idx, n_units, n_params, opt. Z.
    K         : number of model components (K_MODEL).
    chains    : number of MCMC chains.
    seed      : base RNG seed; chain c's mu_k/Sigma_k draw uses seed + c.
    w         : fractional-likelihood weight for the MLE fit (bayesm default).

    Returns
    -------
    Stacked model state, shape-compatible with eb.set_initial_values(...,
    multiple_chains=True).
    """
    prior = getattr(model, "prior_hparams", None)
    if prior is None:
        raise ValueError(
            "model has no 'prior_hparams' attribute - this initialisation "
            "requires the marginalised model from src.mixturemodel."
            "build_mixture_hbmnl_model."
        )

    n_units  = int(data_dict["n_units"])
    n_params = int(data_dict["n_params"])
    K_comp   = int(prior["K"])
    a_mu     = float(prior["a_mu"])
    nu       = float(prior["nu"])
    V        = jnp.asarray(prior["V"])
    eye_P    = jnp.eye(n_params)

    # beta_i <- fractional MLE (bayesm's oldbetas); deterministic, shared
    # across chains, identical to bayesm_gibbs.py's own beta_i init.
    X_units, y_units, mask = _prepare_unit_data(data_dict)
    _, betafmle, _, _ = _fractional_candidate_prep(X_units, y_units, mask, w)

    # ind <- contiguous equal-size blocks (bayesm's ind); pvec <- uniform
    # (bayesm's oldprob). Both deterministic, shared across chains.
    ind0  = jnp.asarray(bayesm_initial_indicators(n_units, K_comp))
    pvec0 = jnp.ones(K_comp) / K_comp

    interface = gs.LieselInterface(model)
    states = []
    for c in range(chains):
        # mu_k/Sigma_k <- Rossi's own first Gibbs draw, conditional on
        # (ind0, betafmle), using this chain's own seed -- exactly what
        # bayesm's per-chain RNG stream would produce for iteration 1.
        comps = _niw_conjugate_draw(
            jax.random.PRNGKey(seed + c), betafmle, ind0,
            K_comp, n_params, a_mu, nu, V, eye_P, _FILL_TRIL,
        )
        position = {
            "beta_i": betafmle,
            "pvec_latent": _SOFTMAX_C.inverse(pvec0),
            "mu_k": comps["mu_k"],
            "sigma_inv_chol_k_latent": comps["sigma_inv_chol_k_latent"],
        }
        states.append(interface.update_state(position, model.state))

    return stack_leaves(states)
