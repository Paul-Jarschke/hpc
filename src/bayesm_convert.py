"""
Convert bayesm raw posterior draws into the canonical posterior_raw.pkl format —
the SAME keys/shapes the Liesel runners produce, so the identical analysis /
label_switching / marginal_comparison / post_process code works unchanged.

Vendored from the study's run_single_bayesm_experiment.py (HierarchicalBayesianMNL):
the R sampler (src/bayesm_sampler.R) writes float64 column-major .bin draws + a
dims.json describing shapes/axis order; here we read them, rebuild the TFP
FillScaleTriL latent from the exported precision, and assemble the canonical dict:

    mu_k                     (chains, draws, K, P)
    sigma_inv_chol_k_latent  (chains, draws, K, P(P+1)/2)   - TFP FillScaleTriL
    pvec                     (chains, draws, K)             - simplex (NOT pvec_latent)
    beta_i                   (chains, draws, N, P)
    Delta                    (chains, draws, D, P)          - only if demographics

The crucial detail: bayesm samples Sigma_k directly (Gibbs); R exports the PRECISION
Sigma^{-1} = rooti @ rooti.T (bayesm's own definition). Here we take its lower
Cholesky L (L L.T = precision) and map it through FillScaleTriL().inverse — the exact
representation the Liesel model stores, so analysis._sigma_from_latent round-trips it
back to Sigma. (bayesm emits 'pvec' as a simplex, never 'pvec_latent'; analysis._recover_pvec
prefers 'pvec' when present.)
"""

import json
import pathlib

import numpy as np


def _read_bin(path, shape):
    """Read an R column-major float64 dump and reshape with Fortran order."""
    arr = np.fromfile(path, dtype="<f8")
    return arr.reshape(tuple(shape), order="F")


def precision_to_latent(prec):
    """prec: (..., P, P) precision matrices (Sigma^{-1}) -> TFP FillScaleTriL latent
    (..., P(P+1)/2) such that analysis._sigma_from_latent() round-trips back to Sigma."""
    import jax.numpy as jnp
    import tensorflow_probability.substrates.jax.bijectors as tfb

    # Symmetrise to kill tiny asymmetry, then lower-Cholesky: L L^T = precision.
    prec = 0.5 * (prec + np.swapaxes(prec, -1, -2))
    L = np.linalg.cholesky(prec)                       # lower-triangular, positive diag
    return np.asarray(tfb.FillScaleTriL().inverse(jnp.asarray(L)))


def _as_list(x):
    """jsonlite auto-unboxes length-1 vectors to scalars; force list form."""
    if x is None:
        return []
    return list(x) if isinstance(x, (list, tuple)) else [x]


def read_raw_draws(raw_dir):
    """Read <raw_dir>/{dims.json, meta_r.json, *_chain*.bin} written by the R sampler
    and return (canonical_posterior_dict, meta_r, dims)."""
    raw_dir = pathlib.Path(raw_dir)
    with open(raw_dir / "dims.json") as f:
        dims = json.load(f)
    with open(raw_dir / "meta_r.json") as f:
        meta_r = json.load(f)
    for k in ("seeds", "durations_s", "loglike_mean"):
        meta_r[k] = _as_list(meta_r.get(k))

    C = int(dims["chains"])
    pc = dims["per_chain"]
    has_Z = bool(dims["has_Z"])

    def stack(name):
        shape = pc[name]
        return np.stack(
            [_read_bin(raw_dir / f"{name}_chain{c}.bin", shape) for c in range(C)], axis=0
        )

    prec = stack("prec")                                   # (C, S, K, P, P)
    sigma_inv_chol_k_latent = precision_to_latent(prec)    # (C, S, K, P(P+1)/2)
    if not np.all(np.isfinite(sigma_inv_chol_k_latent)):
        raise FloatingPointError(
            "Non-finite values in reconstructed sigma_inv_chol_k_latent "
            "(precision -> Cholesky -> FillScaleTriL inverse) — likely a degenerate / "
            "empty mixture component in the bayesm draws."
        )

    canonical = {
        "mu_k": stack("mu"),                               # (C, S, K, P)
        "sigma_inv_chol_k_latent": sigma_inv_chol_k_latent,
        "pvec": stack("pvec"),                             # (C, S, K) simplex
        "beta_i": stack("beta"),                           # (C, S, N, P)
    }
    if has_Z:
        canonical["Delta"] = stack("delta")                # (C, S, D, P)

    return {k: np.asarray(v) for k, v in canonical.items()}, meta_r, dims
