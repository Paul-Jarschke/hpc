import numpy as np
import os
import json

import jax.numpy as jnp


def generate_mixture_simulated_data(
        n_units=2000, n_obs=100, n_alts=4, n_components=2, n_params=None,
        n_demos=2, custom_pvec=None, custom_indicators=None, seed=123):
    """
    Simulate data for a Bayesian Hierarchical Multinomial Logit model with a mixture-of-normals heterogeneity distribution.

    Follows Rossi (2006) §5.5 specification:
      - Z is centred; no intercept column
      - mu_k  ~ N(0, I / A_MU),  A_MU = 1/16 for standardised X
      - Sigma_k ~ Diagonal with variances ~ Uniform(0.5, 2.0)
      - Continuous X attributes are standardised globally before choice simulation

    Parameters
    ----------
    n_units             : number of decision-making units (households / individuals)
    n_obs               : number of choice occasions per unit
    n_alts              : number of alternatives per choice occasion
    n_components        : number of mixture components K
    n_params            : total number of beta parameters (defaults to n_alts)
    n_demos             : number of demographic covariates in Z
    custom_pvec         : optional fixed mixture weights; will be normalised
    custom_indicators   : optional fixed component assignments per unit
    seed                : numpy random seed for reproducibility

    Returns
    -------
    dict with arrays X, y, Z, unit_idx and all TRUE_ ground-truth parameters
    """

    np.random.seed(seed)

    if n_params is None:
        n_params = n_alts
    if n_params < n_alts - 1:
        raise ValueError(f"n_params ({n_params}) must be at least n_alts - 1")

    n_ascs       = n_alts - 1
    n_continuous = n_params - n_ascs

    # ------------------------------------------------------------------
    # Demographics — column-wise centred, no intercept  (Rossi §5.5)
    # ------------------------------------------------------------------
    Z  = np.random.normal(0, 1, size=(n_units, n_demos))
    Z -= Z.mean(axis=0)

    Delta_true = np.random.normal(0, 0.5, size=(n_demos, n_params))

    # Mixture weights
    if custom_pvec is not None:
        true_pvec  = np.array(custom_pvec, dtype=float)
        true_pvec /= true_pvec.sum()
    else:
        raw_p     = np.random.uniform(0.5, 2.0, n_components)
        true_pvec = raw_p / raw_p.sum()

    # ------------------------------------------------------------------
    # Component parameters
    #   mu_k    ~ N(0, I / A_MU)              — prior-consistent, A_MU = 1/16  (§5.5)
    #   Sigma_k ~ Diagonal, variances ~ U(0.5, 2.0)
    # ------------------------------------------------------------------
    A_MU = 1.0 / 16.0                   # Rossi: a_mu = 1/16 → SD = 4

    true_mu_k    = np.zeros((n_components, n_params))
    true_Sigma_k = np.zeros((n_components, n_params, n_params))

    for k in range(n_components):
        true_Sigma_k[k] = np.diag(np.random.uniform(0.5, 2.0, n_params))
        true_mu_k[k]    = np.random.normal(0.0, np.sqrt(1.0 / A_MU), size=n_params)

    # ------------------------------------------------------------------
    # Individual-level betas
    # ------------------------------------------------------------------
    if custom_indicators is not None:
        true_indicators = np.array(custom_indicators)
    else:
        true_indicators = np.random.choice(n_components, size=n_units, p=true_pvec)

    beta_true = np.zeros((n_units, n_params))
    for i in range(n_units):
        k            = true_indicators[i]
        mu_i         = Z[i] @ Delta_true + true_mu_k[k]
        beta_true[i] = np.random.multivariate_normal(mu_i, true_Sigma_k[k])

    # ------------------------------------------------------------------
    # Design matrix X
    #   Pass 1 — fill X_array (ASC block fixed; continuous block random)
    #   Standardise — per continuous attribute across all obs × alts
    #   Pass 2 — simulate choices on standardised X
    # ------------------------------------------------------------------
    n_total = n_units * n_obs
    X_array = np.zeros((n_total, n_alts, n_params))

    flat_idx = 0
    for i in range(n_units):
        for t in range(n_obs):
            X_it = np.zeros((n_alts, n_params))
            for a in range(1, n_alts):          # alt 0 is reference
                X_it[a, a - 1] = 1.0
            if n_continuous > 0:
                X_it[:, n_ascs:] = np.random.uniform(
                    1.0, 5.0, size=(n_alts, n_continuous)
                )
            X_array[flat_idx] = X_it
            flat_idx += 1

    # Rossi §5.5: "standardise the X variables"
    if n_continuous > 0:
        for c in range(n_continuous):
            col   = X_array[:, :, n_ascs + c]
            mu_c  = col.mean()
            std_c = col.std() + 1e-8
            X_array[:, :, n_ascs + c] = (col - mu_c) / std_c

    X_list, y_list, unit_idx_list = [], [], []

    flat_idx = 0
    for i in range(n_units):
        for t in range(n_obs):
            X_it  = X_array[flat_idx]
            U_it  = X_it @ beta_true[i]
            exp_U = np.exp(U_it - U_it.max())
            probs = exp_U / exp_U.sum()
            y_it  = int(np.random.choice(n_alts, p=probs))
            X_list.append(X_it)
            y_list.append(y_it)
            unit_idx_list.append(i)
            flat_idx += 1

    asc_names  = [f"Alt{a}" for a in range(1, n_ascs + 1)]
    cont_names = ["Price"] if n_continuous == 1 else [f"X{c + 1}" for c in range(n_continuous)]
    param_names = asc_names + cont_names
    demo_names  = [f"z{d + 1}" for d in range(n_demos)]

    return {
        "X":               jnp.array(X_list),
        "y":               jnp.array(y_list),
        "Z":               jnp.array(Z),
        "unit_idx":        jnp.array(unit_idx_list),
        "n_units":         n_units,
        "n_params":        n_params,
        "n_demos":         n_demos,
        "K":               n_components,
        "n_alts":          n_alts,
        "param_names":     param_names,
        "demo_names":      demo_names,
        "TRUE_DELTA":      Delta_true,
        "TRUE_BETA":       beta_true,
        "TRUE_PVEC":       true_pvec,
        "TRUE_MU_K":       true_mu_k,
        "TRUE_SIGMA_K":    true_Sigma_k,
        "TRUE_INDICATORS": true_indicators,
        "DGP_A_MU":        float(A_MU),
    }


def save_to_json(data, filename="sim_data.json"):
    """Serialise all arrays to lists and write to a JSON file."""

    def convert_recursive(obj):
        if isinstance(obj, (np.ndarray, jnp.ndarray)):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert_recursive(v) for k, v in obj.items()}
        if isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
            return obj.item()
        return obj

    serializable_data = convert_recursive(data)

    dir_name = os.path.dirname(filename)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    with open(filename, "w") as f:
        json.dump(serializable_data, f, indent=4)

    print(f"Saved to {os.path.abspath(filename)}")