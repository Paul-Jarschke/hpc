"""
Diagnostics and parameter-recovery utilities for the mixture HBMNL model.

Convention used throughout
--------------------------
    K       — number of MODEL components (K_MODEL).
              Drives every loop over posterior draws.
              Posterior arrays always have this many components.
    K_true  — number of TRUE components in the data-generating process.
              Ground-truth arrays (true_mu, true_pvec, true_sigma) have only K_true entries.
              When K_MODEL > K_true the extra model components are "spurious":
              They have no true counterpart, so truth overlays are skipped for them rather than indexing out of bounds.

Functions that overlay ground truth (summarize_mu_k, plot_pvec_diagnostics,
summarize_pvec) therefore take an optional `K_true` argument. If it is omitted
they fall back to assuming K == K_true (the correctly-specified case).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from matplotlib.lines import Line2D
from scipy.optimize import linear_sum_assignment
import tensorflow_probability.substrates.jax.bijectors as tfb
from IPython.display import display
import arviz as az


# --------------------------------------------------------------------------- #
# 1. MCMC diagnostics
# --------------------------------------------------------------------------- #
def plot_cholesky_traces(samples_dict, n_params, k_idx=0,
                         param_name="sigma_inv_chol_k_latent", figsize=(15, 12)):
    """Trace plots of the latent Cholesky entries for one model component."""
    latent_samples = samples_dict[param_name][:, :, k_idx, :]
    n_chains, n_draws, n_latent = latent_samples.shape

    fig, axes = plt.subplots(n_params, n_params, figsize=figsize,
                             sharex=True, sharey=False)
    if n_params == 1:
        axes = np.array([[axes]])

    latent_idx = 0
    for i in range(n_params):
        for j in range(n_params):
            ax = axes[i, j]
            if i >= j and latent_idx < n_latent:
                for chain in range(n_chains):
                    ax.plot(latent_samples[chain, :, latent_idx], label=f"Chain {chain}")
                ax.set_title(f"Latent L[{i},{j}]", fontsize=10)
                latent_idx += 1
            else:
                ax.axis("off")
            ax.grid(True)
            if j == 0:
                ax.set_ylabel("Value")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=n_chains,
               bbox_to_anchor=(0.5, 0.02))
    plt.suptitle(f"Cholesky Factor of Precision Matrix - Component {k_idx + 1}", fontsize=20)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()


def compute_acf(x, nlags=30):
    x_centered = x - np.mean(x)
    norm = np.sum(x_centered ** 2)
    if norm == 0:
        return np.zeros(nlags)
    acf = np.correlate(x_centered, x_centered, mode="full")
    return (acf[acf.size // 2:] / norm)[:nlags]


def plot_goose_style_diagnostics(delta_array, demo_idx, param_idx,
                                 demo_name, param_name, n_lags=30):
    n_chains = delta_array.shape[0]
    fig = plt.figure(figsize=(12, 7))
    fig.suptitle(f"Δ[{demo_name}, {param_name}]", fontsize=14, y=0.95)
    gs_layout = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1], hspace=0.3, wspace=0.2)

    ax_trace = fig.add_subplot(gs_layout[0, :])
    for chain in range(n_chains):
        ax_trace.plot(delta_array[chain, :, demo_idx, param_idx], label=f"{chain}")
    ax_trace.set_xlabel("Iteration")
    ax_trace.set_ylabel("Value")
    ax_trace.grid(True, alpha=1)
    ax_trace.legend(title="Chain", loc="center left",
                    bbox_to_anchor=(1.02, 0.5), frameon=False)

    ax_dens = fig.add_subplot(gs_layout[1, 0])
    all_draws = delta_array[:, :, demo_idx, param_idx].reshape(-1)
    ci_low, ci_high = np.percentile(all_draws, [2.5, 97.5])
    for chain in range(n_chains):
        sns.kdeplot(delta_array[chain, :, demo_idx, param_idx], ax=ax_dens, fill=False)
    ax_dens.axvline(ci_low, color="black", linestyle=":", lw=1.2, label="95% CI")
    ax_dens.axvline(ci_high, color="black", linestyle=":", lw=1.2)
    ax_dens.set_xlabel("Value")
    ax_dens.set_ylabel("Density")
    ax_dens.set_title(f"95% CI: [{ci_low:.3f}, {ci_high:.3f}]", fontsize=10)
    ax_dens.grid(True)

    ax_acf = fig.add_subplot(gs_layout[1, 1])
    for chain in range(n_chains):
        ax_acf.plot(compute_acf(delta_array[chain, :, demo_idx, param_idx], nlags=n_lags),
                    alpha=1)
    ax_acf.set_xlabel("Lag")
    ax_acf.set_ylabel("Autocorrelation")
    ax_acf.set_ylim(-0.1, 1.05)
    ax_acf.grid(True)
    plt.show()


# --------------------------------------------------------------------------- #
# 2. Covariance recovery
# --------------------------------------------------------------------------- #
def recover_covariance_matrices(latent_samples_sorted):
    """Map latent Cholesky-of-precision draws back to covariance matrices."""
    bijector_tril = tfb.FillScaleTriL()

    def latent_to_sigma(latent_vec):
        L = bijector_tril.forward(latent_vec)
        precision = L @ L.T
        return jnp.linalg.inv(precision)

    v_latent_to_sigma = jax.vmap(jax.vmap(jax.vmap(latent_to_sigma)))
    return v_latent_to_sigma(latent_samples_sorted)


def plot_final_covariance_complete(samples, true_matrix=None,
                                   empirical_matrix=None, component_idx=0):
    n_dim = samples.shape[-1]
    fig, axes = plt.subplots(n_dim, n_dim, figsize=(18, 16))
    if n_dim == 1:
        axes = np.array([[axes]])
    flattened_samples = samples.reshape(-1, n_dim, n_dim)

    diag_color, off_diag_color, true_val_color, emp_val_color = \
        "#002347", "#4682B4", "#D62728", "#2CA02C"

    for i in range(n_dim):
        for j in range(n_dim):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue

            data_vec = flattened_samples[:, i, j]
            current_color = diag_color if (i == j) else off_diag_color
            post_mean = np.mean(data_vec)
            ci_low, ci_high = np.percentile(data_vec, [2.5, 97.5])

            sns.kdeplot(data_vec, ax=ax, fill=True, color=current_color, alpha=0.25, lw=2.5)
            ax.axvline(post_mean, color=current_color, linestyle="-", lw=1.5, alpha=0.8)
            ax.axvline(ci_low, color=current_color, linestyle=":", lw=1.8, alpha=0.7)
            ax.axvline(ci_high, color=current_color, linestyle=":", lw=1.8, alpha=0.7)

            title_parts = [f"Mean: {post_mean:.2f}"]
            if true_matrix is not None:
                ax.axvline(true_matrix[i, j], color=true_val_color, linestyle="--", lw=2)
                title_parts.append(f"True: {true_matrix[i, j]:.2f}")
            if empirical_matrix is not None:
                ax.axvline(empirical_matrix[i, j], color=emp_val_color, linestyle="-.", lw=2)
                title_parts.append(f"Emp: {empirical_matrix[i, j]:.2f}")

            ax.set_title(f"{' | '.join(title_parts)}\n95% CI: [{ci_low:.2f}, {ci_high:.2f}]",
                         fontsize=10, pad=12, fontweight="bold")
            ax.set_yticks([])
            sns.despine(ax=ax, left=True)

            if i == n_dim - 1:
                ax.set_xlabel(f"Variable {j + 1}", fontsize=12, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"Variable {i + 1}", fontsize=12, fontweight="bold")

    legend_elements = [
        Line2D([0], [0], color=off_diag_color, linestyle="-", lw=2, label="Posterior Mean"),
        Line2D([0], [0], color=off_diag_color, linestyle=":", lw=2, label="95% CI"),
    ]
    if true_matrix is not None:
        legend_elements.append(
            Line2D([0], [0], color=true_val_color, linestyle="--", lw=2, label="True Value"))
    if empirical_matrix is not None:
        legend_elements.append(
            Line2D([0], [0], color=emp_val_color, linestyle="-.", lw=2,
                   label="Empirical Sub-sample"))

    fig.legend(handles=legend_elements, loc="upper right",
               bbox_to_anchor=(0.9, 0.9), fontsize=14, frameon=True)
    plt.suptitle(f"Posterior Σ_k - Component {component_idx}", fontsize=24, y=0.98)
    plt.subplots_adjust(hspace=0.6, wspace=0.2)
    plt.show()


# --------------------------------------------------------------------------- #
# 3. Global parameters (mu_k & Delta)
# --------------------------------------------------------------------------- #
def summarize_mu_k(mu_samples, K, P, param_names, true_mu=None, K_true=None):
    """
    Summarise posterior component means mu_k.

    K       : number of MODEL components (K_MODEL) — drives the loop.
    true_mu : ground-truth means, shape (K_true, P). May have fewer rows than K
              when the model is overspecified.
    K_true  : number of true components. Required to overlay ground truth; if
              None, assumes K_true == K (correctly-specified case).
    """
    if true_mu is not None and K_true is None:
        K_true = K  # correctly-specified fallback

    mu_flat = mu_samples.reshape(-1, K, P)
    post_mu_mean = mu_flat.mean(axis=0)            # (K, P)

    # Match model components to true components via rectangular assignment.
    # mapping[k] = matched true-component index, or None for spurious comps.
    mapping = {k: None for k in range(K)}
    if true_mu is not None:
        cost = np.sum(
            (post_mu_mean[:, None, :] - true_mu[None, :K_true, :]) ** 2, axis=-1
        )                                          # (K, K_true)
        row_ind, col_ind = linear_sum_assignment(cost)
        for r, c in zip(row_ind, col_ind):
            mapping[int(r)] = int(c)

    print("\n=== Component Means (mu_k) Summary Tables ===")
    for k in range(K):
        true_k = mapping[k]
        if true_k is not None:
            header = f"--- MCMC Component {k} (matched to True Component {true_k}) ---"
        elif true_mu is not None:
            header = f"--- MCMC Component {k} ---"
        else:
            header = f"--- MCMC Component {k} ---"
        print(f"\n{header}")

        df = pd.DataFrame({
            "Parameter":      param_names,
            "Posterior_Mean": mu_flat[:, k, :].mean(axis=0),
            "Posterior_Std":  mu_flat[:, k, :].std(axis=0),
        })
        if true_k is not None:
            df.insert(1, "True_Value", true_mu[true_k])
            df["Diff_Abs"] = np.abs(true_mu[true_k] - df["Posterior_Mean"])
        display(df.round(4).set_index("Parameter"))


def plot_mu_k_diagnostics(mu_samples, K, P, param_names, n_lags=30):
    """Goose-style diagnostics (trace, distribution, ACF) for each mu_k[k, p]."""
    n_chains = mu_samples.shape[0]
    for k in range(K):
        for p in range(P):
            fig = plt.figure(figsize=(12, 7))
            fig.suptitle(f"mu_k[Component {k + 1}, {param_names[p]}]", fontsize=14, y=0.95)
            gs_layout = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1], hspace=0.3, wspace=0.2)

            ax_trace = fig.add_subplot(gs_layout[0, :])
            for chain in range(n_chains):
                ax_trace.plot(mu_samples[chain, :, k, p], label=f"{chain}")
            ax_trace.set_xlabel("Iteration")
            ax_trace.set_ylabel("Value")
            ax_trace.grid(True, alpha=1)
            ax_trace.legend(title="Chain", loc="center left",
                            bbox_to_anchor=(1.02, 0.5), frameon=False)

            ax_dens = fig.add_subplot(gs_layout[1, 0])
            all_draws = mu_samples[:, :, k, p].reshape(-1)
            ci_low, ci_high = np.percentile(all_draws, [2.5, 97.5])
            for chain in range(n_chains):
                sns.kdeplot(mu_samples[chain, :, k, p], ax=ax_dens, fill=False)
            ax_dens.axvline(ci_low, color="black", linestyle=":", lw=1.2, label="95% CI")
            ax_dens.axvline(ci_high, color="black", linestyle=":", lw=1.2)
            ax_dens.set_xlabel("Value")
            ax_dens.set_ylabel("Density")
            ax_dens.set_title(f"95% CI: [{ci_low:.3f}, {ci_high:.3f}]", fontsize=10)
            ax_dens.grid(True)

            ax_acf = fig.add_subplot(gs_layout[1, 1])
            for chain in range(n_chains):
                ax_acf.plot(compute_acf(mu_samples[chain, :, k, p], nlags=n_lags), alpha=1)
            ax_acf.set_xlabel("Lag")
            ax_acf.set_ylabel("Autocorrelation")
            ax_acf.set_ylim(-0.1, 1.05)
            ax_acf.grid(True)
            plt.show()


def generate_delta_summaries(delta_samples, param_names, demo_names, true_delta=None):
    mean, std = np.mean(delta_samples, axis=0), np.std(delta_samples, axis=0)
    df_post = pd.DataFrame(index=demo_names, columns=param_names)

    for i in range(len(demo_names)):
        for j in range(len(param_names)):
            df_post.iloc[i, j] = f"{mean[i, j]:.2f} ({std[i, j]:.2f})"

    print("\n=== Posterior distribution of Delta (mean + std) ===")
    display(df_post)

    if true_delta is not None:
        diff_mean = np.abs(true_delta - mean)
        df_diff = pd.DataFrame(index=demo_names, columns=param_names)
        for i in range(len(demo_names)):
            for j in range(len(param_names)):
                df_diff.iloc[i, j] = f"{diff_mean[i, j]:.3f}"
        print("\n=== Absolute Difference in Delta (|True - Posterior|) ===")
        display(df_diff)


def plot_delta_distributions(delta_samples, param_names, demo_names, true_delta=None):
    n_demos, n_params = len(demo_names), len(param_names)
    fig, axes = plt.subplots(n_demos, n_params, figsize=(4 * n_params, 3.5 * n_demos))

    for d in range(n_demos):
        for p in range(n_params):
            ax = axes[d, p] if n_demos > 1 else (axes[p] if n_params > 1 else axes)
            samples = delta_samples[:, d, p]
            ci_low, ci_high = np.percentile(samples, [2.5, 97.5])

            sns.kdeplot(samples, ax=ax, fill=True, color="#1f77b4", alpha=0.5, label="Posterior")
            ax.axvline(ci_low, color="#1f77b4", linestyle=":", lw=1.5, label="95% CI")
            ax.axvline(ci_high, color="#1f77b4", linestyle=":", lw=1.5)
            if true_delta is not None:
                ax.axvline(true_delta[d, p], color="#D62728", linestyle="--", lw=2,
                           label="True Value")

            if d == 0:
                ax.set_title(param_names[p], fontweight="bold")
            if p == 0:
                ax.set_ylabel(demo_names[d], fontweight="bold")
            ax.grid(True, alpha=0.3)

            if d == 0 and p == 0:
                handles, labels = ax.get_legend_handles_labels()
                unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels))
                          if l not in labels[:i]]
                ax.legend(*zip(*unique), loc="upper right")

    plt.suptitle("Posterior Distributions: Global Shift Matrix (Delta)", fontsize=18, y=1.05)
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------- #
# 4. Parameter recovery (betas) — unaffected by K_MODEL vs K_TRUE
#    (each unit has exactly one true beta regardless of component count)
# --------------------------------------------------------------------------- #
def plot_beta_scatter(beta_samples, true_betas, param_names):
    if true_betas is None:
        return
    n_units, P = beta_samples.shape[2], len(param_names)
    liesel_beta_full = beta_samples.reshape(-1, n_units, P)
    l_beta_means = np.mean(liesel_beta_full, axis=0)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for i, param in enumerate(param_names):
        ax.scatter(true_betas[:, i], l_beta_means[:, i], alpha=0.6, s=25,
                   label=param, color=colors[i % len(colors)])

    min_val = min(true_betas.min(), l_beta_means.min())
    max_val = max(true_betas.max(), l_beta_means.max())
    ax.plot([min_val, max_val], [min_val, max_val], "k--", lw=2, alpha=0.7,
            label="Perfect Recovery")
    ax.set_title("Parameter Recovery: Posterior Beta Means vs True Betas", fontsize=16)
    ax.set_xlabel("True Simulated Beta Values", fontsize=12)
    ax.set_ylabel("Liesel Posterior Means", fontsize=12)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.show()


def plot_beta_distributions(samples, p_names, title_prefix, true_vals=None,
                            color_l="#1f77b4", color_t="#d62728"):
    n = len(p_names)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = np.array([axes])

    for i, (ax, param) in enumerate(zip(axes.flatten(), p_names)):
        if i >= len(p_names):
            ax.axis("off")
            continue

        l_vals = samples[:, i]
        ci_low, ci_high = np.percentile(l_vals, [2.5, 97.5])

        sns.kdeplot(l_vals, ax=ax, fill=True, color=color_l, alpha=0.5, label="Posterior")
        ax.axvline(ci_low, color=color_l, linestyle=":", lw=1.5, label="95% CI")
        ax.axvline(ci_high, color=color_l, linestyle=":", lw=1.5)
        if true_vals is not None:
            ax.axvline(true_vals[i], color=color_t, linestyle="--", lw=2,
                       label=f"True Value ({true_vals[i]:.2f})")

        ax.set_title(param, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.2)

        if i == 0:
            handles, labels = ax.get_legend_handles_labels()
            unique = [(h, l) for j2, (h, l) in enumerate(zip(handles, labels))
                      if l not in labels[:j2]]
            ax.legend(*zip(*unique), loc="upper right")

    plt.suptitle(title_prefix, y=1.02, fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()


def plot_beta_unit_diagnostics(beta_samples, unit_idx, param_names, n_lags=30):
    """Trace, distribution, and ACF for all parameters of one household.

    beta_samples : (chains, draws, N, P)
    """
    unit_draws = beta_samples[:, :, unit_idx, :]   # (chains, draws, P)
    n_chains, _, P = unit_draws.shape

    fig, axes = plt.subplots(P, 3, figsize=(15, 4 * P))
    if P == 1:
        axes = axes[np.newaxis, :]

    for p in range(P):
        draws_p = unit_draws[:, :, p]
        all_draws = draws_p.reshape(-1)
        ci_low, ci_high = np.percentile(all_draws, [2.5, 97.5])

        ax_trace, ax_dens, ax_acf = axes[p, 0], axes[p, 1], axes[p, 2]

        for chain in range(n_chains):
            ax_trace.plot(draws_p[chain], label=f"{chain}")
        ax_trace.set_ylabel(param_names[p], fontweight="bold")
        ax_trace.set_xlabel("Iteration")
        ax_trace.grid(True, alpha=0.4)

        for chain in range(n_chains):
            sns.kdeplot(draws_p[chain], ax=ax_dens, fill=False)
        ax_dens.axvline(ci_low, color="black", linestyle=":", lw=1.2)
        ax_dens.axvline(ci_high, color="black", linestyle=":", lw=1.2)
        ax_dens.set_xlabel("Value")
        ax_dens.grid(True, alpha=0.4)

        for chain in range(n_chains):
            ax_acf.plot(compute_acf(draws_p[chain], nlags=n_lags), alpha=1)
        ax_acf.set_xlabel("Lag")
        ax_acf.set_ylabel("Autocorrelation")
        ax_acf.set_ylim(-0.1, 1.05)
        ax_acf.grid(True, alpha=0.4)

        if p == 0:
            ax_trace.set_title("Trace", fontsize=11)
            ax_dens.set_title(f"Distribution - 95% CI: [{ci_low:.3f}, {ci_high:.3f}]", fontsize=10)
            ax_acf.set_title("ACF", fontsize=11)
            ax_trace.legend(title="Chain", loc="center left",
                            bbox_to_anchor=(1.02, 0.5), frameon=False)
        else:
            ax_dens.set_title(f"95% CI: [{ci_low:.3f}, {ci_high:.3f}]", fontsize=10)

    plt.suptitle(f"Posterior Diagnostics - Household {unit_idx}", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------- #
# 5. Component probabilities (pvec)
# --------------------------------------------------------------------------- #
def plot_pvec_diagnostics(pvec_samples, K, true_pvec=None, K_true=None, n_lags=30):
    """
    Diagnostics for each model component's weight.

    K       : number of MODEL components (K_MODEL) — drives the loop.
    true_pvec : ground-truth weights, length K_true.
    K_true  : number of true components. For k >= K_true the component is
              spurious and no true value is overlaid. If None, assumes K_true==K.
    """
    if true_pvec is not None and K_true is None:
        K_true = K

    n_chains = pvec_samples.shape[0]
    for k in range(K):
        has_true = (true_pvec is not None and k < K_true)

        fig = plt.figure(figsize=(12, 7))
        fig.suptitle(f"Diagnostics: pvec[{k}]", fontsize=14, y=0.95)
        gs_layout = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1], hspace=0.35, wspace=0.25)

        ax_trace = fig.add_subplot(gs_layout[0, :])
        for chain in range(n_chains):
            ax_trace.plot(pvec_samples[chain, :, k], label=f"Chain {chain}")
        ax_trace.set_xlabel("Iteration")
        ax_trace.set_ylabel("Probability")
        ax_trace.set_ylim(-0.05, 1.05)
        ax_trace.grid(True, alpha=0.4)
        ax_trace.legend(title="Chain", loc="center left",
                        bbox_to_anchor=(1.02, 0.5), frameon=False)

        ax_dens = fig.add_subplot(gs_layout[1, 0])
        all_draws = pvec_samples[:, :, k].reshape(-1)
        post_mean, ci_low, ci_high = all_draws.mean(), *np.percentile(all_draws, [2.5, 97.5])
        for chain in range(n_chains):
            sns.kdeplot(pvec_samples[chain, :, k], ax=ax_dens, fill=False)
        ax_dens.axvline(post_mean, color="black", linestyle="-", lw=1.5,
                        label=f"Post. Mean: {post_mean:.3f}")
        ax_dens.axvline(ci_low, color="black", linestyle=":", lw=1.2)
        ax_dens.axvline(ci_high, color="black", linestyle=":", lw=1.2)
        ax_dens.set_xlabel("Probability")
        ax_dens.set_ylabel("Density")
        ax_dens.set_xlim(0, 1)
        ax_dens.legend(fontsize=9)
        ax_dens.grid(True, alpha=0.3)
        ax_dens.set_title(f"95% CI: [{ci_low:.3f}, {ci_high:.3f}]", fontsize=10)

        ax_acf = fig.add_subplot(gs_layout[1, 1])
        for chain in range(n_chains):
            ax_acf.plot(compute_acf(pvec_samples[chain, :, k], nlags=n_lags),
                        alpha=0.8, label=f"Chain {chain}")
        ax_acf.axhline(0, color="black", lw=0.8, linestyle="--")
        ax_acf.set_xlabel("Lag")
        ax_acf.set_ylabel("Autocorrelation")
        ax_acf.set_ylim(-0.15, 1.05)
        ax_acf.grid(True, alpha=0.3)
        plt.show()


def summarize_pvec(pvec_samples_sorted, K, true_pvec=None, K_true=None):
    """
    Posterior summary table for component weights.

    K       : number of MODEL components (K_MODEL) — drives the loop.
    true_pvec : ground-truth weights, length K_true. For k >= K_true the
              True_pvec / True_in_CI fields are NaN (spurious component).
    K_true  : number of true components. If None, assumes K_true == K.
    """
    if true_pvec is not None and K_true is None:
        K_true = K

    flat = pvec_samples_sorted.reshape(-1, K)
    means = flat.mean(axis=0)
    sorted_indices = np.argsort(means)[::-1]  # descending by posterior mean

    true_pvec_desc = None
    if true_pvec is not None:
        true_pvec_desc = np.sort(true_pvec)[::-1]  # match rank-to-rank

    rows = []
    for rank, k in enumerate(sorted_indices):
        draws = flat[:, k]
        ci_low, ci_high = np.percentile(draws, [2.5, 97.5])
        row = {
            "Component":      k,
            "Posterior_Mean": draws.mean(),
            "Posterior_Std":  draws.std(),
            "CI_2.5%":        ci_low,
            "CI_97.5%":       ci_high,
        }
        if true_pvec_desc is not None:
            if rank < K_true:
                row["True_pvec"]  = float(true_pvec_desc[rank])
                row["True_in_CI"] = bool(ci_low <= true_pvec_desc[rank] <= ci_high)
            else:
                row["True_pvec"]  = np.nan
                row["True_in_CI"] = np.nan
        rows.append(row)

    print("\n=== pvec: Posterior Summary ===")
    display(pd.DataFrame(rows).set_index("Component").round(4))


# --------------------------------------------------------------------------- #
# 6. Export for marginal-density comparison
#    (K here is K_MODEL — the export simply serialises all model components)
# --------------------------------------------------------------------------- #
def export_posterior_to_pickle(samples, K, P, filename, output_dir="results"):
    """
    Export Liesel posterior draws (mu, sigma, std, pvec) to a .pkl file for
    downstream marginal-density comparison. K is K_MODEL.
    """
    import pathlib
    import pickle

    print("Preparing posterior samples for export...")

    # mu
    mu_samples_flat = np.array(samples["mu_k"]).reshape(-1, K, P)
    R = mu_samples_flat.shape[0]

    # Sigma (from latent Cholesky-of-precision)
    latent_draws = samples["sigma_inv_chol_k_latent"]
    bijector_tril = tfb.FillScaleTriL()

    def latent_to_sigma(latent_vec):
        L = bijector_tril.forward(latent_vec)
        precision = L @ L.T
        return jnp.linalg.inv(precision)

    v_lts = jax.vmap(jax.vmap(jax.vmap(latent_to_sigma)))
    sigma_samples = v_lts(latent_draws)
    sigma_samples_flat = np.array(sigma_samples).reshape(-1, K, P, P)

    # pvec
    if "pvec" in samples:
        pvec_samples = samples["pvec"]
    elif "pvec_latent" in samples:
        pvec_samples = tfb.SoftmaxCentered().forward(samples["pvec_latent"])
    else:
        raise KeyError("Neither 'pvec' nor 'pvec_latent' found in samples.")
    flat_pvec = np.array(pvec_samples).reshape(-1, K)

    # Std (diagonal of Sigma_k)
    std_draws = np.zeros((R, K, P))
    for k in range(K):
        for j in range(P):
            std_draws[:, k, j] = np.sqrt(np.maximum(sigma_samples_flat[:, k, j, j], 0.0))

    # Sanity checks
    assert mu_samples_flat.shape == (sigma_samples_flat.shape[0], K, P), \
        "mu and sigma R-dimension mismatch"
    assert flat_pvec.shape[0] == mu_samples_flat.shape[0], \
        "pvec and mu R-dimension mismatch"
    assert np.allclose(flat_pvec.sum(axis=1), 1.0, atol=1e-4), \
        "pvec rows do not sum to 1"

    print(f" - R (total posterior draws) : {R}")
    print(f" - K (model components)      : {K}")
    print(f" - P (parameters)            : {P}")
    print(f" - pvec mean per component   : {flat_pvec.mean(axis=0).round(4)}")

    data_dict = {
        "mu":    mu_samples_flat,
        "std":   std_draws,
        "pvec":  flat_pvec,
        "sigma": sigma_samples_flat,
    }

    filepath = pathlib.Path(output_dir) / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(data_dict, f)

    print(f"\nSaved {K}-component draws -> {filepath.absolute()}")
    print(f"   mu    shape : {mu_samples_flat.shape}")
    print(f"   sigma shape : {sigma_samples_flat.shape}")
    print(f"   std   shape : {std_draws.shape}")
    print(f"   pvec  shape : {flat_pvec.shape}")


def _recover_pvec(posterior_samples):
    if "pvec" in posterior_samples:
        return np.asarray(posterior_samples["pvec"])
    if "pvec_latent" in posterior_samples:
        return np.asarray(tfb.SoftmaxCentered().forward(posterior_samples["pvec_latent"]))
    raise KeyError("Neither 'pvec' nor 'pvec_latent' in posterior samples.")


def _sigma_from_latent(latent):                      # (C,S,K,n_latent) -> (C,S,K,P,P)
    b = tfb.FillScaleTriL()
    L = np.asarray(b.forward(latent))                # Cholesky of precision
    prec = np.einsum("...ij,...kj->...ik", L, L)     # L Lᵀ = Σ⁻¹
    return np.linalg.inv(prec)


def invariant_convergence_summary(posterior_samples, include_cov=True):
    """
    R-hat and ESS for LABEL-INVARIANT functionals — the honest convergence check
    for a mixture, since per-component R-hat is meaningless under label switching.
    Arrays expected shape (chains, draws, ...).
    """
    mu   = np.asarray(posterior_samples["mu_k"])     # (C,S,K,P)
    pvec = _recover_pvec(posterior_samples)          # (C,S,K)
    C, S, K, P = mu.shape
    rows = []

    mix_mean = np.einsum("csk,cskp->csp", pvec, mu)  # E[u] = Σ_k p_k μ_k
    for p in range(P):
        a = mix_mean[:, :, p]
        rows.append({"quantity": f"E[u]_{p}", "rhat": float(az.rhat(a)), "ess": float(az.ess(a))})

    pvec_sorted = np.sort(pvec, axis=-1)
    for k in range(K):
        a = pvec_sorted[:, :, k]
        rows.append({"quantity": f"pvec_sorted_{k}", "rhat": float(az.rhat(a)), "ess": float(az.ess(a))})

    if include_cov:
        Sigma = _sigma_from_latent(np.asarray(posterior_samples["sigma_inv_chol_k_latent"]))
        outer  = np.einsum("cskp,cskq->cskpq", mu, mu)
        second = np.einsum("csk,cskpq->cspq", pvec, Sigma + outer)
        mbar   = np.einsum("csp,csq->cspq", mix_mean, mix_mean)
        tr     = np.einsum("cspp->cs", second - mbar)
        rows.append({"quantity": "tr(Cov[u])", "rhat": float(az.rhat(tr)), "ess": float(az.ess(tr))})

    return pd.DataFrame(rows).set_index("quantity")


def per_chain_mu_means(posterior_samples):
    """Per-chain posterior mean of mu_k — if chains are each internally tight but
    disagree slot-by-slot, that's between-chain label switching, not a broken run."""
    mu = np.asarray(posterior_samples["mu_k"])
    return {f"chain_{c}": mu[c].mean(axis=0) for c in range(mu.shape[0])}