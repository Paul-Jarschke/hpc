"""
ECR.iterative.1 relabeling for the mixture HBMNL posterior (Papastamoulis &
Iliopoulos 2010; Papastamoulis 2016, JSS - the `label.switching` R package).

This module is ADDITIVE: it never mutates the saved draws and never changes any
existing analysis code. It returns a relabeled COPY of the posterior plus a report.

Why ECR.iterative.1 (of the three ECR variants in label.switching)
------------------------------------------------------------------
`label.switching` offers `ecr` (needs a pivot), `ecr.iterative.1` (pivot-free,
uses the hard allocation matrix z) and `ecr.iterative.2` (pivot-free, uses the
m x n x K classification-probability array p). For this study, up to K=5:

  * `ecr` is rejected: its pivot allocation is not robust when independent
    chains have collapsed to different, incompatible pivots.
  * `ecr.iterative.1` is chosen: pivot-free, and it works on the HARD allocations
    z = argmax_k r_{ik}. Because Liesel marginalizes the allocations, we have to
    RECONSTRUCT responsibilities post-hoc anyway; hard allocations are far more
    robust to the noise in that reconstruction than the full soft p that
    `ecr.iterative.2` would consume. It is also the simplest of the three.
  * Overspecified K_MODEL>K_TRUE is handled for free: collapsed (empty) components
    claim essentially no observations, so they never compete for a real
    component's slot - no special casing, no mass weighting.

The algorithm (faithful to the package)
---------------------------------------
Inputs: z, an m x n matrix of allocations z[t,i] in {0..K-1}. Maintain a
per-iteration relabel map tau_t : {0..K-1} -> {0..K-1} (raw label -> reference
label), initialised to the identity. Iterate to a fixed point:

  1. q[i,k] = (1/m) * sum_t 1{ tau_t(z[t,i]) = k }            (running reference)
  2. for each iteration t, choose tau_t to MAXIMISE sum_i q[i, tau_t(z[t,i])],
     i.e. solve the K x K assignment  max_sigma sum_j A_t[j, sigma(j)]  where
     A_t[j,k] = sum_{i : z[t,i]=j} q[i,k]   (via scipy.optimize.linear_sum_assignment)
  3. repeat until no tau_t changes.

Reconstructing the allocations (Rossi Eq. 5.5.19)
-------------------------------------------------
The household-specific component mean is mu_k + Z_i @ Delta (Z@Delta MUST be
included). Responsibilities r_{tik} ∝ pvec_{tk} * N(beta_{ti} ; mu_{tk}+Z_i Delta_t,
Sigma_{tk}); z[t,i] = argmax_k r_{tik}. This is identical for NUTS, HMC and bayesm
(bayesm's own z is deliberately ignored so one method serves all three).

What is and isn't fixed
-----------------------
Relabeling removes PERMUTATION ambiguity (one mode). It cannot remove genuine
MULTIMODALITY (chains in different partition modes of the mixture weight
posterior). The report classifies the outcome honestly. Component-level recovery
is illustrative-only; the load-bearing inference is on the label-invariant
functionals (analysis.invariant_convergence_summary), which relabeling leaves
mathematically unchanged.

Dependency-free: numpy, scipy, arviz only (no scikit-learn).
"""

import numpy as np
import pandas as pd
import arviz as az
from scipy.optimize import linear_sum_assignment

from src import analysis


# --------------------------------------------------------------------------- #
# 1. Reconstruct hard allocations  z[t,i] = argmax_k r_{tik}   (Rossi Eq 5.5.19)
# --------------------------------------------------------------------------- #
def reconstruct_allocations(posterior_samples, Z=None, chunk=512):
    """Reconstruct per-unit hard allocations z of shape (C, S, N).

    Parameters
    ----------
    posterior_samples : dict with mu_k (C,S,K,P), pvec/pvec_latent,
                        sigma_inv_chol_k_latent (C,S,K,nlat) and beta_i (C,S,N,P).
    Z                 : (N, D) demographics; if given (and 'Delta' present) the
                        household-specific mean mu_k + Z_i @ Delta is used.
    chunk             : draws processed per batch (caps peak memory).
    """
    mu    = np.asarray(posterior_samples["mu_k"])                     # (C,S,K,P)
    pvec  = np.asarray(analysis._recover_pvec(posterior_samples))    # (C,S,K)
    Sigma = np.asarray(
        analysis._sigma_from_latent(np.asarray(posterior_samples["sigma_inv_chol_k_latent"]))
    )                                                                # (C,S,K,P,P)
    beta  = np.asarray(posterior_samples["beta_i"])                  # (C,S,N,P)

    C, S, K, P = mu.shape
    N = beta.shape[2]
    M = C * S
    mu    = mu.reshape(M, K, P)
    pvec  = pvec.reshape(M, K)
    Sigma = Sigma.reshape(M, K, P, P)
    beta  = beta.reshape(M, N, P)

    use_delta = (Z is not None) and ("Delta" in posterior_samples)
    if use_delta:
        Z = np.asarray(Z)                                            # (N,D)
        Delta = np.asarray(posterior_samples["Delta"]).reshape(M, Z.shape[1], P)

    logpvec = np.log(np.clip(pvec, 1e-300, None))                    # (M,K)
    log2pi  = np.log(2.0 * np.pi)
    eyeP    = np.eye(P)
    z = np.empty((M, N), dtype=np.int16)
    conf = np.empty(M, dtype=float)   # per-draw allocation confidence (for pivot pick)

    for a in range(0, M, chunk):
        b = min(a + chunk, M)
        Sc   = Sigma[a:b] + 1e-6 * eyeP                              # (m,K,P,P) jitter
        Sinv = np.linalg.inv(Sc)                                     # (m,K,P,P)
        _, slogdet = np.linalg.slogdet(Sc)                          # (m,K)  (already this chunk)
        if use_delta:
            ZD  = np.einsum("nd,mdp->mnp", Z, Delta[a:b])            # (m,N,P)
            loc = mu[a:b][:, None, :, :] + ZD[:, :, None, :]         # (m,N,K,P)
        else:
            loc = np.broadcast_to(mu[a:b][:, None, :, :], (b - a, N, K, P))
        diff = beta[a:b][:, :, None, :] - loc                        # (m,N,K,P)
        quad = np.einsum("mnkp,mkpq,mnkq->mnk", diff, Sinv, diff)     # (m,N,K)
        logr = logpvec[a:b][:, None, :] - 0.5 * (P * log2pi + slogdet[:, None, :] + quad)
        z[a:b] = np.argmax(logr, axis=2).astype(np.int16)
        # confidence = mean over units of the max log-responsibility (after normalising)
        lse = np.logaddexp.reduce(logr, axis=2)                      # (m,N)
        conf[a:b] = (logr.max(axis=2) - lse).mean(axis=1)            # mean log P(assigned)

    return z.reshape(C, S, N), conf.reshape(C, S)


# --------------------------------------------------------------------------- #
# 2. ECR.iterative.1  (pivot-free; Papastamoulis & Iliopoulos 2010)
# --------------------------------------------------------------------------- #
def ecr_iterative_1(z, K, pivot=0, maxiter=100):
    """Pivot-free ECR on hard allocations z (C,S,N) in {0..K-1}.

    The running reference q[i,k] = mean_t 1{tau_t(z[t,i])=k} is SEEDED from the
    one-hot allocation of a single pivot draw (the highest-confidence draw, passed
    in via `pivot`), then refined to the global mean by the iteration. Seeding is
    essential: from a uniform reference (identity init under balanced switching)
    every assignment is tied and ECR sticks at the degenerate fixed point. Because
    the reference is then iterated to the global mean, the final labeling is robust
    to the exact pivot (unlike the non-iterative `ecr`, whose pivot allocation is
    not robust across independently-collapsed chains).

    Returns tau (C,S,K) raw->ref maps, converged, n_iter, switching_rate.
    """
    C, S, N = z.shape
    M = C * S
    zf = z.reshape(M, N)
    masks = [(zf == j) for j in range(K)]                   # per-component membership

    # Seed the reference from the pivot draw's hard allocation (one-hot).
    q = np.zeros((N, K))
    q[np.arange(N), zf[pivot]] = 1.0

    tau = np.tile(np.arange(K), (M, 1))
    prev_new = None
    converged = False
    n_iter = 0
    for it in range(maxiter):
        n_iter = it + 1
        # A_t[j,k] = sum_{i: z[t,i]=j} q[i,k]  -> assign to MAXIMISE agreement.
        A = np.zeros((M, K, K))
        for j in range(K):
            A[:, j, :] = masks[j].astype(np.float64) @ q     # (M,N)@(N,K) -> (M,K)
        new = np.empty_like(tau)
        for t in range(M):
            _, col = linear_sum_assignment(-A[t])            # raw j -> ref col[j]
            new[t] = col
        tau = new
        if prev_new is not None and np.array_equal(new, prev_new):
            converged = True
            break
        prev_new = new.copy()
        # Update reference: relabel allocations and recompute frequencies.
        zr = np.take_along_axis(tau, zf, axis=1)
        for k in range(K):
            q[:, k] = (zr == k).mean(axis=0)

    switching_rate = float(np.mean(np.any(tau != np.arange(K), axis=1)))
    return tau.reshape(C, S, K), converged, n_iter, switching_rate


# --------------------------------------------------------------------------- #
# 3. Apply permutations to the component-indexed arrays
# --------------------------------------------------------------------------- #
def _apply_perms_axis(arr, perms):
    """Reorder the K axis (axis=2) of a (C,S,K,...) array by per-draw perms (C,S,K),
    where perms[c,s,k] = OLD component index that goes into NEW slot k."""
    C, S, K = perms.shape
    flat = arr.reshape(C * S, *arr.shape[2:])
    idx = perms.reshape(C * S, K)
    idx = idx.reshape((C * S, K) + (1,) * (flat.ndim - 2))
    idx = np.broadcast_to(idx, (C * S, K) + flat.shape[2:])
    return np.take_along_axis(flat, idx, axis=1).reshape(arr.shape)


def relabel_run(posterior_samples, K, Z=None, K_true=None, maxiter=100):
    """Relabel a mixture posterior with ECR.iterative.1.

    Returns (relabeled, report). `relabeled` is a COPY of posterior_samples with
    'mu_k', 'pvec' (simplex), 'Sigma' and 'sigma_inv_chol_k_latent' permuted to one
    global labeling; beta_i / Delta are untouched and 'pvec_latent' is dropped
    (a K-1 SoftmaxCentered latent cannot be permuted consistently - 'pvec' is
    authoritative). `report` carries the permutations, convergence and switching rate.
    """
    if K_true is None:
        K_true = K

    relabeled = dict(posterior_samples)
    mu = np.asarray(posterior_samples["mu_k"])
    C, S, _, P = mu.shape

    if K == 1:
        return relabeled, {
            "K": 1, "K_true": K_true, "converged": True, "n_iter": 0,
            "switching_rate": 0.0, "permutations": np.zeros((C, S, 1), dtype=int),
            "note": "single component - label switching not applicable",
        }

    # ECR on reconstructed allocations; seed from the highest-confidence draw.
    z, conf = reconstruct_allocations(posterior_samples, Z=Z)
    pivot = int(np.argmax(conf.reshape(-1)))
    tau, converged, n_iter, switching_rate = ecr_iterative_1(z, K, pivot=pivot, maxiter=maxiter)

    # tau maps raw label j -> reference slot. To reorder arrays indexed by raw
    # component, NEW slot k must take OLD component tau^{-1}(k): perm = argsort(tau).
    perm = np.argsort(tau, axis=2)                      # (C,S,K): old index per new slot

    # Canonical global anchoring: order slots by descending mean weight so the
    # labeling is reproducible (slot 0 = heaviest component).
    pvec = np.asarray(analysis._recover_pvec(posterior_samples))
    pvec_re = _apply_perms_axis(pvec, perm)
    order = np.argsort(-pvec_re.reshape(C * S, K).mean(axis=0))
    perm = perm[:, :, order]

    Sigma = np.asarray(
        analysis._sigma_from_latent(np.asarray(posterior_samples["sigma_inv_chol_k_latent"]))
    )
    relabeled["mu_k"]  = _apply_perms_axis(mu, perm)
    relabeled["pvec"]  = _apply_perms_axis(pvec, perm)            # simplex, authoritative
    relabeled["Sigma"] = _apply_perms_axis(Sigma, perm)
    relabeled["sigma_inv_chol_k_latent"] = _apply_perms_axis(
        np.asarray(posterior_samples["sigma_inv_chol_k_latent"]), perm
    )
    relabeled.pop("pvec_latent", None)

    return relabeled, {
        "K": K, "K_true": K_true,
        "converged": bool(converged), "n_iter": n_iter,
        "switching_rate": switching_rate,
        "permutations": perm,
        "live_slots": list(range(min(K_true, K))),   # post-anchor heaviest K_true slots
        "note": "" if converged else f"ECR.iterative.1 did NOT converge in {maxiter} sweeps",
    }


# --------------------------------------------------------------------------- #
# 4. Diagnostics (before / after) and an honest verdict
# --------------------------------------------------------------------------- #
def component_convergence_table(posterior_samples, K, K_true=None, label="", all_slots=True):
    """Per-component R-hat AND ESS on raw mu_k[...,k,p] and pvec[...,k].

    Slots are ordered by descending mean weight and flagged 'live' (the K_true
    heaviest) vs empty. all_slots=True reports every component; False keeps only
    the live ones (empty/overspecified slots are unidentified noise)."""
    if K_true is None:
        K_true = K
    mu   = np.asarray(posterior_samples["mu_k"])
    pvec = np.asarray(analysis._recover_pvec(posterior_samples))
    C, S, _, P = mu.shape
    order = np.argsort(-pvec.reshape(C * S, K).mean(axis=0))   # all K, descending weight
    live_set = set(int(k) for k in order[:min(K_true, K)])
    slots = order if all_slots else order[:min(K_true, K)]

    rows = []
    for k in slots:
        is_live = int(k) in live_set
        rows.append({"slot": int(k), "live": is_live, "quantity": "pvec",
                     "rhat": float(az.rhat(pvec[:, :, k])),
                     "ess":  float(az.ess(pvec[:, :, k]))})
        for p in range(P):
            rows.append({"slot": int(k), "live": is_live, "quantity": f"mu[{p}]",
                         "rhat": float(az.rhat(mu[:, :, k, p])),
                         "ess":  float(az.ess(mu[:, :, k, p]))})
    df = pd.DataFrame(rows)
    if label:
        df.insert(0, "stage", label)
    return df


def plot_before_after_traces(before, after, K, title="", true_vals=None, K_true=None, ylim=None):
    """Overlaid chain traces for ALL K components, raw (before) vs relabeled (after).

    before, after : (C, S, K) arrays of a component-indexed scalar (e.g. pvec, or
    one column of mu). A K x 2 grid (left = before, right = after) makes the label
    switching - and its removal - directly visible for every component.
    true_vals : optional length-K_true ground truth; drawn as a dashed line on the
    matching 'after' rows (after relabeling, slots are ordered by descending weight,
    so true values sorted the same way line up rank-to-rank).
    ylim : optional (lo, hi) y-axis limits applied to every subplot, e.g.
    (-0.05, 1.05) for pvec so the scale matches the analysis notebook's pvec traces."""
    import matplotlib.pyplot as plt

    before = np.asarray(before)
    after = np.asarray(after)
    C = before.shape[0]
    fig, axes = plt.subplots(K, 2, figsize=(13, 1.9 * K), squeeze=False, sharex=True)
    true_sorted = np.sort(true_vals)[::-1] if true_vals is not None else None
    for k in range(K):
        for c in range(C):
            axes[k, 0].plot(before[c, :, k], lw=0.5, alpha=0.75)
            axes[k, 1].plot(after[c, :, k], lw=0.5, alpha=0.75)
        if true_sorted is not None and k < len(true_sorted):
            axes[k, 1].axhline(true_sorted[k], color="k", ls="--", lw=1.0)
        axes[k, 0].set_ylabel(f"comp {k}")
        axes[k, 0].grid(True, alpha=0.3)
        axes[k, 1].grid(True, alpha=0.3)
        if ylim is not None:
            axes[k, 0].set_ylim(*ylim)
            axes[k, 1].set_ylim(*ylim)
    axes[0, 0].set_title(f"{title}  -  BEFORE (raw, label-switched)")
    axes[0, 1].set_title(f"{title}  -  AFTER (ECR relabeled)")
    axes[K - 1, 0].set_xlabel("draw")
    axes[K - 1, 1].set_xlabel("draw")
    fig.tight_layout()
    plt.show()


def mixture_mean(posterior_samples):
    """Label-invariant mixture mean E[u] = sum_k pvec_k mu_k, shape (C,S,P)."""
    mu = np.asarray(posterior_samples["mu_k"])
    pv = np.asarray(analysis._recover_pvec(posterior_samples))
    return np.einsum("csk,cskp->csp", pv, mu)


def invariance_guard(before_samples, after_samples, atol=1e-6):
    """NO-CORRUPTION guard (NOT a success signal). E[u] is symmetric in the
    component triples, so it is invariant under ANY joint per-draw permutation -
    a mismatch means the permutation was applied inconsistently across arrays
    (a bug), but a match does NOT certify the relabeling is correct."""
    return bool(np.allclose(mixture_mean(before_samples), mixture_mean(after_samples), atol=atol))


def classify_outcome(report, gate_df, rhat_thresh=1.1):
    """Honest 3-way verdict using the label-INVARIANT gate + the switching rate.

    MULTIMODAL : invariant sorted-pvec R-hat is high -> different partition modes;
                 sorting already removed labels, so it is not a label artifact and
                 relabeling cannot fix it.
    PERMUTATION-FIXED : real switching was present and aligned, gate passes.
    NO-OP : almost no switching was present.
    """
    ps = gate_df[gate_df.index.str.startswith("pvec_sorted")]
    eu = gate_df[gate_df.index.str.startswith("E[u]")]
    max_ps = float(ps["rhat"].max()) if len(ps) else np.nan
    max_eu = float(eu["rhat"].max()) if len(eu) else np.nan

    if np.isfinite(max_ps) and max_ps > rhat_thresh:
        verdict = "MULTIMODAL (genuine - sorting removed labels, so not a label artifact; ECR aligns within-mode only)"
    elif report["switching_rate"] > 0.05:
        verdict = "PERMUTATION-FIXED (label switching was present and has been aligned by ECR)"
    else:
        verdict = "NO-OP (little/no switching detected; relabeling changed almost nothing)"

    return {
        "verdict": verdict,
        "switching_rate": report["switching_rate"],
        "invariant_pvec_sorted_rhat": max_ps,
        "invariant_Eu_rhat": max_eu,
        "gate_passed": bool(np.isfinite(max_eu) and max_eu < rhat_thresh),
    }
