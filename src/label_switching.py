"""
Label-switching post-processing for the mixture WEIGHTS pvec (Papastamoulis &
Iliopoulos 2010; Papastamoulis 2016, JSS - the `label.switching` R package).
Only pvec is relabeled; component means/covariances are not post-processed -
all other inference in the study uses label-invariant functionals
(analysis.invariant_convergence_summary). Input draws are never mutated.

Method: ECR iterative version 1 (Papastamoulis 2016, Algorithm 5) on HARD
allocations z[t,i] = argmax_k r_{tik}. Liesel marginalizes the allocations, so
responsibilities are reconstructed post-hoc: r_{tik} ∝ pvec_{tk} *
N(beta_{ti}; mu_{tk} + Z_i @ Delta_t, Sigma_{tk}) - the Z@Delta term is
required (Rossi Eq. 5.5.19). The same reconstruction serves NUTS, HMC and
bayesm (bayesm's own sampled z is ignored). Hard allocations tolerate the
reconstruction noise better than the soft probabilities `ecr.iterative.2`
needs. Empty components under K_MODEL > K_TRUE claim no observations and need
no special casing.

Relabeling removes PERMUTATION ambiguity only, not genuine MULTIMODALITY
(chains in different partition modes); the report distinguishes the two.

Dependencies: numpy, scipy, arviz.
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
    """Reconstruct per-unit hard allocations z[t,i] = argmax_k r_{tik}, (C,S,N).

    posterior_samples needs mu_k, pvec (or pvec_latent), sigma_inv_chol_k_latent
    and beta_i. If Z (N, D) is given and 'Delta' is present, the household mean
    mu_k + Z_i @ Delta is used. `chunk` caps peak memory.
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

    for a in range(0, M, chunk):
        b = min(a + chunk, M)
        Sc   = Sigma[a:b] + 1e-6 * eyeP                              # (m,K,P,P) jitter
        Sinv = np.linalg.inv(Sc)                                     # (m,K,P,P)
        _, slogdet = np.linalg.slogdet(Sc)                          # (m,K)
        if use_delta:
            ZD  = np.einsum("nd,mdp->mnp", Z, Delta[a:b])            # (m,N,P)
            loc = mu[a:b][:, None, :, :] + ZD[:, :, None, :]         # (m,N,K,P)
        else:
            loc = np.broadcast_to(mu[a:b][:, None, :, :], (b - a, N, K, P))
        diff = beta[a:b][:, :, None, :] - loc                        # (m,N,K,P)
        quad = np.einsum("mnkp,mkpq,mnkq->mnk", diff, Sinv, diff)     # (m,N,K)
        logr = logpvec[a:b][:, None, :] - 0.5 * (P * log2pi + slogdet[:, None, :] + quad)
        z[a:b] = np.argmax(logr, axis=2).astype(np.int16)

    return z.reshape(C, S, N)


# --------------------------------------------------------------------------- #
# 2. ECR.iterative.1  (pivot-free; Papastamoulis & Iliopoulos 2016)
# --------------------------------------------------------------------------- #
def ecr_iterative_1(z, K, maxiter=100):
    """ECR iterative version 1 (Papastamoulis 2016, Algorithm 5) on hard
    allocations z (C,S,N) in {0..K-1}. From identity permutations, alternate
    (2) pivot z*_i = per-unit mode of the relabeled allocations and (3) per
    draw the permutation maximising sum_i 1{tau_t(z[t,i]) = z*_i} (a K x K
    linear assignment), until (4) the total agreement stops improving. The
    objective is monotone and bounded, so termination is guaranteed.

    Returns (tau (C,S,K) raw->ref maps, converged, n_iter, switching_rate).
    """
    C, S, N = z.shape
    M = C * S
    zf = z.reshape(M, N).astype(np.int64)
    masks = [(zf == j).astype(np.float64) for j in range(K)]   # per-component membership

    tau = np.tile(np.arange(K), (M, 1))                        # (1) identity init
    best_F = -1.0
    converged = False
    n_iter = 0
    for it in range(maxiter):
        n_iter = it + 1
        # (2) pivot = per-unit mode of the relabeled allocations, one-hot coded
        zr = np.take_along_axis(tau, zf, axis=1)
        counts = np.stack([(zr == k).sum(axis=0) for k in range(K)], axis=1)  # (N,K)
        zstar = np.zeros((N, K))
        zstar[np.arange(N), counts.argmax(axis=1)] = 1.0
        # (3) A_t[j,k] = #{i : z[t,i] = j, z*_i = k}; maximise sum_j A_t[j, tau(j)]
        A = np.zeros((M, K, K))
        for j in range(K):
            A[:, j, :] = masks[j] @ zstar                      # (M,N)@(N,K) -> (M,K)
        new = np.empty_like(tau)
        F = 0.0
        for t in range(M):
            _, col = linear_sum_assignment(-A[t])              # raw j -> ref col[j]
            new[t] = col
            F += A[t][np.arange(K), col].sum()
        tau = new
        # (4) stop at the first non-improvement of the agreement objective
        if F <= best_F:
            converged = True
            break
        best_F = F

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


def relabel_pvec(posterior_samples, K, Z=None, K_true=None, maxiter=100):
    """Relabel the mixture weights with ECR iterative version 1.

    ONLY pvec is post-processed. Returns (pvec_relabeled (C,S,K), report): the
    per-draw permutations are applied to pvec alone, and the slots are anchored
    by descending mean weight (slot 0 = heaviest) so the labeling is
    reproducible. `report` carries the permutations, convergence, switching
    rate and live slots.
    """
    if K_true is None:
        K_true = K

    pvec = np.asarray(analysis._recover_pvec(posterior_samples))    # (C,S,K)
    C, S = pvec.shape[:2]

    if K == 1:
        return pvec.copy(), {
            "K": 1, "K_true": K_true, "converged": True, "n_iter": 0,
            "switching_rate": 0.0, "permutations": np.zeros((C, S, 1), dtype=int),
            "live_slots": [0],
            "note": "single component - label switching not applicable",
        }

    # ECR (Algorithm 5) on the reconstructed allocations.
    z = reconstruct_allocations(posterior_samples, Z=Z)
    tau, converged, n_iter, switching_rate = ecr_iterative_1(z, K, maxiter=maxiter)

    # tau maps raw label j -> reference slot. To reorder the weight axis,
    # NEW slot k must take OLD component tau^{-1}(k): perm = argsort(tau).
    perm = np.argsort(tau, axis=2)                      # (C,S,K): old index per new slot

    # Canonical global anchoring: order slots by descending mean weight so the
    # labeling is reproducible (slot 0 = heaviest component).
    pvec_re = _apply_perms_axis(pvec, perm)
    order = np.argsort(-pvec_re.reshape(C * S, K).mean(axis=0))
    perm = perm[:, :, order]

    return _apply_perms_axis(pvec, perm), {
        "K": K, "K_true": K_true,
        "converged": bool(converged), "n_iter": n_iter,
        "switching_rate": switching_rate,
        "permutations": perm,
        "live_slots": list(range(min(K_true, K))),   # post-anchor heaviest K_true slots
        "note": "" if converged else f"ECR did NOT converge in {maxiter} sweeps",
    }


# --------------------------------------------------------------------------- #
# 4. Diagnostics (before / after) and outcome classification
# --------------------------------------------------------------------------- #
def pvec_convergence_table(pvec, K, K_true=None, label=""):
    """Per-slot R-hat and ESS of the weights. `pvec` is (C,S,K), raw or
    relabeled. Slots are ordered by descending mean weight; the K_true heaviest
    are flagged 'live' (surplus slots are unidentified noise)."""
    if K_true is None:
        K_true = K
    pvec = np.asarray(pvec)
    C, S = pvec.shape[:2]
    order = np.argsort(-pvec.reshape(C * S, K).mean(axis=0))   # descending weight
    live_set = set(int(k) for k in order[:min(K_true, K)])
    rows = [{"slot": int(k), "live": int(k) in live_set,
             "rhat": float(az.rhat(pvec[:, :, k])),
             "ess":  float(az.ess(pvec[:, :, k]))} for k in order]
    df = pd.DataFrame(rows)
    if label:
        df.insert(0, "stage", label)
    return df


def plot_before_after_traces(before, after, K, title="", true_vals=None, ylim=None):
    """Weight traces for all K slots, raw (left) vs relabeled (right).

    before, after : (C, S, K) pvec arrays. true_vals: optional length-K_true
    truth, drawn on the 'after' rows rank-to-rank (slots are weight-ordered
    after relabeling). ylim: optional (lo, hi) y-limits."""
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


def invariance_guard(pvec_before, pvec_after, atol=1e-8):
    """No-corruption guard, NOT a success signal: a pure relabeling only
    permutes each draw's weights, so the per-draw SORTED pvec must be identical
    before and after; a mismatch means the draws were corrupted. A match does
    not certify the relabeling is correct."""
    b = np.sort(np.asarray(pvec_before), axis=-1)
    a = np.sort(np.asarray(pvec_after), axis=-1)
    return bool(np.allclose(b, a, atol=atol))


def classify_outcome(report, gate_df, rhat_thresh=1.1):
    """Three-way verdict using the label-INVARIANT gate + the switching rate.

    MULTIMODAL : invariant sorted-pvec R-hat is high -> different partition modes;
                 sorting already removed labels, so it is not a label artifact and
                 relabeling cannot fix it.
    PERMUTATION-FIXED : real switching was present and aligned, gate passes.
    NO-OP : almost no switching was present.
    """
    ps = gate_df[gate_df.index.str.startswith("pvec_sorted")]
    max_ps = float(ps["rhat"].max()) if len(ps) else np.nan

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
        "gate_passed": bool(np.isfinite(max_ps) and max_ps < rhat_thresh),
    }
