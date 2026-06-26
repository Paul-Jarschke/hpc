"""
Local validation of the bayesm -> canonical posterior_raw conversion (src/bayesm_convert.py).

The bayesm arm fits in R but the numerical bridge (bayesm precision -> TFP FillScaleTriL
latent -> Sigma) is pure Python/JAX and is the single highest-risk step: if the
parameterisation does not match the Liesel model's FillScaleTriL exactly, every
downstream Sigma recovery / moment / marginal is silently wrong. This test pins it
WITHOUT needing R, by (1) round-tripping known precision matrices and (2) driving the
full read_raw_draws() over synthetic R-style column-major .bin files.

Run:  .venv/Scripts/python.exe tests/test_bayesm_convert.py
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src import analysis
from src import bayesm_convert as bc


def _random_spd(P, rng, scale=1.0):
    A = rng.normal(size=(P, P))
    return scale * (A @ A.T) + P * np.eye(P)        # SPD, well-conditioned


def test_precision_latent_roundtrip():
    """precision -> latent -> _sigma_from_latent must recover Sigma = precision^{-1}."""
    rng = np.random.default_rng(0)
    C, S, K, P = 2, 3, 5, 4
    Sigma = np.empty((C, S, K, P, P))
    for c in range(C):
        for s in range(S):
            for k in range(K):
                Sigma[c, s, k] = _random_spd(P, rng)
    prec = np.linalg.inv(Sigma)

    latent = bc.precision_to_latent(prec)
    assert latent.shape == (C, S, K, P * (P + 1) // 2), f"latent shape {latent.shape}"
    assert np.all(np.isfinite(latent))

    Sigma_rt = np.asarray(analysis._sigma_from_latent(latent))
    err = np.max(np.abs(Sigma_rt - Sigma))
    assert err < 1e-4, f"Sigma round-trip error too large: {err}"
    print(f"[ok] precision<->latent round-trip: max|Sigma_rt - Sigma| = {err:.2e}")


def test_read_bin_fortran_order():
    """_read_bin must reconstruct an R column-major dump with the right logical shape."""
    rng = np.random.default_rng(1)
    shape = (3, 5, 4)                                # (S, K, P)
    arr = rng.normal(size=shape)
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "x.bin"
        # R writes as.double(arr) column-major == numpy ravel(order="F").
        arr.ravel(order="F").astype("<f8").tofile(p)
        back = bc._read_bin(p, shape)
    assert np.allclose(back, arr), "Fortran-order .bin round-trip failed"
    print("[ok] _read_bin Fortran-order round-trip")


def test_read_raw_draws_end_to_end():
    """Drive the full read_raw_draws() over synthetic R-style bins (what the R sampler emits)."""
    rng = np.random.default_rng(2)
    C, S, K, P, N, D = 2, 4, 5, 4, 7, 2
    # ground-truth arrays in the canonical (C,S,...) layout
    mu = rng.normal(size=(C, S, K, P))
    pvec = rng.dirichlet(np.ones(K), size=(C, S))          # (C,S,K) simplex
    beta = rng.normal(size=(C, S, N, P))
    delta = rng.normal(size=(C, S, D, P))
    Sigma = np.empty((C, S, K, P, P))
    for idx in np.ndindex(C, S, K):
        Sigma[idx] = _random_spd(P, rng)
    prec = np.linalg.inv(Sigma)

    with tempfile.TemporaryDirectory() as d:
        raw = Path(d)
        # write one column-major .bin per chain, exactly like src/bayesm_sampler.R
        for c in range(C):
            for name, a in (("mu", mu), ("prec", prec), ("pvec", pvec),
                            ("beta", beta), ("delta", delta)):
                a[c].ravel(order="F").astype("<f8").tofile(raw / f"{name}_chain{c}.bin")
        dims = {"chains": C, "n_samples": S, "K": K, "P": P, "D": D, "N": N,
                "has_Z": True, "order": "F",
                "per_chain": {"mu": [S, K, P], "prec": [S, K, P, P], "pvec": [S, K],
                              "beta": [S, N, P], "delta": [S, D, P]}}
        (raw / "dims.json").write_text(json.dumps(dims))
        (raw / "meta_r.json").write_text(json.dumps(
            {"seeds": [42, 43], "durations_s": [1.0, 1.1], "loglike_mean": [-1.0, -1.1],
             "k_true": 3, "bayesm_version": "3.1-7"}))

        canon, meta_r, dims_out = bc.read_raw_draws(raw)

    assert set(canon) == {"mu_k", "sigma_inv_chol_k_latent", "pvec", "beta_i", "Delta"}
    assert "pvec_latent" not in canon, "bayesm must emit simplex 'pvec', not 'pvec_latent'"
    assert canon["mu_k"].shape == (C, S, K, P)
    assert canon["beta_i"].shape == (C, S, N, P)
    assert canon["Delta"].shape == (C, S, D, P)
    assert canon["sigma_inv_chol_k_latent"].shape == (C, S, K, P * (P + 1) // 2)
    assert np.allclose(canon["mu_k"], mu) and np.allclose(canon["beta_i"], beta)
    assert np.allclose(canon["pvec"], pvec) and np.allclose(canon["Delta"], delta)
    # the reconstructed latent must invert back to the original Sigma
    Sigma_rt = np.asarray(analysis._sigma_from_latent(canon["sigma_inv_chol_k_latent"]))
    err = np.max(np.abs(Sigma_rt - Sigma))
    assert err < 1e-4, f"end-to-end Sigma round-trip error: {err}"
    assert meta_r["seeds"] == [42, 43]
    print(f"[ok] read_raw_draws end-to-end: keys + shapes + Sigma round-trip (err={err:.2e})")


if __name__ == "__main__":
    test_precision_latent_roundtrip()
    test_read_bin_fortran_order()
    test_read_raw_draws_end_to_end()
    print("\nALL BAYESM CONVERSION TESTS PASSED")
