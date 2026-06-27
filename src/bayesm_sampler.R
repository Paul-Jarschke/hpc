# ==============================================================================
# bayesm mixture-HBMNL sampler — vendored + lightly adapted from
# HierarchicalBayesianMNL/run_single_bayesm_experiment.R (the study's R sampler).
#
# Exposes run_bayesm_fit(), which does PURE SAMPLING with bayesm::rhierMnlRwMixture
# and writes the raw posterior draws in a portable bridge form into out_raw_dir for
# src/bayesm_convert.py to convert into the canonical posterior_raw.pkl (identical
# keys/shapes to the Liesel runners). Adaptation vs the study script: the standalone
# commandArgs parser is replaced by function arguments so the harness run.qmd can
# source() and call it in-process (bayesm already loaded); the sampling logic is
# unchanged.
#
# Bridge format written into out_raw_dir:
#     dims.json                  shapes + axis order for every array
#     meta_r.json                sampling metadata (seeds, timings, loglike, ...)
#     mu_chain<c>.bin            float64 LE, R-dim (S, K, P)
#     prec_chain<c>.bin          float64 LE, R-dim (S, K, P, P)   (Sigma^{-1})
#     pvec_chain<c>.bin          float64 LE, R-dim (S, K)
#     beta_chain<c>.bin          float64 LE, R-dim (S, N, P)
#     delta_chain<c>.bin         float64 LE, R-dim (S, D, P)      (only if Z)
#
# Every .bin is column-major (R native); Python reshapes with order="F" using the
# R dims in dims.json. We export the PRECISION (Sigma^{-1} = rooti %*% t(rooti),
# bayesm's own definition) so Python can build the exact TFP FillScaleTriL latent.
# ==============================================================================

suppressMessages(library(bayesm))
suppressMessages(library(jsonlite))


run_bayesm_fit <- function(data_path, out_raw_dir, k_model, chains, r_total,
                           burn_in, thin, seed, a_mu, a_delta, dirichlet_a,
                           scenario = "NA", nprint = 2000L) {

    DATA_PATH   <- data_path
    OUT_RAW_DIR <- out_raw_dir
    SCENARIO    <- scenario
    N_COMP      <- as.integer(k_model)
    CHAINS      <- as.integer(chains)
    R_TOTAL     <- as.integer(r_total)
    BURN_IN     <- as.integer(burn_in)
    THIN        <- as.integer(thin)
    SEED        <- as.integer(seed)
    A_MU        <- as.numeric(a_mu)
    A_DELTA     <- as.numeric(a_delta)
    DIRICHLET_A <- as.numeric(dirichlet_a)
    NPRINT      <- as.integer(nprint)

    stopifnot(!is.null(DATA_PATH), !is.null(OUT_RAW_DIR),
              !is.na(N_COMP), !is.na(CHAINS), !is.na(R_TOTAL),
              !is.na(BURN_IN), !is.na(THIN), !is.na(SEED))

    dir.create(OUT_RAW_DIR, recursive = TRUE, showWarnings = FALSE)

    cat("============================================================\n")
    cat(sprintf("bayesm rhierMnlRwMixture | scenario=%s\n", SCENARIO))
    cat(sprintf("Data file : %s\n", DATA_PATH))
    cat(sprintf("K_MODEL=%d  chains=%d  R=%d  thin=%d  burn-in=%d  seed=%d\n",
                N_COMP, CHAINS, R_TOTAL, THIN, BURN_IN, SEED))
    cat(sprintf("Amu=%g  Ad=%g  dirichlet_a=%g\n", A_MU, A_DELTA, DIRICHLET_A))
    cat("============================================================\n")

    # -- 1. load data ---------------------------------------------------------
    raw <- fromJSON(DATA_PATH, simplifyVector = TRUE)

    n_units  <- as.integer(raw$n_units)
    n_params <- as.integer(raw$n_params)
    n_alts   <- as.integer(raw$n_alts)
    n_demos  <- as.integer(raw$n_demos)
    K_true   <- as.integer(raw$K)
    n_obs    <- as.integer(length(raw$y) / n_units)

    P <- n_params
    D <- n_demos
    N <- n_units
    has_Z <- !is.null(raw$Z) && D > 0

    cat(sprintf("N=%d  obs/unit=%d  alts=%d  P=%d  D=%d  K_true=%d\n",
                N, n_obs, n_alts, P, D, K_true))

    if (length(dim(raw$X)) != 3) {
        stop("raw$X did not simplify to a 3D array; check the data JSON.")
    }

    # -- 2. reconstruct lgtdata (bayesm layout) -------------------------------
    X_aperm <- aperm(raw$X, c(2, 1, 3))                 # (n_alts, n_total, P)
    X_all   <- matrix(X_aperm, ncol = n_params, byrow = FALSE)
    y_all   <- as.integer(unlist(raw$y)) + 1L           # 0-indexed -> 1-indexed

    lgtdata <- vector("list", N)
    rows_per_unit <- n_obs * n_alts
    for (i in seq_len(N)) {
        r0 <- (i - 1L) * rows_per_unit + 1L
        r1 <- i * rows_per_unit
        o0 <- (i - 1L) * n_obs + 1L
        o1 <- i * n_obs
        lgtdata[[i]] <- list(y = y_all[o0:o1], X = X_all[r0:r1, , drop = FALSE])
    }

    if (has_Z) {
        Z <- matrix(unlist(raw$Z), nrow = N, ncol = D, byrow = FALSE)
        data_list <- list(p = n_alts, lgtdata = lgtdata, Z = Z)
    } else {
        data_list <- list(p = n_alts, lgtdata = lgtdata)
    }

    # -- 3. prior matched to the Liesel mixture model -------------------------
    #   Sigma_k^{-1} ~ W(nu, V^{-1}),  nu = P + 3,  V = nu * I
    #   mu_k | Sigma_k ~ N(0, Sigma_k / Amu)
    #   vec(Delta)     ~ N(0, Ad^{-1}),  Ad = A_delta * I
    #   pvec           ~ Dirichlet(a),   a = dirichlet_a * 1
    Prior <- list(
        ncomp = N_COMP,
        nu    = P + 3,
        V     = (P + 3) * diag(P),
        Amu   = A_MU,
        a     = rep(DIRICHLET_A, N_COMP)
    )
    if (has_Z) Prior$Ad <- A_DELTA * diag(D * P)

    # keep = 1: bayesm returns EVERY raw draw; we discard warmup and thin ourselves
    # below (in raw-iteration units) so burn-in is removed BEFORE thinning.
    Mcmc <- list(R = R_TOTAL, keep = 1L, nprint = NPRINT)

    # -- 4. draw bookkeeping (identical across chains -> stackable) ------------
    keep_idx <- seq.int(BURN_IN + 1L, R_TOTAL, by = THIN)
    S        <- length(keep_idx)
    cat(sprintf("Raw draws/chain=%d  warmup discarded=%d  thin=%d  retained=%d\n",
                R_TOTAL, BURN_IN, THIN, S))

    write_bin <- function(arr, path) {
        con <- file(path, "wb")
        writeBin(as.double(arr), con, size = 8, endian = "little")
        close(con)
    }

    # -- 5. seed loop over chains ---------------------------------------------
    seeds        <- integer(CHAINS)
    durations_s  <- numeric(CHAINS)
    loglike_mean <- numeric(CHAINS)
    t_start      <- Sys.time()

    for (chain in seq_len(CHAINS)) {
        cidx  <- chain - 1L
        cseed <- SEED + cidx
        seeds[chain] <- cseed
        cat(sprintf("\n--- chain %d / %d  (seed=%d) ---\n", chain, CHAINS, cseed))

        set.seed(cseed)
        c_t0 <- Sys.time()
        out  <- rhierMnlRwMixture(Data = data_list, Prior = Prior, Mcmc = Mcmc)
        durations_s[chain] <- as.numeric(difftime(Sys.time(), c_t0, units = "secs"))

        # mu (S,K,P) and precision (S,K,P,P) from the component draws
        mu_arr   <- array(0.0, dim = c(S, N_COMP, P))
        prec_arr <- array(0.0, dim = c(S, N_COMP, P, P))
        cd <- out$nmix$compdraw
        for (s in seq_len(S)) {
            comp <- cd[[keep_idx[s]]]
            for (k in seq_len(N_COMP)) {
                mu_arr[s, k, ] <- comp[[k]]$mu
                rt <- comp[[k]]$rooti                 # inverse upper-Cholesky of Sigma
                prec_arr[s, k, , ] <- rt %*% t(rt)    # = Sigma^{-1} (bayesm convention)
            }
        }

        # pvec (S,K) - probdraw may be (K, R_draws) or (R_draws, K)
        pd <- out$nmix$probdraw
        if (nrow(pd) == N_COMP) pd <- t(pd)
        pvec_arr <- pd[keep_idx, , drop = FALSE]

        # beta (S,N,P) from (N,P,R_draws)
        beta_arr <- aperm(out$betadraw[, , keep_idx, drop = FALSE], c(3, 1, 2))

        write_bin(mu_arr,   file.path(OUT_RAW_DIR, sprintf("mu_chain%d.bin",   cidx)))
        write_bin(prec_arr, file.path(OUT_RAW_DIR, sprintf("prec_chain%d.bin", cidx)))
        write_bin(pvec_arr, file.path(OUT_RAW_DIR, sprintf("pvec_chain%d.bin", cidx)))
        write_bin(beta_arr, file.path(OUT_RAW_DIR, sprintf("beta_chain%d.bin", cidx)))

        if (has_Z) {
            # bayesm's Deltadraw row is vec(Delta) for Delta indexed (nvar, nz) = (P, D)
            # — column-major of (P, D), i.e. ROW-major of the (D, P) we want. Reshape to
            # (P, D, S) then permute to canonical (S, D, P) so delta[s, d, p] = effect of
            # demographic d on parameter p (matches TRUE_DELTA and the Liesel Delta).
            # (Reshaping as (D, P) transposes the demographic x parameter mapping when
            # D != P — verified against regress(bayesm beta ~ Z), which matches TRUE_DELTA.)
            Dd <- out$Deltadraw[keep_idx, , drop = FALSE]
            delta_arr <- aperm(array(t(Dd), dim = c(P, D, S)), c(3, 2, 1))
            write_bin(delta_arr, file.path(OUT_RAW_DIR, sprintf("delta_chain%d.bin", cidx)))
        }

        loglike_mean[chain] <- mean(out$loglike[keep_idx])
        cat(sprintf("chain %d done in %.1fs  mean loglike=%.1f\n",
                    chain, durations_s[chain], loglike_mean[chain]))
        rm(out); gc(verbose = FALSE)
    }

    total_duration_s <- as.numeric(difftime(Sys.time(), t_start, units = "secs"))

    # -- 6. write the bridge metadata -----------------------------------------
    per_chain <- list(
        mu   = c(S, N_COMP, P),
        prec = c(S, N_COMP, P, P),
        pvec = c(S, N_COMP),
        beta = c(S, N, P)
    )
    if (has_Z) per_chain$delta <- c(S, D, P)

    dims <- list(
        chains = CHAINS, n_samples = S, K = N_COMP, P = P, D = D, N = N,
        has_Z = has_Z, order = "F", per_chain = per_chain
    )
    write_json(dims, file.path(OUT_RAW_DIR, "dims.json"), auto_unbox = TRUE)

    meta_r <- list(
        scenario = SCENARIO, k_model = N_COMP, k_true = K_true,
        n_params = P, n_units = N, n_demos = D, n_obs = n_obs, n_alts = n_alts,
        chains = CHAINS, r_total = R_TOTAL, thin = THIN, burn_in = BURN_IN,
        n_samples = S, a_mu = A_MU, a_delta = A_DELTA, dirichlet_a = DIRICHLET_A,
        # as.list keeps these as JSON arrays even when CHAINS == 1.
        seeds = as.list(seeds), durations_s = as.list(round(durations_s, 1)),
        total_duration_s = round(total_duration_s, 1),
        loglike_mean = as.list(round(loglike_mean, 2)),
        bayesm_version = as.character(packageVersion("bayesm")),
        r_version = R.version.string,
        started_at = format(t_start, "%Y-%m-%dT%H:%M:%S")
    )
    write_json(meta_r, file.path(OUT_RAW_DIR, "meta_r.json"), auto_unbox = TRUE)

    cat(sprintf("\nAll chains complete in %.1fs. Raw draws -> %s\n",
                total_duration_s, OUT_RAW_DIR))
    invisible(OUT_RAW_DIR)
}
