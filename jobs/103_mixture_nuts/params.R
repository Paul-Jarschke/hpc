library(tidyverse)

# Number of replicate seeds per scenario to include (data_seed in 1..MAX_DATA_SEED).
# 100 = the full grid (4 scenarios x 100 fittable seeds = 400 rows).
MAX_DATA_SEED <- 50

# ..............................................................................
# Build params.csv for ONE sampler slice of the 2-chain mixture experiment
# (jobs 100-103, updated port of HierarchicalBayesianMultinomialLogit @ 893e63f).
# The sampler is read from THIS job's directory name, so this script is
# byte-identical across the four jobs:
#
#   100_mixture_bayesm     -> sampler = bayesm       (R, rhierMnlRwMixture)
#   101_mixture_replicate  -> sampler = bayesm_gibbs (Python/Goose replication)
#   102_mixture_hmc        -> sampler = hmc          (informed init, dense mm)
#   103_mixture_nuts       -> sampler = nuts         (informed init, dense mm)
#
# All jobs run 2 chains. Draw bookkeeping is matched across arms:
#   nuts/hmc      : warmup 2000, posterior 10000 (per chain)
#   bayesm arms   : r_total 42000, burn_in 2000, thin 4 -> 10000 kept (per chain)
#
# Run locally (Rscript or RStudio). This OVERWRITES params.csv.
# ..............................................................................

get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg)) {
    return(dirname(normalizePath(sub("^--file=", "", file_arg))))
  }
  if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
    return(dirname(rstudioapi::getActiveDocumentContext()$path))
  }
  getwd()
}

script_dir <- get_script_dir()
job_name   <- basename(script_dir)

# ---- derive the sampler from the job directory name ----
arm <- str_match(job_name, "^1\\d\\d_mixture_(bayesm|replicate|hmc|nuts)$")[, 2]
if (is.na(arm)) {
  stop("Could not parse the sampler arm from job dir name: ", job_name)
}
sampler  <- if (arm == "replicate") "bayesm_gibbs" else arm
n_chains <- 2L
cat("Job:", job_name, "-> sampler =", sampler, "| n_chains =", n_chains, "\n")

manifest <- read_csv(
  fs::path(script_dir, "..", "..", "data", "in", "k5model_mixture", "manifest.csv"),
  show_col_types = FALSE
)

# bayesm cannot fit a dataset unless every alternative is chosen at least once; those are
# screened in generate_mixture_data.py (all_alts_chosen == 0). Drop them, then keep the first
# MAX_DATA_SEED FITTABLE seeds per scenario, so every sampler uses the SAME fittable datasets.
excluded <- manifest |> dplyr::filter(all_alts_chosen == 0)
if (nrow(excluded) > 0) {
  cat("Screened out", nrow(excluded), "unfittable dataset(s):",
      paste(excluded$dataset_key, collapse = ", "), "\n")
}

params <- manifest |>
  filter(all_alts_chosen == 1) |>
  group_by(k_true) |>
  arrange(data_seed, .by_group = TRUE) |>
  slice_head(n = MAX_DATA_SEED) |>
  ungroup() |>
  select(dataset_key, scenario, k_true, data_seed) |>
  mutate(
    sampler               = sampler,
    n_chains              = n_chains,
    k_model               = 5L,
    warmup                = 2000L,   # nuts/hmc only
    posterior             = 10000L,  # nuts/hmc only
    r_total               = 42000L,  # bayesm/bayesm_gibbs only
    burn_in               = 2000L,   # bayesm/bayesm_gibbs only
    thin                  = 4L,      # bayesm/bayesm_gibbs only
    seed                  = 42L,
    a_mu                  = 0.01,
    a_delta               = 0.01,
    dirichlet_a           = 1.0,
    num_integration_steps = 10L,     # hmc only
  ) |>
  arrange(k_true, data_seed)

write_csv(params, fs::path(script_dir, "params.csv"))
cat("Wrote", nrow(params), "rows to", fs::path(script_dir, "params.csv"), "\n")
