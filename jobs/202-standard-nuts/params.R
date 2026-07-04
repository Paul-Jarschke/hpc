library(tidyverse)

# Number of replicate seeds to include (data_seed in 1..MAX_DATA_SEED).
# 100 = the full grid (1 scenario x 100 fittable seeds = 100 rows).
MAX_DATA_SEED <- 100

# ..............................................................................
# Build params.csv for ONE sampler slice of the STANDARD (single-normal-
# component, no mixture) HBMNL experiment (jobs 200-202, Rossi section 5.4;
# upstream HierarchicalBayesianMultinomialLogit @ 893e63f). The sampler is read
# from THIS job's directory name, so this script is byte-identical across the
# three jobs:
#
#   200-standard-bayesm -> sampler = bayesm (R, rhierMnlRwMixture with ncomp=1)
#   201-standard-hmc    -> sampler = hmc    (Liesel/Goose, dense mm on Sigma)
#   202-standard-nuts   -> sampler = nuts   (Liesel/Goose, dense mm on Sigma)
#
# All jobs run 2 chains. Draw bookkeeping matched across arms:
#   nuts/hmc : warmup 2000, posterior 10000 (per chain)
#   bayesm   : r_total 42000, burn_in 2000, thin 4 -> 10000 kept (per chain)
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
sampler <- str_match(job_name, "^2\\d\\d-standard-(bayesm|hmc|nuts)$")[, 2]
if (is.na(sampler)) {
  stop("Could not parse the sampler arm from job dir name: ", job_name)
}
n_chains <- 2L
cat("Job:", job_name, "-> sampler =", sampler, "| n_chains =", n_chains, "\n")

manifest <- read_csv(
  fs::path(script_dir, "..", "..", "data", "in", "standard_model", "manifest.csv"),
  show_col_types = FALSE
)

# bayesm cannot fit a dataset unless every alternative is chosen at least once; screened
# in generate_standard_data.py (all_alts_chosen == 0). Drop those, then keep the first
# MAX_DATA_SEED FITTABLE seeds, so every sampler uses the SAME fittable datasets.
excluded <- manifest |> dplyr::filter(all_alts_chosen == 0)
if (nrow(excluded) > 0) {
  cat("Screened out", nrow(excluded), "unfittable dataset(s):",
      paste(excluded$dataset_key, collapse = ", "), "\n")
}

params <- manifest |>
  filter(all_alts_chosen == 1) |>
  arrange(data_seed) |>
  slice_head(n = MAX_DATA_SEED) |>
  select(dataset_key, scenario, k_true, data_seed) |>
  mutate(
    sampler               = sampler,
    n_chains              = n_chains,
    k_model               = 1L,      # bayesm's ncomp; the Liesel arms have no K at all
    warmup                = 2000L,   # nuts/hmc only
    posterior             = 10000L,  # nuts/hmc only
    r_total               = 42000L,  # bayesm only
    burn_in               = 2000L,   # bayesm only
    thin                  = 4L,      # bayesm only
    seed                  = 42L,
    a_mu                  = 0.01,
    a_delta               = 0.01,
    dirichlet_a           = 1.0,     # bayesm only (inert with ncomp = 1)
    num_integration_steps = 10L,     # hmc only
  ) |>
  arrange(data_seed)

write_csv(params, fs::path(script_dir, "params.csv"))
cat("Wrote", nrow(params), "rows to", fs::path(script_dir, "params.csv"), "\n")
