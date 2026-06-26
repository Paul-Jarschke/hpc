library(tidyverse)

# Number of replicate seeds per scenario to include (data_seed in 1..MAX_DATA_SEED).
# Set to 50 for the full grid; a smaller value runs a quick subset.
MAX_DATA_SEED <- 50

# ..............................................................................
# Build params.csv for ONE (sampler x n_chains) slice of the k5model_mixture
# experiment. The sampler and chain count are read from THIS job's directory
# name (e.g. "004-k5-nuts-c1" -> sampler=nuts, n_chains=1), so this script is
# byte-identical across the four jobs (004-007).
#
#   grid = 200 datasets (from the manifest) x this (sampler, n_chains)
#        = 200 rows, each = one overspecified K_MODEL=5 fit.
#
# Run locally. This OVERWRITES params.csv (including any timing-probe row).
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

# ---- derive sampler + n_chains from the job directory name ----
sampler  <- str_match(job_name, "-(nuts|hmc)-")[, 2]
n_chains <- as.integer(str_match(job_name, "-c(\\d+)$")[, 2])
if (is.na(sampler) || is.na(n_chains)) {
  stop("Could not parse sampler/n_chains from job dir name: ", job_name)
}
cat("Job:", job_name, "-> sampler =", sampler, "| n_chains =", n_chains, "\n")

manifest <- read_csv(
  fs::path(script_dir, "..", "..", "data", "in", "k5model_mixture", "manifest.csv"),
  show_col_types = FALSE
)

params <- manifest |>
  filter(data_seed <= MAX_DATA_SEED) |>
  select(dataset_key, scenario, k_true, data_seed) |>
  mutate(
    sampler               = sampler,
    n_chains              = n_chains,
    k_model               = 5L,
    warmup                = 2000L,
    posterior             = 10000L,
    seed                  = 42L,
    a_mu                  = 0.01,
    a_delta               = 0.01,
    dirichlet_a           = 1.0,
    num_integration_steps = 10L,
  ) |>
  arrange(k_true, data_seed)

write_csv(params, fs::path(script_dir, "params.csv"))
cat("Wrote", nrow(params), "rows to", fs::path(script_dir, "params.csv"), "\n")
