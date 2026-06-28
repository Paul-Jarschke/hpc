library(tidyverse)

# Number of replicate seeds per scenario to include (data_seed in 1..MAX_DATA_SEED).
# Set to 100 for the full grid; a smaller value runs a quick subset.
MAX_DATA_SEED <- 100

# ..............................................................................
# Build params.csv for ONE n_chains slice of the bayesm arm of the k5model_mixture
# experiment. The chain count is read from THIS job's directory name
# (e.g. "008-k5-bayesm-c1" -> n_chains=1), so this script is byte-identical across
# the bayesm jobs (008, 009). sampler is fixed to "bayesm".
#
#   grid = 200 datasets (manifest) filtered to data_seed<=MAX_DATA_SEED x this n_chains
#        = 40 rows (10 seeds x 4 scenarios), each = one overspecified K_MODEL=5 fit.
#
# bayesm MCMC controls: keep ALL r_total raw draws, drop the first burn_in (warmup),
# then thin by `thin`, in raw-iteration units: (42000 - 2000) / 4 = 10000 retained
# draws, matching the liesel posterior length. Priors (a_mu/a_delta/dirichlet_a) are
# set to the SAME values as the liesel grid for a controlled cross-sampler comparison
# (the liesel model's a_mu/A_delta map exactly onto bayesm's Amu/Ad).
#
# Run locally. This OVERWRITES params.csv.
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

# ---- derive n_chains from the job directory name; sampler is fixed ----
n_chains <- as.integer(str_match(job_name, "-c(\\d+)$")[, 2])
if (is.na(n_chains)) {
  stop("Could not parse n_chains from job dir name: ", job_name)
}
cat("Job:", job_name, "-> sampler = bayesm | n_chains =", n_chains, "\n")

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
    sampler     = "bayesm",
    n_chains    = n_chains,
    k_model     = 5L,
    r_total     = 42000L,
    burn_in     = 2000L,
    thin        = 4L,
    seed        = 42L,
    a_mu        = 0.01,
    a_delta     = 0.01,
    dirichlet_a = 1.0,
  ) |>
  arrange(k_true, data_seed)

write_csv(params, fs::path(script_dir, "params.csv"))
cat("Wrote", nrow(params), "rows to", fs::path(script_dir, "params.csv"), "\n")
